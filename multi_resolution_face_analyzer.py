#!/usr/bin/env python3
"""
Multi-Resolution Face Detection and Reconciliation

This script implements a sophisticated face detection strategy that:
1. Detects faces at multiple resolutions (720p, 1080p, 1440p)
2. Analyzes and reconciles faces detected multiple times
3. Combines results for maximum accuracy and confidence

The approach leverages the strengths of different resolutions and uses
spatial overlap and confidence scoring to merge duplicate detections.
"""

import cv2
import numpy as np
import json
import random
from pathlib import Path
from typing import List, Dict, Any, Tuple, Set
import time
from loguru import logger
import click
from tqdm import tqdm
from dataclasses import dataclass
from collections import defaultdict

# Import our face detection system
from src.soccer_photo_sorter.detectors.face_detector import FaceDetector
from src.soccer_photo_sorter.core.image_processor import ImageProcessor
from src.soccer_photo_sorter.config.settings import Settings
from src.soccer_photo_sorter.utils.cuda_utils import CudaManager


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle numpy types and complex objects."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, '__class__') and 'face_recognition' in str(obj.__class__):
            return str(obj)
        elif hasattr(obj, '__class__') and 'dlib' in str(obj.__class__):
            return str(obj)
        return super(NumpyEncoder, self).default(obj)


@dataclass
class FaceDetection:
    """Represents a single face detection with metadata."""
    bbox: Tuple[int, int, int, int]  # (x, y, width, height)
    confidence: float
    method: str
    resolution: str
    face_id: str
    encoding: Any = None
    center: Tuple[int, int] = None
    
    def __post_init__(self):
        """Calculate center point after initialization."""
        if self.center is None:
            x, y, w, h = self.bbox
            self.center = (x + w // 2, y + h // 2)


@dataclass
class ReconciledFace:
    """Represents a reconciled face from multiple detections."""
    bbox: Tuple[int, int, int, int]
    confidence: float
    detection_count: int
    methods: List[str]
    resolutions: List[str]
    face_id: str
    encoding: Any = None
    center: Tuple[int, int] = None
    
    def __post_init__(self):
        """Calculate center point after initialization."""
        if self.center is None:
            x, y, w, h = self.bbox
            self.center = (x + w // 2, y + h // 2)


class MultiResolutionFaceAnalyzer:
    """Analyzes faces across multiple resolutions and reconciles duplicates."""
    
    def __init__(self, 
                 target_resolutions: List[Tuple[int, int]] = None,
                 overlap_threshold: float = 0.3,
                 confidence_threshold: float = 0.5):
        """
        Initialize multi-resolution face analyzer.
        
        Args:
            target_resolutions: List of (width, height) tuples for detection
            overlap_threshold: IoU threshold for considering faces as duplicates
            confidence_threshold: Minimum confidence for final face acceptance
        """
        # Default resolutions: 720p, 1080p, 1440p
        self.target_resolutions = target_resolutions or [
            (1280, 720),   # 720p
            (1920, 1080),  # 1080p  
            (2560, 1440),  # 1440p
        ]
        
        self.overlap_threshold = overlap_threshold
        self.confidence_threshold = confidence_threshold
        
        # Initialize face detector with low confidence for maximum sensitivity
        self.face_detector = FaceDetector(
            confidence_threshold=0.1,  # Very low threshold to catch all detections
            min_faces=0,  # No minimum requirement
            max_faces=50,  # High maximum to catch all faces
            face_size=20  # Small minimum face size
        )
        
        # Initialize image processor
        settings = Settings()
        self.image_processor = ImageProcessor(settings)
        
        # Check CUDA availability
        cuda_manager = CudaManager()
        self.cuda_available = cuda_manager.is_available
        logger.info(f"CUDA available: {self.cuda_available}")
    
    def resize_image_to_resolution(self, image: np.ndarray, target_width: int, target_height: int) -> np.ndarray:
        """
        Resize image to target resolution while maintaining aspect ratio.
        
        Args:
            image: Input image array
            target_width: Target width
            target_height: Target height
            
        Returns:
            Resized image array
        """
        height, width = image.shape[:2]
        
        # Calculate scaling factor to fit within target resolution
        scale_w = target_width / width
        scale_h = target_height / height
        scale = min(scale_w, scale_h)
        
        # Calculate new dimensions
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        # Resize image
        resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        # Create canvas with target resolution
        canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)
        
        # Center the resized image on the canvas
        y_offset = (target_height - new_height) // 2
        x_offset = (target_width - new_width) // 2
        canvas[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized
        
        return canvas, (scale, x_offset, y_offset)
    
    def detect_faces_at_resolution(self, image: np.ndarray, resolution: Tuple[int, int], 
                                 image_name: str) -> List[FaceDetection]:
        """
        Detect faces at a specific resolution.
        
        Args:
            image: Original image array
            resolution: Target resolution (width, height)
            image_name: Name of the image file
            
        Returns:
            List of FaceDetection objects
        """
        width, height = resolution
        res_key = f"{width}x{height}"
        
        # Resize image to target resolution
        resized_image, (scale, x_offset, y_offset) = self.resize_image_to_resolution(image, width, height)
        
        # Create temporary file for face detector
        temp_path = Path(f"/tmp/temp_multi_face_{image_name}_{width}x{height}.jpg")
        cv2.imwrite(str(temp_path), cv2.cvtColor(resized_image, cv2.COLOR_RGB2BGR))
        
        face_detections = []
        
        try:
            faces = self.face_detector.detect_faces(temp_path)
            
            for i, face in enumerate(faces):
                bbox = face.get('bbox', (0, 0, 0, 0))
                confidence = face.get('confidence', 0.0)
                method = face.get('method', 'unknown')
                face_id = face.get('face_id', f'face_{i}')
                encoding = face.get('encoding')
                
                # Convert coordinates back to original image scale
                x, y, w, h = bbox
                
                # Remove offset and scale back to original
                orig_x = int((x - x_offset) / scale)
                orig_y = int((y - y_offset) / scale)
                orig_w = int(w / scale)
                orig_h = int(h / scale)
                
                # Ensure coordinates are within original image bounds
                orig_height, orig_width = image.shape[:2]
                orig_x = max(0, min(orig_x, orig_width - 1))
                orig_y = max(0, min(orig_y, orig_height - 1))
                orig_w = min(orig_w, orig_width - orig_x)
                orig_h = min(orig_h, orig_height - orig_y)
                
                detection = FaceDetection(
                    bbox=(orig_x, orig_y, orig_w, orig_h),
                    confidence=confidence,
                    method=method,
                    resolution=res_key,
                    face_id=face_id,
                    encoding=encoding
                )
                
                face_detections.append(detection)
                
        except Exception as e:
            logger.error(f"Face detection error at {res_key}: {e}")
        finally:
            # Clean up temp file
            if temp_path.exists():
                temp_path.unlink()
        
        return face_detections
    
    def calculate_iou(self, bbox1: Tuple[int, int, int, int], bbox2: Tuple[int, int, int, int]) -> float:
        """
        Calculate Intersection over Union (IoU) between two bounding boxes.
        
        Args:
            bbox1: First bounding box (x, y, width, height)
            bbox2: Second bounding box (x, y, width, height)
            
        Returns:
            IoU value between 0 and 1
        """
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        # Calculate intersection
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        # Calculate union
        bbox1_area = w1 * h1
        bbox2_area = w2 * h2
        union_area = bbox1_area + bbox2_area - intersection_area
        
        if union_area == 0:
            return 0.0
        
        return intersection_area / union_area
    
    def reconcile_faces(self, all_detections: List[FaceDetection]) -> List[ReconciledFace]:
        """
        Reconcile multiple face detections into unique faces.
        
        Args:
            all_detections: List of all face detections across resolutions
            
        Returns:
            List of reconciled faces
        """
        if not all_detections:
            return []
        
        # Group detections by spatial overlap
        reconciled_faces = []
        processed = set()
        
        for i, detection in enumerate(all_detections):
            if i in processed:
                continue
            
            # Find all detections that overlap with this one
            overlapping_detections = [detection]
            overlapping_indices = {i}
            
            for j, other_detection in enumerate(all_detections):
                if j <= i or j in processed:
                    continue
                
                iou = self.calculate_iou(detection.bbox, other_detection.bbox)
                if iou >= self.overlap_threshold:
                    overlapping_detections.append(other_detection)
                    overlapping_indices.add(j)
            
            # Mark all overlapping detections as processed
            processed.update(overlapping_indices)
            
            # Create reconciled face
            reconciled_face = self._create_reconciled_face(overlapping_detections)
            reconciled_faces.append(reconciled_face)
        
        # Filter by confidence threshold
        filtered_faces = [
            face for face in reconciled_faces 
            if face.confidence >= self.confidence_threshold
        ]
        
        return filtered_faces
    
    def _create_reconciled_face(self, detections: List[FaceDetection]) -> ReconciledFace:
        """
        Create a reconciled face from multiple detections.
        
        Args:
            detections: List of overlapping detections
            
        Returns:
            ReconciledFace object
        """
        if len(detections) == 1:
            detection = detections[0]
            return ReconciledFace(
                bbox=detection.bbox,
                confidence=detection.confidence,
                detection_count=1,
                methods=[detection.method],
                resolutions=[detection.resolution],
                face_id=detection.face_id,
                encoding=detection.encoding
            )
        
        # Calculate weighted average bounding box
        total_weight = sum(det.confidence for det in detections)
        weighted_x = sum(det.bbox[0] * det.confidence for det in detections) / total_weight
        weighted_y = sum(det.bbox[1] * det.confidence for det in detections) / total_weight
        weighted_w = sum(det.bbox[2] * det.confidence for det in detections) / total_weight
        weighted_h = sum(det.bbox[3] * det.confidence for det in detections) / total_weight
        
        # Calculate average confidence
        avg_confidence = sum(det.confidence for det in detections) / len(detections)
        
        # Collect unique methods and resolutions
        methods = list(set(det.method for det in detections))
        resolutions = list(set(det.resolution for det in detections))
        
        # Use the most confident detection's encoding
        best_detection = max(detections, key=lambda d: d.confidence)
        
        return ReconciledFace(
            bbox=(int(weighted_x), int(weighted_y), int(weighted_w), int(weighted_h)),
            confidence=avg_confidence,
            detection_count=len(detections),
            methods=methods,
            resolutions=resolutions,
            face_id=f"reconciled_{len(detections)}_detections",
            encoding=best_detection.encoding
        )
    
    def _draw_face_detections(self, image: np.ndarray, detections: List[FaceDetection], 
                             color: Tuple[int, int, int] = (255, 255, 0)) -> np.ndarray:
        """Draw individual face detections on image."""
        vis_image = image.copy()
        
        for i, detection in enumerate(detections):
            x, y, w, h = detection.bbox
            
            # Draw face bounding box
            cv2.rectangle(vis_image, (x, y), (x + w, y + h), color, 2)
            
            # Draw label
            label = f"{detection.resolution}: {detection.confidence:.2f}"
            cv2.putText(vis_image, label, (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return vis_image
    
    def _draw_reconciled_faces(self, image: np.ndarray, reconciled_faces: List[ReconciledFace]) -> np.ndarray:
        """Draw reconciled faces on image."""
        vis_image = image.copy()
        
        for i, face in enumerate(reconciled_faces):
            x, y, w, h = face.bbox
            
            # Choose color based on detection count
            if face.detection_count >= 3:
                color = (0, 255, 0)  # Green - detected at all resolutions
            elif face.detection_count == 2:
                color = (0, 255, 255)  # Yellow - detected at 2 resolutions
            else:
                color = (0, 165, 255)  # Orange - detected at 1 resolution
            
            # Draw face bounding box
            cv2.rectangle(vis_image, (x, y), (x + w, y + h), color, 3)
            
            # Draw confidence and detection count
            label = f"Face {i+1}: {face.confidence:.2f} ({face.detection_count} detections)"
            cv2.putText(vis_image, label, (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            # Draw methods and resolutions
            methods_str = "+".join(face.methods)
            resolutions_str = "+".join(face.resolutions)
            cv2.putText(vis_image, f"{methods_str} | {resolutions_str}", (x, y + h + 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return vis_image
    
    def analyze_image(self, image_path: Path) -> Dict[str, Any]:
        """
        Analyze a single image with multi-resolution face detection.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Dictionary with analysis results
        """
        logger.debug(f"Analyzing {image_path.name}")
        
        # Load image
        image = self.image_processor.load_image(image_path)
        if image is None:
            logger.warning(f"Failed to load image: {image_path}")
            return {}
        
        all_detections = []
        resolution_results = {}
        
        # Detect faces at each resolution
        for width, height in self.target_resolutions:
            logger.debug(f"Detecting faces at {width}x{height}")
            
            detections = self.detect_faces_at_resolution(image, (width, height), image_path.name)
            all_detections.extend(detections)
            
            resolution_results[f"{width}x{height}"] = {
                'detections': len(detections),
                'faces': [
                    {
                        'bbox': det.bbox,
                        'confidence': det.confidence,
                        'method': det.method,
                        'face_id': det.face_id
                    }
                    for det in detections
                ]
            }
        
        # Reconcile faces
        reconciled_faces = self.reconcile_faces(all_detections)
        
        # Create visualizations
        individual_vis = self._draw_face_detections(image, all_detections)
        reconciled_vis = self._draw_reconciled_faces(image, reconciled_faces)
        
        return {
            'image_name': image_path.name,
            'image_path': str(image_path),
            'total_detections': len(all_detections),
            'reconciled_faces': len(reconciled_faces),
            'resolution_results': resolution_results,
            'reconciled_face_data': [
                {
                    'bbox': face.bbox,
                    'confidence': face.confidence,
                    'detection_count': face.detection_count,
                    'methods': face.methods,
                    'resolutions': face.resolutions,
                    'face_id': face.face_id
                }
                for face in reconciled_faces
            ],
            'individual_visualization': individual_vis,
            'reconciled_visualization': reconciled_vis
        }
    
    def run_multi_resolution_analysis(self, input_pattern: str, output_dir: Path, 
                                    num_images: int = 10, verbose: bool = False) -> None:
        """
        Run multi-resolution face analysis on multiple images.
        
        Args:
            input_pattern: Glob pattern for input images
            output_dir: Output directory for results
            num_images: Number of random images to test
            verbose: Enable verbose logging
        """
        if verbose:
            logger.add(lambda msg: print(msg, end=""), level="DEBUG")
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create visualization directories
        individual_dir = output_dir / 'individual_detections'
        reconciled_dir = output_dir / 'reconciled_faces'
        individual_dir.mkdir(parents=True, exist_ok=True)
        reconciled_dir.mkdir(parents=True, exist_ok=True)
        
        # Get image files
        if input_pattern.startswith('/'):
            # Absolute path pattern
            image_paths = list(Path(input_pattern).parent.glob(Path(input_pattern).name))
        else:
            # Relative path pattern
            image_paths = list(Path().glob(input_pattern))
        
        if not image_paths:
            logger.error(f"No images found matching pattern: {input_pattern}")
            return
        
        # Select random images
        if len(image_paths) > num_images:
            image_paths = random.sample(image_paths, num_images)
        
        logger.info(f"Analyzing {len(image_paths)} images across {len(self.target_resolutions)} resolutions")
        
        all_results = []
        summary_stats = {
            'total_images': len(image_paths),
            'total_detections': 0,
            'total_reconciled_faces': 0,
            'resolution_stats': defaultdict(lambda: {'detections': 0, 'images': 0}),
            'detection_count_stats': defaultdict(int),
            'method_stats': defaultdict(int)
        }
        
        # Process each image
        for image_path in tqdm(image_paths, desc="Analyzing images"):
            result = self.analyze_image(image_path)
            if not result:
                continue
            
            all_results.append(result)
            
            # Update summary statistics
            summary_stats['total_detections'] += result['total_detections']
            summary_stats['total_reconciled_faces'] += result['reconciled_faces']
            
            # Update resolution stats
            for res_key, res_data in result['resolution_results'].items():
                summary_stats['resolution_stats'][res_key]['detections'] += res_data['detections']
                summary_stats['resolution_stats'][res_key]['images'] += 1
            
            # Update detection count stats
            for face_data in result['reconciled_face_data']:
                count = face_data['detection_count']
                summary_stats['detection_count_stats'][count] += 1
                
                # Update method stats
                for method in face_data['methods']:
                    summary_stats['method_stats'][method] += 1
            
            # Save visualizations
            individual_filename = f"{image_path.stem}_individual_detections.jpg"
            reconciled_filename = f"{image_path.stem}_reconciled_faces.jpg"
            
            cv2.imwrite(str(individual_dir / individual_filename), 
                       cv2.cvtColor(result['individual_visualization'], cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(reconciled_dir / reconciled_filename), 
                       cv2.cvtColor(result['reconciled_visualization'], cv2.COLOR_RGB2BGR))
        
        # Prepare results data
        results_data = {
            'analysis_info': {
                'input_pattern': input_pattern,
                'num_images_analyzed': len(image_paths),
                'target_resolutions': [f"{w}x{h}" for w, h in self.target_resolutions],
                'overlap_threshold': self.overlap_threshold,
                'confidence_threshold': self.confidence_threshold,
                'cuda_available': self.cuda_available,
                'analysis_timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            },
            'summary_statistics': dict(summary_stats),
            'detailed_results': all_results
        }
        
        # Save results to JSON
        results_file = output_dir / 'multi_resolution_face_analysis_results.json'
        with open(results_file, 'w') as f:
            json.dump(results_data, f, indent=2, cls=NumpyEncoder)
        
        logger.info(f"Results saved to {results_file}")
        
        # Print summary
        self._print_summary(summary_stats, output_dir)
    
    def _print_summary(self, summary_stats: Dict, output_dir: Path) -> None:
        """Print analysis summary."""
        print("\n🎯 MULTI-RESOLUTION FACE ANALYSIS RESULTS")
        print("=" * 60)
        
        print(f"📊 Images analyzed: {summary_stats['total_images']}")
        print(f"🔍 Total detections: {summary_stats['total_detections']}")
        print(f"✅ Reconciled faces: {summary_stats['total_reconciled_faces']}")
        
        # Calculate efficiency metrics
        if summary_stats['total_detections'] > 0:
            reconciliation_rate = (summary_stats['total_reconciled_faces'] / summary_stats['total_detections']) * 100
            print(f"📈 Reconciliation rate: {reconciliation_rate:.1f}%")
        
        print("\n📏 Resolution Performance:")
        print("-" * 30)
        for res_key, stats in summary_stats['resolution_stats'].items():
            avg_detections = stats['detections'] / stats['images'] if stats['images'] > 0 else 0
            print(f"{res_key}: {avg_detections:.1f} faces/image ({stats['detections']} total)")
        
        print("\n🎯 Detection Confidence:")
        print("-" * 30)
        for count, faces in sorted(summary_stats['detection_count_stats'].items()):
            percentage = (faces / summary_stats['total_reconciled_faces']) * 100 if summary_stats['total_reconciled_faces'] > 0 else 0
            print(f"{count} resolution(s): {faces} faces ({percentage:.1f}%)")
        
        print("\n🔧 Detection Methods:")
        print("-" * 30)
        for method, count in summary_stats['method_stats'].items():
            print(f"{method}: {count} detections")
        
        print(f"\n💾 Results saved to: {output_dir}")
        print(f"🖼️ Individual detections: {output_dir}/individual_detections/")
        print(f"🎯 Reconciled faces: {output_dir}/reconciled_faces/")


@click.command()
@click.argument('input_pattern', type=str)
@click.argument('output_dir', type=click.Path())
@click.option('--num-images', '-n', default=10, help='Number of random images to test')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.option('--overlap-threshold', default=0.3, help='IoU threshold for face reconciliation')
@click.option('--confidence-threshold', default=0.5, help='Minimum confidence for final faces')
def main(input_pattern: str, output_dir: str, num_images: int, verbose: bool, 
         overlap_threshold: float, confidence_threshold: float):
    """
    Multi-resolution face detection and reconciliation analysis.
    
    INPUT_PATTERN: Glob pattern for input images (e.g., "/path/to/images/*.jpg")
    OUTPUT_DIR: Directory to save analysis results and visualizations
    """
    output_path = Path(output_dir)
    
    # Initialize analyzer
    analyzer = MultiResolutionFaceAnalyzer(
        overlap_threshold=overlap_threshold,
        confidence_threshold=confidence_threshold
    )
    
    # Run analysis
    analyzer.run_multi_resolution_analysis(input_pattern, output_path, num_images, verbose)


if __name__ == '__main__':
    main()
