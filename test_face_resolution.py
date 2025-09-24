#!/usr/bin/env python3
"""
Face Detection Resolution Test

This script tests face detection performance across different image resolutions
to determine the optimal size for detection accuracy, similar to facial recognition
improvements with downsampling.

The hypothesis is that face detection models were trained on common
video resolutions (720p, 1080p, 1440p) rather than full-resolution mirrorless
camera images, so downsampling might improve detection accuracy.
"""

import cv2
import numpy as np
import json
import random
from pathlib import Path
from typing import List, Dict, Any, Tuple
import time
from loguru import logger
import click
from tqdm import tqdm

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
            return str(obj)  # Convert face_recognition objects to strings
        elif hasattr(obj, '__class__') and 'dlib' in str(obj.__class__):
            return str(obj)  # Convert dlib objects to strings
        return super(NumpyEncoder, self).default(obj)


class FaceResolutionTester:
    """Test face detection performance across different image resolutions."""
    
    def __init__(self):
        """Initialize the face resolution tester."""
        # Test resolutions (width, height) - same as pose testing for comparison
        self.test_resolutions = [
            (1024, 768),    # XGA
            (1280, 720),    # HD
            (1366, 768),    # Common laptop resolution
            (1600, 900),    # HD+
            (1920, 1080),   # Full HD
            (2048, 1152),   # 2K
            (2560, 1440),   # 2K QHD
            (3200, 1800),   # 3.2K
            (3840, 2160),   # 4K UHD
            (5120, 2880),   # 5K
        ]
        
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
        
        return canvas
    
    def _draw_face_detections(self, image: np.ndarray, faces: List[Dict[str, Any]]) -> np.ndarray:
        """Draw face detections on image."""
        vis_image = image.copy()
        
        for i, face in enumerate(faces):
            bbox = face.get('bbox', (0, 0, 0, 0))
            confidence = face.get('confidence', 0.0)
            method = face.get('method', 'unknown')
            
            x, y, w, h = bbox
            
            # Draw face bounding box - CYAN color
            cv2.rectangle(vis_image, (x, y), (x + w, y + h), (255, 255, 0), 3)
            
            # Draw confidence label
            label = f"Face {i}: {confidence:.2f} ({method})"
            cv2.putText(vis_image, label, (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            
            # Draw face ID if available
            face_id = face.get('face_id', f'face_{i}')
            cv2.putText(vis_image, face_id, (x, y + h + 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        return vis_image
    
    def test_image_at_resolution(self, image: np.ndarray, resolution: Tuple[int, int], 
                                image_name: str) -> Dict[str, Any]:
        """
        Test face detection at a specific resolution.
        
        Args:
            image: Original image array
            resolution: Target resolution (width, height)
            image_name: Name of the image file
            
        Returns:
            Dictionary with detection results
        """
        width, height = resolution
        
        # Resize image to target resolution
        resized_image = self.resize_image_to_resolution(image, width, height)
        
        # Measure detection time
        start_time = time.time()
        
        # Detect faces using the face detector
        # We need to create a temporary file for the face detector
        temp_path = Path(f"/tmp/temp_face_test_{image_name}_{width}x{height}.jpg")
        cv2.imwrite(str(temp_path), cv2.cvtColor(resized_image, cv2.COLOR_RGB2BGR))
        
        try:
            faces = self.face_detector.detect_faces(temp_path)
        except Exception as e:
            logger.error(f"Face detection error: {e}")
            faces = []
        finally:
            # Clean up temp file
            if temp_path.exists():
                temp_path.unlink()
        
        detection_time = time.time() - start_time
        
        # Calculate detection metrics
        total_faces = len(faces)
        high_confidence_faces = sum(1 for face in faces if face.get('confidence', 0) > 0.5)
        avg_confidence = np.mean([face.get('confidence', 0) for face in faces]) if faces else 0
        
        # Count faces by detection method
        method_counts = {}
        for face in faces:
            method = face.get('method', 'unknown')
            method_counts[method] = method_counts.get(method, 0) + 1
        
        return {
            'resolution': f"{width}x{height}",
            'width': width,
            'height': height,
            'total_faces': total_faces,
            'high_confidence_faces': high_confidence_faces,
            'avg_confidence': float(avg_confidence),
            'detection_time': detection_time,
            'method_counts': method_counts,
            'image_name': image_name,
            'faces': faces  # Include face data for visualization
        }
    
    def run_resolution_test(self, input_pattern: str, output_dir: Path, 
                           num_images: int = 10, verbose: bool = False) -> None:
        """
        Run face detection resolution test.
        
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
        
        # Create annotated images directory
        annotated_dir = output_dir / 'annotated_images'
        annotated_dir.mkdir(parents=True, exist_ok=True)
        
        all_results = []
        resolution_summaries = {}
        
        # Initialize resolution summaries
        for width, height in self.test_resolutions:
            res_key = f"{width}x{height}"
            resolution_summaries[res_key] = {
                'total_faces': 0,
                'high_confidence_faces': 0,
                'total_images': 0,
                'total_time': 0.0,
                'method_counts': {}
            }
        
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
        
        logger.info(f"Testing {len(image_paths)} images across {len(self.test_resolutions)} resolutions")
        
        # Process each image
        for image_path in tqdm(image_paths, desc="Testing images"):
            logger.debug(f"Processing {image_path.name}")
            
            # Load image
            image = self.image_processor.load_image(image_path)
            if image is None:
                logger.warning(f"Failed to load image: {image_path}")
                continue
            
            image_results = []
            
            # Test at each resolution
            for width, height in self.test_resolutions:
                logger.debug(f"Testing {image_path.name} at {width}x{height}")
                
                result = self.test_image_at_resolution(image, (width, height), image_path.name)
                image_results.append(result)
                
                # Update resolution summary
                res_key = f"{width}x{height}"
                resolution_summaries[res_key]['total_faces'] += result['total_faces']
                resolution_summaries[res_key]['high_confidence_faces'] += result['high_confidence_faces']
                resolution_summaries[res_key]['total_images'] += 1
                resolution_summaries[res_key]['total_time'] += result['detection_time']
                
                # Update method counts
                for method, count in result['method_counts'].items():
                    resolution_summaries[res_key]['method_counts'][method] = \
                        resolution_summaries[res_key]['method_counts'].get(method, 0) + count
                
                # Generate annotated image
                annotated_image = self._draw_face_detections(image, result['faces'])
                annotated_filename = f"{image_path.stem}_{width}x{height}_annotated.jpg"
                annotated_path = annotated_dir / annotated_filename
                cv2.imwrite(str(annotated_path), cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
                logger.debug(f"Saved annotated image: {annotated_filename}")
            
            all_results.append({
                'image_name': image_path.name,
                'image_path': str(image_path),
                'results': image_results
            })
        
        # Calculate summary statistics
        for res_key, summary in resolution_summaries.items():
            if summary['total_images'] > 0:
                summary['avg_faces_per_image'] = summary['total_faces'] / summary['total_images']
                summary['avg_high_confidence_faces'] = summary['high_confidence_faces'] / summary['total_images']
                summary['avg_time_per_image'] = summary['total_time'] / summary['total_images']
        
        # Prepare results data
        results_data = {
            'test_info': {
                'input_pattern': input_pattern,
                'num_images_tested': len(image_paths),
                'resolutions_tested': len(self.test_resolutions),
                'cuda_available': self.cuda_available,
                'test_timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            },
            'resolution_summaries': resolution_summaries,
            'detailed_results': all_results
        }
        
        # Save results to JSON
        results_file = output_dir / 'face_resolution_test_results.json'
        with open(results_file, 'w') as f:
            json.dump(results_data, f, indent=2, cls=NumpyEncoder)
        
        logger.info(f"Results saved to {results_file}")
        
        # Print summary
        self._print_summary(resolution_summaries, output_dir)
    
    def _print_summary(self, resolution_summaries: Dict, output_dir: Path) -> None:
        """Print test summary."""
        print("\n🎯 FACE RESOLUTION TEST RESULTS")
        print("=" * 50)
        
        # Sort resolutions by average faces per image
        sorted_resolutions = sorted(
            resolution_summaries.items(),
            key=lambda x: x[1].get('avg_faces_per_image', 0),
            reverse=True
        )
        
        print("📈 TOP PERFORMING RESOLUTIONS:")
        print("-" * 30)
        
        for i, (res_key, summary) in enumerate(sorted_resolutions[:5], 1):
            avg_faces = summary.get('avg_faces_per_image', 0)
            avg_time = summary.get('avg_time_per_image', 0)
            avg_conf_faces = summary.get('avg_high_confidence_faces', 0)
            
            print(f"{i}. {res_key}:")
            print(f"   - Avg faces per image: {avg_faces:.1f}")
            print(f"   - Avg high-confidence faces: {avg_conf_faces:.1f}")
            print(f"   - Avg time per image: {avg_time:.3f}s")
            
            # Show method breakdown
            methods = summary.get('method_counts', {})
            if methods:
                method_str = ", ".join([f"{method}: {count}" for method, count in methods.items()])
                print(f"   - Detection methods: {method_str}")
            print()


@click.command()
@click.argument('input_pattern', type=str)
@click.argument('output_dir', type=click.Path())
@click.option('--num-images', '-n', default=10, help='Number of random images to test')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
def main(input_pattern: str, output_dir: str, num_images: int, verbose: bool):
    """
    Test face detection performance across different image resolutions.
    
    INPUT_PATTERN: Glob pattern for input images (e.g., "/path/to/images/*.jpg")
    OUTPUT_DIR: Directory to save test results and annotated images
    """
    output_path = Path(output_dir)
    
    # Initialize tester
    tester = FaceResolutionTester()
    
    # Run test
    tester.run_resolution_test(input_pattern, output_path, num_images, verbose)
    
    print(f"\n🎯 FACE RESOLUTION TEST RESULTS")
    print("=" * 50)
    print(f"📊 Images tested: {num_images}")
    print(f"📏 Resolutions tested: {len(tester.test_resolutions)}")
    print(f"💾 Results saved to: {output_path}")
    print(f"🖼️ Annotated images saved to: {output_path}/annotated_images/")


if __name__ == '__main__':
    main()
