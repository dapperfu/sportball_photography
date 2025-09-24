#!/usr/bin/env python3
"""
Standalone Comprehensive Multi-Resolution Analysis Tool

This tool performs comprehensive analysis of face and pose detection performance
across multiple resolutions, generating annotated images for human evaluation.

Author: Claude Sonnet 4 (claude-3-5-sonnet-20241022)
Generated via Cursor IDE (cursor.sh) with AI assistance
"""

import json
import logging
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import click
import cv2
import numpy as np
from tqdm import tqdm
from loguru import logger
import mediapipe as mp


@dataclass
class DetectionResult:
    """Result of detection at a specific resolution."""
    resolution: str
    width: int
    height: int
    face_count: int
    pose_count: int
    face_confidences: List[float]
    pose_confidences: List[float]
    processing_time: float
    success: bool
    error: Optional[str] = None


@dataclass
class ImageAnalysis:
    """Complete analysis of an image across all resolutions."""
    filename: str
    original_size: Tuple[int, int]
    results: List[DetectionResult]
    best_face_resolution: Optional[str] = None
    best_pose_resolution: Optional[str] = None


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle numpy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


class StandaloneAnalyzer:
    """Standalone analyzer for multi-resolution detection testing."""
    
    def __init__(self):
        """Initialize the analyzer with detectors."""
        # Initialize MediaPipe pose detection
        self.mp_pose = mp.solutions.pose
        self.pose_detector = self.mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            enable_segmentation=False,
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3
        )
        
        # Initialize OpenCV face detection
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Define target resolutions for analysis
        self.resolutions = {
            '480p': (854, 480),
            '720p': (1280, 720),
            '900p': (1600, 900),
            '1080p': (1920, 1080),
            '1440p': (2560, 1440),
            '1800p': (3200, 1800),
            '4K': (3840, 2160),
            '5K': (5120, 2880)
        }
        
        logger.info("Initialized standalone analyzer")
    
    def analyze_image(self, image_path: Path) -> ImageAnalysis:
        """
        Analyze a single image across all resolutions.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Complete analysis results
        """
        logger.info(f"Analyzing {image_path.name}")
        
        # Load original image
        original_image = cv2.imread(str(image_path))
        if original_image is None:
            logger.error(f"Failed to load image: {image_path}")
            return ImageAnalysis(
                filename=image_path.name,
                original_size=(0, 0),
                results=[],
                error="Failed to load image"
            )
        
        original_height, original_width = original_image.shape[:2]
        results = []
        
        # Analyze at each resolution
        for res_name, (target_width, target_height) in self.resolutions.items():
            try:
                # Resize image
                resized_image = cv2.resize(original_image, (target_width, target_height))
                
                # Detect faces
                face_start = cv2.getTickCount()
                gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
                face_time = (cv2.getTickCount() - face_start) / cv2.getTickFrequency()
                
                # Detect poses
                pose_start = cv2.getTickCount()
                rgb_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
                pose_results = self.pose_detector.process(rgb_image)
                pose_time = (cv2.getTickCount() - pose_start) / cv2.getTickFrequency()
                
                # Extract results
                face_count = len(faces)
                face_confidences = [0.8] * face_count  # OpenCV doesn't provide confidence
                
                pose_count = 1 if pose_results.pose_landmarks else 0
                pose_confidences = [0.8] if pose_count > 0 else []
                
                total_time = face_time + pose_time
                
                result = DetectionResult(
                    resolution=res_name,
                    width=target_width,
                    height=target_height,
                    face_count=face_count,
                    pose_count=pose_count,
                    face_confidences=face_confidences,
                    pose_confidences=pose_confidences,
                    processing_time=total_time,
                    success=True
                )
                
                results.append(result)
                logger.debug(f"{res_name}: {face_count} faces, {pose_count} poses, {total_time:.3f}s")
                
            except Exception as e:
                logger.error(f"Error analyzing {res_name}: {e}")
                results.append(DetectionResult(
                    resolution=res_name,
                    width=target_width,
                    height=target_height,
                    face_count=0,
                    pose_count=0,
                    face_confidences=[],
                    pose_confidences=[],
                    processing_time=0.0,
                    success=False,
                    error=str(e)
                ))
        
        # Determine best resolutions
        best_face_resolution = self._find_best_resolution(results, 'face')
        best_pose_resolution = self._find_best_resolution(results, 'pose')
        
        return ImageAnalysis(
            filename=image_path.name,
            original_size=(original_width, original_height),
            results=results,
            best_face_resolution=best_face_resolution,
            best_pose_resolution=best_pose_resolution
        )
    
    def _find_best_resolution(self, results: List[DetectionResult], detection_type: str) -> Optional[str]:
        """Find the best resolution for a detection type."""
        if not results:
            return None
        
        if detection_type == 'face':
            # Best resolution has most faces with good confidence
            best_res = max(results, key=lambda r: (
                r.face_count,
                np.mean(r.face_confidences) if r.face_confidences else 0
            ))
        else:  # pose
            # Best resolution has most poses with good confidence
            best_res = max(results, key=lambda r: (
                r.pose_count,
                np.mean(r.pose_confidences) if r.pose_confidences else 0
            ))
        
        return best_res.resolution if best_res.success else None
    
    def create_annotated_images(self, analysis: ImageAnalysis, output_dir: Path) -> None:
        """Create annotated images for visual evaluation."""
        logger.info(f"Creating annotated images for {analysis.filename}")
        
        # Load original image
        image_path = Path(analysis.filename)
        original_image = cv2.imread(str(image_path))
        if original_image is None:
            logger.error(f"Failed to load image: {image_path}")
            return
        
        # Create annotated images for each resolution
        for result in analysis.results:
            if not result.success:
                continue
            
            # Resize image
            resized_image = cv2.resize(original_image, (result.width, result.height))
            
            # Create face annotated image
            face_image = resized_image.copy()
            gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
            self._draw_face_annotations(face_image, faces)
            
            face_filename = f"{image_path.stem}_{result.resolution}_faces.jpg"
            cv2.imwrite(str(output_dir / "faces" / face_filename), face_image)
            
            # Create pose annotated image
            pose_image = resized_image.copy()
            rgb_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
            pose_results = self.pose_detector.process(rgb_image)
            self._draw_pose_annotations(pose_image, pose_results)
            
            pose_filename = f"{image_path.stem}_{result.resolution}_poses.jpg"
            cv2.imwrite(str(output_dir / "poses" / pose_filename), pose_image)
    
    def _draw_face_annotations(self, image: np.ndarray, faces: List[Tuple]) -> None:
        """Draw face detection annotations."""
        for i, (x, y, w, h) in enumerate(faces):
            # Draw bounding box
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Draw label
            label = f"Face {i+1}"
            cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    def _draw_pose_annotations(self, image: np.ndarray, pose_results) -> None:
        """Draw pose detection annotations."""
        if pose_results.pose_landmarks:
            landmarks = pose_results.pose_landmarks.landmark
            h, w = image.shape[:2]
            
            # Draw keypoints
            for landmark in landmarks:
                if landmark.visibility > 0.5:
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    cv2.circle(image, (x, y), 3, (255, 0, 0), -1)
            
            # Draw pose connections
            connections = self.mp_pose.POSE_CONNECTIONS
            for connection in connections:
                start_point = landmarks[connection[0]]
                end_point = landmarks[connection[1]]
                
                if start_point.visibility > 0.5 and end_point.visibility > 0.5:
                    start_x = int(start_point.x * w)
                    start_y = int(start_point.y * h)
                    end_x = int(end_point.x * w)
                    end_y = int(end_point.y * h)
                    
                    cv2.line(image, (start_x, start_y), (end_x, end_y), (255, 0, 0), 2)


@click.command()
@click.argument('input_pattern', type=str)
@click.argument('output_dir', type=str)
@click.option('--num-images', '-n', default=5, help='Number of images to analyze')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
def main(input_pattern: str, output_dir: str, num_images: int, verbose: bool):
    """Run comprehensive multi-resolution analysis."""
    
    # Setup logging
    if verbose:
        logger.add("comprehensive_analysis.log", level="DEBUG")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    (output_path / "faces").mkdir(exist_ok=True)
    (output_path / "poses").mkdir(exist_ok=True)
    
    # Find input images
    if input_pattern.startswith('/'):
        # Absolute path
        parent_dir = Path(input_pattern).parent
        pattern = Path(input_pattern).name
        image_files = list(parent_dir.glob(pattern))
    else:
        # Relative path
        image_files = list(Path('.').glob(input_pattern))
    
    if not image_files:
        logger.error(f"No images found matching pattern: {input_pattern}")
        return
    
    # Select random images
    selected_images = random.sample(image_files, min(num_images, len(image_files)))
    logger.info(f"Selected {len(selected_images)} images for analysis")
    
    # Initialize analyzer
    analyzer = StandaloneAnalyzer()
    
    # Analyze images
    all_analyses = []
    for image_path in tqdm(selected_images, desc="Analyzing images"):
        analysis = analyzer.analyze_image(image_path)
        all_analyses.append(analysis)
        
        # Create annotated images
        analyzer.create_annotated_images(analysis, output_path)
    
    # Save results
    results_data = {
        'analysis_timestamp': str(Path().cwd()),
        'total_images': len(all_analyses),
        'resolutions_tested': list(analyzer.resolutions.keys()),
        'analyses': [asdict(analysis) for analysis in all_analyses]
    }
    
    results_file = output_path / "comprehensive_analysis_results.json"
    with open(results_file, 'w') as f:
        json.dump(results_data, f, indent=2, cls=NumpyEncoder)
    
    # Generate summary
    summary_file = output_path / "analysis_summary.md"
    with open(summary_file, 'w') as f:
        f.write("# Comprehensive Multi-Resolution Analysis Results\n\n")
        f.write(f"**Analysis Date**: {Path().cwd()}\n")
        f.write(f"**Images Analyzed**: {len(all_analyses)}\n")
        f.write(f"**Resolutions Tested**: {', '.join(analyzer.resolutions.keys())}\n\n")
        
        f.write("## Summary by Resolution\n\n")
        f.write("| Resolution | Avg Faces | Avg Poses | Avg Time (s) |\n")
        f.write("|------------|-----------|-----------|--------------|\n")
        
        for res_name in analyzer.resolutions.keys():
            res_analyses = [a for a in all_analyses if any(r.resolution == res_name for r in a.results)]
            if res_analyses:
                avg_faces = np.mean([r.face_count for a in res_analyses for r in a.results if r.resolution == res_name])
                avg_poses = np.mean([r.pose_count for a in res_analyses for r in a.results if r.resolution == res_name])
                avg_time = np.mean([r.processing_time for a in res_analyses for r in a.results if r.resolution == res_name])
                f.write(f"| {res_name} | {avg_faces:.1f} | {avg_poses:.1f} | {avg_time:.3f} |\n")
        
        f.write("\n## Best Resolutions per Image\n\n")
        f.write("| Image | Best Face Resolution | Best Pose Resolution |\n")
        f.write("|-------|---------------------|---------------------|\n")
        
        for analysis in all_analyses:
            f.write(f"| {analysis.filename} | {analysis.best_face_resolution or 'N/A'} | {analysis.best_pose_resolution or 'N/A'} |\n")
        
        f.write("\n## Recommendations\n\n")
        f.write("- **Most faces detected**: Higher resolutions generally detect more faces\n")
        f.write("- **Best balance**: 720p-1080p often provides good speed/accuracy balance\n")
        f.write("- **Quality vs Speed**: Higher resolutions are slower but may detect more\n")
        f.write("- **Human evaluation**: Check annotated images in faces/ and poses/ directories\n")
        f.write("- **Remember**: Most detections doesn't always mean better quality!\n")
    
    logger.info(f"Analysis complete! Results saved to {output_path}")
    logger.info(f"Annotated images saved to {output_path}/faces/ and {output_path}/poses/")
    logger.info(f"Summary report: {summary_file}")


if __name__ == '__main__':
    main()
