#!/usr/bin/env python3
"""
Pose Detection Resolution Test

This script tests pose detection performance across different image resolutions
to determine the optimal size for detection accuracy, similar to facial recognition
improvements with downsampling.

The hypothesis is that MediaPipe pose detection models were trained on common
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
import mediapipe as mp


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle numpy types and MediaPipe objects."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, '__class__') and 'mediapipe' in str(obj.__class__):
            # Skip MediaPipe objects
            return None
        return super(NumpyEncoder, self).default(obj)

# Import our pose detection system
from src.soccer_photo_sorter.detectors.pose_detector import PoseDetector
from src.soccer_photo_sorter.core.image_processor import ImageProcessor
from src.soccer_photo_sorter.config.settings import Settings
from src.soccer_photo_sorter.utils.cuda_utils import CudaManager


class PoseResolutionTester:
    """Test pose detection across different image resolutions."""
    
    def __init__(self, settings: Settings):
        """Initialize the resolution tester."""
        self.settings = settings
        self.cuda_manager = CudaManager()
        self.image_processor = ImageProcessor(settings, self.cuda_manager)
        
        # Initialize pose detector with low confidence for maximum sensitivity
        self.pose_detector = PoseDetector(
            confidence_threshold=0.1,  # Very low threshold to catch all detections
            min_detection_confidence=0.1,
            min_tracking_confidence=0.1
        )
        
        # Initialize MediaPipe drawing utilities for visualization
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Define test resolutions (width, height)
        self.test_resolutions = [
            (1280, 720),   # 720p HD
            (1920, 1080),  # 1080p FHD
            (2560, 1440),  # 1440p QHD
            (3840, 2160),  # 4K UHD
            (640, 480),    # VGA
            (800, 600),    # SVGA
            (1024, 768),   # XGA
            (1366, 768),   # HD
            (1600, 900),   # HD+
            (2048, 1152),  # 2K
            (3200, 1800),  # 3K
            (5120, 2880),  # 5K
        ]
        
        logger.info(f"Initialized PoseResolutionTester with {len(self.test_resolutions)} test resolutions")
    
    def resize_image(self, image: np.ndarray, target_width: int, target_height: int) -> np.ndarray:
        """
        Resize image to target dimensions while maintaining aspect ratio.
        
        Args:
            image: Input image array
            target_width: Target width
            target_height: Target height
            
        Returns:
            Resized image array
        """
        original_height, original_width = image.shape[:2]
        
        # Calculate scaling factor to fit within target dimensions
        scale_w = target_width / original_width
        scale_h = target_height / original_height
        scale = min(scale_w, scale_h)
        
        # Calculate new dimensions
        new_width = int(original_width * scale)
        new_height = int(original_height * scale)
        
        # Resize image
        resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        # Create canvas with target dimensions
        canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)
        
        # Center the resized image on the canvas
        y_offset = (target_height - new_height) // 2
        x_offset = (target_width - new_width) // 2
        canvas[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized
        
        return canvas
    
    def _draw_pose_landmarks(self, image: np.ndarray, poses: List[Dict[str, Any]]) -> np.ndarray:
        """Draw pose landmarks and connections on image."""
        vis_image = image.copy()
        
        # Define pose connections (MediaPipe pose connections)
        pose_connections = [
            (0, 1), (1, 2), (2, 3), (3, 7),  # Face
            (0, 4), (4, 5), (5, 6), (6, 8),  # Face
            (9, 10),  # Mouth
            (11, 12),  # Shoulders
            (11, 13), (13, 15),  # Left arm
            (12, 14), (14, 16),  # Right arm
            (11, 23), (12, 24),  # Torso
            (23, 24),  # Hips
            (23, 25), (25, 27), (27, 29), (29, 31),  # Left leg
            (24, 26), (26, 28), (28, 30), (30, 32),  # Right leg
        ]
        
        for person_id, pose_data in enumerate(poses):
            keypoints = pose_data.get('keypoints', {})
            
            # Draw pose landmarks
            visible_landmarks = []
            for kp_id in range(33):  # MediaPipe has 33 pose landmarks
                if kp_id in keypoints:
                    kp = keypoints[kp_id]
                    # Check visibility threshold (MediaPipe uses visibility score)
                    if kp.visibility > 0.5:  # Threshold for visibility
                        x = int(kp.x)
                        y = int(kp.y)
                        visible_landmarks.append((kp_id, x, y))
                        
                        # Draw landmark point - CYAN color
                        cv2.circle(vis_image, (x, y), 8, (255, 255, 0), -1)
            
            # Draw pose connections
            for connection in pose_connections:
                start_id, end_id = connection
                start_kp = next((kp for kp in visible_landmarks if kp[0] == start_id), None)
                end_kp = next((kp for kp in visible_landmarks if kp[0] == end_id), None)
                
                if start_kp and end_kp:
                    cv2.line(vis_image, (start_kp[1], start_kp[2]), (end_kp[1], end_kp[2]), (255, 255, 0), 4)
            
            # Add person ID label
            if visible_landmarks:
                # Use nose position (landmark 0) for label
                nose_kp = next((kp for kp in visible_landmarks if kp[0] == 0), None)
                if nose_kp:
                    nose_x, nose_y = nose_kp[1], nose_kp[2]
                    cv2.putText(vis_image, f"Person {person_id}", (nose_x - 30, nose_y - 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 3)
        
        return vis_image
    
    def _draw_bounding_boxes(self, image: np.ndarray, poses: List[Dict[str, Any]], 
                           jersey_regions: List, body_regions: List) -> np.ndarray:
        """Draw bounding boxes for humans and jerseys."""
        vis_image = image.copy()
        
        # Draw jersey bounding boxes (GREEN)
        for region in jersey_regions:
            if hasattr(region, 'x'):  # JerseyRegion namedtuple
                x, y, width, height = region.x, region.y, region.width, region.height
                person_id = region.person_id
                confidence = region.confidence
            else:  # Dict format
                bbox = region.get('bounding_box', {})
                if bbox:
                    x = int(bbox['x'])
                    y = int(bbox['y'])
                    width = int(bbox['width'])
                    height = int(bbox['height'])
                    person_id = region.get('person_id', 0)
                    confidence = region.get('confidence', 0.0)
                else:
                    continue
            
            # Draw jersey bounding box (thick green)
            cv2.rectangle(vis_image, (x, y), (x + width, y + height), (0, 255, 0), 6)
            
            # Draw label
            label = f"Jersey P{person_id} ({confidence:.2f})"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 3)[0]
            
            # Draw label background
            cv2.rectangle(vis_image, (x, y - label_size[1] - 20), 
                         (x + label_size[0], y), (0, 255, 0), -1)
            
            # Draw label text
            cv2.putText(vis_image, label, (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 3)
        
        # Draw body bounding boxes (RED)
        for region in body_regions:
            bbox = region.get('bounding_box', {})
            if bbox:
                x = int(bbox['x'])
                y = int(bbox['y'])
                width = int(bbox['width'])
                height = int(bbox['height'])
                person_id = region.get('person_id', 0)
                confidence = region.get('confidence', 0.0)
                
                # Draw body bounding box (thick red)
                cv2.rectangle(vis_image, (x, y), (x + width, y + height), (255, 0, 0), 6)
                
                # Draw label
                label = f"Body P{person_id} ({confidence:.2f})"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 3)[0]
                
                # Draw label background
                cv2.rectangle(vis_image, (x, y + height), 
                             (x + label_size[0], y + height + label_size[1] + 20), (255, 0, 0), -1)
                
                # Draw label text
                cv2.putText(vis_image, label, (x, y + height + 15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
        
        return vis_image
    
    def create_annotated_image(self, image: np.ndarray, poses: List[Dict[str, Any]], 
                             jersey_regions: List, body_regions: List) -> np.ndarray:
        """Create fully annotated image with poses, bounding boxes, and labels."""
        # Start with pose landmarks
        vis_image = self._draw_pose_landmarks(image, poses)
        
        # Add bounding boxes
        vis_image = self._draw_bounding_boxes(vis_image, poses, jersey_regions, body_regions)
        
        return vis_image
    
    def test_image_at_resolution(self, image: np.ndarray, resolution: Tuple[int, int], 
                                image_name: str) -> Dict[str, Any]:
        """
        Test pose detection on an image at a specific resolution.
        
        Args:
            image: Original image array
            resolution: Target resolution (width, height)
            image_name: Name of the image file
            
        Returns:
            Dictionary with detection results
        """
        width, height = resolution
        
        # Resize image
        resized_image = self.resize_image(image, width, height)
        
        # Test pose detection
        start_time = time.time()
        poses = self.pose_detector.detect_poses(resized_image)
        detection_time = time.time() - start_time
        
        # Extract jersey regions
        jersey_regions = self.pose_detector.extract_jersey_regions(resized_image)
        
        # Extract body regions
        body_regions = self.pose_detector.extract_body_regions(resized_image)
        
        # Calculate detection metrics
        total_poses = len(poses)
        visible_poses = sum(1 for pose in poses if pose.get('confidence', 0) > 0.5)
        avg_confidence = np.mean([pose.get('confidence', 0) for pose in poses]) if poses else 0
        
        return {
            'resolution': f"{width}x{height}",
            'width': width,
            'height': height,
            'total_poses': total_poses,
            'visible_poses': visible_poses,
            'avg_confidence': float(avg_confidence),
            'jersey_regions': len(jersey_regions),
            'body_regions': len(body_regions),
            'detection_time': detection_time,
            'image_name': image_name,
            'poses': poses,
            'jersey_regions_data': jersey_regions,
            'body_regions_data': body_regions,
            'resized_image': resized_image
        }
    
    def run_resolution_test(self, image_paths: List[Path], output_dir: Path) -> Dict[str, Any]:
        """
        Run resolution test on multiple images.
        
        Args:
            image_paths: List of image file paths
            output_dir: Output directory for results
            
        Returns:
            Dictionary with comprehensive test results
        """
        logger.info(f"Starting resolution test with {len(image_paths)} images")
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create annotated images directory
        annotated_dir = output_dir / 'annotated_images'
        annotated_dir.mkdir(parents=True, exist_ok=True)
        
        all_results = []
        resolution_summaries = {}
        
        # Initialize resolution summaries
        for width, height in self.test_resolutions:
            resolution_summaries[f"{width}x{height}"] = {
                'total_poses': 0,
                'visible_poses': 0,
                'total_images': 0,
                'total_time': 0,
                'jersey_regions': 0,
                'body_regions': 0,
                'confidences': []
            }
        
        # Process each image
        for image_path in tqdm(image_paths, desc="Testing images"):
            logger.info(f"Processing {image_path.name}")
            
            # Load image
            image = self.image_processor.load_image(image_path)
            if image is None:
                logger.error(f"Failed to load image: {image_path}")
                continue
            
            image_results = []
            
            # Test at each resolution
            for width, height in self.test_resolutions:
                logger.debug(f"Testing {image_path.name} at {width}x{height}")
                
                result = self.test_image_at_resolution(image, (width, height), image_path.name)
                image_results.append(result)
                
                # Create annotated image
                if result['poses'] or result['jersey_regions_data'] or result['body_regions_data']:
                    annotated_image = self.create_annotated_image(
                        result['resized_image'],
                        result['poses'],
                        result['jersey_regions_data'],
                        result['body_regions_data']
                    )
                    
                    # Save annotated image
                    base_name = image_path.stem
                    annotated_filename = f"{base_name}_{width}x{height}_annotated.jpg"
                    annotated_path = annotated_dir / annotated_filename
                    cv2.imwrite(str(annotated_path), annotated_image)
                    logger.debug(f"Saved annotated image: {annotated_filename}")
                
                # Update resolution summary
                res_key = f"{width}x{height}"
                resolution_summaries[res_key]['total_poses'] += result['total_poses']
                resolution_summaries[res_key]['visible_poses'] += result['visible_poses']
                resolution_summaries[res_key]['total_images'] += 1
                resolution_summaries[res_key]['total_time'] += result['detection_time']
                resolution_summaries[res_key]['jersey_regions'] += result['jersey_regions']
                resolution_summaries[res_key]['body_regions'] += result['body_regions']
                resolution_summaries[res_key]['confidences'].append(result['avg_confidence'])
            
            all_results.append({
                'image_name': image_path.name,
                'original_size': f"{image.shape[1]}x{image.shape[0]}",
                'results': image_results
            })
        
        # Calculate summary statistics
        for res_key, summary in resolution_summaries.items():
            if summary['total_images'] > 0:
                summary['avg_poses_per_image'] = summary['total_poses'] / summary['total_images']
                summary['avg_visible_poses_per_image'] = summary['visible_poses'] / summary['total_images']
                summary['avg_time_per_image'] = summary['total_time'] / summary['total_images']
                summary['avg_jersey_regions_per_image'] = summary['jersey_regions'] / summary['total_images']
                summary['avg_body_regions_per_image'] = summary['body_regions'] / summary['total_images']
                summary['avg_confidence'] = np.mean(summary['confidences']) if summary['confidences'] else 0
                summary['std_confidence'] = np.std(summary['confidences']) if summary['confidences'] else 0
        
        # Save detailed results
        results_data = {
            'test_summary': {
                'total_images_tested': len(image_paths),
                'total_resolutions_tested': len(self.test_resolutions),
                'test_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'test_resolutions': [f"{w}x{h}" for w, h in self.test_resolutions]
            },
            'resolution_summaries': resolution_summaries,
            'detailed_results': all_results
        }
        
        # Save results to JSON
        results_file = output_dir / 'pose_resolution_test_results.json'
        with open(results_file, 'w') as f:
            json.dump(results_data, f, indent=2, cls=NumpyEncoder)
        
        logger.info(f"Results saved to {results_file}")
        
        return results_data


@click.command()
@click.argument('input_pattern', type=str)
@click.argument('output_dir', type=click.Path())
@click.option('--num-images', '-n', default=10, help='Number of random images to test')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.option('--save-annotations', '-a', is_flag=True, help='Save annotated images with stick figures and bounding boxes')
def main(input_pattern: str, output_dir: str, num_images: int, verbose: bool, save_annotations: bool):
    """
    Test pose detection performance across different image resolutions.
    
    INPUT_PATTERN: Glob pattern for input images (e.g., '/path/to/images/*.jpg')
    OUTPUT_DIR: Directory to save test results
    """
    if verbose:
        logger.add("pose_resolution_test.log", level="DEBUG")
    
    # Initialize settings
    settings = Settings()
    
    # Find images matching the pattern
    input_path = Path(input_pattern)
    if input_path.is_file():
        image_paths = [input_path]
    else:
        # Use glob pattern
        image_paths = list(Path(input_pattern).parent.glob(input_pattern.split('/')[-1]))
    
    if not image_paths:
        logger.error(f"No images found matching pattern: {input_pattern}")
        return
    
    # Select random subset
    if len(image_paths) > num_images:
        image_paths = random.sample(image_paths, num_images)
    
    logger.info(f"Selected {len(image_paths)} images for testing")
    for img_path in image_paths:
        logger.info(f"  - {img_path.name}")
    
    # Initialize tester
    tester = PoseResolutionTester(settings)
    
    # Run test
    output_path = Path(output_dir)
    results = tester.run_resolution_test(image_paths, output_path)
    
    # Print summary
    print("\n🎯 POSE RESOLUTION TEST RESULTS")
    print("=" * 50)
    print(f"📊 Images tested: {len(image_paths)}")
    print(f"📏 Resolutions tested: {len(tester.test_resolutions)}")
    print(f"💾 Results saved to: {output_path}")
    print(f"🖼️ Annotated images saved to: {output_path}/annotated_images/")
    
    print("\n📈 TOP PERFORMING RESOLUTIONS:")
    print("-" * 30)
    
    # Sort resolutions by average poses detected
    sorted_resolutions = sorted(
        results['resolution_summaries'].items(),
        key=lambda x: x[1]['avg_poses_per_image'],
        reverse=True
    )
    
    for i, (resolution, stats) in enumerate(sorted_resolutions[:5]):
        print(f"{i+1}. {resolution}:")
        print(f"   - Avg poses per image: {stats['avg_poses_per_image']:.2f}")
        print(f"   - Avg confidence: {stats['avg_confidence']:.3f}")
        print(f"   - Avg time per image: {stats['avg_time_per_image']:.3f}s")
        print(f"   - Jersey regions: {stats['avg_jersey_regions_per_image']:.2f}")
        print()


if __name__ == '__main__':
    main()
