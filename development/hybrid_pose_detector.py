#!/usr/bin/env python3
"""
Hybrid Human Detection + Pose Analysis using YOLOv8 + MediaPipe

This tool uses YOLOv8 to detect humans first, then applies pose detection
to each detected human's bounding box for maximum efficiency and accuracy.

Author: Claude Sonnet 4 (claude-3-5-sonnet-20241022)
Generated via Cursor IDE (cursor.sh) with AI assistance
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
import click
from loguru import logger
import json
import mediapipe as mp


class HybridPoseDetector:
    """Hybrid detector combining YOLOv8 human detection with MediaPipe pose analysis."""
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the hybrid pose detector.
        
        Args:
            model_path: Path to custom YOLO model (optional)
        """
        # Initialize YOLOv8
        try:
            from ultralytics import YOLO
            self.YOLO = YOLO
            self.yolo_available = True
        except ImportError:
            logger.error("ultralytics not installed. Install with: pip install ultralytics")
            self.yolo_available = False
            return
        
        # Load YOLO model
        if model_path and Path(model_path).exists():
            self.model = YOLO(model_path)
            logger.info(f"Loaded custom model from {model_path}")
        else:
            # Use pre-trained YOLOv8 model
            self.model = YOLO('yolov8n.pt')  # nano version for speed
            logger.info("Loaded pre-trained YOLOv8n model")
        
        # Initialize MediaPipe pose detection
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=True,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # COCO class names
        self.coco_classes = {
            0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus',
            6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant',
            11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat',
            16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear',
            22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag',
            27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard',
            32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove',
            36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle',
            40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl',
            46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli',
            51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake',
            56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table',
            61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard',
            67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink',
            72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors',
            77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'
        }
        
        # Person class ID in COCO
        self.person_class_id = 0
        
        logger.info("Initialized HybridPoseDetector with YOLOv8 + MediaPipe")
    
    def detect_humans_and_poses(self, image: np.ndarray, confidence_threshold: float = 0.5) -> Dict[str, Any]:
        """
        Detect humans using YOLOv8, then analyze poses for each detected human.
        
        Args:
            image: Input image (BGR format)
            confidence_threshold: Minimum confidence for human detection
            
        Returns:
            Dictionary with detection results
        """
        if not self.yolo_available:
            return {'humans': [], 'poses': [], 'total_humans': 0, 'total_poses': 0}
        
        try:
            # Step 1: Detect humans using YOLOv8
            results = self.model(image, conf=confidence_threshold)
            
            humans = []
            poses = []
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for i, box in enumerate(boxes):
                        # Get class ID and confidence
                        class_id = int(box.cls[0])
                        confidence = float(box.conf[0])
                        
                        # Check if it's a person
                        if class_id == self.person_class_id:
                            # Get bounding box coordinates
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                            
                            # Calculate center and dimensions
                            center_x = (x1 + x2) // 2
                            center_y = (y1 + y2) // 2
                            width = x2 - x1
                            height = y2 - y1
                            
                            # Analyze person characteristics
                            person_info = self._analyze_person(image, x1, y1, x2, y2)
                            
                            human_data = {
                                'bbox': (x1, y1, width, height),
                                'center': (center_x, center_y),
                                'confidence': confidence,
                                'method': 'yolov8',
                                'class_name': 'person',
                                'person_id': len(humans) + 1,
                                **person_info
                            }
                            humans.append(human_data)
                            
                            # Step 2: Extract human region and analyze pose
                            pose_data = self._analyze_pose_in_region(image, x1, y1, x2, y2, len(humans))
                            if pose_data:
                                poses.append(pose_data)
            
            return {
                'humans': humans,
                'poses': poses,
                'total_humans': len(humans),
                'total_poses': len(poses)
            }
            
        except Exception as e:
            logger.error(f"Hybrid detection failed: {e}")
            return {'humans': [], 'poses': [], 'total_humans': 0, 'total_poses': 0}
    
    def _analyze_person(self, image: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> Dict[str, Any]:
        """Analyze person characteristics from detected region."""
        # Extract person region
        person_region = image[y1:y2, x1:x2]
        
        if person_region.size == 0:
            return {
                'size_category': 'unknown',
                'position': 'unknown',
                'color_dominant': 'unknown',
                'aspect_ratio': 0.0
            }
        
        # Calculate aspect ratio
        width = x2 - x1
        height = y2 - y1
        aspect_ratio = height / width if width > 0 else 0.0
        
        # Determine size category based on bounding box area
        image_area = image.shape[0] * image.shape[1]
        person_area = width * height
        area_ratio = person_area / image_area
        
        if area_ratio > 0.1:
            size_category = 'large'
        elif area_ratio > 0.05:
            size_category = 'medium'
        elif area_ratio > 0.01:
            size_category = 'small'
        else:
            size_category = 'tiny'
        
        # Determine position in image
        img_height, img_width = image.shape[:2]
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        
        if center_x < img_width // 3:
            horizontal_pos = 'left'
        elif center_x > 2 * img_width // 3:
            horizontal_pos = 'right'
        else:
            horizontal_pos = 'center'
        
        if center_y < img_height // 3:
            vertical_pos = 'top'
        elif center_y > 2 * img_height // 3:
            vertical_pos = 'bottom'
        else:
            vertical_pos = 'middle'
        
        position = f"{vertical_pos}_{horizontal_pos}"
        
        # Analyze dominant color in person region
        hsv_region = cv2.cvtColor(person_region, cv2.COLOR_BGR2HSV)
        hist_h = cv2.calcHist([hsv_region], [0], None, [180], [0, 180])
        dominant_hue = np.argmax(hist_h)
        
        # Map hue to color names
        if dominant_hue < 10 or dominant_hue > 170:
            color_dominant = 'red'
        elif 10 <= dominant_hue <= 25:
            color_dominant = 'orange'
        elif 25 <= dominant_hue <= 35:
            color_dominant = 'yellow'
        elif 35 <= dominant_hue <= 85:
            color_dominant = 'green'
        elif 85 <= dominant_hue <= 130:
            color_dominant = 'blue'
        elif 130 <= dominant_hue <= 170:
            color_dominant = 'purple'
        else:
            color_dominant = 'unknown'
        
        return {
            'size_category': size_category,
            'position': position,
            'color_dominant': color_dominant,
            'aspect_ratio': aspect_ratio,
            'area_ratio': area_ratio
        }
    
    def _analyze_pose_in_region(self, image: np.ndarray, x1: int, y1: int, x2: int, y2: int, person_id: int) -> Optional[Dict[str, Any]]:
        """
        Analyze pose within a specific human bounding box region.
        
        Args:
            image: Original image
            x1, y1, x2, y2: Bounding box coordinates
            person_id: ID of the person
            
        Returns:
            Pose analysis results or None if no pose detected
        """
        try:
            # Extract human region with some padding
            padding = 20  # Add padding around the bounding box
            x1_padded = max(0, x1 - padding)
            y1_padded = max(0, y1 - padding)
            x2_padded = min(image.shape[1], x2 + padding)
            y2_padded = min(image.shape[0], y2 + padding)
            
            human_region = image[y1_padded:y2_padded, x1_padded:x2_padded]
            
            if human_region.size == 0:
                return None
            
            # Convert BGR to RGB for MediaPipe
            rgb_region = cv2.cvtColor(human_region, cv2.COLOR_BGR2RGB)
            
            # Run pose detection on the human region
            pose_results = self.pose.process(rgb_region)
            
            if pose_results.pose_landmarks:
                # Extract pose landmarks
                landmarks = pose_results.pose_landmarks.landmark
                
                # Convert landmarks to image coordinates (relative to the human region)
                pose_points = []
                for landmark in landmarks:
                    if landmark.visibility > 0.5:  # Only include visible landmarks
                        # Convert relative coordinates to absolute coordinates
                        abs_x = int(landmark.x * human_region.shape[1]) + x1_padded
                        abs_y = int(landmark.y * human_region.shape[0]) + y1_padded
                        pose_points.append({
                            'x': abs_x,
                            'y': abs_y,
                            'visibility': landmark.visibility,
                            'z': landmark.z
                        })
                    else:
                        pose_points.append(None)
                
                # Analyze pose characteristics
                pose_analysis = self._analyze_pose_characteristics(pose_points, x1, y1, x2, y2)
                
                return {
                    'person_id': person_id,
                    'bbox': (x1, y1, x2 - x1, y2 - y1),
                    'landmarks': pose_points,
                    'total_landmarks': len([p for p in pose_points if p is not None]),
                    'method': 'mediapipe',
                    **pose_analysis
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Pose analysis failed for person {person_id}: {e}")
            return None
    
    def _analyze_pose_characteristics(self, landmarks: List[Optional[Dict]], x1: int, y1: int, x2: int, y2: int) -> Dict[str, Any]:
        """Analyze pose characteristics from landmarks."""
        # Filter out None landmarks
        valid_landmarks = [lm for lm in landmarks if lm is not None]
        
        if len(valid_landmarks) < 10:  # Need minimum landmarks for analysis
            return {
                'pose_quality': 'poor',
                'pose_type': 'unknown',
                'body_orientation': 'unknown',
                'arm_positions': 'unknown',
                'leg_positions': 'unknown'
            }
        
        # Analyze pose quality based on landmark visibility
        avg_visibility = sum(lm['visibility'] for lm in valid_landmarks) / len(valid_landmarks)
        
        if avg_visibility > 0.8:
            pose_quality = 'excellent'
        elif avg_visibility > 0.6:
            pose_quality = 'good'
        elif avg_visibility > 0.4:
            pose_quality = 'fair'
        else:
            pose_quality = 'poor'
        
        # Analyze body orientation (simplified)
        # Use nose, left shoulder, right shoulder for orientation
        nose = landmarks[0] if landmarks[0] else None
        left_shoulder = landmarks[11] if landmarks[11] else None
        right_shoulder = landmarks[12] if landmarks[12] else None
        
        if nose and left_shoulder and right_shoulder:
            # Calculate shoulder angle
            shoulder_angle = np.arctan2(
                right_shoulder['y'] - left_shoulder['y'],
                right_shoulder['x'] - left_shoulder['x']
            ) * 180 / np.pi
            
            if abs(shoulder_angle) < 30:
                body_orientation = 'facing_camera'
            elif abs(shoulder_angle) > 150:
                body_orientation = 'facing_away'
            else:
                body_orientation = 'side_view'
        else:
            body_orientation = 'unknown'
        
        # Analyze arm positions (simplified)
        left_wrist = landmarks[15] if landmarks[15] else None
        right_wrist = landmarks[16] if landmarks[16] else None
        left_elbow = landmarks[13] if landmarks[13] else None
        right_elbow = landmarks[14] if landmarks[14] else None
        
        arm_positions = []
        if left_wrist and left_elbow:
            if left_wrist['y'] < left_elbow['y']:
                arm_positions.append('left_raised')
        if right_wrist and right_elbow:
            if right_wrist['y'] < right_elbow['y']:
                arm_positions.append('right_raised')
        
        if not arm_positions:
            arm_positions = ['both_down']
        
        # Analyze leg positions (simplified)
        left_ankle = landmarks[27] if landmarks[27] else None
        right_ankle = landmarks[28] if landmarks[28] else None
        left_knee = landmarks[25] if landmarks[25] else None
        right_knee = landmarks[26] if landmarks[26] else None
        
        leg_positions = []
        if left_ankle and left_knee:
            if left_ankle['y'] < left_knee['y']:
                leg_positions.append('left_raised')
        if right_ankle and right_knee:
            if right_ankle['y'] < right_knee['y']:
                leg_positions.append('right_raised')
        
        if not leg_positions:
            leg_positions = ['both_down']
        
        # Determine pose type based on arm and leg positions
        if 'left_raised' in arm_positions or 'right_raised' in arm_positions:
            if 'left_raised' in leg_positions or 'right_raised' in leg_positions:
                pose_type = 'dynamic'
            else:
                pose_type = 'arm_action'
        elif 'left_raised' in leg_positions or 'right_raised' in leg_positions:
            pose_type = 'leg_action'
        else:
            pose_type = 'static'
        
        return {
            'pose_quality': pose_quality,
            'pose_type': pose_type,
            'body_orientation': body_orientation,
            'arm_positions': arm_positions,
            'leg_positions': leg_positions,
            'avg_visibility': avg_visibility
        }
    
    def draw_detections(self, image: np.ndarray, detections: Dict[str, Any]) -> np.ndarray:
        """Draw both human detections and pose landmarks on the image."""
        annotated = image.copy()
        
        humans = detections.get('humans', [])
        poses = detections.get('poses', [])
        
        # Color mapping for different categories
        human_colors = {
            'large': (0, 255, 0),      # Green
            'medium': (0, 165, 255),   # Orange
            'small': (0, 255, 255),    # Yellow
            'tiny': (255, 0, 255),     # Magenta
            'unknown': (255, 255, 255)  # White
        }
        
        pose_colors = {
            'excellent': (0, 255, 0),  # Green
            'good': (0, 165, 255),     # Orange
            'fair': (0, 255, 255),     # Yellow
            'poor': (255, 0, 255),     # Magenta
            'unknown': (255, 255, 255)  # White
        }
        
        # Draw humans
        for human in humans:
            x1, y1, w, h = human['bbox']
            x2, y2 = x1 + w, y1 + h
            confidence = human['confidence']
            size_category = human['size_category']
            position = human['position']
            color_dominant = human['color_dominant']
            person_id = human.get('person_id', 0)
            
            # Get color for this person's size category
            color = human_colors.get(size_category, (255, 255, 255))
            
            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            # Draw center point
            center_x, center_y = human['center']
            cv2.circle(annotated, (center_x, center_y), 5, color, -1)
            
            # Draw person ID
            cv2.putText(annotated, f"P{person_id}", (x1, y1 - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Draw confidence
            confidence_label = f"{confidence:.2f}"
            cv2.putText(annotated, confidence_label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Draw characteristics
            char_label = f"{size_category} | {position} | {color_dominant}"
            cv2.putText(annotated, char_label, (x1, y2 + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Draw poses
        for pose in poses:
            person_id = pose['person_id']
            landmarks = pose['landmarks']
            pose_quality = pose['pose_quality']
            pose_type = pose['pose_type']
            body_orientation = pose['body_orientation']
            
            # Get color for pose quality
            color = pose_colors.get(pose_quality, (255, 255, 255))
            
            # Draw pose landmarks
            for i, landmark in enumerate(landmarks):
                if landmark is not None:
                    x, y = landmark['x'], landmark['y']
                    visibility = landmark['visibility']
                    
                    # Draw landmark point
                    cv2.circle(annotated, (x, y), 3, color, -1)
                    
                    # Draw landmark number for key points
                    if i in [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]:  # Key landmarks
                        cv2.putText(annotated, str(i), (x + 5, y - 5),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
            
            # Draw pose information
            pose_info = f"P{person_id}: {pose_type} | {body_orientation} | {pose_quality}"
            cv2.putText(annotated, pose_info, (10, 30 + len(poses) * 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Draw summary
        summary_text = f"Humans: {len(humans)} | Poses: {len(poses)}"
        cv2.putText(annotated, summary_text, (50, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return annotated
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded models."""
        if not self.yolo_available:
            return {"error": "YOLO not available"}
        
        return {
            "yolo_model": str(self.model.model_name) if hasattr(self.model, 'model_name') else "YOLOv8",
            "pose_model": "MediaPipe Pose",
            "person_class_id": self.person_class_id,
            "person_class_name": self.coco_classes[self.person_class_id]
        }


@click.command()
@click.argument('input_pattern', type=str)
@click.argument('output_dir', type=str)
@click.option('--max-images', '-m', default=5, type=int, help='Maximum number of images to process')
@click.option('--confidence', '-c', default=0.5, help='Confidence threshold for human detection')
@click.option('--model-path', help='Path to custom YOLO model')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
def main(input_pattern: str, output_dir: str, max_images: int, confidence: float, model_path: str, verbose: bool):
    """Detect humans with YOLOv8 and analyze poses with MediaPipe."""
    
    # Setup logging
    if verbose:
        logger.add("hybrid_pose_detection.log", level="DEBUG")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find images
    if input_pattern.startswith('/'):
        parent_dir = Path(input_pattern).parent
        pattern = Path(input_pattern).name
        image_files = list(parent_dir.glob(pattern))
    else:
        image_files = list(Path('.').glob(input_pattern))
    
    if not image_files:
        logger.error(f"No images found matching pattern: {input_pattern}")
        return
    
    # Limit images
    image_files = image_files[:max_images]
    
    logger.info(f"Processing {len(image_files)} images with hybrid pose detection")
    
    # Initialize detector
    detector = HybridPoseDetector(model_path)
    
    if not detector.yolo_available:
        logger.error("YOLO not available. Install with: pip install ultralytics")
        return
    
    # Print model info
    model_info = detector.get_model_info()
    logger.info(f"Using YOLO model: {model_info['yolo_model']}")
    logger.info(f"Using pose model: {model_info['pose_model']}")
    logger.info(f"Person class: {model_info['person_class_name']} (ID: {model_info['person_class_id']})")
    
    # Process images
    total_humans = 0
    total_poses = 0
    results = []
    
    for image_path in image_files:
        logger.info(f"Analyzing {image_path.name}")
        
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            logger.error(f"Failed to load {image_path.name}")
            continue
        
        # Detect humans and analyze poses
        detections = detector.detect_humans_and_poses(image, confidence_threshold=confidence)
        
        humans = detections.get('humans', [])
        poses = detections.get('poses', [])
        
        if humans or poses:
            total_humans += len(humans)
            total_poses += len(poses)
            logger.info(f"  Detected {len(humans)} human(s) and {len(poses)} pose(s)")
            
            for human in humans:
                logger.info(f"    Person {human['person_id']}: {human['size_category']} {human['position']} {human['color_dominant']} (confidence: {human['confidence']:.2f})")
            
            for pose in poses:
                logger.info(f"    Pose {pose['person_id']}: {pose['pose_type']} {pose['body_orientation']} {pose['pose_quality']} ({pose['total_landmarks']} landmarks)")
        else:
            logger.info(f"  No humans or poses detected")
        
        # Create annotated image
        annotated = detector.draw_detections(image, detections)
        
        # Save annotated image
        output_filename = f"{image_path.stem}_hybrid_pose_detection.jpg"
        output_file = output_path / output_filename
        cv2.imwrite(str(output_file), annotated)
        
        logger.info(f"  Saved: {output_filename}")
        
        # Store results
        results.append({
            'image': image_path.name,
            'humans_detected': len(humans),
            'poses_detected': len(poses),
            'humans': humans,
            'poses': poses
        })
    
    # Save results summary
    summary_path = output_path / "hybrid_pose_detection_summary.json"
    with open(summary_path, 'w') as f:
        json.dump({
            'total_images': len(image_files),
            'total_humans': total_humans,
            'total_poses': total_poses,
            'human_detection_rate': total_humans / len(image_files),
            'pose_detection_rate': total_poses / len(image_files),
            'confidence_threshold': confidence,
            'model_info': model_info,
            'results': results
        }, f, indent=2)
    
    # Summary
    logger.info(f"Hybrid pose detection complete!")
    logger.info(f"Images processed: {len(image_files)}")
    logger.info(f"Total humans detected: {total_humans}")
    logger.info(f"Total poses detected: {total_poses}")
    logger.info(f"Human detection rate: {total_humans/len(image_files)*100:.1f}%")
    logger.info(f"Pose detection rate: {total_poses/len(image_files)*100:.1f}%")
    logger.info(f"Results saved to {output_path}")


if __name__ == '__main__':
    main()
