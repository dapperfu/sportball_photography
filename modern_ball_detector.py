#!/usr/bin/env python3
"""
Modern Ball Detection using YOLOv8

This tool uses YOLOv8 (ultralytics) for accurate soccer ball detection
and classification, replacing the traditional Hough circle approach.

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


class ModernBallDetector:
    """Modern ball detector using YOLOv8 for accurate detection."""
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the modern ball detector.
        
        Args:
            model_path: Path to custom YOLO model (optional)
        """
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
        
        # COCO class names (YOLOv8 uses COCO dataset)
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
        
        # Sports ball class ID in COCO
        self.sports_ball_class_id = 32
        
        logger.info("Initialized ModernBallDetector with YOLOv8")
    
    def detect_balls(self, image: np.ndarray, confidence_threshold: float = 0.5) -> List[Dict[str, Any]]:
        """
        Detect balls using YOLOv8.
        
        Args:
            image: Input image (BGR format)
            confidence_threshold: Minimum confidence for detection
            
        Returns:
            List of detected balls
        """
        if not self.yolo_available:
            return []
        
        try:
            # Run YOLO inference
            results = self.model(image, conf=confidence_threshold)
            
            balls = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for i, box in enumerate(boxes):
                        # Get class ID and confidence
                        class_id = int(box.cls[0])
                        confidence = float(box.conf[0])
                        
                        # Check if it's a sports ball
                        if class_id == self.sports_ball_class_id:
                            # Get bounding box coordinates
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                            
                            # Calculate center and dimensions
                            center_x = (x1 + x2) // 2
                            center_y = (y1 + y2) // 2
                            width = x2 - x1
                            height = y2 - y1
                            
                            # Classify ball type based on context
                            ball_type = self._classify_ball_type(image, x1, y1, x2, y2)
                            
                            balls.append({
                                'bbox': (x1, y1, width, height),
                                'center': (center_x, center_y),
                                'type': ball_type,
                                'confidence': confidence,
                                'method': 'yolov8',
                                'class_name': 'sports ball'
                            })
            
            return balls
            
        except Exception as e:
            logger.error(f"YOLO detection failed: {e}")
            return []
    
    def _classify_ball_type(self, image: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> str:
        """
        Classify ball type based on color analysis of detected region.
        
        Args:
            image: Original image
            x1, y1, x2, y2: Bounding box coordinates
            
        Returns:
            Ball type classification
        """
        # Extract ball region
        ball_region = image[y1:y2, x1:x2]
        
        if ball_region.size == 0:
            return "unknown"
        
        # Convert to HSV for better color analysis
        hsv_region = cv2.cvtColor(ball_region, cv2.COLOR_BGR2HSV)
        
        # Calculate dominant colors
        hist_h = cv2.calcHist([hsv_region], [0], None, [180], [0, 180])
        hist_s = cv2.calcHist([hsv_region], [1], None, [256], [0, 256])
        hist_v = cv2.calcHist([hsv_region], [2], None, [256], [0, 256])
        
        dominant_hue = np.argmax(hist_h)
        dominant_saturation = np.argmax(hist_s)
        dominant_value = np.argmax(hist_v)
        
        # Classify based on color characteristics
        if dominant_value > 200 and dominant_saturation < 50:  # White/light
            return "soccer"
        elif 5 <= dominant_hue <= 15 and dominant_saturation > 120:  # Orange
            return "basketball"
        elif 20 <= dominant_hue <= 30 and dominant_saturation > 120:  # Yellow
            return "tennis"
        elif 10 <= dominant_hue <= 20 and dominant_saturation > 80:  # Brown/orange
            return "rugby"
        elif dominant_hue < 10 or dominant_hue > 170:  # Red
            return "soccer"
        
        return "sports ball"  # Generic fallback
    
    def draw_detections(self, image: np.ndarray, balls: List[Dict[str, Any]]) -> np.ndarray:
        """
        Draw ball detections on the image.
        
        Args:
            image: Original image
            balls: List of detected balls
            
        Returns:
            Annotated image
        """
        annotated = image.copy()
        
        if not balls:
            # Draw "No ball detected" message
            cv2.putText(annotated, "No ball detected", (50, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return annotated
        
        # Color mapping for different ball types
        colors = {
            "soccer": (0, 255, 0),      # Green
            "basketball": (0, 165, 255), # Orange
            "tennis": (0, 255, 255),    # Yellow
            "rugby": (0, 0, 255),       # Red
            "sports ball": (255, 0, 255), # Magenta
            "unknown": (255, 255, 255)  # White
        }
        
        for i, ball in enumerate(balls):
            x1, y1, w, h = ball['bbox']
            x2, y2 = x1 + w, y1 + h
            ball_type = ball['type']
            confidence = ball['confidence']
            
            # Get color for this ball type
            color = colors.get(ball_type, (255, 255, 255))
            
            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            # Draw center point
            center_x, center_y = ball['center']
            cv2.circle(annotated, (center_x, center_y), 5, color, -1)
            
            # Draw label
            label = f"{ball_type.title()} ({confidence:.2f})"
            cv2.putText(annotated, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Draw confidence bar
            bar_width = 100
            bar_height = 8
            bar_x = x1
            bar_y = y2 + 20
            
            # Background bar
            cv2.rectangle(annotated, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (100, 100, 100), -1)
            
            # Confidence bar
            confidence_width = int(bar_width * confidence)
            cv2.rectangle(annotated, (bar_x, bar_y), (bar_x + confidence_width, bar_y + bar_height), color, -1)
            
            # Ball ID
            cv2.putText(annotated, f"Ball {i+1}", (x1, y1 - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return annotated
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        if not self.yolo_available:
            return {"error": "YOLO not available"}
        
        return {
            "model_name": str(self.model.model_name) if hasattr(self.model, 'model_name') else "YOLOv8",
            "model_path": str(self.model.model) if hasattr(self.model, 'model') else "pre-trained",
            "classes": len(self.coco_classes),
            "sports_ball_class_id": self.sports_ball_class_id,
            "sports_ball_class_name": self.coco_classes[self.sports_ball_class_id]
        }


@click.command()
@click.argument('input_pattern', type=str)
@click.argument('output_dir', type=str)
@click.option('--max-images', '-m', default=5, type=int, help='Maximum number of images to process')
@click.option('--confidence', '-c', default=0.5, help='Confidence threshold for detection')
@click.option('--model-path', help='Path to custom YOLO model')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
def main(input_pattern: str, output_dir: str, max_images: int, confidence: float, model_path: str, verbose: bool):
    """Detect balls using modern YOLOv8 object detection."""
    
    # Setup logging
    if verbose:
        logger.add("modern_ball_detection.log", level="DEBUG")
    
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
    
    logger.info(f"Processing {len(image_files)} images with modern ball detection")
    
    # Initialize detector
    detector = ModernBallDetector(model_path)
    
    if not detector.yolo_available:
        logger.error("YOLO not available. Install with: pip install ultralytics")
        return
    
    # Print model info
    model_info = detector.get_model_info()
    logger.info(f"Using model: {model_info['model_name']}")
    logger.info(f"Sports ball class: {model_info['sports_ball_class_name']} (ID: {model_info['sports_ball_class_id']})")
    
    # Process images
    total_balls = 0
    results = []
    
    for image_path in image_files:
        logger.info(f"Analyzing {image_path.name}")
        
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            logger.error(f"Failed to load {image_path.name}")
            continue
        
        # Detect balls
        balls = detector.detect_balls(image, confidence_threshold=confidence)
        
        if balls:
            total_balls += len(balls)
            logger.info(f"  Detected {len(balls)} ball(s)")
            for i, ball in enumerate(balls):
                logger.info(f"    Ball {i+1}: {ball['type']} (confidence: {ball['confidence']:.2f})")
        else:
            logger.info(f"  No balls detected")
        
        # Create annotated image
        annotated = detector.draw_detections(image, balls)
        
        # Save annotated image
        output_filename = f"{image_path.stem}_modern_ball_detection.jpg"
        output_file = output_path / output_filename
        cv2.imwrite(str(output_file), annotated)
        
        logger.info(f"  Saved: {output_filename}")
        
        # Store results
        results.append({
            'image': image_path.name,
            'balls_detected': len(balls),
            'balls': balls
        })
    
    # Save results summary
    summary_path = output_path / "modern_ball_detection_summary.json"
    with open(summary_path, 'w') as f:
        json.dump({
            'total_images': len(image_files),
            'total_balls': total_balls,
            'detection_rate': total_balls / len(image_files),
            'confidence_threshold': confidence,
            'model_info': model_info,
            'results': results
        }, f, indent=2)
    
    # Summary
    logger.info(f"Modern ball detection complete!")
    logger.info(f"Images processed: {len(image_files)}")
    logger.info(f"Total balls detected: {total_balls}")
    logger.info(f"Detection rate: {total_balls/len(image_files)*100:.1f}%")
    logger.info(f"Results saved to {output_path}")


if __name__ == '__main__':
    main()
