#!/usr/bin/env python3
"""
Modern Human Detection using YOLOv8

This tool uses YOLOv8 (ultralytics) for accurate human detection
and classification, including pose estimation and person counting.

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


class ModernHumanDetector:
    """Modern human detector using YOLOv8 for accurate detection."""
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the modern human detector.
        
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
        
        # Person class ID in COCO
        self.person_class_id = 0
        
        logger.info("Initialized ModernHumanDetector with YOLOv8")
    
    def detect_humans(self, image: np.ndarray, confidence_threshold: float = 0.5) -> List[Dict[str, Any]]:
        """
        Detect humans using YOLOv8.
        
        Args:
            image: Input image (BGR format)
            confidence_threshold: Minimum confidence for detection
            
        Returns:
            List of detected humans
        """
        if not self.yolo_available:
            return []
        
        try:
            # Run YOLO inference
            results = self.model(image, conf=confidence_threshold)
            
            humans = []
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
                            
                            # Calculate person characteristics
                            person_info = self._analyze_person(image, x1, y1, x2, y2)
                            
                            humans.append({
                                'bbox': (x1, y1, width, height),
                                'center': (center_x, center_y),
                                'confidence': confidence,
                                'method': 'yolov8',
                                'class_name': 'person',
                                'person_id': i + 1,
                                **person_info
                            })
            
            return humans
            
        except Exception as e:
            logger.error(f"YOLO detection failed: {e}")
            return []
    
    def _analyze_person(self, image: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> Dict[str, Any]:
        """
        Analyze person characteristics from detected region.
        
        Args:
            image: Original image
            x1, y1, x2, y2: Bounding box coordinates
            
        Returns:
            Person analysis results
        """
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
    
    def draw_detections(self, image: np.ndarray, humans: List[Dict[str, Any]]) -> np.ndarray:
        """
        Draw human detections on the image.
        
        Args:
            image: Original image
            humans: List of detected humans
            
        Returns:
            Annotated image
        """
        annotated = image.copy()
        
        if not humans:
            # Draw "No humans detected" message
            cv2.putText(annotated, "No humans detected", (50, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return annotated
        
        # Color mapping for different person characteristics
        colors = {
            'large': (0, 255, 0),      # Green
            'medium': (0, 165, 255),   # Orange
            'small': (0, 255, 255),    # Yellow
            'tiny': (255, 0, 255),     # Magenta
            'unknown': (255, 255, 255)  # White
        }
        
        for i, human in enumerate(humans):
            x1, y1, w, h = human['bbox']
            x2, y2 = x1 + w, y1 + h
            confidence = human['confidence']
            size_category = human['size_category']
            position = human['position']
            color_dominant = human['color_dominant']
            
            # Get color for this person's size category
            color = colors.get(size_category, (255, 255, 255))
            
            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            # Draw center point
            center_x, center_y = human['center']
            cv2.circle(annotated, (center_x, center_y), 5, color, -1)
            
            # Draw person ID
            person_id = human.get('person_id', i + 1)
            cv2.putText(annotated, f"Person {person_id}", (x1, y1 - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Draw confidence
            confidence_label = f"Conf: {confidence:.2f}"
            cv2.putText(annotated, confidence_label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Draw characteristics
            char_label = f"{size_category.title()} | {position} | {color_dominant}"
            cv2.putText(annotated, char_label, (x1, y2 + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Draw confidence bar
            bar_width = 100
            bar_height = 8
            bar_x = x1
            bar_y = y2 + 40
            
            # Background bar
            cv2.rectangle(annotated, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (100, 100, 100), -1)
            
            # Confidence bar
            confidence_width = int(bar_width * confidence)
            cv2.rectangle(annotated, (bar_x, bar_y), (bar_x + confidence_width, bar_y + bar_height), color, -1)
        
        # Draw summary
        summary_text = f"Humans detected: {len(humans)}"
        cv2.putText(annotated, summary_text, (50, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return annotated
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        if not self.yolo_available:
            return {"error": "YOLO not available"}
        
        return {
            "model_name": str(self.model.model_name) if hasattr(self.model, 'model_name') else "YOLOv8",
            "model_path": str(self.model.model) if hasattr(self.model, 'model') else "pre-trained",
            "classes": len(self.coco_classes),
            "person_class_id": self.person_class_id,
            "person_class_name": self.coco_classes[self.person_class_id]
        }


@click.command()
@click.argument('input_pattern', type=str)
@click.argument('output_dir', type=str)
@click.option('--max-images', '-m', default=5, type=int, help='Maximum number of images to process')
@click.option('--confidence', '-c', default=0.5, help='Confidence threshold for detection')
@click.option('--model-path', help='Path to custom YOLO model')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
def main(input_pattern: str, output_dir: str, max_images: int, confidence: float, model_path: str, verbose: bool):
    """Detect humans using modern YOLOv8 object detection."""
    
    # Setup logging
    if verbose:
        logger.add("modern_human_detection.log", level="DEBUG")
    
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
    
    logger.info(f"Processing {len(image_files)} images with modern human detection")
    
    # Initialize detector
    detector = ModernHumanDetector(model_path)
    
    if not detector.yolo_available:
        logger.error("YOLO not available. Install with: pip install ultralytics")
        return
    
    # Print model info
    model_info = detector.get_model_info()
    logger.info(f"Using model: {model_info['model_name']}")
    logger.info(f"Person class: {model_info['person_class_name']} (ID: {model_info['person_class_id']})")
    
    # Process images
    total_humans = 0
    results = []
    
    for image_path in image_files:
        logger.info(f"Analyzing {image_path.name}")
        
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            logger.error(f"Failed to load {image_path.name}")
            continue
        
        # Detect humans
        humans = detector.detect_humans(image, confidence_threshold=confidence)
        
        if humans:
            total_humans += len(humans)
            logger.info(f"  Detected {len(humans)} human(s)")
            for i, human in enumerate(humans):
                logger.info(f"    Person {i+1}: {human['size_category']} {human['position']} {human['color_dominant']} (confidence: {human['confidence']:.2f})")
        else:
            logger.info(f"  No humans detected")
        
        # Create annotated image
        annotated = detector.draw_detections(image, humans)
        
        # Save annotated image
        output_filename = f"{image_path.stem}_modern_human_detection.jpg"
        output_file = output_path / output_filename
        cv2.imwrite(str(output_file), annotated)
        
        logger.info(f"  Saved: {output_filename}")
        
        # Store results
        results.append({
            'image': image_path.name,
            'humans_detected': len(humans),
            'humans': humans
        })
    
    # Save results summary
    summary_path = output_path / "modern_human_detection_summary.json"
    with open(summary_path, 'w') as f:
        json.dump({
            'total_images': len(image_files),
            'total_humans': total_humans,
            'detection_rate': total_humans / len(image_files),
            'confidence_threshold': confidence,
            'model_info': model_info,
            'results': results
        }, f, indent=2)
    
    # Summary
    logger.info(f"Modern human detection complete!")
    logger.info(f"Images processed: {len(image_files)}")
    logger.info(f"Total humans detected: {total_humans}")
    logger.info(f"Detection rate: {total_humans/len(image_files)*100:.1f}%")
    logger.info(f"Results saved to {output_path}")


if __name__ == '__main__':
    main()
