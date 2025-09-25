#!/usr/bin/env python3
"""
YOLOv8 Object Detection Tool with GPU Support and Parallel Processing

This tool detects objects (humans, balls, etc.) in images using YOLOv8 and saves 
comprehensive detection data to JSON sidecar files. Includes all metadata needed 
to extract objects from full-resolution images using percentage-based coordinates.

Author: Claude Sonnet 4 (claude-3-5-sonnet-20241022)
Generated via Cursor IDE (cursor.sh) with AI assistance
"""

import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Set
import click
import cv2
import numpy as np
from tqdm import tqdm
from loguru import logger

# Try to import ultralytics for YOLOv8
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    logger.warning("ultralytics not available - YOLOv8 detection will be skipped")

# COCO class names for YOLOv8
COCO_CLASSES = {
    0: 'person',
    1: 'bicycle',
    2: 'car',
    3: 'motorcycle',
    4: 'airplane',
    5: 'bus',
    6: 'train',
    7: 'truck',
    8: 'boat',
    9: 'traffic light',
    10: 'fire hydrant',
    11: 'stop sign',
    12: 'parking meter',
    13: 'bench',
    14: 'bird',
    15: 'cat',
    16: 'dog',
    17: 'horse',
    18: 'sheep',
    19: 'cow',
    20: 'elephant',
    21: 'bear',
    22: 'zebra',
    23: 'giraffe',
    24: 'backpack',
    25: 'umbrella',
    26: 'handbag',
    27: 'tie',
    28: 'suitcase',
    29: 'frisbee',
    30: 'skis',
    31: 'snowboard',
    32: 'sports ball',
    33: 'kite',
    34: 'baseball bat',
    35: 'baseball glove',
    36: 'skateboard',
    37: 'surfboard',
    38: 'tennis racket',
    39: 'bottle',
    40: 'wine glass',
    41: 'cup',
    42: 'fork',
    43: 'knife',
    44: 'spoon',
    45: 'bowl',
    46: 'banana',
    47: 'apple',
    48: 'sandwich',
    49: 'orange',
    50: 'broccoli',
    51: 'carrot',
    52: 'hot dog',
    53: 'pizza',
    54: 'donut',
    55: 'cake',
    56: 'chair',
    57: 'couch',
    58: 'potted plant',
    59: 'bed',
    60: 'dining table',
    61: 'toilet',
    62: 'tv',
    63: 'laptop',
    64: 'mouse',
    65: 'remote',
    66: 'keyboard',
    67: 'cell phone',
    68: 'microwave',
    69: 'oven',
    70: 'toaster',
    71: 'sink',
    72: 'refrigerator',
    73: 'book',
    74: 'clock',
    75: 'vase',
    76: 'scissors',
    77: 'teddy bear',
    78: 'hair drier',
    79: 'toothbrush'
}


@dataclass
class DetectedObject:
    """Information about a detected object."""
    object_id: int
    class_name: str
    class_id: int
    # Coordinates in original image (pixels)
    x: int
    y: int
    width: int
    height: int
    # Coordinates as percentages of image dimensions
    x_percent: float
    y_percent: float
    width_percent: float
    height_percent: float
    confidence: float
    # Crop area with padding (for extraction)
    crop_x_percent: float = 0.0
    crop_y_percent: float = 0.0
    crop_width_percent: float = 0.0
    crop_height_percent: float = 0.0
    # Scale factor used for detection
    detection_scale_factor: float = 1.0


@dataclass
class DetectionResult:
    """Result of object detection from an image."""
    image_path: str
    image_width: int
    image_height: int
    objects_found: int
    detection_time: float
    detected_objects: List[DetectedObject]
    error: Optional[str] = None


class YOLOv8ObjectDetector:
    """YOLOv8 object detection with GPU support and comprehensive data collection."""
    
    def __init__(self, 
                 model_path: str = "yolov8n.pt",
                 border_padding: float = 0.25, 
                 use_gpu: bool = True,
                 target_objects: Optional[Set[str]] = None,
                 confidence_threshold: float = 0.5):
        """
        Initialize YOLOv8 object detector.
        
        Args:
            model_path: Path to YOLOv8 model file
            border_padding: Percentage of padding around detected objects (0.25 = 25%)
            use_gpu: Whether to use GPU acceleration if available
            target_objects: Set of object class names to detect (None = all)
            confidence_threshold: Minimum confidence for detections
        """
        self.border_padding = border_padding
        self.use_gpu = use_gpu
        self.confidence_threshold = confidence_threshold
        self.target_objects = target_objects or set()
        
        if not YOLO_AVAILABLE:
            raise ImportError("ultralytics package is required for YOLOv8 detection")
        
        # Load YOLOv8 model
        try:
            self.model = YOLO(model_path)
            logger.info(f"Loaded YOLOv8 model: {model_path}")
        except Exception as e:
            logger.error(f"Failed to load YOLOv8 model: {e}")
            raise
        
        # Check for GPU support
        if self.use_gpu:
            try:
                # Check if CUDA is available
                import torch
                if torch.cuda.is_available():
                    device_count = torch.cuda.device_count()
                    logger.info(f"GPU support available: {device_count} CUDA devices")
                    self.device = "cuda"
                else:
                    logger.warning("CUDA not available, falling back to CPU")
                    self.use_gpu = False
                    self.device = "cpu"
            except ImportError:
                logger.warning("PyTorch not available, falling back to CPU")
                self.use_gpu = False
                self.device = "cpu"
        else:
            self.device = "cpu"
        
        # Create class name to ID mapping
        self.class_name_to_id = {name: class_id for class_id, name in COCO_CLASSES.items()}
        
        # Filter target objects if specified
        if self.target_objects:
            # Convert to lowercase for case-insensitive matching
            target_objects_lower = {obj.lower() for obj in self.target_objects}
            self.target_class_ids = set()
            
            for class_name, class_id in self.class_name_to_id.items():
                if class_name.lower() in target_objects_lower:
                    self.target_class_ids.add(class_id)
            
            logger.info(f"Target objects: {sorted(self.target_objects)}")
            logger.info(f"Target class IDs: {sorted(self.target_class_ids)}")
        else:
            self.target_class_ids = None  # Detect all objects
        
        logger.info(f"Initialized YOLOv8 detector with {border_padding*100:.0f}% border padding, GPU: {self.use_gpu}")
    
    def detect_objects_in_image(self, image_path: Path, force: bool = False) -> DetectionResult:
        """
        Detect objects in a single image with comprehensive data collection.
        
        Args:
            image_path: Path to the input image
            force: Whether to force detection even if JSON sidecar exists
            
        Returns:
            Detection result with object information
        """
        # Check if JSON sidecar already exists and contains YOLOv8 data
        json_path = image_path.parent / f"{image_path.stem}.json"
        if json_path.exists() and not force:
            try:
                with open(json_path, 'r') as f:
                    json_data = json.load(f)
                if 'yolov8' in json_data:
                    logger.info(f"Skipping {image_path.name} - YOLOv8 data already exists (use --force to override)")
                    return DetectionResult(
                        image_path=str(image_path),
                        image_width=0,
                        image_height=0,
                        objects_found=0,
                        detection_time=0.0,
                        detected_objects=[],
                        error="Skipped - YOLOv8 data already exists"
                    )
            except Exception as e:
                logger.debug(f"Could not read JSON sidecar {json_path}: {e}")
                # Continue with detection if JSON is invalid
        
        logger.info(f"Detecting objects in {image_path.name}")
        
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            return DetectionResult(
                image_path=str(image_path),
                image_width=0,
                image_height=0,
                objects_found=0,
                detection_time=0.0,
                detected_objects=[],
                error="Failed to load image"
            )
        
        original_height, original_width = image.shape[:2]
        
        # Resize image to 1080p for optimal detection performance
        target_width = 1920  # 1080p width
        target_height = 1080  # 1080p height
        
        # Calculate scaling factor to fit within 1080p while maintaining aspect ratio
        scale_factor = min(target_width / original_width, target_height / original_height)
        
        # Keep original image for final coordinates, use resized for detection
        original_image = image.copy()
        
        if scale_factor < 1.0:
            # Only resize if image is larger than 1080p
            new_width = int(original_width * scale_factor)
            new_height = int(original_height * scale_factor)
            image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
            logger.debug(f"Resized image from {original_width}x{original_height} to {new_width}x{new_height}")
        else:
            scale_factor = 1.0  # No scaling needed
        
        # Run YOLOv8 detection
        start_time = cv2.getTickCount()
        
        try:
            # Run inference
            results = self.model(image, device=self.device, conf=self.confidence_threshold)
            detection_time = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()
            
            detected_objects = []
            
            # Process detection results
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for i, box in enumerate(boxes):
                        # Get box coordinates (x1, y1, x2, y2)
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = box.conf[0].cpu().numpy()
                        class_id = int(box.cls[0].cpu().numpy())
                        
                        # Filter by target objects if specified
                        if self.target_class_ids is not None and class_id not in self.target_class_ids:
                            continue
                        
                        # Scale coordinates back to original image size
                        orig_x1 = int(x1 / scale_factor)
                        orig_y1 = int(y1 / scale_factor)
                        orig_x2 = int(x2 / scale_factor)
                        orig_y2 = int(y2 / scale_factor)
                        
                        # Calculate width and height
                        orig_w = orig_x2 - orig_x1
                        orig_h = orig_y2 - orig_y1
                        
                        # Calculate percentage coordinates
                        x_percent = orig_x1 / original_width
                        y_percent = orig_y1 / original_height
                        width_percent = orig_w / original_width
                        height_percent = orig_h / original_height
                        
                        # Calculate padded coordinates for cropping
                        padding_x_percent = width_percent * self.border_padding
                        padding_y_percent = height_percent * self.border_padding
                        
                        # Calculate crop area with padding (as percentages)
                        crop_x_percent = max(0.0, x_percent - padding_x_percent)
                        crop_y_percent = max(0.0, y_percent - padding_y_percent)
                        crop_width_percent = min(1.0 - crop_x_percent, width_percent + 2 * padding_x_percent)
                        crop_height_percent = min(1.0 - crop_y_percent, height_percent + 2 * padding_y_percent)
                        
                        # Get class name
                        class_name = COCO_CLASSES.get(class_id, f"class_{class_id}")
                        
                        # Create detected object
                        detected_object = DetectedObject(
                            object_id=i+1,
                            class_name=class_name,
                            class_id=class_id,
                            x=orig_x1,
                            y=orig_y1,
                            width=orig_w,
                            height=orig_h,
                            x_percent=x_percent,
                            y_percent=y_percent,
                            width_percent=width_percent,
                            height_percent=height_percent,
                            confidence=float(confidence),
                            crop_x_percent=crop_x_percent,
                            crop_y_percent=crop_y_percent,
                            crop_width_percent=crop_width_percent,
                            crop_height_percent=crop_height_percent,
                            detection_scale_factor=scale_factor
                        )
                        
                        detected_objects.append(detected_object)
                        logger.debug(f"Detected {class_name}: {orig_w}x{orig_h} at ({orig_x1}, {orig_y1}), confidence: {confidence:.3f}")
            
        except Exception as e:
            logger.error(f"Error during YOLOv8 detection: {e}")
            return DetectionResult(
                image_path=str(image_path),
                image_width=original_width,
                image_height=original_height,
                objects_found=0,
                detection_time=0.0,
                detected_objects=[],
                error=str(e)
            )
        
        return DetectionResult(
            image_path=str(image_path),
            image_width=original_width,
            image_height=original_height,
            objects_found=len(detected_objects),
            detection_time=detection_time,
            detected_objects=detected_objects
        )
    
    def create_detection_json(self, image_path: Path, detection_result: DetectionResult) -> None:
        """
        Create JSON sidecar file for an image with all detection data.
        
        Args:
            image_path: Path to the original image
            detection_result: Detection result data
        """
        from datetime import datetime
        
        # Check if JSON sidecar already exists
        json_path = image_path.parent / f"{image_path.stem}.json"
        existing_data = {}
        
        if json_path.exists():
            try:
                with open(json_path, 'r') as f:
                    existing_data = json.load(f)
            except Exception as e:
                logger.warning(f"Could not read existing JSON {json_path}: {e}")
                existing_data = {}
        
        # Create YOLOv8 data structure
        yolov8_data = {
            "metadata": {
                "extraction_timestamp": datetime.now().isoformat(),
                "tool_version": "1.0.0",
                "model_path": str(self.model.ckpt_path) if hasattr(self.model, 'ckpt_path') else "yolov8n.pt",
                "border_padding_percentage": self.border_padding * 100,
                "confidence_threshold": self.confidence_threshold,
                "total_objects_found": detection_result.objects_found,
                "image_path": str(image_path),
                "image_dimensions": {
                    "width": detection_result.image_width,
                    "height": detection_result.image_height
                },
                "detection_time_seconds": detection_result.detection_time,
                "gpu_enabled": self.use_gpu,
                "target_objects": list(self.target_objects) if self.target_objects else "all"
            },
            "objects": []
        }
        
        # Merge with existing data
        image_data = existing_data.copy()
        image_data["yolov8"] = yolov8_data
        
        # Add each detected object to the JSON
        for obj in detection_result.detected_objects:
            object_data = {
                "object_id": obj.object_id,
                "class_name": obj.class_name,
                "class_id": obj.class_id,
                "coordinates_pixels": {
                    "x": obj.x,
                    "y": obj.y,
                    "width": obj.width,
                    "height": obj.height
                },
                "coordinates_percent": {
                    "x": obj.x_percent,
                    "y": obj.y_percent,
                    "width": obj.width_percent,
                    "height": obj.height_percent
                },
                "confidence": obj.confidence,
                "border_padding": self.border_padding,
                "crop_area_percent": {
                    "x": obj.crop_x_percent,
                    "y": obj.crop_y_percent,
                    "width": obj.crop_width_percent,
                    "height": obj.crop_height_percent
                },
                "detection_scale_factor": obj.detection_scale_factor
            }
            
            image_data["yolov8"]["objects"].append(object_data)
        
        # Create JSON filename based on the image filename
        json_filename = f"{image_path.stem}.json"
        json_path = image_path.parent / json_filename
        
        # Save JSON file
        with open(json_path, 'w') as f:
            json.dump(image_data, f, indent=2, cls=NumpyEncoder)
        
        logger.debug(f"Detection JSON sidecar saved: {json_filename}")
    
    def detect_objects_in_images(self, image_pattern: str, max_images: Optional[int] = None, force: bool = False) -> List[DetectionResult]:
        """
        Detect objects in multiple images with parallel processing.
        
        Args:
            image_pattern: Pattern to match image files or directory path
            max_images: Maximum number of images to process
            force: Whether to force detection even if JSON sidecar exists
            
        Returns:
            List of detection results
        """
        # Check if input is a directory
        input_path = Path(image_pattern)
        if input_path.is_dir():
            # If it's a directory, find all image files in it
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
            image_files = []
            for file_path in input_path.iterdir():
                if file_path.is_file() and file_path.suffix.lower() in image_extensions:
                    image_files.append(file_path)
        else:
            # Find images using pattern
            if image_pattern.startswith('/'):
                # Absolute path
                parent_dir = Path(image_pattern).parent
                pattern = Path(image_pattern).name
                image_files = list(parent_dir.glob(pattern))
            else:
                # Relative path
                image_files = list(Path('.').glob(image_pattern))
        
        if not image_files:
            logger.error(f"No images found matching pattern: {image_pattern}")
            return []
        
        # Limit number of images if specified
        if max_images:
            image_files = image_files[:max_images]
        
        logger.info(f"Found {len(image_files)} images to process")
        
        # Process images in parallel
        results = []
        max_workers = min(4, len(image_files))  # Limit to 4 workers
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_image = {
                executor.submit(self.detect_objects_in_image, image_path, force): image_path 
                for image_path in image_files
            }
            
            # Process completed tasks with progress bar
            for future in tqdm(as_completed(future_to_image), 
                             total=len(image_files), 
                             desc="Detecting objects"):
                try:
                    result = future.result()
                    results.append(result)
                    
                    # Create JSON sidecar if objects were found
                    if result.detected_objects:
                        image_path = future_to_image[future]
                        self.create_detection_json(image_path, result)
                        
                except Exception as e:
                    image_path = future_to_image[future]
                    logger.error(f"Error processing {image_path}: {e}")
                    # Create error result
                    error_result = DetectionResult(
                        image_path=str(image_path),
                        image_width=0,
                        image_height=0,
                        objects_found=0,
                        detection_time=0.0,
                        detected_objects=[],
                        error=str(e)
                    )
                    results.append(error_result)
        
        return results


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


def list_available_objects():
    """List all available object classes for detection."""
    print("Available object classes for detection:")
    print("=" * 50)
    
    # Group by category for better readability
    categories = {
        'People & Animals': ['person', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe'],
        'Vehicles': ['bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat'],
        'Sports': ['sports ball', 'frisbee', 'skis', 'snowboard', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket'],
        'Furniture': ['chair', 'couch', 'bed', 'dining table', 'toilet'],
        'Electronics': ['tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster'],
        'Food': ['banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake'],
        'Other': ['traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'potted plant', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
    }
    
    for category, objects in categories.items():
        print(f"\n{category}:")
        for obj in objects:
            # Find the class ID by searching through COCO_CLASSES
            class_id = None
            for cid, name in COCO_CLASSES.items():
                if name == obj:
                    class_id = cid
                    break
            if class_id is not None:
                print(f"  {obj} (ID: {class_id})")
    
    print(f"\nTotal: {len(COCO_CLASSES)} object classes available")
    print("\nUsage examples:")
    print("  --objects person,sports ball  # Detect people and sports balls")
    print("  --objects car,truck,bus        # Detect vehicles")
    print("  --objects person              # Detect only people")


@click.command()
@click.argument('input_pattern', type=str, required=False)
@click.option('--objects', '-o', 
              help='Comma-separated list of objects to detect (e.g., "person,sports ball"). Use --list to see all options.')
@click.option('--list', 'list_objects', is_flag=True, help='List all available object classes')
@click.option('--border-padding', '-b', default=0.25, help='Border padding percentage (0.25 = 25%)')
@click.option('--confidence', '-c', default=0.5, help='Confidence threshold for detections (0.0-1.0)')
@click.option('--model', '-m', default='yolov8n.pt', help='Path to YOLOv8 model file')
@click.option('--max-images', '-n', default=None, type=int, help='Maximum number of images to process')
@click.option('--gpu/--no-gpu', default=True, help='Use GPU acceleration if available')
@click.option('--force', '-f', is_flag=True, help='Force detection even if JSON sidecar exists')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
def main(input_pattern: Optional[str], objects: Optional[str], list_objects: bool, border_padding: float, 
         confidence: float, model: str, max_images: Optional[int], gpu: bool, force: bool, verbose: bool):
    """Detect objects in images using YOLOv8 and save comprehensive data to JSON sidecar files."""
    
    # Handle --list option
    if list_objects:
        list_available_objects()
        return
    
    # Check if input_pattern is provided when not listing
    if not input_pattern:
        click.echo("Error: INPUT_PATTERN is required when not using --list option")
        click.echo("Use --help for more information")
        return 1
    
    # Setup logging
    if verbose:
        logger.add("yolo_object_detection.log", level="DEBUG")
    
    # Parse target objects
    target_objects = None
    if objects:
        target_objects = {obj.strip().lower() for obj in objects.split(',')}
        logger.info(f"Target objects: {sorted(target_objects)}")
    
    # Initialize detector
    try:
        detector = YOLOv8ObjectDetector(
            model_path=model,
            border_padding=border_padding, 
            use_gpu=gpu,
            target_objects=target_objects,
            confidence_threshold=confidence
        )
    except Exception as e:
        logger.error(f"Failed to initialize detector: {e}")
        return 1
    
    # Detect objects
    logger.info(f"Starting object detection with {border_padding*100:.0f}% border padding, confidence: {confidence}")
    results = detector.detect_objects_in_images(input_pattern, max_images, force)
    
    if not results:
        logger.error("No images processed")
        return 1
    
    # Calculate summary statistics
    total_images = len(results)
    total_objects_found = sum(result.objects_found for result in results)
    total_time = sum(result.detection_time for result in results)
    
    # Count objects by class
    class_counts = {}
    for result in results:
        for obj in result.detected_objects:
            class_counts[obj.class_name] = class_counts.get(obj.class_name, 0) + 1
    
    logger.info(f"Object detection complete!")
    logger.info(f"Processed {total_images} images")
    logger.info(f"Found {total_objects_found} objects")
    logger.info(f"Total detection time: {total_time:.2f}s")
    logger.info(f"Average time per image: {total_time/total_images:.2f}s")
    logger.info(f"GPU enabled: {detector.use_gpu}")
    
    if class_counts:
        logger.info("Objects detected by class:")
        for class_name, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"  {class_name}: {count}")
    
    return 0


if __name__ == "__main__":
    main()
