"""
Object Detection Module

YOLOv8-based object detection with GPU support, parallel processing,
and comprehensive data collection for the sportball package.

Author: Claude Sonnet 4 (claude-3-5-sonnet-20241022)
Generated via Cursor IDE (cursor.sh) with AI assistance
"""

import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Set, Dict, Any, Union, Tuple
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


class ObjectDetector:
    """
    YOLOv8 object detection with GPU support and comprehensive data collection.
    
    This class provides object detection capabilities using YOLOv8, with support
    for GPU acceleration, parallel processing, and comprehensive metadata collection
    for integration with the sportball package.
    """
    
    def __init__(self, 
                 model_path: str = "yolov8n.pt",
                 border_padding: float = 0.25, 
                 enable_gpu: bool = True,
                 target_objects: Optional[Set[str]] = None,
                 confidence_threshold: float = 0.5,
                 cache_enabled: bool = True,
                 gpu_batch_size: int = 8):
        """
        Initialize YOLOv8 object detector.
        
        Args:
            model_path: Path to YOLOv8 model file
            border_padding: Percentage of padding around detected objects (0.25 = 25%)
            enable_gpu: Whether to use GPU acceleration if available
            target_objects: Set of object class names to detect (None = all)
            confidence_threshold: Minimum confidence for detections
            cache_enabled: Whether to enable result caching
            gpu_batch_size: Number of images to process in GPU batches
        """
        self.border_padding = border_padding
        self.enable_gpu = enable_gpu
        self.confidence_threshold = confidence_threshold
        self.target_objects = target_objects or set()
        self.cache_enabled = cache_enabled
        self.gpu_batch_size = gpu_batch_size
        
        if not YOLO_AVAILABLE:
            raise ImportError("ultralytics package is required for YOLOv8 detection")
        
        # Check for GPU support first
        if self.enable_gpu:
            try:
                # Check if CUDA is available
                import torch
                if torch.cuda.is_available():
                    device_count = torch.cuda.device_count()
                    logger.info(f"GPU support available: {device_count} CUDA devices")
                    self.device = "cuda"
                else:
                    logger.warning("CUDA not available, falling back to CPU")
                    self.enable_gpu = False
                    self.device = "cpu"
            except ImportError:
                logger.warning("PyTorch not available, falling back to CPU")
                self.enable_gpu = False
                self.device = "cpu"
        else:
            self.device = "cpu"
        
        # Load YOLOv8 model with proper device configuration
        try:
            # Suppress Ultralytics logging
            import logging as std_logging
            std_logging.getLogger('ultralytics').setLevel(std_logging.WARNING)
            
            # Load model and move to appropriate device
            self.model = YOLO(model_path)
            
            # Configure model for GPU if available
            if self.enable_gpu and self.device == "cuda":
                try:
                    # Move model to GPU
                    self.model.to(self.device)
                    logger.info(f"YOLOv8 model moved to GPU: {self.device}")
                except Exception as e:
                    logger.warning(f"Failed to move YOLOv8 model to GPU: {e}")
                    self.device = "cpu"
                    self.model.to(self.device)
            else:
                self.model.to(self.device)
                logger.info(f"YOLOv8 model using device: {self.device}")
            
            logger.info(f"Loaded YOLOv8 model: {model_path}")
        except Exception as e:
            logger.error(f"Failed to load YOLOv8 model: {e}")
            raise
        
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
        
        logger.info(f"Initialized ObjectDetector with {border_padding*100:.0f}% border padding, GPU: {self.enable_gpu}")
    
    def detect_objects(self, 
                      image_paths: Union[Path, List[Path]], 
                      save_sidecar: bool = True,
                      force: bool = False,
                      max_workers: Optional[int] = None,
                      **kwargs) -> Dict[str, Any]:
        """
        Detect objects in images.
        
        Args:
            image_paths: Single image path or list of image paths
            save_sidecar: Whether to save results to sidecar files
            force: Whether to force detection even if sidecar exists
            max_workers: Maximum number of parallel workers
            **kwargs: Additional arguments (confidence, classes, etc.)
            
        Returns:
            Dictionary containing detection results
        """
        if isinstance(image_paths, Path):
            image_paths = [image_paths]
        
        # Override settings from kwargs
        confidence = kwargs.get('confidence', self.confidence_threshold)
        classes = kwargs.get('classes', None)
        
        # Update target objects if classes specified
        if classes:
            self.target_objects = set(classes)
            # Recalculate target class IDs
            target_objects_lower = {obj.lower() for obj in self.target_objects}
            self.target_class_ids = set()
            for class_name, class_id in self.class_name_to_id.items():
                if class_name.lower() in target_objects_lower:
                    self.target_class_ids.add(class_id)
        
        logger.info(f"Detecting objects in {len(image_paths)} images")
        
        # Process images
        if len(image_paths) == 1:
            # Single image processing
            result = self._detect_single_image(image_paths[0], force, save_sidecar)
            return {str(image_paths[0]): self._format_result(result)}
        else:
            # Multiple images processing
            results = self._detect_multiple_images(image_paths, force, max_workers, save_sidecar)
            return {str(path): self._format_result(result) for path, result in zip(image_paths, results)}
    
    def _detect_single_image(self, image_path: Path, force: bool = False, save_sidecar: bool = True) -> DetectionResult:
        """Detect objects in a single image."""
        result = self.detect_objects_in_image(image_path, force)
        
        # Save sidecar if requested
        if save_sidecar:
            self._save_sidecar(result)
        
        return result
    
    def _detect_multiple_images(self, image_paths: List[Path], force: bool, max_workers: Optional[int], save_sidecar: bool = True) -> List[DetectionResult]:
        """Detect objects in multiple images with optimized processing."""
        if self.enable_gpu and self.device == "cuda":
            # Use GPU batch processing for better GPU utilization
            return self._detect_multiple_images_gpu_batch(image_paths, force, save_sidecar)
        else:
            # Use CPU parallel processing
            return self._detect_multiple_images_cpu_parallel(image_paths, force, max_workers, save_sidecar)
    
    def _detect_multiple_images_gpu_batch(self, image_paths: List[Path], force: bool, save_sidecar: bool = True) -> List[DetectionResult]:
        """Detect objects in multiple images using sequential processing with progress bar."""
        logger.info(f"Processing {len(image_paths)} images sequentially")
        
        results = []
        
        # Use tqdm for progress tracking
        try:
            progress_bar = tqdm(total=len(image_paths), desc="Detecting objects", unit="images")
        except ImportError:
            progress_bar = None
        
        for i, image_path in enumerate(image_paths):
            try:
                result = self.detect_objects_in_image(image_path, force)
                results.append(result)
                
                # Save sidecar if requested
                if save_sidecar:
                    self._save_sidecar(result)
                
                # Update progress bar
                if progress_bar:
                    progress_bar.update(1)
                    
            except Exception as e:
                logger.error(f"Error processing {image_path}: {e}")
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
                # Update progress bar even on error
                if progress_bar:
                    progress_bar.update(1)
        
        # Close progress bar
        if progress_bar:
            progress_bar.close()
        
        return results
    
    def _detect_multiple_images_cpu_parallel(self, image_paths: List[Path], force: bool, max_workers: Optional[int], save_sidecar: bool = True) -> List[DetectionResult]:
        """Detect objects in multiple images using sequential processing with progress bar."""
        logger.info(f"Processing {len(image_paths)} images sequentially")
        
        results = []
        
        # Use tqdm for progress tracking
        try:
            progress_bar = tqdm(total=len(image_paths), desc="Detecting objects", unit="images")
        except ImportError:
            progress_bar = None
        
        for i, image_path in enumerate(image_paths):
            try:
                result = self.detect_objects_in_image(image_path, force)
                results.append(result)
                
                # Save sidecar if requested
                if save_sidecar:
                    self._save_sidecar(result)
                
                # Update progress bar
                if progress_bar:
                    progress_bar.update(1)
                    
            except Exception as e:
                logger.error(f"Error processing {image_path}: {e}")
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
                # Update progress bar even on error
                if progress_bar:
                    progress_bar.update(1)
        
        # Close progress bar
        if progress_bar:
            progress_bar.close()
        
        return results
    
    def _calculate_optimal_batch_size(self) -> int:
        """Calculate optimal batch size based on GPU memory and image size."""
        try:
            import torch
            
            if not torch.cuda.is_available():
                return self.gpu_batch_size
            
            # Get GPU memory info
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            
            # Estimate memory usage per 1080p image (3 channels, uint8)
            # 1920 * 1080 * 3 bytes = ~6.2 MB per image
            # Plus model overhead, intermediate tensors, etc.
            memory_per_image_mb = 25  # Conservative estimate including model overhead
            
            # Calculate how many images we can fit in GPU memory
            # Leave 20% memory free for model and intermediate operations
            available_memory_mb = gpu_memory_gb * 1024 * 0.8
            max_images = int(available_memory_mb / memory_per_image_mb)
            
            # Use the smaller of: calculated max, configured batch size, or 16 (safety limit)
            optimal_batch_size = min(max_images, self.gpu_batch_size, 16)
            
            # Ensure minimum batch size of 1
            optimal_batch_size = max(optimal_batch_size, 1)
            
            logger.info(f"GPU memory: {gpu_memory_gb:.1f} GB, estimated {memory_per_image_mb} MB per image")
            logger.info(f"Calculated optimal batch size: {optimal_batch_size}")
            
            return optimal_batch_size
            
        except Exception as e:
            logger.warning(f"Failed to calculate optimal batch size: {e}, using default: {self.gpu_batch_size}")
            return self.gpu_batch_size
    
    def _process_gpu_batch(self, image_paths: List[Path], force: bool) -> List[DetectionResult]:
        """Process a batch of images using GPU batch inference."""
        import torch
        
        # Load and preprocess all images in the batch
        batch_images = []
        batch_metadata = []
        
        for image_path in image_paths:
            # Check if we should skip this image
            if not force:
                original_image_path = image_path.resolve() if image_path.is_symlink() else image_path
                json_path = original_image_path.parent / f"{original_image_path.stem}.json"
                if json_path.exists():
                    try:
                        with open(json_path, 'r') as f:
                            json_data = json.load(f)
                        if 'yolov8' in json_data:
                            logger.info(f"Skipping {image_path.name} - YOLOv8 data already exists")
                            continue
                    except Exception:
                        pass
            
            # Load image
            image = cv2.imread(str(image_path))
            if image is None:
                logger.error(f"Failed to load image: {image_path}")
                continue
            
            original_height, original_width = image.shape[:2]
            
            # Resize image to 1080p for optimal detection performance
            target_width = 1920
            target_height = 1080
            scale_factor = min(target_width / original_width, target_height / original_height)
            
            if scale_factor < 1.0:
                new_width = int(original_width * scale_factor)
                new_height = int(original_height * scale_factor)
                image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
            else:
                scale_factor = 1.0
            
            batch_images.append(image)
            batch_metadata.append({
                'path': image_path,
                'original_width': original_width,
                'original_height': original_height,
                'scale_factor': scale_factor
            })
        
        if not batch_images:
            return []
        
        # Run batch inference
        start_time = cv2.getTickCount()
        results = []
        
        try:
            # Use YOLO's batch processing capability
            batch_results = self.model(batch_images, device=self.device, conf=self.confidence_threshold, verbose=False)
            detection_time = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()
            
            # Process results for each image
            for i, (result, metadata) in enumerate(zip(batch_results, batch_metadata)):
                detected_objects = []
                
                boxes = result.boxes
                if boxes is not None:
                    for j, box in enumerate(boxes):
                        # Get box coordinates (x1, y1, x2, y2)
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = box.conf[0].cpu().numpy()
                        class_id = int(box.cls[0].cpu().numpy())
                        
                        # Filter by target objects if specified
                        if self.target_class_ids is not None and class_id not in self.target_class_ids:
                            continue
                        
                        # Scale coordinates back to original image size
                        orig_x1 = int(x1 / metadata['scale_factor'])
                        orig_y1 = int(y1 / metadata['scale_factor'])
                        orig_x2 = int(x2 / metadata['scale_factor'])
                        orig_y2 = int(y2 / metadata['scale_factor'])
                        
                        # Calculate width and height
                        orig_w = orig_x2 - orig_x1
                        orig_h = orig_y2 - orig_y1
                        
                        # Calculate percentage coordinates
                        x_percent = orig_x1 / metadata['original_width']
                        y_percent = orig_y1 / metadata['original_height']
                        width_percent = orig_w / metadata['original_width']
                        height_percent = orig_h / metadata['original_height']
                        
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
                            object_id=j+1,
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
                            detection_scale_factor=metadata['scale_factor']
                        )
                        
                        detected_objects.append(detected_object)
                
                # Create detection result
                detection_result = DetectionResult(
                    image_path=str(metadata['path']),
                    image_width=metadata['original_width'],
                    image_height=metadata['original_height'],
                    objects_found=len(detected_objects),
                    detection_time=detection_time / len(batch_images),  # Average time per image
                    detected_objects=detected_objects
                )
                
                results.append(detection_result)
                
                # Save sidecar if requested
                if save_sidecar:
                    self._save_sidecar(detection_result)
        
        except Exception as e:
            logger.error(f"GPU batch processing error: {e}")
            # Fallback to individual processing
            for metadata in batch_metadata:
                try:
                    result = self.detect_objects_in_image(metadata['path'], force)
                    results.append(result)
                except Exception as img_error:
                    logger.error(f"Error processing {metadata['path']}: {img_error}")
                    error_result = DetectionResult(
                        image_path=str(metadata['path']),
                        image_width=metadata['original_width'],
                        image_height=metadata['original_height'],
                        objects_found=0,
                        detection_time=0.0,
                        detected_objects=[],
                        error=str(img_error)
                    )
                    results.append(error_result)
        
        return results
    
    def detect_objects_in_image(self, image_path: Path, force: bool = False) -> DetectionResult:
        """
        Detect objects in a single image with comprehensive data collection.
        
        Args:
            image_path: Path to the input image
            force: Whether to force detection even if JSON sidecar exists
            
        Returns:
            Detection result with object information
        """
        # Resolve symlink to get the original image path
        original_image_path = image_path.resolve() if image_path.is_symlink() else image_path
        
        # Check if JSON sidecar already exists and contains YOLOv8 data
        json_path = original_image_path.parent / f"{original_image_path.stem}.json"
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
            results = self.model(image, device=self.device, conf=self.confidence_threshold, verbose=False)
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
    
    def _format_result(self, result: DetectionResult) -> Dict[str, Any]:
        """Format detection result for sportball compatibility."""
        if result.error:
            return {
                "success": False,
                "error": result.error,
                "objects": [],
                "metadata": {
                    "image_path": result.image_path,
                    "image_width": result.image_width,
                    "image_height": result.image_height,
                    "detection_time": result.detection_time
                }
            }
        
        # Format objects for sportball compatibility
        objects = []
        for obj in result.detected_objects:
            objects.append({
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
                "crop_area_percent": {
                    "x": obj.crop_x_percent,
                    "y": obj.crop_y_percent,
                    "width": obj.crop_width_percent,
                    "height": obj.crop_height_percent
                },
                "detection_scale_factor": obj.detection_scale_factor
            })
        
        return {
            "success": True,
            "objects": objects,
            "metadata": {
                "image_path": result.image_path,
                "image_width": result.image_width,
                "image_height": result.image_height,
                "objects_found": result.objects_found,
                "detection_time": result.detection_time,
                "model_path": str(self.model.ckpt_path) if hasattr(self.model, 'ckpt_path') else "yolov8n.pt",
                "border_padding_percentage": self.border_padding * 100,
                "confidence_threshold": self.confidence_threshold,
                "gpu_enabled": self.enable_gpu,
                "target_objects": list(self.target_objects) if self.target_objects else "all"
            }
        }
    
    def extract_objects(self, 
                      image_path: Path, 
                      detection_data: Dict[str, Any], 
                      output_dir: Path,
                      object_types: Optional[List[str]] = None,
                      create_annotated: bool = True,
                      create_individual: bool = True,
                      annotate_individual: bool = False,
                      **kwargs) -> Dict[str, Any]:
        """
        Extract detected objects from an image.
        
        Args:
            image_path: Path to the input image
            detection_data: Detection data from detect_objects
            output_dir: Output directory for extracted objects
            object_types: Types of objects to extract (None for all)
            create_annotated: Whether to create annotated image
            create_individual: Whether to create individual object files
            annotate_individual: Whether to add labels to individual extracted objects
            **kwargs: Additional arguments
            
        Returns:
            Dictionary containing extraction results
        """
        if not detection_data.get('success', False):
            return {"success": False, "error": "No valid detection data"}
        
        objects = detection_data.get('objects', [])
        if not objects:
            return {"success": True, "objects_extracted": 0, "message": "No objects to extract"}
        
        # Filter by object types if specified
        if object_types:
            filtered_objects = []
            for obj in objects:
                if obj['class_name'].lower() in [t.lower() for t in object_types]:
                    filtered_objects.append(obj)
            objects = filtered_objects
        
        if not objects:
            return {"success": True, "objects_extracted": 0, "message": "No objects match filter criteria"}
        
        # Load original image
        image = cv2.imread(str(image_path))
        if image is None:
            return {"success": False, "error": "Failed to load image"}
        
        original_height, original_width = image.shape[:2]
        
        # Create output subdirectories
        image_output_dir = output_dir / image_path.stem
        image_output_dir.mkdir(parents=True, exist_ok=True)
        
        annotated_image_path = None
        individual_objects = []
        
        # Track class-specific counters for better naming
        class_counters = {}
        
        # Process each detected object
        for i, obj_data in enumerate(objects):
            try:
                # Get object information
                class_name = obj_data['class_name']
                class_id = obj_data['class_id']
                confidence = obj_data['confidence']
                
                # Increment class-specific counter
                class_counters[class_name] = class_counters.get(class_name, 0) + 1
                class_counter = class_counters[class_name]
                
                # Get coordinates (use pixel coordinates for extraction)
                coords_pixels = obj_data['coordinates_pixels']
                x = int(coords_pixels['x'])
                y = int(coords_pixels['y'])
                width = int(coords_pixels['width'])
                height = int(coords_pixels['height'])
                
                # Get crop area for individual extraction
                crop_percent = obj_data['crop_area_percent']
                crop_x = int(crop_percent['x'] * original_width)
                crop_y = int(crop_percent['y'] * original_height)
                crop_width = int(crop_percent['width'] * original_width)
                crop_height = int(crop_percent['height'] * original_height)
                
                # Ensure coordinates are within image bounds
                crop_x = max(0, crop_x)
                crop_y = max(0, crop_y)
                crop_width = min(crop_width, original_width - crop_x)
                crop_height = min(crop_height, original_height - crop_y)
                
                # Create individual object filename using class-specific counter
                object_filename = f"{image_path.stem}_{class_name}_{class_counter:02d}.jpg"
                object_path = image_output_dir / object_filename
                
                # Extract individual object
                if create_individual and crop_width > 0 and crop_height > 0:
                    object_image = image[crop_y:crop_y+crop_height, crop_x:crop_x+crop_width]
                    
                    # Add annotation to individual object if requested
                    if annotate_individual:
                        object_image = self._annotate_individual_object(
                            object_image, class_name, confidence
                        )
                    
                    cv2.imwrite(str(object_path), object_image)
                    
                    individual_objects.append({
                        'filename': object_filename,
                        'path': str(object_path),
                        'class_name': class_name,
                        'class_id': class_id,
                        'confidence': confidence,
                        'crop_coordinates': {
                            'x': crop_x, 'y': crop_y, 
                            'width': crop_width, 'height': crop_height
                        },
                        'original_coordinates': {
                            'x': x, 'y': y, 'width': width, 'height': height
                        }
                    })
                
                # Create annotated image
                if create_annotated:
                    # Create label text
                    label = f"{class_name}_{i+1:02d} ({confidence:.2f})"
                    
                    # Get color for this class
                    color = self._get_color_for_class(class_name)
                    
                    # Draw annotation
                    image = self._draw_annotation(image, x, y, width, height, label, color)
                
            except Exception as e:
                logger.error(f"Error processing object {i+1}: {e}")
                continue
        
        # Save annotated image
        if create_annotated and objects:
            annotated_filename = f"{image_path.stem}_annotated.jpg"
            annotated_image_path = image_output_dir / annotated_filename
            cv2.imwrite(str(annotated_image_path), image)
        
        return {
            "success": True,
            "objects_extracted": len(objects),
            "annotated_image_path": str(annotated_image_path) if annotated_image_path else None,
            "individual_objects": individual_objects,
            "output_directory": str(image_output_dir)
        }
    
    def _get_color_for_class(self, class_name: str) -> Tuple[int, int, int]:
        """Get color for a specific object class."""
        # Define colors for common object classes
        color_map = {
            'person': (0, 255, 0),        # Green
            'sports ball': (255, 0, 0),   # Red
            'car': (0, 0, 255),           # Blue
            'truck': (255, 255, 0),      # Cyan
            'bus': (255, 0, 255),         # Magenta
            'bicycle': (0, 255, 255),     # Yellow
            'motorcycle': (128, 0, 128),  # Purple
            'airplane': (255, 165, 0),    # Orange
            'boat': (0, 128, 255),        # Light Blue
            'train': (128, 128, 0),       # Olive
        }
        
        return color_map.get(class_name, (255, 255, 255))  # Default to white
    
    def _draw_annotation(self, image: np.ndarray, x: int, y: int, width: int, height: int, 
                        label: str, color: Tuple[int, int, int]) -> np.ndarray:
        """Draw annotation on image."""
        annotated_image = image.copy()
        
        # Draw rectangle
        cv2.rectangle(annotated_image, (x, y), (x + width, y + height), color, 2)
        
        # Draw label background
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2
        )
        
        # Position label above the bounding box
        label_x = x
        label_y = y - 10 if y - 10 > text_height else y + height + text_height + 10
        
        # Draw label background rectangle
        cv2.rectangle(annotated_image, 
                     (label_x, label_y - text_height - baseline),
                     (label_x + text_width, label_y + baseline),
                     color, -1)
        
        # Draw label text
        cv2.putText(annotated_image, label, (label_x, label_y - baseline),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        return annotated_image
    
    def _annotate_individual_object(self, object_image: np.ndarray, class_name: str, 
                                   confidence: float) -> np.ndarray:
        """Add annotation to an individual extracted object."""
        annotated_image = object_image.copy()
        
        # Create label text
        label = f"{class_name} {confidence:.2f}"
        
        # Get text size for positioning
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2
        )
        
        # Position label at top-left corner of the object
        label_x = 10
        label_y = text_height + 10
        
        # Ensure label fits within image bounds
        img_height, img_width = annotated_image.shape[:2]
        if label_x + text_width > img_width:
            label_x = img_width - text_width - 10
        if label_y > img_height:
            label_y = img_height - 10
        
        # Draw background rectangle for label
        cv2.rectangle(annotated_image,
                     (label_x, label_y - text_height - baseline),
                     (label_x + text_width, label_y + baseline),
                     (0, 0, 0), -1)  # Black background
        
        # Draw label text
        cv2.putText(annotated_image, label, (label_x, label_y - baseline),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        return annotated_image
    
    def get_available_classes(self) -> Dict[int, str]:
        """Get all available object classes."""
        return COCO_CLASSES.copy()
    
    def list_available_objects(self) -> None:
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
        print("  --classes person,sports ball  # Detect people and sports balls")
        print("  --classes car,truck,bus        # Detect vehicles")
        print("  --classes person              # Detect only people")
    
    def _save_sidecar(self, detection_result: DetectionResult) -> bool:
        """
        Save detection result to sidecar file.
        
        Args:
            detection_result: Detection result to save
            
        Returns:
            True if successful, False otherwise
        """
        try:
            image_path = Path(detection_result.image_path)
            
            # Format the result for JSON serialization
            formatted_result = self._format_result(detection_result)
            
            # Save to sidecar file
            sidecar_path = image_path.with_suffix('.json')
            
            # Load existing data if file exists
            existing_data = {}
            if sidecar_path.exists():
                try:
                    with open(sidecar_path, 'r') as f:
                        existing_data = json.load(f)
                except Exception as e:
                    logger.warning(f"Could not read existing sidecar {sidecar_path}: {e}")
                    existing_data = {}
            
            # Merge the new data with existing data
            merged_data = existing_data.copy()
            merged_data['yolov8'] = formatted_result
            
            # Update sidecar_info if it exists, otherwise create new
            if 'sidecar_info' in existing_data:
                merged_data['sidecar_info'].update({
                    'last_updated': __import__('datetime').datetime.now().isoformat(),
                    'last_operation': 'yolov8'
                })
            else:
                merged_data['sidecar_info'] = {
                    'operation_type': 'yolov8',
                    'created_at': __import__('datetime').datetime.now().isoformat(),
                    'last_updated': __import__('datetime').datetime.now().isoformat(),
                    'last_operation': 'yolov8',
                    'image_path': str(image_path),
                    'symlink_path': str(image_path),
                    'symlink_info': {
                        'symlink_path': str(image_path),
                        'target_path': str(image_path),
                        'is_symlink': False
                    }
                }
            
            # Save the merged data
            with open(sidecar_path, 'w') as f:
                json.dump(merged_data, f, indent=2, cls=NumpyEncoder)
            
            logger.debug(f"Saved YOLOv8 detection results to {sidecar_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save sidecar for {detection_result.image_path}: {e}")
            return False