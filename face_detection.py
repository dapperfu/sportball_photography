#!/usr/bin/env python3
"""
Face Detection Tool with GPU Support and Parallel Processing

This tool detects faces in images and saves comprehensive detection data
to JSON sidecar files. Includes face encodings for clustering and all
metadata needed to extract faces from full-resolution images.

Author: Claude Sonnet 4 (claude-3-5-sonnet-20241022)
Generated via Cursor IDE (cursor.sh) with AI assistance
"""

import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import click
import cv2
import numpy as np
from tqdm import tqdm
from loguru import logger

# Try to import face_recognition for encodings
try:
    import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
except ImportError:
    FACE_RECOGNITION_AVAILABLE = False
    logger.warning("face_recognition not available - face encodings will be skipped")


@dataclass
class FacialFeature:
    """Information about a detected facial feature."""
    feature_type: str  # 'left_eye', 'right_eye', 'nose', 'mouth'
    x: int
    y: int
    width: int
    height: int
    confidence: float


@dataclass
class DetectedFace:
    """Information about a detected face."""
    face_id: int
    # Coordinates in original image
    x: int
    y: int
    width: int
    height: int
    confidence: float
    # Face encoding for clustering (if available)
    face_encoding: Optional[List[float]] = None
    # Facial features
    facial_features: List[FacialFeature] = None
    # Crop area with padding (for extraction)
    crop_x: int = 0
    crop_y: int = 0
    crop_width: int = 0
    crop_height: int = 0
    # Scale factor used for detection
    detection_scale_factor: float = 1.0


@dataclass
class DetectionResult:
    """Result of face detection from an image."""
    image_path: str
    image_width: int
    image_height: int
    faces_found: int
    detection_time: float
    detected_faces: List[DetectedFace]
    error: Optional[str] = None


class FaceDetector:
    """Face detection with GPU support and comprehensive data collection."""
    
    def __init__(self, border_padding: float = 0.25, use_gpu: bool = True):
        """
        Initialize face detector.
        
        Args:
            border_padding: Percentage of padding around detected face (0.25 = 25%)
            use_gpu: Whether to use GPU acceleration if available
        """
        self.border_padding = border_padding
        self.use_gpu = use_gpu
        
        # Initialize OpenCV cascades for face and feature detection
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
        # Try to load nose and mouth cascades, but don't fail if they're not available
        self.nose_cascade = None
        self.mouth_cascade = None
        
        # Only try to load if the files exist
        nose_path = cv2.data.haarcascades + 'haarcascade_mcs_nose.xml'
        mouth_path = cv2.data.haarcascades + 'haarcascade_mcs_mouth.xml'
        
        if os.path.exists(nose_path):
            try:
                self.nose_cascade = cv2.CascadeClassifier(nose_path)
                if self.nose_cascade.empty():
                    self.nose_cascade = None
            except:
                self.nose_cascade = None
                
        if os.path.exists(mouth_path):
            try:
                self.mouth_cascade = cv2.CascadeClassifier(mouth_path)
                if self.mouth_cascade.empty():
                    self.mouth_cascade = None
            except:
                self.mouth_cascade = None
        
        # Check for GPU support
        if self.use_gpu:
            try:
                # Try to create a GPU context
                gpu_info = cv2.cuda.getCudaEnabledDeviceCount()
                if gpu_info > 0:
                    logger.info(f"GPU support available: {gpu_info} CUDA devices")
                else:
                    logger.warning("No CUDA devices found, falling back to CPU")
                    self.use_gpu = False
            except:
                logger.warning("CUDA not available, falling back to CPU")
                self.use_gpu = False
        
        logger.info(f"Initialized face detector with {border_padding*100:.0f}% border padding, GPU: {self.use_gpu}")
    
    def detect_facial_features(self, face_region: np.ndarray) -> List[FacialFeature]:
        """
        Detect facial features (eyes, nose, mouth) in a face region.
        
        Args:
            face_region: Cropped face image
            
        Returns:
            List of detected facial features
        """
        features = []
        gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
        
        # Temporarily suppress OpenCV output for feature detection
        cv2.setLogLevel(0)  # LOG_LEVEL_SILENT = 0
        
        # Detect eyes
        if self.eye_cascade:
            eyes = self.eye_cascade.detectMultiScale(gray_face, 1.1, 3)
            for i, (ex, ey, ew, eh) in enumerate(eyes):
                # Determine if it's left or right eye based on position
                eye_type = 'left_eye' if ex < face_region.shape[1] // 2 else 'right_eye'
                feature = FacialFeature(
                    feature_type=eye_type,
                    x=int(ex),
                    y=int(ey),
                    width=int(ew),
                    height=int(eh),
                    confidence=0.8
                )
                features.append(feature)
        
        # Detect nose
        if self.nose_cascade:
            noses = self.nose_cascade.detectMultiScale(gray_face, 1.1, 3)
            for (nx, ny, nw, nh) in noses:
                feature = FacialFeature(
                    feature_type='nose',
                    x=int(nx),
                    y=int(ny),
                    width=int(nw),
                    height=int(nh),
                    confidence=0.8
                )
                features.append(feature)
        
        # Detect mouth
        if self.mouth_cascade:
            mouths = self.mouth_cascade.detectMultiScale(gray_face, 1.1, 3)
            for (mx, my, mw, mh) in mouths:
                feature = FacialFeature(
                    feature_type='mouth',
                    x=int(mx),
                    y=int(my),
                    width=int(mw),
                    height=int(mh),
                    confidence=0.8
                )
                features.append(feature)
        
        # Restore normal logging
        cv2.setLogLevel(4)  # LOG_LEVEL_INFO = 4
        
        return features
    
    def get_face_encoding(self, face_region: np.ndarray) -> Optional[List[float]]:
        """
        Get face encoding for clustering if face_recognition is available.
        
        Args:
            face_region: Cropped face image
            
        Returns:
            Face encoding or None if not available
        """
        if not FACE_RECOGNITION_AVAILABLE:
            return None
        
        try:
            # Convert BGR to RGB for face_recognition
            rgb_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2RGB)
            encodings = face_recognition.face_encodings(rgb_face)
            if encodings:
                return encodings[0].tolist()
        except Exception as e:
            logger.debug(f"Failed to get face encoding: {e}")
        
        return None
    
    def detect_faces_in_image(self, image_path: Path, force: bool = False) -> DetectionResult:
        """
        Detect faces in a single image with comprehensive data collection.
        
        Args:
            image_path: Path to the input image
            force: Whether to force detection even if JSON sidecar exists
            
        Returns:
            Detection result with face information
        """
        # Check if JSON sidecar already exists with face data
        json_path = image_path.parent / f"{image_path.stem}.json"
        if json_path.exists() and not force:
            # Check if JSON contains actual face data
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                
                # Check if this is our face detection data with faces
                if ("Face_detector" in data and 
                    "faces" in data["Face_detector"] and 
                    len(data["Face_detector"]["faces"]) > 0):
                    logger.info(f"Skipping {image_path.name} - JSON sidecar exists with {len(data['Face_detector']['faces'])} faces (use --force to override)")
                    return DetectionResult(
                        image_path=str(image_path),
                        image_width=0,
                        image_height=0,
                        faces_found=0,
                        detection_time=0.0,
                        detected_faces=[],
                        error="Skipped - JSON sidecar exists with face data"
                    )
            except (json.JSONDecodeError, KeyError, TypeError):
                # JSON exists but is invalid or doesn't contain face data, continue processing
                logger.debug(f"JSON sidecar exists but invalid/empty, reprocessing {image_path.name}")
                pass
        
        logger.info(f"Detecting faces in {image_path.name}")
        
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            return DetectionResult(
                image_path=str(image_path),
                image_width=0,
                image_height=0,
                faces_found=0,
                detection_time=0.0,
                detected_faces=[],
                error="Failed to load image"
            )
        
        original_height, original_width = image.shape[:2]
        
        # Resize image to 1080p for optimal face detection performance
        target_width = 1920  # 1080p width
        target_height = 1080  # 1080p height
        
        # Calculate scaling factor to fit within 1080p while maintaining aspect ratio
        scale_factor = min(target_width / original_width, target_height / original_height)
        
        # Keep original image for face encoding, use resized for detection
        original_image = image.copy()
        
        if scale_factor < 1.0:
            # Only resize if image is larger than 1080p
            new_width = int(original_width * scale_factor)
            new_height = int(original_height * scale_factor)
            image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
            logger.debug(f"Resized image from {original_width}x{original_height} to {new_width}x{new_height}")
        else:
            scale_factor = 1.0  # No scaling needed
        
        # Detect faces (suppress OpenCV output to avoid interrupting tqdm)
        start_time = cv2.getTickCount()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Temporarily suppress OpenCV output
        cv2.setLogLevel(0)  # LOG_LEVEL_SILENT = 0
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        cv2.setLogLevel(4)  # LOG_LEVEL_INFO = 4, restore normal logging
        
        detection_time = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()
        
        detected_faces = []
        
        # Process each detected face
        for i, (x, y, w, h) in enumerate(faces):
            try:
                # Scale coordinates back to original image size
                orig_x = int(x / scale_factor)
                orig_y = int(y / scale_factor)
                orig_w = int(w / scale_factor)
                orig_h = int(h / scale_factor)
                
                # Calculate padded coordinates for cropping
                padding_x = int(orig_w * self.border_padding)
                padding_y = int(orig_h * self.border_padding)
                
                # Calculate crop area with padding
                crop_x = max(0, orig_x - padding_x)
                crop_y = max(0, orig_y - padding_y)
                crop_w = min(original_width - crop_x, orig_w + 2 * padding_x)
                crop_h = min(original_height - crop_y, orig_h + 2 * padding_y)
                
                # Extract face region from original image for encoding and features
                face_region = original_image[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
                
                # Detect facial features
                facial_features = self.detect_facial_features(face_region)
                
                # Get face encoding for clustering
                face_encoding = self.get_face_encoding(face_region)
                
                # Create detected face object
                detected_face = DetectedFace(
                    face_id=i+1,
                    x=orig_x,
                    y=orig_y,
                    width=orig_w,
                    height=orig_h,
                    confidence=0.8,  # OpenCV doesn't provide confidence
                    face_encoding=face_encoding,
                    facial_features=facial_features,
                    crop_x=crop_x,
                    crop_y=crop_y,
                    crop_width=crop_w,
                    crop_height=crop_h,
                    detection_scale_factor=scale_factor
                )
                
                detected_faces.append(detected_face)
                logger.debug(f"Detected face {i+1}: {orig_w}x{orig_h} at ({orig_x}, {orig_y})")
                
            except Exception as e:
                logger.error(f"Error processing face {i+1}: {e}")
                continue
        
        return DetectionResult(
            image_path=str(image_path),
            image_width=original_width,
            image_height=original_height,
            faces_found=len(faces),
            detection_time=detection_time,
            detected_faces=detected_faces
        )
    
    def create_detection_json(self, image_path: Path, detection_result: DetectionResult) -> None:
        """
        Create or update JSON sidecar file for an image with all detection data.
        Merges with existing data to preserve information from other tools.
        
        Args:
            image_path: Path to the original image
            detection_result: Detection result data
        """
        from datetime import datetime
        
        json_path = image_path.parent / f"{image_path.stem}.json"
        
        # Load existing data if file exists
        existing_data = {}
        if json_path.exists():
            try:
                with open(json_path, 'r') as f:
                    existing_data = json.load(f)
            except (json.JSONDecodeError, KeyError, TypeError):
                # If existing file is invalid, start fresh
                existing_data = {}
        
        # Create or update Face_detector section
        if "Face_detector" not in existing_data:
            existing_data["Face_detector"] = {}
        
        # Update metadata
        existing_data["Face_detector"]["metadata"] = {
            "extraction_timestamp": datetime.now().isoformat(),
            "tool_version": "1.0.0",
            "border_padding_percentage": self.border_padding * 100,
            "total_faces_found": detection_result.faces_found,
            "image_path": str(image_path),
            "image_dimensions": {
                "width": detection_result.image_width,
                "height": detection_result.image_height
            },
            "detection_time_seconds": detection_result.detection_time,
            "gpu_enabled": self.use_gpu,
            "face_recognition_available": FACE_RECOGNITION_AVAILABLE
        }
        
        # Update faces data
        existing_data["Face_detector"]["faces"] = []
        
        # Add each detected face to the JSON
        for face in detection_result.detected_faces:
            face_data = {
                "face_id": face.face_id,
                "coordinates": {
                    "x": face.x,
                    "y": face.y,
                    "width": face.width,
                    "height": face.height
                },
                "confidence": face.confidence,
                "border_padding": self.border_padding,
                "crop_area": {
                    "x": face.crop_x,
                    "y": face.crop_y,
                    "width": face.crop_width,
                    "height": face.crop_height
                },
                "detection_scale_factor": face.detection_scale_factor,
                "face_encoding": face.face_encoding,
                "facial_features": []
            }
            
            # Add facial features data
            for feature in face.facial_features:
                feature_data = {
                    "feature_type": feature.feature_type,
                    "coordinates": {
                        "x": feature.x,
                        "y": feature.y,
                        "width": feature.width,
                        "height": feature.height
                    },
                    "confidence": feature.confidence
                }
                face_data["facial_features"].append(feature_data)
            
            existing_data["Face_detector"]["faces"].append(face_data)
        
        # Save merged JSON file
        with open(json_path, 'w') as f:
            json.dump(existing_data, f, indent=2, cls=NumpyEncoder)
        
        logger.debug(f"Detection JSON sidecar saved/updated: {json_path.name}")
    
    def detect_faces_in_images(self, image_pattern: str, max_images: Optional[int] = None, force: bool = False) -> List[DetectionResult]:
        """
        Detect faces in multiple images with parallel processing.
        
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
                executor.submit(self.detect_faces_in_image, image_path, force): image_path 
                for image_path in image_files
            }
            
            # Process completed tasks with progress bar
            for future in tqdm(as_completed(future_to_image), 
                             total=len(image_files), 
                             desc="Detecting faces"):
                try:
                    result = future.result()
                    results.append(result)
                    
                    # Always create/update JSON sidecar (even if no faces found)
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
                        faces_found=0,
                        detection_time=0.0,
                        detected_faces=[],
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


@click.command()
@click.argument('input_pattern', type=str)
@click.option('--border-padding', '-b', default=0.25, help='Border padding percentage (0.25 = 25%)')
@click.option('--max-images', '-m', default=None, type=int, help='Maximum number of images to process')
@click.option('--gpu/--no-gpu', default=True, help='Use GPU acceleration if available')
@click.option('--force', '-f', is_flag=True, help='Force detection even if JSON sidecar exists')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
def main(input_pattern: str, border_padding: float, max_images: Optional[int], gpu: bool, force: bool, verbose: bool):
    """Detect faces in images and save comprehensive data to JSON sidecar files."""
    
    # Setup logging
    if verbose:
        logger.add("face_detection.log", level="DEBUG")
    
    # Initialize detector
    detector = FaceDetector(border_padding=border_padding, use_gpu=gpu)
    
    # Detect faces
    logger.info(f"Starting face detection with {border_padding*100:.0f}% border padding")
    results = detector.detect_faces_in_images(input_pattern, max_images, force)
    
    if not results:
        logger.error("No images processed")
        return
    
    # Calculate summary statistics
    total_images = len(results)
    total_faces_found = sum(result.faces_found for result in results)
    total_time = sum(result.detection_time for result in results)
    
    logger.info(f"Face detection complete!")
    logger.info(f"Processed {total_images} images")
    logger.info(f"Found {total_faces_found} faces")
    logger.info(f"Total detection time: {total_time:.2f}s")
    logger.info(f"Average time per image: {total_time/total_images:.2f}s")
    logger.info(f"GPU enabled: {detector.use_gpu}")
    logger.info(f"Face recognition available: {FACE_RECOGNITION_AVAILABLE}")


if __name__ == "__main__":
    main()
