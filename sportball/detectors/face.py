"""
Face Detection Module

Face detection and recognition functionality for sportball.

Author: Claude Sonnet 4 (claude-3-5-sonnet-20241022)
Generated via Cursor IDE (cursor.sh) with AI assistance
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from loguru import logger

try:
    import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
except ImportError:
    FACE_RECOGNITION_AVAILABLE = False
    logger.warning("face_recognition not available - face encodings will be skipped")

from ..decorators import gpu_accelerated, cached_result


@dataclass
class DetectedFace:
    """Information about a detected face."""
    face_id: int
    bbox: tuple  # (x, y, width, height)
    confidence: float
    landmarks: Optional[List[tuple]] = None
    encoding: Optional[List[float]] = None


@dataclass
class FaceDetectionResult:
    """Result of face detection operation."""
    faces: List[DetectedFace]
    face_count: int
    success: bool
    processing_time: float
    error: Optional[str] = None


class FaceDetector:
    """
    Face detection and recognition using OpenCV and face_recognition.
    """
    
    def __init__(self, 
                 enable_gpu: bool = True,
                 cache_enabled: bool = True,
                 confidence_threshold: float = 0.5,
                 min_face_size: int = 64,
                 batch_size: int = 8):
        """
        Initialize face detector.
        
        Args:
            enable_gpu: Whether to enable GPU acceleration
            cache_enabled: Whether to enable result caching
            confidence_threshold: Minimum confidence for face detection
            min_face_size: Minimum face size in pixels
            batch_size: Batch size for processing multiple images
        """
        self.enable_gpu = enable_gpu
        self.cache_enabled = cache_enabled
        self.confidence_threshold = confidence_threshold
        self.min_face_size = min_face_size
        self.batch_size = batch_size
        
        # Initialize OpenCV face cascade
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Initialize face recognition if available
        self.face_recognition_available = FACE_RECOGNITION_AVAILABLE
        
        self.logger = logger.bind(component="face_detector")
        self.logger.info("Initialized FaceDetector")
    
    @gpu_accelerated(fallback_cpu=True)
    @cached_result(expire_seconds=3600)  # Cache for 1 hour
    def detect_faces(self, 
                    image_path: Path, 
                    confidence: Optional[float] = None,
                    min_faces: int = 1,
                    max_faces: Optional[int] = None,
                    face_size: int = 64) -> FaceDetectionResult:
        """
        Detect faces in an image.
        
        Args:
            image_path: Path to the image file
            confidence: Detection confidence threshold
            min_faces: Minimum number of faces to detect
            max_faces: Maximum number of faces to detect
            face_size: Minimum face size in pixels
            
        Returns:
            FaceDetectionResult containing detected faces
        """
        import time
        start_time = time.time()
        
        try:
            # Load image
            image = cv2.imread(str(image_path))
            if image is None:
                return FaceDetectionResult(
                    faces=[],
                    face_count=0,
                    success=False,
                    processing_time=0,
                    error="Failed to load image"
                )
            
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Detect faces using OpenCV
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(face_size, face_size),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            # Filter faces by confidence and size
            detected_faces = []
            for i, (x, y, w, h) in enumerate(faces):
                if w >= face_size and h >= face_size:
                    # Calculate confidence (simplified)
                    face_confidence = min(1.0, (w * h) / (face_size * face_size))
                    
                    if confidence is None or face_confidence >= confidence:
                        detected_face = DetectedFace(
                            face_id=i,
                            bbox=(x, y, w, h),
                            confidence=face_confidence
                        )
                        
                        # Add face encoding if available
                        if self.face_recognition_available:
                            try:
                                face_encoding = self._get_face_encoding(image, (x, y, w, h))
                                detected_face.encoding = face_encoding
                            except Exception as e:
                                self.logger.warning(f"Failed to get face encoding: {e}")
                        
                        detected_faces.append(detected_face)
            
            # Apply min/max face constraints
            if len(detected_faces) < min_faces:
                return FaceDetectionResult(
                    faces=[],
                    face_count=0,
                    success=False,
                    processing_time=time.time() - start_time,
                    error=f"Not enough faces detected (found {len(detected_faces)}, required {min_faces})"
                )
            
            if max_faces and len(detected_faces) > max_faces:
                # Sort by confidence and take top N
                detected_faces.sort(key=lambda f: f.confidence, reverse=True)
                detected_faces = detected_faces[:max_faces]
            
            processing_time = time.time() - start_time
            
            return FaceDetectionResult(
                faces=detected_faces,
                face_count=len(detected_faces),
                success=True,
                processing_time=processing_time
            )
            
        except Exception as e:
            self.logger.error(f"Face detection failed: {e}")
            return FaceDetectionResult(
                faces=[],
                face_count=0,
                success=False,
                processing_time=time.time() - start_time,
                error=str(e)
            )
    
    def detect_faces_batch(self, 
                          image_paths: List[Path], 
                          confidence: Optional[float] = None,
                          min_faces: int = 1,
                          max_faces: Optional[int] = None,
                          face_size: int = 64) -> Dict[str, FaceDetectionResult]:
        """
        Detect faces in multiple images using batch processing.
        
        Args:
            image_paths: List of image paths
            confidence: Detection confidence threshold
            min_faces: Minimum number of faces to detect
            max_faces: Maximum number of faces to detect
            face_size: Minimum face size in pixels
            
        Returns:
            Dictionary mapping image paths to detection results
        """
        self.logger.info(f"Processing {len(image_paths)} images in batches of {self.batch_size}")
        
        results = {}
        
        # Process images in batches
        for i in range(0, len(image_paths), self.batch_size):
            batch_paths = image_paths[i:i + self.batch_size]
            self.logger.info(f"Processing face detection batch {i//self.batch_size + 1}: {len(batch_paths)} images")
            
            try:
                batch_results = self._process_face_batch(
                    batch_paths, confidence, min_faces, max_faces, face_size
                )
                results.update(batch_results)
            except Exception as e:
                self.logger.error(f"Face batch processing failed: {e}")
                # Fallback to individual processing
                for image_path in batch_paths:
                    try:
                        result = self.detect_faces(image_path, confidence, min_faces, max_faces, face_size)
                        results[str(image_path)] = result
                    except Exception as img_error:
                        self.logger.error(f"Error processing {image_path}: {img_error}")
                        results[str(image_path)] = FaceDetectionResult(
                            faces=[],
                            face_count=0,
                            success=False,
                            processing_time=0.0,
                            error=str(img_error)
                        )
        
        return results
    
    def _process_face_batch(self, 
                           image_paths: List[Path], 
                           confidence: Optional[float],
                           min_faces: int,
                           max_faces: Optional[int],
                           face_size: int) -> Dict[str, FaceDetectionResult]:
        """Process a batch of images for face detection."""
        import time
        start_time = time.time()
        
        results = {}
        
        # Load all images in the batch
        batch_images = []
        batch_metadata = []
        
        for image_path in image_paths:
            try:
                image = cv2.imread(str(image_path))
                if image is None:
                    results[str(image_path)] = FaceDetectionResult(
                        faces=[],
                        face_count=0,
                        success=False,
                        processing_time=0.0,
                        error="Failed to load image"
                    )
                    continue
                
                batch_images.append(image)
                batch_metadata.append({
                    'path': image_path,
                    'original_height': image.shape[0],
                    'original_width': image.shape[1]
                })
            except Exception as e:
                results[str(image_path)] = FaceDetectionResult(
                    faces=[],
                    face_count=0,
                    success=False,
                    processing_time=0.0,
                    error=str(e)
                )
        
        if not batch_images:
            return results
        
        # Process each image in the batch
        batch_processing_time = time.time() - start_time
        
        for i, (image, metadata) in enumerate(zip(batch_images, batch_metadata)):
            try:
                # Convert to grayscale for face detection
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                
                # Detect faces using OpenCV
                faces = self.face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(face_size, face_size),
                    flags=cv2.CASCADE_SCALE_IMAGE
                )
                
                # Filter faces by confidence and size
                detected_faces = []
                for j, (x, y, w, h) in enumerate(faces):
                    if w >= face_size and h >= face_size:
                        # Calculate confidence (simplified)
                        face_confidence = min(1.0, (w * h) / (face_size * face_size))
                        
                        if confidence is None or face_confidence >= confidence:
                            detected_face = DetectedFace(
                                face_id=j,
                                bbox=(x, y, w, h),
                                confidence=face_confidence
                            )
                            
                            # Add face encoding if available
                            if self.face_recognition_available:
                                try:
                                    face_encoding = self._get_face_encoding(image, (x, y, w, h))
                                    detected_face.encoding = face_encoding
                                except Exception as e:
                                    self.logger.warning(f"Failed to get face encoding: {e}")
                            
                            detected_faces.append(detected_face)
                
                # Apply min/max face constraints
                if len(detected_faces) < min_faces:
                    results[str(metadata['path'])] = FaceDetectionResult(
                        faces=[],
                        face_count=0,
                        success=False,
                        processing_time=batch_processing_time / len(batch_images),
                        error=f"Not enough faces detected (found {len(detected_faces)}, required {min_faces})"
                    )
                    continue
                
                if max_faces and len(detected_faces) > max_faces:
                    # Sort by confidence and take top N
                    detected_faces.sort(key=lambda f: f.confidence, reverse=True)
                    detected_faces = detected_faces[:max_faces]
                
                results[str(metadata['path'])] = FaceDetectionResult(
                    faces=detected_faces,
                    face_count=len(detected_faces),
                    success=True,
                    processing_time=batch_processing_time / len(batch_images)
                )
                
            except Exception as e:
                results[str(metadata['path'])] = FaceDetectionResult(
                    faces=[],
                    face_count=0,
                    success=False,
                    processing_time=batch_processing_time / len(batch_images),
                    error=str(e)
                )
        
        return results
    
    def _get_face_encoding(self, image: np.ndarray, bbox: tuple) -> Optional[List[float]]:
        """
        Get face encoding for a detected face.
        
        Args:
            image: Input image
            bbox: Face bounding box (x, y, w, h)
            
        Returns:
            Face encoding or None if failed
        """
        if not self.face_recognition_available:
            return None
        
        try:
            x, y, w, h = bbox
            face_image = image[y:y+h, x:x+w]
            
            # Convert BGR to RGB
            face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            
            # Get face encodings
            encodings = face_recognition.face_encodings(face_rgb)
            
            if encodings:
                return encodings[0].tolist()
            
        except Exception as e:
            self.logger.warning(f"Failed to get face encoding: {e}")
        
        return None
    
    def extract_faces(self, 
                     image_path: Path, 
                     output_dir: Path,
                     face_size: int = 64,
                     padding: int = 10) -> Dict[str, Any]:
        """
        Extract detected faces from an image.
        
        Args:
            image_path: Path to the input image
            output_dir: Directory to save extracted faces
            face_size: Size of extracted faces
            padding: Padding around face in pixels
            
        Returns:
            Dictionary containing extraction results
        """
        try:
            # Detect faces first
            detection_result = self.detect_faces(image_path)
            
            if not detection_result.success:
                return {
                    "success": False,
                    "error": detection_result.error,
                    "faces_extracted": 0
                }
            
            # Load image
            image = cv2.imread(str(image_path))
            if image is None:
                return {
                    "success": False,
                    "error": "Failed to load image",
                    "faces_extracted": 0
                }
            
            # Create output directory
            output_dir.mkdir(parents=True, exist_ok=True)
            
            extracted_faces = []
            
            for i, face in enumerate(detection_result.faces):
                x, y, w, h = face.bbox
                
                # Add padding
                x_start = max(0, x - padding)
                y_start = max(0, y - padding)
                x_end = min(image.shape[1], x + w + padding)
                y_end = min(image.shape[0], y + h + padding)
                
                # Extract face
                face_image = image[y_start:y_end, x_start:x_end]
                
                # Resize if needed
                if face_size > 0:
                    face_image = cv2.resize(face_image, (face_size, face_size))
                
                # Save face
                face_filename = f"{image_path.stem}_face_{i:02d}.jpg"
                face_path = output_dir / face_filename
                
                cv2.imwrite(str(face_path), face_image)
                
                extracted_faces.append({
                    "face_id": face.face_id,
                    "bbox": face.bbox,
                    "confidence": face.confidence,
                    "output_path": str(face_path)
                })
            
            return {
                "success": True,
                "faces_extracted": len(extracted_faces),
                "faces": extracted_faces
            }
            
        except Exception as e:
            self.logger.error(f"Face extraction failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "faces_extracted": 0
            }
    
    def _format_result(self, result: FaceDetectionResult, image_path: Path) -> Dict[str, Any]:
        """
        Format face detection result for JSON serialization.
        
        Args:
            result: Face detection result
            image_path: Path to the image file
            
        Returns:
            Dictionary containing formatted face detection data
        """
        if not result.success:
            return {
                "success": False,
                "error": result.error,
                "faces": [],
                "metadata": {
                    "image_path": str(image_path),
                    "faces_found": 0,
                    "processing_time": float(result.processing_time),
                    "extraction_timestamp": __import__('datetime').datetime.now().isoformat()
                }
            }
        
        # Format faces for sportball compatibility
        faces = []
        for face in result.faces:
            face_data = {
                "face_id": int(face.face_id),
                "bbox": {
                    "x": int(face.bbox[0]),
                    "y": int(face.bbox[1]),
                    "width": int(face.bbox[2]),
                    "height": int(face.bbox[3])
                },
                "confidence": float(face.confidence)
            }
            
            # Add encoding if available
            if face.encoding is not None:
                # Convert numpy array to list for JSON serialization
                if hasattr(face.encoding, 'tolist'):
                    face_data["encoding"] = face.encoding.tolist()
                else:
                    face_data["encoding"] = list(face.encoding)
            
            faces.append(face_data)
        
        return {
            "success": True,
            "faces": faces,
            "metadata": {
                "image_path": str(image_path),
                "faces_found": int(result.face_count),
                "processing_time": float(result.processing_time),
                "extraction_timestamp": __import__('datetime').datetime.now().isoformat(),
                "face_size_threshold": int(getattr(self, 'face_size', 64)),
                "confidence_threshold": float(getattr(self, 'confidence_threshold', 0.5))
            }
        }