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
    
    def tune_gpu_batch_size(self, 
                           test_image_paths: Optional[List[Path]] = None,
                           max_test_images: int = 50,
                           start_batch_size: int = 1,
                           max_batch_size: int = 64,
                           image_size: tuple = (1920, 1080)) -> int:
        """
        Automatically tune GPU batch size by testing until memory limit is reached.
        
        Args:
            test_image_paths: List of test image paths (if None, creates synthetic images)
            max_test_images: Maximum number of test images to use
            start_batch_size: Starting batch size for testing
            max_batch_size: Maximum batch size to test
            image_size: Size of test images (width, height)
            
        Returns:
            Optimal batch size that doesn't cause memory errors
        """
        if not self.enable_gpu:
            self.logger.info("GPU not enabled, skipping batch size tuning")
            return self.batch_size
        
        try:
            import torch
            import cv2
            import numpy as np
            
            if not torch.cuda.is_available():
                self.logger.info("CUDA not available, skipping batch size tuning")
                return self.batch_size
            
            self.logger.info(f"Starting GPU batch size tuning (max: {max_batch_size})")
            
            # Create test images if not provided
            if test_image_paths is None:
                test_image_paths = self._create_test_images(max_test_images, image_size)
            
            # Limit test images
            test_image_paths = test_image_paths[:max_test_images]
            
            optimal_batch_size = start_batch_size
            last_successful_batch_size = start_batch_size
            
            # Binary search approach for efficiency
            low, high = start_batch_size, max_batch_size
            
            while low <= high:
                mid_batch_size = (low + high) // 2
                
                self.logger.info(f"Testing batch size: {mid_batch_size}")
                
                if self._test_batch_size(test_image_paths, mid_batch_size):
                    # Success - try larger batch size
                    optimal_batch_size = mid_batch_size
                    last_successful_batch_size = mid_batch_size
                    low = mid_batch_size + 1
                    self.logger.info(f"✅ Batch size {mid_batch_size} successful")
                else:
                    # Failure - try smaller batch size
                    high = mid_batch_size - 1
                    self.logger.warning(f"❌ Batch size {mid_batch_size} failed (GPU memory)")
            
            # Clean up test images if we created them
            if test_image_paths and len(test_image_paths) > 0:
                self._cleanup_test_images(test_image_paths)
            
            self.logger.info(f"🎯 Optimal GPU batch size: {optimal_batch_size}")
            return optimal_batch_size
            
        except Exception as e:
            self.logger.error(f"GPU batch size tuning failed: {e}")
            return self.batch_size
    
    def _create_test_images(self, count: int, image_size: tuple) -> List[Path]:
        """Create synthetic test images for batch size tuning."""
        import cv2
        import numpy as np
        import tempfile
        from pathlib import Path
        
        test_images = []
        temp_dir = Path(tempfile.mkdtemp(prefix="sportball_gpu_tuning_"))
        
        try:
            for i in range(count):
                # Create a synthetic image with some faces (rectangles)
                image = np.random.randint(0, 255, (image_size[1], image_size[0], 3), dtype=np.uint8)
                
                # Add some rectangular "faces" for detection
                for _ in range(np.random.randint(1, 4)):
                    x = np.random.randint(0, image_size[0] - 100)
                    y = np.random.randint(0, image_size[1] - 100)
                    w = np.random.randint(50, 150)
                    h = np.random.randint(50, 150)
                    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 255), -1)
                
                # Save test image
                test_path = temp_dir / f"test_image_{i:03d}.jpg"
                cv2.imwrite(str(test_path), image)
                test_images.append(test_path)
                
        except Exception as e:
            self.logger.error(f"Failed to create test images: {e}")
            # Clean up on error
            self._cleanup_test_images(test_images)
            return []
        
        return test_images
    
    def _cleanup_test_images(self, test_image_paths: List[Path]):
        """Clean up temporary test images."""
        import shutil
        
        try:
            if test_image_paths:
                # Get the temp directory from the first image
                temp_dir = test_image_paths[0].parent
                if temp_dir.name.startswith("sportball_gpu_tuning_"):
                    shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception as e:
            self.logger.warning(f"Failed to cleanup test images: {e}")
    
    def _test_batch_size(self, test_image_paths: List[Path], batch_size: int) -> bool:
        """Test if a specific batch size works without GPU memory errors."""
        try:
            import torch
            
            # Clear GPU cache before test
            torch.cuda.empty_cache()
            
            # Test with a subset of images
            test_batch = test_image_paths[:batch_size]
            
            # Try to process the batch
            results = self._process_face_batch(
                test_batch, 
                confidence=0.5, 
                min_faces=0,  # Allow 0 faces for test
                max_faces=None, 
                face_size=64
            )
            
            # Check if we got results
            success = len(results) == len(test_batch)
            
            # Clear GPU cache after test
            torch.cuda.empty_cache()
            
            return success
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                # Clear GPU cache on memory error
                try:
                    import torch
                    torch.cuda.empty_cache()
                except:
                    pass
                return False
            else:
                # Other runtime errors
                return False
        except Exception:
            # Any other error
            return False
    
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
    
    def _format_result(self, result: FaceDetectionResult, image_path: Path, image_width: int = None, image_height: int = None) -> Dict[str, Any]:
        """
        Format face detection result for JSON serialization.
        
        Args:
            result: Face detection result
            image_path: Path to the image file
            image_width: Image width in pixels (for ratio calculation)
            image_height: Image height in pixels (for ratio calculation)
            
        Returns:
            Dictionary containing formatted face detection data
        """
        # Load image dimensions if not provided
        if image_width is None or image_height is None:
            try:
                import cv2
                image = cv2.imread(str(image_path))
                if image is not None:
                    image_height, image_width = image.shape[:2]
                else:
                    # Fallback to default dimensions if image can't be loaded
                    image_width = 1920
                    image_height = 1080
            except Exception:
                # Fallback to default dimensions
                image_width = 1920
                image_height = 1080
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
            # Convert pixel coordinates to normalized ratios (0-1)
            x_pixel, y_pixel, w_pixel, h_pixel = face.bbox
            
            face_data = {
                "face_id": int(face.face_id),
                "bbox": {
                    "x": float(x_pixel) / image_width,
                    "y": float(y_pixel) / image_height,
                    "width": float(w_pixel) / image_width,
                    "height": float(h_pixel) / image_height
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
                "image_width": int(image_width),
                "image_height": int(image_height),
                "faces_found": int(result.face_count),
                "processing_time": float(result.processing_time),
                "extraction_timestamp": __import__('datetime').datetime.now().isoformat(),
                "face_size_threshold": int(getattr(self, 'face_size', 64)),
                "confidence_threshold": float(getattr(self, 'confidence_threshold', 0.5))
            }
        }