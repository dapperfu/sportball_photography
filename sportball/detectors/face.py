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

try:
    import insightface
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False
    logger.warning("insightface not available - InsightFace detection will be skipped")

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
        
        # Initialize logger first
        self.logger = logger.bind(component="face_detector")
        
        # Initialize face detection models
        self.device = "cpu"
        self.gpu_model = None
        
        # Try to initialize GPU-based face detection first
        if self.enable_gpu:
            try:
                import torch
                if torch.cuda.is_available():
                    self.device = "cuda"
                    self._initialize_gpu_models()
                    self.logger.info("GPU-based face detection initialized")
                else:
                    self.logger.warning("CUDA not available, falling back to CPU")
                    self._initialize_cpu_models()
            except ImportError:
                self.logger.warning("PyTorch not available, falling back to CPU")
                self._initialize_cpu_models()
        else:
            self._initialize_cpu_models()
        
        # Initialize face recognition if available
        self.face_recognition_available = FACE_RECOGNITION_AVAILABLE
    
    def _initialize_gpu_models(self):
        """Initialize GPU-based face detection models."""
        if FACE_RECOGNITION_AVAILABLE:
            # Use face_recognition library with CUDA-enabled dlib
            self.gpu_model = "face_recognition"
            self.logger.info("face_recognition model loaded with CUDA support")
            
            # Also initialize CPU models as fallback
            self._initialize_cpu_models()
            return
        
        # If face_recognition not available, fall back to CPU
        self.logger.warning("face_recognition not available, falling back to CPU")
        self.device = "cpu"
        self._initialize_cpu_models()
    
    def _initialize_cpu_models(self):
        """Initialize CPU-based face detection models using face_recognition."""
        self.device = "cpu"
        if FACE_RECOGNITION_AVAILABLE:
            self.gpu_model = "face_recognition"
            self.logger.info("face_recognition model loaded for CPU processing")
        else:
            raise ImportError("face_recognition library is required but not available. Please install it with: pip install face_recognition")
    
    def _detect_faces_gpu(self, image, confidence: Optional[float], face_size: int):
        """Detect faces using GPU-based face_recognition library."""
        try:
            import face_recognition
            import numpy as np
            
            # Convert BGR to RGB for face_recognition
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Use face_recognition library for detection
            # Try CNN model first (GPU-accelerated), fall back to HOG if needed
            try:
                face_locations = face_recognition.face_locations(
                    image_rgb,
                    model="cnn"  # CNN model uses GPU acceleration
                )
            except Exception:
                # Fall back to HOG model if CNN fails
                face_locations = face_recognition.face_locations(
                    image_rgb,
                    model="hog"
                )
            
            faces = []
            for (top, right, bottom, left) in face_locations:
                # Convert face_recognition format (top, right, bottom, left) to (x, y, w, h)
                x, y, w, h = left, top, right - left, bottom - top
                
                if w >= face_size and h >= face_size:
                    # face_recognition doesn't provide confidence scores directly
                    # We'll use a default high confidence for detected faces
                    face_confidence = 0.9  # face_recognition is quite accurate
                    
                    if confidence is None or face_confidence >= confidence:
                        faces.append((x, y, w, h))
            
            return faces
                
        except Exception as e:
            self.logger.error(f"GPU face detection failed: {e}")
            return self._detect_faces_cpu(image, face_size)
    
    def _detect_faces_cpu(self, image, face_size: int):
        """Detect faces using CPU-based face_recognition library."""
        try:
            import face_recognition
            
            # Convert BGR to RGB for face_recognition
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Detect face locations
            face_locations = face_recognition.face_locations(rgb_image, model="hog")
            
            # Convert to the format expected by the rest of the code
            faces = []
            for (top, right, bottom, left) in face_locations:
                # Convert to (x, y, w, h) format
                x, y, w, h = left, top, right - left, bottom - top
                
                # Check minimum face size
                if w >= face_size and h >= face_size:
                    faces.append((x, y, w, h))
            
            return faces
        except Exception as e:
            self.logger.error(f"CPU face detection failed: {e}")
            return []
    
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
            
            # Use GPU model if available, otherwise fall back to CPU
            if self.gpu_model is not None and self.device == "cuda":
                faces = self._detect_faces_gpu(image, confidence, face_size)
            else:
                faces = self._detect_faces_cpu(image, face_size)
            
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
        
        # Use GPU batch processing if available
        if self.device == "cuda" and self.gpu_model == "face_recognition":
            self.logger.info("Using GPU batch processing")
            return self._process_gpu_batch(batch_images, batch_metadata, confidence, min_faces, max_faces, face_size, start_time)
        else:
            self.logger.info(f"Using CPU batch processing (device={self.device}, gpu_model={self.gpu_model})")
            # Fall back to CPU processing
            return self._process_cpu_batch(batch_images, batch_metadata, confidence, min_faces, max_faces, face_size, start_time)
    
    def _process_gpu_batch(self, batch_images, batch_metadata, confidence, min_faces, max_faces, face_size, start_time):
        """Process batch using GPU-accelerated face_recognition library with true batch processing."""
        import time
        results = {}
        
        try:
            import face_recognition
            import numpy as np
            
            # Monitor GPU memory usage
            self._log_gpu_memory("Before GPU batch processing")
            
            # Convert all images to RGB for face_recognition
            batch_images_rgb = []
            for image in batch_images:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                batch_images_rgb.append(image_rgb)
            
            self._log_gpu_memory("After loading images to GPU memory")
            
            # Use true batch processing with face_recognition
            # This will load multiple images into GPU memory and process them together
            batch_face_locations = face_recognition.batch_face_locations(
                batch_images_rgb,
                number_of_times_to_upsample=1,
                batch_size=len(batch_images_rgb)  # Process all images in one batch
            )
            
            self._log_gpu_memory("After batch face detection")
            
            # Process results for each image
            for i, (face_locations, metadata) in enumerate(zip(batch_face_locations, batch_metadata)):
                try:
                    # Convert face_recognition format to our format
                    detected_faces = []
                    for j, (top, right, bottom, left) in enumerate(face_locations):
                        x, y, w, h = left, top, right - left, bottom - top
                        
                        if w >= face_size and h >= face_size:
                            face_confidence = 0.9  # face_recognition doesn't provide confidence
                            
                            if confidence is None or face_confidence >= confidence:
                                detected_face = DetectedFace(
                                    face_id=j,
                                    bbox=(x, y, w, h),
                                    confidence=face_confidence
                                )
                                
                                # Add face encoding
                                try:
                                    face_encoding = face_recognition.face_encodings(
                                        batch_images_rgb[i], 
                                        [(top, right, bottom, left)]
                                    )[0]
                                    detected_face.encoding = face_encoding
                                except Exception as e:
                                    self.logger.warning(f"Failed to encode face: {e}")
                                
                                detected_faces.append(detected_face)
                    
                    # Apply face count limits
                    if max_faces and len(detected_faces) > max_faces:
                        detected_faces = detected_faces[:max_faces]
                    
                    # Check minimum face count
                    success = len(detected_faces) >= min_faces
                    
                    results[str(metadata['path'])] = FaceDetectionResult(
                        faces=detected_faces,
                        face_count=len(detected_faces),
                        success=success,
                        processing_time=(time.time() - start_time) / len(batch_images),
                        error=None if success else f"Found {len(detected_faces)} faces, need at least {min_faces}"
                    )
                    
                except Exception as e:
                    results[str(metadata['path'])] = FaceDetectionResult(
                        faces=[],
                        face_count=0,
                        success=False,
                        processing_time=0.0,
                        error=str(e)
                    )
            
        except Exception as e:
            self.logger.error(f"GPU batch processing failed: {e}")
            # Fall back to CPU processing
            return self._process_cpu_batch(batch_images, batch_metadata, confidence, min_faces, max_faces, face_size, start_time)
        
        self._log_gpu_memory("After GPU batch processing")
        return results
    
    def _process_cpu_batch(self, batch_images, batch_metadata, confidence, min_faces, max_faces, face_size, start_time):
        """Process batch using CPU-based face_recognition."""
        import time
        import face_recognition
        results = {}
        
        # Process each image in the batch
        batch_processing_time = time.time() - start_time
        
        for i, (image, metadata) in enumerate(zip(batch_images, batch_metadata)):
            try:
                # Convert BGR to RGB for face_recognition
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Detect face locations
                face_locations = face_recognition.face_locations(rgb_image, model="hog")
                
                # Filter faces by confidence and size
                detected_faces = []
                for j, (top, right, bottom, left) in enumerate(face_locations):
                    # Convert to (x, y, w, h) format
                    x, y, w, h = left, top, right - left, bottom - top
                    
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
    
    def _log_gpu_memory(self, context: str):
        """Log current GPU memory usage."""
        try:
            import torch
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**2  # MB
                reserved = torch.cuda.memory_reserved() / 1024**2     # MB
                self.logger.info(f"GPU Memory [{context}]: {allocated:.1f}MB allocated, {reserved:.1f}MB reserved")
            else:
                self.logger.info(f"GPU Memory [{context}]: CUDA not available")
        except Exception as e:
            self.logger.warning(f"Failed to get GPU memory info [{context}]: {e}")


class InsightFaceDetector:
    """
    Face detection using InsightFace library for high-performance face detection and recognition.
    """
    
    def __init__(self, 
                 enable_gpu: bool = True,
                 cache_enabled: bool = True,
                 confidence_threshold: float = 0.5,
                 min_face_size: int = 64,
                 batch_size: int = 8,
                 model_name: str = "buffalo_l"):
        """
        Initialize InsightFace detector.
        
        Args:
            enable_gpu: Whether to enable GPU acceleration
            cache_enabled: Whether to enable result caching
            confidence_threshold: Minimum confidence for face detection
            min_face_size: Minimum face size in pixels
            batch_size: Batch size for processing multiple images
            model_name: InsightFace model name (buffalo_l, buffalo_m, buffalo_s)
        """
        self.enable_gpu = enable_gpu
        self.cache_enabled = cache_enabled
        self.confidence_threshold = confidence_threshold
        self.min_face_size = min_face_size
        self.batch_size = batch_size
        self.model_name = model_name
        
        # Initialize logger
        self.logger = logger.bind(component="insightface_detector")
        
        # Initialize InsightFace model
        self.app = None
        self.device = "cpu"
        
        if not INSIGHTFACE_AVAILABLE:
            self.logger.error("InsightFace not available - install with: pip install insightface")
            return
        
        try:
            # Set device
            if self.enable_gpu:
                try:
                    import torch
                    if torch.cuda.is_available():
                        self.device = "cuda:0"
                        self.logger.info("Using GPU for InsightFace")
                    else:
                        self.device = "cpu"
                        self.logger.warning("CUDA not available, using CPU for InsightFace")
                except ImportError:
                    self.device = "cpu"
                    self.logger.warning("PyTorch not available, using CPU for InsightFace")
            else:
                self.device = "cpu"
                self.logger.info("Using CPU for InsightFace")
            
            # Initialize InsightFace app
            self.app = insightface.app.FaceAnalysis(
                name=self.model_name,
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider'] if self.device.startswith('cuda') else ['CPUExecutionProvider']
            )
            self.app.prepare(ctx_id=0 if self.device == "cpu" else 0, det_size=(640, 640))
            
            self.logger.info(f"InsightFace initialized with model: {self.model_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize InsightFace: {e}")
            self.app = None
    
    @gpu_accelerated(fallback_cpu=True)
    @cached_result(expire_seconds=3600)  # Cache for 1 hour
    def detect_faces(self, 
                    image_path: Path, 
                    confidence: Optional[float] = None,
                    min_faces: int = 1,
                    max_faces: Optional[int] = None,
                    face_size: int = 64) -> FaceDetectionResult:
        """
        Detect faces in an image using InsightFace.
        
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
            if self.app is None:
                return FaceDetectionResult(
                    faces=[],
                    face_count=0,
                    success=False,
                    processing_time=0,
                    error="InsightFace not initialized"
                )
            
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
            
            # Detect faces using InsightFace
            faces = self.app.get(image)
            
            # Filter faces by confidence and size
            detected_faces = []
            for i, face in enumerate(faces):
                # Get bounding box
                bbox = face.bbox.astype(int)
                x, y, x2, y2 = bbox
                w, h = x2 - x, y2 - y
                
                # Check minimum face size
                if w >= face_size and h >= face_size:
                    # Get confidence score
                    face_confidence = float(face.det_score)
                    
                    # Apply confidence threshold
                    conf_threshold = confidence if confidence is not None else self.confidence_threshold
                    if face_confidence >= conf_threshold:
                        detected_face = DetectedFace(
                            face_id=i,
                            bbox=(x, y, w, h),
                            confidence=face_confidence,
                            landmarks=face.kps.tolist() if hasattr(face, 'kps') else None,
                            encoding=face.embedding.tolist() if hasattr(face, 'embedding') else None
                        )
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
            self.logger.error(f"InsightFace detection failed: {e}")
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
        Detect faces in multiple images using parallel GPU batch processing.
        
        Args:
            image_paths: List of image paths
            confidence: Detection confidence threshold
            min_faces: Minimum number of faces to detect
            max_faces: Maximum number of faces to detect
            face_size: Minimum face size in pixels
            
        Returns:
            Dictionary mapping image paths to detection results
        """
        self.logger.info(f"Processing {len(image_paths)} images with InsightFace parallel GPU processing")
        
        if self.device.startswith('cuda') and self.app is not None:
            return self._process_parallel_gpu_batch(image_paths, confidence, min_faces, max_faces, face_size)
        else:
            # Fall back to individual processing
            return self._process_cpu_batch(image_paths, confidence, min_faces, max_faces, face_size)
    
    def _process_parallel_gpu_batch(self, 
                                   image_paths: List[Path], 
                                   confidence: Optional[float],
                                   min_faces: int,
                                   max_faces: Optional[int],
                                   face_size: int) -> Dict[str, FaceDetectionResult]:
        """Process batch using parallel GPU processing with multiple workers."""
        import time
        import cv2
        import numpy as np
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        start_time = time.time()
        results = {}
        
        # Split images into smaller batches for parallel processing
        # Each worker will process a batch of images
        batch_size = min(32, len(image_paths))  # Process 32 images per batch
        num_workers = min(4, len(image_paths) // batch_size + 1)  # Up to 4 parallel workers
        
        self.logger.info(f"Processing {len(image_paths)} images with {num_workers} parallel workers, batch size {batch_size}")
        
        def process_batch_worker(batch_paths, worker_id):
            """Process a batch of images in a separate thread."""
            worker_results = {}
            
            try:
                # Create a new InsightFace app instance for this worker
                worker_app = insightface.app.FaceAnalysis(
                    name=self.model_name,
                    providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
                )
                worker_app.prepare(ctx_id=0, det_size=(640, 640))
                
                self.logger.info(f"Worker {worker_id}: Processing {len(batch_paths)} images")
                
                # Load all images in this batch
                batch_images = []
                batch_metadata = []
                
                for img_path in batch_paths:
                    try:
                        image = cv2.imread(str(img_path))
                        if image is not None:
                            batch_images.append(image)
                            batch_metadata.append({
                                'path': img_path,
                                'height': image.shape[0],
                                'width': image.shape[1]
                            })
                        else:
                            worker_results[str(img_path)] = FaceDetectionResult(
                                faces=[], face_count=0, success=False, processing_time=0.0,
                                error="Failed to load image"
                            )
                    except Exception as e:
                        worker_results[str(img_path)] = FaceDetectionResult(
                            faces=[], face_count=0, success=False, processing_time=0.0,
                            error=str(e)
                        )
                
                if not batch_images:
                    return worker_results
                
                # Process each image in the batch
                for img, metadata in zip(batch_images, batch_metadata):
                    try:
                        faces = worker_app.get(img)
                        
                        # Filter faces by confidence and size
                        detected_faces = []
                        for j, face in enumerate(faces):
                            # Get bounding box
                            bbox = face.bbox.astype(int)
                            x, y, x2, y2 = bbox
                            w, h = x2 - x, y2 - y
                            
                            # Check minimum face size
                            if w >= face_size and h >= face_size:
                                # Get confidence score
                                face_confidence = float(face.det_score)
                                
                                # Apply confidence threshold
                                conf_threshold = confidence if confidence is not None else self.confidence_threshold
                                if face_confidence >= conf_threshold:
                                    detected_face = DetectedFace(
                                        face_id=j,
                                        bbox=(x, y, w, h),
                                        confidence=face_confidence,
                                        landmarks=face.kps.tolist() if hasattr(face, 'kps') else None,
                                        encoding=face.embedding.tolist() if hasattr(face, 'embedding') else None
                                    )
                                    detected_faces.append(detected_face)
                        
                        # Apply min/max face constraints
                        success = len(detected_faces) >= min_faces
                        if max_faces and len(detected_faces) > max_faces:
                            detected_faces.sort(key=lambda f: f.confidence, reverse=True)
                            detected_faces = detected_faces[:max_faces]
                        
                        worker_results[str(metadata['path'])] = FaceDetectionResult(
                            faces=detected_faces,
                            face_count=len(detected_faces),
                            success=success,
                            processing_time=0.0,  # Will be calculated at the end
                            error=None if success else f"Found {len(detected_faces)} faces, need at least {min_faces}"
                        )
                        
                    except Exception as e:
                        worker_results[str(metadata['path'])] = FaceDetectionResult(
                            faces=[], face_count=0, success=False, processing_time=0.0, error=str(e)
                        )
                
                self.logger.info(f"Worker {worker_id}: Completed {len(batch_paths)} images")
                
            except Exception as e:
                self.logger.error(f"Worker {worker_id} failed: {e}")
                # Mark all images in this batch as failed
                for img_path in batch_paths:
                    worker_results[str(img_path)] = FaceDetectionResult(
                        faces=[], face_count=0, success=False, processing_time=0.0, error=str(e)
                    )
            
            return worker_results
        
        # Split images into batches for parallel processing
        batches = []
        for i in range(0, len(image_paths), batch_size):
            batch = image_paths[i:i + batch_size]
            batches.append(batch)
        
        # Process batches in parallel
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Submit all batches
            future_to_batch = {
                executor.submit(process_batch_worker, batch, i): batch 
                for i, batch in enumerate(batches)
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_batch):
                try:
                    batch_results = future.result()
                    results.update(batch_results)
                except Exception as e:
                    self.logger.error(f"Batch processing failed: {e}")
                    # Mark all images in this batch as failed
                    batch = future_to_batch[future]
                    for img_path in batch:
                        results[str(img_path)] = FaceDetectionResult(
                            faces=[], face_count=0, success=False, processing_time=0.0, error=str(e)
                        )
        
        # Calculate processing times
        total_time = time.time() - start_time
        avg_time_per_image = total_time / len(image_paths)
        
        # Update processing times in results
        for result in results.values():
            result.processing_time = avg_time_per_image
        
        self.logger.info(f"Parallel GPU processing completed: {len(image_paths)} images in {total_time:.2f}s ({avg_time_per_image:.3f}s per image)")
        
        return results
    
    def _process_gpu_batch(self, 
                          image_paths: List[Path], 
                          confidence: Optional[float],
                          min_faces: int,
                          max_faces: Optional[int],
                          face_size: int) -> Dict[str, FaceDetectionResult]:
        """Process batch using true GPU batch processing with InsightFace."""
        import time
        import cv2
        import numpy as np
        
        start_time = time.time()
        results = {}
        
        try:
            # Load all images into GPU memory at once
            batch_images = []
            batch_metadata = []
            
            for image_path in image_paths:
                try:
                    image = cv2.imread(str(image_path))
                    if image is not None:
                        batch_images.append(image)
                        batch_metadata.append({
                            'path': image_path,
                            'height': image.shape[0],
                            'width': image.shape[1]
                        })
                    else:
                        results[str(image_path)] = FaceDetectionResult(
                            faces=[], face_count=0, success=False, processing_time=0.0,
                            error="Failed to load image"
                        )
                except Exception as e:
                    results[str(image_path)] = FaceDetectionResult(
                        faces=[], face_count=0, success=False, processing_time=0.0,
                        error=str(e)
                    )
            
            if not batch_images:
                return results
            
            # Process all images in parallel using GPU
            self.logger.info(f"Processing {len(batch_images)} images simultaneously on GPU")
            
            # Use InsightFace's batch processing capability
            batch_faces = self.app.get(batch_images)
            
            # Process results
            for i, (image, metadata) in enumerate(zip(batch_images, batch_metadata)):
                try:
                    if i < len(batch_faces):
                        faces = batch_faces[i] if isinstance(batch_faces[i], list) else [batch_faces[i]]
                    else:
                        faces = []
                    
                    # Filter faces by confidence and size
                    detected_faces = []
                    for j, face in enumerate(faces):
                        # Get bounding box
                        bbox = face.bbox.astype(int)
                        x, y, x2, y2 = bbox
                        w, h = x2 - x, y2 - y
                        
                        # Check minimum face size
                        if w >= face_size and h >= face_size:
                            # Get confidence score
                            face_confidence = float(face.det_score)
                            
                            # Apply confidence threshold
                            conf_threshold = confidence if confidence is not None else self.confidence_threshold
                            if face_confidence >= conf_threshold:
                                detected_face = DetectedFace(
                                    face_id=j,
                                    bbox=(x, y, w, h),
                                    confidence=face_confidence,
                                    landmarks=face.kps.tolist() if hasattr(face, 'kps') else None,
                                    encoding=face.embedding.tolist() if hasattr(face, 'embedding') else None
                                )
                                detected_faces.append(detected_face)
                    
                    # Apply min/max face constraints
                    success = len(detected_faces) >= min_faces
                    if max_faces and len(detected_faces) > max_faces:
                        detected_faces.sort(key=lambda f: f.confidence, reverse=True)
                        detected_faces = detected_faces[:max_faces]
                    
                    processing_time = (time.time() - start_time) / len(batch_images)
                    
                    results[str(metadata['path'])] = FaceDetectionResult(
                        faces=detected_faces,
                        face_count=len(detected_faces),
                        success=success,
                        processing_time=processing_time,
                        error=None if success else f"Found {len(detected_faces)} faces, need at least {min_faces}"
                    )
                    
                except Exception as e:
                    results[str(metadata['path'])] = FaceDetectionResult(
                        faces=[], face_count=0, success=False, processing_time=0.0, error=str(e)
                    )
            
            total_time = time.time() - start_time
            self.logger.info(f"GPU batch processing completed: {len(batch_images)} images in {total_time:.2f}s ({total_time/len(batch_images):.3f}s per image)")
            
        except Exception as e:
            self.logger.error(f"GPU batch processing failed: {e}")
            # Fall back to individual processing
            return self._process_cpu_batch(image_paths, confidence, min_faces, max_faces, face_size)
        
        return results
    
    def _process_cpu_batch(self, 
                          image_paths: List[Path], 
                          confidence: Optional[float],
                          min_faces: int,
                          max_faces: Optional[int],
                          face_size: int) -> Dict[str, FaceDetectionResult]:
        """Process batch using individual CPU processing."""
        results = {}
        
        for i, image_path in enumerate(image_paths):
            try:
                result = self.detect_faces(image_path, confidence, min_faces, max_faces, face_size)
                results[str(image_path)] = result
            except Exception as e:
                self.logger.error(f"Error processing {image_path}: {e}")
                results[str(image_path)] = FaceDetectionResult(
                    faces=[],
                    face_count=0,
                    success=False,
                    processing_time=0.0,
                    error=str(e)
                )
        
        return results
    
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
                face_filename = f"{image_path.stem}_insightface_face_{i:02d}.jpg"
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
            self.logger.error(f"InsightFace extraction failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "faces_extracted": 0
            }
    
    def _format_result(self, result: FaceDetectionResult, image_path: Path, image_width: int = None, image_height: int = None, face_size: int = 64) -> Dict[str, Any]:
        """
        Format face detection result for JSON serialization.
        
        Args:
            result: Face detection result
            image_path: Path to the image file
            image_width: Image width in pixels (for ratio calculation)
            image_height: Image height in pixels (for ratio calculation)
            face_size: Minimum face size threshold used
            
        Returns:
            Dictionary containing formatted face detection data
        """
        # Load image dimensions if not provided
        if image_width is None or image_height is None:
            try:
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
                    "extraction_timestamp": __import__('datetime').datetime.now().isoformat(),
                    "detector": "insightface"
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
            
            # Add landmarks if available
            if face.landmarks is not None:
                face_data["landmarks"] = face.landmarks
            
            # Add encoding if available
            if face.encoding is not None:
                face_data["encoding"] = face.encoding
            
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
                "detector": "insightface",
                "model_name": self.model_name,
                "face_size_threshold": int(face_size),
                "confidence_threshold": float(self.confidence_threshold)
            }
        }