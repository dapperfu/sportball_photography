"""
Detection Tool Adapters

Adapters to make existing detection tools compatible with the tool-agnostic framework.

Author: Claude Sonnet 4 (claude-3-5-sonnet-20241022)
Generated via Cursor IDE (cursor.sh) with AI assistance
"""

from pathlib import Path
from typing import Dict, List, Optional, Any
from loguru import logger

from .base import DetectionTool, DetectionResult, DetectionConfig


class FaceDetectionAdapter(DetectionTool):
    """
    Adapter for face detection tools to work with the tool-agnostic framework.
    """
    
    def __init__(self, config: Optional[DetectionConfig] = None):
        super().__init__(config)
        self.face_detector = None
        self.detector_type = "insightface"  # Default to InsightFace
    
    def initialize(self) -> bool:
        """Initialize the face detection adapter."""
        try:
            # Import face detection modules
            from ..detectors.face import InsightFaceDetector, FaceDetector
            
            # Create detector based on configuration
            if self.config.tool_params.get('detector_type', 'insightface') == 'insightface':
                self.face_detector = InsightFaceDetector(
                    enable_gpu=self.config.enable_gpu,
                    cache_enabled=self.config.cache_enabled,
                    confidence_threshold=self.config.confidence_threshold,
                    min_face_size=self.config.min_size,
                    batch_size=self.config.batch_size,
                    model_name=self.config.tool_params.get('model_name', 'buffalo_l'),
                    verbose=self.config.tool_params.get('verbose', False)
                )
                self.detector_type = "insightface"
            else:
                self.face_detector = FaceDetector(
                    enable_gpu=self.config.enable_gpu,
                    cache_enabled=self.config.cache_enabled,
                    confidence_threshold=self.config.confidence_threshold,
                    min_face_size=self.config.min_size,
                    batch_size=self.config.batch_size
                )
                self.detector_type = "opencv"
            
            self._initialized = True
            self.logger.info(f"Initialized face detection adapter with {self.detector_type}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize face detection adapter: {e}")
            return False
    
    def detect(self, image_path: Path, **kwargs) -> DetectionResult:
        """Detect faces in a single image."""
        if not self._ensure_initialized():
            return DetectionResult(
                success=False,
                error="Face detection adapter not initialized",
                tool_name=self.config.tool_name
            )
        
        try:
            # Perform face detection
            result = self.face_detector.detect_faces(
                image_path,
                confidence=kwargs.get('confidence', self.config.confidence_threshold),
                min_faces=kwargs.get('min_faces', 1),
                max_faces=kwargs.get('max_faces', self.config.max_detections),
                face_size=kwargs.get('face_size', self.config.min_size)
            )
            
            # Convert to generic DetectionResult
            if result.success:
                detection_data = {
                    'faces': [face.as_dict() for face in result.faces],
                    'face_count': result.face_count,
                    'detector_type': self.detector_type
                }
                
                return DetectionResult(
                    success=True,
                    data=detection_data,
                    tool_name=self.config.tool_name,
                    processing_time=result.processing_time,
                    image_path=image_path
                )
            else:
                return DetectionResult(
                    success=False,
                    error=result.error,
                    tool_name=self.config.tool_name,
                    processing_time=result.processing_time,
                    image_path=image_path
                )
                
        except Exception as e:
            self.logger.error(f"Face detection failed: {e}")
            return DetectionResult(
                success=False,
                error=str(e),
                tool_name=self.config.tool_name,
                image_path=image_path
            )
    
    def detect_batch(self, image_paths: List[Path], **kwargs) -> Dict[str, DetectionResult]:
        """Detect faces in multiple images."""
        if not self._ensure_initialized():
            return {
                str(path): DetectionResult(
                    success=False,
                    error="Face detection adapter not initialized",
                    tool_name=self.config.tool_name
                ) for path in image_paths
            }
        
        try:
            # Perform batch face detection
            results = self.face_detector.detect_faces_batch(
                image_paths,
                confidence=kwargs.get('confidence', self.config.confidence_threshold),
                min_faces=kwargs.get('min_faces', 1),
                max_faces=kwargs.get('max_faces', self.config.max_detections),
                face_size=kwargs.get('face_size', self.config.min_size)
            )
            
            # Convert to generic DetectionResult format
            converted_results = {}
            for image_path, result in results.items():
                if result.success:
                    detection_data = {
                        'faces': [face.as_dict() for face in result.faces],
                        'face_count': result.face_count,
                        'detector_type': self.detector_type
                    }
                    
                    converted_results[image_path] = DetectionResult(
                        success=True,
                        data=detection_data,
                        tool_name=self.config.tool_name,
                        processing_time=result.processing_time,
                        image_path=Path(image_path)
                    )
                else:
                    converted_results[image_path] = DetectionResult(
                        success=False,
                        error=result.error,
                        tool_name=self.config.tool_name,
                        processing_time=result.processing_time,
                        image_path=Path(image_path)
                    )
            
            return converted_results
            
        except Exception as e:
            self.logger.error(f"Batch face detection failed: {e}")
            return {
                str(path): DetectionResult(
                    success=False,
                    error=str(e),
                    tool_name=self.config.tool_name,
                    image_path=path
                ) for path in image_paths
            }


class ObjectDetectionAdapter(DetectionTool):
    """
    Adapter for object detection tools to work with the tool-agnostic framework.
    """
    
    def __init__(self, config: Optional[DetectionConfig] = None):
        super().__init__(config)
        self.object_detector = None
    
    def initialize(self) -> bool:
        """Initialize the object detection adapter."""
        try:
            # Import object detection module
            from ..detectors.object import ObjectDetector
            
            # Create detector
            self.object_detector = ObjectDetector(
                model_path=self.config.tool_params.get('model_path', 'yolov8n.pt'),
                border_padding=self.config.tool_params.get('border_padding', 0.25),
                enable_gpu=self.config.enable_gpu,
                target_objects=self.config.tool_params.get('target_objects'),
                confidence_threshold=self.config.confidence_threshold,
                cache_enabled=self.config.cache_enabled,
                gpu_batch_size=self.config.batch_size
            )
            
            self._initialized = True
            self.logger.info("Initialized object detection adapter")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize object detection adapter: {e}")
            return False
    
    def detect(self, image_path: Path, **kwargs) -> DetectionResult:
        """Detect objects in a single image."""
        if not self._ensure_initialized():
            return DetectionResult(
                success=False,
                error="Object detection adapter not initialized",
                tool_name=self.config.tool_name
            )
        
        try:
            # Perform object detection
            results = self.object_detector.detect_objects(
                [image_path],
                save_sidecar=False,  # We'll handle sidecar saving separately
                force=kwargs.get('force', self.config.force_reprocess),
                **kwargs
            )
            
            # Extract result for the single image
            if str(image_path) in results:
                result_data = results[str(image_path)]
                
                if result_data.get('success', False):
                    detection_data = {
                        'objects': result_data.get('objects', []),
                        'object_count': result_data.get('objects_found', 0),
                        'detector_type': 'yolov8'
                    }
                    
                    return DetectionResult(
                        success=True,
                        data=detection_data,
                        tool_name=self.config.tool_name,
                        processing_time=result_data.get('detection_time', 0.0),
                        image_path=image_path
                    )
                else:
                    return DetectionResult(
                        success=False,
                        error=result_data.get('error', 'Unknown error'),
                        tool_name=self.config.tool_name,
                        processing_time=result_data.get('detection_time', 0.0),
                        image_path=image_path
                    )
            else:
                return DetectionResult(
                    success=False,
                    error="No detection result found",
                    tool_name=self.config.tool_name,
                    image_path=image_path
                )
                
        except Exception as e:
            self.logger.error(f"Object detection failed: {e}")
            return DetectionResult(
                success=False,
                error=str(e),
                tool_name=self.config.tool_name,
                image_path=image_path
            )
    
    def detect_batch(self, image_paths: List[Path], **kwargs) -> Dict[str, DetectionResult]:
        """Detect objects in multiple images."""
        if not self._ensure_initialized():
            return {
                str(path): DetectionResult(
                    success=False,
                    error="Object detection adapter not initialized",
                    tool_name=self.config.tool_name
                ) for path in image_paths
            }
        
        try:
            # Perform batch object detection
            results = self.object_detector.detect_objects(
                image_paths,
                save_sidecar=False,  # We'll handle sidecar saving separately
                force=kwargs.get('force', self.config.force_reprocess),
                **kwargs
            )
            
            # Convert to generic DetectionResult format
            converted_results = {}
            for image_path, result_data in results.items():
                if result_data.get('success', False):
                    detection_data = {
                        'objects': result_data.get('objects', []),
                        'object_count': result_data.get('objects_found', 0),
                        'detector_type': 'yolov8'
                    }
                    
                    converted_results[image_path] = DetectionResult(
                        success=True,
                        data=detection_data,
                        tool_name=self.config.tool_name,
                        processing_time=result_data.get('detection_time', 0.0),
                        image_path=Path(image_path)
                    )
                else:
                    converted_results[image_path] = DetectionResult(
                        success=False,
                        error=result_data.get('error', 'Unknown error'),
                        tool_name=self.config.tool_name,
                        processing_time=result_data.get('detection_time', 0.0),
                        image_path=Path(image_path)
                    )
            
            return converted_results
            
        except Exception as e:
            self.logger.error(f"Batch object detection failed: {e}")
            return {
                str(path): DetectionResult(
                    success=False,
                    error=str(e),
                    tool_name=self.config.tool_name,
                    image_path=path
                ) for path in image_paths
            }


class QualityAssessmentAdapter(DetectionTool):
    """
    Adapter for quality assessment tools to work with the tool-agnostic framework.
    """
    
    def __init__(self, config: Optional[DetectionConfig] = None):
        super().__init__(config)
        self.quality_assessor = None
    
    def initialize(self) -> bool:
        """Initialize the quality assessment adapter."""
        try:
            # Import quality assessment module
            from ..detectors.quality import QualityAssessor
            
            # Create assessor
            self.quality_assessor = QualityAssessor(
                cache_enabled=self.config.cache_enabled
            )
            
            self._initialized = True
            self.logger.info("Initialized quality assessment adapter")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize quality assessment adapter: {e}")
            return False
    
    def detect(self, image_path: Path, **kwargs) -> DetectionResult:
        """Assess quality of a single image."""
        if not self._ensure_initialized():
            return DetectionResult(
                success=False,
                error="Quality assessment adapter not initialized",
                tool_name=self.config.tool_name
            )
        
        try:
            # Perform quality assessment
            result = self.quality_assessor.assess_quality(image_path, **kwargs)
            
            if result.get('success', False):
                detection_data = {
                    'quality_score': result.get('quality_score', 0.0),
                    'quality_metrics': result.get('metrics', {}),
                    'assessor_type': 'quality'
                }
                
                return DetectionResult(
                    success=True,
                    data=detection_data,
                    tool_name=self.config.tool_name,
                    processing_time=result.get('processing_time', 0.0),
                    image_path=image_path
                )
            else:
                return DetectionResult(
                    success=False,
                    error=result.get('error', 'Unknown error'),
                    tool_name=self.config.tool_name,
                    processing_time=result.get('processing_time', 0.0),
                    image_path=image_path
                )
                
        except Exception as e:
            self.logger.error(f"Quality assessment failed: {e}")
            return DetectionResult(
                success=False,
                error=str(e),
                tool_name=self.config.tool_name,
                image_path=image_path
            )
    
    def detect_batch(self, image_paths: List[Path], **kwargs) -> Dict[str, DetectionResult]:
        """Assess quality of multiple images."""
        if not self._ensure_initialized():
            return {
                str(path): DetectionResult(
                    success=False,
                    error="Quality assessment adapter not initialized",
                    tool_name=self.config.tool_name
                ) for path in image_paths
            }
        
        try:
            # Perform batch quality assessment
            results = self.quality_assessor.assess_quality_batch(image_paths, **kwargs)
            
            # Convert to generic DetectionResult format
            converted_results = {}
            for image_path, result in results.items():
                if result.get('success', False):
                    detection_data = {
                        'quality_score': result.get('quality_score', 0.0),
                        'quality_metrics': result.get('metrics', {}),
                        'assessor_type': 'quality'
                    }
                    
                    converted_results[image_path] = DetectionResult(
                        success=True,
                        data=detection_data,
                        tool_name=self.config.tool_name,
                        processing_time=result.get('processing_time', 0.0),
                        image_path=Path(image_path)
                    )
                else:
                    converted_results[image_path] = DetectionResult(
                        success=False,
                        error=result.get('error', 'Unknown error'),
                        tool_name=self.config.tool_name,
                        processing_time=result.get('processing_time', 0.0),
                        image_path=Path(image_path)
                    )
            
            return converted_results
            
        except Exception as e:
            self.logger.error(f"Batch quality assessment failed: {e}")
            return {
                str(path): DetectionResult(
                    success=False,
                    error=str(e),
                    tool_name=self.config.tool_name,
                    image_path=path
                ) for path in image_paths
            }
