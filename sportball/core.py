"""
Sportball Core Module

This module provides the core functionality for the sportball package,
integrating all detection, analysis, and processing capabilities.

Author: Claude Sonnet 4 (claude-3-5-sonnet-20241022)
Generated via Cursor IDE (cursor.sh) with AI assistance
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from loguru import logger

from .sidecar import SidecarManager
from .decorators import (
    gpu_accelerated,
    parallel_processing,
    progress_tracked,
    cached_result,
    timing_decorator
)


class SportballCore:
    """
    Core class that provides unified access to all sportball functionality.
    
    This class serves as the main entry point for all sportball operations,
    providing a clean API for face detection, object detection, game analysis,
    and other sports photo processing tasks.
    """
    
    def __init__(self, 
                 base_dir: Optional[Path] = None,
                 enable_gpu: bool = True,
                 max_workers: Optional[int] = None,
                 cache_enabled: bool = True):
        """
        Initialize the SportballCore.
        
        Args:
            base_dir: Base directory for operations
            enable_gpu: Whether to enable GPU acceleration
            max_workers: Maximum number of parallel workers
            cache_enabled: Whether to enable result caching
        """
        self.base_dir = base_dir or Path.cwd()
        self.enable_gpu = enable_gpu
        self.max_workers = max_workers
        self.cache_enabled = cache_enabled
        
        # Initialize sidecar manager
        self.sidecar = SidecarManager(self.base_dir)
        
        # Initialize detectors (lazy loading)
        self._face_detector = None
        self._object_detector = None
        self._game_detector = None
        self._ball_detector = None
        self._quality_assessor = None
        
        self.logger = logger.bind(component="core")
        self.logger.info("Initialized SportballCore")
    
    @property
    def face_detector(self):
        """Lazy-loaded face detector."""
        if self._face_detector is None:
            from .detectors.face import FaceDetector
            self._face_detector = FaceDetector(
                enable_gpu=self.enable_gpu,
                cache_enabled=self.cache_enabled
            )
        return self._face_detector
    
    @property
    def object_detector(self):
        """Lazy-loaded object detector."""
        if self._object_detector is None:
            from .detectors.object import ObjectDetector
            self._object_detector = ObjectDetector(
                enable_gpu=self.enable_gpu,
                cache_enabled=self.cache_enabled
            )
        return self._object_detector
    
    @property
    def game_detector(self):
        """Lazy-loaded game detector."""
        if self._game_detector is None:
            from .detectors.game import GameDetector
            self._game_detector = GameDetector(
                cache_enabled=self.cache_enabled
            )
        return self._game_detector
    
    @property
    def ball_detector(self):
        """Lazy-loaded ball detector."""
        if self._ball_detector is None:
            from .detectors.ball import BallDetector
            self._ball_detector = BallDetector(
                enable_gpu=self.enable_gpu,
                cache_enabled=self.cache_enabled
            )
        return self._ball_detector
    
    @property
    def quality_assessor(self):
        """Lazy-loaded quality assessor."""
        if self._quality_assessor is None:
            from .detectors.quality import QualityAssessor
            self._quality_assessor = QualityAssessor(
                cache_enabled=self.cache_enabled
            )
        return self._quality_assessor
    
    @timing_decorator
    @gpu_accelerated(fallback_cpu=True)
    def detect_faces(self, 
                     image_paths: Union[Path, List[Path]], 
                     save_sidecar: bool = True,
                     **kwargs) -> Dict[str, Any]:
        """
        Detect faces in images.
        
        Args:
            image_paths: Single image path or list of image paths
            save_sidecar: Whether to save results to sidecar files
            **kwargs: Additional arguments for face detection
            
        Returns:
            Dictionary containing detection results
        """
        if isinstance(image_paths, Path):
            image_paths = [image_paths]
        
        self.logger.info(f"Detecting faces in {len(image_paths)} images")
        
        results = {}
        for image_path in image_paths:
            try:
                # Check cache first
                if self.cache_enabled:
                    cached_data = self.sidecar.load_data(image_path, "face_detection")
                    if cached_data:
                        results[str(image_path)] = cached_data
                        continue
                
                # Perform detection
                detection_result = self.face_detector.detect_faces(image_path, **kwargs)
                
                # Save to sidecar if requested
                if save_sidecar:
                    self.sidecar.save_data(
                        image_path, 
                        "face_detection", 
                        detection_result,
                        metadata={"kwargs": kwargs}
                    )
                
                results[str(image_path)] = detection_result
                
            except Exception as e:
                self.logger.error(f"Face detection failed for {image_path}: {e}")
                results[str(image_path)] = {"error": str(e), "success": False}
        
        return results
    
    @timing_decorator
    @gpu_accelerated(fallback_cpu=True)
    def detect_objects(self, 
                      image_paths: Union[Path, List[Path]], 
                      save_sidecar: bool = True,
                      **kwargs) -> Dict[str, Any]:
        """
        Detect objects in images.
        
        Args:
            image_paths: Single image path or list of image paths
            save_sidecar: Whether to save results to sidecar files
            **kwargs: Additional arguments for object detection
            
        Returns:
            Dictionary containing detection results
        """
        if isinstance(image_paths, Path):
            image_paths = [image_paths]
        
        self.logger.info(f"Detecting objects in {len(image_paths)} images")
        
        results = {}
        for image_path in image_paths:
            try:
                # Check cache first
                if self.cache_enabled:
                    cached_data = self.sidecar.load_data(image_path, "object_detection")
                    if cached_data:
                        results[str(image_path)] = cached_data
                        continue
                
                # Perform detection
                detection_result = self.object_detector.detect_objects(image_path, **kwargs)
                
                # Save to sidecar if requested
                if save_sidecar:
                    self.sidecar.save_data(
                        image_path, 
                        "object_detection", 
                        detection_result,
                        metadata={"kwargs": kwargs}
                    )
                
                results[str(image_path)] = detection_result
                
            except Exception as e:
                self.logger.error(f"Object detection failed for {image_path}: {e}")
                results[str(image_path)] = {"error": str(e), "success": False}
        
        return results
    
    @timing_decorator
    def detect_games(self, 
                    photo_directory: Path, 
                    pattern: str = "*_*",
                    save_sidecar: bool = True,
                    **kwargs) -> Dict[str, Any]:
        """
        Detect game boundaries in a directory of photos.
        
        Args:
            photo_directory: Directory containing photos
            pattern: File pattern to match
            save_sidecar: Whether to save results to sidecar files
            **kwargs: Additional arguments for game detection
            
        Returns:
            Dictionary containing game detection results
        """
        self.logger.info(f"Detecting games in {photo_directory} with pattern {pattern}")
        
        try:
            # Perform game detection
            detection_result = self.game_detector.detect_games(
                photo_directory, 
                pattern=pattern, 
                **kwargs
            )
            
            # Save to sidecar if requested
            if save_sidecar:
                sidecar_path = photo_directory / "game_detection.json"
                self.sidecar.save_data(
                    sidecar_path, 
                    "game_detection", 
                    detection_result,
                    metadata={"pattern": pattern, "kwargs": kwargs}
                )
            
            return detection_result
            
        except Exception as e:
            self.logger.error(f"Game detection failed: {e}")
            return {"error": str(e), "success": False}
    
    @timing_decorator
    @gpu_accelerated(fallback_cpu=True)
    def detect_balls(self, 
                    image_paths: Union[Path, List[Path]], 
                    save_sidecar: bool = True,
                    **kwargs) -> Dict[str, Any]:
        """
        Detect balls in images.
        
        Args:
            image_paths: Single image path or list of image paths
            save_sidecar: Whether to save results to sidecar files
            **kwargs: Additional arguments for ball detection
            
        Returns:
            Dictionary containing detection results
        """
        if isinstance(image_paths, Path):
            image_paths = [image_paths]
        
        self.logger.info(f"Detecting balls in {len(image_paths)} images")
        
        results = {}
        for image_path in image_paths:
            try:
                # Check cache first
                if self.cache_enabled:
                    cached_data = self.sidecar.load_data(image_path, "ball_detection")
                    if cached_data:
                        results[str(image_path)] = cached_data
                        continue
                
                # Perform detection
                detection_result = self.ball_detector.detect_balls(image_path, **kwargs)
                
                # Save to sidecar if requested
                if save_sidecar:
                    self.sidecar.save_data(
                        image_path, 
                        "ball_detection", 
                        detection_result,
                        metadata={"kwargs": kwargs}
                    )
                
                results[str(image_path)] = detection_result
                
            except Exception as e:
                self.logger.error(f"Ball detection failed for {image_path}: {e}")
                results[str(image_path)] = {"error": str(e), "success": False}
        
        return results
    
    @timing_decorator
    def assess_quality(self, 
                      image_paths: Union[Path, List[Path]], 
                      save_sidecar: bool = True,
                      **kwargs) -> Dict[str, Any]:
        """
        Assess photo quality.
        
        Args:
            image_paths: Single image path or list of image paths
            save_sidecar: Whether to save results to sidecar files
            **kwargs: Additional arguments for quality assessment
            
        Returns:
            Dictionary containing quality assessment results
        """
        if isinstance(image_paths, Path):
            image_paths = [image_paths]
        
        self.logger.info(f"Assessing quality of {len(image_paths)} images")
        
        results = {}
        for image_path in image_paths:
            try:
                # Check cache first
                if self.cache_enabled:
                    cached_data = self.sidecar.load_data(image_path, "quality_assessment")
                    if cached_data:
                        results[str(image_path)] = cached_data
                        continue
                
                # Perform assessment
                assessment_result = self.quality_assessor.assess_quality(image_path, **kwargs)
                
                # Save to sidecar if requested
                if save_sidecar:
                    self.sidecar.save_data(
                        image_path, 
                        "quality_assessment", 
                        assessment_result,
                        metadata={"kwargs": kwargs}
                    )
                
                results[str(image_path)] = assessment_result
                
            except Exception as e:
                self.logger.error(f"Quality assessment failed for {image_path}: {e}")
                results[str(image_path)] = {"error": str(e), "success": False}
        
        return results
    
    def extract_objects(self, 
                       image_paths: Union[Path, List[Path]], 
                       output_dir: Path,
                       object_types: Optional[List[str]] = None,
                       **kwargs) -> Dict[str, Any]:
        """
        Extract detected objects from images.
        
        Args:
            image_paths: Single image path or list of image paths
            output_dir: Directory to save extracted objects
            object_types: Types of objects to extract (None for all)
            **kwargs: Additional arguments for extraction
            
        Returns:
            Dictionary containing extraction results
        """
        if isinstance(image_paths, Path):
            image_paths = [image_paths]
        
        self.logger.info(f"Extracting objects from {len(image_paths)} images")
        
        # Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results = {}
        for image_path in image_paths:
            try:
                # Load detection data
                detection_data = self.sidecar.load_data(image_path, "object_detection")
                if not detection_data:
                    # Perform detection first
                    detection_data = self.detect_objects(image_path, save_sidecar=True)
                
                # Extract objects
                extraction_result = self.object_detector.extract_objects(
                    image_path, 
                    detection_data, 
                    output_dir,
                    object_types=object_types,
                    **kwargs
                )
                
                results[str(image_path)] = extraction_result
                
            except Exception as e:
                self.logger.error(f"Object extraction failed for {image_path}: {e}")
                results[str(image_path)] = {"error": str(e), "success": False}
        
        return results
    
    def get_sidecar_summary(self, directory: Optional[Path] = None) -> Dict[str, Any]:
        """
        Get a summary of sidecar files in a directory.
        
        Args:
            directory: Directory to analyze (defaults to base_dir)
            
        Returns:
            Dictionary containing sidecar summary
        """
        target_dir = directory or self.base_dir
        return self.sidecar.get_operation_summary(target_dir)
    
    def cleanup_cache(self):
        """Clean up cached data."""
        self.sidecar.clear_cache()
        self.logger.info("Cleared all cached data")
    
    def cleanup_orphaned_sidecars(self, directory: Optional[Path] = None) -> int:
        """
        Remove orphaned sidecar files.
        
        Args:
            directory: Directory to clean up (defaults to base_dir)
            
        Returns:
            Number of orphaned files removed
        """
        target_dir = directory or self.base_dir
        removed_count = self.sidecar.cleanup_orphaned_sidecars(target_dir)
        self.logger.info(f"Removed {removed_count} orphaned sidecar files")
        return removed_count
