"""
Sportball Core Module

This module provides the core functionality for the sportball package,
integrating all detection, analysis, and processing capabilities.

Author: Claude Sonnet 4 (claude-3-5-sonnet-20241022)
Generated via Cursor IDE (cursor.sh) with AI assistance
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from loguru import logger

from .sidecar import SidecarManager
from .decorators import (
    gpu_accelerated,
    timing_decorator
)
from .detection.integration import DetectionIntegration


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
                 cache_enabled: bool = True,
                 verbose: bool = False):
        """
        Initialize the SportballCore.
        
        Args:
            base_dir: Base directory for operations
            enable_gpu: Whether to enable GPU acceleration
            max_workers: Maximum number of parallel workers
            cache_enabled: Whether to enable result caching
            verbose: Whether to show verbose output
        """
        self.base_dir = base_dir or Path.cwd()
        self.enable_gpu = enable_gpu
        self.max_workers = max_workers
        self.cache_enabled = cache_enabled
        self.verbose = verbose
        
        # Initialize sidecar manager
        self.sidecar = SidecarManager(self.base_dir)
        
        # Initialize tool-agnostic detection integration
        self.detection = DetectionIntegration(
            base_dir=self.base_dir,
            enable_rust=True,
            max_workers=self.max_workers
        )
        
        # Initialize detectors (lazy loading)
        self._face_detector = None
        self._object_detector = None
        self._game_detector = None
        self._quality_assessor = None
        self._face_clustering = None
        
        self.logger = logger.bind(component="core")
        self.logger.info("Initialized SportballCore")
    
    @property
    def face_detector(self):
        """Lazy-loaded face detector (defaults to InsightFace)."""
        if self._face_detector is None:
            from .detectors.face import InsightFaceDetector
            self._face_detector = InsightFaceDetector(
                enable_gpu=self.enable_gpu,
                cache_enabled=self.cache_enabled,
                verbose=self.verbose
            )
        return self._face_detector
    
    def get_face_detector(self, batch_size: int = 8):
        """Get face detector with custom batch size (defaults to FaceDetector)."""
        from .detectors.face import FaceDetector
        return FaceDetector(
            enable_gpu=self.enable_gpu,
            batch_size=batch_size,
            cache_enabled=self.cache_enabled
        )
    
    def get_opencv_detector(self, batch_size: int = 8):
        """Get OpenCV face detector with custom batch size."""
        from .detectors.face import FaceDetector
        return FaceDetector(
            enable_gpu=self.enable_gpu,
            cache_enabled=self.cache_enabled,
            batch_size=batch_size
        )
    
    def get_insightface_detector(self, batch_size: int = 8, model_name: str = "buffalo_l"):
        """Get InsightFace detector with custom batch size and model."""
        from .detectors.face import InsightFaceDetector
        return InsightFaceDetector(
            enable_gpu=self.enable_gpu,
            cache_enabled=self.cache_enabled,
            batch_size=batch_size,
            model_name=model_name,
            verbose=self.verbose
        )
    
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
    
    def get_object_detector(self, gpu_batch_size: int = 8):
        """Get object detector with custom batch size."""
        from .detectors.object import ObjectDetector
        return ObjectDetector(
            enable_gpu=self.enable_gpu,
            cache_enabled=self.cache_enabled,
            gpu_batch_size=gpu_batch_size
        )
    
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
    def quality_assessor(self):
        """Lazy-loaded quality assessor."""
        if self._quality_assessor is None:
            from .detectors.quality import QualityAssessor
            self._quality_assessor = QualityAssessor(
                cache_enabled=self.cache_enabled
            )
        return self._quality_assessor
    
    @property
    def face_clustering(self):
        """Lazy-loaded face clustering."""
        if self._face_clustering is None:
            from .detectors.face_clustering import FaceClustering
            self._face_clustering = FaceClustering(
                cache_enabled=self.cache_enabled,
                verbose=self.verbose
            )
        return self._face_clustering
    
    @timing_decorator
    @gpu_accelerated(fallback_cpu=True)
    def detect_faces(self, 
                     image_paths: Union[Path, List[Path]], 
                     save_sidecar: bool = True,
                     **kwargs) -> Dict[str, Any]:
        """
        Detect faces in images using sequential processing.
        
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
        
        # Use InsightFace detector (default and most reliable)
        face_detector = self.face_detector
        
        # Perform sequential detection
        results = face_detector.detect_faces_batch(image_paths, **kwargs)
        
        # Save to sidecar if requested
        if save_sidecar:
            for image_path in image_paths:
                if str(image_path) in results:
                    # Load image dimensions for ratio calculation
                    try:
                        import cv2
                        image = cv2.imread(str(image_path))
                        if image is not None:
                            image_height, image_width = image.shape[:2]
                        else:
                            image_width = image_height = None
                    except Exception:
                        image_width = image_height = None
                    
                    # Format the result for JSON serialization
                    formatted_result = face_detector._format_result(
                        results[str(image_path)], 
                        image_path,
                        image_width,
                        image_height
                    )
                    self.sidecar.save_data_merge(
                        image_path, 
                        "face_detection", 
                        formatted_result,
                        metadata={"kwargs": kwargs}
                    )
        
        return results
    
    @timing_decorator
    @gpu_accelerated(fallback_cpu=True)
    def detect_objects(self, 
                      image_paths: Union[Path, List[Path]], 
                      save_sidecar: bool = True,
                      gpu_batch_size: int = 8,
                      force: bool = False,
                      **kwargs) -> Dict[str, Any]:
        """
        Detect objects in images.
        
        Args:
            image_paths: Single image path or list of image paths
            save_sidecar: Whether to save results to sidecar files
            gpu_batch_size: GPU batch size for processing multiple images
            **kwargs: Additional arguments for object detection
            
        Returns:
            Dictionary containing detection results
        """
        if isinstance(image_paths, Path):
            image_paths = [image_paths]
        
        self.logger.info(f"Detecting objects in {len(image_paths)} images")
        
        # Use custom object detector with specified batch size
        object_detector = self.get_object_detector(gpu_batch_size=gpu_batch_size)
        
        # Perform batch detection
        results = object_detector.detect_objects(image_paths, save_sidecar=save_sidecar, force=force, **kwargs)
        
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
                sidecar_data = self.sidecar.load_data(image_path, "yolov8")
                if sidecar_data and 'yolov8' in sidecar_data:
                    detection_data = sidecar_data['yolov8']
                else:
                    # Perform detection first
                    detection_data = self.detect_objects(image_path, save_sidecar=True)
                    if detection_data and str(image_path) in detection_data:
                        detection_data = detection_data[str(image_path)]
                
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
    
    def extract_faces(self, 
                     image_paths: Union[Path, List[Path]], 
                     output_dir: Path,
                     padding: int = 20,
                     max_workers: Optional[int] = None,
                     **kwargs) -> Dict[str, Any]:
        """
        Extract detected faces from images with parallel processing.
        
        Args:
            image_paths: Single image path or list of image paths
            output_dir: Directory to save extracted faces
            padding: Padding around face in pixels (default: 20px)
            max_workers: Maximum number of parallel workers (None for auto)
            **kwargs: Additional arguments for extraction
            
        Returns:
            Dictionary containing extraction results
        """
        if isinstance(image_paths, Path):
            image_paths = [image_paths]
        
        self.logger.info(f"Extracting faces from {len(image_paths)} images")
        
        # Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine number of workers
        if max_workers is None:
            max_workers = min(len(image_paths), self.max_workers or 4)
        
        results = {}
        
        # Use parallel processing for multiple images
        if len(image_paths) > 1 and max_workers > 1:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            from tqdm import tqdm
            
            def extract_single_image(image_path: Path) -> tuple:
                """Extract faces from a single image."""
                try:
                    # Load detection data
                    detection_data = self.sidecar.load_data(image_path, "face_detection")
                    if not detection_data:
                        # Perform detection first
                        detection_data = self.detect_faces(image_path, save_sidecar=True)
                    
                    # Extract faces using InsightFace detector (most reliable)
                    face_detector = self.face_detector
                    extraction_result = face_detector.extract_faces(
                        image_path, 
                        output_dir,
                        padding=padding,
                        **kwargs
                    )
                    
                    return str(image_path), extraction_result
                    
                except Exception as e:
                    self.logger.error(f"Face extraction failed for {image_path}: {e}")
                    return str(image_path), {"error": str(e), "success": False}
            
            # Process images in parallel with progress bar
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all tasks
                future_to_path = {
                    executor.submit(extract_single_image, image_path): image_path 
                    for image_path in image_paths
                }
                
                # Process completed tasks with progress bar
                with tqdm(total=len(image_paths), desc="Extracting faces", unit="images") as pbar:
                    for future in as_completed(future_to_path):
                        image_path, result = future.result()
                        results[image_path] = result
                        pbar.update(1)
                        
                        # Update progress bar description with current stats
                        successful = sum(1 for r in results.values() if r.get('success', False))
                        total_faces = sum(r.get('faces_extracted', 0) for r in results.values())
                        pbar.set_postfix({
                            'successful': successful,
                            'faces': total_faces
                        })
        
        else:
            # Sequential processing for single image or when parallel processing is disabled
            for image_path in image_paths:
                try:
                    # Load detection data
                    detection_data = self.sidecar.load_data(image_path, "face_detection")
                    if not detection_data:
                        # Perform detection first
                        detection_data = self.detect_faces(image_path, save_sidecar=True)
                    
                    # Extract faces using InsightFace detector (most reliable)
                    face_detector = self.face_detector
                    extraction_result = face_detector.extract_faces(
                        image_path, 
                        output_dir,
                        padding=padding,
                        **kwargs
                    )
                    
                    results[str(image_path)] = extraction_result
                    
                except Exception as e:
                    self.logger.error(f"Face extraction failed for {image_path}: {e}")
                    results[str(image_path)] = {"error": str(e), "success": False}
        
        return results
    
    @timing_decorator
    def cluster_faces(self, 
                     input_source: Union[Path, Dict[str, Any]], 
                     similarity_threshold: float = 0.6,
                     min_cluster_size: int = 2,
                     algorithm: str = "dbscan",
                     max_faces: Optional[int] = None,
                     save_sidecar: bool = True,
                     **kwargs) -> Dict[str, Any]:
        """
        Cluster similar faces together.
        
        Args:
            input_source: Directory path or detection results dictionary
            similarity_threshold: Minimum similarity for faces to be in same cluster
            min_cluster_size: Minimum number of faces required to form a cluster
            algorithm: Clustering algorithm ('dbscan', 'agglomerative', 'kmeans')
            max_faces: Maximum number of faces to cluster (None for all)
            save_sidecar: Whether to save results to sidecar files
            **kwargs: Additional arguments for clustering
            
        Returns:
            Dictionary containing clustering results
        """
        self.logger.info(f"Clustering faces with {algorithm} algorithm")
        
        # Get face clustering instance with custom parameters
        face_clustering = self.get_face_clustering(
            similarity_threshold=similarity_threshold,
            min_cluster_size=min_cluster_size,
            algorithm=algorithm
        )
        
        # Perform clustering
        if isinstance(input_source, Path):
            # Cluster from directory
            clustering_result = face_clustering.cluster_faces_from_directory(
                input_source, max_faces=max_faces
            )
        else:
            # Cluster from detection results
            clustering_result = face_clustering.cluster_faces_from_detections(
                input_source, max_faces=max_faces
            )
        
        # Save to sidecar if requested
        if save_sidecar and clustering_result.success:
            if isinstance(input_source, Path):
                sidecar_path = input_source / "face_clustering.json"
            else:
                # Use base directory for detection results
                sidecar_path = self.base_dir / "face_clustering.json"
            
            self.sidecar.save_data(
                sidecar_path, 
                "face_clustering", 
                clustering_result.as_dict(),
                metadata={
                    "similarity_threshold": similarity_threshold,
                    "min_cluster_size": min_cluster_size,
                    "algorithm": algorithm,
                    "max_faces": max_faces,
                    "kwargs": kwargs
                }
            )
        
        return clustering_result.as_dict()
    
    def get_face_clustering(self, 
                           similarity_threshold: float = 0.6,
                           min_cluster_size: int = 2,
                           algorithm: str = "dbscan"):
        """Get face clustering instance with custom parameters."""
        from .detectors.face_clustering import FaceClustering
        return FaceClustering(
            similarity_threshold=similarity_threshold,
            min_cluster_size=min_cluster_size,
            algorithm=algorithm,
            cache_enabled=self.cache_enabled,
            verbose=self.verbose
        )
    
    def export_face_clusters(self, 
                           clustering_result: Dict[str, Any],
                           output_dir: Path,
                           export_format: str = "json",
                           create_visualization: bool = True,
                           **kwargs) -> Dict[str, Any]:
        """
        Export face clustering results.
        
        Args:
            clustering_result: Result from cluster_faces method
            output_dir: Directory to save export files
            export_format: Export format ('json', 'csv', 'both')
            create_visualization: Whether to create cluster visualization images
            **kwargs: Additional arguments for export
            
        Returns:
            Dictionary containing export results
        """
        self.logger.info(f"Exporting face clusters to {output_dir}")
        
        # Get face clustering instance
        face_clustering = self.face_clustering
        
        # Convert dict back to FaceClusteringResult for export
        from .detectors.face_clustering import FaceClusteringResult, FaceCluster
        
        # Reconstruct clusters
        clusters = []
        for cluster_data in clustering_result.get('clusters', []):
            cluster = FaceCluster(
                cluster_id=cluster_data['cluster_id'],
                face_ids=cluster_data['face_ids'],
                centroid_encoding=cluster_data.get('centroid_encoding'),
                face_count=cluster_data.get('face_count', 0),
                confidence_scores=cluster_data.get('confidence_scores', []),
                image_paths=cluster_data.get('image_paths', [])
            )
            clusters.append(cluster)
        
        # Reconstruct result
        result = FaceClusteringResult(
            clusters=clusters,
            unclustered_faces=clustering_result.get('unclustered_faces', []),
            total_faces=clustering_result.get('total_faces', 0),
            cluster_count=clustering_result.get('cluster_count', 0),
            success=clustering_result.get('success', False),
            processing_time=clustering_result.get('processing_time', 0.0),
            algorithm_used=clustering_result.get('algorithm_used', 'dbscan'),
            parameters=clustering_result.get('parameters', {}),
            error=clustering_result.get('error')
        )
        
        # Export results
        export_results = face_clustering.export_clusters(
            result, output_dir, export_format
        )
        
        # Create visualization if requested
        if create_visualization and result.success:
            # Load detection results for visualization
            detection_results = {}
            for cluster in clusters:
                for image_path in cluster.image_paths:
                    if image_path not in detection_results:
                        # Try to load from sidecar
                        sidecar_data = self.sidecar.load_data(Path(image_path), "face_detection")
                        if sidecar_data:
                            detection_results[image_path] = sidecar_data
            
            if detection_results:
                viz_results = face_clustering.visualize_clusters(
                    result, detection_results, output_dir / "visualizations"
                )
                export_results['visualization'] = viz_results
        
        return export_results
    
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
    
    # Tool-agnostic detection methods
    
    def detect_with_tool(self, tool_name: str, image_paths: Union[Path, List[Path]], 
                        config_override: Optional[Dict[str, Any]] = None,
                        **kwargs) -> Union[Any, Dict[str, Any]]:
        """
        Perform detection using a specific tool (tool-agnostic).
        
        Args:
            tool_name: Name of the detection tool
            image_paths: Single image path or list of image paths
            config_override: Optional configuration overrides
            **kwargs: Additional detection parameters
            
        Returns:
            Detection result(s)
        """
        return self.detection.detect_with_tool(tool_name, image_paths, config_override, **kwargs)
    
    def validate_sidecar_files(self, directory: Path, 
                             operation_type: Optional[str] = None,
                             use_rust: bool = True) -> List[Dict[str, Any]]:
        """
        Validate sidecar files in a directory (massively parallel).
        
        Args:
            directory: Directory to search for sidecar files
            operation_type: Optional operation type filter
            use_rust: Whether to use Rust implementation if available
            
        Returns:
            List of validation results
        """
        return self.detection.validate_sidecar_files(directory, operation_type, use_rust)
    
    def get_available_detection_tools(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information about all available detection tools.
        
        Returns:
            Dictionary mapping tool names to their information
        """
        return self.detection.get_available_tools()
    
    def register_custom_detection_tool(self, tool_name: str, tool_class, 
                                     config: Optional[Dict[str, Any]] = None) -> None:
        """
        Register a custom detection tool.
        
        Args:
            tool_name: Name of the tool
            tool_class: DetectionTool subclass
            config: Optional configuration for the tool
        """
        from .detection.base import DetectionConfig
        detection_config = DetectionConfig()
        if config:
            detection_config.update_from_dict(config)
        self.detection.register_custom_tool(tool_name, tool_class, detection_config)
    
    def get_detection_performance_info(self) -> Dict[str, Any]:
        """
        Get performance information about the detection system.
        
        Returns:
            Dictionary with performance information
        """
        return self.detection.get_performance_info()
    
    def benchmark_detection_performance(self, test_files: List[Path], 
                                      iterations: int = 3) -> Dict[str, Any]:
        """
        Benchmark performance of the detection system.
        
        Args:
            test_files: List of test files to use
            iterations: Number of benchmark iterations
            
        Returns:
            Dictionary with benchmark results
        """
        return self.detection.benchmark_performance(test_files, iterations)
