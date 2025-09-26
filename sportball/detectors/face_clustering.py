"""
Face Clustering Module

Face clustering functionality for grouping similar faces together.

Author: Claude Sonnet 4 (claude-3-5-sonnet-20241022)
Generated via Cursor IDE (cursor.sh) with AI assistance
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict
from loguru import logger
import json
import time

try:
    from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
    from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn not available - install with: pip install scikit-learn")

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    logger.warning("OpenCV not available - face clustering will be limited")

from ..decorators import cached_result


@dataclass
class FaceCluster:
    """Information about a face cluster."""
    cluster_id: int
    face_ids: List[str]  # List of face identifiers (image_path:face_id)
    centroid_encoding: Optional[List[float]] = None
    face_count: int = 0
    confidence_scores: List[float] = None
    image_paths: List[str] = None
    
    def __post_init__(self):
        """Initialize derived fields."""
        if self.face_count == 0:
            self.face_count = len(self.face_ids)
        if self.confidence_scores is None:
            self.confidence_scores = []
        if self.image_paths is None:
            self.image_paths = []
    
    def as_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        result = asdict(self)
        return result


@dataclass
class FaceClusteringResult:
    """Result of face clustering operation."""
    clusters: List[FaceCluster]
    unclustered_faces: List[str]  # Face IDs that couldn't be clustered
    total_faces: int
    cluster_count: int
    success: bool
    processing_time: float
    algorithm_used: str
    parameters: Dict[str, Any]
    error: Optional[str] = None
    
    def as_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        result = asdict(self)
        if 'clusters' in result:
            result['clusters'] = [cluster.as_dict() for cluster in self.clusters]
        return result


class FaceClustering:
    """
    Face clustering using various algorithms to group similar faces together.
    """
    
    def __init__(self, 
                 similarity_threshold: float = 0.6,
                 min_cluster_size: int = 2,
                 algorithm: str = "dbscan",
                 cache_enabled: bool = True,
                 verbose: bool = False):
        """
        Initialize face clustering.
        
        Args:
            similarity_threshold: Minimum similarity for faces to be in same cluster
            min_cluster_size: Minimum number of faces required to form a cluster
            algorithm: Clustering algorithm ('dbscan', 'agglomerative', 'kmeans')
            cache_enabled: Whether to enable result caching
            verbose: Whether to show verbose output
        """
        self.similarity_threshold = similarity_threshold
        self.min_cluster_size = min_cluster_size
        self.algorithm = algorithm.lower()
        self.cache_enabled = cache_enabled
        self.verbose = verbose
        
        # Initialize logger
        self.logger = logger.bind(component="face_clustering")
        
        if not SKLEARN_AVAILABLE:
            self.logger.error("scikit-learn not available - install with: pip install scikit-learn")
            raise ImportError("scikit-learn is required for face clustering")
        
        # Validate algorithm
        valid_algorithms = ['dbscan', 'agglomerative', 'kmeans']
        if self.algorithm not in valid_algorithms:
            raise ValueError(f"Invalid algorithm '{self.algorithm}'. Must be one of: {valid_algorithms}")
        
        self.logger.info(f"Face clustering initialized with {self.algorithm} algorithm")
    
    def cluster_faces_from_detections(self, 
                                    detection_results: Dict[str, Any],
                                    max_faces: Optional[int] = None) -> FaceClusteringResult:
        """
        Cluster faces from face detection results.
        
        Args:
            detection_results: Dictionary mapping image paths to face detection results
            max_faces: Maximum number of faces to cluster (None for all)
            
        Returns:
            FaceClusteringResult containing clustered faces
        """
        import time
        start_time = time.time()
        
        try:
            # Extract face encodings and metadata
            face_encodings, face_metadata = self._extract_face_data(detection_results, max_faces)
            
            if len(face_encodings) == 0:
                return FaceClusteringResult(
                    clusters=[],
                    unclustered_faces=[],
                    total_faces=0,
                    cluster_count=0,
                    success=False,
                    processing_time=time.time() - start_time,
                    algorithm_used=self.algorithm,
                    parameters=self._get_algorithm_parameters(),
                    error="No face encodings found in detection results"
                )
            
            # Perform clustering
            cluster_labels = self._perform_clustering(face_encodings)
            
            # Group faces into clusters
            clusters, unclustered_faces = self._group_faces_into_clusters(
                face_encodings, face_metadata, cluster_labels
            )
            
            processing_time = time.time() - start_time
            
            result = FaceClusteringResult(
                clusters=clusters,
                unclustered_faces=unclustered_faces,
                total_faces=len(face_encodings),
                cluster_count=len(clusters),
                success=True,
                processing_time=processing_time,
                algorithm_used=self.algorithm,
                parameters=self._get_algorithm_parameters()
            )
            
            if self.verbose:
                self.logger.info(f"Clustering completed: {len(clusters)} clusters, {len(unclustered_faces)} unclustered faces")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Face clustering failed: {e}")
            return FaceClusteringResult(
                clusters=[],
                unclustered_faces=[],
                total_faces=0,
                cluster_count=0,
                success=False,
                processing_time=time.time() - start_time,
                algorithm_used=self.algorithm,
                parameters=self._get_algorithm_parameters(),
                error=str(e)
            )
    
    def cluster_faces_from_directory(self, 
                                   directory: Path,
                                   pattern: str = "*.json",
                                   max_faces: Optional[int] = None) -> FaceClusteringResult:
        """
        Cluster faces from sidecar files in a directory.
        
        Args:
            directory: Directory containing face detection sidecar files
            pattern: File pattern to match sidecar files
            max_faces: Maximum number of faces to cluster (None for all)
            
        Returns:
            FaceClusteringResult containing clustered faces
        """
        try:
            # Find sidecar files
            sidecar_files = list(directory.glob(pattern))
            if not sidecar_files:
                return FaceClusteringResult(
                    clusters=[],
                    unclustered_faces=[],
                    total_faces=0,
                    cluster_count=0,
                    success=False,
                    processing_time=0.0,
                    algorithm_used=self.algorithm,
                    parameters=self._get_algorithm_parameters(),
                    error=f"No sidecar files found matching pattern '{pattern}' in {directory}"
                )
            
            # Load detection results
            detection_results = {}
            for sidecar_file in sidecar_files:
                try:
                    with open(sidecar_file, 'r') as f:
                        data = json.load(f)
                        if data.get('success', False) and 'faces' in data:
                            # Extract image path from sidecar filename
                            image_path = sidecar_file.stem.replace('_face_detection', '')
                            detection_results[image_path] = data
                except Exception as e:
                    self.logger.warning(f"Failed to load sidecar file {sidecar_file}: {e}")
            
            return self.cluster_faces_from_detections(detection_results, max_faces)
            
        except Exception as e:
            self.logger.error(f"Directory clustering failed: {e}")
            return FaceClusteringResult(
                clusters=[],
                unclustered_faces=[],
                total_faces=0,
                cluster_count=0,
                success=False,
                processing_time=0.0,
                algorithm_used=self.algorithm,
                parameters=self._get_algorithm_parameters(),
                error=str(e)
            )
    
    def _extract_face_data(self, 
                          detection_results: Dict[str, Any], 
                          max_faces: Optional[int] = None) -> Tuple[List[np.ndarray], List[Dict[str, Any]]]:
        """
        Extract face encodings and metadata from detection results.
        
        Args:
            detection_results: Dictionary mapping image paths to face detection results
            max_faces: Maximum number of faces to extract
            
        Returns:
            Tuple of (face_encodings, face_metadata)
        """
        face_encodings = []
        face_metadata = []
        
        for image_path, detection_data in detection_results.items():
            if not detection_data.get('success', False):
                continue
            
            faces = detection_data.get('faces', [])
            for face_idx, face_data in enumerate(faces):
                # Check if face has encoding
                if 'encoding' not in face_data or face_data['encoding'] is None:
                    continue
                
                # Convert encoding to numpy array
                encoding = np.array(face_data['encoding'], dtype=np.float32)
                face_encodings.append(encoding)
                
                # Store metadata
                face_id = f"{image_path}:{face_data.get('face_id', face_idx)}"
                metadata = {
                    'face_id': face_id,
                    'image_path': image_path,
                    'face_idx': face_idx,
                    'bbox': face_data.get('bbox', {}),
                    'confidence': face_data.get('confidence', 0.0)
                }
                face_metadata.append(metadata)
                
                # Check max_faces limit
                if max_faces and len(face_encodings) >= max_faces:
                    break
            
            if max_faces and len(face_encodings) >= max_faces:
                break
        
        return face_encodings, face_metadata
    
    def _perform_clustering(self, face_encodings: List[np.ndarray]) -> np.ndarray:
        """
        Perform clustering on face encodings.
        
        Args:
            face_encodings: List of face encoding vectors
            
        Returns:
            Array of cluster labels
        """
        if len(face_encodings) < 2:
            return np.array([0] * len(face_encodings))
        
        # Convert to numpy array
        X = np.array(face_encodings)
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        if self.algorithm == "dbscan":
            return self._dbscan_clustering(X_scaled)
        elif self.algorithm == "agglomerative":
            return self._agglomerative_clustering(X_scaled)
        elif self.algorithm == "kmeans":
            return self._kmeans_clustering(X_scaled)
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")
    
    def _dbscan_clustering(self, X: np.ndarray) -> np.ndarray:
        """Perform DBSCAN clustering."""
        # Convert similarity threshold to distance threshold
        # For cosine similarity: distance = 1 - similarity
        eps = 1.0 - self.similarity_threshold
        
        clustering = DBSCAN(
            eps=eps,
            min_samples=self.min_cluster_size,
            metric='cosine'
        )
        
        return clustering.fit_predict(X)
    
    def _agglomerative_clustering(self, X: np.ndarray) -> np.ndarray:
        """Perform Agglomerative clustering."""
        # Estimate number of clusters based on similarity threshold
        n_clusters = max(1, len(X) // self.min_cluster_size)
        
        clustering = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage='average',
            metric='cosine'
        )
        
        return clustering.fit_predict(X)
    
    def _kmeans_clustering(self, X: np.ndarray) -> np.ndarray:
        """Perform K-means clustering."""
        # Estimate number of clusters
        n_clusters = max(1, min(len(X) // self.min_cluster_size, len(X)))
        
        clustering = KMeans(
            n_clusters=n_clusters,
            random_state=42,
            n_init=10
        )
        
        return clustering.fit_predict(X)
    
    def _group_faces_into_clusters(self, 
                                 face_encodings: List[np.ndarray],
                                 face_metadata: List[Dict[str, Any]],
                                 cluster_labels: np.ndarray) -> Tuple[List[FaceCluster], List[str]]:
        """
        Group faces into clusters based on cluster labels.
        
        Args:
            face_encodings: List of face encoding vectors
            face_metadata: List of face metadata dictionaries
            cluster_labels: Array of cluster labels
            
        Returns:
            Tuple of (clusters, unclustered_faces)
        """
        clusters = []
        unclustered_faces = []
        
        # Group faces by cluster label
        cluster_groups = {}
        for i, label in enumerate(cluster_labels):
            if label == -1:  # Noise/unclustered
                unclustered_faces.append(face_metadata[i]['face_id'])
            else:
                if label not in cluster_groups:
                    cluster_groups[label] = []
                cluster_groups[label].append(i)
        
        # Create FaceCluster objects
        for cluster_id, face_indices in cluster_groups.items():
            if len(face_indices) < self.min_cluster_size:
                # Move to unclustered
                for idx in face_indices:
                    unclustered_faces.append(face_metadata[idx]['face_id'])
                continue
            
            # Extract cluster data
            cluster_encodings = [face_encodings[i] for i in face_indices]
            cluster_face_ids = [face_metadata[i]['face_id'] for i in face_indices]
            cluster_confidences = [face_metadata[i]['confidence'] for i in face_indices]
            cluster_image_paths = [face_metadata[i]['image_path'] for i in face_indices]
            
            # Calculate centroid
            centroid_encoding = np.mean(cluster_encodings, axis=0).tolist()
            
            cluster = FaceCluster(
                cluster_id=cluster_id,
                face_ids=cluster_face_ids,
                centroid_encoding=centroid_encoding,
                face_count=len(face_indices),
                confidence_scores=cluster_confidences,
                image_paths=cluster_image_paths
            )
            clusters.append(cluster)
        
        return clusters, unclustered_faces
    
    def _get_algorithm_parameters(self) -> Dict[str, Any]:
        """Get current algorithm parameters."""
        return {
            'similarity_threshold': self.similarity_threshold,
            'min_cluster_size': self.min_cluster_size,
            'algorithm': self.algorithm
        }
    
    def export_clusters(self, 
                       clustering_result: FaceClusteringResult,
                       output_dir: Path,
                       export_format: str = "json") -> Dict[str, Any]:
        """
        Export clustering results to files.
        
        Args:
            clustering_result: Result of face clustering
            output_dir: Directory to save export files
            export_format: Export format ('json', 'csv', 'both')
            
        Returns:
            Dictionary containing export results
        """
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            
            export_results = {
                'success': True,
                'files_created': [],
                'error': None
            }
            
            if export_format in ['json', 'both']:
                # Export JSON
                json_file = output_dir / "face_clusters.json"
                with open(json_file, 'w') as f:
                    json.dump(clustering_result.as_dict(), f, indent=2)
                export_results['files_created'].append(str(json_file))
            
            if export_format in ['csv', 'both']:
                # Export CSV
                csv_file = output_dir / "face_clusters.csv"
                self._export_clusters_csv(clustering_result, csv_file)
                export_results['files_created'].append(str(csv_file))
            
            return export_results
            
        except Exception as e:
            self.logger.error(f"Export failed: {e}")
            return {
                'success': False,
                'files_created': [],
                'error': str(e)
            }
    
    def _export_clusters_csv(self, clustering_result: FaceClusteringResult, csv_file: Path):
        """Export clustering results to CSV format."""
        import csv
        
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Write header
            writer.writerow([
                'cluster_id', 'face_id', 'image_path', 'confidence', 
                'face_count', 'algorithm', 'similarity_threshold'
            ])
            
            # Write cluster data
            for cluster in clustering_result.clusters:
                for i, face_id in enumerate(cluster.face_ids):
                    image_path = cluster.image_paths[i] if i < len(cluster.image_paths) else ""
                    confidence = cluster.confidence_scores[i] if i < len(cluster.confidence_scores) else 0.0
                    
                    writer.writerow([
                        cluster.cluster_id,
                        face_id,
                        image_path,
                        confidence,
                        cluster.face_count,
                        clustering_result.algorithm_used,
                        clustering_result.parameters.get('similarity_threshold', 0.6)
                    ])
            
            # Write unclustered faces
            for face_id in clustering_result.unclustered_faces:
                writer.writerow([
                    -1,  # -1 for unclustered
                    face_id,
                    "",  # image_path
                    0.0,  # confidence
                    1,    # face_count
                    clustering_result.algorithm_used,
                    clustering_result.parameters.get('similarity_threshold', 0.6)
                ])
    
    def visualize_clusters(self, 
                          clustering_result: FaceClusteringResult,
                          detection_results: Dict[str, Any],
                          output_dir: Path,
                          max_faces_per_cluster: int = 10) -> Dict[str, Any]:
        """
        Create visualization of face clusters.
        
        Args:
            clustering_result: Result of face clustering
            detection_results: Original face detection results
            output_dir: Directory to save visualization images
            max_faces_per_cluster: Maximum number of faces to show per cluster
            
        Returns:
            Dictionary containing visualization results
        """
        if not CV2_AVAILABLE:
            return {
                'success': False,
                'error': 'OpenCV not available for visualization'
            }
        
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            
            visualization_results = {
                'success': True,
                'images_created': [],
                'error': None
            }
            
            # Create cluster visualization images
            for cluster in clustering_result.clusters:
                if cluster.face_count == 0:
                    continue
                
                # Limit faces per cluster
                faces_to_show = cluster.face_ids[:max_faces_per_cluster]
                
                # Create cluster image
                cluster_image = self._create_cluster_image(
                    faces_to_show, detection_results, cluster
                )
                
                if cluster_image is not None:
                    # Save cluster image
                    cluster_file = output_dir / f"cluster_{cluster.cluster_id:03d}.jpg"
                    cv2.imwrite(str(cluster_file), cluster_image)
                    visualization_results['images_created'].append(str(cluster_file))
            
            return visualization_results
            
        except Exception as e:
            self.logger.error(f"Visualization failed: {e}")
            return {
                'success': False,
                'images_created': [],
                'error': str(e)
            }
    
    def _create_cluster_image(self, 
                            face_ids: List[str],
                            detection_results: Dict[str, Any],
                            cluster: FaceCluster) -> Optional[np.ndarray]:
        """Create a visualization image for a cluster."""
        try:
            face_images = []
            
            for face_id in face_ids:
                # Parse face_id (image_path:face_idx)
                if ':' not in face_id:
                    continue
                
                image_path, face_idx_str = face_id.split(':', 1)
                try:
                    face_idx = int(face_idx_str)
                except ValueError:
                    continue
                
                # Find the detection result
                detection_data = None
                for path, data in detection_results.items():
                    if path in image_path or image_path in path:
                        detection_data = data
                        break
                
                if not detection_data or not detection_data.get('success', False):
                    continue
                
                faces = detection_data.get('faces', [])
                if face_idx >= len(faces):
                    continue
                
                face_data = faces[face_idx]
                bbox = face_data.get('bbox', {})
                
                # Load and crop face
                try:
                    image = cv2.imread(image_path)
                    if image is None:
                        continue
                    
                    # Get image dimensions
                    h, w = image.shape[:2]
                    
                    # Convert normalized bbox to pixel coordinates
                    x = int(bbox.get('x', 0) * w)
                    y = int(bbox.get('y', 0) * h)
                    face_w = int(bbox.get('width', 0) * w)
                    face_h = int(bbox.get('height', 0) * h)
                    
                    # Crop face with padding
                    padding = 20
                    x_start = max(0, x - padding)
                    y_start = max(0, y - padding)
                    x_end = min(w, x + face_w + padding)
                    y_end = min(h, y + face_h + padding)
                    
                    face_crop = image[y_start:y_end, x_start:x_end]
                    
                    # Resize to standard size
                    face_crop = cv2.resize(face_crop, (128, 128))
                    face_images.append(face_crop)
                    
                except Exception as e:
                    self.logger.warning(f"Failed to process face {face_id}: {e}")
                    continue
            
            if not face_images:
                return None
            
            # Create grid layout
            cols = min(5, len(face_images))
            rows = (len(face_images) + cols - 1) // cols
            
            # Create output image
            cell_size = 128
            output_w = cols * cell_size
            output_h = rows * cell_size
            output_image = np.zeros((output_h, output_w, 3), dtype=np.uint8)
            
            # Place face images
            for i, face_img in enumerate(face_images):
                row = i // cols
                col = i % cols
                
                y_start = row * cell_size
                x_start = col * cell_size
                y_end = y_start + cell_size
                x_end = x_start + cell_size
                
                output_image[y_start:y_end, x_start:x_end] = face_img
            
            return output_image
            
        except Exception as e:
            self.logger.error(f"Failed to create cluster image: {e}")
            return None
