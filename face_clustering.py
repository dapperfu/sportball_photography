#!/usr/bin/env python3
"""
AI-Based Face Clustering Tool

This tool reads face recognition JSON data and uses AI clustering algorithms
to group similar faces into likely persons. It then extracts and crops faces
with configurable padding and saves them to organized directories.

Features:
- Multiple clustering algorithms (DBSCAN, K-means, Hierarchical)
- Face encoding extraction from JSON data
- Configurable padding for face crops
- Organized output directory structure
- Parallel processing for performance
- Comprehensive error handling and logging

Author: Claude Sonnet 4 (claude-3-5-sonnet-20241022)
Generated via Cursor IDE (cursor.sh) with AI assistance
"""

import json
import logging
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any, Union
import click
import cv2
import numpy as np
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table
from rich.panel import Panel
from rich.logging import RichHandler
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import warnings

# Configure rich console
console = Console()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, rich_tracebacks=True)]
)
logger = logging.getLogger("face_clustering")

# Suppress sklearn warnings
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')


@dataclass
class FaceData:
    """Represents face data from JSON with encoding."""
    image_path: Path
    face_id: int
    x: int
    y: int
    width: int
    height: int
    confidence: float
    face_encoding: Optional[List[float]]
    crop_x: int
    crop_y: int
    crop_width: int
    crop_height: int
    detection_scale_factor: float


@dataclass
class ClusterResult:
    """Result of face clustering."""
    cluster_id: int
    faces: List[FaceData]
    center_encoding: Optional[np.ndarray]
    confidence_score: float
    face_count: int


class FaceClusteringEngine:
    """AI-based face clustering engine with multiple algorithms."""
    
    def __init__(self, 
                 padding: float = 0.25,
                 min_cluster_size: int = 2,
                 max_clusters: int = 50,
                 algorithm: str = 'dbscan',
                 max_workers: int = 4):
        """
        Initialize face clustering engine.
        
        Args:
            padding: Padding percentage for face crops (0.25 = 25%)
            min_cluster_size: Minimum faces per cluster
            max_clusters: Maximum number of clusters for K-means
            algorithm: Clustering algorithm ('dbscan', 'kmeans', 'hierarchical')
            max_workers: Maximum parallel workers
        """
        self.padding = padding
        self.min_cluster_size = min_cluster_size
        self.max_clusters = max_clusters
        self.algorithm = algorithm.lower()
        self.max_workers = max_workers
        self.face_data: List[FaceData] = []
        self.clusters: List[ClusterResult] = []
        self.performance_metrics = {}
        
        # Validate algorithm
        if self.algorithm not in ['dbscan', 'kmeans', 'hierarchical']:
            raise ValueError(f"Unsupported algorithm: {algorithm}. Use 'dbscan', 'kmeans', or 'hierarchical'")
    
    def load_face_data_from_json(self, json_paths: List[Path]) -> List[FaceData]:
        """
        Load face data from JSON files.
        
        Args:
            json_paths: List of JSON file paths
            
        Returns:
            List of FaceData objects
        """
        face_data = []
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            
            task = progress.add_task("Loading face data from JSON files...", total=len(json_paths))
            
            for json_path in json_paths:
                try:
                    with open(json_path, 'r') as f:
                        data = json.load(f)
                    
                    # Extract image path (JSON file should be named like image.json)
                    image_path = json_path.parent / json_path.stem
                    
                    # Handle different image extensions
                    for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
                        potential_image = image_path.with_suffix(ext)
                        if potential_image.exists():
                            image_path = potential_image
                            break
                    
                    if not image_path.exists():
                        logger.warning(f"Image not found for JSON: {json_path}")
                        continue
                    
                    # Extract face data from JSON
                    if 'Face_detector' in data and 'faces' in data['Face_detector']:
                        faces = data['Face_detector']['faces']
                        
                        for face in faces:
                            # Only include faces with encodings
                            if face.get('face_encoding') is not None:
                                face_data.append(FaceData(
                                    image_path=image_path,
                                    face_id=face['face_id'],
                                    x=face['coordinates']['x'],
                                    y=face['coordinates']['y'],
                                    width=face['coordinates']['width'],
                                    height=face['coordinates']['height'],
                                    confidence=face['confidence'],
                                    face_encoding=face['face_encoding'],
                                    crop_x=face['crop_area']['x'],
                                    crop_y=face['crop_area']['y'],
                                    crop_width=face['crop_area']['width'],
                                    crop_height=face['crop_area']['height'],
                                    detection_scale_factor=face['detection_scale_factor']
                                ))
                    
                    elif 'Face_extractor' in data and 'faces' in data['Face_extractor']:
                        faces = data['Face_extractor']['faces']
                        
                        for face in faces:
                            # For extracted faces, we need to reconstruct the encoding
                            # This is a simplified approach - in practice, you'd want to
                            # re-extract encodings from the cropped face images
                            face_data.append(FaceData(
                                image_path=image_path,
                                face_id=face['face_id'],
                                x=face['coordinates']['x'],
                                y=face['coordinates']['y'],
                                width=face['coordinates']['width'],
                                height=face['coordinates']['height'],
                                confidence=face['confidence'],
                                face_encoding=None,  # Will need to be re-extracted
                                crop_x=face['coordinates']['x'],
                                crop_y=face['coordinates']['y'],
                                crop_width=face['coordinates']['width'],
                                crop_height=face['coordinates']['height'],
                                detection_scale_factor=1.0
                            ))
                
                except Exception as e:
                    logger.error(f"Error loading JSON {json_path}: {e}")
                    continue
                
                progress.advance(task)
        
        self.face_data = face_data
        console.print(f"✅ Loaded {len(face_data)} faces from {len(json_paths)} JSON files", style="green")
        return face_data
    
    def extract_face_encodings(self) -> np.ndarray:
        """
        Extract face encodings from face data.
        
        Returns:
            Array of face encodings
        """
        encodings = []
        valid_indices = []
        
        for i, face in enumerate(self.face_data):
            if face.face_encoding is not None:
                encodings.append(face.face_encoding)
                valid_indices.append(i)
        
        if not encodings:
            raise ValueError("No face encodings found in the data")
        
        # Convert to numpy array
        encodings_array = np.array(encodings)
        
        # Standardize the encodings
        scaler = StandardScaler()
        encodings_array = scaler.fit_transform(encodings_array)
        
        console.print(f"✅ Extracted {len(encodings)} face encodings", style="green")
        return encodings_array, valid_indices
    
    def cluster_faces(self, encodings: np.ndarray, valid_indices: List[int]) -> List[ClusterResult]:
        """
        Cluster faces using the specified algorithm.
        
        Args:
            encodings: Face encodings array
            valid_indices: Indices of faces with valid encodings
            
        Returns:
            List of ClusterResult objects
        """
        console.print(f"🤖 Clustering faces using {self.algorithm} algorithm...", style="blue")
        
        if self.algorithm == 'dbscan':
            clusters = self._cluster_dbscan(encodings)
        elif self.algorithm == 'kmeans':
            clusters = self._cluster_kmeans(encodings)
        elif self.algorithm == 'hierarchical':
            clusters = self._cluster_hierarchical(encodings)
        else:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}")
        
        # Convert cluster labels to ClusterResult objects
        cluster_results = []
        unique_labels = np.unique(clusters)
        
        for cluster_id in unique_labels:
            if cluster_id == -1:  # Skip noise points in DBSCAN
                continue
            
            # Get faces in this cluster
            cluster_mask = clusters == cluster_id
            cluster_indices = np.where(cluster_mask)[0]
            
            if len(cluster_indices) < self.min_cluster_size:
                continue
            
            # Get face data for this cluster
            cluster_faces = [self.face_data[valid_indices[i]] for i in cluster_indices]
            
            # Calculate cluster center
            cluster_encodings = encodings[cluster_mask]
            center_encoding = np.mean(cluster_encodings, axis=0)
            
            # Calculate confidence score (average confidence of faces)
            confidence_score = np.mean([face.confidence for face in cluster_faces])
            
            cluster_results.append(ClusterResult(
                cluster_id=cluster_id,
                faces=cluster_faces,
                center_encoding=center_encoding,
                confidence_score=confidence_score,
                face_count=len(cluster_faces)
            ))
        
        # Sort clusters by face count (descending)
        cluster_results.sort(key=lambda x: x.face_count, reverse=True)
        
        # Reassign cluster IDs starting from 1
        for i, cluster in enumerate(cluster_results, 1):
            cluster.cluster_id = i
        
        self.clusters = cluster_results
        console.print(f"✅ Created {len(cluster_results)} clusters", style="green")
        return cluster_results
    
    def _cluster_dbscan(self, encodings: np.ndarray) -> np.ndarray:
        """Cluster using DBSCAN algorithm."""
        # Estimate eps based on data
        from sklearn.neighbors import NearestNeighbors
        nbrs = NearestNeighbors(n_neighbors=min(4, len(encodings))).fit(encodings)
        distances, indices = nbrs.kneighbors(encodings)
        distances = np.sort(distances[:, -1])
        
        # Use elbow method to find optimal eps
        eps = np.percentile(distances, 75)  # Use 75th percentile as eps
        
        dbscan = DBSCAN(eps=eps, min_samples=self.min_cluster_size)
        return dbscan.fit_predict(encodings)
    
    def _cluster_kmeans(self, encodings: np.ndarray) -> np.ndarray:
        """Cluster using K-means algorithm."""
        # Determine optimal number of clusters
        n_clusters = min(self.max_clusters, max(2, len(encodings) // self.min_cluster_size))
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        return kmeans.fit_predict(encodings)
    
    def _cluster_hierarchical(self, encodings: np.ndarray) -> np.ndarray:
        """Cluster using Hierarchical clustering."""
        # Determine optimal number of clusters
        n_clusters = min(self.max_clusters, max(2, len(encodings) // self.min_cluster_size))
        
        hierarchical = AgglomerativeClustering(n_clusters=n_clusters)
        return hierarchical.fit_predict(encodings)
    
    def crop_and_save_faces(self, output_dir: Path) -> Dict[str, Path]:
        """
        Crop faces from images and save to organized directories.
        
        Args:
            output_dir: Output directory for organized faces
            
        Returns:
            Dictionary mapping cluster IDs to output directories
        """
        if not self.clusters:
            raise ValueError("No clusters available. Run clustering first.")
        
        console.print(f"📁 Creating organized face directories...", style="blue")
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        cluster_dirs = {}
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            
            total_faces = sum(len(cluster.faces) for cluster in self.clusters)
            task = progress.add_task("Cropping and saving faces...", total=total_faces)
            
            for cluster in self.clusters:
                # Create cluster directory
                cluster_dir_name = f"person_{cluster.cluster_id:02d}"
                cluster_dir = output_dir / cluster_dir_name
                cluster_dir.mkdir(exist_ok=True)
                cluster_dirs[f"person_{cluster.cluster_id:02d}"] = cluster_dir
                
                # Process faces in parallel
                with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    futures = []
                    
                    for face_idx, face in enumerate(cluster.faces):
                        future = executor.submit(
                            self._crop_and_save_face,
                            face,
                            cluster_dir,
                            face_idx
                        )
                        futures.append(future)
                    
                    # Wait for all faces to be processed
                    for future in as_completed(futures):
                        try:
                            future.result()
                            progress.advance(task)
                        except Exception as e:
                            logger.error(f"Error processing face: {e}")
                            progress.advance(task)
        
        console.print(f"✅ Saved faces to {len(cluster_dirs)} directories", style="green")
        return cluster_dirs
    
    def _crop_and_save_face(self, face: FaceData, cluster_dir: Path, face_idx: int) -> None:
        """
        Crop a single face from its image and save it.
        
        Args:
            face: Face data
            cluster_dir: Cluster output directory
            face_idx: Index of face in cluster
        """
        try:
            # Load the image
            image = cv2.imread(str(face.image_path))
            if image is None:
                logger.error(f"Could not load image: {face.image_path}")
                return
            
            # Calculate crop coordinates with padding
            img_height, img_width = image.shape[:2]
            
            # Use crop coordinates from JSON (already includes padding)
            x1 = max(0, face.crop_x)
            y1 = max(0, face.crop_y)
            x2 = min(img_width, face.crop_x + face.crop_width)
            y2 = min(img_height, face.crop_y + face.crop_height)
            
            # Crop the face
            cropped_face = image[y1:y2, x1:x2]
            
            if cropped_face.size == 0:
                logger.error(f"Empty crop for face in {face.image_path}")
                return
            
            # Generate output filename
            image_stem = face.image_path.stem
            face_filename = f"{image_stem}_face_{face_idx:03d}.jpg"
            output_path = cluster_dir / face_filename
            
            # Save the cropped face
            cv2.imwrite(str(output_path), cropped_face)
            
        except Exception as e:
            logger.error(f"Error cropping face from {face.image_path}: {e}")
    
    def generate_clustering_report(self, output_dir: Path) -> Path:
        """
        Generate a comprehensive clustering report.
        
        Args:
            output_dir: Output directory for the report
            
        Returns:
            Path to the generated report file
        """
        report_path = output_dir / "face_clustering_report.json"
        
        # Generate report data
        report_data = {
            'summary': {
                'total_faces': len(self.face_data),
                'total_clusters': len(self.clusters),
                'algorithm_used': self.algorithm,
                'padding_percentage': self.padding,
                'min_cluster_size': self.min_cluster_size,
                'max_clusters': self.max_clusters,
                'generated_at': time.strftime('%Y-%m-%d %H:%M:%S')
            },
            'clusters': [],
            'performance': self.performance_metrics
        }
        
        # Add cluster information
        for cluster in self.clusters:
            cluster_info = {
                'cluster_id': cluster.cluster_id,
                'face_count': cluster.face_count,
                'confidence_score': cluster.confidence_score,
                'faces': []
            }
            
            for face in cluster.faces:
                face_info = {
                    'image_path': str(face.image_path),
                    'face_id': face.face_id,
                    'coordinates': {
                        'x': face.x,
                        'y': face.y,
                        'width': face.width,
                        'height': face.height
                    },
                    'confidence': face.confidence,
                    'crop_area': {
                        'x': face.crop_x,
                        'y': face.crop_y,
                        'width': face.crop_width,
                        'height': face.crop_height
                    }
                }
                cluster_info['faces'].append(face_info)
            
            report_data['clusters'].append(cluster_info)
        
        # Write report to file
        with open(report_path, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        console.print(f"📊 Clustering report generated: {report_path}", style="green")
        return report_path
    
    def display_clustering_results(self):
        """Display clustering results in a beautiful table."""
        if not self.clusters:
            console.print("❌ No clustering results to display", style="red")
            return
        
        table = Table(title="Face Clustering Results")
        table.add_column("Cluster ID", style="cyan", no_wrap=True)
        table.add_column("Face Count", style="green", justify="right")
        table.add_column("Avg Confidence", style="yellow", justify="right")
        table.add_column("Sample Images", style="magenta")
        
        for cluster in self.clusters:
            # Get sample image names (first 3)
            sample_images = [face.image_path.name for face in cluster.faces[:3]]
            sample_text = ", ".join(sample_images)
            if len(cluster.faces) > 3:
                sample_text += f" (+{len(cluster.faces) - 3} more)"
            
            table.add_row(
                f"person_{cluster.cluster_id:02d}",
                str(cluster.face_count),
                f"{cluster.confidence_score:.3f}",
                sample_text
            )
        
        console.print(table)
    
    def display_performance_metrics(self):
        """Display performance metrics."""
        if not self.performance_metrics:
            return
        
        metrics_table = Table(title="Performance Metrics")
        metrics_table.add_column("Metric", style="cyan")
        metrics_table.add_column("Value", style="green")
        
        for metric, value in self.performance_metrics.items():
            metrics_table.add_row(metric.replace('_', ' ').title(), str(value))
        
        console.print(metrics_table)


@click.command()
@click.option('--input', '-i',
              type=click.Path(exists=True, file_okay=False, path_type=Path),
              required=True,
              help='Input directory containing JSON files with face data')
@click.option('--output', '-o',
              type=click.Path(path_type=Path),
              default=Path('./results/clustered_faces'),
              help='Output directory for organized face clusters')
@click.option('--pattern', '-p',
              default='*.json',
              help='File pattern to match JSON files (default: *.json)')
@click.option('--algorithm', '-a',
              type=click.Choice(['dbscan', 'kmeans', 'hierarchical']),
              default='dbscan',
              help='Clustering algorithm to use')
@click.option('--padding',
              type=float,
              default=0.25,
              help='Padding percentage for face crops (default: 0.25 = 25%)')
@click.option('--min-cluster-size',
              type=int,
              default=2,
              help='Minimum faces per cluster')
@click.option('--max-clusters',
              type=int,
              default=50,
              help='Maximum number of clusters (for K-means and Hierarchical)')
@click.option('--workers', '-w',
              type=int,
              default=4,
              help='Number of parallel workers')
@click.option('--verbose', '-v',
              is_flag=True,
              help='Enable verbose logging')
@click.option('--quiet', '-q',
              is_flag=True,
              help='Suppress output except errors')
@click.version_option(version='1.0.0')
def main(input: Path, output: Path, pattern: str, algorithm: str, padding: float,
         min_cluster_size: int, max_clusters: int, workers: int, verbose: bool, quiet: bool):
    """
    AI-Based Face Clustering Tool
    
    Reads face recognition JSON data and uses AI clustering algorithms to group
    similar faces into likely persons. Extracts and crops faces with configurable
    padding and saves them to organized directories.
    
    The tool expects JSON files created by the face detection system, containing
    face encodings and metadata. It will automatically find corresponding image
    files and crop faces based on the detection data.
    
    Examples:
    
    \b
    # Basic clustering with DBSCAN
    python face_clustering.py --input /path/to/json/files --output /path/to/output
    
    \b
    # Use K-means with custom parameters
    python face_clustering.py --input /path/to/json/files --algorithm kmeans --max-clusters 20
    
    \b
    # High performance with 8 workers
    python face_clustering.py --input /path/to/json/files --workers 8
    
    \b
    # Custom padding and cluster size
    python face_clustering.py --input /path/to/json/files --padding 0.3 --min-cluster-size 3
    """
    
    # Configure logging
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    elif quiet:
        logging.getLogger().setLevel(logging.ERROR)
    
    # Display header
    if not quiet:
        console.print(Panel.fit(
            "[bold blue]AI-Based Face Clustering Tool[/bold blue]\n"
            "Intelligent face grouping and organization",
            border_style="blue"
        ))
    
    try:
        start_time = time.time()
        
        # Find JSON files
        if not quiet:
            console.print(f"\n[bold]🔍 Finding JSON files...[/bold]")
            console.print(f"Input: {input}")
            console.print(f"Pattern: {pattern}")
        
        json_files = list(input.glob(pattern))
        json_files = [f for f in json_files if f.is_file() and f.suffix.lower() == '.json']
        
        if not json_files:
            console.print("❌ No JSON files found", style="red")
            return 1
        
        console.print(f"📄 Found {len(json_files)} JSON files", style="green")
        
        # Create clustering engine
        engine = FaceClusteringEngine(
            padding=padding,
            min_cluster_size=min_cluster_size,
            max_clusters=max_clusters,
            algorithm=algorithm,
            max_workers=workers
        )
        
        # Step 1: Load face data
        if not quiet:
            console.print(f"\n[bold]📊 Loading face data...[/bold]")
        
        face_data = engine.load_face_data_from_json(json_files)
        
        if not face_data:
            console.print("❌ No face data found in JSON files", style="red")
            return 1
        
        # Step 2: Extract encodings
        if not quiet:
            console.print(f"\n[bold]🧠 Extracting face encodings...[/bold]")
        
        encodings, valid_indices = engine.extract_face_encodings()
        
        # Step 3: Cluster faces
        if not quiet:
            console.print(f"\n[bold]🤖 Clustering faces...[/bold]")
        
        clusters = engine.cluster_faces(encodings, valid_indices)
        
        if not clusters:
            console.print("❌ No clusters created", style="red")
            return 1
        
        # Step 4: Crop and save faces
        if not quiet:
            console.print(f"\n[bold]✂️  Cropping and saving faces...[/bold]")
        
        cluster_dirs = engine.crop_and_save_faces(output)
        
        # Step 5: Generate report
        if not quiet:
            console.print(f"\n[bold]📊 Generating report...[/bold]")
        
        report_path = engine.generate_clustering_report(output)
        
        # Store performance metrics
        engine.performance_metrics = {
            'total_time': time.time() - start_time,
            'json_files_processed': len(json_files),
            'faces_clustered': len(face_data),
            'clusters_created': len(clusters),
            'algorithm_used': algorithm,
            'parallel_workers': workers
        }
        
        # Display results
        if not quiet:
            console.print(f"\n[bold green]✅ Face Clustering Complete![/bold green]")
            console.print(f"Output directory: {output}")
            console.print(f"Report: {report_path}")
            
            # Display results table
            engine.display_clustering_results()
            
            # Display performance metrics
            engine.display_performance_metrics()
        
        return 0
        
    except Exception as e:
        console.print(f"❌ Error during clustering: {e}", style="red")
        if verbose:
            import traceback
            console.print(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())
