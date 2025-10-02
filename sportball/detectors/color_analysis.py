"""
Color Analysis Module

Comprehensive color analysis for jersey color detection using computer vision
techniques including clustering algorithms and multiple color spaces.

Author: Claude Sonnet 4 (claude-3-5-sonnet-20241022)
Generated via Cursor IDE (cursor.sh) with AI assistance
"""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple, Union
import numpy as np
from PIL import Image
from loguru import logger

# Try to import OpenCV for image processing
try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False
    logger.warning("opencv-python not available - color analysis will be limited")

# Try to import scikit-learn for clustering
try:
    from sklearn.cluster import KMeans
    from sklearn.metrics import pairwise_distances_argmin_min
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn not available - K-means clustering will be skipped")


@dataclass
class ColorCluster:
    """Represents a color cluster."""
    
    rgb_color: Tuple[int, int, int]
    hsv_color: Tuple[float, float, float]
    lab_color: Tuple[float, float, float]
    percentage: float
    pixel_count: int
    cluster_id: int


@dataclass
class JerseyColorAnalysis:
    """Result of jersey color analysis."""
    
    dominant_colors: List[ColorCluster]
    jersey_colors: List[ColorCluster]  # Filtered jersey-specific colors
    background_colors: List[ColorCluster]  # Filtered background colors
    confidence: float
    color_space_used: str
    processing_time: float


@dataclass
class ColorAnalysisResult:
    """Result of color analysis on an image."""
    
    success: bool
    jersey_analysis: Optional[JerseyColorAnalysis]
    error: Optional[str] = None
    image_path: Optional[Path] = None


class ColorAnalyzer:
    """
    Color analysis engine for jersey color detection.
    
    This class provides comprehensive color analysis capabilities using computer
    vision techniques including clustering algorithms and multiple color spaces
    for robust jersey color identification.
    """
    
    def __init__(
        self,
        n_clusters: int = 8,
        color_space: str = "rgb",
        similarity_threshold: float = 0.15,
        background_filter_threshold: float = 0.3,
        cache_enabled: bool = True,
    ):
        """
        Initialize the ColorAnalyzer.
        
        Args:
            n_clusters: Number of clusters for K-means clustering
            color_space: Primary color space for analysis ('rgb', 'hsv', 'lab')
            similarity_threshold: Threshold for grouping similar colors
            background_filter_threshold: Threshold for filtering background colors
            cache_enabled: Whether to enable result caching
        """
        self.n_clusters = n_clusters
        self.color_space = color_space
        self.similarity_threshold = similarity_threshold
        self.background_filter_threshold = background_filter_threshold
        self.cache_enabled = cache_enabled
        
        self.logger = logger.bind(component="color_analyzer")
        
        # Validate dependencies
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for color clustering")
        
        if not OPENCV_AVAILABLE:
            self.logger.warning("OpenCV not available - limited color space support")
        
        self.logger.info("Initialized ColorAnalyzer")
    
    def analyze_jersey_colors(
        self,
        image_paths: Union[Path, List[Path]],
        pose_results: Optional[Dict[str, Any]] = None,
        save_sidecar: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Analyze jersey colors in images.
        
        Args:
            image_paths: Single image path or list of image paths
            pose_results: Optional pose detection results for region extraction
            save_sidecar: Whether to save results to sidecar files
            **kwargs: Additional arguments for color analysis
            
        Returns:
            Dictionary containing color analysis results
        """
        if isinstance(image_paths, Path):
            image_paths = [image_paths]
        
        self.logger.info(f"Analyzing jersey colors in {len(image_paths)} images")
        
        results = {}
        
        for image_path in image_paths:
            try:
                # Check cache first
                if self.cache_enabled:
                    cached_data = self._load_cached_result(image_path)
                    if cached_data:
                        results[str(image_path)] = cached_data
                        continue
                
                # Get pose results for this image if available
                pose_result = None
                if pose_results and str(image_path) in pose_results:
                    pose_result = pose_results[str(image_path)]
                
                # Perform color analysis
                color_result = self._analyze_image_colors(image_path, pose_result)
                
                # Save to sidecar if requested
                if save_sidecar and color_result.success:
                    self._save_to_sidecar(image_path, color_result)
                
                results[str(image_path)] = color_result.as_dict()
                
            except Exception as e:
                self.logger.error(f"Color analysis failed for {image_path}: {e}")
                results[str(image_path)] = {
                    "success": False,
                    "error": str(e),
                    "jersey_analysis": None
                }
        
        return results
    
    def _analyze_image_colors(
        self, 
        image_path: Path, 
        pose_result: Optional[Dict[str, Any]] = None
    ) -> ColorAnalysisResult:
        """Analyze colors in a single image."""
        start_time = time.perf_counter()
        
        try:
            # Load image
            image = Image.open(image_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            image_array = np.array(image)
            
            # Extract jersey regions if pose data is available
            jersey_regions = []
            if pose_result and pose_result.get("success", False):
                jersey_regions = self._extract_jersey_regions_from_pose(
                    image_array, pose_result
                )
            
            # If no pose data or no regions found, use full image
            if not jersey_regions:
                jersey_regions = [image_array]
            
            # Analyze colors in jersey regions
            jersey_analysis = self._analyze_regions_colors(jersey_regions)
            
            processing_time = time.perf_counter() - start_time
            
            return ColorAnalysisResult(
                success=True,
                jersey_analysis=jersey_analysis,
                image_path=image_path
            )
            
        except Exception as e:
            processing_time = time.perf_counter() - start_time
            return ColorAnalysisResult(
                success=False,
                jersey_analysis=None,
                error=str(e),
                image_path=image_path
            )
    
    def _extract_jersey_regions_from_pose(
        self, 
        image_array: np.ndarray, 
        pose_result: Dict[str, Any]
    ) -> List[np.ndarray]:
        """Extract jersey regions based on pose detection."""
        jersey_regions = []
        
        poses = pose_result.get("poses", [])
        for pose in poses:
            upper_body_bbox = pose.get("upper_body_bbox")
            if upper_body_bbox:
                x, y, w, h = upper_body_bbox
                
                # Extract region
                region = image_array[y:y+h, x:x+w]
                
                if region.size > 0:
                    jersey_regions.append(region)
        
        return jersey_regions
    
    def _analyze_regions_colors(self, regions: List[np.ndarray]) -> JerseyColorAnalysis:
        """Analyze colors in jersey regions."""
        start_time = time.perf_counter()
        
        # Combine all regions
        if len(regions) == 1:
            combined_image = regions[0]
        else:
            # Concatenate regions vertically
            combined_image = np.vstack(regions)
        
        # Reshape image for clustering
        pixels = combined_image.reshape(-1, 3)
        
        # Convert to different color spaces
        rgb_pixels = pixels.astype(np.float32)
        
        if OPENCV_AVAILABLE:
            # Convert to HSV
            hsv_image = cv2.cvtColor(combined_image, cv2.COLOR_RGB2HSV)
            hsv_pixels = hsv_image.reshape(-1, 3).astype(np.float32)
            
            # Convert to LAB
            lab_image = cv2.cvtColor(combined_image, cv2.COLOR_RGB2LAB)
            lab_pixels = lab_image.reshape(-1, 3).astype(np.float32)
        else:
            # Fallback without OpenCV
            hsv_pixels = self._rgb_to_hsv_fallback(rgb_pixels)
            lab_pixels = self._rgb_to_lab_fallback(rgb_pixels)
        
        # Choose color space for clustering
        if self.color_space == "hsv":
            clustering_pixels = hsv_pixels
        elif self.color_space == "lab":
            clustering_pixels = lab_pixels
        else:  # rgb
            clustering_pixels = rgb_pixels
        
        # Perform K-means clustering
        kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=42,
            n_init=10
        )
        
        cluster_labels = kmeans.fit_predict(clustering_pixels)
        cluster_centers = kmeans.cluster_centers_
        
        # Calculate cluster statistics
        clusters = []
        total_pixels = len(pixels)
        
        for i, center in enumerate(cluster_centers):
            cluster_mask = cluster_labels == i
            pixel_count = np.sum(cluster_mask)
            percentage = pixel_count / total_pixels
            
            # Convert center back to RGB
            if self.color_space == "hsv":
                rgb_center = self._hsv_to_rgb(center)
                hsv_center = center
                lab_center = self._rgb_to_lab_fallback(rgb_center.reshape(1, -1))[0]
            elif self.color_space == "lab":
                rgb_center = self._lab_to_rgb_fallback(center.reshape(1, -1))[0]
                hsv_center = self._rgb_to_hsv_fallback(rgb_center.reshape(1, -1))[0]
                lab_center = center
            else:  # rgb
                rgb_center = center
                hsv_center = self._rgb_to_hsv_fallback(center.reshape(1, -1))[0]
                lab_center = self._rgb_to_lab_fallback(center.reshape(1, -1))[0]
            
            cluster = ColorCluster(
                rgb_color=tuple(int(c) for c in rgb_center),
                hsv_color=tuple(float(c) for c in hsv_center),
                lab_color=tuple(float(c) for c in lab_center),
                percentage=percentage,
                pixel_count=pixel_count,
                cluster_id=i
            )
            clusters.append(cluster)
        
        # Sort clusters by percentage (dominant colors first)
        clusters.sort(key=lambda x: x.percentage, reverse=True)
        
        # Filter jersey colors vs background colors
        jersey_colors, background_colors = self._filter_jersey_colors(clusters)
        
        # Calculate confidence based on color separation
        confidence = self._calculate_color_confidence(jersey_colors, background_colors)
        
        processing_time = time.perf_counter() - start_time
        
        return JerseyColorAnalysis(
            dominant_colors=clusters,
            jersey_colors=jersey_colors,
            background_colors=background_colors,
            confidence=confidence,
            color_space_used=self.color_space,
            processing_time=processing_time
        )
    
    def _filter_jersey_colors(
        self, 
        clusters: List[ColorCluster]
    ) -> Tuple[List[ColorCluster], List[ColorCluster]]:
        """Filter clusters into jersey colors and background colors."""
        jersey_colors = []
        background_colors = []
        
        for cluster in clusters:
            # Simple heuristic: colors with high saturation are more likely to be jerseys
            hsv = cluster.hsv_color
            saturation = hsv[1]  # Saturation component
            
            # Colors with low saturation are likely background
            if saturation < self.background_filter_threshold:
                background_colors.append(cluster)
            else:
                jersey_colors.append(cluster)
        
        return jersey_colors, background_colors
    
    def _calculate_color_confidence(
        self, 
        jersey_colors: List[ColorCluster], 
        background_colors: List[ColorCluster]
    ) -> float:
        """Calculate confidence in color analysis."""
        if not jersey_colors:
            return 0.0
        
        # Confidence based on:
        # 1. Number of distinct jersey colors
        # 2. Separation from background colors
        # 3. Dominance of jersey colors
        
        jersey_percentage = sum(c.percentage for c in jersey_colors)
        background_percentage = sum(c.percentage for c in background_colors)
        
        # Base confidence from jersey color dominance
        dominance_confidence = jersey_percentage / (jersey_percentage + background_percentage)
        
        # Bonus for having multiple distinct jersey colors
        distinct_colors_bonus = min(len(jersey_colors) / 3.0, 1.0)  # Cap at 1.0
        
        # Penalty for too many colors (might indicate noise)
        if len(jersey_colors) > 5:
            distinct_colors_bonus *= 0.8
        
        confidence = (dominance_confidence * 0.7) + (distinct_colors_bonus * 0.3)
        
        return min(confidence, 1.0)
    
    def group_similar_colors(self, colors: List[ColorCluster]) -> List[List[ColorCluster]]:
        """Group similar colors together."""
        if len(colors) <= 1:
            return [colors]
        
        groups = []
        used = set()
        
        for i, color1 in enumerate(colors):
            if i in used:
                continue
            
            group = [color1]
            used.add(i)
            
            for j, color2 in enumerate(colors[i+1:], i+1):
                if j in used:
                    continue
                
                # Calculate color similarity
                similarity = self._calculate_color_similarity(color1, color2)
                
                if similarity < self.similarity_threshold:
                    group.append(color2)
                    used.add(j)
            
            groups.append(group)
        
        return groups
    
    def _calculate_color_similarity(self, color1: ColorCluster, color2: ColorCluster) -> float:
        """Calculate similarity between two colors using LAB color space."""
        # Use LAB color space for perceptual similarity
        lab1 = np.array(color1.lab_color)
        lab2 = np.array(color2.lab_color)
        
        # Euclidean distance in LAB space
        distance = np.linalg.norm(lab1 - lab2)
        
        # Normalize to 0-1 range (approximate)
        normalized_distance = distance / 100.0
        
        return normalized_distance
    
    def _rgb_to_hsv_fallback(self, rgb_pixels: np.ndarray) -> np.ndarray:
        """Fallback RGB to HSV conversion without OpenCV."""
        # Simple RGB to HSV conversion
        rgb_normalized = rgb_pixels / 255.0
        
        hsv_pixels = np.zeros_like(rgb_normalized)
        
        for i, pixel in enumerate(rgb_normalized):
            r, g, b = pixel
            
            max_val = max(r, g, b)
            min_val = min(r, g, b)
            delta = max_val - min_val
            
            # Value
            hsv_pixels[i, 2] = max_val
            
            # Saturation
            if max_val == 0:
                hsv_pixels[i, 1] = 0
            else:
                hsv_pixels[i, 1] = delta / max_val
            
            # Hue
            if delta == 0:
                hsv_pixels[i, 0] = 0
            elif max_val == r:
                hsv_pixels[i, 0] = ((g - b) / delta) % 6
            elif max_val == g:
                hsv_pixels[i, 0] = (b - r) / delta + 2
            else:
                hsv_pixels[i, 0] = (r - g) / delta + 4
            
            hsv_pixels[i, 0] *= 60  # Convert to degrees
        
        return hsv_pixels * 255  # Scale back to 0-255
    
    def _rgb_to_lab_fallback(self, rgb_pixels: np.ndarray) -> np.ndarray:
        """Fallback RGB to LAB conversion without OpenCV."""
        # Simplified RGB to LAB conversion
        # This is a basic implementation - for production use, consider using
        # a proper color space conversion library
        
        rgb_normalized = rgb_pixels / 255.0
        
        # Simple approximation to LAB
        lab_pixels = np.zeros_like(rgb_normalized)
        
        for i, pixel in enumerate(rgb_normalized):
            r, g, b = pixel
            
            # Simple linear transformation approximation
            lab_pixels[i, 0] = 0.299 * r + 0.587 * g + 0.114 * b  # L
            lab_pixels[i, 1] = 0.5 * (r - g)  # A
            lab_pixels[i, 2] = 0.25 * (r + g - 2 * b)  # B
        
        return lab_pixels * 255
    
    def _hsv_to_rgb(self, hsv_pixel: np.ndarray) -> np.ndarray:
        """Convert HSV pixel to RGB."""
        h, s, v = hsv_pixel
        
        h = h / 60.0
        i = int(h)
        f = h - i
        
        p = v * (1 - s)
        q = v * (1 - s * f)
        t = v * (1 - s * (1 - f))
        
        if i == 0:
            return np.array([v, t, p])
        elif i == 1:
            return np.array([q, v, p])
        elif i == 2:
            return np.array([p, v, t])
        elif i == 3:
            return np.array([p, q, v])
        elif i == 4:
            return np.array([t, p, v])
        else:
            return np.array([v, p, q])
    
    def _lab_to_rgb_fallback(self, lab_pixels: np.ndarray) -> np.ndarray:
        """Fallback LAB to RGB conversion without OpenCV."""
        # Simplified LAB to RGB conversion
        lab_normalized = lab_pixels / 255.0
        
        rgb_pixels = np.zeros_like(lab_normalized)
        
        for i, pixel in enumerate(lab_normalized):
            l, a, b = pixel
            
            # Simple inverse transformation
            r = l + 0.5 * a
            g = l - 0.25 * a + 0.25 * b
            blue = l - 0.25 * b
            
            rgb_pixels[i] = [r, g, blue]
        
        # Clamp values
        rgb_pixels = np.clip(rgb_pixels, 0, 1)
        
        return rgb_pixels * 255
    
    def _load_cached_result(self, image_path: Path) -> Optional[Dict[str, Any]]:
        """Load cached color analysis result."""
        # This would integrate with the sidecar system
        # For now, return None to always perform analysis
        return None
    
    def _save_to_sidecar(self, image_path: Path, result: ColorAnalysisResult):
        """Save color analysis result to sidecar file."""
        # This would integrate with the sidecar system
        # Implementation would go here
        pass


# Add as_dict method to ColorAnalysisResult
def as_dict(self) -> Dict[str, Any]:
    """Convert ColorAnalysisResult to dictionary."""
    result = {
        "success": self.success,
        "error": self.error,
        "image_path": str(self.image_path) if self.image_path else None
    }
    
    if self.jersey_analysis:
        result["jersey_analysis"] = {
            "dominant_colors": [
                {
                    "rgb_color": cluster.rgb_color,
                    "hsv_color": cluster.hsv_color,
                    "lab_color": cluster.lab_color,
                    "percentage": cluster.percentage,
                    "pixel_count": cluster.pixel_count,
                    "cluster_id": cluster.cluster_id
                }
                for cluster in self.jersey_analysis.dominant_colors
            ],
            "jersey_colors": [
                {
                    "rgb_color": cluster.rgb_color,
                    "hsv_color": cluster.hsv_color,
                    "lab_color": cluster.lab_color,
                    "percentage": cluster.percentage,
                    "pixel_count": cluster.pixel_count,
                    "cluster_id": cluster.cluster_id
                }
                for cluster in self.jersey_analysis.jersey_colors
            ],
            "background_colors": [
                {
                    "rgb_color": cluster.rgb_color,
                    "hsv_color": cluster.hsv_color,
                    "lab_color": cluster.lab_color,
                    "percentage": cluster.percentage,
                    "pixel_count": cluster.pixel_count,
                    "cluster_id": cluster.cluster_id
                }
                for cluster in self.jersey_analysis.background_colors
            ],
            "confidence": self.jersey_analysis.confidence,
            "color_space_used": self.jersey_analysis.color_space_used,
            "processing_time": self.jersey_analysis.processing_time
        }
    else:
        result["jersey_analysis"] = None
    
    return result


# Add the method to ColorAnalysisResult
ColorAnalysisResult.as_dict = as_dict
