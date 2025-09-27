"""
Photo Quality Assessment Module for Sportball

This module provides comprehensive photo quality assessment capabilities for sports photography,
including focus/sharpness detection, exposure analysis, composition evaluation, and more.

Author: Claude Sonnet 4 (claude-3-5-sonnet-20241022)
Generated via Cursor IDE (cursor.sh) with AI assistance
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any, Union
from dataclasses import dataclass
from loguru import logger
import math


@dataclass
class QualityMetrics:
    """Photo quality metrics."""
    overall_score: float  # Overall quality score (0-1)
    sharpness_score: float  # Sharpness/blur assessment (0-1)
    focus_score: float  # Focus quality using multiple algorithms (0-1)
    exposure_score: float  # Exposure quality (0-1)
    composition_score: float  # Composition quality (0-1)
    action_score: float  # Action timing quality (0-1)
    color_score: float  # Color quality (0-1)
    noise_score: float  # Noise level (0-1, higher is better)
    contrast_score: float  # Contrast quality (0-1)


@dataclass
class QualityAssessment:
    """Complete quality assessment result."""
    metrics: QualityMetrics
    recommendations: List[str]
    quality_grade: str  # A, B, C, D, F
    processing_time: float
    success: bool
    error: Optional[str] = None


class QualityAssessor:
    """
    Photo quality assessment system for sports photography.
    
    This class provides comprehensive quality assessment including:
    - Focus/sharpness detection using Laplacian variance
    - Exposure analysis
    - Composition evaluation
    - Action timing assessment
    - Color quality analysis
    - Noise detection
    - Contrast evaluation
    """
    
    def __init__(self, cache_enabled: bool = True):
        """
        Initialize the quality assessor.
        
        Args:
            cache_enabled: Whether to enable caching of results
        """
        self.cache_enabled = cache_enabled
        self.logger = logger
        
        # Quality thresholds (can be adjusted based on use case)
        self.thresholds = {
            'sharpness': {
                'excellent': 1000,
                'good': 500,
                'fair': 200,
                'poor': 100,
                'very_poor': 50
            },
            'exposure': {
                'optimal_min': 50,
                'optimal_max': 200,
                'acceptable_min': 30,
                'acceptable_max': 220
            },
            'contrast': {
                'excellent': 0.7,
                'good': 0.5,
                'fair': 0.3,
                'poor': 0.2
            }
        }
        
        self.logger.info("Initialized QualityAssessor")
    
    def assess_quality(self, 
                      image_path: Union[Path, str], 
                      **kwargs) -> Dict[str, Any]:
        """
        Assess photo quality from image file.
        
        Args:
            image_path: Path to the image file
            **kwargs: Additional assessment parameters
            
        Returns:
            Dictionary containing quality assessment results
        """
        image_path = Path(image_path)
        
        if not image_path.exists():
            return {
                "error": f"Image file not found: {image_path}",
                "success": False
            }
        
        try:
            # Load image
            image = cv2.imread(str(image_path))
            if image is None:
                return {
                    "error": f"Failed to load image: {image_path}",
                    "success": False
                }
            
            # Perform assessment
            assessment = self._assess_image_quality(image)
            
            # Convert to dictionary format for compatibility
            result = {
                "success": assessment.success,
                "quality": {
                    "overall_score": assessment.metrics.overall_score,
                    "sharpness": assessment.metrics.sharpness_score,
                    "focus": assessment.metrics.focus_score,
                    "exposure": assessment.metrics.exposure_score,
                    "composition": assessment.metrics.composition_score,
                    "action": assessment.metrics.action_score,
                    "color": assessment.metrics.color_score,
                    "noise": assessment.metrics.noise_score,
                    "contrast": assessment.metrics.contrast_score,
                    "grade": assessment.quality_grade,
                    "recommendations": assessment.recommendations
                },
                "processing_time": assessment.processing_time,
                "image_path": str(image_path),
                "image_width": image.shape[1],
                "image_height": image.shape[0]
            }
            
            if assessment.error:
                result["error"] = assessment.error
            
            return result
            
        except Exception as e:
            self.logger.error(f"Quality assessment failed for {image_path}: {e}")
            return {
                "error": str(e),
                "success": False,
                "image_path": str(image_path)
            }
    
    def _assess_image_quality(self, image: np.ndarray) -> QualityAssessment:
        """
        Assess image quality using multiple metrics.
        
        Args:
            image: Input image array
            
        Returns:
            QualityAssessment object with complete results
        """
        start_time = cv2.getTickCount()
        
        try:
            # Convert to grayscale for some analyses
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Assess individual quality metrics
            sharpness_score = self._assess_sharpness(image)
            focus_score = self._assess_focus(image)
            exposure_score = self._assess_exposure(image)
            composition_score = self._assess_composition(image)
            action_score = self._assess_action_timing(image)
            color_score = self._assess_color_quality(image)
            noise_score = self._assess_noise(image)
            contrast_score = self._assess_contrast(image)
            
            # Calculate overall score (weighted average)
            # Focus and sharpness are both important for image quality
            weights = {
                'sharpness': 0.15,
                'focus': 0.20,  # Focus is critical for sports photography
                'exposure': 0.20,
                'composition': 0.15,
                'action': 0.15,
                'color': 0.10,
                'noise': 0.05,
                'contrast': 0.05
            }
            
            overall_score = (
                sharpness_score * weights['sharpness'] +
                focus_score * weights['focus'] +
                exposure_score * weights['exposure'] +
                composition_score * weights['composition'] +
                action_score * weights['action'] +
                color_score * weights['color'] +
                noise_score * weights['noise'] +
                contrast_score * weights['contrast']
            )
            
            # Create metrics object
            metrics = QualityMetrics(
                overall_score=overall_score,
                sharpness_score=sharpness_score,
                focus_score=focus_score,
                exposure_score=exposure_score,
                composition_score=composition_score,
                action_score=action_score,
                color_score=color_score,
                noise_score=noise_score,
                contrast_score=contrast_score
            )
            
            # Generate recommendations
            recommendations = self._generate_recommendations(metrics)
            
            # Determine quality grade
            quality_grade = self._determine_quality_grade(overall_score)
            
            processing_time = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()
            
            return QualityAssessment(
                metrics=metrics,
                recommendations=recommendations,
                quality_grade=quality_grade,
                processing_time=processing_time,
                success=True
            )
            
        except Exception as e:
            self.logger.error(f"Quality assessment failed: {e}")
            processing_time = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()
            return QualityAssessment(
                metrics=QualityMetrics(0, 0, 0, 0, 0, 0, 0, 0),
                recommendations=[],
                quality_grade='F',
                processing_time=processing_time,
                success=False,
                error=str(e)
            )
    
    def _assess_sharpness(self, image: np.ndarray) -> float:
        """
        Assess image sharpness using Laplacian variance.
        
        Args:
            image: Input image array
            
        Returns:
            Sharpness score (0-1, higher is sharper)
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Calculate Laplacian variance (higher = sharper)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Normalize to 0-1 scale using thresholds
        thresholds = self.thresholds['sharpness']
        
        if laplacian_var >= thresholds['excellent']:
            return 1.0
        elif laplacian_var >= thresholds['good']:
            return 0.8
        elif laplacian_var >= thresholds['fair']:
            return 0.6
        elif laplacian_var >= thresholds['poor']:
            return 0.4
        elif laplacian_var >= thresholds['very_poor']:
            return 0.2
        else:
            return 0.0
    
    def _assess_focus(self, image: np.ndarray) -> float:
        """
        Assess image focus using multiple established algorithms.
        
        Implements several focus detection algorithms from computer vision literature:
        - Tenengrad (gradient-based)
        - Sobel variance
        - Brenner gradient
        - Laplacian variance (enhanced)
        - Variance of Laplacian
        - Modified Laplacian
        
        Args:
            image: Input image array
            
        Returns:
            Focus score (0-1, higher is more focused)
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Normalize to 0-1 range
        gray = gray.astype(np.float64) / 255.0
        
        # 1. Tenengrad Algorithm (Tenenbaum, 1970)
        # Uses gradient magnitude to measure focus
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        tenengrad_score = np.sum(grad_x**2 + grad_y**2)
        
        # 2. Sobel Variance
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        sobel_variance = np.var(sobel_magnitude)
        
        # 3. Brenner Gradient
        # Measures focus by calculating sum of squared differences
        brenner_score = np.sum((gray[:-2, :] - gray[2:, :])**2)
        
        # 4. Enhanced Laplacian Variance
        laplacian = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
        laplacian_variance = np.var(laplacian)
        
        # 5. Variance of Laplacian (Pech-Pacheco et al., 2000)
        laplacian_mean = np.mean(laplacian)
        variance_of_laplacian = np.mean((laplacian - laplacian_mean)**2)
        
        # 6. Modified Laplacian (Nayar & Nakagawa, 1994)
        # Uses second derivatives in both directions
        mod_laplacian_x = np.abs(cv2.Laplacian(gray, cv2.CV_64F, ksize=1))
        mod_laplacian_y = np.abs(cv2.Laplacian(gray.T, cv2.CV_64F, ksize=1).T)
        modified_laplacian = np.sum(mod_laplacian_x + mod_laplacian_y)
        
        # Normalize scores to 0-1 range using adaptive thresholds
        # These thresholds are based on typical image sizes and characteristics
        height, width = gray.shape
        image_size_factor = height * width / (1000 * 1000)  # Normalize for image size
        
        # Normalize each algorithm score
        scores = {
            'tenengrad': min(1.0, tenengrad_score / (10000 * image_size_factor)),
            'sobel_var': min(1.0, sobel_variance / (0.1 * image_size_factor)),
            'brenner': min(1.0, brenner_score / (5000 * image_size_factor)),
            'laplacian_var': min(1.0, laplacian_variance / (0.1 * image_size_factor)),
            'var_of_laplacian': min(1.0, variance_of_laplacian / (0.01 * image_size_factor)),
            'modified_laplacian': min(1.0, modified_laplacian / (1000 * image_size_factor))
        }
        
        # Weighted combination of algorithms
        # Tenengrad and Sobel variance are generally most reliable
        weights = {
            'tenengrad': 0.25,
            'sobel_var': 0.25,
            'brenner': 0.15,
            'laplacian_var': 0.15,
            'var_of_laplacian': 0.10,
            'modified_laplacian': 0.10
        }
        
        # Calculate weighted average
        focus_score = sum(scores[algo] * weights[algo] for algo in scores)
        
        # Apply sigmoid-like function to map to 0-1 range more smoothly
        focus_score = 1 / (1 + np.exp(-10 * (focus_score - 0.5)))
        
        return focus_score
    
    def _assess_exposure(self, image: np.ndarray) -> float:
        """
        Assess image exposure quality.
        
        Args:
            image: Input image array
            
        Returns:
            Exposure score (0-1, higher is better)
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Calculate mean brightness
        mean_brightness = np.mean(gray)
        
        # Assess exposure using thresholds
        thresholds = self.thresholds['exposure']
        
        if thresholds['optimal_min'] <= mean_brightness <= thresholds['optimal_max']:
            return 1.0
        elif thresholds['acceptable_min'] <= mean_brightness <= thresholds['acceptable_max']:
            return 0.8
        else:
            # Calculate penalty for extreme values
            if mean_brightness < thresholds['acceptable_min']:
                penalty = (thresholds['acceptable_min'] - mean_brightness) / thresholds['acceptable_min']
            else:
                penalty = (mean_brightness - thresholds['acceptable_max']) / (255 - thresholds['acceptable_max'])
            
            return max(0.0, 0.4 - penalty * 0.4)
    
    def _assess_composition(self, image: np.ndarray) -> float:
        """
        Assess image composition quality using rule of thirds.
        
        Args:
            image: Input image array
            
        Returns:
            Composition score (0-1, higher is better)
        """
        height, width = image.shape[:2]
        
        # Define rule of thirds lines
        third_width = width / 3
        third_height = height / 3
        
        # Convert to grayscale for edge detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect edges
        edges = cv2.Canny(gray, 50, 150)
        
        # Check for interesting elements along rule of thirds lines
        score = 0.0
        
        # Horizontal lines
        for line_y in [third_height, 2 * third_height]:
            y_start = max(0, int(line_y - 10))
            y_end = min(height, int(line_y + 10))
            line_edges = edges[y_start:y_end, :]
            edge_density = np.sum(line_edges > 0) / (line_edges.size + 1e-6)
            score += min(0.5, edge_density * 10)
        
        # Vertical lines
        for line_x in [third_width, 2 * third_width]:
            x_start = max(0, int(line_x - 10))
            x_end = min(width, int(line_x + 10))
            line_edges = edges[:, x_start:x_end]
            edge_density = np.sum(line_edges > 0) / (line_edges.size + 1e-6)
            score += min(0.5, edge_density * 10)
        
        return min(1.0, score)
    
    def _assess_action_timing(self, image: np.ndarray) -> float:
        """
        Assess action timing quality (motion blur detection).
        
        Args:
            image: Input image array
            
        Returns:
            Action timing score (0-1, higher is better)
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect motion blur using gradient analysis
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # Calculate gradient magnitude
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # High gradient variance indicates good action timing (sharp motion)
        gradient_variance = np.var(gradient_magnitude)
        
        # Normalize score
        if gradient_variance > 10000:
            return 1.0
        elif gradient_variance > 5000:
            return 0.8
        elif gradient_variance > 2000:
            return 0.6
        elif gradient_variance > 1000:
            return 0.4
        else:
            return 0.2
    
    def _assess_color_quality(self, image: np.ndarray) -> float:
        """
        Assess color quality and saturation.
        
        Args:
            image: Input image array
            
        Returns:
            Color quality score (0-1, higher is better)
        """
        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Calculate mean saturation
        mean_saturation = np.mean(hsv[:, :, 1])
        
        # Calculate color diversity (number of distinct colors)
        # Reshape image to list of pixels
        pixels = image.reshape(-1, 3)
        
        # Count unique colors (simplified)
        unique_colors = len(np.unique(pixels.view(np.void), axis=0))
        color_diversity = min(1.0, unique_colors / 10000)  # Normalize
        
        # Combine saturation and diversity
        saturation_score = min(1.0, mean_saturation / 128)  # Normalize to 0-1
        overall_score = (saturation_score + color_diversity) / 2
        
        return overall_score
    
    def _assess_noise(self, image: np.ndarray) -> float:
        """
        Assess image noise level.
        
        Args:
            image: Input image array
            
        Returns:
            Noise score (0-1, higher is better, i.e., less noise)
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Calculate noise using Laplacian (high frequency content)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        noise_level = np.var(laplacian)
        
        # Normalize (lower noise = higher score)
        if noise_level < 100:
            return 1.0
        elif noise_level < 500:
            return 0.8
        elif noise_level < 1000:
            return 0.6
        elif noise_level < 2000:
            return 0.4
        else:
            return 0.2
    
    def _assess_contrast(self, image: np.ndarray) -> float:
        """
        Assess image contrast quality.
        
        Args:
            image: Input image array
            
        Returns:
            Contrast score (0-1, higher is better)
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Calculate standard deviation as contrast measure
        contrast = np.std(gray) / 255.0
        
        # Normalize using thresholds
        thresholds = self.thresholds['contrast']
        
        if contrast >= thresholds['excellent']:
            return 1.0
        elif contrast >= thresholds['good']:
            return 0.8
        elif contrast >= thresholds['fair']:
            return 0.6
        elif contrast >= thresholds['poor']:
            return 0.4
        else:
            return 0.2
    
    def _generate_recommendations(self, metrics: QualityMetrics) -> List[str]:
        """
        Generate improvement recommendations based on quality metrics.
        
        Args:
            metrics: Quality metrics object
            
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        if metrics.sharpness_score < 0.5:
            recommendations.append("Consider using faster shutter speed or better focus technique")
        
        if metrics.focus_score < 0.5:
            recommendations.append("Image appears out of focus - check autofocus settings or manual focus")
        
        if metrics.exposure_score < 0.5:
            recommendations.append("Adjust exposure settings - image may be over/under exposed")
        
        if metrics.composition_score < 0.5:
            recommendations.append("Consider applying rule of thirds for better composition")
        
        if metrics.action_score < 0.5:
            recommendations.append("Use faster shutter speed to freeze action")
        
        if metrics.color_score < 0.5:
            recommendations.append("Consider adjusting saturation or white balance")
        
        if metrics.noise_score < 0.5:
            recommendations.append("Reduce ISO or use noise reduction techniques")
        
        if metrics.contrast_score < 0.5:
            recommendations.append("Increase contrast in post-processing")
        
        if not recommendations:
            recommendations.append("Image quality is good - no major improvements needed")
        
        return recommendations
    
    def _determine_quality_grade(self, overall_score: float) -> str:
        """
        Determine quality grade based on overall score.
        
        Args:
            overall_score: Overall quality score (0-1)
            
        Returns:
            Quality grade (A, B, C, D, F)
        """
        if overall_score >= 0.85:
            return "A"
        elif overall_score >= 0.70:
            return "B"
        elif overall_score >= 0.55:
            return "C"
        elif overall_score >= 0.40:
            return "D"
        else:
            return "F"
