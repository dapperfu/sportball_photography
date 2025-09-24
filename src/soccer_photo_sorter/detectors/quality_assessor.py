"""
Photo Quality Assessment Module

This module provides photo quality assessment capabilities for sports photography,
evaluating blur, composition, exposure, and action timing.

Author: Claude Sonnet 4 (claude-3-5-sonnet-20241022)
Generated via Cursor IDE (cursor.sh) with AI assistance
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from loguru import logger
import math


@dataclass
class QualityMetrics:
    """Photo quality metrics."""
    overall_score: float  # Overall quality score (0-1)
    sharpness_score: float  # Sharpness/blur assessment (0-1)
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
    quality_grade: str  # 'A', 'B', 'C', 'D', 'F'
    processing_time: float
    success: bool
    error: Optional[str] = None


class QualityAssessor:
    """Photo quality assessment system for sports photography."""
    
    def __init__(self, 
                 quality_thresholds: Optional[Dict[str, float]] = None):
        """
        Initialize the quality assessor.
        
        Args:
            quality_thresholds: Custom quality thresholds
        """
        # Default quality thresholds
        self.thresholds = quality_thresholds or {
            'excellent_sharpness': 0.8,
            'good_sharpness': 0.6,
            'fair_sharpness': 0.4,
            'excellent_exposure': 0.8,
            'good_exposure': 0.6,
            'fair_exposure': 0.4,
            'excellent_composition': 0.8,
            'good_composition': 0.6,
            'fair_composition': 0.4,
            'excellent_action': 0.8,
            'good_action': 0.6,
            'fair_action': 0.4
        }
        
        # Quality grade thresholds
        self.grade_thresholds = {
            'A': 0.85,  # Excellent
            'B': 0.70,  # Good
            'C': 0.55,  # Fair
            'D': 0.40,  # Poor
            'F': 0.0    # Very Poor
        }
        
        logger.info("Initialized QualityAssessor")
    
    def assess_photo_quality(self, image: np.ndarray) -> QualityAssessment:
        """
        Assess the quality of a sports photo.
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            Quality assessment result
        """
        start_time = cv2.getTickCount()
        
        try:
            # Calculate individual quality metrics
            sharpness_score = self._assess_sharpness(image)
            exposure_score = self._assess_exposure(image)
            composition_score = self._assess_composition(image)
            action_score = self._assess_action_timing(image)
            color_score = self._assess_color_quality(image)
            noise_score = self._assess_noise_level(image)
            contrast_score = self._assess_contrast(image)
            
            # Calculate overall score (weighted average)
            overall_score = self._calculate_overall_score(
                sharpness_score, exposure_score, composition_score,
                action_score, color_score, noise_score, contrast_score
            )
            
            # Create quality metrics
            metrics = QualityMetrics(
                overall_score=overall_score,
                sharpness_score=sharpness_score,
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
            logger.error(f"Quality assessment failed: {e}")
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
        """Assess image sharpness using Laplacian variance."""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Calculate Laplacian variance (higher = sharper)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Normalize to 0-1 scale (empirical thresholds)
        # These thresholds may need adjustment based on your specific use case
        if laplacian_var > 1000:
            return 1.0
        elif laplacian_var > 500:
            return 0.8
        elif laplacian_var > 200:
            return 0.6
        elif laplacian_var > 100:
            return 0.4
        elif laplacian_var > 50:
            return 0.2
        else:
            return 0.0
    
    def _assess_exposure(self, image: np.ndarray) -> float:
        """Assess image exposure quality."""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Calculate histogram
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        
        # Calculate exposure metrics
        mean_brightness = np.mean(gray)
        
        # Check for overexposure (too many bright pixels)
        overexposed_pixels = np.sum(hist[200:256])
        overexposure_ratio = overexposed_pixels / (image.shape[0] * image.shape[1])
        
        # Check for underexposure (too many dark pixels)
        underexposed_pixels = np.sum(hist[0:50])
        underexposure_ratio = underexposed_pixels / (image.shape[0] * image.shape[1])
        
        # Calculate exposure score
        exposure_score = 1.0
        
        # Penalize overexposure
        if overexposure_ratio > 0.1:  # More than 10% overexposed
            exposure_score -= overexposure_ratio * 2
        
        # Penalize underexposure
        if underexposure_ratio > 0.1:  # More than 10% underexposed
            exposure_score -= underexposure_ratio * 2
        
        # Check if brightness is in good range
        if mean_brightness < 50 or mean_brightness > 200:
            exposure_score -= 0.3
        
        return max(0.0, min(1.0, exposure_score))
    
    def _assess_composition(self, image: np.ndarray) -> float:
        """Assess image composition quality."""
        height, width = image.shape[:2]
        
        # Rule of thirds assessment
        rule_of_thirds_score = self._assess_rule_of_thirds(image)
        
        # Symmetry assessment
        symmetry_score = self._assess_symmetry(image)
        
        # Balance assessment
        balance_score = self._assess_balance(image)
        
        # Leading lines assessment
        leading_lines_score = self._assess_leading_lines(image)
        
        # Combine composition scores
        composition_score = (
            rule_of_thirds_score * 0.4 +
            symmetry_score * 0.2 +
            balance_score * 0.2 +
            leading_lines_score * 0.2
        )
        
        return composition_score
    
    def _assess_rule_of_thirds(self, image: np.ndarray) -> float:
        """Assess rule of thirds compliance."""
        height, width = image.shape[:2]
        
        # Define rule of thirds lines
        vertical_lines = [width // 3, 2 * width // 3]
        horizontal_lines = [height // 3, 2 * height // 3]
        
        # Convert to grayscale for edge detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # Check for strong edges near rule of thirds lines
        score = 0.0
        
        # Check vertical lines
        for x in vertical_lines:
            line_region = edges[:, max(0, x-10):min(width, x+10)]
            edge_density = np.sum(line_region > 0) / line_region.size
            score += edge_density
        
        # Check horizontal lines
        for y in horizontal_lines:
            line_region = edges[max(0, y-10):min(height, y+10), :]
            edge_density = np.sum(line_region > 0) / line_region.size
            score += edge_density
        
        return min(1.0, score / 4.0)  # Normalize by number of lines
    
    def _assess_symmetry(self, image: np.ndarray) -> float:
        """Assess image symmetry."""
        height, width = image.shape[:2]
        
        # Check horizontal symmetry
        top_half = image[:height//2, :]
        bottom_half = cv2.flip(image[height//2:, :], 0)
        
        # Resize to same dimensions
        if top_half.shape[0] != bottom_half.shape[0]:
            min_height = min(top_half.shape[0], bottom_half.shape[0])
            top_half = top_half[:min_height, :]
            bottom_half = bottom_half[:min_height, :]
        
        # Calculate similarity
        diff = cv2.absdiff(top_half, bottom_half)
        similarity = 1.0 - (np.mean(diff) / 255.0)
        
        return max(0.0, similarity)
    
    def _assess_balance(self, image: np.ndarray) -> float:
        """Assess image balance."""
        height, width = image.shape[:2]
        
        # Divide image into quadrants
        top_left = image[:height//2, :width//2]
        top_right = image[:height//2, width//2:]
        bottom_left = image[height//2:, :width//2]
        bottom_right = image[height//2:, width//2:]
        
        # Calculate average brightness for each quadrant
        quadrants = [top_left, top_right, bottom_left, bottom_right]
        brightness_values = [np.mean(cv2.cvtColor(q, cv2.COLOR_BGR2GRAY)) for q in quadrants]
        
        # Calculate balance (lower variance = better balance)
        variance = np.var(brightness_values)
        balance_score = max(0.0, 1.0 - variance / 10000.0)  # Normalize variance
        
        return balance_score
    
    def _assess_leading_lines(self, image: np.ndarray) -> float:
        """Assess presence of leading lines."""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect lines using Hough transform
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=50, maxLineGap=10)
        
        if lines is None:
            return 0.0
        
        # Analyze line directions
        horizontal_lines = 0
        vertical_lines = 0
        diagonal_lines = 0
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = math.atan2(y2 - y1, x2 - x1) * 180 / math.pi
            
            if abs(angle) < 15 or abs(angle - 180) < 15:
                horizontal_lines += 1
            elif abs(angle - 90) < 15 or abs(angle + 90) < 15:
                vertical_lines += 1
            else:
                diagonal_lines += 1
        
        # Score based on line variety
        total_lines = len(lines)
        if total_lines == 0:
            return 0.0
        
        # Good composition has variety in line directions
        variety_score = min(1.0, (horizontal_lines + vertical_lines + diagonal_lines) / 10.0)
        
        return variety_score
    
    def _assess_action_timing(self, image: np.ndarray) -> float:
        """Assess action timing quality (simplified)."""
        # This is a simplified implementation
        # In a full implementation, this would analyze motion blur patterns
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect motion blur by analyzing edge patterns
        edges = cv2.Canny(gray, 50, 150)
        
        # Calculate edge density
        edge_density = np.sum(edges > 0) / edges.size
        
        # Higher edge density often indicates better action capture
        if edge_density > 0.1:
            return 0.8
        elif edge_density > 0.05:
            return 0.6
        elif edge_density > 0.02:
            return 0.4
        else:
            return 0.2
    
    def _assess_color_quality(self, image: np.ndarray) -> float:
        """Assess color quality and saturation."""
        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Calculate saturation
        saturation = hsv[:, :, 1]
        mean_saturation = np.mean(saturation)
        
        # Calculate color variance
        color_variance = np.var(saturation)
        
        # Good color quality has moderate saturation and good variance
        saturation_score = min(1.0, mean_saturation / 128.0)  # Normalize to 0-1
        variance_score = min(1.0, color_variance / 10000.0)  # Normalize variance
        
        color_score = (saturation_score + variance_score) / 2.0
        
        return color_score
    
    def _assess_noise_level(self, image: np.ndarray) -> float:
        """Assess image noise level."""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Calculate noise as difference between original and blurred
        noise = cv2.absdiff(gray, blurred)
        noise_level = np.mean(noise)
        
        # Convert to quality score (lower noise = higher score)
        if noise_level < 10:
            return 1.0
        elif noise_level < 20:
            return 0.8
        elif noise_level < 30:
            return 0.6
        elif noise_level < 40:
            return 0.4
        else:
            return 0.2
    
    def _assess_contrast(self, image: np.ndarray) -> float:
        """Assess image contrast."""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Calculate contrast using standard deviation
        contrast = np.std(gray)
        
        # Normalize contrast score
        if contrast > 50:
            return 1.0
        elif contrast > 40:
            return 0.8
        elif contrast > 30:
            return 0.6
        elif contrast > 20:
            return 0.4
        else:
            return 0.2
    
    def _calculate_overall_score(self, sharpness: float, exposure: float, composition: float,
                                action: float, color: float, noise: float, contrast: float) -> float:
        """Calculate overall quality score."""
        # Weighted average of all metrics
        weights = {
            'sharpness': 0.25,    # Most important for sports photos
            'exposure': 0.20,     # Very important
            'composition': 0.15,  # Important for aesthetics
            'action': 0.15,       # Important for sports
            'color': 0.10,        # Moderately important
            'noise': 0.10,        # Moderately important
            'contrast': 0.05      # Less important
        }
        
        overall_score = (
            sharpness * weights['sharpness'] +
            exposure * weights['exposure'] +
            composition * weights['composition'] +
            action * weights['action'] +
            color * weights['color'] +
            noise * weights['noise'] +
            contrast * weights['contrast']
        )
        
        return overall_score
    
    def _generate_recommendations(self, metrics: QualityMetrics) -> List[str]:
        """Generate improvement recommendations."""
        recommendations = []
        
        if metrics.sharpness_score < self.thresholds['fair_sharpness']:
            recommendations.append("Consider using faster shutter speed or image stabilization to reduce blur")
        
        if metrics.exposure_score < self.thresholds['fair_exposure']:
            recommendations.append("Adjust exposure settings - image may be over or underexposed")
        
        if metrics.composition_score < self.thresholds['fair_composition']:
            recommendations.append("Consider repositioning to improve composition using rule of thirds")
        
        if metrics.action_score < self.thresholds['fair_action']:
            recommendations.append("Try capturing action at peak moment for better timing")
        
        if metrics.color_score < 0.4:
            recommendations.append("Consider adjusting color saturation or white balance")
        
        if metrics.noise_score < 0.4:
            recommendations.append("Reduce ISO or use noise reduction techniques")
        
        if metrics.contrast_score < 0.4:
            recommendations.append("Increase contrast or adjust lighting conditions")
        
        if not recommendations:
            recommendations.append("Photo quality is excellent - no specific improvements needed")
        
        return recommendations
    
    def _determine_quality_grade(self, overall_score: float) -> str:
        """Determine quality grade based on overall score."""
        for grade, threshold in self.grade_thresholds.items():
            if overall_score >= threshold:
                return grade
        return 'F'
    
    def draw_quality_assessment(self, image: np.ndarray, assessment: QualityAssessment) -> np.ndarray:
        """Draw quality assessment results on the image."""
        annotated = image.copy()
        
        # Draw quality grade
        grade_color = self._get_grade_color(assessment.quality_grade)
        cv2.putText(annotated, f"Quality Grade: {assessment.quality_grade}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, grade_color, 2)
        
        # Draw overall score
        cv2.putText(annotated, f"Overall Score: {assessment.metrics.overall_score:.2f}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, grade_color, 2)
        
        # Draw individual metrics
        y_offset = 100
        metrics_text = [
            f"Sharpness: {assessment.metrics.sharpness_score:.2f}",
            f"Exposure: {assessment.metrics.exposure_score:.2f}",
            f"Composition: {assessment.metrics.composition_score:.2f}",
            f"Action: {assessment.metrics.action_score:.2f}",
            f"Color: {assessment.metrics.color_score:.2f}",
            f"Noise: {assessment.metrics.noise_score:.2f}",
            f"Contrast: {assessment.metrics.contrast_score:.2f}"
        ]
        
        for text in metrics_text:
            cv2.putText(annotated, text, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 25
        
        return annotated
    
    def _get_grade_color(self, grade: str) -> Tuple[int, int, int]:
        """Get color for quality grade."""
        color_map = {
            'A': (0, 255, 0),    # Green
            'B': (0, 255, 255),  # Yellow
            'C': (0, 165, 255),  # Orange
            'D': (0, 0, 255),    # Red
            'F': (0, 0, 128)     # Dark Red
        }
        return color_map.get(grade, (255, 255, 255))
