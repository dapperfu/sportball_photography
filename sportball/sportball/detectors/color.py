"""
Color detection module for soccer photo sorter.

This module provides color detection and classification functionality
for identifying jersey colors in soccer photographs.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from loguru import logger

from ..core.image_processor import ImageProcessor
from ..config.settings import ColorConfig


class ColorDetector:
    """Detector for jersey colors in soccer photos."""
    
    def __init__(self, 
                 confidence_threshold: float = 0.8,
                 color_config: Optional[ColorConfig] = None):
        """
        Initialize color detector.
        
        Args:
            confidence_threshold: Minimum confidence for color detection
            color_config: Optional color configuration
        """
        self.confidence_threshold = confidence_threshold
        self.color_config = color_config or ColorConfig()
        self.image_processor = ImageProcessor()
        
        # Color space conversion matrices
        self._color_spaces = {
            'RGB': None,  # Default
            'HSV': cv2.COLOR_RGB2HSV,
            'LAB': cv2.COLOR_RGB2LAB,
        }
    
    def detect_colors(self, image_path: Path) -> List[Tuple[str, float]]:
        """
        Detect dominant colors in image.
        
        Args:
            image_path: Path to image file
            
        Returns:
            List of (color_name, confidence) tuples
        """
        try:
            # Load image
            image = self.image_processor.load_image(image_path)
            if image is None:
                return []
            
            # Resize for processing
            image = self.image_processor.resize_image(image)
            
            # Preprocess image
            processed_image = self.image_processor.preprocess_image(image)
            
            # Detect dominant colors
            dominant_colors = self.image_processor.detect_dominant_colors(processed_image)
            
            # Classify colors
            detected_colors = []
            for color_rgb in dominant_colors:
                color_name = self.image_processor.classify_color(
                    color_rgb, 
                    self.color_config.colors
                )
                
                if color_name:
                    # Calculate confidence based on color distance
                    confidence = self._calculate_color_confidence(
                        color_rgb, 
                        self.color_config.colors[color_name]
                    )
                    
                    if confidence >= self.confidence_threshold:
                        detected_colors.append((color_name, confidence))
            
            # Remove duplicates and sort by confidence
            detected_colors = list(set(detected_colors))
            detected_colors.sort(key=lambda x: x[1], reverse=True)
            
            logger.debug(f"Detected colors in {image_path.name}: {detected_colors}")
            return detected_colors
            
        except Exception as e:
            logger.error(f"Error detecting colors in {image_path}: {e}")
            return []
    
    def _calculate_color_confidence(self, 
                                   detected_color: Tuple[int, int, int],
                                   color_config: Dict[str, Any]) -> float:
        """
        Calculate confidence score for color detection.
        
        Args:
            detected_color: Detected color (R, G, B)
            color_config: Color configuration dictionary
            
        Returns:
            Confidence score (0.0-1.0)
        """
        rgb_range = color_config['rgb_range']
        tolerance = color_config['tolerance']
        
        # Calculate Euclidean distance
        distance = np.sqrt(
            (detected_color[0] - rgb_range[0])**2 +
            (detected_color[1] - rgb_range[1])**2 +
            (detected_color[2] - rgb_range[2])**2
        )
        
        # Convert distance to confidence (closer = higher confidence)
        if distance <= tolerance:
            confidence = 1.0 - (distance / tolerance) * 0.3  # 0.7 to 1.0 range
        else:
            confidence = 0.0
        
        return max(0.0, min(1.0, confidence))
    
    def detect_jersey_colors(self, image_path: Path) -> List[Tuple[str, float]]:
        """
        Detect colors specifically in jersey regions.
        
        Args:
            image_path: Path to image file
            
        Returns:
            List of (color_name, confidence) tuples
        """
        try:
            # Load image
            image = self.image_processor.load_image(image_path)
            if image is None:
                return []
            
            # Resize for processing
            image = self.image_processor.resize_image(image)
            
            # Extract jersey regions
            jersey_regions = self.image_processor.extract_jersey_regions(image)
            
            if not jersey_regions:
                logger.debug(f"No jersey regions found in {image_path.name}")
                return []
            
            # Analyze each jersey region
            all_colors = []
            for region in jersey_regions:
                # Detect dominant colors in region
                dominant_colors = self.image_processor.detect_dominant_colors(region)
                
                # Classify colors
                for color_rgb in dominant_colors:
                    color_name = self.image_processor.classify_color(
                        color_rgb, 
                        self.color_config.colors
                    )
                    
                    if color_name:
                        confidence = self._calculate_color_confidence(
                            color_rgb, 
                            self.color_config.colors[color_name]
                        )
                        
                        if confidence >= self.confidence_threshold:
                            all_colors.append((color_name, confidence))
            
            # Remove duplicates and sort by confidence
            all_colors = list(set(all_colors))
            all_colors.sort(key=lambda x: x[1], reverse=True)
            
            logger.debug(f"Detected jersey colors in {image_path.name}: {all_colors}")
            return all_colors
            
        except Exception as e:
            logger.error(f"Error detecting jersey colors in {image_path}: {e}")
            return []
    
    def analyze_color_distribution(self, 
                                  image_path: Path,
                                  color_space: str = 'RGB') -> Dict[str, Any]:
        """
        Analyze color distribution in image.
        
        Args:
            image_path: Path to image file
            color_space: Color space for analysis ('RGB', 'HSV', 'LAB')
            
        Returns:
            Color distribution analysis dictionary
        """
        try:
            # Load image
            image = self.image_processor.load_image(image_path)
            if image is None:
                return {}
            
            # Resize for processing
            image = self.image_processor.resize_image(image)
            
            # Convert color space if needed
            if color_space != 'RGB' and color_space in self._color_spaces:
                conversion_code = self._color_spaces[color_space]
                if conversion_code:
                    image = cv2.cvtColor(image, conversion_code)
            
            # Extract color histogram
            histograms = self.image_processor.extract_color_histogram(image)
            
            # Analyze histogram
            analysis = {
                'color_space': color_space,
                'histograms': histograms,
                'dominant_colors': [],
                'color_variance': {},
            }
            
            # Calculate color variance for each channel
            for channel, hist in histograms.items():
                mean_val = np.mean(hist)
                variance = np.var(hist)
                analysis['color_variance'][channel] = {
                    'mean': float(mean_val),
                    'variance': float(variance),
                    'std_dev': float(np.sqrt(variance))
                }
            
            # Detect dominant colors
            dominant_colors = self.image_processor.detect_dominant_colors(image)
            for color_rgb in dominant_colors:
                color_name = self.image_processor.classify_color(
                    color_rgb, 
                    self.color_config.colors
                )
                
                confidence = 0.0
                if color_name:
                    confidence = self._calculate_color_confidence(
                        color_rgb, 
                        self.color_config.colors[color_name]
                    )
                
                analysis['dominant_colors'].append({
                    'rgb': color_rgb,
                    'name': color_name,
                    'confidence': confidence
                })
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing color distribution in {image_path}: {e}")
            return {}
    
    def get_color_statistics(self, 
                           image_paths: List[Path]) -> Dict[str, Any]:
        """
        Get color statistics across multiple images.
        
        Args:
            image_paths: List of image file paths
            
        Returns:
            Color statistics dictionary
        """
        color_counts = {}
        total_images = len(image_paths)
        processed_images = 0
        
        for image_path in image_paths:
            try:
                detected_colors = self.detect_colors(image_path)
                
                for color_name, confidence in detected_colors:
                    if color_name not in color_counts:
                        color_counts[color_name] = {
                            'count': 0,
                            'total_confidence': 0.0,
                            'files': []
                        }
                    
                    color_counts[color_name]['count'] += 1
                    color_counts[color_name]['total_confidence'] += confidence
                    color_counts[color_name]['files'].append({
                        'file': image_path.name,
                        'confidence': confidence
                    })
                
                processed_images += 1
                
            except Exception as e:
                logger.error(f"Error processing {image_path}: {e}")
        
        # Calculate statistics
        statistics = {
            'total_images': total_images,
            'processed_images': processed_images,
            'color_distribution': color_counts,
            'most_common_colors': [],
        }
        
        # Sort colors by frequency
        sorted_colors = sorted(
            color_counts.items(), 
            key=lambda x: x[1]['count'], 
            reverse=True
        )
        
        for color_name, data in sorted_colors:
            avg_confidence = data['total_confidence'] / data['count']
            statistics['most_common_colors'].append({
                'color': color_name,
                'count': data['count'],
                'percentage': (data['count'] / processed_images) * 100,
                'avg_confidence': avg_confidence
            })
        
        return statistics
    
    def update_color_config(self, color_config: ColorConfig) -> None:
        """
        Update color configuration.
        
        Args:
            color_config: New color configuration
        """
        self.color_config = color_config
        logger.info("Color configuration updated")
    
    def add_custom_color(self, 
                        name: str, 
                        rgb_range: List[int], 
                        tolerance: int) -> None:
        """
        Add custom color definition.
        
        Args:
            name: Color name
            rgb_range: RGB color range [R, G, B]
            tolerance: Color tolerance
        """
        self.color_config.colors[name] = {
            'rgb_range': rgb_range,
            'tolerance': tolerance
        }
        logger.info(f"Added custom color: {name}")
    
    def __repr__(self) -> str:
        """String representation of color detector."""
        return f"ColorDetector(threshold={self.confidence_threshold}, colors={len(self.color_config.colors)})"
