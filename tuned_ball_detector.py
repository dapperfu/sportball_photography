#!/usr/bin/env python3
"""
Tuned Ball Detection Tool

This tool uses optimized parameters to detect exactly one ball per image
with high accuracy and minimal false positives.

Author: Claude Sonnet 4 (claude-3-5-sonnet-20241022)
Generated via Cursor IDE (cursor.sh) with AI assistance
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
import click
from loguru import logger
import math


class TunedBallDetector:
    """Tuned ball detector optimized for single ball detection."""
    
    def __init__(self):
        """Initialize the tuned ball detector."""
        # Optimized Hough Circle parameters for single ball detection
        self.hough_params = {
            'dp': 1,           # Inverse ratio of accumulator resolution
            'minDist': 100,    # Increased minimum distance between centers
            'param1': 100,     # Increased upper threshold for edge detection
            'param2': 50,      # Increased accumulator threshold
            'minRadius': 20,   # Increased minimum radius
            'maxRadius': 80    # Decreased maximum radius
        }
        
        # Ball color signatures (HSV ranges) - more restrictive
        self.ball_color_signatures = {
            'soccer': {
                'white': {'lower': (0, 0, 200), 'upper': (180, 30, 255)},
                'black': {'lower': (0, 0, 0), 'upper': (180, 255, 50)},
            },
            'basketball': {
                'orange': {'lower': (5, 120, 120), 'upper': (15, 255, 255)},
            },
            'tennis': {
                'yellow': {'lower': (20, 120, 120), 'upper': (30, 255, 255)},
            },
            'rugby': {
                'brown': {'lower': (10, 80, 80), 'upper': (20, 255, 200)},
            }
        }
        
        logger.info("Initialized TunedBallDetector")
    
    def detect_single_ball(self, image: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Detect a single ball with high confidence.
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            Best ball detection or None if no ball found
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        
        # Detect circles with tuned parameters
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            **self.hough_params
        )
        
        if circles is None:
            logger.debug("No circles detected with Hough transform")
            return None
        
        circles = np.round(circles[0, :]).astype("int")
        logger.debug(f"Detected {len(circles)} circles with Hough transform")
        
        # Score and filter circles
        scored_circles = []
        for (x, y, r) in circles:
            score = self._score_circle(image, x, y, r)
            if score > 0.5:  # Only keep high-confidence detections
                scored_circles.append({
                    'center': (x, y),
                    'radius': r,
                    'score': score,
                    'method': 'hough_circles'
                })
        
        if not scored_circles:
            logger.debug("No circles passed scoring threshold")
            return None
        
        # Return the best scoring circle
        best_circle = max(scored_circles, key=lambda c: c['score'])
        
        # Classify ball type
        ball_type = self._classify_ball_type(image, best_circle['center'][0], best_circle['center'][1], best_circle['radius'])
        
        return {
            'center': best_circle['center'],
            'radius': best_circle['radius'],
            'type': ball_type,
            'confidence': best_circle['score'],
            'method': best_circle['method']
        }
    
    def _score_circle(self, image: np.ndarray, x: int, y: int, r: int) -> float:
        """
        Score a detected circle based on multiple criteria.
        
        Args:
            image: Original image
            x, y, r: Circle parameters
            
        Returns:
            Confidence score (0-1)
        """
        height, width = image.shape[:2]
        
        # Check if circle is within image bounds
        if x - r < 0 or x + r >= width or y - r < 0 or y + r >= height:
            return 0.0
        
        # Extract circle region
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.circle(mask, (x, y), r, 255, -1)
        
        # Calculate edge density in the circle
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        circle_edges = cv2.bitwise_and(edges, mask)
        edge_density = np.sum(circle_edges > 0) / (np.pi * r * r)
        
        # Calculate color uniformity
        circle_region = image[y-r:y+r, x-r:x+r]
        if circle_region.size == 0:
            return 0.0
        
        # Convert to HSV for better color analysis
        hsv_region = cv2.cvtColor(circle_region, cv2.COLOR_BGR2HSV)
        
        # Calculate color variance (lower variance = more uniform = more likely to be a ball)
        color_variance = np.var(hsv_region)
        color_uniformity = max(0, 1.0 - color_variance / 10000.0)
        
        # Calculate circularity (how circular the detected region is)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            contour = contours[0]
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
            else:
                circularity = 0.0
        else:
            circularity = 0.0
        
        # Calculate size appropriateness (balls should be reasonable size)
        image_area = height * width
        circle_area = np.pi * r * r
        size_ratio = circle_area / image_area
        
        # Ideal size ratio for a ball in a sports photo (0.001 to 0.01)
        if 0.001 <= size_ratio <= 0.01:
            size_score = 1.0
        elif 0.0005 <= size_ratio <= 0.02:
            size_score = 0.8
        else:
            size_score = 0.3
        
        # Combine scores
        total_score = (
            edge_density * 0.3 +           # Edge density
            color_uniformity * 0.3 +        # Color uniformity
            circularity * 0.2 +             # Circularity
            size_score * 0.2                # Size appropriateness
        )
        
        return min(1.0, total_score)
    
    def _classify_ball_type(self, image: np.ndarray, x: int, y: int, r: int) -> str:
        """
        Classify ball type based on color analysis.
        
        Args:
            image: Original image
            x, y, r: Circle parameters
            
        Returns:
            Ball type classification
        """
        # Extract ball region
        x1, y1 = max(0, x-r), max(0, y-r)
        x2, y2 = min(image.shape[1], x+r), min(image.shape[0], y+r)
        ball_region = image[y1:y2, x1:x2]
        
        if ball_region.size == 0:
            return "unknown"
        
        # Convert to HSV
        hsv_region = cv2.cvtColor(ball_region, cv2.COLOR_BGR2HSV)
        
        # Calculate color histogram
        hist_h = cv2.calcHist([hsv_region], [0], None, [180], [0, 180])
        hist_s = cv2.calcHist([hsv_region], [1], None, [256], [0, 256])
        hist_v = cv2.calcHist([hsv_region], [2], None, [256], [0, 256])
        
        # Get dominant colors
        dominant_hue = np.argmax(hist_h)
        dominant_saturation = np.argmax(hist_s)
        dominant_value = np.argmax(hist_v)
        
        # Classify based on color characteristics
        if dominant_value > 200 and dominant_saturation < 50:  # White/light
            return "soccer"
        elif 5 <= dominant_hue <= 15 and dominant_saturation > 120:  # Orange
            return "basketball"
        elif 20 <= dominant_hue <= 30 and dominant_saturation > 120:  # Yellow
            return "tennis"
        elif 10 <= dominant_hue <= 20 and dominant_saturation > 80:  # Brown/orange
            return "rugby"
        
        return "unknown"
    
    def draw_ball_detection(self, image: np.ndarray, ball_detection: Optional[Dict[str, Any]]) -> np.ndarray:
        """
        Draw ball detection on the image.
        
        Args:
            image: Original image
            ball_detection: Ball detection result
            
        Returns:
            Annotated image
        """
        annotated = image.copy()
        
        if ball_detection is None:
            # Draw "No ball detected" message
            cv2.putText(annotated, "No ball detected", (50, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return annotated
        
        x, y = ball_detection['center']
        r = ball_detection['radius']
        ball_type = ball_detection['type']
        confidence = ball_detection['confidence']
        
        # Color based on ball type
        colors = {
            "soccer": (0, 255, 0),      # Green
            "basketball": (0, 165, 255), # Orange
            "tennis": (0, 255, 255),    # Yellow
            "rugby": (0, 0, 255),       # Red
            "unknown": (255, 255, 255)  # White
        }
        color = colors.get(ball_type, (255, 255, 255))
        
        # Draw circle
        cv2.circle(annotated, (x, y), r, color, 3)
        
        # Draw center point
        cv2.circle(annotated, (x, y), 3, color, -1)
        
        # Draw label
        label = f"{ball_type.title()} Ball ({confidence:.2f})"
        cv2.putText(annotated, label, (x - r, y - r - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # Draw confidence bar
        bar_width = 100
        bar_height = 10
        bar_x = x - bar_width // 2
        bar_y = y + r + 30
        
        # Background bar
        cv2.rectangle(annotated, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (100, 100, 100), -1)
        
        # Confidence bar
        confidence_width = int(bar_width * confidence)
        cv2.rectangle(annotated, (bar_x, bar_y), (bar_x + confidence_width, bar_y + bar_height), color, -1)
        
        return annotated


@click.command()
@click.argument('input_pattern', type=str)
@click.argument('output_dir', type=str)
@click.option('--max-images', '-m', default=5, type=int, help='Maximum number of images to process')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
def main(input_pattern: str, output_dir: str, max_images: int, verbose: bool):
    """Detect single ball per image with tuned parameters."""
    
    # Setup logging
    if verbose:
        logger.add("tuned_ball_detection.log", level="DEBUG")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find images
    if input_pattern.startswith('/'):
        parent_dir = Path(input_pattern).parent
        pattern = Path(input_pattern).name
        image_files = list(parent_dir.glob(pattern))
    else:
        image_files = list(Path('.').glob(input_pattern))
    
    if not image_files:
        logger.error(f"No images found matching pattern: {input_pattern}")
        return
    
    # Limit images
    image_files = image_files[:max_images]
    
    logger.info(f"Processing {len(image_files)} images for tuned ball detection")
    
    # Initialize detector
    detector = TunedBallDetector()
    
    # Process images
    balls_detected = 0
    for image_path in image_files:
        logger.info(f"Analyzing {image_path.name}")
        
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            logger.error(f"Failed to load {image_path.name}")
            continue
        
        # Detect ball
        ball_detection = detector.detect_single_ball(image)
        
        if ball_detection:
            balls_detected += 1
            logger.info(f"  Ball detected: {ball_detection['type']} (confidence: {ball_detection['confidence']:.2f})")
        else:
            logger.info(f"  No ball detected")
        
        # Create annotated image
        annotated = detector.draw_ball_detection(image, ball_detection)
        
        # Save annotated image
        output_filename = f"{image_path.stem}_tuned_ball_detection.jpg"
        output_file = output_path / output_filename
        cv2.imwrite(str(output_file), annotated)
        
        logger.info(f"  Saved: {output_filename}")
    
    # Summary
    logger.info(f"Tuned ball detection complete!")
    logger.info(f"Images processed: {len(image_files)}")
    logger.info(f"Balls detected: {balls_detected}")
    logger.info(f"Detection rate: {balls_detected/len(image_files)*100:.1f}%")
    logger.info(f"Results saved to {output_path}")


if __name__ == '__main__':
    main()
