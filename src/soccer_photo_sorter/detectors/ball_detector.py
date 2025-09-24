"""
Ball Detection Module

This module provides ball detection capabilities for various sports including
soccer, rugby, basketball, and other ball sports.

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
class BallDetection:
    """Information about a detected ball."""
    ball_type: str  # 'soccer', 'rugby', 'basketball', 'tennis', 'unknown'
    center_x: int
    center_y: int
    radius: int
    confidence: float
    bounding_box: Tuple[int, int, int, int]  # (x, y, width, height)
    color_info: Dict[str, Any]
    motion_vector: Optional[Tuple[float, float]] = None  # (dx, dy) for tracking


@dataclass
class BallTrackingResult:
    """Result of ball detection and tracking."""
    balls_detected: List[BallDetection]
    processing_time: float
    success: bool
    error: Optional[str] = None


class BallDetector:
    """Ball detection and tracking system for sports analysis."""
    
    def __init__(self, 
                 confidence_threshold: float = 0.5,
                 min_ball_radius: int = 5,
                 max_ball_radius: int = 100,
                 enable_tracking: bool = True):
        """
        Initialize the ball detector.
        
        Args:
            confidence_threshold: Minimum confidence for ball detection
            min_ball_radius: Minimum ball radius in pixels
            max_ball_radius: Maximum ball radius in pixels
            enable_tracking: Enable motion tracking between frames
        """
        self.confidence_threshold = confidence_threshold
        self.min_ball_radius = min_ball_radius
        self.max_ball_radius = max_ball_radius
        self.enable_tracking = enable_tracking
        
        # Initialize Hough Circle Transform parameters
        self.hough_params = {
            'dp': 1,           # Inverse ratio of accumulator resolution
            'minDist': 20,     # Minimum distance between circle centers
            'param1': 50,      # Upper threshold for edge detection
            'param2': 30,      # Accumulator threshold for center detection
            'minRadius': min_ball_radius,
            'maxRadius': max_ball_radius
        }
        
        # Ball type color signatures (HSV ranges)
        self.ball_color_signatures = {
            'soccer': {
                'white': {'lower': (0, 0, 200), 'upper': (180, 30, 255)},
                'black': {'lower': (0, 0, 0), 'upper': (180, 255, 50)},
                'pattern': {'lower': (0, 0, 50), 'upper': (180, 255, 200)}
            },
            'rugby': {
                'white': {'lower': (0, 0, 200), 'upper': (180, 30, 255)},
                'brown': {'lower': (10, 50, 50), 'upper': (20, 255, 200)},
                'pattern': {'lower': (0, 0, 50), 'upper': (180, 255, 200)}
            },
            'basketball': {
                'orange': {'lower': (5, 100, 100), 'upper': (15, 255, 255)},
                'brown': {'lower': (10, 50, 50), 'upper': (20, 255, 200)}
            },
            'tennis': {
                'yellow': {'lower': (20, 100, 100), 'upper': (30, 255, 255)},
                'white': {'lower': (0, 0, 200), 'upper': (180, 30, 255)}
            }
        }
        
        # Previous frame data for tracking
        self.previous_balls: List[BallDetection] = []
        
        logger.info(f"Initialized BallDetector with confidence threshold {confidence_threshold}")
    
    def detect_balls(self, image: np.ndarray) -> BallTrackingResult:
        """
        Detect balls in an image using multiple detection methods.
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            Ball tracking result with detected balls
        """
        start_time = cv2.getTickCount()
        
        try:
            # Convert to different color spaces for better detection
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # Apply multiple detection methods
            circles_hough = self._detect_circles_hough(gray)
            circles_template = self._detect_circles_template_matching(gray)
            color_regions = self._detect_ball_color_regions(hsv)
            
            # Combine and filter results
            all_detections = circles_hough + circles_template + color_regions
            filtered_detections = self._filter_and_merge_detections(all_detections)
            
            # Classify ball types
            classified_balls = self._classify_ball_types(image, filtered_detections)
            
            # Apply tracking if enabled
            if self.enable_tracking and len(self.previous_balls) > 0:
                classified_balls = self._apply_motion_tracking(classified_balls)
            
            # Update previous frame data
            self.previous_balls = classified_balls.copy()
            
            processing_time = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()
            
            return BallTrackingResult(
                balls_detected=classified_balls,
                processing_time=processing_time,
                success=True
            )
            
        except Exception as e:
            logger.error(f"Ball detection failed: {e}")
            processing_time = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()
            return BallTrackingResult(
                balls_detected=[],
                processing_time=processing_time,
                success=False,
                error=str(e)
            )
    
    def _detect_circles_hough(self, gray_image: np.ndarray) -> List[BallDetection]:
        """Detect circular objects using Hough Circle Transform."""
        circles = cv2.HoughCircles(
            gray_image,
            cv2.HOUGH_GRADIENT,
            **self.hough_params
        )
        
        detections = []
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for (x, y, r) in circles:
                if self.min_ball_radius <= r <= self.max_ball_radius:
                    detection = BallDetection(
                        ball_type='unknown',
                        center_x=x,
                        center_y=y,
                        radius=r,
                        confidence=0.7,  # Default confidence for Hough circles
                        bounding_box=(x-r, y-r, 2*r, 2*r),
                        color_info={}
                    )
                    detections.append(detection)
        
        return detections
    
    def _detect_circles_template_matching(self, gray_image: np.ndarray) -> List[BallDetection]:
        """Detect circular objects using template matching."""
        detections = []
        
        # Create circular templates of different sizes
        template_sizes = [10, 15, 20, 25, 30, 40, 50]
        
        for size in template_sizes:
            if size < self.min_ball_radius or size > self.max_ball_radius:
                continue
                
            # Create circular template
            template = np.zeros((size*2+1, size*2+1), dtype=np.uint8)
            cv2.circle(template, (size, size), size, 255, -1)
            
            # Apply template matching
            result = cv2.matchTemplate(gray_image, template, cv2.TM_CCOEFF_NORMED)
            locations = np.where(result >= self.confidence_threshold)
            
            for pt in zip(*locations[::-1]):
                x, y = pt
                detection = BallDetection(
                    ball_type='unknown',
                    center_x=x + size,
                    center_y=y + size,
                    radius=size,
                    confidence=result[y, x],
                    bounding_box=(x, y, size*2, size*2),
                    color_info={}
                )
                detections.append(detection)
        
        return detections
    
    def _detect_ball_color_regions(self, hsv_image: np.ndarray) -> List[BallDetection]:
        """Detect ball-like regions based on color signatures."""
        detections = []
        
        for ball_type, color_ranges in self.ball_color_signatures.items():
            for color_name, (lower, upper) in color_ranges.items():
                # Create mask for color range
                mask = cv2.inRange(hsv_image, np.array(lower), np.array(upper))
                
                # Find contours
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for contour in contours:
                    # Check if contour is roughly circular
                    area = cv2.contourArea(contour)
                    if area < 50:  # Too small
                        continue
                    
                    # Calculate circularity
                    perimeter = cv2.arcLength(contour, True)
                    if perimeter == 0:
                        continue
                    
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    
                    if circularity > 0.7:  # Roughly circular
                        # Get bounding circle
                        (x, y), radius = cv2.minEnclosingCircle(contour)
                        x, y, radius = int(x), int(y), int(radius)
                        
                        if self.min_ball_radius <= radius <= self.max_ball_radius:
                            detection = BallDetection(
                                ball_type=ball_type,
                                center_x=x,
                                center_y=y,
                                radius=radius,
                                confidence=circularity,
                                bounding_box=(x-radius, y-radius, 2*radius, 2*radius),
                                color_info={'color_name': color_name, 'area': area}
                            )
                            detections.append(detection)
        
        return detections
    
    def _filter_and_merge_detections(self, detections: List[BallDetection]) -> List[BallDetection]:
        """Filter and merge overlapping detections."""
        if not detections:
            return []
        
        # Sort by confidence
        detections.sort(key=lambda x: x.confidence, reverse=True)
        
        filtered = []
        for detection in detections:
            # Check if this detection overlaps with any already filtered detection
            overlaps = False
            for existing in filtered:
                if self._detections_overlap(detection, existing):
                    overlaps = True
                    break
            
            if not overlaps:
                filtered.append(detection)
        
        return filtered
    
    def _detections_overlap(self, det1: BallDetection, det2: BallDetection, threshold: float = 0.5) -> bool:
        """Check if two detections overlap significantly."""
        # Calculate intersection area
        x1, y1, w1, h1 = det1.bounding_box
        x2, y2, w2, h2 = det2.bounding_box
        
        # Calculate intersection rectangle
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)
        
        if x_right < x_left or y_bottom < y_top:
            return False
        
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        union_area = w1 * h1 + w2 * h2 - intersection_area
        
        if union_area == 0:
            return False
        
        overlap_ratio = intersection_area / union_area
        return overlap_ratio > threshold
    
    def _classify_ball_types(self, image: np.ndarray, detections: List[BallDetection]) -> List[BallDetection]:
        """Classify detected circular objects as specific ball types."""
        classified = []
        
        for detection in detections:
            # Extract region around the ball
            x, y, w, h = detection.bounding_box
            x = max(0, x)
            y = max(0, y)
            w = min(w, image.shape[1] - x)
            h = min(h, image.shape[0] - y)
            
            if w <= 0 or h <= 0:
                continue
            
            ball_region = image[y:y+h, x:x+w]
            
            # Analyze color distribution
            ball_type = self._analyze_ball_color_signature(ball_region)
            
            # Update detection with classified type
            detection.ball_type = ball_type
            classified.append(detection)
        
        return classified
    
    def _analyze_ball_color_signature(self, ball_region: np.ndarray) -> str:
        """Analyze color signature to determine ball type."""
        if ball_region.size == 0:
            return 'unknown'
        
        # Convert to HSV
        hsv_region = cv2.cvtColor(ball_region, cv2.COLOR_BGR2HSV)
        
        # Calculate color histogram
        hist_h = cv2.calcHist([hsv_region], [0], None, [180], [0, 180])
        hist_s = cv2.calcHist([hsv_region], [1], None, [256], [0, 256])
        hist_v = cv2.calcHist([hsv_region], [2], None, [256], [0, 256])
        
        # Analyze dominant colors
        dominant_hue = np.argmax(hist_h)
        dominant_saturation = np.argmax(hist_s)
        dominant_value = np.argmax(hist_v)
        
        # Classify based on color characteristics
        if dominant_value > 200 and dominant_saturation < 50:  # White/light
            if dominant_hue < 20 or dominant_hue > 160:  # White
                return 'soccer'  # Likely soccer ball (white/black pattern)
        elif 5 <= dominant_hue <= 15:  # Orange/brown
            return 'basketball'  # Likely basketball
        elif 20 <= dominant_hue <= 30:  # Yellow
            return 'tennis'  # Likely tennis ball
        elif 10 <= dominant_hue <= 20:  # Brown/orange
            return 'rugby'  # Likely rugby ball
        
        return 'unknown'
    
    def _apply_motion_tracking(self, current_detections: List[BallDetection]) -> List[BallDetection]:
        """Apply motion tracking to connect detections across frames."""
        if not self.previous_balls:
            return current_detections
        
        tracked_detections = []
        
        for current_det in current_detections:
            # Find closest previous detection
            closest_prev = None
            min_distance = float('inf')
            
            for prev_det in self.previous_balls:
                distance = math.sqrt(
                    (current_det.center_x - prev_det.center_x) ** 2 +
                    (current_det.center_y - prev_det.center_y) ** 2
                )
                
                if distance < min_distance and distance < 50:  # Max movement threshold
                    min_distance = distance
                    closest_prev = prev_det
            
            # Calculate motion vector if we found a match
            if closest_prev:
                dx = current_det.center_x - closest_prev.center_x
                dy = current_det.center_y - closest_prev.center_y
                current_det.motion_vector = (dx, dy)
                
                # Inherit ball type if it was previously classified
                if closest_prev.ball_type != 'unknown':
                    current_det.ball_type = closest_prev.ball_type
            
            tracked_detections.append(current_det)
        
        return tracked_detections
    
    def draw_ball_detections(self, image: np.ndarray, detections: List[BallDetection]) -> np.ndarray:
        """Draw ball detections on the image."""
        annotated = image.copy()
        
        for detection in detections:
            # Draw circle
            cv2.circle(annotated, (detection.center_x, detection.center_y), 
                      detection.radius, (0, 255, 0), 2)
            
            # Draw bounding box
            x, y, w, h = detection.bounding_box
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (255, 0, 0), 1)
            
            # Draw label
            label = f"{detection.ball_type} ({detection.confidence:.2f})"
            cv2.putText(annotated, label, 
                       (detection.center_x - detection.radius, detection.center_y - detection.radius - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Draw motion vector if available
            if detection.motion_vector:
                dx, dy = detection.motion_vector
                end_x = detection.center_x + int(dx * 2)  # Scale for visibility
                end_y = detection.center_y + int(dy * 2)
                cv2.arrowedLine(annotated, (detection.center_x, detection.center_y),
                              (end_x, end_y), (0, 0, 255), 2)
        
        return annotated
