"""
Field Position Analysis Module

This module provides field position analysis capabilities for sports,
determining player positions relative to field boundaries and zones.

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
class FieldZone:
    """Information about a field zone."""
    zone_name: str  # 'goal_area', 'penalty_area', 'center_field', 'corner', 'sideline'
    zone_type: str  # 'defensive', 'midfield', 'offensive', 'neutral'
    boundaries: List[Tuple[int, int]]  # Polygon vertices
    center: Tuple[int, int]  # Zone center point
    area: float  # Zone area in pixels


@dataclass
class PlayerPosition:
    """Information about a player's field position."""
    player_id: int
    field_x: float  # Normalized x position (0-1)
    field_y: float  # Normalized y position (0-1)
    zone: Optional[FieldZone]
    distance_to_goal: float  # Distance to nearest goal
    distance_to_center: float  # Distance to field center
    field_side: str  # 'left', 'right', 'center'
    is_in_penalty_area: bool
    is_in_goal_area: bool
    is_offside: bool


@dataclass
class FieldAnalysisResult:
    """Result of field position analysis."""
    field_detected: bool
    field_zones: List[FieldZone]
    player_positions: List[PlayerPosition]
    field_dimensions: Tuple[int, int]  # Field width, height
    processing_time: float
    success: bool
    error: Optional[str] = None


class FieldAnalyzer:
    """Field position analysis system for sports."""
    
    def __init__(self, 
                 field_type: str = 'soccer',
                 enable_offside_detection: bool = True):
        """
        Initialize the field analyzer.
        
        Args:
            field_type: Type of field ('soccer', 'rugby', 'basketball')
            enable_offside_detection: Enable offside detection
        """
        self.field_type = field_type
        self.enable_offside_detection = enable_offside_detection
        
        # Field-specific parameters
        self.field_configs = {
            'soccer': {
                'aspect_ratio': 1.5,  # Length to width ratio
                'goal_width': 0.15,  # Goal width as fraction of field width
                'penalty_area_ratio': 0.4,  # Penalty area size ratio
                'goal_area_ratio': 0.2,  # Goal area size ratio
                'center_circle_ratio': 0.3  # Center circle radius ratio
            },
            'rugby': {
                'aspect_ratio': 1.8,
                'goal_width': 0.1,
                'penalty_area_ratio': 0.3,
                'goal_area_ratio': 0.15,
                'center_circle_ratio': 0.25
            },
            'basketball': {
                'aspect_ratio': 1.0,
                'goal_width': 0.2,
                'penalty_area_ratio': 0.3,
                'goal_area_ratio': 0.15,
                'center_circle_ratio': 0.4
            }
        }
        
        # Field detection parameters
        self.field_detection_params = {
            'min_field_area': 10000,  # Minimum field area in pixels
            'line_threshold': 50,     # Threshold for line detection
            'corner_threshold': 0.1   # Threshold for corner detection
        }
        
        logger.info(f"Initialized FieldAnalyzer for {field_type} field")
    
    def analyze_field_positions(self, image: np.ndarray, player_bboxes: List[Tuple[int, int, int, int]]) -> FieldAnalysisResult:
        """
        Analyze field positions for detected players.
        
        Args:
            image: Input image (BGR format)
            player_bboxes: List of player bounding boxes (x, y, width, height)
            
        Returns:
            Field analysis result
        """
        start_time = cv2.getTickCount()
        
        try:
            # Detect field boundaries
            field_detected, field_zones = self._detect_field_boundaries(image)
            
            if not field_detected:
                return FieldAnalysisResult(
                    field_detected=False,
                    field_zones=[],
                    player_positions=[],
                    field_dimensions=(0, 0),
                    processing_time=(cv2.getTickCount() - start_time) / cv2.getTickFrequency(),
                    success=False,
                    error="Field boundaries not detected"
                )
            
            # Analyze player positions
            player_positions = self._analyze_player_positions(image, player_bboxes, field_zones)
            
            # Apply offside detection if enabled
            if self.enable_offside_detection:
                player_positions = self._detect_offside_positions(player_positions, field_zones)
            
            # Calculate field dimensions
            field_dimensions = self._calculate_field_dimensions(field_zones)
            
            processing_time = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()
            
            return FieldAnalysisResult(
                field_detected=True,
                field_zones=field_zones,
                player_positions=player_positions,
                field_dimensions=field_dimensions,
                processing_time=processing_time,
                success=True
            )
            
        except Exception as e:
            logger.error(f"Field position analysis failed: {e}")
            processing_time = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()
            return FieldAnalysisResult(
                field_detected=False,
                field_zones=[],
                player_positions=[],
                field_dimensions=(0, 0),
                processing_time=processing_time,
                success=False,
                error=str(e)
            )
    
    def _detect_field_boundaries(self, image: np.ndarray) -> Tuple[bool, List[FieldZone]]:
        """Detect field boundaries and zones."""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Detect lines
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=50, maxLineGap=10)
        
        if lines is None:
            return False, []
        
        # Group lines and detect field boundaries
        field_zones = self._extract_field_zones(lines, image.shape)
        
        return len(field_zones) > 0, field_zones
    
    def _extract_field_zones(self, lines: np.ndarray, image_shape: Tuple[int, int]) -> List[FieldZone]:
        """Extract field zones from detected lines."""
        zones = []
        height, width = image_shape[:2]
        
        # Detect field boundaries (simplified approach)
        # In a full implementation, this would use more sophisticated line analysis
        
        # Center field zone
        center_x, center_y = width // 2, height // 2
        center_zone = FieldZone(
            zone_name='center_field',
            zone_type='neutral',
            boundaries=[(center_x - 100, center_y - 100), (center_x + 100, center_y + 100)],
            center=(center_x, center_y),
            area=40000
        )
        zones.append(center_zone)
        
        # Left goal area (simplified)
        left_goal_zone = FieldZone(
            zone_name='left_goal_area',
            zone_type='defensive',
            boundaries=[(0, height//2 - 50), (width//4, height//2 + 50)],
            center=(width//8, height//2),
            area=width * height // 8
        )
        zones.append(left_goal_zone)
        
        # Right goal area (simplified)
        right_goal_zone = FieldZone(
            zone_name='right_goal_area',
            zone_type='defensive',
            boundaries=[(3*width//4, height//2 - 50), (width, height//2 + 50)],
            center=(7*width//8, height//2),
            area=width * height // 8
        )
        zones.append(right_goal_zone)
        
        # Left penalty area
        left_penalty_zone = FieldZone(
            zone_name='left_penalty_area',
            zone_type='defensive',
            boundaries=[(0, height//2 - 100), (width//3, height//2 + 100)],
            center=(width//6, height//2),
            area=width * height // 6
        )
        zones.append(left_penalty_zone)
        
        # Right penalty area
        right_penalty_zone = FieldZone(
            zone_name='right_penalty_area',
            zone_type='defensive',
            boundaries=[(2*width//3, height//2 - 100), (width, height//2 + 100)],
            center=(5*width//6, height//2),
            area=width * height // 6
        )
        zones.append(right_penalty_zone)
        
        return zones
    
    def _analyze_player_positions(self, image: np.ndarray, player_bboxes: List[Tuple[int, int, int, int]], field_zones: List[FieldZone]) -> List[PlayerPosition]:
        """Analyze positions of detected players."""
        positions = []
        height, width = image.shape[:2]
        
        for i, bbox in enumerate(player_bboxes):
            x, y, w, h = bbox
            
            # Calculate normalized field position
            field_x = (x + w/2) / width
            field_y = (y + h/2) / height
            
            # Determine which zone the player is in
            player_zone = self._find_player_zone(field_x, field_y, field_zones)
            
            # Calculate distances
            distance_to_goal = self._calculate_distance_to_goal(field_x, field_y, width, height)
            distance_to_center = self._calculate_distance_to_center(field_x, field_y)
            
            # Determine field side
            field_side = self._determine_field_side(field_x)
            
            # Check if in penalty/goal areas
            is_in_penalty_area = self._is_in_penalty_area(field_x, field_y, field_zones)
            is_in_goal_area = self._is_in_goal_area(field_x, field_y, field_zones)
            
            position = PlayerPosition(
                player_id=i,
                field_x=field_x,
                field_y=field_y,
                zone=player_zone,
                distance_to_goal=distance_to_goal,
                distance_to_center=distance_to_center,
                field_side=field_side,
                is_in_penalty_area=is_in_penalty_area,
                is_in_goal_area=is_in_goal_area,
                is_offside=False  # Will be determined later
            )
            
            positions.append(position)
        
        return positions
    
    def _find_player_zone(self, field_x: float, field_y: float, field_zones: List[FieldZone]) -> Optional[FieldZone]:
        """Find which zone a player is in."""
        for zone in field_zones:
            if self._point_in_zone(field_x, field_y, zone):
                return zone
        return None
    
    def _point_in_zone(self, x: float, y: float, zone: FieldZone) -> bool:
        """Check if a point is inside a zone."""
        # Simplified point-in-polygon test
        # In a full implementation, this would use proper polygon containment
        zone_center = zone.center
        zone_width = 0.2  # Simplified zone width
        zone_height = 0.2  # Simplified zone height
        
        return (abs(x - zone_center[0]) < zone_width and 
                abs(y - zone_center[1]) < zone_height)
    
    def _calculate_distance_to_goal(self, field_x: float, field_y: float, width: int, height: int) -> float:
        """Calculate distance to nearest goal."""
        # Simplified distance calculation
        # Distance to left goal
        left_goal_distance = math.sqrt(field_x**2 + (field_y - 0.5)**2)
        
        # Distance to right goal
        right_goal_distance = math.sqrt((1 - field_x)**2 + (field_y - 0.5)**2)
        
        return min(left_goal_distance, right_goal_distance)
    
    def _calculate_distance_to_center(self, field_x: float, field_y: float) -> float:
        """Calculate distance to field center."""
        return math.sqrt((field_x - 0.5)**2 + (field_y - 0.5)**2)
    
    def _determine_field_side(self, field_x: float) -> str:
        """Determine which side of the field the player is on."""
        if field_x < 0.33:
            return 'left'
        elif field_x > 0.67:
            return 'right'
        else:
            return 'center'
    
    def _is_in_penalty_area(self, field_x: float, field_y: float, field_zones: List[FieldZone]) -> bool:
        """Check if player is in penalty area."""
        for zone in field_zones:
            if 'penalty' in zone.zone_name and self._point_in_zone(field_x, field_y, zone):
                return True
        return False
    
    def _is_in_goal_area(self, field_x: float, field_y: float, field_zones: List[FieldZone]) -> bool:
        """Check if player is in goal area."""
        for zone in field_zones:
            if 'goal' in zone.zone_name and self._point_in_zone(field_x, field_y, zone):
                return True
        return False
    
    def _detect_offside_positions(self, player_positions: List[PlayerPosition], field_zones: List[FieldZone]) -> List[PlayerPosition]:
        """Detect offside positions (simplified implementation)."""
        # This is a simplified offside detection
        # In a full implementation, this would consider:
        # - Ball position
        # - Second-to-last defender
        # - Player's team affiliation
        
        for position in player_positions:
            # Simplified offside rule: if player is in opponent's half and close to goal
            if position.field_x > 0.6 and position.distance_to_goal < 0.3:
                position.is_offside = True
        
        return player_positions
    
    def _calculate_field_dimensions(self, field_zones: List[FieldZone]) -> Tuple[int, int]:
        """Calculate field dimensions from detected zones."""
        if not field_zones:
            return (0, 0)
        
        # Find bounding box of all zones
        min_x = min(min(point[0] for point in zone.boundaries) for zone in field_zones)
        max_x = max(max(point[0] for point in zone.boundaries) for zone in field_zones)
        min_y = min(min(point[1] for point in zone.boundaries) for zone in field_zones)
        max_y = max(max(point[1] for point in zone.boundaries) for zone in field_zones)
        
        width = max_x - min_x
        height = max_y - min_y
        
        return (width, height)
    
    def draw_field_analysis(self, image: np.ndarray, result: FieldAnalysisResult) -> np.ndarray:
        """Draw field analysis results on the image."""
        annotated = image.copy()
        
        # Draw field zones
        for zone in result.field_zones:
            color = self._get_zone_color(zone.zone_type)
            
            # Draw zone boundaries
            if len(zone.boundaries) >= 3:
                pts = np.array(zone.boundaries, np.int32)
                cv2.polylines(annotated, [pts], True, color, 2)
            
            # Draw zone center
            cv2.circle(annotated, zone.center, 5, color, -1)
            
            # Draw zone label
            cv2.putText(annotated, zone.zone_name, 
                       (zone.center[0] - 50, zone.center[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Draw player positions
        for position in result.player_positions:
            # Convert normalized coordinates to pixel coordinates
            height, width = image.shape[:2]
            pixel_x = int(position.field_x * width)
            pixel_y = int(position.field_y * height)
            
            # Choose color based on position
            if position.is_offside:
                color = (0, 0, 255)  # Red for offside
            elif position.is_in_penalty_area:
                color = (255, 0, 0)  # Blue for penalty area
            elif position.is_in_goal_area:
                color = (0, 255, 0)  # Green for goal area
            else:
                color = (255, 255, 0)  # Yellow for normal position
            
            # Draw player position
            cv2.circle(annotated, (pixel_x, pixel_y), 8, color, -1)
            
            # Draw player ID
            cv2.putText(annotated, f"P{position.player_id}", 
                       (pixel_x + 10, pixel_y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Draw additional info
            info_text = f"Zone: {position.zone.zone_name if position.zone else 'None'}"
            cv2.putText(annotated, info_text, 
                       (pixel_x + 10, pixel_y + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
        
        return annotated
    
    def _get_zone_color(self, zone_type: str) -> Tuple[int, int, int]:
        """Get color for zone type."""
        color_map = {
            'defensive': (0, 0, 255),    # Blue
            'offensive': (0, 255, 0),    # Green
            'midfield': (255, 255, 0),   # Yellow
            'neutral': (128, 128, 128)   # Gray
        }
        return color_map.get(zone_type, (255, 255, 255))
