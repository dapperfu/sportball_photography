"""
Action Classification Module

This module provides action recognition capabilities for sports analysis,
classifying player actions like running, jumping, kicking, etc.

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
class ActionDetection:
    """Information about a detected player action."""
    action_type: str  # 'running', 'jumping', 'kicking', 'tackling', 'passing', 'shooting', 'standing'
    confidence: float
    player_bbox: Tuple[int, int, int, int]  # Player bounding box
    action_region: Tuple[int, int, int, int]  # Region where action occurs
    key_points: List[Tuple[int, int]]  # Key points for action
    motion_vector: Optional[Tuple[float, float]] = None
    speed_estimate: Optional[float] = None  # Estimated speed in pixels/frame


@dataclass
class ActionClassificationResult:
    """Result of action classification."""
    actions_detected: List[ActionDetection]
    processing_time: float
    success: bool
    error: Optional[str] = None


class ActionClassifier:
    """Action classification system for sports analysis."""
    
    def __init__(self, 
                 confidence_threshold: float = 0.6,
                 enable_speed_estimation: bool = True):
        """
        Initialize the action classifier.
        
        Args:
            confidence_threshold: Minimum confidence for action detection
            enable_speed_estimation: Enable player speed estimation
        """
        self.confidence_threshold = confidence_threshold
        self.enable_speed_estimation = enable_speed_estimation
        
        # Initialize MediaPipe pose detection for action analysis
        try:
            import mediapipe as mp
            self.mp_pose = mp.solutions.pose
            self.pose_detector = self.mp_pose.Pose(
                static_image_mode=True,
                model_complexity=2,
                enable_segmentation=False,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            self.mp_drawing = mp.solutions.drawing_utils
            self.mp_drawing_styles = mp.solutions.drawing_styles
            self.mediapipe_available = True
        except ImportError:
            logger.warning("MediaPipe not available, using OpenCV-based pose estimation")
            self.mediapipe_available = False
        
        # Action-specific parameters
        self.action_thresholds = {
            'running': {
                'leg_angle_range': (30, 150),  # Leg angle range for running
                'arm_swing_threshold': 0.3,   # Arm swing detection threshold
                'speed_threshold': 5.0        # Minimum speed for running
            },
            'jumping': {
                'vertical_displacement': 20,  # Minimum vertical movement
                'leg_extension': 0.8,         # Leg extension ratio
                'air_time_threshold': 0.2     # Minimum time in air
            },
            'kicking': {
                'leg_extension': 0.9,        # Leg extension for kick
                'foot_elevation': 0.3,       # Foot elevation above ground
                'ball_proximity': 30         # Distance to ball
            },
            'tackling': {
                'body_lean': 0.4,            # Body lean angle
                'arm_extension': 0.6,       # Arm extension for tackle
                'speed_threshold': 3.0      # Minimum speed for tackle
            },
            'passing': {
                'arm_extension': 0.7,       # Arm extension for pass
                'ball_proximity': 50,       # Distance to ball
                'body_orientation': 0.3      # Body orientation change
            },
            'shooting': {
                'leg_extension': 0.95,      # Maximum leg extension
                'ball_proximity': 40,       # Distance to ball
                'follow_through': 0.5       # Follow-through motion
            }
        }
        
        # Previous frame data for motion analysis
        self.previous_poses: List[Dict] = []
        
        logger.info(f"Initialized ActionClassifier with confidence threshold {confidence_threshold}")
    
    def classify_actions(self, image: np.ndarray, player_bboxes: List[Tuple[int, int, int, int]]) -> ActionClassificationResult:
        """
        Classify actions for detected players.
        
        Args:
            image: Input image (BGR format)
            player_bboxes: List of player bounding boxes (x, y, width, height)
            
        Returns:
            Action classification result
        """
        start_time = cv2.getTickCount()
        
        try:
            actions = []
            
            for bbox in player_bboxes:
                # Extract player region
                x, y, w, h = bbox
                player_region = image[y:y+h, x:x+w]
                
                if player_region.size == 0:
                    continue
                
                # Detect pose in player region
                pose_data = self._detect_pose(player_region)
                
                if pose_data:
                    # Classify action based on pose
                    action = self._classify_player_action(pose_data, bbox, image)
                    if action and action.confidence >= self.confidence_threshold:
                        actions.append(action)
            
            # Apply temporal analysis if previous frame data is available
            if self.previous_poses and self.enable_speed_estimation:
                actions = self._apply_temporal_analysis(actions)
            
            # Update previous frame data
            self.previous_poses = [action.__dict__ for action in actions]
            
            processing_time = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()
            
            return ActionClassificationResult(
                actions_detected=actions,
                processing_time=processing_time,
                success=True
            )
            
        except Exception as e:
            logger.error(f"Action classification failed: {e}")
            processing_time = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()
            return ActionClassificationResult(
                actions_detected=[],
                processing_time=processing_time,
                success=False,
                error=str(e)
            )
    
    def _detect_pose(self, player_region: np.ndarray) -> Optional[Dict]:
        """Detect pose in player region."""
        if self.mediapipe_available:
            return self._detect_pose_mediapipe(player_region)
        else:
            return self._detect_pose_opencv(player_region)
    
    def _detect_pose_mediapipe(self, player_region: np.ndarray) -> Optional[Dict]:
        """Detect pose using MediaPipe."""
        try:
            # Convert BGR to RGB
            rgb_region = cv2.cvtColor(player_region, cv2.COLOR_BGR2RGB)
            
            # Detect pose
            results = self.pose_detector.process(rgb_region)
            
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                
                # Extract key points
                key_points = {}
                for i, landmark in enumerate(landmarks):
                    if landmark.visibility > 0.5:  # Only visible landmarks
                        key_points[i] = {
                            'x': landmark.x,
                            'y': landmark.y,
                            'z': landmark.z,
                            'visibility': landmark.visibility
                        }
                
                return {
                    'landmarks': key_points,
                    'method': 'mediapipe',
                    'confidence': np.mean([lm['visibility'] for lm in key_points.values()])
                }
        except Exception as e:
            logger.debug(f"MediaPipe pose detection failed: {e}")
        
        return None
    
    def _detect_pose_opencv(self, player_region: np.ndarray) -> Optional[Dict]:
        """Detect pose using OpenCV-based approach."""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(player_region, cv2.COLOR_BGR2GRAY)
            
            # Apply edge detection
            edges = cv2.Canny(gray, 50, 150)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return None
            
            # Find largest contour (likely the player)
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Extract key points from contour
            key_points = self._extract_key_points_from_contour(largest_contour, player_region.shape)
            
            return {
                'landmarks': key_points,
                'method': 'opencv',
                'confidence': 0.6  # Default confidence for OpenCV method
            }
        except Exception as e:
            logger.debug(f"OpenCV pose detection failed: {e}")
        
        return None
    
    def _extract_key_points_from_contour(self, contour: np.ndarray, image_shape: Tuple[int, int]) -> Dict:
        """Extract key points from contour for OpenCV-based pose detection."""
        # Get contour moments
        M = cv2.moments(contour)
        
        if M["m00"] == 0:
            return {}
        
        # Calculate centroid
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)
        
        # Estimate key points based on contour geometry
        key_points = {
            0: {'x': cx / image_shape[1], 'y': cy / image_shape[0], 'visibility': 0.8},  # Center
            11: {'x': (x + w * 0.2) / image_shape[1], 'y': (y + h * 0.3) / image_shape[0], 'visibility': 0.7},  # Left shoulder
            12: {'x': (x + w * 0.8) / image_shape[1], 'y': (y + h * 0.3) / image_shape[0], 'visibility': 0.7},  # Right shoulder
            23: {'x': (x + w * 0.3) / image_shape[1], 'y': (y + h * 0.7) / image_shape[0], 'visibility': 0.7},  # Left hip
            24: {'x': (x + w * 0.7) / image_shape[1], 'y': (y + h * 0.7) / image_shape[0], 'visibility': 0.7},  # Right hip
            25: {'x': (x + w * 0.2) / image_shape[1], 'y': (y + h * 0.9) / image_shape[0], 'visibility': 0.6},  # Left knee
            26: {'x': (x + w * 0.8) / image_shape[1], 'y': (y + h * 0.9) / image_shape[0], 'visibility': 0.6},  # Right knee
        }
        
        return key_points
    
    def _classify_player_action(self, pose_data: Dict, bbox: Tuple[int, int, int, int], image: np.ndarray) -> Optional[ActionDetection]:
        """Classify action for a single player based on pose data."""
        landmarks = pose_data['landmarks']
        
        if not landmarks:
            return None
        
        # Calculate pose features
        features = self._calculate_pose_features(landmarks)
        
        # Classify action based on features
        action_scores = {}
        
        # Running detection
        running_score = self._detect_running(features)
        action_scores['running'] = running_score
        
        # Jumping detection
        jumping_score = self._detect_jumping(features)
        action_scores['jumping'] = jumping_score
        
        # Kicking detection
        kicking_score = self._detect_kicking(features)
        action_scores['kicking'] = kicking_score
        
        # Tackling detection
        tackling_score = self._detect_tackling(features)
        action_scores['tackling'] = tackling_score
        
        # Passing detection
        passing_score = self._detect_passing(features)
        action_scores['passing'] = passing_score
        
        # Shooting detection
        shooting_score = self._detect_shooting(features)
        action_scores['shooting'] = shooting_score
        
        # Standing detection (default)
        standing_score = self._detect_standing(features)
        action_scores['standing'] = standing_score
        
        # Find best action
        best_action = max(action_scores.items(), key=lambda x: x[1])
        action_type, confidence = best_action
        
        if confidence < self.confidence_threshold:
            return None
        
        # Create action detection
        action_region = self._calculate_action_region(landmarks, bbox)
        key_points = self._extract_action_key_points(landmarks, bbox)
        
        return ActionDetection(
            action_type=action_type,
            confidence=confidence,
            player_bbox=bbox,
            action_region=action_region,
            key_points=key_points
        )
    
    def _calculate_pose_features(self, landmarks: Dict) -> Dict[str, float]:
        """Calculate features from pose landmarks."""
        features = {}
        
        # Convert landmarks to pixel coordinates (assuming normalized coordinates)
        pixel_landmarks = {}
        for idx, lm in landmarks.items():
            pixel_landmarks[idx] = {
                'x': lm['x'] * 1000,  # Scale to reasonable pixel values
                'y': lm['y'] * 1000,
                'visibility': lm['visibility']
            }
        
        # Calculate key distances and angles
        if 11 in pixel_landmarks and 12 in pixel_landmarks:  # Shoulders
            shoulder_width = abs(pixel_landmarks[12]['x'] - pixel_landmarks[11]['x'])
            features['shoulder_width'] = shoulder_width
        
        if 23 in pixel_landmarks and 24 in pixel_landmarks:  # Hips
            hip_width = abs(pixel_landmarks[24]['x'] - pixel_landmarks[23]['x'])
            features['hip_width'] = hip_width
        
        # Calculate leg angles
        if 23 in pixel_landmarks and 25 in pixel_landmarks:  # Left leg
            left_leg_angle = self._calculate_angle(
                pixel_landmarks[23], pixel_landmarks[25]
            )
            features['left_leg_angle'] = left_leg_angle
        
        if 24 in pixel_landmarks and 26 in pixel_landmarks:  # Right leg
            right_leg_angle = self._calculate_angle(
                pixel_landmarks[24], pixel_landmarks[26]
            )
            features['right_leg_angle'] = right_leg_angle
        
        # Calculate body lean
        if 11 in pixel_landmarks and 12 in pixel_landmarks and 23 in pixel_landmarks and 24 in pixel_landmarks:
            shoulder_center = {
                'x': (pixel_landmarks[11]['x'] + pixel_landmarks[12]['x']) / 2,
                'y': (pixel_landmarks[11]['y'] + pixel_landmarks[12]['y']) / 2
            }
            hip_center = {
                'x': (pixel_landmarks[23]['x'] + pixel_landmarks[24]['x']) / 2,
                'y': (pixel_landmarks[23]['y'] + pixel_landmarks[24]['y']) / 2
            }
            body_lean = abs(shoulder_center['x'] - hip_center['x'])
            features['body_lean'] = body_lean
        
        return features
    
    def _calculate_angle(self, point1: Dict, point2: Dict) -> float:
        """Calculate angle between two points."""
        dx = point2['x'] - point1['x']
        dy = point2['y'] - point1['y']
        angle = math.degrees(math.atan2(dy, dx))
        return abs(angle)
    
    def _detect_running(self, features: Dict) -> float:
        """Detect running action."""
        score = 0.0
        
        # Check leg angles
        if 'left_leg_angle' in features and 'right_leg_angle' in features:
            left_angle = features['left_leg_angle']
            right_angle = features['right_leg_angle']
            
            # Running typically has alternating leg positions
            angle_diff = abs(left_angle - right_angle)
            if 30 <= angle_diff <= 120:  # Good range for running
                score += 0.4
        
        # Check body lean (runners lean forward)
        if 'body_lean' in features:
            if features['body_lean'] > 10:  # Significant lean
                score += 0.3
        
        return min(score, 1.0)
    
    def _detect_jumping(self, features: Dict) -> float:
        """Detect jumping action."""
        score = 0.0
        
        # Check leg extension
        if 'left_leg_angle' in features and 'right_leg_angle' in features:
            left_angle = features['left_leg_angle']
            right_angle = features['right_leg_angle']
            
            # Jumping typically has extended legs
            avg_leg_angle = (left_angle + right_angle) / 2
            if avg_leg_angle > 120:  # Extended legs
                score += 0.5
        
        return min(score, 1.0)
    
    def _detect_kicking(self, features: Dict) -> float:
        """Detect kicking action."""
        score = 0.0
        
        # Check leg extension (one leg extended)
        if 'left_leg_angle' in features and 'right_leg_angle' in features:
            left_angle = features['left_leg_angle']
            right_angle = features['right_leg_angle']
            
            # One leg should be more extended than the other
            angle_diff = abs(left_angle - right_angle)
            if angle_diff > 30:  # Significant difference
                score += 0.6
        
        return min(score, 1.0)
    
    def _detect_tackling(self, features: Dict) -> float:
        """Detect tackling action."""
        score = 0.0
        
        # Check body lean (tacklers lean forward)
        if 'body_lean' in features:
            if features['body_lean'] > 15:  # Strong lean
                score += 0.5
        
        # Check leg angles (low stance)
        if 'left_leg_angle' in features and 'right_leg_angle' in features:
            avg_leg_angle = (features['left_leg_angle'] + features['right_leg_angle']) / 2
            if avg_leg_angle < 90:  # Low stance
                score += 0.4
        
        return min(score, 1.0)
    
    def _detect_passing(self, features: Dict) -> float:
        """Detect passing action."""
        score = 0.0
        
        # Check body orientation (passing involves body rotation)
        if 'body_lean' in features:
            if 5 <= features['body_lean'] <= 20:  # Moderate lean
                score += 0.4
        
        return min(score, 1.0)
    
    def _detect_shooting(self, features: Dict) -> float:
        """Detect shooting action."""
        score = 0.0
        
        # Check leg extension (strong leg extension for shot)
        if 'left_leg_angle' in features and 'right_leg_angle' in features:
            max_leg_angle = max(features['left_leg_angle'], features['right_leg_angle'])
            if max_leg_angle > 140:  # Very extended leg
                score += 0.6
        
        return min(score, 1.0)
    
    def _detect_standing(self, features: Dict) -> float:
        """Detect standing action (default state)."""
        score = 0.5  # Default score for standing
        
        # Check if legs are in neutral position
        if 'left_leg_angle' in features and 'right_leg_angle' in features:
            left_angle = features['left_leg_angle']
            right_angle = features['right_leg_angle']
            
            # Standing typically has similar leg angles
            angle_diff = abs(left_angle - right_angle)
            if angle_diff < 20:  # Similar leg positions
                score += 0.3
        
        return min(score, 1.0)
    
    def _calculate_action_region(self, landmarks: Dict, bbox: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        """Calculate region where action occurs."""
        x, y, w, h = bbox
        
        # For now, return the player bounding box
        # In more advanced implementations, this could be more specific
        return bbox
    
    def _extract_action_key_points(self, landmarks: Dict, bbox: Tuple[int, int, int, int]) -> List[Tuple[int, int]]:
        """Extract key points for the action."""
        x, y, w, h = bbox
        key_points = []
        
        # Convert normalized coordinates to pixel coordinates
        for idx, lm in landmarks.items():
            if lm['visibility'] > 0.5:
                pixel_x = int(x + lm['x'] * w)
                pixel_y = int(y + lm['y'] * h)
                key_points.append((pixel_x, pixel_y))
        
        return key_points
    
    def _apply_temporal_analysis(self, actions: List[ActionDetection]) -> List[ActionDetection]:
        """Apply temporal analysis for motion tracking."""
        # This is a simplified implementation
        # In a full implementation, this would track motion between frames
        for action in actions:
            # Estimate speed based on motion vector
            if action.motion_vector:
                dx, dy = action.motion_vector
                speed = math.sqrt(dx*dx + dy*dy)
                action.speed_estimate = speed
        
        return actions
    
    def draw_action_detections(self, image: np.ndarray, actions: List[ActionDetection]) -> np.ndarray:
        """Draw action detections on the image."""
        annotated = image.copy()
        
        # Color mapping for different actions
        action_colors = {
            'running': (0, 255, 0),    # Green
            'jumping': (255, 0, 0),    # Blue
            'kicking': (0, 0, 255),    # Red
            'tackling': (255, 255, 0), # Cyan
            'passing': (255, 0, 255),  # Magenta
            'shooting': (0, 255, 255), # Yellow
            'standing': (128, 128, 128) # Gray
        }
        
        for action in actions:
            color = action_colors.get(action.action_type, (255, 255, 255))
            
            # Draw player bounding box
            x, y, w, h = action.player_bbox
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
            
            # Draw action region
            ax, ay, aw, ah = action.action_region
            cv2.rectangle(annotated, (ax, ay), (ax + aw, ay + ah), color, 1)
            
            # Draw key points
            for kx, ky in action.key_points:
                cv2.circle(annotated, (kx, ky), 3, color, -1)
            
            # Draw label
            label = f"{action.action_type} ({action.confidence:.2f})"
            if action.speed_estimate:
                label += f" Speed: {action.speed_estimate:.1f}"
            
            cv2.putText(annotated, label, (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return annotated
