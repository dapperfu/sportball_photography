#!/usr/bin/env python3
"""
Phase 1 Visual Analysis Tool

This tool generates comprehensive annotated outputs with all Phase 1 analysis overlays
for visual verification of detection accuracy.

Author: Claude Sonnet 4 (claude-3-5-sonnet-20241022)
Generated via Cursor IDE (cursor.sh) with AI assistance
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Any
import click
from loguru import logger
import math


class Phase1VisualAnalyzer:
    """Visual analysis tool for Phase 1 features with comprehensive overlays."""
    
    def __init__(self):
        """Initialize the visual analyzer."""
        # Initialize OpenCV face detection
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Initialize MediaPipe pose detection
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
            logger.warning("MediaPipe not available")
            self.mediapipe_available = False
        
        logger.info("Initialized Phase1VisualAnalyzer")
    
    def analyze_image(self, image_path: Path) -> Dict[str, Any]:
        """Perform comprehensive visual analysis on an image."""
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            return {"error": "Failed to load image"}
        
        # Perform all analyses
        ball_detections = self._detect_balls(image)
        face_detections = self._detect_faces(image)
        pose_detections = self._detect_poses(image)
        action_analysis = self._analyze_actions(image, face_detections)
        field_analysis = self._analyze_field_positions(image, face_detections)
        quality_assessment = self._assess_quality(image)
        
        return {
            "image_path": str(image_path),
            "ball_detections": ball_detections,
            "face_detections": face_detections,
            "pose_detections": pose_detections,
            "action_analysis": action_analysis,
            "field_analysis": field_analysis,
            "quality_assessment": quality_assessment,
            "image_shape": image.shape
        }
    
    def _detect_balls(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Detect balls using multiple methods."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        balls = []
        
        # Method 1: Hough Circles
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=20,
            param1=50,
            param2=30,
            minRadius=5,
            maxRadius=100
        )
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for i, (x, y, r) in enumerate(circles):
                # Classify ball type based on color
                ball_type = self._classify_ball_type(image, x, y, r)
                balls.append({
                    "id": i,
                    "center": (x, y),
                    "radius": r,
                    "type": ball_type,
                    "method": "hough_circles",
                    "confidence": 0.7
                })
        
        # Method 2: Template Matching
        template_balls = self._detect_balls_template_matching(gray)
        balls.extend(template_balls)
        
        # Method 3: Color-based detection
        color_balls = self._detect_balls_color_based(image)
        balls.extend(color_balls)
        
        return balls
    
    def _detect_balls_template_matching(self, gray: np.ndarray) -> List[Dict[str, Any]]:
        """Detect balls using template matching."""
        balls = []
        template_sizes = [10, 15, 20, 25, 30, 40, 50]
        
        for size in template_sizes:
            # Create circular template
            template = np.zeros((size*2+1, size*2+1), dtype=np.uint8)
            cv2.circle(template, (size, size), size, 255, -1)
            
            # Apply template matching
            result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
            locations = np.where(result >= 0.5)
            
            for pt in zip(*locations[::-1]):
                x, y = pt
                balls.append({
                    "id": len(balls),
                    "center": (x + size, y + size),
                    "radius": size,
                    "type": "unknown",
                    "method": "template_matching",
                    "confidence": result[y, x]
                })
        
        return balls
    
    def _detect_balls_color_based(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Detect balls based on color signatures."""
        balls = []
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Define color ranges for different ball types
        color_ranges = {
            'soccer': [(0, 0, 200), (180, 30, 255)],  # White
            'basketball': [(5, 100, 100), (15, 255, 255)],  # Orange
            'tennis': [(20, 100, 100), (30, 255, 255)],  # Yellow
            'rugby': [(10, 50, 50), (20, 255, 200)]  # Brown
        }
        
        for ball_type, (lower, upper) in color_ranges.items():
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 50:  # Minimum area
                    perimeter = cv2.arcLength(contour, True)
                    if perimeter > 0:
                        circularity = 4 * np.pi * area / (perimeter * perimeter)
                        if circularity > 0.7:  # Roughly circular
                            (x, y), radius = cv2.minEnclosingCircle(contour)
                            balls.append({
                                "id": len(balls),
                                "center": (int(x), int(y)),
                                "radius": int(radius),
                                "type": ball_type,
                                "method": "color_based",
                                "confidence": circularity
                            })
        
        return balls
    
    def _classify_ball_type(self, image: np.ndarray, x: int, y: int, r: int) -> str:
        """Classify ball type based on color analysis."""
        # Extract ball region
        x1, y1 = max(0, x-r), max(0, y-r)
        x2, y2 = min(image.shape[1], x+r), min(image.shape[0], y+r)
        ball_region = image[y1:y2, x1:x2]
        
        if ball_region.size == 0:
            return "unknown"
        
        # Convert to HSV
        hsv_region = cv2.cvtColor(ball_region, cv2.COLOR_BGR2HSV)
        
        # Calculate dominant colors
        hist_h = cv2.calcHist([hsv_region], [0], None, [180], [0, 180])
        dominant_hue = np.argmax(hist_h)
        
        # Classify based on hue
        if dominant_hue < 20 or dominant_hue > 160:  # White/light
            return "soccer"
        elif 5 <= dominant_hue <= 15:  # Orange/brown
            return "basketball"
        elif 20 <= dominant_hue <= 30:  # Yellow
            return "tennis"
        elif 10 <= dominant_hue <= 20:  # Brown/orange
            return "rugby"
        
        return "unknown"
    
    def _detect_faces(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Detect faces in the image."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        
        face_detections = []
        for i, (x, y, w, h) in enumerate(faces):
            face_detections.append({
                "id": i,
                "bbox": (x, y, w, h),
                "center": (x + w//2, y + h//2),
                "confidence": 0.8
            })
        
        return face_detections
    
    def _detect_poses(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Detect poses in the image."""
        if not self.mediapipe_available:
            return []
        
        try:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.pose_detector.process(rgb_image)
            
            pose_detections = []
            if results.pose_landmarks:
                landmarks = []
                for landmark in results.pose_landmarks.landmark:
                    if landmark.visibility > 0.5:
                        x = int(landmark.x * image.shape[1])
                        y = int(landmark.y * image.shape[0])
                        landmarks.append((x, y))
                
                pose_detections.append({
                    "id": 0,
                    "landmarks": landmarks,
                    "confidence": 0.8
                })
            
            return pose_detections
        except Exception as e:
            logger.debug(f"Pose detection failed: {e}")
            return []
    
    def _analyze_actions(self, image: np.ndarray, face_detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze player actions based on pose data."""
        actions = []
        
        if not self.mediapipe_available:
            return actions
        
        try:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.pose_detector.process(rgb_image)
            
            if results.pose_landmarks:
                # Analyze pose for actions
                landmarks = results.pose_landmarks.landmark
                
                # Check for running (leg movement)
                if self._is_running(landmarks):
                    actions.append({
                        "type": "running",
                        "confidence": 0.7,
                        "region": self._get_action_region(landmarks, image.shape)
                    })
                
                # Check for jumping (leg extension)
                if self._is_jumping(landmarks):
                    actions.append({
                        "type": "jumping",
                        "confidence": 0.6,
                        "region": self._get_action_region(landmarks, image.shape)
                    })
                
                # Check for kicking (one leg extended)
                if self._is_kicking(landmarks):
                    actions.append({
                        "type": "kicking",
                        "confidence": 0.8,
                        "region": self._get_action_region(landmarks, image.shape)
                    })
                
                # Default to standing if no specific action detected
                if not actions:
                    actions.append({
                        "type": "standing",
                        "confidence": 0.5,
                        "region": self._get_action_region(landmarks, image.shape)
                    })
        
        except Exception as e:
            logger.debug(f"Action analysis failed: {e}")
        
        return actions
    
    def _is_running(self, landmarks) -> bool:
        """Check if pose indicates running."""
        # Simplified running detection based on leg angles
        try:
            left_hip = landmarks[23]
            left_knee = landmarks[25]
            right_hip = landmarks[24]
            right_knee = landmarks[26]
            
            # Calculate leg angles (simplified)
            left_angle = abs(left_hip.y - left_knee.y)
            right_angle = abs(right_hip.y - right_knee.y)
            
            # Running typically has alternating leg positions
            return abs(left_angle - right_angle) > 0.1
        except:
            return False
    
    def _is_jumping(self, landmarks) -> bool:
        """Check if pose indicates jumping."""
        try:
            left_knee = landmarks[25]
            right_knee = landmarks[26]
            
            # Jumping typically has extended legs
            avg_knee_height = (left_knee.y + right_knee.y) / 2
            return avg_knee_height < 0.3  # Knees high up
        except:
            return False
    
    def _is_kicking(self, landmarks) -> bool:
        """Check if pose indicates kicking."""
        try:
            left_knee = landmarks[25]
            right_knee = landmarks[26]
            
            # Kicking typically has one leg more extended
            return abs(left_knee.y - right_knee.y) > 0.2
        except:
            return False
    
    def _get_action_region(self, landmarks, image_shape: Tuple[int, int, int]) -> Tuple[int, int, int, int]:
        """Get bounding box for action region."""
        try:
            # Get bounding box of all visible landmarks
            visible_landmarks = [lm for lm in landmarks if lm.visibility > 0.5]
            if not visible_landmarks:
                return (0, 0, image_shape[1], image_shape[0])
            
            x_coords = [int(lm.x * image_shape[1]) for lm in visible_landmarks]
            y_coords = [int(lm.y * image_shape[0]) for lm in visible_landmarks]
            
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
            
            return (x_min, y_min, x_max - x_min, y_max - y_min)
        except:
            return (0, 0, image_shape[1], image_shape[0])
    
    def _analyze_field_positions(self, image: np.ndarray, face_detections: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze field positions of detected players."""
        height, width = image.shape[:2]
        
        field_analysis = {
            "field_detected": True,  # Simplified - assume field is detected
            "zones": [
                {"name": "left_goal", "type": "defensive", "center": (width//8, height//2)},
                {"name": "center_field", "type": "neutral", "center": (width//2, height//2)},
                {"name": "right_goal", "type": "defensive", "center": (7*width//8, height//2)}
            ],
            "player_positions": []
        }
        
        for face in face_detections:
            x, y, w, h = face["bbox"]
            center_x, center_y = face["center"]
            
            # Normalize position
            norm_x = center_x / width
            norm_y = center_y / height
            
            # Determine field side
            if norm_x < 0.33:
                field_side = "left"
            elif norm_x > 0.67:
                field_side = "right"
            else:
                field_side = "center"
            
            # Determine zone
            zone = "center_field"
            if norm_x < 0.25:
                zone = "left_goal"
            elif norm_x > 0.75:
                zone = "right_goal"
            
            field_analysis["player_positions"].append({
                "player_id": face["id"],
                "field_x": norm_x,
                "field_y": norm_y,
                "field_side": field_side,
                "zone": zone,
                "is_offside": norm_x > 0.6 and norm_y < 0.3  # Simplified offside
            })
        
        return field_analysis
    
    def _assess_quality(self, image: np.ndarray) -> Dict[str, Any]:
        """Assess photo quality."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Sharpness assessment
        laplacian_var = cv2.Laplacian(gray, cv2.COLOR_BGR2GRAY).var()
        if laplacian_var > 1000:
            sharpness_score = 1.0
        elif laplacian_var > 500:
            sharpness_score = 0.8
        elif laplacian_var > 200:
            sharpness_score = 0.6
        elif laplacian_var > 100:
            sharpness_score = 0.4
        else:
            sharpness_score = 0.2
        
        # Exposure assessment
        mean_brightness = np.mean(gray)
        if 50 <= mean_brightness <= 200:
            exposure_score = 1.0
        elif 30 <= mean_brightness <= 220:
            exposure_score = 0.8
        else:
            exposure_score = 0.4
        
        # Overall quality
        overall_score = (sharpness_score + exposure_score) / 2
        
        # Quality grade
        if overall_score >= 0.85:
            grade = "A"
        elif overall_score >= 0.70:
            grade = "B"
        elif overall_score >= 0.55:
            grade = "C"
        elif overall_score >= 0.40:
            grade = "D"
        else:
            grade = "F"
        
        return {
            "overall_score": overall_score,
            "sharpness_score": sharpness_score,
            "exposure_score": exposure_score,
            "grade": grade,
            "recommendations": self._get_quality_recommendations(sharpness_score, exposure_score)
        }
    
    def _get_quality_recommendations(self, sharpness: float, exposure: float) -> List[str]:
        """Get quality improvement recommendations."""
        recommendations = []
        
        if sharpness < 0.4:
            recommendations.append("Use faster shutter speed to reduce blur")
        if exposure < 0.4:
            recommendations.append("Adjust exposure settings")
        
        if not recommendations:
            recommendations.append("Photo quality is good")
        
        return recommendations
    
    def create_annotated_image(self, analysis: Dict[str, Any]) -> np.ndarray:
        """Create annotated image with all analysis overlays."""
        image = cv2.imread(analysis["image_path"])
        if image is None:
            return None
        
        annotated = image.copy()
        
        # Draw ball detections
        annotated = self._draw_ball_detections(annotated, analysis["ball_detections"])
        
        # Draw face detections
        annotated = self._draw_face_detections(annotated, analysis["face_detections"])
        
        # Draw pose detections
        annotated = self._draw_pose_detections(annotated, analysis["pose_detections"])
        
        # Draw action analysis
        annotated = self._draw_action_analysis(annotated, analysis["action_analysis"])
        
        # Draw field analysis
        annotated = self._draw_field_analysis(annotated, analysis["field_analysis"])
        
        # Draw quality assessment
        annotated = self._draw_quality_assessment(annotated, analysis["quality_assessment"])
        
        return annotated
    
    def _draw_ball_detections(self, image: np.ndarray, ball_detections: List[Dict[str, Any]]) -> np.ndarray:
        """Draw ball detection overlays."""
        for ball in ball_detections:
            x, y = ball["center"]
            r = ball["radius"]
            ball_type = ball["type"]
            method = ball["method"]
            confidence = ball["confidence"]
            
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
            cv2.circle(image, (x, y), r, color, 2)
            
            # Draw label
            label = f"{ball_type} ({method}) {confidence:.2f}"
            cv2.putText(image, label, (x - r, y - r - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return image
    
    def _draw_face_detections(self, image: np.ndarray, face_detections: List[Dict[str, Any]]) -> np.ndarray:
        """Draw face detection overlays."""
        for face in face_detections:
            x, y, w, h = face["bbox"]
            face_id = face["id"]
            confidence = face["confidence"]
            
            # Draw bounding box
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
            
            # Draw label
            label = f"Face {face_id} ({confidence:.2f})"
            cv2.putText(image, label, (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        return image
    
    def _draw_pose_detections(self, image: np.ndarray, pose_detections: List[Dict[str, Any]]) -> np.ndarray:
        """Draw pose detection overlays."""
        if not self.mediapipe_available:
            return image
        
        for pose in pose_detections:
            landmarks = pose["landmarks"]
            
            # Draw landmarks
            for x, y in landmarks:
                cv2.circle(image, (x, y), 3, (0, 255, 255), -1)
            
            # Draw pose connections (simplified)
            if len(landmarks) >= 2:
                for i in range(len(landmarks) - 1):
                    cv2.line(image, landmarks[i], landmarks[i + 1], (0, 255, 255), 1)
        
        return image
    
    def _draw_action_analysis(self, image: np.ndarray, action_analysis: List[Dict[str, Any]]) -> np.ndarray:
        """Draw action analysis overlays."""
        for action in action_analysis:
            action_type = action["type"]
            confidence = action["confidence"]
            region = action["region"]
            
            # Color based on action type
            colors = {
                "running": (0, 255, 0),    # Green
                "jumping": (255, 0, 0),   # Blue
                "kicking": (0, 0, 255),   # Red
                "tackling": (255, 255, 0), # Cyan
                "passing": (255, 0, 255), # Magenta
                "shooting": (0, 255, 255), # Yellow
                "standing": (128, 128, 128) # Gray
            }
            color = colors.get(action_type, (255, 255, 255))
            
            # Draw action region
            x, y, w, h = region
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            
            # Draw label
            label = f"{action_type} ({confidence:.2f})"
            cv2.putText(image, label, (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return image
    
    def _draw_field_analysis(self, image: np.ndarray, field_analysis: Dict[str, Any]) -> np.ndarray:
        """Draw field analysis overlays."""
        height, width = image.shape[:2]
        
        # Draw field zones
        for zone in field_analysis["zones"]:
            center = zone["center"]
            zone_name = zone["name"]
            zone_type = zone["type"]
            
            # Color based on zone type
            colors = {
                "defensive": (0, 0, 255),    # Red
                "neutral": (128, 128, 128),  # Gray
                "offensive": (0, 255, 0)     # Green
            }
            color = colors.get(zone_type, (255, 255, 255))
            
            # Draw zone center
            cv2.circle(image, center, 10, color, -1)
            
            # Draw zone label
            cv2.putText(image, zone_name, (center[0] - 50, center[1] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Draw player positions
        for pos in field_analysis["player_positions"]:
            player_id = pos["player_id"]
            field_x = pos["field_x"]
            field_y = pos["field_y"]
            field_side = pos["field_side"]
            zone = pos["zone"]
            is_offside = pos["is_offside"]
            
            # Convert to pixel coordinates
            pixel_x = int(field_x * width)
            pixel_y = int(field_y * height)
            
            # Color based on position
            if is_offside:
                color = (0, 0, 255)  # Red for offside
            elif field_side == "left":
                color = (255, 0, 0)  # Blue for left side
            elif field_side == "right":
                color = (0, 255, 0)  # Green for right side
            else:
                color = (255, 255, 0)  # Yellow for center
            
            # Draw player position
            cv2.circle(image, (pixel_x, pixel_y), 8, color, -1)
            
            # Draw player info
            label = f"P{player_id} ({zone})"
            if is_offside:
                label += " OFFSIDE"
            
            cv2.putText(image, label, (pixel_x + 10, pixel_y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        return image
    
    def _draw_quality_assessment(self, image: np.ndarray, quality_assessment: Dict[str, Any]) -> np.ndarray:
        """Draw quality assessment overlays."""
        overall_score = quality_assessment["overall_score"]
        grade = quality_assessment["grade"]
        sharpness = quality_assessment["sharpness_score"]
        exposure = quality_assessment["exposure_score"]
        
        # Color based on grade
        colors = {
            "A": (0, 255, 0),    # Green
            "B": (0, 255, 255),  # Yellow
            "C": (0, 165, 255),  # Orange
            "D": (0, 0, 255),    # Red
            "F": (0, 0, 128)     # Dark Red
        }
        color = colors.get(grade, (255, 255, 255))
        
        # Draw quality info
        cv2.putText(image, f"Quality Grade: {grade}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(image, f"Overall Score: {overall_score:.2f}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(image, f"Sharpness: {sharpness:.2f}", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        cv2.putText(image, f"Exposure: {exposure:.2f}", (10, 110),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return image


@click.command()
@click.argument('input_pattern', type=str)
@click.argument('output_dir', type=str)
@click.option('--max-images', '-m', default=5, type=int, help='Maximum number of images to process')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
def main(input_pattern: str, output_dir: str, max_images: int, verbose: bool):
    """Generate Phase 1 analysis with visual overlays for verification."""
    
    # Setup logging
    if verbose:
        logger.add("phase1_visual_analysis.log", level="DEBUG")
    
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
    
    logger.info(f"Processing {len(image_files)} images for Phase 1 visual analysis")
    
    # Initialize analyzer
    analyzer = Phase1VisualAnalyzer()
    
    # Process images
    for image_path in image_files:
        logger.info(f"Analyzing {image_path.name}")
        
        # Perform analysis
        analysis = analyzer.analyze_image(image_path)
        
        if "error" in analysis:
            logger.error(f"Analysis failed for {image_path.name}: {analysis['error']}")
            continue
        
        # Create annotated image
        annotated = analyzer.create_annotated_image(analysis)
        
        if annotated is not None:
            # Save annotated image
            output_filename = f"{image_path.stem}_phase1_analysis.jpg"
            output_file = output_path / output_filename
            cv2.imwrite(str(output_file), annotated)
            
            logger.info(f"Saved annotated image: {output_filename}")
            
            # Print analysis summary
            logger.info(f"  Balls detected: {len(analysis['ball_detections'])}")
            logger.info(f"  Faces detected: {len(analysis['face_detections'])}")
            logger.info(f"  Poses detected: {len(analysis['pose_detections'])}")
            logger.info(f"  Actions detected: {len(analysis['action_analysis'])}")
            logger.info(f"  Quality grade: {analysis['quality_assessment']['grade']}")
    
    logger.info(f"Phase 1 visual analysis complete! Results saved to {output_path}")


if __name__ == '__main__':
    main()
