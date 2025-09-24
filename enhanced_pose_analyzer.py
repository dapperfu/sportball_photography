#!/usr/bin/env python3
"""
Enhanced Pose Analysis Tool for Soccer Photos

This tool performs comprehensive pose analysis on soccer photos, assuming humans
are mostly full-size in the shots. It creates detailed JSON sidecar files with
analysis results and provides progress tracking with tqdm.

Features:
- Multi-person pose detection using MediaPipe
- Detailed pose keypoint analysis
- Jersey region extraction and analysis
- Body region analysis for number detection
- JSON sidecar files for all analysis results
- Progress tracking with tqdm
- Configurable limits for testing
- Comprehensive error handling and logging
"""

import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

import click
import cv2
import numpy as np
from tqdm import tqdm
from loguru import logger
import mediapipe as mp


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle numpy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from soccer_photo_sorter.detectors import PoseDetector
from soccer_photo_sorter.core.image_processor import ImageProcessor
from soccer_photo_sorter.config.settings import Settings


class EnhancedPoseAnalyzer:
    """Enhanced pose analyzer with comprehensive analysis and JSON output."""
    
    def __init__(self, 
                 pose_confidence: float = 0.5,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5,
                 enable_multi_person: bool = True):
        """
        Initialize enhanced pose analyzer.
        
        Args:
            pose_confidence: Minimum confidence for pose detection
            min_detection_confidence: MediaPipe minimum detection confidence
            min_tracking_confidence: MediaPipe minimum tracking confidence
            enable_multi_person: Enable multi-person detection
        """
        self.pose_confidence = pose_confidence
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.enable_multi_person = enable_multi_person
        
        # Initialize components
        self.settings = Settings()
        self.image_processor = ImageProcessor(self.settings)
        self.pose_detector = PoseDetector(
            confidence_threshold=pose_confidence,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        # Enable multi-person detection
        self.pose_detector.enable_multi_person_detection(enable_multi_person)
        
        # Initialize MediaPipe drawing utilities for visualizations
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Analysis statistics
        self.stats = {
            'total_images': 0,
            'processed_images': 0,
            'successful_analyses': 0,
            'failed_analyses': 0,
            'total_people_detected': 0,
            'total_jersey_regions': 0,
            'total_body_regions': 0,
            'processing_time': 0.0,
            'errors': []
        }
    
    def analyze_image(self, image_path: Path) -> Dict[str, Any]:
        """
        Perform comprehensive pose analysis on a single image.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Comprehensive analysis results dictionary
        """
        start_time = time.time()
        
        try:
            # Load and preprocess image
            image = self.image_processor.load_image(image_path)
            if image is None:
                return self._create_error_result(image_path, "Failed to load image")
            
            # Resize for processing
            original_size = image.shape[:2]
            processed_image = self.image_processor.resize_image(image)
            processed_size = processed_image.shape[:2]
            
            # Perform pose detection
            poses = self.pose_detector.detect_poses(processed_image)
            
            # Extract jersey regions
            jersey_regions = self.pose_detector.extract_jersey_regions(processed_image)
            
            # Extract body regions
            body_regions = self.pose_detector.extract_body_regions(processed_image)
            
            # Analyze each pose in detail
            detailed_poses = []
            for i, pose_data in enumerate(poses):
                pose_analysis = self._analyze_pose_detailed(pose_data, i, original_size, processed_size)
                detailed_poses.append(pose_analysis)
            
            # Analyze jersey regions
            detailed_jersey_regions = []
            for i, region in enumerate(jersey_regions):
                jersey_analysis = self._analyze_jersey_region(region, i, processed_image)
                detailed_jersey_regions.append(jersey_analysis)
            
            # Analyze body regions
            detailed_body_regions = []
            for i, region in enumerate(body_regions):
                body_analysis = self._analyze_body_region(region, i, processed_image)
                detailed_body_regions.append(body_analysis)
            
            # Create comprehensive result
            analysis_result = {
                'image_info': {
                    'file_path': str(image_path),
                    'file_name': image_path.name,
                    'file_size_bytes': image_path.stat().st_size if image_path.exists() else 0,
                    'original_size': {'width': original_size[1], 'height': original_size[0]},
                    'processed_size': {'width': processed_size[1], 'height': processed_size[0]},
                    'scale_factor': {
                        'width': original_size[1] / processed_size[1],
                        'height': original_size[0] / processed_size[0]
                    }
                },
                'pose_analysis': {
                    'num_people_detected': len(poses),
                    'poses': detailed_poses,
                    'detection_method': 'multi_person' if self.enable_multi_person else 'single_person',
                    'confidence_threshold': self.pose_confidence
                },
                'jersey_analysis': {
                    'num_jersey_regions': len(jersey_regions),
                    'regions': detailed_jersey_regions
                },
                'body_analysis': {
                    'num_body_regions': len(body_regions),
                    'regions': detailed_body_regions
                },
                'processing_info': {
                    'processing_time_seconds': time.time() - start_time,
                    'timestamp': datetime.now().isoformat(),
                    'analyzer_version': '1.0.0',
                    'settings': {
                        'pose_confidence': self.pose_confidence,
                        'min_detection_confidence': self.min_detection_confidence,
                        'min_tracking_confidence': self.min_tracking_confidence,
                        'enable_multi_person': self.enable_multi_person
                    }
                }
            }
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"Error analyzing image {image_path}: {e}")
            return self._create_error_result(image_path, str(e))
    
    def _analyze_pose_detailed(self, pose_data: Dict[str, Any], person_id: int, 
                              original_size: Tuple[int, int], processed_size: Tuple[int, int]) -> Dict[str, Any]:
        """Analyze a single pose in detail."""
        keypoints = pose_data['keypoints']
        confidence = pose_data['confidence']
        
        # Analyze keypoint visibility and positions
        keypoint_analysis = {}
        visible_keypoints = 0
        total_keypoints = len(keypoints)
        
        for kp_id, keypoint in keypoints.items():
            visibility = keypoint.visibility
            is_visible = visibility > 0.5
            
            keypoint_analysis[kp_id] = {
                'x': float(keypoint.x),
                'y': float(keypoint.y),
                'visibility': float(visibility),
                'is_visible': is_visible,
                'landmark_name': self._get_landmark_name(kp_id)
            }
            
            if is_visible:
                visible_keypoints += 1
        
        # Calculate pose quality metrics
        pose_quality = {
            'overall_confidence': float(confidence),
            'visibility_ratio': visible_keypoints / total_keypoints if total_keypoints > 0 else 0,
            'visible_keypoints': visible_keypoints,
            'total_keypoints': total_keypoints,
            'pose_completeness': self._calculate_pose_completeness(keypoint_analysis)
        }
        
        # Analyze pose orientation and stance
        pose_orientation = self._analyze_pose_orientation(keypoint_analysis)
        
        return {
            'person_id': person_id,
            'keypoints': keypoint_analysis,
            'pose_quality': pose_quality,
            'pose_orientation': pose_orientation,
            'bounding_box': self._calculate_pose_bounding_box(keypoint_analysis),
            'anatomical_regions': self._analyze_anatomical_regions(keypoint_analysis)
        }
    
    def _analyze_jersey_region(self, region, region_id: int, image: np.ndarray) -> Dict[str, Any]:
        """Analyze a jersey region in detail."""
        # Extract jersey image
        jersey_image = image[region.y:region.y + region.height, 
                           region.x:region.x + region.width]
        
        # Analyze jersey color distribution
        color_analysis = self._analyze_jersey_colors(jersey_image)
        
        # Analyze jersey texture and patterns
        texture_analysis = self._analyze_jersey_texture(jersey_image)
        
        return {
            'region_id': region_id,
            'person_id': region.person_id,
            'bounding_box': {
                'x': int(region.x),
                'y': int(region.y),
                'width': int(region.width),
                'height': int(region.height),
                'area': int(region.width * region.height)
            },
            'confidence': float(region.confidence),
            'color_analysis': color_analysis,
            'texture_analysis': texture_analysis,
            'region_quality': self._assess_jersey_region_quality(region, jersey_image)
        }
    
    def _analyze_body_region(self, region: Dict[str, Any], region_id: int, image: np.ndarray) -> Dict[str, Any]:
        """Analyze a body region in detail."""
        bbox = region['bounding_box']
        body_image = region['image']
        
        # Analyze body region for number detection potential
        number_detection_potential = self._assess_number_detection_potential(body_image)
        
        # Analyze body region characteristics
        body_characteristics = self._analyze_body_characteristics(body_image)
        
        return {
            'region_id': region_id,
            'person_id': region['person_id'],
            'bounding_box': bbox,
            'confidence': float(region['confidence']),
            'number_detection_potential': number_detection_potential,
            'body_characteristics': body_characteristics,
            'region_quality': self._assess_body_region_quality(bbox, body_image)
        }
    
    def _analyze_jersey_colors(self, jersey_image: np.ndarray) -> Dict[str, Any]:
        """Analyze colors in jersey region."""
        if jersey_image.size == 0:
            return {'error': 'Empty jersey image'}
        
        # Convert to different color spaces
        hsv_image = cv2.cvtColor(jersey_image, cv2.COLOR_RGB2HSV)
        
        # Calculate color statistics
        rgb_mean = np.mean(jersey_image, axis=(0, 1))
        rgb_std = np.std(jersey_image, axis=(0, 1))
        hsv_mean = np.mean(hsv_image, axis=(0, 1))
        hsv_std = np.std(hsv_image, axis=(0, 1))
        
        # Detect dominant colors
        dominant_colors = self.image_processor.detect_dominant_colors(jersey_image)
        
        return {
            'rgb_mean': rgb_mean.tolist() if hasattr(rgb_mean, 'tolist') else list(rgb_mean),
            'rgb_std': rgb_std.tolist() if hasattr(rgb_std, 'tolist') else list(rgb_std),
            'hsv_mean': hsv_mean.tolist() if hasattr(hsv_mean, 'tolist') else list(hsv_mean),
            'hsv_std': hsv_std.tolist() if hasattr(hsv_std, 'tolist') else list(hsv_std),
            'dominant_colors': [color.tolist() if hasattr(color, 'tolist') else list(color) for color in dominant_colors[:5]],  # Top 5 colors
            'color_variance': float(np.var(jersey_image)),
            'brightness': float(np.mean(cv2.cvtColor(jersey_image, cv2.COLOR_RGB2GRAY)))
        }
    
    def _analyze_jersey_texture(self, jersey_image: np.ndarray) -> Dict[str, Any]:
        """Analyze texture characteristics of jersey region."""
        if jersey_image.size == 0:
            return {'error': 'Empty jersey image'}
        
        # Convert to grayscale
        gray = cv2.cvtColor(jersey_image, cv2.COLOR_RGB2GRAY)
        
        # Calculate texture features
        # Laplacian variance (texture measure)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Local Binary Pattern approximation
        lbp_variance = np.var(gray)
        
        # Edge density
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        return {
            'laplacian_variance': float(laplacian_var),
            'lbp_variance': float(lbp_variance),
            'edge_density': float(edge_density),
            'texture_complexity': 'high' if laplacian_var > 100 else 'medium' if laplacian_var > 50 else 'low'
        }
    
    def _assess_number_detection_potential(self, body_image: np.ndarray) -> Dict[str, Any]:
        """Assess potential for number detection in body region."""
        if body_image.size == 0:
            return {'error': 'Empty body image'}
        
        # Convert to grayscale
        gray = cv2.cvtColor(body_image, cv2.COLOR_RGB2GRAY)
        
        # Analyze contrast
        contrast = np.std(gray)
        
        # Analyze brightness
        brightness = np.mean(gray)
        
        # Analyze potential text regions
        # Use adaptive threshold to find potential text
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        
        # Find contours that might be numbers
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        potential_text_regions = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            if 100 < area < 5000:  # Reasonable size for jersey numbers
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h
                if 0.3 < aspect_ratio < 3.0:  # Reasonable aspect ratio for numbers
                    potential_text_regions += 1
        
        return {
            'contrast': float(contrast),
            'brightness': float(brightness),
            'potential_text_regions': potential_text_regions,
            'detection_potential': 'high' if contrast > 50 and potential_text_regions > 0 else 'medium' if contrast > 30 else 'low',
            'recommended_preprocessing': 'adaptive_threshold' if contrast > 30 else 'histogram_equalization'
        }
    
    def _analyze_body_characteristics(self, body_image: np.ndarray) -> Dict[str, Any]:
        """Analyze general body region characteristics."""
        if body_image.size == 0:
            return {'error': 'Empty body image'}
        
        # Basic image statistics
        height, width = body_image.shape[:2]
        
        # Color analysis
        rgb_mean = np.mean(body_image, axis=(0, 1))
        
        # Brightness and contrast
        gray = cv2.cvtColor(body_image, cv2.COLOR_RGB2GRAY)
        brightness = np.mean(gray)
        contrast = np.std(gray)
        
        return {
            'dimensions': {'width': width, 'height': height},
            'aspect_ratio': width / height,
            'rgb_mean': rgb_mean.tolist() if hasattr(rgb_mean, 'tolist') else list(rgb_mean),
            'brightness': float(brightness),
            'contrast': float(contrast),
            'image_quality': 'good' if contrast > 40 and 50 < brightness < 200 else 'fair' if contrast > 20 else 'poor'
        }
    
    def _get_landmark_name(self, landmark_id: int) -> str:
        """Get human-readable landmark name."""
        landmark_names = {
            0: 'nose', 1: 'left_eye_inner', 2: 'left_eye', 3: 'left_eye_outer',
            4: 'right_eye_inner', 5: 'right_eye', 6: 'right_eye_outer',
            7: 'left_ear', 8: 'right_ear', 9: 'mouth_left', 10: 'mouth_right',
            11: 'left_shoulder', 12: 'right_shoulder', 13: 'left_elbow',
            14: 'right_elbow', 15: 'left_wrist', 16: 'right_wrist',
            17: 'left_pinky', 18: 'right_pinky', 19: 'left_index',
            20: 'right_index', 21: 'left_thumb', 22: 'right_thumb',
            23: 'left_hip', 24: 'right_hip', 25: 'left_knee',
            26: 'right_knee', 27: 'left_ankle', 28: 'right_ankle',
            29: 'left_heel', 30: 'right_heel', 31: 'left_foot_index',
            32: 'right_foot_index'
        }
        return landmark_names.get(landmark_id, f'landmark_{landmark_id}')
    
    def _calculate_pose_completeness(self, keypoint_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate pose completeness metrics."""
        # Define important body parts
        torso_keypoints = [11, 12, 23, 24]  # shoulders and hips
        arms_keypoints = [13, 14, 15, 16]   # elbows and wrists
        legs_keypoints = [25, 26, 27, 28]  # knees and ankles
        head_keypoints = [0, 1, 2, 5, 7, 8]  # head and face
        
        def get_visibility_ratio(keypoint_ids):
            visible = sum(1 for kp_id in keypoint_ids 
                         if kp_id in keypoint_analysis and keypoint_analysis[kp_id]['is_visible'])
            return visible / len(keypoint_ids) if keypoint_ids else 0
        
        return {
            'torso_completeness': get_visibility_ratio(torso_keypoints),
            'arms_completeness': get_visibility_ratio(arms_keypoints),
            'legs_completeness': get_visibility_ratio(legs_keypoints),
            'head_completeness': get_visibility_ratio(head_keypoints),
            'overall_completeness': np.mean([
                get_visibility_ratio(torso_keypoints),
                get_visibility_ratio(arms_keypoints),
                get_visibility_ratio(legs_keypoints),
                get_visibility_ratio(head_keypoints)
            ])
        }
    
    def _analyze_pose_orientation(self, keypoint_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze pose orientation and stance."""
        # Get key shoulder and hip positions
        left_shoulder = keypoint_analysis.get(11)
        right_shoulder = keypoint_analysis.get(12)
        left_hip = keypoint_analysis.get(23)
        right_hip = keypoint_analysis.get(24)
        
        orientation = 'unknown'
        stance = 'unknown'
        
        if all(kp and kp['is_visible'] for kp in [left_shoulder, right_shoulder]):
            # Calculate shoulder angle
            shoulder_angle = np.arctan2(
                right_shoulder['y'] - left_shoulder['y'],
                right_shoulder['x'] - left_shoulder['x']
            ) * 180 / np.pi
            
            if abs(shoulder_angle) < 15:
                orientation = 'frontal'
            elif abs(shoulder_angle) > 75:
                orientation = 'side'
            else:
                orientation = 'angled'
        
        if all(kp and kp['is_visible'] for kp in [left_hip, right_hip]):
            # Calculate hip angle
            hip_angle = np.arctan2(
                right_hip['y'] - left_hip['y'],
                right_hip['x'] - left_hip['x']
            ) * 180 / np.pi
            
            if abs(hip_angle) < 15:
                stance = 'standing_straight'
            elif abs(hip_angle) > 30:
                stance = 'leaning'
            else:
                stance = 'slight_lean'
        
        return {
            'orientation': orientation,
            'stance': stance,
            'shoulder_angle': shoulder_angle if 'shoulder_angle' in locals() else None,
            'hip_angle': hip_angle if 'hip_angle' in locals() else None
        }
    
    def _calculate_pose_bounding_box(self, keypoint_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate bounding box for pose."""
        visible_keypoints = [kp for kp in keypoint_analysis.values() if kp['is_visible']]
        
        if not visible_keypoints:
            return {'x': 0, 'y': 0, 'width': 0, 'height': 0}
        
        x_coords = [kp['x'] for kp in visible_keypoints]
        y_coords = [kp['y'] for kp in visible_keypoints]
        
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        
        return {
            'x': float(x_min),
            'y': float(y_min),
            'width': float(x_max - x_min),
            'height': float(y_max - y_min)
        }
    
    def _analyze_anatomical_regions(self, keypoint_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze anatomical regions of the pose."""
        regions = {
            'head': {'keypoints': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 'visible': 0},
            'torso': {'keypoints': [11, 12, 23, 24], 'visible': 0},
            'left_arm': {'keypoints': [11, 13, 15], 'visible': 0},
            'right_arm': {'keypoints': [12, 14, 16], 'visible': 0},
            'left_leg': {'keypoints': [23, 25, 27], 'visible': 0},
            'right_leg': {'keypoints': [24, 26, 28], 'visible': 0}
        }
        
        for region_name, region_data in regions.items():
            visible_count = sum(1 for kp_id in region_data['keypoints']
                              if kp_id in keypoint_analysis and keypoint_analysis[kp_id]['is_visible'])
            regions[region_name]['visible'] = visible_count
            regions[region_name]['completeness'] = visible_count / len(region_data['keypoints'])
        
        return regions
    
    def _assess_jersey_region_quality(self, region, jersey_image: np.ndarray) -> Dict[str, Any]:
        """Assess quality of jersey region extraction."""
        if jersey_image.size == 0:
            return {'quality': 'poor', 'reason': 'empty_region'}
        
        # Check region size
        area = region.width * region.height
        if area < 1000:
            return {'quality': 'poor', 'reason': 'too_small', 'area': area}
        
        # Check aspect ratio
        aspect_ratio = region.width / region.height
        if aspect_ratio < 0.5 or aspect_ratio > 2.0:
            return {'quality': 'fair', 'reason': 'unusual_aspect_ratio', 'aspect_ratio': aspect_ratio}
        
        # Check image quality
        gray = cv2.cvtColor(jersey_image, cv2.COLOR_RGB2GRAY)
        contrast = np.std(gray)
        brightness = np.mean(gray)
        
        if contrast < 20 or brightness < 30 or brightness > 225:
            return {'quality': 'fair', 'reason': 'poor_image_quality', 'contrast': contrast, 'brightness': brightness}
        
        return {'quality': 'good', 'area': area, 'aspect_ratio': aspect_ratio, 'contrast': contrast, 'brightness': brightness}
    
    def _assess_body_region_quality(self, bbox: Dict[str, Any], body_image: np.ndarray) -> Dict[str, Any]:
        """Assess quality of body region extraction."""
        if body_image.size == 0:
            return {'quality': 'poor', 'reason': 'empty_region'}
        
        # Check region size
        area = bbox['width'] * bbox['height']
        if area < 2000:
            return {'quality': 'poor', 'reason': 'too_small', 'area': area}
        
        # Check aspect ratio
        aspect_ratio = bbox['width'] / bbox['height']
        if aspect_ratio < 0.3 or aspect_ratio > 3.0:
            return {'quality': 'fair', 'reason': 'unusual_aspect_ratio', 'aspect_ratio': aspect_ratio}
        
        # Check image quality
        gray = cv2.cvtColor(body_image, cv2.COLOR_RGB2GRAY)
        contrast = np.std(gray)
        brightness = np.mean(gray)
        
        if contrast < 15 or brightness < 20 or brightness > 235:
            return {'quality': 'fair', 'reason': 'poor_image_quality', 'contrast': contrast, 'brightness': brightness}
        
        return {'quality': 'good', 'area': area, 'aspect_ratio': aspect_ratio, 'contrast': contrast, 'brightness': brightness}
    
    def _create_error_result(self, image_path: Path, error_message: str) -> Dict[str, Any]:
        """Create error result for failed analysis."""
        return {
            'image_info': {
                'file_path': str(image_path),
                'file_name': image_path.name,
                'file_size_bytes': image_path.stat().st_size if image_path.exists() else 0
            },
            'error': {
                'message': error_message,
                'timestamp': datetime.now().isoformat()
            },
            'processing_info': {
                'processing_time_seconds': 0.0,
                'timestamp': datetime.now().isoformat(),
                'analyzer_version': '1.0.0'
            }
        }
    
    def save_analysis_to_json(self, analysis_result: Dict[str, Any], output_path: Path) -> None:
        """Save analysis result to JSON file."""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(analysis_result, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
            logger.debug(f"Saved analysis to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save analysis to {output_path}: {e}")
            raise
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return self.stats.copy()
    
    def create_visualization(self, image: np.ndarray, analysis_result: Dict[str, Any]) -> np.ndarray:
        """
        Create annotated visualization of pose analysis results.
        
        Args:
            image: Original image array (RGB format)
            analysis_result: Analysis results dictionary
            
        Returns:
            Annotated image with pose landmarks, bounding boxes, and labels
        """
        if 'error' in analysis_result:
            # Return original image with error message
            vis_image = image.copy()
            cv2.putText(vis_image, "Analysis Error", (50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return vis_image
        
        vis_image = image.copy()
        
        # Get scale factors for coordinate conversion
        image_info = analysis_result.get('image_info', {})
        scale_factor = image_info.get('scale_factor', {'width': 1.0, 'height': 1.0})
        
        # Draw pose landmarks and connections
        pose_analysis = analysis_result.get('pose_analysis', {})
        poses = pose_analysis.get('poses', [])
        
        for pose_data in poses:
            vis_image = self._draw_pose_landmarks(vis_image, pose_data, scale_factor)
        
        # Draw jersey region bounding boxes
        jersey_analysis = analysis_result.get('jersey_analysis', {})
        jersey_regions = jersey_analysis.get('regions', [])
        
        for region_data in jersey_regions:
            vis_image = self._draw_jersey_region(vis_image, region_data, scale_factor)
        
        # Draw body region bounding boxes
        body_analysis = analysis_result.get('body_analysis', {})
        body_regions = body_analysis.get('regions', [])
        
        for region_data in body_regions:
            vis_image = self._draw_body_region(vis_image, region_data, scale_factor)
        
        # Add summary information
        vis_image = self._add_summary_info(vis_image, analysis_result)
        
        return vis_image
    
    def _draw_pose_landmarks(self, image: np.ndarray, pose_data: Dict[str, Any], scale_factor: Dict[str, float]) -> np.ndarray:
        """Draw pose landmarks and connections on image."""
        keypoints = pose_data.get('keypoints', {})
        person_id = pose_data.get('person_id', 0)
        
        # Define pose connections (MediaPipe pose connections)
        pose_connections = [
            (0, 1), (1, 2), (2, 3), (3, 7),  # Face
            (0, 4), (4, 5), (5, 6), (6, 8),  # Face
            (9, 10),  # Mouth
            (11, 12),  # Shoulders
            (11, 13), (13, 15),  # Left arm
            (12, 14), (14, 16),  # Right arm
            (11, 23), (12, 24),  # Torso
            (23, 24),  # Hips
            (23, 25), (25, 27), (27, 29), (29, 31),  # Left leg
            (24, 26), (26, 28), (28, 30), (30, 32),  # Right leg
        ]
        
        # Draw pose landmarks
        visible_landmarks = []
        for kp_id in range(33):  # MediaPipe has 33 pose landmarks
            if str(kp_id) in keypoints:
                kp = keypoints[str(kp_id)]
                if kp['is_visible']:
                    # Scale coordinates from processed image to original image
                    x = int(kp['x'] * scale_factor['width'])
                    y = int(kp['y'] * scale_factor['height'])
                    visible_landmarks.append((kp_id, x, y))
                    
                    # Draw landmark point (larger for original image) - CYAN color
                    cv2.circle(image, (x, y), 8, (255, 255, 0), -1)
        
        # Draw pose connections
        for connection in pose_connections:
            start_id, end_id = connection
            start_kp = next((kp for kp in visible_landmarks if kp[0] == start_id), None)
            end_kp = next((kp for kp in visible_landmarks if kp[0] == end_id), None)
            
            if start_kp and end_kp:
                cv2.line(image, (start_kp[1], start_kp[2]), (end_kp[1], end_kp[2]), (255, 255, 0), 4)
        
        # Add person ID label
        if visible_landmarks:
            # Use nose position (landmark 0) for label
            nose_kp = next((kp for kp in visible_landmarks if kp[0] == 0), None)
            if nose_kp:
                nose_x, nose_y = nose_kp[1], nose_kp[2]
                cv2.putText(image, f"Person {person_id}", (nose_x - 30, nose_y - 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 3)
        
        return image
    
    def _draw_jersey_region(self, image: np.ndarray, region_data: Dict[str, Any], scale_factor: Dict[str, float]) -> np.ndarray:
        """Draw jersey region bounding box on image with color swatch."""
        bbox = region_data.get('bounding_box', {})
        person_id = region_data.get('person_id', 0)
        confidence = region_data.get('confidence', 0.0)
        
        if not bbox:
            return image
        
        # Scale coordinates from processed image to original image
        x = int(bbox['x'] * scale_factor['width'])
        y = int(bbox['y'] * scale_factor['height'])
        width = int(bbox['width'] * scale_factor['width'])
        height = int(bbox['height'] * scale_factor['height'])
        
        # Draw bounding box (thicker for original image)
        cv2.rectangle(image, (x, y), (x + width, y + height), (0, 255, 0), 6)
        
        # Extract jersey region and calculate average color
        jersey_color = self._calculate_jersey_color(image, x, y, width, height)
        
        # Draw color swatch in upper-left corner of bounding box
        # Make swatch 33% of the bounding box size for better visibility
        swatch_size = max(20, min(int(min(width, height) * 0.33), 100))  # Between 20-100 pixels
        swatch_x = x + 5
        swatch_y = y + 5
        
        # Draw swatch background (white border)
        cv2.rectangle(image, (swatch_x - 2, swatch_y - 2), 
                     (swatch_x + swatch_size + 2, swatch_y + swatch_size + 2), (255, 255, 255), -1)
        
        # Draw color swatch
        cv2.rectangle(image, (swatch_x, swatch_y), 
                     (swatch_x + swatch_size, swatch_y + swatch_size), jersey_color, -1)
        
        # Draw label
        label = f"Jersey P{person_id} ({confidence:.2f})"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 3)[0]
        
        # Draw label background
        cv2.rectangle(image, (x, y - label_size[1] - 20), 
                     (x + label_size[0], y), (0, 255, 0), -1)
        
        # Draw label text
        cv2.putText(image, label, (x, y - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 3)
        
        return image
    
    def _calculate_jersey_color(self, image: np.ndarray, x: int, y: int, width: int, height: int) -> tuple:
        """
        Calculate dominant jersey color using improved algorithm with better background rejection.
        
        Args:
            image: Original image array (RGB format)
            x, y, width, height: Jersey region coordinates
            
        Returns:
            RGB color tuple (B, G, R) for OpenCV
        """
        # Ensure coordinates are within image bounds
        x = max(0, min(x, image.shape[1] - 1))
        y = max(0, min(y, image.shape[0] - 1))
        width = min(width, image.shape[1] - x)
        height = min(height, image.shape[0] - y)
        
        if width <= 0 or height <= 0:
            return (128, 128, 128)  # Default gray color
        
        # Extract jersey region
        jersey_region = image[y:y+height, x:x+width]
        
        # Focus on center area of jersey to avoid edges/background
        center_padding = 0.2  # Use 60% of center area
        center_x = int(width * center_padding)
        center_y = int(height * center_padding)
        center_w = int(width * (1 - 2 * center_padding))
        center_h = int(height * (1 - 2 * center_padding))
        
        if center_w > 0 and center_h > 0:
            center_region = jersey_region[center_y:center_y+center_h, center_x:center_x+center_w]
        else:
            center_region = jersey_region
        
        # Reshape for easier processing
        pixels = center_region.reshape(-1, 3)
        
        if len(pixels) == 0:
            return (128, 128, 128)  # Default gray color
        
        # Convert to HSV for better color analysis
        hsv_region = cv2.cvtColor(center_region, cv2.COLOR_RGB2HSV)
        hsv_pixels = hsv_region.reshape(-1, 3)
        
        # Improved background rejection
        # 1. Reject very dark pixels (shadows, black areas)
        brightness_mask = hsv_pixels[:, 2] > 30
        
        # 2. Reject very bright pixels (highlights, white areas) - but be more lenient for bright colors
        # For bright colors like yellow, we need higher value threshold
        brightness_upper_mask = hsv_pixels[:, 2] < 250  # More lenient upper bound
        
        # 3. Reject low saturation pixels (grays, whites) - but be more lenient
        # Yellow can have lower saturation than other colors
        saturation_mask = hsv_pixels[:, 1] > 20  # Lower threshold for yellow
        
        # 4. Additional check: reject pixels that are too close to pure white/black
        rgb_pixels = pixels
        white_distance = np.sqrt(np.sum((rgb_pixels - [255, 255, 255])**2, axis=1))
        black_distance = np.sqrt(np.sum((rgb_pixels - [0, 0, 0])**2, axis=1))
        
        not_white_mask = white_distance > 30  # Not too close to white
        not_black_mask = black_distance > 30   # Not too close to black
        
        # Combine all masks
        valid_mask = (brightness_mask & brightness_upper_mask & 
                     saturation_mask & not_white_mask & not_black_mask)
        
        # If too few valid pixels, relax the constraints
        if np.sum(valid_mask) < len(pixels) * 0.1:  # Less than 10% valid
            # Try with relaxed constraints
            relaxed_mask = (brightness_mask & brightness_upper_mask & not_white_mask & not_black_mask)
            if np.sum(relaxed_mask) > len(pixels) * 0.05:  # At least 5% valid
                valid_mask = relaxed_mask
            else:
                # Use all pixels if still too few
                valid_mask = np.ones(len(pixels), dtype=bool)
        
        valid_pixels = pixels[valid_mask]
        
        if len(valid_pixels) == 0:
            return (128, 128, 128)  # Default gray color
        
        # Use K-means clustering to find dominant colors
        dominant_color = self._find_dominant_color(valid_pixels)
        
        # Convert RGB to BGR for OpenCV
        return (int(dominant_color[2]), int(dominant_color[1]), int(dominant_color[0]))
    
    def _find_dominant_color(self, pixels: np.ndarray) -> np.ndarray:
        """
        Find the statistically dominant color using K-means clustering.
        
        Args:
            pixels: Array of RGB pixel values
            
        Returns:
            Dominant RGB color
        """
        try:
            from sklearn.cluster import KMeans
        except ImportError:
            # Fallback to simple averaging if sklearn not available
            return np.mean(pixels, axis=0).astype(int)
        
        # Determine number of clusters based on pixel count and color diversity
        n_pixels = len(pixels)
        
        # Check color diversity to avoid over-clustering
        color_std = np.std(pixels, axis=0)
        color_diversity = np.mean(color_std)
        
        if n_pixels < 30 or color_diversity < 10:
            # Not enough pixels or very uniform color - use simple average
            return np.mean(pixels, axis=0).astype(int)
        elif n_pixels < 100:
            n_clusters = 2
        elif n_pixels < 300:
            n_clusters = 3
        else:
            n_clusters = min(4, max(2, n_pixels // 100))  # 2-4 clusters max
        
        try:
            # Apply K-means clustering with error handling
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10, max_iter=100)
            kmeans.fit(pixels)
            
            # Get cluster centers and labels
            cluster_centers = kmeans.cluster_centers_
            labels = kmeans.labels_
            
            # Count pixels in each cluster
            cluster_counts = np.bincount(labels)
            
            # Find the cluster with the most pixels (dominant color)
            dominant_cluster_idx = np.argmax(cluster_counts)
            dominant_color = cluster_centers[dominant_cluster_idx]
            
            return dominant_color.astype(int)
            
        except Exception:
            # Fallback to simple averaging if clustering fails
            return np.mean(pixels, axis=0).astype(int)
    
    def _draw_body_region(self, image: np.ndarray, region_data: Dict[str, Any], scale_factor: Dict[str, float]) -> np.ndarray:
        """Draw body region bounding box on image."""
        bbox = region_data.get('bounding_box', {})
        person_id = region_data.get('person_id', 0)
        confidence = region_data.get('confidence', 0.0)
        
        if not bbox:
            return image
        
        # Scale coordinates from processed image to original image
        x = int(bbox['x'] * scale_factor['width'])
        y = int(bbox['y'] * scale_factor['height'])
        width = int(bbox['width'] * scale_factor['width'])
        height = int(bbox['height'] * scale_factor['height'])
        
        # Draw bounding box (thicker for original image)
        cv2.rectangle(image, (x, y), (x + width, y + height), (255, 0, 0), 6)
        
        # Draw label
        label = f"Body P{person_id} ({confidence:.2f})"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 3)[0]
        
        # Draw label background
        cv2.rectangle(image, (x, y + height), 
                     (x + label_size[0], y + height + label_size[1] + 20), (255, 0, 0), -1)
        
        # Draw label text
        cv2.putText(image, label, (x, y + height + 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
        
        return image
    
    def _add_summary_info(self, image: np.ndarray, analysis_result: Dict[str, Any]) -> np.ndarray:
        """Add summary information to the visualization."""
        pose_analysis = analysis_result.get('pose_analysis', {})
        jersey_analysis = analysis_result.get('jersey_analysis', {})
        body_analysis = analysis_result.get('body_analysis', {})
        
        num_people = pose_analysis.get('num_people_detected', 0)
        num_jerseys = jersey_analysis.get('num_jersey_regions', 0)
        num_bodies = body_analysis.get('num_body_regions', 0)
        
        # Create summary text
        summary_lines = [
            f"People: {num_people}",
            f"Jerseys: {num_jerseys}",
            f"Bodies: {num_bodies}"
        ]
        
        # Draw summary background
        text_height = 25
        bg_height = len(summary_lines) * text_height + 20
        cv2.rectangle(image, (10, 10), (200, bg_height), (0, 0, 0), -1)
        cv2.rectangle(image, (10, 10), (200, bg_height), (255, 255, 255), 2)
        
        # Draw summary text
        for i, line in enumerate(summary_lines):
            y_pos = 30 + i * text_height
            cv2.putText(image, line, (20, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return image
    
    def save_visualization(self, vis_image: np.ndarray, image_path: Path, output_dir: Path) -> Path:
        """Save visualization image to output directory."""
        # Create visualization filename
        vis_filename = f"{image_path.stem}_pose_visualization.jpg"
        vis_path = output_dir / vis_filename
        
        # Convert RGB to BGR for OpenCV
        vis_image_bgr = cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR)
        
        # Save image
        cv2.imwrite(str(vis_path), vis_image_bgr)
        
        return vis_path


@click.command()
@click.argument('input_dir', type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.argument('output_dir', type=click.Path(path_type=Path))
@click.option('--limit', '-l', type=int, default=None, help='Limit number of images to process (for testing)')
@click.option('--pose-confidence', '-c', default=0.3, help='Pose detection confidence threshold (lower = more sensitive)')
@click.option('--min-detection-confidence', default=0.3, help='MediaPipe minimum detection confidence (lower = more sensitive)')
@click.option('--min-tracking-confidence', default=0.3, help='MediaPipe minimum tracking confidence (lower = more sensitive)')
@click.option('--single-person', is_flag=True, help='Use single-person detection only')
@click.option('--save-visualizations', '-v', is_flag=True, help='Save annotated images with bounding boxes')
@click.option('--verbose', is_flag=True, help='Enable verbose logging')
@click.option('--dry-run', is_flag=True, help='Preview what would be processed without creating files')
def main(input_dir: Path, output_dir: Path, limit: Optional[int], pose_confidence: float,
         min_detection_confidence: float, min_tracking_confidence: float, single_person: bool,
         save_visualizations: bool, verbose: bool, dry_run: bool):
    """
    Enhanced Pose Analysis Tool for Soccer Photos
    
    Performs comprehensive pose analysis on soccer photos, assuming humans
    are mostly full-size in the shots. Creates detailed JSON sidecar files
    with analysis results.
    
    INPUT_DIR: Directory containing input images
    OUTPUT_DIR: Directory to save JSON analysis files
    """
    
    # Setup logging
    if verbose:
        logger.add(sys.stderr, level="DEBUG")
    else:
        logger.add(sys.stderr, level="INFO")
    
    # Create output directory
    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create visualizations subdirectory if needed
        if save_visualizations:
            vis_dir = output_dir / "visualizations"
            vis_dir.mkdir(parents=True, exist_ok=True)
    
    # Find image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG', '.tiff', '.tif', '.TIFF', '.TIF'}
    image_files = []
    
    for file_path in input_dir.rglob('*'):
        if file_path.is_file() and file_path.suffix in image_extensions:
            image_files.append(file_path)
    
    # Sort files for consistent processing order
    image_files.sort()
    
    # Apply limit if specified
    if limit:
        image_files = image_files[:limit]
        click.echo(f"Limited to {limit} images for testing")
    
    if not image_files:
        click.echo("No image files found in input directory", err=True)
        return
    
    click.echo(f"Found {len(image_files)} images to process")
    click.echo(f"Input directory: {input_dir}")
    click.echo(f"Output directory: {output_dir}")
    click.echo(f"Pose confidence threshold: {pose_confidence}")
    click.echo(f"Detection mode: {'single-person' if single_person else 'multi-person'}")
    click.echo(f"Save visualizations: {'Yes' if save_visualizations else 'No'}")
    
    if dry_run:
        click.echo("DRY RUN MODE - No files will be created")
        for i, image_file in enumerate(image_files[:10]):  # Show first 10
            click.echo(f"  Would process: {image_file.name}")
        if len(image_files) > 10:
            click.echo(f"  ... and {len(image_files) - 10} more files")
        return
    
    # Initialize analyzer
    try:
        analyzer = EnhancedPoseAnalyzer(
            pose_confidence=pose_confidence,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            enable_multi_person=not single_person
        )
    except ImportError as e:
        click.echo(f"Error: {e}", err=True)
        click.echo("Make sure MediaPipe is installed: pip install mediapipe", err=True)
        return
    
    # Process images with progress bar
    start_time = time.time()
    successful_analyses = 0
    failed_analyses = 0
    
    with tqdm(total=len(image_files), desc="Processing images", unit="img") as pbar:
        for image_file in image_files:
            try:
                # Perform analysis
                analysis_result = analyzer.analyze_image(image_file)
                
                # Save JSON sidecar file
                json_filename = f"{image_file.stem}_pose_analysis.json"
                json_path = output_dir / json_filename
                
                analyzer.save_analysis_to_json(analysis_result, json_path)
                
                # Create and save visualization if requested
                if save_visualizations:
                    try:
                        # Load original image for visualization
                        original_image = analyzer.image_processor.load_image(image_file)
                        if original_image is not None:
                            # Create visualization
                            vis_image = analyzer.create_visualization(original_image, analysis_result)
                            
                            # Save visualization
                            vis_path = analyzer.save_visualization(vis_image, image_file, vis_dir)
                            logger.debug(f"Saved visualization to {vis_path}")
                    except Exception as e:
                        logger.error(f"Failed to create visualization for {image_file}: {e}")
                
                # Update statistics
                if 'error' not in analysis_result:
                    successful_analyses += 1
                    analyzer.stats['total_people_detected'] += analysis_result['pose_analysis']['num_people_detected']
                    analyzer.stats['total_jersey_regions'] += analysis_result['jersey_analysis']['num_jersey_regions']
                    analyzer.stats['total_body_regions'] += analysis_result['body_analysis']['num_body_regions']
                else:
                    failed_analyses += 1
                    analyzer.stats['errors'].append({
                        'file': str(image_file),
                        'error': analysis_result['error']['message']
                    })
                
                # Update progress bar description
                pbar.set_description(f"Processed {image_file.name}")
                pbar.update(1)
                
            except Exception as e:
                logger.error(f"Failed to process {image_file}: {e}")
                failed_analyses += 1
                analyzer.stats['errors'].append({
                    'file': str(image_file),
                    'error': str(e)
                })
                pbar.update(1)
    
    # Calculate final statistics
    total_time = time.time() - start_time
    analyzer.stats.update({
        'total_images': len(image_files),
        'processed_images': len(image_files),
        'successful_analyses': successful_analyses,
        'failed_analyses': failed_analyses,
        'processing_time': total_time
    })
    
    # Save summary statistics
    summary_path = output_dir / "analysis_summary.json"
    summary_data = {
        'processing_summary': analyzer.get_statistics(),
        'settings': {
            'pose_confidence': pose_confidence,
            'min_detection_confidence': min_detection_confidence,
            'min_tracking_confidence': min_tracking_confidence,
            'enable_multi_person': not single_person,
            'input_directory': str(input_dir),
            'output_directory': str(output_dir)
        },
        'timestamp': datetime.now().isoformat()
    }
    
    try:
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
    except Exception as e:
        logger.error(f"Failed to save summary: {e}")
    
    # Display final results
    click.echo(f"\n🎯 ANALYSIS COMPLETE!")
    click.echo(f"=" * 50)
    click.echo(f"📁 Images processed: {len(image_files)}")
    click.echo(f"✅ Successful analyses: {successful_analyses}")
    click.echo(f"❌ Failed analyses: {failed_analyses}")
    click.echo(f"👥 Total people detected: {analyzer.stats['total_people_detected']}")
    click.echo(f"👕 Total jersey regions: {analyzer.stats['total_jersey_regions']}")
    click.echo(f"🦴 Total body regions: {analyzer.stats['total_body_regions']}")
    click.echo(f"⏱️  Total processing time: {total_time:.1f}s")
    click.echo(f"⏱️  Average time per image: {total_time/len(image_files):.2f}s")
    click.echo(f"💾 JSON files saved to: {output_dir}")
    click.echo(f"📄 Summary saved to: {summary_path}")
    if save_visualizations:
        click.echo(f"🖼️  Visualizations saved to: {vis_dir}")
    
    if failed_analyses > 0:
        click.echo(f"\n⚠️  {failed_analyses} analyses failed. Check logs for details.")
        click.echo("Failed files:")
        for error in analyzer.stats['errors'][:5]:  # Show first 5 errors
            click.echo(f"  - {Path(error['file']).name}: {error['error']}")
        if len(analyzer.stats['errors']) > 5:
            click.echo(f"  ... and {len(analyzer.stats['errors']) - 5} more errors")


if __name__ == '__main__':
    main()
