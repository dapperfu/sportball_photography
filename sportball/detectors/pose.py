"""
Pose Detection Module

MediaPipe-based pose detection for identifying upper body regions
for jersey color analysis in sports photography.

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

# Try to import MediaPipe for pose detection
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    logger.warning("mediapipe not available - pose detection will be skipped")

# Try to import MMPose as alternative
try:
    from mmpose.apis import MMPoseInferencer
    MMPOSE_AVAILABLE = True
except ImportError:
    MMPOSE_AVAILABLE = False
    logger.warning("mmpose not available - MMPose detection will be skipped")


@dataclass
class PoseKeypoint:
    """Represents a pose keypoint."""
    
    x: float
    y: float
    confidence: float
    name: str


@dataclass
class PoseDetection:
    """Represents a pose detection result."""
    
    keypoints: List[PoseKeypoint]
    upper_body_bbox: Optional[Tuple[int, int, int, int]] = None  # x, y, width, height
    confidence: float = 0.0
    person_id: int = 0


@dataclass
class PoseDetectionResult:
    """Result of pose detection on an image."""
    
    success: bool
    poses: List[PoseDetection]
    processing_time: float
    error: Optional[str] = None
    image_path: Optional[Path] = None


class PoseDetector:
    """
    Pose detection for identifying upper body regions for jersey color analysis.
    
    This class provides pose detection capabilities using MediaPipe (primary)
    or MMPose (alternative) for identifying upper body regions where jersey
    colors are most prominent and consistent.
    """
    
    # MediaPipe pose keypoint indices for upper body
    UPPER_BODY_KEYPOINTS = {
        'left_shoulder': 11,
        'right_shoulder': 12,
        'left_elbow': 13,
        'right_elbow': 14,
        'left_wrist': 15,
        'right_wrist': 16,
        'nose': 0,
        'left_eye': 2,
        'right_eye': 5,
        'left_ear': 7,
        'right_ear': 8,
    }
    
    # Keypoint names for MediaPipe
    KEYPOINT_NAMES = [
        'nose', 'left_eye_inner', 'left_eye', 'left_eye_outer',
        'right_eye_inner', 'right_eye', 'right_eye_outer',
        'left_ear', 'right_ear', 'mouth_left', 'mouth_right',
        'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist', 'left_pinky', 'right_pinky',
        'left_index', 'right_index', 'left_thumb', 'right_thumb',
        'left_hip', 'right_hip', 'left_knee', 'right_knee',
        'left_ankle', 'right_ankle', 'left_heel', 'right_heel',
        'left_foot_index', 'right_foot_index'
    ]
    
    def __init__(
        self,
        backend: str = "mediapipe",
        confidence_threshold: float = 0.7,
        cache_enabled: bool = True,
        enable_gpu: bool = True,
    ):
        """
        Initialize the PoseDetector.
        
        Args:
            backend: Pose detection backend ('mediapipe' or 'mmpose')
            confidence_threshold: Minimum confidence for pose detections
            cache_enabled: Whether to enable result caching
            enable_gpu: Whether to use GPU acceleration if available
        """
        self.backend = backend
        self.confidence_threshold = confidence_threshold
        self.cache_enabled = cache_enabled
        self.enable_gpu = enable_gpu
        
        self.logger = logger.bind(component="pose_detector")
        
        # Initialize the selected backend
        if backend == "mediapipe":
            self._init_mediapipe()
        elif backend == "mmpose":
            self._init_mmpose()
        else:
            raise ValueError(f"Unsupported backend: {backend}")
    
    def _init_mediapipe(self):
        """Initialize MediaPipe pose detection."""
        if not MEDIAPIPE_AVAILABLE:
            raise ImportError("MediaPipe is required for pose detection")
        
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Initialize pose detection
        self.pose = self.mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,  # Use high complexity for better accuracy
            enable_segmentation=False,
            min_detection_confidence=self.confidence_threshold,
            min_tracking_confidence=self.confidence_threshold,
        )
        
        self.logger.info("Initialized MediaPipe pose detection")
    
    def _init_mmpose(self):
        """Initialize MMPose detection."""
        if not MMPOSE_AVAILABLE:
            raise ImportError("MMPose is required for pose detection")
        
        # Initialize MMPose inferencer
        self.mmpose_inferencer = MMPoseInferencer(
            pose2d='rtmpose-m_8xb256-420e_coco-256x192',
            device='cuda' if self.enable_gpu else 'cpu'
        )
        
        self.logger.info("Initialized MMPose detection")
    
    def detect_poses(
        self,
        image_paths: Union[Path, List[Path]],
        save_sidecar: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Detect poses in images.
        
        Args:
            image_paths: Single image path or list of image paths
            save_sidecar: Whether to save results to sidecar files
            **kwargs: Additional arguments for pose detection
            
        Returns:
            Dictionary containing pose detection results
        """
        if isinstance(image_paths, Path):
            image_paths = [image_paths]
        
        self.logger.info(f"Detecting poses in {len(image_paths)} images")
        
        results = {}
        
        for image_path in image_paths:
            try:
                # Check cache first
                if self.cache_enabled:
                    cached_data = self._load_cached_result(image_path)
                    if cached_data:
                        results[str(image_path)] = cached_data
                        continue
                
                # Perform pose detection
                pose_result = self._detect_poses_in_image(image_path)
                
                # Save to sidecar if requested
                if save_sidecar and pose_result.success:
                    self._save_to_sidecar(image_path, pose_result)
                
                results[str(image_path)] = pose_result.as_dict()
                
            except Exception as e:
                self.logger.error(f"Pose detection failed for {image_path}: {e}")
                results[str(image_path)] = {
                    "success": False,
                    "error": str(e),
                    "poses": [],
                    "processing_time": 0.0
                }
        
        return results
    
    def _detect_poses_in_image(self, image_path: Path) -> PoseDetectionResult:
        """Detect poses in a single image."""
        start_time = time.perf_counter()
        
        try:
            # Load image
            image = Image.open(image_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            image_array = np.array(image)
            
            if self.backend == "mediapipe":
                return self._detect_poses_mediapipe(image_array, image_path)
            elif self.backend == "mmpose":
                return self._detect_poses_mmpose(image_array, image_path)
            else:
                raise ValueError(f"Unsupported backend: {self.backend}")
                
        except Exception as e:
            processing_time = time.perf_counter() - start_time
            return PoseDetectionResult(
                success=False,
                poses=[],
                processing_time=processing_time,
                error=str(e),
                image_path=image_path
            )
    
    def _detect_poses_mediapipe(self, image_array: np.ndarray, image_path: Path) -> PoseDetectionResult:
        """Detect poses using MediaPipe."""
        start_time = time.perf_counter()
        
        try:
            # Convert to RGB for MediaPipe
            rgb_image = image_array.copy()
            
            # Run pose detection
            results = self.pose.process(rgb_image)
            
            poses = []
            
            if results.pose_landmarks:
                # Extract keypoints
                keypoints = []
                for i, landmark in enumerate(results.pose_landmarks.landmark):
                    if i < len(self.KEYPOINT_NAMES):
                        keypoint = PoseKeypoint(
                            x=landmark.x,
                            y=landmark.y,
                            confidence=landmark.visibility,
                            name=self.KEYPOINT_NAMES[i]
                        )
                        keypoints.append(keypoint)
                
                # Create pose detection
                pose_detection = PoseDetection(
                    keypoints=keypoints,
                    confidence=self._calculate_pose_confidence(keypoints),
                    person_id=0
                )
                
                # Calculate upper body bounding box
                pose_detection.upper_body_bbox = self._calculate_upper_body_bbox(
                    keypoints, image_array.shape
                )
                
                poses.append(pose_detection)
            
            processing_time = time.perf_counter() - start_time
            
            return PoseDetectionResult(
                success=True,
                poses=poses,
                processing_time=processing_time,
                image_path=image_path
            )
            
        except Exception as e:
            processing_time = time.perf_counter() - start_time
            return PoseDetectionResult(
                success=False,
                poses=[],
                processing_time=processing_time,
                error=str(e),
                image_path=image_path
            )
    
    def _detect_poses_mmpose(self, image_array: np.ndarray, image_path: Path) -> PoseDetectionResult:
        """Detect poses using MMPose."""
        start_time = time.perf_counter()
        
        try:
            # Run MMPose inference
            results = self.mmpose_inferencer(image_array)
            
            poses = []
            
            # Process results (MMPose format may vary)
            # This is a simplified implementation - actual MMPose integration
            # would need to be adapted based on the specific MMPose version
            
            processing_time = time.perf_counter() - start_time
            
            return PoseDetectionResult(
                success=True,
                poses=poses,
                processing_time=processing_time,
                image_path=image_path
            )
            
        except Exception as e:
            processing_time = time.perf_counter() - start_time
            return PoseDetectionResult(
                success=False,
                poses=[],
                processing_time=processing_time,
                error=str(e),
                image_path=image_path
            )
    
    def _calculate_pose_confidence(self, keypoints: List[PoseKeypoint]) -> float:
        """Calculate overall pose confidence from keypoints."""
        if not keypoints:
            return 0.0
        
        # Focus on upper body keypoints for jersey analysis
        upper_body_indices = [
            self.UPPER_BODY_KEYPOINTS['left_shoulder'],
            self.UPPER_BODY_KEYPOINTS['right_shoulder'],
            self.UPPER_BODY_KEYPOINTS['left_elbow'],
            self.UPPER_BODY_KEYPOINTS['right_elbow'],
        ]
        
        upper_body_confidences = []
        for idx in upper_body_indices:
            if idx < len(keypoints):
                upper_body_confidences.append(keypoints[idx].confidence)
        
        if upper_body_confidences:
            return sum(upper_body_confidences) / len(upper_body_confidences)
        else:
            # Fallback to average of all keypoints
            return sum(kp.confidence for kp in keypoints) / len(keypoints)
    
    def _calculate_upper_body_bbox(
        self, 
        keypoints: List[PoseKeypoint], 
        image_shape: Tuple[int, int, int]
    ) -> Tuple[int, int, int, int]:
        """Calculate bounding box for upper body region."""
        height, width = image_shape[:2]
        
        # Get upper body keypoints
        upper_body_points = []
        for keypoint_name, idx in self.UPPER_BODY_KEYPOINTS.items():
            if idx < len(keypoints):
                kp = keypoints[idx]
                if kp.confidence > self.confidence_threshold:
                    upper_body_points.append((kp.x, kp.y))
        
        if not upper_body_points:
            # Fallback to shoulders only
            shoulder_points = []
            for idx in [self.UPPER_BODY_KEYPOINTS['left_shoulder'], 
                       self.UPPER_BODY_KEYPOINTS['right_shoulder']]:
                if idx < len(keypoints):
                    kp = keypoints[idx]
                    if kp.confidence > self.confidence_threshold:
                        shoulder_points.append((kp.x, kp.y))
            
            if shoulder_points:
                upper_body_points = shoulder_points
        
        if not upper_body_points:
            # No valid upper body points found
            return (0, 0, width, height)
        
        # Calculate bounding box
        x_coords = [p[0] for p in upper_body_points]
        y_coords = [p[1] for p in upper_body_points]
        
        min_x = min(x_coords)
        max_x = max(x_coords)
        min_y = min(y_coords)
        max_y = max(y_coords)
        
        # Add padding around the upper body region
        padding_x = (max_x - min_x) * 0.2  # 20% padding
        padding_y = (max_y - min_y) * 0.3  # 30% padding
        
        # Convert to pixel coordinates
        x1 = max(0, int((min_x - padding_x) * width))
        y1 = max(0, int((min_y - padding_y) * height))
        x2 = min(width, int((max_x + padding_x) * width))
        y2 = min(height, int((max_y + padding_y) * height))
        
        return (x1, y1, x2 - x1, y2 - y1)
    
    def _load_cached_result(self, image_path: Path) -> Optional[Dict[str, Any]]:
        """Load cached pose detection result."""
        # This would integrate with the sidecar system
        # For now, return None to always perform detection
        return None
    
    def _save_to_sidecar(self, image_path: Path, result: PoseDetectionResult):
        """Save pose detection result to sidecar file."""
        # This would integrate with the sidecar system
        # Implementation would go here
        pass
    
    def extract_jersey_regions(
        self, 
        image_path: Path, 
        pose_result: PoseDetectionResult
    ) -> List[np.ndarray]:
        """
        Extract jersey regions from image based on pose detection.
        
        Args:
            image_path: Path to the image
            pose_result: Pose detection result
            
        Returns:
            List of jersey region arrays
        """
        if not pose_result.success or not pose_result.poses:
            return []
        
        try:
            # Load image
            image = Image.open(image_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            image_array = np.array(image)
            
            jersey_regions = []
            
            for pose in pose_result.poses:
                if pose.upper_body_bbox:
                    x, y, w, h = pose.upper_body_bbox
                    
                    # Extract region
                    region = image_array[y:y+h, x:x+w]
                    
                    if region.size > 0:
                        jersey_regions.append(region)
            
            return jersey_regions
            
        except Exception as e:
            self.logger.error(f"Failed to extract jersey regions from {image_path}: {e}")
            return []


# Add as_dict method to PoseDetectionResult
def as_dict(self) -> Dict[str, Any]:
    """Convert PoseDetectionResult to dictionary."""
    return {
        "success": self.success,
        "poses": [
            {
                "keypoints": [
                    {
                        "x": kp.x,
                        "y": kp.y,
                        "confidence": kp.confidence,
                        "name": kp.name
                    }
                    for kp in pose.keypoints
                ],
                "upper_body_bbox": pose.upper_body_bbox,
                "confidence": pose.confidence,
                "person_id": pose.person_id
            }
            for pose in self.poses
        ],
        "processing_time": self.processing_time,
        "error": self.error,
        "image_path": str(self.image_path) if self.image_path else None
    }


# Add the method to PoseDetectionResult
PoseDetectionResult.as_dict = as_dict
