"""
Tests for Jersey Color Splitting Functionality

Comprehensive tests for the jersey color-based game splitting system.

Author: Claude Sonnet 4 (claude-3-5-sonnet-20241022)
Generated via Cursor IDE (cursor.sh) with AI assistance
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
import numpy as np
from PIL import Image

from sportball.detectors.pose import PoseDetector, PoseDetectionResult, PoseDetection, PoseKeypoint
from sportball.detectors.color_analysis import ColorAnalyzer, ColorAnalysisResult, ColorCluster
from sportball.detectors.jersey_splitting import JerseyGameSplitter, JerseySplittingResult
from sportball.core import SportballCore


class TestPoseDetector:
    """Test pose detection functionality."""
    
    def test_pose_detector_initialization(self):
        """Test pose detector initialization."""
        with patch('sportball.detectors.pose.MEDIAPIPE_AVAILABLE', True):
            with patch('sportball.detectors.pose.mp') as mock_mp:
                mock_mp.solutions.pose.Pose.return_value = Mock()
                
                detector = PoseDetector()
                assert detector.backend == "mediapipe"
                assert detector.confidence_threshold == 0.7
    
    def test_pose_detector_mediapipe_unavailable(self):
        """Test pose detector when MediaPipe is unavailable."""
        with patch('sportball.detectors.pose.MEDIAPIPE_AVAILABLE', False):
            with pytest.raises(ImportError):
                PoseDetector()
    
    def test_calculate_pose_confidence(self):
        """Test pose confidence calculation."""
        with patch('sportball.detectors.pose.MEDIAPIPE_AVAILABLE', True):
            with patch('sportball.detectors.pose.mp') as mock_mp:
                mock_mp.solutions.pose.Pose.return_value = Mock()
                
                detector = PoseDetector()
                
                # Create mock keypoints
                keypoints = [
                    PoseKeypoint(0.5, 0.5, 0.8, "left_shoulder"),
                    PoseKeypoint(0.6, 0.5, 0.9, "right_shoulder"),
                    PoseKeypoint(0.5, 0.6, 0.7, "left_elbow"),
                    PoseKeypoint(0.6, 0.6, 0.8, "right_elbow"),
                ]
                
                confidence = detector._calculate_pose_confidence(keypoints)
                assert 0.0 <= confidence <= 1.0
    
    def test_calculate_upper_body_bbox(self):
        """Test upper body bounding box calculation."""
        with patch('sportball.detectors.pose.MEDIAPIPE_AVAILABLE', True):
            with patch('sportball.detectors.pose.mp') as mock_mp:
                mock_mp.solutions.pose.Pose.return_value = Mock()
                
                detector = PoseDetector()
                
                # Create mock keypoints
                keypoints = [
                    PoseKeypoint(0.4, 0.3, 0.8, "left_shoulder"),
                    PoseKeypoint(0.6, 0.3, 0.9, "right_shoulder"),
                ]
                
                bbox = detector._calculate_upper_body_bbox(keypoints, (480, 640, 3))
                assert len(bbox) == 4
                assert all(coord >= 0 for coord in bbox)


class TestColorAnalyzer:
    """Test color analysis functionality."""
    
    def test_color_analyzer_initialization(self):
        """Test color analyzer initialization."""
        with patch('sportball.detectors.color_analysis.SKLEARN_AVAILABLE', True):
            analyzer = ColorAnalyzer()
            assert analyzer.n_clusters == 8
            assert analyzer.color_space == "rgb"
            assert analyzer.similarity_threshold == 0.15
    
    def test_color_analyzer_sklearn_unavailable(self):
        """Test color analyzer when scikit-learn is unavailable."""
        with patch('sportball.detectors.color_analysis.SKLEARN_AVAILABLE', False):
            with pytest.raises(ImportError):
                ColorAnalyzer()
    
    def test_calculate_color_similarity(self):
        """Test color similarity calculation."""
        with patch('sportball.detectors.color_analysis.SKLEARN_AVAILABLE', True):
            analyzer = ColorAnalyzer()
            
            color1 = {"lab_color": (50.0, 10.0, 20.0)}
            color2 = {"lab_color": (52.0, 12.0, 22.0)}
            
            similarity = analyzer._calculate_color_similarity(color1, color2)
            assert similarity >= 0.0
    
    def test_filter_jersey_colors(self):
        """Test jersey color filtering."""
        with patch('sportball.detectors.color_analysis.SKLEARN_AVAILABLE', True):
            analyzer = ColorAnalyzer()
            
            # Create mock clusters
            clusters = [
                ColorCluster(
                    rgb_color=(255, 0, 0),
                    hsv_color=(0, 1.0, 1.0),  # High saturation (jersey)
                    lab_color=(50.0, 10.0, 20.0),
                    percentage=0.3,
                    pixel_count=1000,
                    cluster_id=0
                ),
                ColorCluster(
                    rgb_color=(128, 128, 128),
                    hsv_color=(0, 0.0, 0.5),  # Low saturation (background)
                    lab_color=(50.0, 0.0, 0.0),
                    percentage=0.2,
                    pixel_count=500,
                    cluster_id=1
                ),
            ]
            
            jersey_colors, background_colors = analyzer._filter_jersey_colors(clusters)
            assert len(jersey_colors) == 1
            assert len(background_colors) == 1


class TestJerseyGameSplitter:
    """Test jersey game splitting functionality."""
    
    def test_jersey_splitter_initialization(self):
        """Test jersey splitter initialization."""
        with patch('sportball.detectors.pose.MEDIAPIPE_AVAILABLE', True):
            with patch('sportball.detectors.pose.mp') as mock_mp:
                mock_mp.solutions.pose.Pose.return_value = Mock()
                
                with patch('sportball.detectors.color_analysis.SKLEARN_AVAILABLE', True):
                    splitter = JerseyGameSplitter()
                    assert splitter.pose_confidence_threshold == 0.7
                    assert splitter.color_similarity_threshold == 0.15
                    assert splitter.min_team_photos == 5
    
    def test_group_similar_colors(self):
        """Test grouping similar colors."""
        with patch('sportball.detectors.pose.MEDIAPIPE_AVAILABLE', True):
            with patch('sportball.detectors.pose.mp') as mock_mp:
                mock_mp.solutions.pose.Pose.return_value = Mock()
                
                with patch('sportball.detectors.color_analysis.SKLEARN_AVAILABLE', True):
                    splitter = JerseyGameSplitter()
                    
                    colors = [
                        {"lab_color": (50.0, 10.0, 20.0), "percentage": 0.3},
                        {"lab_color": (52.0, 12.0, 22.0), "percentage": 0.2},
                        {"lab_color": (80.0, 5.0, 10.0), "percentage": 0.1},
                    ]
                    
                    groups = splitter._group_similar_colors(colors)
                    assert len(groups) >= 1
    
    def test_calculate_split_confidence(self):
        """Test split confidence calculation."""
        with patch('sportball.detectors.pose.MEDIAPIPE_AVAILABLE', True):
            with patch('sportball.detectors.pose.mp') as mock_mp:
                mock_mp.solutions.pose.Pose.return_value = Mock()
                
                with patch('sportball.detectors.color_analysis.SKLEARN_AVAILABLE', True):
                    splitter = JerseyGameSplitter()
                    
                    photo_jersey_colors = [
                        {"lab_color": (50.0, 10.0, 20.0)},
                        {"lab_color": (80.0, 5.0, 10.0)},
                    ]
                    matched_teams = ["Team_1", "Team_2"]
                    jersey_groups = []
                    
                    confidence = splitter._calculate_split_confidence(
                        photo_jersey_colors, matched_teams, jersey_groups
                    )
                    assert 0.0 <= confidence <= 1.0


class TestSportballCoreIntegration:
    """Test SportballCore integration."""
    
    def test_core_jersey_splitter_property(self):
        """Test jersey splitter property in SportballCore."""
        with patch('sportball.detectors.pose.MEDIAPIPE_AVAILABLE', True):
            with patch('sportball.detectors.pose.mp') as mock_mp:
                mock_mp.solutions.pose.Pose.return_value = Mock()
                
                with patch('sportball.detectors.color_analysis.SKLEARN_AVAILABLE', True):
                    core = SportballCore()
                    assert core.jersey_splitter is not None
    
    def test_split_games_by_jersey_color_method(self):
        """Test split_games_by_jersey_color method."""
        with patch('sportball.detectors.pose.MEDIAPIPE_AVAILABLE', True):
            with patch('sportball.detectors.pose.mp') as mock_mp:
                mock_mp.solutions.pose.Pose.return_value = Mock()
                
                with patch('sportball.detectors.color_analysis.SKLEARN_AVAILABLE', True):
                    core = SportballCore()
                    
                    # Create temporary test image
                    with tempfile.TemporaryDirectory() as temp_dir:
                        temp_path = Path(temp_dir)
                        test_image = temp_path / "test.jpg"
                        
                        # Create a simple test image
                        image = Image.new('RGB', (100, 100), color='red')
                        image.save(test_image)
                        
                        # Mock the jersey splitter to avoid actual processing
                        mock_result = JerseySplittingResult(
                            success=True,
                            split_decisions=[],
                            detected_teams=[],
                            processing_time=1.0
                        )
                        
                        with patch.object(core, 'get_jersey_splitter') as mock_get_splitter:
                            mock_splitter = Mock()
                            mock_splitter.split_games_by_jersey_color.return_value = mock_result
                            mock_splitter.get_split_summary.return_value = {"total_photos": 1}
                            mock_get_splitter.return_value = mock_splitter
                            
                            result = core.split_games_by_jersey_color([test_image])
                            assert result["success"] is True
                            assert "summary" in result


class TestCLIIntegration:
    """Test CLI integration."""
    
    def test_jersey_splitting_cli_options(self):
        """Test that CLI options are properly defined."""
        from sportball.cli.commands.game_commands import split
        
        # Check that the command has the expected options
        command_options = [option.name for option in split.params]
        
        expected_options = [
            'split_by_jersey',
            'pose_confidence', 
            'color_similarity',
            'min_team_photos'
        ]
        
        for option in expected_options:
            assert option in command_options


if __name__ == "__main__":
    pytest.main([__file__])
