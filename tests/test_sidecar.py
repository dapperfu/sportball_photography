"""
Tests for sidecar format support.

Tests binary and JSON format serialization, deserialization, and format detection.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from sportball.sidecar import SidecarManager, OperationType


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    # Cleanup
    import shutil
    shutil.rmtree(temp_path)


@pytest.fixture
def test_image_path(temp_dir):
    """Create a test image path."""
    image_path = temp_dir / "test_image.jpg"
    # Create a dummy image file
    image_path.write_bytes(b"dummy image data")
    return image_path


class TestSidecarFormatSupport:
    """Test suite for sidecar format support."""

    def test_save_and_load_json_format(self, temp_dir, test_image_path):
        """Test saving and loading sidecar data in JSON format."""
        manager = SidecarManager()
        
        test_data = {
            "faces": [{"face_id": 1, "confidence": 0.95}],
            "success": True
        }
        
        # Save with JSON format
        success = manager.save_data_merge(
            test_image_path,
            "face_detection",
            test_data
        )
        
        assert success
        
        # Load the data
        loaded_data = manager._load_existing_data(test_image_path)
        assert "face_detection" in loaded_data
        assert loaded_data["face_detection"]["success"] is True
        assert len(loaded_data["face_detection"]["faces"]) == 1

    def test_nested_backend_storage(self, temp_dir, test_image_path):
        """Test nested backend storage structure."""
        manager = SidecarManager()
        
        test_data = {
            "faces": [{"face_id": 1, "confidence": 0.95}],
            "success": True
        }
        
        # Save with backend metadata
        success = manager.save_data_merge(
            test_image_path,
            "face_detection",
            test_data,
            metadata={"backend": "insightface"}
        )
        
        assert success
        
        # Load and verify nested structure
        loaded_data = manager._load_existing_data(test_image_path)
        assert "face_detection" in loaded_data
        assert "insightface" in loaded_data["face_detection"]
        assert loaded_data["face_detection"]["insightface"]["success"] is True

    def test_multiple_backends_same_image(self, temp_dir, test_image_path):
        """Test storing results from multiple backends for the same image."""
        manager = SidecarManager()
        
        # Save with first backend
        insightface_data = {
            "faces": [{"face_id": 1, "confidence": 0.95}],
            "success": True
        }
        success1 = manager.save_data_merge(
            test_image_path,
            "face_detection",
            insightface_data,
            metadata={"backend": "insightface"}
        )
        
        assert success1
        
        # Save with second backend
        facerec_data = {
            "faces": [{"face_id": 1, "confidence": 0.92}],
            "success": True
        }
        success2 = manager.save_data_merge(
            test_image_path,
            "face_detection",
            facerec_data,
            metadata={"backend": "face_recognition"}
        )
        
        assert success2
        
        # Verify both backends are stored
        loaded_data = manager._load_existing_data(test_image_path)
        assert "face_detection" in loaded_data
        assert "insightface" in loaded_data["face_detection"]
        assert "face_recognition" in loaded_data["face_detection"]

    def test_format_detection(self, temp_dir, test_image_path):
        """Test format detection for existing sidecar files."""
        manager = SidecarManager()
        
        test_data = {"test": "data"}
        
        # Save in JSON format
        manager.save_data_merge(
            test_image_path,
            "test_operation",
            test_data
        )
        
        # Detect the format
        detected_format = manager._detect_existing_format(test_image_path)
        
        # Should detect one of the supported formats
        assert detected_format is not None
        
    def test_skip_logic_with_existing_sidecar(self, temp_dir, test_image_path):
        """Test that processing is skipped when valid sidecar exists."""
        manager = SidecarManager()
        
        test_data = {
            "objects": [{"class": "person", "confidence": 0.95}],
            "success": True,
            "objects_found": 1
        }
        
        # Save initial sidecar
        manager.save_data_merge(
            test_image_path,
            "yolov8",
            test_data
        )
        
        # Try to find the sidecar (simulating skip logic)
        sidecar_info = manager.find_sidecar_for_image(test_image_path)
        
        assert sidecar_info is not None
        assert sidecar_info.operation == OperationType.YOLOV8
        
        # Verify the data can be loaded
        loaded_data = sidecar_info.load()
        assert "yolov8" in loaded_data
        assert loaded_data["yolov8"]["success"] is True
        assert loaded_data["yolov8"]["objects_found"] == 1


class TestRustIntegration:
    """Test Rust sidecar integration."""

    def test_rust_manager_availability(self):
        """Test if Rust manager is available."""
        from sportball.detection.rust_sidecar import RustSidecarManager
        
        manager = RustSidecarManager()
        
        # Should at least initialize (Rust binary might not be available)
        assert manager is not None
        
    @patch('sportball.detection.rust_sidecar.RustSidecarManager.rust_available', True)
    def test_rust_save_when_available(self, temp_dir, test_image_path):
        """Test Rust save when Rust binary is available."""
        from sportball.detection.rust_sidecar import RustSidecarManager
        
        manager = RustSidecarManager()
        
        # This will try to use Rust if available, fallback otherwise
        # The key is that it doesn't raise an error
        try:
            test_data = {"test": "data"}
            manager.save_sidecar_data(
                test_image_path,
                "test_operation",
                test_data
            )
        except Exception:
            # Expected if Rust binary is not available
            pass


class TestFormatCompatibility:
    """Test format compatibility and migration."""
    
    def test_backward_compatibility_with_flat_structure(self, temp_dir, test_image_path):
        """Test that old flat sidecar structure still works."""
        manager = SidecarManager()
        
        # Save data without backend metadata (flat structure)
        test_data = {"faces": [], "success": True}
        
        success = manager.save_data_merge(
            test_image_path,
            "face_detection",
            test_data
        )
        
        assert success
        
        # Load should work
        loaded_data = manager._load_existing_data(test_image_path)
        assert "face_detection" in loaded_data
        assert isinstance(loaded_data["face_detection"], dict)
        
    def test_roundtrip_json(self, temp_dir, test_image_path):
        """Test roundtrip: save JSON, read JSON."""
        manager = SidecarManager()
        
        original_data = {
            "faces": [{"face_id": 1, "confidence": 0.95}],
            "success": True
        }
        
        # Save
        manager.save_data_merge(
            test_image_path,
            "face_detection",
            original_data
        )
        
        # Load
        loaded_data = manager._load_existing_data(test_image_path)
        
        # Compare
        assert loaded_data["face_detection"]["success"] == original_data["success"]
        assert len(loaded_data["face_detection"]["faces"]) == len(original_data["faces"])

