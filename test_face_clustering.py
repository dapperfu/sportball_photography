#!/usr/bin/env python3
"""
Test script for Face Clustering functionality.

This script demonstrates how to use the face clustering tool with sample data.
It creates a simple test case and shows the expected workflow.

Author: Claude Sonnet 4 (claude-3-5-sonnet-20241022)
Generated via Cursor IDE (cursor.sh) with AI assistance
"""

import json
import tempfile
from pathlib import Path
from typing import List
import numpy as np
from face_clustering import FaceClusteringEngine


def create_sample_json_data(output_dir: Path) -> List[Path]:
    """
    Create sample JSON data for testing face clustering.
    
    Args:
        output_dir: Directory to create sample JSON files
        
        Returns:
            List of created JSON file paths
    """
    import cv2
    
    output_dir.mkdir(parents=True, exist_ok=True)
    json_files = []
    
    # Create sample face data with encodings
    sample_faces = [
        {
            "image_name": "photo_001.jpg",
            "faces": [
                {
                    "face_id": 1,
                    "coordinates": {"x": 100, "y": 150, "width": 200, "height": 250},
                    "confidence": 0.95,
                    "face_encoding": np.random.rand(128).tolist(),  # Random 128-dim encoding
                    "crop_area": {"x": 75, "y": 125, "width": 250, "height": 300},
                    "detection_scale_factor": 1.0
                }
            ]
        },
        {
            "image_name": "photo_002.jpg", 
            "faces": [
                {
                    "face_id": 1,
                    "coordinates": {"x": 120, "y": 160, "width": 180, "height": 230},
                    "confidence": 0.92,
                    "face_encoding": np.random.rand(128).tolist(),
                    "crop_area": {"x": 95, "y": 135, "width": 230, "height": 280},
                    "detection_scale_factor": 1.0
                }
            ]
        },
        {
            "image_name": "photo_003.jpg",
            "faces": [
                {
                    "face_id": 1,
                    "coordinates": {"x": 80, "y": 120, "width": 220, "height": 270},
                    "confidence": 0.88,
                    "face_encoding": np.random.rand(128).tolist(),
                    "crop_area": {"x": 55, "y": 95, "width": 270, "height": 320},
                    "detection_scale_factor": 1.0
                }
            ]
        }
    ]
    
    # Create sample images and JSON files
    for face_data in sample_faces:
        # Create a dummy image
        image_path = output_dir / face_data["image_name"]
        dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        cv2.imwrite(str(image_path), dummy_image)
        
        # Create JSON data
        json_data = {
            "Face_detector": {
                "faces": face_data["faces"]
            }
        }
        
        json_filename = face_data["image_name"].replace(".jpg", ".json")
        json_path = output_dir / json_filename
        
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2)
        
        json_files.append(json_path)
    
    return json_files


def test_face_clustering():
    """Test the face clustering functionality."""
    print("🧪 Testing Face Clustering Engine...")
    
    # Create temporary directory for test data
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create sample JSON data
        print("📄 Creating sample JSON data...")
        json_files = create_sample_json_data(temp_path / "json_data")
        print(f"✅ Created {len(json_files)} sample JSON files")
        
        # Create clustering engine
        print("🤖 Initializing clustering engine...")
        engine = FaceClusteringEngine(
            padding=0.25,
            min_cluster_size=2,
            max_clusters=10,
            algorithm='dbscan',
            max_workers=2
        )
        
        # Load face data
        print("📊 Loading face data...")
        face_data = engine.load_face_data_from_json(json_files)
        print(f"✅ Loaded {len(face_data)} faces")
        
        # Extract encodings
        print("🧠 Extracting face encodings...")
        encodings, valid_indices = engine.extract_face_encodings()
        print(f"✅ Extracted {len(encodings)} encodings")
        
        # Cluster faces
        print("🔗 Clustering faces...")
        clusters = engine.cluster_faces(encodings, valid_indices)
        print(f"✅ Created {len(clusters)} clusters")
        
        # Display results
        print("\n📋 Clustering Results:")
        engine.display_clustering_results()
        
        print("\n✅ Face clustering test completed successfully!")


if __name__ == "__main__":
    test_face_clustering()
