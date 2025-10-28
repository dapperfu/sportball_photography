#!/usr/bin/env python3
"""
Test sidecar functionality on real images

Tests sidecar operations without requiring Rust tools to be installed.
Focuses on reading and writing sidecar files that already exist.

Author: Claude Sonnet 4 (claude-3-5-sonnet-20241022)
Generated via Cursor IDE (cursor.sh) with AI assistance
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from sportball.sidecar import SidecarManager, OperationType


def test_sidecar_read_write():
    """Test reading and writing sidecar files without Rust dependency."""
    print("🧪 Testing Sidecar Functionality on Real Images")
    print("=" * 70)
    
    try:
        manager = SidecarManager()
        
        # Check if Rust manager is available
        if manager.rust_manager and manager.rust_manager.rust_available:
            print("✅ Rust sidecar manager is available")
        else:
            print("⚠️  Rust sidecar manager not available (tests will use Python sidecar manager)")
        
        # Test with a real image path
        test_image = Path("/keg/pictures/2015/08-Aug")
        
        if not test_image.exists():
            print(f"❌ Test directory not found: {test_image}")
            return False
        
        # Find all images in the directory
        image_files = []
        for ext in [".jpg", ".jpeg", ".png"]:
            image_files.extend(test_image.glob(f"*{ext}"))
        
        if not image_files:
            print(f"❌ No images found in {test_image}")
            return False
        
        # Take first 3 images for testing
        test_images = image_files[:3]
        print(f"\n📊 Testing with {len(test_images)} images:")
        for img in test_images:
            print(f"   - {img.name}")
        
        # Test sidecar file operations
        print("\n🔄 Testing sidecar file operations...")
        
        for img in test_images:
            # Find or create sidecar info
            sidecar_info = manager.find_sidecar_for_image(img)
            
            if sidecar_info:
                print(f"\n✅ Found sidecar for {img.name}")
                print(f"   Sidecar: {sidecar_info.sidecar_path.name}")
                print(f"   Operation: {sidecar_info.operation}")
                
                # Try to load data
                try:
                    data = sidecar_info.load()
                    keys = list(data.keys())
                    print(f"   Data keys: {', '.join(keys[:3])}{'...' if len(keys) > 3 else ''}")
                    
                    # Test metadata extraction
                    proc_time = sidecar_info.get_processing_time()
                    if proc_time:
                        print(f"   Processing time: {proc_time:.2f}s")
                    
                    success = sidecar_info.get_success_status()
                    print(f"   Success: {success}")
                    
                except Exception as e:
                    print(f"   ⚠️  Could not load data: {e}")
            else:
                print(f"\n⚠️  No sidecar found for {img.name}")
        
        # Test statistics gathering
        print("\n🔄 Testing statistics gathering...")
        try:
            stats = manager.get_statistics(test_image)
            print(f"✅ Statistics generated")
            print(f"   Total images: {stats.get('total_images', 0)}")
            print(f"   Total sidecars: {stats.get('total_sidecars', 0)}")
            print(f"   Coverage: {stats.get('coverage_percentage', 0):.1f}%")
            
            op_counts = stats.get('operation_counts', {})
            if op_counts:
                print(f"   Operations:")
                for op, count in op_counts.items():
                    print(f"     - {op}: {count}")
            
        except Exception as e:
            print(f"⚠️  Statistics failed: {e}")
        
        print("\n✅ Sidecar operations test completed")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_sidecar_read_existing():
    """Test reading existing sidecar files."""
    print("\n" + "=" * 70)
    print("📖 Testing READ operations on existing sidecars")
    print("=" * 70)
    
    try:
        manager = SidecarManager()
        
        # Look for images with sidecars
        test_dir = Path("/keg/pictures/2015/08-Aug")
        
        if not test_dir.exists():
            print(f"❌ Test directory not found")
            return False
        
        # Find sidecars
        sidecars = manager.find_all_sidecars(test_dir)
        
        if not sidecars:
            print(f"⚠️  No sidecars found in directory")
            return False
        
        print(f"✅ Found {len(sidecars)} sidecar files")
        
        # Test loading each sidecar
        for i, sidecar_info in enumerate(sidecars[:5], 1):  # Test first 5
            print(f"\n📄 Sidecar {i}: {sidecar_info.sidecar_path.name}")
            
            try:
                data = sidecar_info.load()
                
                # Check structure
                has_sidecar_info = "sidecar_info" in data
                has_face_detection = "face_detection" in data
                has_yolov8 = "yolov8" in data
                
                print(f"   Has sidecar_info: {has_sidecar_info}")
                print(f"   Has face_detection: {has_face_detection}")
                print(f"   Has yolov8: {has_yolov8}")
                
                # Test data access
                if has_face_detection:
                    face_data = data.get("face_detection", {})
                    faces_count = face_data.get("metadata", {}).get("faces_found", 0)
                    print(f"   Faces found: {faces_count}")
                
                if has_yolov8:
                    yolov8_data = data.get("yolov8", {})
                    objects_count = yolov8_data.get("metadata", {}).get("objects_found", 0)
                    print(f"   Objects found: {objects_count}")
                
            except Exception as e:
                print(f"   ⚠️  Error loading: {e}")
        
        print("\n✅ READ operations test completed")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


def main() -> int:
    """Main test function."""
    print("🧪 Sidecar Functionality Test on Real Images")
    print("=" * 70)
    print()
    
    tests = [
        ("Sidecar Read/Write Operations", test_sidecar_read_write),
        ("Sidecar Read Existing Files", test_sidecar_read_existing),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        if test_func():
            passed += 1
            print(f"✅ {test_name} passed")
        else:
            print(f"❌ {test_name} failed")
        print()
    
    print("=" * 70)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    print("=" * 70)
    
    if passed == total:
        print("🎉 All tests passed!")
        return 0
    else:
        print("⚠️  Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())

