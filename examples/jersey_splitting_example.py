#!/usr/bin/env python3
"""
Jersey Color Splitting Example

Example script demonstrating jersey color-based game splitting functionality.

Author: Claude Sonnet 4 (claude-3-5-sonnet-20241022)
Generated via Cursor IDE (cursor.sh) with AI assistance
"""

import sys
from pathlib import Path
from sportball.core import SportballCore


def main():
    """Main example function."""
    if len(sys.argv) < 3:
        print("Usage: python jersey_splitting_example.py <input_directory> <output_directory>")
        print("\nExample:")
        print("  python jersey_splitting_example.py /path/to/photos /path/to/output")
        sys.exit(1)
    
    input_dir = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    
    if not input_dir.exists():
        print(f"Error: Input directory {input_dir} does not exist")
        sys.exit(1)
    
    print(f"🎨 Jersey Color Splitting Example")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print()
    
    # Initialize SportballCore
    print("Initializing SportballCore...")
    core = SportballCore(enable_gpu=True, verbose=True)
    
    # Find image files
    image_files = []
    for ext in [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]:
        image_files.extend(input_dir.rglob(f"*{ext}"))
        image_files.extend(input_dir.rglob(f"*{ext.upper()}"))
    
    if not image_files:
        print("❌ No image files found in input directory")
        sys.exit(1)
    
    print(f"Found {len(image_files)} images to process")
    print()
    
    # Check for existing pose detection data
    missing_pose_data = []
    for image_file in image_files:
        sidecar_data = core.sidecar.load_data(image_file, "pose_detection")
        if not sidecar_data:
            missing_pose_data.append(image_file)
    
    if missing_pose_data:
        print(f"⚠️  Warning: {len(missing_pose_data)} images missing pose detection data")
        print("Consider running pose detection first:")
        print(f"  sb pose detect {input_dir}")
        print()
    
    # Perform jersey splitting
    print("Starting jersey color analysis...")
    try:
        results = core.split_games_by_jersey_color(
            image_files,
            output_dir=output_dir,
            save_sidecar=True,
            pose_confidence_threshold=0.7,
            color_similarity_threshold=0.15,
            min_team_photos=5,
        )
        
        if results["success"]:
            print("✅ Jersey splitting completed successfully!")
            
            # Display results
            summary = results.get("summary", {})
            detected_teams = results.get("detected_teams", [])
            
            print(f"\n📊 Results Summary:")
            print(f"  Total photos: {summary.get('total_photos', 0)}")
            print(f"  Detected teams: {len(detected_teams)}")
            print(f"  Split photos: {summary.get('split_photos', 0)}")
            print(f"  Single team photos: {summary.get('single_team_photos', 0)}")
            print(f"  Multi-team photos: {summary.get('multi_team_photos', 0)}")
            print(f"  No team photos: {summary.get('no_team_photos', 0)}")
            print(f"  Average confidence: {summary.get('average_confidence', 0.0):.2f}")
            print(f"  Processing time: {summary.get('processing_time', 0.0):.2f}s")
            
            print(f"\n🏆 Detected Teams:")
            for team in detected_teams:
                color_rgb = team.get("dominant_color", {}).get("rgb_color", (0, 0, 0))
                print(f"  {team.get('team_name', 'Unknown')}: RGB{color_rgb} "
                      f"({team.get('photo_count', 0)} photos, "
                      f"confidence: {team.get('confidence', 0.0):.2f})")
            
            print(f"\n📁 Organized photos saved to: {output_dir}")
            
        else:
            print(f"❌ Jersey splitting failed: {results.get('error', 'Unknown error')}")
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ Error during jersey splitting: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
