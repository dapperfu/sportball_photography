#!/usr/bin/env python3
"""
Simple script to run game detection on September 20th soccer photos.

This script uses the GameDetector to automatically organize photos into game folders.
"""

import sys
from pathlib import Path
from game_detector import GameDetector, GameDetectionConfig

def main():
    """Run game detection on the September 20th photos."""
    
    # Configuration
    input_dir = Path("/keg/pictures/incoming/2025/09-Sep")
    output_dir = Path("/projects/soccer_photo_sorter/results/game_detection")
    pattern = "20250920_*"
    
    print("🏈 Soccer Game Detection System")
    print("=" * 50)
    print(f"Input Directory: {input_dir}")
    print(f"Output Directory: {output_dir}")
    print(f"Pattern: {pattern}")
    print()
    
    # Create configuration
    config = GameDetectionConfig(
        min_game_duration_minutes=30,  # Minimum 30 minutes for a game
        min_gap_minutes=10,            # 10+ minute gap indicates new game
        min_photos_per_game=50,        # At least 50 photos per game
        max_photos_per_game=2000       # Reasonable upper limit
    )
    
    # Create detector
    detector = GameDetector(config)
    
    try:
        # Run detection
        print("🔍 Analyzing photos and detecting games...")
        results = detector.detect_games_from_directory(
            input_dir=input_dir,
            output_dir=output_dir,
            pattern=pattern,
            create_symlinks=True  # Create symlinks for review
        )
        
        if results:
            print("\n✅ Game Detection Complete!")
            print("=" * 50)
            
            summary = results['summary']
            print(f"📊 Summary:")
            print(f"   Total Games: {summary['total_games']}")
            print(f"   Total Photos: {summary['total_photos']}")
            print(f"   Report: {results['report_path']}")
            
            print(f"\n🎮 Detected Games:")
            for game in summary['games']:
                print(f"   Game {game['game_id']:2d}: {game['start_time']} - {game['end_time']} "
                      f"({game['duration_minutes']:5.1f} min, {game['photo_count']:4d} photos)")
                if game['gap_before_minutes']:
                    print(f"              Gap before: {game['gap_before_minutes']:5.1f} min")
                if game['gap_after_minutes']:
                    print(f"              Gap after:  {game['gap_after_minutes']:5.1f} min")
            
            print(f"\n📁 Game Folders Created:")
            for game_id, folder_path in results['game_folders'].items():
                print(f"   {game_id}: {folder_path}")
            
            print(f"\n🔗 Symlinks created for easy review.")
            print(f"   Review the folders in: {output_dir}")
            print(f"   Detailed report: {results['report_path']}")
            
        else:
            print("❌ No games detected. Check your input directory and pattern.")
            return 1
            
    except Exception as e:
        print(f"❌ Error during game detection: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
