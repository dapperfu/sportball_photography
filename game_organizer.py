#!/usr/bin/env python3
"""
Game Organizer - The ONE tool for soccer photo organization.

Mostly automated detection with optional manual splits from text file.

Author: Claude Sonnet 4 (claude-3-5-sonnet-20241022)
Generated via Cursor IDE (cursor.sh) with AI assistance
"""

import sys
from pathlib import Path
from unified_game_organizer import UnifiedGameOrganizer, GameDetectionConfig

def main():
    """Main function for the ONE game organizer tool."""
    
    print("🏈 Game Organizer")
    print("=" * 30)
    print("Mostly automated detection with optional manual splits")
    print()
    
    # Configuration
    input_dir = Path("/keg/pictures/incoming/2025/09-Sep")
    output_dir = Path("/projects/soccer_photo_sorter/results/games")
    pattern = "20250920_*"
    
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
    
    # Create organizer
    organizer = UnifiedGameOrganizer(config)
    
    try:
        # Step 1: Automated detection
        print("🤖 Running automated game detection...")
        results = organizer.detect_games(input_dir, pattern)
        
        if not results:
            print("❌ No games detected automatically")
            return 1
        
        # Show automated results
        print(f"\n📊 Automated Detection Results:")
        for game in organizer.detector.games:
            duration = (game.end_time - game.start_time).total_seconds() / 60
            print(f"   Game {game.game_id}: {game.start_time.strftime('%H:%M:%S')} - {game.end_time.strftime('%H:%M:%S')} ({duration:.1f} min, {game.photo_count} photos)")
        
        # Step 2: Optional manual splits
        split_file = Path("manual_splits.txt")
        if split_file.exists():
            print(f"\n✋ Loading manual splits from {split_file}...")
            organizer.load_split_file(split_file)
        else:
            print(f"\n💡 To add manual splits:")
            print(f"   1. Create {split_file} with one timestamp per line")
            print(f"   2. Example: 14:00:00")
            print(f"   3. Re-run this script")
        
        # Step 3: Apply splits and create folders
        print(f"\n🔧 Applying splits and creating organized folders...")
        organizer.apply_manual_splits()
        game_folders = organizer.create_organized_folders(output_dir, create_symlinks=True)
        
        # Step 4: Generate report
        print(f"\n📊 Generating report...")
        report_path = organizer.generate_comprehensive_report(output_dir)
        
        # Final summary
        print(f"\n✅ Game Organization Complete!")
        print("=" * 40)
        print(f"📊 Summary:")
        print(f"   Automated games: {len(organizer.detector.games)}")
        print(f"   Manual splits: {len(organizer.splitter.manual_splits) if organizer.splitter else 0}")
        print(f"   Final games: {len(organizer.final_games)}")
        print(f"   Total photos: {sum(game.photo_count for game in organizer.final_games)}")
        
        print(f"\n📁 Game Folders Created:")
        for game_id, folder_path in game_folders.items():
            print(f"   {game_id}: {folder_path}")
        
        print(f"\n📊 Final Games:")
        for game in organizer.final_games:
            duration = (game.end_time - game.start_time).total_seconds() / 60
            print(f"   Game {game.game_id}: {game.start_time.strftime('%H:%M:%S')} - {game.end_time.strftime('%H:%M:%S')} ({duration:.1f} min, {game.photo_count} photos)")
        
        print(f"\n🔗 Symlinks created for easy review")
        print(f"   Review the folders in: {output_dir}")
        print(f"   Report: {report_path}")
        
    except Exception as e:
        print(f"❌ Error during organization: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
