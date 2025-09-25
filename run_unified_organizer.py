#!/usr/bin/env python3
"""
Simple script to run the unified game organizer.

This demonstrates the complete workflow: automated detection + manual splitting.
"""

import sys
from pathlib import Path
from unified_game_organizer import UnifiedGameOrganizer, GameDetectionConfig

def main():
    """Run the unified game organizer."""
    
    print("🎮 Unified Game Organizer")
    print("=" * 50)
    print("This combines automated detection with manual splitting capabilities")
    print()
    
    # Configuration
    input_dir = Path("/keg/pictures/incoming/2025/09-Sep")
    output_dir = Path("/projects/soccer_photo_sorter/results/unified_games")
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
        print("🤖 Step 1: Running automated game detection...")
        results = organizer.detect_games(input_dir, pattern)
        
        if not results:
            print("❌ No games detected automatically")
            return 1
        
        # Show automated results
        print(f"\n📊 Automated Detection Results:")
        for game in organizer.detector.games:
            duration = (game.end_time - game.start_time).total_seconds() / 60
            print(f"   Game {game.game_id}: {game.start_time.strftime('%H:%M:%S')} - {game.end_time.strftime('%H:%M:%S')} ({duration:.1f} min, {game.photo_count} photos)")
        
        # Step 2: Example manual split (if you ran from one game to another)
        print(f"\n✋ Step 2: Adding example manual split...")
        print("   Adding split at 14:00:00 (example: if you ran from one game to another)")
        organizer.add_manual_split("14:00:00")
        
        # Show manual splits
        print(f"\n📝 Current manual splits:")
        organizer.list_manual_splits()
        
        # Step 3: Apply manual splits
        print(f"\n🔧 Step 3: Applying manual splits...")
        final_games = organizer.apply_manual_splits()
        
        # Step 4: Create organized folders
        print(f"\n📁 Step 4: Creating organized folders...")
        game_folders = organizer.create_organized_folders(output_dir, create_symlinks=True)
        
        # Step 5: Generate comprehensive report
        print(f"\n📊 Step 5: Generating comprehensive report...")
        report_path = organizer.generate_comprehensive_report(output_dir)
        
        # Final summary
        print(f"\n✅ Unified Organization Complete!")
        print("=" * 50)
        print(f"📊 Summary:")
        print(f"   Automated games detected: {len(organizer.detector.games)}")
        print(f"   Manual splits added: {len(organizer.splitter.manual_splits)}")
        print(f"   Final games created: {len(final_games)}")
        print(f"   Total photos organized: {sum(game.photo_count for game in final_games)}")
        
        print(f"\n📁 Game Folders Created:")
        for game_id, folder_path in game_folders.items():
            print(f"   {game_id}: {folder_path}")
        
        print(f"\n📊 Final Games:")
        for game in final_games:
            duration = (game.end_time - game.start_time).total_seconds() / 60
            print(f"   Game {game.game_id}: {game.start_time.strftime('%H:%M:%S')} - {game.end_time.strftime('%H:%M:%S')} ({duration:.1f} min, {game.photo_count} photos)")
        
        print(f"\n🔗 Symlinks created for easy review")
        print(f"   Review the folders in: {output_dir}")
        print(f"   Comprehensive report: {report_path}")
        
        print(f"\n💡 To run interactively:")
        print(f"   python unified_game_organizer.py --input {input_dir} --output {output_dir} --interactive")
        
    except Exception as e:
        print(f"❌ Error during organization: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
