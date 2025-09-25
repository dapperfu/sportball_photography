#!/usr/bin/env python3
"""
Simple Split File Demo

This demonstrates the lightweight split file system:
1. Generate a simple text file with suggested splits
2. Edit the file to add manual splits
3. Apply the splits to create organized games
"""

import sys
from pathlib import Path
from simple_split_file import SimpleSplitFileManager
from game_detector import GameDetector, GameDetectionConfig

def main():
    """Demonstrate the simple split file workflow."""
    
    print("📝 Simple Split File System")
    print("=" * 40)
    print("Lightweight system with plain text files")
    print("One timestamp per line - nothing more!")
    print()
    
    # Configuration
    input_dir = Path("/keg/pictures/incoming/2025/09-Sep")
    pattern = "20250920_*"
    split_file = Path("game_splits.txt")
    output_dir = Path("/projects/soccer_photo_sorter/results/simple_splits")
    
    print(f"Input Directory: {input_dir}")
    print(f"Pattern: {pattern}")
    print(f"Split File: {split_file}")
    print(f"Output Directory: {output_dir}")
    print()
    
    # Create detector and manager
    detector_config = GameDetectionConfig()
    detector = GameDetector(detector_config)
    manager = SimpleSplitFileManager(detector)
    
    try:
        # Step 1: Generate split file
        print("🔍 Step 1: Generating split file...")
        manager.generate_split_file(input_dir, split_file, pattern)
        
        # Show what was generated
        print(f"\n📄 Generated split file content:")
        with open(split_file, 'r') as f:
            content = f.read()
            print(content)
        
        # Step 2: Simulate manual editing
        print(f"\n✏️  Step 2: Manual editing (simulated)")
        print(f"   Split file saved to: {split_file}")
        print(f"   You can now edit this file to add manual splits")
        print(f"   Example: Add '14:00:00' on a new line")
        print()
        
        # Simulate adding manual splits
        print("   Simulating manual edits...")
        with open(split_file, 'a') as f:
            f.write("\n# Manual splits added:\n")
            f.write("14:00:00\n")
            f.write("15:30:00\n")
        
        print(f"   Added manual splits: 14:00:00, 15:30:00")
        
        # Step 3: Load and apply splits
        print(f"\n📂 Step 3: Loading and applying splits...")
        manager.load_split_file(split_file)
        
        # Apply splits
        final_games = manager.apply_splits_to_detector()
        
        # Create organized folders
        print(f"\n📁 Step 4: Creating organized folders...")
        game_folders = manager.create_organized_folders(output_dir, final_games, create_symlinks=True)
        
        # Show final results
        print(f"\n✅ Simple Split System Complete!")
        print("=" * 40)
        print(f"📊 Summary:")
        print(f"   Original automated games: {len(detector.games)}")
        print(f"   Manual splits applied: {len(manager.split_timestamps)}")
        print(f"   Final games created: {len(final_games)}")
        print(f"   Total photos organized: {sum(game.photo_count for game in final_games)}")
        
        print(f"\n📊 Final Games:")
        for game in final_games:
            duration = (game.end_time - game.start_time).total_seconds() / 60
            print(f"   Game {game.game_id}: {game.start_time.strftime('%H:%M:%S')} - {game.end_time.strftime('%H:%M:%S')} ({duration:.1f} min, {game.photo_count} photos)")
        
        print(f"\n📁 Game Folders Created:")
        for game_id, folder_path in game_folders.items():
            print(f"   {game_id}: {folder_path}")
        
        print(f"\n💡 Next Steps:")
        print(f"   1. Review the split file: {split_file}")
        print(f"   2. Edit it to add/remove manual splits")
        print(f"   3. Run: python simple_split_file.py --load {split_file} --apply --output {output_dir}")
        print(f"   4. Review the organized folders")
        
    except Exception as e:
        print(f"❌ Error during workflow: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
