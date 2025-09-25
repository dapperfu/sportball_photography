#!/usr/bin/env python3
"""
Simple script to demonstrate manual game splitting.

This script shows how to add manual splits to the detected games.
"""

import sys
from pathlib import Path
from game_detector import GameDetector, GameDetectionConfig
from manual_game_splitter import ManualGameSplitter

def main():
    """Demonstrate manual game splitting."""
    
    print("🎮 Manual Game Splitter Demo")
    print("=" * 50)
    
    # Configuration
    input_dir = Path("/keg/pictures/incoming/2025/09-Sep")
    pattern = "20250920_*"
    
    # Create detector and load games
    config = GameDetectionConfig()
    detector = GameDetector(config)
    
    print("🔍 Loading games from directory...")
    results = detector.detect_games_from_directory(
        input_dir=input_dir,
        output_dir=Path("/tmp"),  # Temporary output
        pattern=pattern,
        create_symlinks=False
    )
    
    if not results:
        print("❌ No games detected")
        return 1
    
    print(f"✅ Loaded {len(detector.games)} games")
    
    # Create splitter
    splitter = ManualGameSplitter(detector)
    
    # Example: Add a manual split at 14:00:00 (if you ran from one game to another)
    print("\n📝 Example: Adding manual split at 14:00:00")
    splitter.add_manual_split("14:00:00")
    
    # Show current splits
    print("\n📝 Current manual splits:")
    splitter.list_manual_splits()
    
    # Apply splits and show results
    print("\n🔧 Applying manual splits...")
    new_games = splitter.apply_manual_splits()
    
    print(f"\n📊 Results with manual splits:")
    print(f"   Original games: {len(detector.games)}")
    print(f"   New games: {len(new_games)}")
    
    for game in new_games:
        duration = (game.end_time - game.start_time).total_seconds() / 60
        print(f"   Game {game.game_id}: {game.start_time.strftime('%H:%M:%S')} - {game.end_time.strftime('%H:%M:%S')} ({duration:.1f} min, {game.photo_count} photos)")
    
    # Save splits for later use
    splitter.save_splits("example_manual_splits.json")
    
    print(f"\n💡 To use interactively, run:")
    print(f"   python manual_game_splitter.py --input {input_dir} --interactive")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
