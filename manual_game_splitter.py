#!/usr/bin/env python3
"""
Manual Game Splitter for Soccer Photo Sorting

This module allows manual insertion of game boundaries at specific timestamps,
useful for cases where you ran from one game to another without significant gaps.

Author: Claude Sonnet 4 (claude-3-5-sonnet-20241022)
Generated via Cursor IDE (cursor.sh) with AI assistance
"""

import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
import json
from game_detector import GameDetector, GameDetectionConfig, GameSession


class ManualGameSplitter:
    """
    Allows manual insertion of game boundaries at specific timestamps.
    """
    
    def __init__(self, detector: GameDetector):
        """
        Initialize the manual splitter.
        
        Args:
            detector: GameDetector instance with loaded games
        """
        self.detector = detector
        self.manual_splits: List[datetime] = []
    
    def add_manual_split(self, timestamp_str: str) -> bool:
        """
        Add a manual split at a specific timestamp.
        
        Args:
            timestamp_str: Timestamp in format "HH:MM:SS" or "HHMMSS"
            
        Returns:
            True if split was added successfully
        """
        try:
            # Parse timestamp - handle both formats
            if ':' in timestamp_str:
                # Format: "HH:MM:SS"
                timestamp = datetime.strptime(timestamp_str, "%H:%M:%S")
            else:
                # Format: "HHMMSS"
                timestamp = datetime.strptime(timestamp_str, "%H%M%S")
            
            # Convert to full datetime for September 20th, 2025
            full_timestamp = datetime(2025, 9, 20, timestamp.hour, timestamp.minute, timestamp.second)
            
            self.manual_splits.append(full_timestamp)
            self.manual_splits.sort()  # Keep sorted
            
            print(f"✅ Manual split added at {timestamp_str}")
            return True
            
        except ValueError as e:
            print(f"❌ Invalid timestamp format '{timestamp_str}': {e}")
            print("   Use format: HH:MM:SS (e.g., '14:30:00') or HHMMSS (e.g., '143000')")
            return False
    
    def remove_manual_split(self, timestamp_str: str) -> bool:
        """
        Remove a manual split at a specific timestamp.
        
        Args:
            timestamp_str: Timestamp in format "HH:MM:SS" or "HHMMSS"
            
        Returns:
            True if split was removed successfully
        """
        try:
            # Parse timestamp
            if ':' in timestamp_str:
                timestamp = datetime.strptime(timestamp_str, "%H:%M:%S")
            else:
                timestamp = datetime.strptime(timestamp_str, "%H%M%S")
            
            full_timestamp = datetime(2025, 9, 20, timestamp.hour, timestamp.minute, timestamp.second)
            
            if full_timestamp in self.manual_splits:
                self.manual_splits.remove(full_timestamp)
                print(f"✅ Manual split removed at {timestamp_str}")
                return True
            else:
                print(f"❌ No manual split found at {timestamp_str}")
                return False
                
        except ValueError as e:
            print(f"❌ Invalid timestamp format '{timestamp_str}': {e}")
            return False
    
    def list_manual_splits(self):
        """List all current manual splits."""
        if not self.manual_splits:
            print("📝 No manual splits configured")
            return
        
        print("📝 Current manual splits:")
        for i, split in enumerate(self.manual_splits, 1):
            print(f"   {i}. {split.strftime('%H:%M:%S')}")
    
    def apply_manual_splits(self) -> List[GameSession]:
        """
        Apply manual splits to the detected games, creating new game sessions.
        
        Returns:
            List of GameSession objects with manual splits applied
        """
        if not self.manual_splits:
            print("📝 No manual splits to apply")
            return self.detector.games
        
        print(f"🔧 Applying {len(self.manual_splits)} manual splits...")
        
        new_games = []
        game_id = 1
        
        for game in self.detector.games:
            # Check if any manual splits fall within this game
            splits_in_game = [s for s in self.manual_splits 
                            if game.start_time <= s <= game.end_time]
            
            if not splits_in_game:
                # No splits in this game, keep as is
                game.game_id = game_id
                new_games.append(game)
                game_id += 1
            else:
                # Split this game at manual split points
                all_timestamps = [game.start_time] + splits_in_game + [game.end_time]
                all_timestamps.sort()
                
                # Create sub-games
                for i in range(len(all_timestamps) - 1):
                    start_time = all_timestamps[i]
                    end_time = all_timestamps[i + 1]
                    
                    # Find photos in this time range
                    game_photos = [photo for photo in game.photo_files
                                 if start_time <= self.detector.extract_timestamp_from_filename(photo.name) <= end_time]
                    
                    if game_photos:  # Only create game if it has photos
                        sub_game = GameSession(
                            game_id=game_id,
                            start_time=start_time,
                            end_time=end_time,
                            photo_count=len(game_photos),
                            photo_files=game_photos,
                            gap_before=game.gap_before if i == 0 else None,
                            gap_after=game.gap_after if i == len(all_timestamps) - 2 else None
                        )
                        new_games.append(sub_game)
                        game_id += 1
                        
                        print(f"   Game {sub_game.game_id}: {start_time.strftime('%H:%M:%S')} - {end_time.strftime('%H:%M:%S')} ({len(game_photos)} photos)")
        
        print(f"✅ Created {len(new_games)} games with manual splits applied")
        return new_games
    
    def interactive_mode(self):
        """Run interactive mode for adding/removing manual splits."""
        print("\n🎮 Manual Game Splitter - Interactive Mode")
        print("=" * 50)
        print("Commands:")
        print("  add <timestamp>  - Add split at timestamp (e.g., 'add 14:30:00')")
        print("  remove <timestamp> - Remove split at timestamp")
        print("  list             - List current splits")
        print("  apply            - Apply splits and show results")
        print("  save             - Save splits to file")
        print("  load <file>      - Load splits from file")
        print("  quit             - Exit interactive mode")
        print()
        
        while True:
            try:
                command = input("splitter> ").strip().lower()
                
                if command == "quit" or command == "exit":
                    break
                elif command == "list":
                    self.list_manual_splits()
                elif command == "apply":
                    new_games = self.apply_manual_splits()
                    print(f"\n📊 Results: {len(new_games)} games total")
                    for game in new_games:
                        duration = (game.end_time - game.start_time).total_seconds() / 60
                        print(f"   Game {game.game_id}: {game.start_time.strftime('%H:%M:%S')} - {game.end_time.strftime('%H:%M:%S')} ({duration:.1f} min, {game.photo_count} photos)")
                elif command.startswith("add "):
                    timestamp = command[4:].strip()
                    self.add_manual_split(timestamp)
                elif command.startswith("remove "):
                    timestamp = command[7:].strip()
                    self.remove_manual_split(timestamp)
                elif command == "save":
                    self.save_splits()
                elif command.startswith("load "):
                    filename = command[5:].strip()
                    self.load_splits(filename)
                else:
                    print("❌ Unknown command. Type 'quit' to exit.")
                    
            except KeyboardInterrupt:
                print("\n👋 Exiting interactive mode...")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    def save_splits(self, filename: str = "manual_splits.json"):
        """Save manual splits to a file."""
        splits_data = {
            'manual_splits': [split.isoformat() for split in self.manual_splits],
            'created_at': datetime.now().isoformat()
        }
        
        with open(filename, 'w') as f:
            json.dump(splits_data, f, indent=2)
        
        print(f"💾 Manual splits saved to {filename}")
    
    def load_splits(self, filename: str):
        """Load manual splits from a file."""
        try:
            with open(filename, 'r') as f:
                splits_data = json.load(f)
            
            self.manual_splits = [datetime.fromisoformat(ts) for ts in splits_data['manual_splits']]
            self.manual_splits.sort()
            
            print(f"📂 Loaded {len(self.manual_splits)} manual splits from {filename}")
            
        except FileNotFoundError:
            print(f"❌ File {filename} not found")
        except Exception as e:
            print(f"❌ Error loading splits: {e}")


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Manual game splitter for soccer photos")
    parser.add_argument("--input", "-i", type=Path, required=True, help="Input directory")
    parser.add_argument("--pattern", "-p", default="20250920_*", help="File pattern")
    parser.add_argument("--add-split", "-a", help="Add split at timestamp (HH:MM:SS)")
    parser.add_argument("--remove-split", "-r", help="Remove split at timestamp")
    parser.add_argument("--list", "-l", action="store_true", help="List current splits")
    parser.add_argument("--apply", action="store_true", help="Apply splits and show results")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    parser.add_argument("--save", help="Save splits to file")
    parser.add_argument("--load", help="Load splits from file")
    
    args = parser.parse_args()
    
    # Create detector and load games
    config = GameDetectionConfig()
    detector = GameDetector(config)
    
    print("🔍 Loading games from directory...")
    results = detector.detect_games_from_directory(
        input_dir=args.input,
        output_dir=Path("/tmp"),  # Temporary output
        pattern=args.pattern,
        create_symlinks=False  # Don't create files for this
    )
    
    if not results:
        print("❌ No games detected")
        return 1
    
    # Create splitter
    splitter = ManualGameSplitter(detector)
    
    # Load splits if specified
    if args.load:
        splitter.load_splits(args.load)
    
    # Handle commands
    if args.add_split:
        splitter.add_manual_split(args.add_split)
    
    if args.remove_split:
        splitter.remove_manual_split(args.remove_split)
    
    if args.list:
        splitter.list_manual_splits()
    
    if args.apply:
        new_games = splitter.apply_manual_splits()
        print(f"\n📊 Final Results: {len(new_games)} games")
        for game in new_games:
            duration = (game.end_time - game.start_time).total_seconds() / 60
            print(f"   Game {game.game_id}: {game.start_time.strftime('%H:%M:%S')} - {game.end_time.strftime('%H:%M:%S')} ({duration:.1f} min, {game.photo_count} photos)")
    
    if args.save:
        splitter.save_splits(args.save)
    
    if args.interactive:
        splitter.interactive_mode()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
