#!/usr/bin/env python3
"""
Simple Split File System for Game Splitting

This module handles generation and ingestion of simple plain text files
with one timestamp per line for manual game splits.

Author: Claude Sonnet 4 (claude-3-5-sonnet-20241022)
Generated via Cursor IDE (cursor.sh) with AI assistance
"""

import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict
from game_detector import GameDetector, GameDetectionConfig, GameSession


class SimpleSplitFileManager:
    """
    Manages simple plain text split files.
    
    Format: One timestamp per line in ISO format or HH:MM:SS
    """
    
    def __init__(self, detector: Optional[GameDetector] = None):
        """
        Initialize the split file manager.
        
        Args:
            detector: Optional GameDetector instance
        """
        self.detector = detector
        self.split_timestamps: List[datetime] = []
    
    def generate_split_file(self, 
                           input_dir: Path,
                           output_file: Path,
                           pattern: str = "20250920_*") -> Path:
        """
        Generate a simple split file from automated detection.
        
        Args:
            input_dir: Input directory containing photos
            output_file: Path to save split file
            pattern: File pattern to match
            
        Returns:
            Path to generated split file
        """
        print("🔍 Running automated detection to generate split file...")
        
        # Run automated detection
        results = self.detector.detect_games_from_directory(
            input_dir=input_dir,
            output_dir=Path("/tmp"),  # Temporary output
            create_symlinks=False
        )
        
        if not results:
            raise ValueError("No games detected automatically")
        
        # Generate split file content
        content_lines = []
        content_lines.append("# Game Split File")
        content_lines.append("# Generated from automated detection")
        content_lines.append(f"# Source: {input_dir}")
        content_lines.append(f"# Pattern: {pattern}")
        content_lines.append(f"# Generated: {datetime.now().isoformat()}")
        content_lines.append("")
        content_lines.append("# Add manual splits below (one timestamp per line)")
        content_lines.append("# Format: HH:MM:SS or YYYY-MM-DDTHH:MM:SS")
        content_lines.append("# Example: 14:00:00")
        content_lines.append("")
        
        # Add suggested splits based on game boundaries
        content_lines.append("# Suggested splits (uncomment to use):")
        for game in self.detector.games:
            # Add split at start of each game (except first)
            if game.game_id > 1:
                timestamp = game.start_time.strftime("%H:%M:%S")
                content_lines.append(f"# {timestamp}  # Start of Game {game.game_id}")
        
        content_lines.append("")
        content_lines.append("# Manual splits:")
        
        # Write to file
        with open(output_file, 'w') as f:
            f.write('\n'.join(content_lines))
        
        print(f"📝 Split file generated: {output_file}")
        print(f"   Edit this file to add manual splits (one timestamp per line)")
        
        return output_file
    
    def load_split_file(self, split_file: Path) -> List[datetime]:
        """
        Load timestamps from split file.
        
        Args:
            split_file: Path to split file
            
        Returns:
            List of datetime objects
        """
        timestamps = []
        
        with open(split_file, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                
                # Skip empty lines and comments
                if not line or line.startswith('#'):
                    continue
                
                # Remove inline comments
                if '#' in line:
                    line = line.split('#')[0].strip()
                
                if not line:
                    continue
                
                try:
                    # Try parsing as HH:MM:SS first
                    if ':' in line and len(line) <= 8:
                        timestamp = datetime.strptime(line, "%H:%M:%S")
                        # Convert to full datetime for September 20th, 2025
                        full_timestamp = datetime(2025, 9, 20, timestamp.hour, timestamp.minute, timestamp.second)
                        timestamps.append(full_timestamp)
                    # Try parsing as ISO format
                    elif 'T' in line:
                        timestamp = datetime.fromisoformat(line)
                        timestamps.append(timestamp)
                    else:
                        print(f"⚠️  Line {line_num}: Invalid timestamp format '{line}'")
                        
                except ValueError as e:
                    print(f"⚠️  Line {line_num}: Could not parse timestamp '{line}': {e}")
        
        # Sort timestamps
        timestamps.sort()
        
        print(f"📂 Loaded {len(timestamps)} timestamps from {split_file}")
        for i, ts in enumerate(timestamps, 1):
            print(f"   {i}. {ts.strftime('%H:%M:%S')}")
        
        self.split_timestamps = timestamps
        return timestamps
    
    def apply_splits_to_detector(self) -> List[GameSession]:
        """
        Apply loaded splits to create final game sessions.
        
        Returns:
            List of final GameSession objects
        """
        if not self.split_timestamps:
            print("❌ No splits loaded")
            return self.detector.games
        
        print("🔧 Applying manual splits...")
        
        # Create manual splitter
        from manual_game_splitter import ManualGameSplitter
        splitter = ManualGameSplitter(self.detector)
        
        # Add each timestamp as a split
        for timestamp in self.split_timestamps:
            timestamp_str = timestamp.strftime("%H:%M:%S")
            splitter.add_manual_split(timestamp_str)
        
        # Apply splits
        final_games = splitter.apply_manual_splits()
        
        print(f"✅ Applied {len(self.split_timestamps)} splits: {len(final_games)} final games")
        return final_games
    
    def create_organized_folders(self, 
                               output_dir: Path,
                               final_games: List[GameSession],
                               create_symlinks: bool = True) -> Dict[str, Path]:
        """
        Create organized folders for the final games.
        
        Args:
            output_dir: Output directory for game folders
            final_games: List of final GameSession objects
            create_symlinks: Whether to create symlinks (True) or copy files (False)
            
        Returns:
            Dictionary mapping game IDs to folder paths
        """
        print(f"📁 Creating organized folders for {len(final_games)} games...")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        game_folders = {}
        
        for game in final_games:
            # Create game folder name
            game_folder_name = f"Game_{game.game_id:02d}_{game.start_time.strftime('%H%M')}-{game.end_time.strftime('%H%M')}"
            game_folder = output_dir / game_folder_name
            game_folder.mkdir(exist_ok=True)
            
            print(f"   Creating {game_folder_name} with {game.photo_count} photos")
            
            # Copy or symlink photos
            for photo_path in game.photo_files:
                if create_symlinks:
                    symlink_path = game_folder / photo_path.name
                    if not symlink_path.exists():
                        symlink_path.symlink_to(photo_path)
                else:
                    dest_path = game_folder / photo_path.name
                    if not dest_path.exists():
                        import shutil
                        shutil.copy2(photo_path, dest_path)
            
            game_folders[f"Game_{game.game_id:02d}"] = game_folder
        
        return game_folders


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Simple split file manager for game splitting")
    parser.add_argument("--input", "-i", type=Path, help="Input directory")
    parser.add_argument("--pattern", "-p", default="20250920_*", help="File pattern")
    parser.add_argument("--generate", "-g", action="store_true", help="Generate split file")
    parser.add_argument("--load", "-l", type=Path, help="Load split file")
    parser.add_argument("--apply", "-a", action="store_true", help="Apply splits")
    parser.add_argument("--output", "-o", type=Path, help="Output directory for organized games")
    parser.add_argument("--copy", action="store_true", help="Copy files instead of creating symlinks")
    parser.add_argument("--split-file", "-s", type=Path, help="Split file path")
    
    args = parser.parse_args()
    
    # Create detector
    config = GameDetectionConfig()
    detector = GameDetector(config)
    
    # Create manager
    manager = SimpleSplitFileManager(detector)
    
    try:
        if args.generate:
            if not args.input:
                print("❌ --input required for --generate")
                return 1
            
            # Generate split file
            split_file = args.split_file or Path("game_splits.txt")
            manager.generate_split_file(args.input, split_file, args.pattern)
            
            print(f"\n💡 Next steps:")
            print(f"   1. Edit {split_file} to add manual splits")
            print(f"   2. Run: python simple_split_file.py --load {split_file} --apply --output /path/to/output")
        
        elif args.load:
            # Load split file
            manager.load_split_file(args.load)
            
            if args.apply:
                # Apply splits
                final_games = manager.apply_splits_to_detector()
                
                if args.output:
                    # Create organized folders
                    game_folders = manager.create_organized_folders(
                        args.output, 
                        final_games, 
                        create_symlinks=not args.copy
                    )
                    
                    print(f"\n✅ Organization complete!")
                    print(f"   Output directory: {args.output}")
                    print(f"   Games created: {len(final_games)}")
                    print(f"   Total photos: {sum(game.photo_count for game in final_games)}")
                else:
                    print(f"\n📊 Final Results: {len(final_games)} games")
                    for game in final_games:
                        duration = (game.end_time - game.start_time).total_seconds() / 60
                        print(f"   Game {game.game_id}: {game.start_time.strftime('%H:%M:%S')} - {game.end_time.strftime('%H:%M:%S')} ({duration:.1f} min, {game.photo_count} photos)")
        
        else:
            print("❌ Must specify --generate or --load")
            return 1
        
        return 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
