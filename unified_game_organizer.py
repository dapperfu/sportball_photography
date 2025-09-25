#!/usr/bin/env python3
"""
Unified Game Organizer for Soccer Photo Sorting

This module combines automated game detection with manual splitting capabilities,
providing a complete workflow for organizing soccer photos into game folders.

Author: Claude Sonnet 4 (claude-3-5-sonnet-20241022)
Generated via Cursor IDE (cursor.sh) with AI assistance
"""

import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
import json
from game_detector import GameDetector, GameDetectionConfig, GameSession
from manual_game_splitter import ManualGameSplitter


class UnifiedGameOrganizer:
    """
    Unified system that combines automated detection with manual splitting.
    
    This class provides a complete workflow:
    1. Automatically detect games based on temporal gaps
    2. Allow manual splits to be added for edge cases
    3. Create organized folders with symlinks or copies
    4. Generate comprehensive reports
    """
    
    def __init__(self, config: Optional[GameDetectionConfig] = None):
        """
        Initialize the unified organizer.
        
        Args:
            config: Optional configuration for game detection
        """
        self.config = config or GameDetectionConfig()
        self.detector = GameDetector(self.config)
        self.splitter = None
        self.final_games: List[GameSession] = []
        
    def detect_games(self, 
                    input_dir: Path, 
                    pattern: str = "20250920_*") -> Dict:
        """
        Run automated game detection.
        
        Args:
            input_dir: Input directory containing photos
            pattern: File pattern to match
            
        Returns:
            Dictionary containing detection results
        """
        print("🤖 Running automated game detection...")
        
        results = self.detector.detect_games_from_directory(
            input_dir=input_dir,
            output_dir=Path("/tmp"),  # Temporary output
            pattern=pattern,
            create_symlinks=False  # Don't create files yet
        )
        
        if results:
            print(f"✅ Automated detection found {len(self.detector.games)} games")
            self.splitter = ManualGameSplitter(self.detector)
            return results
        else:
            print("❌ No games detected automatically")
            return {}
    
    def add_manual_split(self, timestamp_str: str) -> bool:
        """
        Add a manual split point.
        
        Args:
            timestamp_str: Timestamp in format "HH:MM:SS" or "HHMMSS"
            
        Returns:
            True if split was added successfully
        """
        if not self.splitter:
            print("❌ Run automated detection first")
            return False
        
        return self.splitter.add_manual_split(timestamp_str)
    
    def remove_manual_split(self, timestamp_str: str) -> bool:
        """
        Remove a manual split point.
        
        Args:
            timestamp_str: Timestamp in format "HH:MM:SS" or "HHMMSS"
            
        Returns:
            True if split was removed successfully
        """
        if not self.splitter:
            print("❌ Run automated detection first")
            return False
        
        return self.splitter.remove_manual_split(timestamp_str)
    
    def list_manual_splits(self):
        """List all manual splits."""
        if not self.splitter:
            print("❌ Run automated detection first")
            return
        
        self.splitter.list_manual_splits()
    
    def apply_manual_splits(self) -> List[GameSession]:
        """
        Apply manual splits to create final game list.
        
        Returns:
            List of final GameSession objects
        """
        if not self.splitter:
            print("❌ Run automated detection first")
            return []
        
        print("🔧 Applying manual splits to automated detection...")
        self.final_games = self.splitter.apply_manual_splits()
        
        print(f"📊 Final Results:")
        print(f"   Automated games: {len(self.detector.games)}")
        print(f"   Manual splits: {len(self.splitter.manual_splits)}")
        print(f"   Final games: {len(self.final_games)}")
        
        return self.final_games
    
    def create_organized_folders(self, 
                               output_dir: Path,
                               create_symlinks: bool = True,
                               use_final_games: bool = True) -> Dict[str, Path]:
        """
        Create organized folders for the games.
        
        Args:
            output_dir: Output directory for game folders
            create_symlinks: Whether to create symlinks (True) or copy files (False)
            use_final_games: Whether to use final games (True) or original games (False)
            
        Returns:
            Dictionary mapping game IDs to folder paths
        """
        games_to_organize = self.final_games if use_final_games else self.detector.games
        
        if not games_to_organize:
            print("❌ No games to organize")
            return {}
        
        print(f"📁 Creating organized folders for {len(games_to_organize)} games...")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        game_folders = {}
        
        for game in games_to_organize:
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
    
    def generate_comprehensive_report(self, output_dir: Path) -> Path:
        """
        Generate a comprehensive report including both automated and manual results.
        
        Args:
            output_dir: Output directory
            
        Returns:
            Path to the generated report
        """
        report_path = output_dir / "comprehensive_game_report.json"
        
        report = {
            'workflow_info': {
                'automated_detection': True,
                'manual_splits_applied': len(self.splitter.manual_splits) if self.splitter else 0,
                'final_games_count': len(self.final_games),
                'created_at': datetime.now().isoformat()
            },
            'detection_config': {
                'min_game_duration_minutes': self.config.min_game_duration_minutes,
                'max_gap_minutes': self.config.max_gap_minutes,
                'min_gap_minutes': self.config.min_gap_minutes,
                'min_photos_per_game': self.config.min_photos_per_game,
                'max_photos_per_game': self.config.max_photos_per_game
            },
            'automated_results': {
                'total_games': len(self.detector.games),
                'total_photos': sum(game.photo_count for game in self.detector.games),
                'games': []
            },
            'manual_splits': [split.isoformat() for split in self.splitter.manual_splits] if self.splitter else [],
            'final_results': {
                'total_games': len(self.final_games),
                'total_photos': sum(game.photo_count for game in self.final_games),
                'games': []
            }
        }
        
        # Add automated games
        for game in self.detector.games:
            duration_minutes = (game.end_time - game.start_time).total_seconds() / 60
            game_info = {
                'game_id': game.game_id,
                'start_time': game.start_time.strftime('%H:%M:%S'),
                'end_time': game.end_time.strftime('%H:%M:%S'),
                'duration_minutes': round(duration_minutes, 1),
                'photo_count': game.photo_count,
                'gap_before_minutes': round(game.gap_before / 60, 1) if game.gap_before else None,
                'gap_after_minutes': round(game.gap_after / 60, 1) if game.gap_after else None
            }
            report['automated_results']['games'].append(game_info)
        
        # Add final games
        for game in self.final_games:
            duration_minutes = (game.end_time - game.start_time).total_seconds() / 60
            game_info = {
                'game_id': game.game_id,
                'start_time': game.start_time.strftime('%H:%M:%S'),
                'end_time': game.end_time.strftime('%H:%M:%S'),
                'duration_minutes': round(duration_minutes, 1),
                'photo_count': game.photo_count,
                'gap_before_minutes': round(game.gap_before / 60, 1) if game.gap_before else None,
                'gap_after_minutes': round(game.gap_after / 60, 1) if game.gap_after else None
            }
            report['final_results']['games'].append(game_info)
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"📊 Comprehensive report saved to {report_path}")
        return report_path
    
    def interactive_workflow(self, 
                           input_dir: Path, 
                           output_dir: Path,
                           pattern: str = "20250920_*"):
        """
        Run interactive workflow combining automated detection and manual splitting.
        
        Args:
            input_dir: Input directory containing photos
            output_dir: Output directory for organized games
            pattern: File pattern to match
        """
        print("🎮 Unified Game Organizer - Interactive Workflow")
        print("=" * 60)
        
        # Step 1: Automated detection
        print("\n🤖 Step 1: Automated Game Detection")
        results = self.detect_games(input_dir, pattern)
        
        if not results:
            print("❌ No games detected. Exiting.")
            return
        
        # Show automated results
        print(f"\n📊 Automated Detection Results:")
        for game in self.detector.games:
            duration = (game.end_time - game.start_time).total_seconds() / 60
            print(f"   Game {game.game_id}: {game.start_time.strftime('%H:%M:%S')} - {game.end_time.strftime('%H:%M:%S')} ({duration:.1f} min, {game.photo_count} photos)")
        
        # Step 2: Manual splitting
        print(f"\n✋ Step 2: Manual Splitting (Optional)")
        print("Add manual splits for cases where you ran from one game to another:")
        print("Commands: add <timestamp>, remove <timestamp>, list, apply, skip")
        
        while True:
            try:
                command = input("\nmanual> ").strip().lower()
                
                if command == "skip" or command == "done":
                    break
                elif command == "list":
                    self.list_manual_splits()
                elif command == "apply":
                    self.apply_manual_splits()
                    break
                elif command.startswith("add "):
                    timestamp = command[4:].strip()
                    self.add_manual_split(timestamp)
                elif command.startswith("remove "):
                    timestamp = command[7:].strip()
                    self.remove_manual_split(timestamp)
                else:
                    print("❌ Unknown command. Use: add <timestamp>, remove <timestamp>, list, apply, skip")
                    
            except KeyboardInterrupt:
                print("\n👋 Skipping manual splits...")
                break
        
        # Step 3: Apply splits if any were added
        if self.splitter and self.splitter.manual_splits:
            self.apply_manual_splits()
        else:
            self.final_games = self.detector.games
        
        # Step 4: Create organized folders
        print(f"\n📁 Step 3: Creating Organized Folders")
        create_symlinks = input("Create symlinks (y) or copy files (n)? [y]: ").strip().lower()
        create_symlinks = create_symlinks != 'n'
        
        game_folders = self.create_organized_folders(output_dir, create_symlinks)
        
        # Step 5: Generate report
        print(f"\n📊 Step 4: Generating Report")
        report_path = self.generate_comprehensive_report(output_dir)
        
        # Final summary
        print(f"\n✅ Workflow Complete!")
        print(f"   Automated games: {len(self.detector.games)}")
        print(f"   Manual splits: {len(self.splitter.manual_splits) if self.splitter else 0}")
        print(f"   Final games: {len(self.final_games)}")
        print(f"   Total photos: {sum(game.photo_count for game in self.final_games)}")
        print(f"   Output directory: {output_dir}")
        print(f"   Report: {report_path}")
        
        if create_symlinks:
            print(f"   Note: Symlinks created for easy review")


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Unified game organizer for soccer photos")
    parser.add_argument("--input", "-i", type=Path, required=True, help="Input directory")
    parser.add_argument("--output", "-o", type=Path, required=True, help="Output directory")
    parser.add_argument("--pattern", "-p", default="20250920_*", help="File pattern")
    parser.add_argument("--add-split", "-a", help="Add split at timestamp (HH:MM:SS)")
    parser.add_argument("--remove-split", "-r", help="Remove split at timestamp")
    parser.add_argument("--list", "-l", action="store_true", help="List current splits")
    parser.add_argument("--copy", action="store_true", help="Copy files instead of creating symlinks")
    parser.add_argument("--interactive", action="store_true", help="Run interactive workflow")
    parser.add_argument("--min-duration", type=int, default=30, help="Minimum game duration (minutes)")
    parser.add_argument("--min-gap", type=int, default=10, help="Minimum gap to separate games (minutes)")
    
    args = parser.parse_args()
    
    # Create configuration
    config = GameDetectionConfig(
        min_game_duration_minutes=args.min_duration,
        min_gap_minutes=args.min_gap
    )
    
    # Create organizer
    organizer = UnifiedGameOrganizer(config)
    
    if args.interactive:
        # Run interactive workflow
        organizer.interactive_workflow(args.input, args.output, args.pattern)
    else:
        # Run automated workflow with optional manual splits
        print("🤖 Running automated detection...")
        results = organizer.detect_games(args.input, args.pattern)
        
        if not results:
            print("❌ No games detected")
            return 1
        
        # Handle manual splits if specified
        if args.add_split:
            organizer.add_manual_split(args.add_split)
        
        if args.remove_split:
            organizer.remove_manual_split(args.remove_split)
        
        if args.list:
            organizer.list_manual_splits()
        
        # Apply splits and create folders
        organizer.apply_manual_splits()
        organizer.create_organized_folders(args.output, create_symlinks=not args.copy)
        organizer.generate_comprehensive_report(args.output)
        
        print("✅ Organization complete!")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
