#!/usr/bin/env python3
"""
Game Split Configuration Management

This module handles generation, editing, and ingestion of game split configuration files.
Allows users to create, modify, and apply manual splits through configuration files.

Author: Claude Sonnet 4 (claude-3-5-sonnet-20241022)
Generated via Cursor IDE (cursor.sh) with AI assistance
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, asdict
from game_detector import GameDetector, GameDetectionConfig, GameSession


@dataclass
class ManualSplit:
    """Represents a manual split point."""
    
    timestamp: str  # Format: "HH:MM:SS"
    description: str  # Human-readable description
    reason: str  # Why this split was added
    confidence: str  # "high", "medium", "low"


@dataclass
class GameSplitConfig:
    """Configuration for game splitting."""
    
    # Metadata
    created_at: str
    created_by: str
    source_directory: str
    file_pattern: str
    total_photos: int
    
    # Detection settings
    detection_config: Dict[str, Any]
    
    # Automated results
    automated_games: List[Dict[str, Any]]
    
    # Manual splits
    manual_splits: List[ManualSplit]
    
    # Notes and comments
    notes: str
    version: str = "1.0"


class GameSplitConfigManager:
    """
    Manages game split configuration files.
    
    This class provides functionality to:
    1. Generate configuration files from automated detection
    2. Load and validate existing configurations
    3. Apply configurations to create final game organization
    """
    
    def __init__(self, detector: Optional[GameDetector] = None):
        """
        Initialize the configuration manager.
        
        Args:
            detector: Optional GameDetector instance
        """
        self.detector = detector
        self.config: Optional[GameSplitConfig] = None
    
    def generate_config_from_detection(self, 
                                     input_dir: Path,
                                     pattern: str = "20250920_*",
                                     notes: str = "",
                                     created_by: str = "User") -> GameSplitConfig:
        """
        Generate a configuration file from automated detection results.
        
        Args:
            input_dir: Input directory containing photos
            pattern: File pattern to match
            notes: Optional notes about the detection
            created_by: Who created this configuration
            
        Returns:
            GameSplitConfig object ready for editing
        """
        print("🔍 Running automated detection to generate configuration...")
        
        # Run automated detection
        results = self.detector.detect_games_from_directory(
            input_dir=input_dir,
            output_dir=Path("/tmp"),  # Temporary output
            pattern=pattern,
            create_symlinks=False
        )
        
        if not results:
            raise ValueError("No games detected automatically")
        
        # Extract automated games data
        automated_games = []
        for game in self.detector.games:
            duration_minutes = (game.end_time - game.start_time).total_seconds() / 60
            game_data = {
                'game_id': game.game_id,
                'start_time': game.start_time.strftime('%H:%M:%S'),
                'end_time': game.end_time.strftime('%H:%M:%S'),
                'duration_minutes': round(duration_minutes, 1),
                'photo_count': game.photo_count,
                'gap_before_minutes': round(game.gap_before / 60, 1) if game.gap_before else None,
                'gap_after_minutes': round(game.gap_after / 60, 1) if game.gap_after else None
            }
            automated_games.append(game_data)
        
        # Create configuration
        config = GameSplitConfig(
            created_at=datetime.now().isoformat(),
            created_by=created_by,
            source_directory=str(input_dir),
            file_pattern=pattern,
            total_photos=sum(game.photo_count for game in self.detector.games),
            detection_config={
                'min_game_duration_minutes': self.detector.config.min_game_duration_minutes,
                'max_gap_minutes': self.detector.config.max_gap_minutes,
                'min_gap_minutes': self.detector.config.min_gap_minutes,
                'min_photos_per_game': self.detector.config.min_photos_per_game,
                'max_photos_per_game': self.detector.config.max_photos_per_game
            },
            automated_games=automated_games,
            manual_splits=[],  # Empty - ready for user to add
            notes=notes
        )
        
        self.config = config
        print(f"✅ Configuration generated with {len(automated_games)} automated games")
        return config
    
    def save_config(self, filepath: Path) -> Path:
        """
        Save configuration to file.
        
        Args:
            filepath: Path to save configuration file
            
        Returns:
            Path to saved file
        """
        if not self.config:
            raise ValueError("No configuration loaded")
        
        # Convert to dictionary for JSON serialization
        config_dict = asdict(self.config)
        
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        print(f"💾 Configuration saved to {filepath}")
        return filepath
    
    def load_config(self, filepath: Path) -> GameSplitConfig:
        """
        Load configuration from file.
        
        Args:
            filepath: Path to configuration file
            
        Returns:
            Loaded GameSplitConfig object
        """
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        
        # Convert manual splits back to objects
        manual_splits = [ManualSplit(**split) for split in config_dict['manual_splits']]
        
        # Create config object
        config = GameSplitConfig(
            created_at=config_dict['created_at'],
            created_by=config_dict['created_by'],
            source_directory=config_dict['source_directory'],
            file_pattern=config_dict['file_pattern'],
            total_photos=config_dict['total_photos'],
            detection_config=config_dict['detection_config'],
            automated_games=config_dict['automated_games'],
            manual_splits=manual_splits,
            notes=config_dict['notes'],
            version=config_dict.get('version', '1.0')
        )
        
        self.config = config
        print(f"📂 Configuration loaded from {filepath}")
        print(f"   Created: {config.created_at}")
        print(f"   Created by: {config.created_by}")
        print(f"   Automated games: {len(config.automated_games)}")
        print(f"   Manual splits: {len(config.manual_splits)}")
        
        return config
    
    def validate_config(self) -> List[str]:
        """
        Validate the loaded configuration.
        
        Returns:
            List of validation errors (empty if valid)
        """
        if not self.config:
            return ["No configuration loaded"]
        
        errors = []
        
        # Validate manual splits
        for i, split in enumerate(self.config.manual_splits):
            try:
                # Validate timestamp format
                datetime.strptime(split.timestamp, "%H:%M:%S")
            except ValueError:
                errors.append(f"Manual split {i+1}: Invalid timestamp format '{split.timestamp}'")
            
            # Validate confidence
            if split.confidence not in ["high", "medium", "low"]:
                errors.append(f"Manual split {i+1}: Invalid confidence '{split.confidence}'")
        
        # Validate automated games
        if not self.config.automated_games:
            errors.append("No automated games found")
        
        return errors
    
    def apply_config_to_detector(self) -> List[GameSession]:
        """
        Apply the configuration to create final game sessions.
        
        Returns:
            List of final GameSession objects
        """
        if not self.config:
            raise ValueError("No configuration loaded")
        
        if not self.detector:
            raise ValueError("No detector available")
        
        print("🔧 Applying configuration to create final games...")
        
        # Create manual splitter
        from manual_game_splitter import ManualGameSplitter
        splitter = ManualGameSplitter(self.detector)
        
        # Add manual splits from configuration
        for split in self.config.manual_splits:
            splitter.add_manual_split(split.timestamp)
            print(f"   Added split at {split.timestamp}: {split.description}")
        
        # Apply splits
        final_games = splitter.apply_manual_splits()
        
        print(f"✅ Configuration applied: {len(final_games)} final games created")
        return final_games
    
    def print_config_summary(self):
        """Print a summary of the loaded configuration."""
        if not self.config:
            print("❌ No configuration loaded")
            return
        
        print(f"\n📊 Configuration Summary")
        print(f"=" * 50)
        print(f"Created: {self.config.created_at}")
        print(f"Created by: {self.config.created_by}")
        print(f"Source: {self.config.source_directory}")
        print(f"Pattern: {self.config.file_pattern}")
        print(f"Total photos: {self.config.total_photos}")
        print(f"Automated games: {len(self.config.automated_games)}")
        print(f"Manual splits: {len(self.config.manual_splits)}")
        
        if self.config.automated_games:
            print(f"\n🤖 Automated Games:")
            for game in self.config.automated_games:
                print(f"   Game {game['game_id']}: {game['start_time']} - {game['end_time']} "
                      f"({game['duration_minutes']} min, {game['photo_count']} photos)")
        
        if self.config.manual_splits:
            print(f"\n✋ Manual Splits:")
            for i, split in enumerate(self.config.manual_splits, 1):
                print(f"   {i}. {split.timestamp} - {split.description} ({split.confidence} confidence)")
                print(f"      Reason: {split.reason}")
        
        if self.config.notes:
            print(f"\n📝 Notes:")
            print(f"   {self.config.notes}")


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Game split configuration manager")
    parser.add_argument("--input", "-i", type=Path, help="Input directory")
    parser.add_argument("--pattern", "-p", default="20250920_*", help="File pattern")
    parser.add_argument("--generate", "-g", action="store_true", help="Generate configuration file")
    parser.add_argument("--load", "-l", type=Path, help="Load configuration file")
    parser.add_argument("--save", "-s", type=Path, help="Save configuration file")
    parser.add_argument("--validate", "-v", action="store_true", help="Validate configuration")
    parser.add_argument("--apply", "-a", action="store_true", help="Apply configuration")
    parser.add_argument("--summary", action="store_true", help="Print configuration summary")
    parser.add_argument("--notes", help="Notes for generated configuration")
    parser.add_argument("--created-by", default="User", help="Creator name")
    
    args = parser.parse_args()
    
    # Create detector
    config = GameDetectionConfig()
    detector = GameDetector(config)
    
    # Create manager
    manager = GameSplitConfigManager(detector)
    
    try:
        if args.generate:
            if not args.input:
                print("❌ --input required for --generate")
                return 1
            
            # Generate configuration
            config = manager.generate_config_from_detection(
                input_dir=args.input,
                pattern=args.pattern,
                notes=args.notes or "",
                created_by=args.created_by
            )
            
            # Save if requested
            if args.save:
                manager.save_config(args.save)
            else:
                # Save to default filename
                default_file = Path(f"game_split_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
                manager.save_config(default_file)
        
        elif args.load:
            # Load configuration
            manager.load_config(args.load)
        
        else:
            print("❌ Must specify --generate or --load")
            return 1
        
        # Additional operations
        if args.validate:
            errors = manager.validate_config()
            if errors:
                print("❌ Validation errors:")
                for error in errors:
                    print(f"   {error}")
            else:
                print("✅ Configuration is valid")
        
        if args.summary:
            manager.print_config_summary()
        
        if args.apply:
            final_games = manager.apply_config_to_detector()
            print(f"\n📊 Final Results: {len(final_games)} games")
            for game in final_games:
                duration = (game.end_time - game.start_time).total_seconds() / 60
                print(f"   Game {game.game_id}: {game.start_time.strftime('%H:%M:%S')} - {game.end_time.strftime('%H:%M:%S')} ({duration:.1f} min, {game.photo_count} photos)")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
