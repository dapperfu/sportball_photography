#!/usr/bin/env python3
"""
Configuration File Workflow Demo

This script demonstrates the complete workflow:
1. Generate configuration file from automated detection
2. Edit the configuration file (manual step)
3. Load and apply the configuration
"""

import sys
from pathlib import Path
from game_split_config import GameSplitConfigManager
from game_detector import GameDetector, GameDetectionConfig

def main():
    """Demonstrate the configuration file workflow."""
    
    print("📋 Game Split Configuration Workflow")
    print("=" * 50)
    print("This demonstrates the complete configuration file workflow:")
    print("1. Generate configuration from automated detection")
    print("2. Edit configuration file manually")
    print("3. Load and apply configuration")
    print()
    
    # Configuration
    input_dir = Path("/keg/pictures/incoming/2025/09-Sep")
    pattern = "20250920_*"
    config_file = Path("game_split_config_demo.json")
    
    print(f"Input Directory: {input_dir}")
    print(f"Pattern: {pattern}")
    print(f"Config File: {config_file}")
    print()
    
    # Create detector and manager
    detector_config = GameDetectionConfig()
    detector = GameDetector(detector_config)
    manager = GameSplitConfigManager(detector)
    
    try:
        # Step 1: Generate configuration
        print("🔍 Step 1: Generating configuration from automated detection...")
        config = manager.generate_config_from_detection(
            input_dir=input_dir,
            pattern=pattern,
            notes="Demo configuration for September 20th soccer photos",
            created_by="Demo User"
        )
        
        # Save configuration
        manager.save_config(config_file)
        
        # Show what was generated
        print(f"\n📊 Generated Configuration:")
        manager.print_config_summary()
        
        # Step 2: Simulate manual editing
        print(f"\n✏️  Step 2: Manual editing (simulated)")
        print(f"   Configuration file saved to: {config_file}")
        print(f"   You can now edit this file to add manual splits")
        print(f"   Example manual splits to add:")
        print(f"   - Add split at 14:00:00 for game transition")
        print(f"   - Add split at 15:30:00 for halftime")
        print()
        
        # Simulate adding manual splits
        print("   Simulating manual edits...")
        config.manual_splits = [
            {
                "timestamp": "14:00:00",
                "description": "Game transition - ran from one game to another",
                "reason": "Photographer moved between games without stopping",
                "confidence": "high"
            },
            {
                "timestamp": "15:30:00", 
                "description": "Halftime break",
                "reason": "Clear break in action for halftime",
                "confidence": "medium"
            }
        ]
        
        # Save updated configuration
        manager.save_config(config_file)
        print(f"   Updated configuration saved to: {config_file}")
        
        # Step 3: Load and apply configuration
        print(f"\n📂 Step 3: Loading and applying configuration...")
        manager.load_config(config_file)
        
        # Validate configuration
        errors = manager.validate_config()
        if errors:
            print("❌ Validation errors:")
            for error in errors:
                print(f"   {error}")
        else:
            print("✅ Configuration is valid")
        
        # Show loaded configuration
        print(f"\n📊 Loaded Configuration:")
        manager.print_config_summary()
        
        # Apply configuration
        print(f"\n🔧 Applying configuration...")
        final_games = manager.apply_config_to_detector()
        
        # Show final results
        print(f"\n✅ Final Results:")
        print(f"   Original automated games: {len(detector.games)}")
        print(f"   Manual splits applied: {len(config.manual_splits)}")
        print(f"   Final games created: {len(final_games)}")
        
        print(f"\n📊 Final Games:")
        for game in final_games:
            duration = (game.end_time - game.start_time).total_seconds() / 60
            print(f"   Game {game.game_id}: {game.start_time.strftime('%H:%M:%S')} - {game.end_time.strftime('%H:%M:%S')} ({duration:.1f} min, {game.photo_count} photos)")
        
        print(f"\n💡 Next Steps:")
        print(f"   1. Review the configuration file: {config_file}")
        print(f"   2. Edit it to add/remove manual splits as needed")
        print(f"   3. Run: python game_split_config.py --load {config_file} --apply")
        print(f"   4. Create organized folders with the final games")
        
    except Exception as e:
        print(f"❌ Error during workflow: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
