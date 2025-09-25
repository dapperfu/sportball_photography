#!/usr/bin/env python3
"""
Demo script showing command-line usage of the configuration system.

This script demonstrates the various command-line options available.
"""

import subprocess
import sys
from pathlib import Path

def run_command(cmd, description):
    """Run a command and show the description."""
    print(f"\n🔧 {description}")
    print(f"Command: {cmd}")
    print("-" * 60)
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(result.stdout)
        else:
            print(f"Error: {result.stderr}")
    except Exception as e:
        print(f"Exception: {e}")

def main():
    """Demonstrate command-line usage."""
    
    print("📋 Configuration System Command-Line Demo")
    print("=" * 60)
    print("This demonstrates the various command-line options available")
    print("for the game split configuration system.")
    
    # Base command
    base_cmd = "python game_split_config.py"
    input_dir = "/keg/pictures/incoming/2025/09-Sep"
    
    # 1. Generate configuration
    cmd1 = f"{base_cmd} --input {input_dir} --generate --save config_generated.json --notes 'Generated config for demo'"
    run_command(cmd1, "Generate configuration from automated detection")
    
    # 2. Load and show summary
    cmd2 = f"{base_cmd} --load config_generated.json --summary"
    run_command(cmd2, "Load configuration and show summary")
    
    # 3. Validate configuration
    cmd3 = f"{base_cmd} --load config_generated.json --validate"
    run_command(cmd3, "Validate configuration")
    
    # 4. Apply configuration
    cmd4 = f"{base_cmd} --load config_generated.json --apply"
    run_command(cmd4, "Apply configuration to create final games")
    
    print(f"\n💡 Additional Commands:")
    print(f"   # Generate with custom pattern:")
    print(f"   {base_cmd} --input {input_dir} --pattern '20250920_*' --generate --save my_config.json")
    print(f"   ")
    print(f"   # Load and apply in one command:")
    print(f"   {base_cmd} --load my_config.json --apply --summary")
    print(f"   ")
    print(f"   # Generate with custom creator:")
    print(f"   {base_cmd} --input {input_dir} --generate --created-by 'John Doe' --save config.json")
    
    print(f"\n📝 Configuration File Format:")
    print(f"   The generated JSON file contains:")
    print(f"   - Metadata (created_at, created_by, source_directory)")
    print(f"   - Detection configuration parameters")
    print(f"   - Automated games detected")
    print(f"   - Manual splits (empty initially)")
    print(f"   - Notes and version information")
    
    print(f"\n✏️  Manual Editing:")
    print(f"   You can edit the JSON file to add manual splits:")
    print(f"   - Add entries to the 'manual_splits' array")
    print(f"   - Each split needs: timestamp, description, reason, confidence")
    print(f"   - Timestamp format: 'HH:MM:SS' (e.g., '14:00:00')")
    print(f"   - Confidence: 'high', 'medium', or 'low'")

if __name__ == "__main__":
    main()
