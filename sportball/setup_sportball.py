#!/usr/bin/env python3
"""
Setup script for Sportball package.

This script sets up the sportball package for development and installation.

Author: Claude Sonnet 4 (claude-3-5-sonnet-20241022)
Generated via Cursor IDE (cursor.sh) with AI assistance
"""

import subprocess
import sys
from pathlib import Path


def run_command(cmd: str, description: str) -> bool:
    """Run a command and return success status."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        if e.stdout:
            print(f"STDOUT: {e.stdout}")
        if e.stderr:
            print(f"STDERR: {e.stderr}")
        return False


def main():
    """Main setup function."""
    print("🚀 Setting up Sportball package...")
    
    # Check if we're in the right directory
    if not Path("sportball").exists():
        print("❌ sportball directory not found. Please run this script from the sportball project root.")
        return 1
    
    # Install the package in development mode
    if not run_command("pip install -e .", "Installing sportball package"):
        return 1
    
    # Install development dependencies
    if not run_command("pip install -e .[dev]", "Installing development dependencies"):
        print("⚠️  Development dependencies installation failed, but package is installed")
    
    # Run basic tests
    if not run_command("python -c \"import sportball; print('Sportball imported successfully')\"", "Testing package import"):
        return 1
    
    # Test CLI
    if not run_command("sportball --help", "Testing CLI"):
        return 1
    
    print("\n🎉 Sportball setup completed successfully!")
    print("\n📋 Next steps:")
    print("1. Test the CLI: sportball --help")
    print("2. Try face detection: sportball face detect /path/to/images")
    print("3. Check documentation: sportball --help")
    print("4. Run tests: pytest")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
