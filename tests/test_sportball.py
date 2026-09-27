#!/usr/bin/env python3
"""
Test script for Sportball package.

This script tests the basic functionality of the sportball package.

Author: Claude Sonnet 4 (claude-3-5-sonnet-20241022)
Generated via Cursor IDE (cursor.sh) with AI assistance
"""

import sys
from pathlib import Path

# Add the parent directory to the path so we can import sportball
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_imports():
    """Test that all modules can be imported."""
    print("🔄 Testing imports...")

    try:
        import sportball

        print("✅ Main sportball module imported")

        from sportball import SportballCore

        print("✅ Core classes imported")

        from sportball.decorators import (
            parallel_processing,
            progress_tracked,
            timing_decorator,
        )

        print("✅ Decorators imported")

        return True

    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False


def test_core_initialization():
    """Test that SportballCore can be initialized."""
    print("🔄 Testing core initialization...")

    try:
        from sportball import SportballCore

        core = SportballCore(enable_gpu=False, max_workers=2)
        print("✅ SportballCore initialized successfully")

        # Test properties
        assert core.enable_gpu == False
        assert core.max_workers == 2
        assert core.cache_enabled == True
        print("✅ Core properties set correctly")

        return True

    except Exception as e:
        print(f"❌ Core initialization failed: {e}")
        return False


def test_decorators():
    """Test that decorators can be applied."""
    print("🔄 Testing decorators...")

    try:
        from sportball.decorators import (
            timing_decorator,
        )

        # Test decorator application
        @timing_decorator
        def test_function():
            return "test"

        result = test_function()
        assert result == "test"
        print("✅ Decorators can be applied")

        return True

    except Exception as e:
        print(f"❌ Decorator test failed: {e}")
        return False


def test_cli_import():
    """Test that CLI can be imported."""
    print("🔄 Testing CLI import...")

    try:
        from sportball.cli.ultra_minimal_main import cli

        print("✅ CLI module imported")

        # Test that cli is callable
        assert callable(cli)
        print("✅ CLI is callable")

        return True

    except Exception as e:
        print(f"❌ CLI import failed: {e}")
        return False


def main():
    """Main test function."""
    print("🧪 Testing Sportball package...")

    tests = [
        test_imports,
        test_core_initialization,
        test_decorators,
        test_cli_import,
    ]

    passed = 0
    for test in tests:
        if test():
            passed += 1
        print()

    print(f"{passed}/{len(tests)} tests passed")
    return passed == len(tests)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
