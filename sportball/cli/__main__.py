"""
Minimal CLI entry point to avoid heavy imports at startup.

This module provides a lightweight entry point that only imports
the CLI when needed, avoiding the heavy dependencies in the main
sportball package.
"""


def main():
    """Main entry point for the CLI."""
    from .main import cli

    cli()


if __name__ == "__main__":
    main()
