"""
Rust Sidecar Integration

Python wrapper for the high-performance Rust sidecar implementation.
ALL sidecar operations require Rust and will raise RuntimeError if unavailable.

Author: Claude Sonnet 4 (claude-3-5-sonnet-20241022)
Generated via Cursor IDE (cursor.sh) with AI assistance
"""

import json
import subprocess
import tempfile
import os
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from loguru import logger

# Try to import the Python package (built via maturin)
try:
    import image_sidecar_rust
    PYTHON_PACKAGE_AVAILABLE = True
except ImportError:
    PYTHON_PACKAGE_AVAILABLE = False
    image_sidecar_rust = None



@dataclass
class RustSidecarConfig:
    """Configuration for Rust sidecar integration."""

    enable_rust: bool = True
    rust_binary_path: Optional[Path] = None
    max_workers: int = 16
    timeout: int = 300


class RustSidecarManager:
    """
    Python wrapper for the Rust sidecar implementation.

    Provides high-performance sidecar operations. Rust is REQUIRED.
    Raises RuntimeError if Rust is unavailable.
    """

    def __init__(self, config: Optional[RustSidecarConfig] = None):
        """
        Initialize Rust sidecar manager.

        Args:
            config: Configuration for the manager
        """
        self.config = config or RustSidecarConfig()
        self.logger = logger.bind(component="rust_sidecar")
        self.rust_available = False

        # Check if Rust is available
        self._check_rust_availability()

    def _check_rust_availability(self) -> None:
        """Check if Rust Python package or binary is available."""
        if not self.config.enable_rust:
            self.rust_available = False
            return

        # First, try to use the Python package (preferred)
        if PYTHON_PACKAGE_AVAILABLE:
            self.rust_available = True
            self.logger.info("Rust Python package is available")
            return

        # Fall back to binary subprocess approach
        # Check for Rust binary
        if self.config.rust_binary_path and self.config.rust_binary_path.exists():
            self.rust_available = True
            self.logger.info(f"Rust binary found: {self.config.rust_binary_path}")
            return

        # Try to find Rust binary in PATH
        try:
            result = subprocess.run(
                ["which", "sportball-sidecar-rust"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                self.config.rust_binary_path = Path(result.stdout.strip())
                self.rust_available = True
                self.logger.info(
                    f"Rust binary found in PATH: {self.config.rust_binary_path}"
                )
                return
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

        # Try to find binary in parent directory
        parent_dir = Path(__file__).parent.parent.parent.parent.parent
        rust_binary = (
            parent_dir
            / "image-sidecar-rust"
            / "target"
            / "release"
            / "sportball-sidecar-rust"
        )

        # Also try relative path from current working directory
        if not rust_binary.exists():
            rust_binary = Path(
                "../image-sidecar-rust/target/release/sportball-sidecar-rust"
            )
        if rust_binary.exists():
            self.config.rust_binary_path = rust_binary
            self.rust_available = True
            self.logger.info(
                f"Rust binary found in parent directory: {self.config.rust_binary_path}"
            )
            return

        # Rust not available
        self.rust_available = False
        self.logger.error("Rust not available. Sidecar operations require Rust.")
        raise RuntimeError(
            "Rust implementation not available. Sidecar operations require "
            "the image-sidecar-rust module. Please ensure Rust tools are installed."
        )

    def validate_sidecars(
        self, directory: Path, operation_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Validate sidecar files in a directory.

        Args:
            directory: Directory containing sidecar files
            operation_filter: Optional operation type filter

        Returns:
            List of validation results
        """
        if not self.rust_available:
            raise RuntimeError("Rust not available. Sidecar operations require Rust.")

        try:
            # Create temporary file with directory path
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".txt", delete=False
            ) as f:
                f.write(f"{directory}\n")
                temp_file = f.name

            try:
                # Run Rust binary
                cmd = [
                    str(self.config.rust_binary_path),
                    "validate",
                    "--input",
                    str(directory),
                    "--workers",
                    str(self.config.max_workers),
                ]

                if operation_filter:
                    cmd.extend(["--operation-type", operation_filter])

                result = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=self.config.timeout
                )

                if result.returncode != 0:
                    raise RuntimeError(f"Rust validation failed: {result.stderr}")

                # Parse results
                output_data = json.loads(result.stdout)
                return output_data.get("results", [])

            finally:
                # Clean up temporary file
                os.unlink(temp_file)

        except Exception as e:
            self.logger.error(f"Rust sidecar validation failed: {e}")
            raise

    def get_statistics(
        self, directory: Path, operation_filter: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get comprehensive statistics about sidecar files.

        Args:
            directory: Directory containing sidecar files
            operation_filter: Optional operation type filter

        Returns:
            Dictionary with statistics
        """
        if not self.rust_available:
            raise RuntimeError("Rust not available. Sidecar operations require Rust.")

        try:
            # Run Rust binary
            cmd = [
                str(self.config.rust_binary_path),
                "stats",
                "--input",
                str(directory),
            ]

            if operation_filter:
                cmd.extend(["--operation-type", operation_filter])

            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=self.config.timeout
            )

            if result.returncode != 0:
                raise RuntimeError(f"Rust statistics failed: {result.stderr}")

            # Parse results
            return json.loads(result.stdout)

        except Exception as e:
            self.logger.error(f"Rust sidecar statistics failed: {e}")
            raise

    def cleanup_orphaned_sidecars(self, directory: Path, dry_run: bool = False) -> int:
        """
        Clean up orphaned sidecar files.

        Args:
            directory: Directory to clean up
            dry_run: If True, only show what would be cleaned

        Returns:
            Number of orphaned files removed
        """
        if not self.rust_available:
            raise RuntimeError("Rust not available. Sidecar operations require Rust.")

        try:
            # Run Rust binary
            cmd = [
                str(self.config.rust_binary_path),
                "cleanup",
                "--input",
                str(directory),
            ]

            if dry_run:
                cmd.append("--dry-run")

            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=self.config.timeout
            )

            if result.returncode != 0:
                raise RuntimeError(f"Rust cleanup failed: {result.stderr}")

            # Parse output to get count
            output = result.stdout.strip()
            if "Removed" in output:
                # Extract number from "Removed X orphaned sidecar files"
                import re

                match = re.search(r"Removed (\d+)", output)
                if match:
                    return int(match.group(1))

            return 0

        except Exception as e:
            self.logger.error(f"Rust sidecar cleanup failed: {e}")
            raise

    def export_sidecar_data(
        self,
        directory: Path,
        output_path: Path,
        operation_filter: Optional[str] = None,
        format: str = "json",
    ) -> int:
        """
        Export sidecar data to file.

        Args:
            directory: Directory containing sidecar files
            output_path: Output file path
            operation_filter: Optional operation type filter
            format: Export format (json, csv)

        Returns:
            Number of sidecar files exported
        """
        if not self.rust_available:
            raise RuntimeError("Rust not available. Sidecar operations require Rust.")

        try:
            # Run Rust binary
            cmd = [
                str(self.config.rust_binary_path),
                "export",
                "--input",
                str(directory),
                "--output",
                str(output_path),
                "--format",
                format,
            ]

            if operation_filter:
                cmd.extend(["--operation-type", operation_filter])

            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=self.config.timeout
            )

            if result.returncode != 0:
                raise RuntimeError(f"Rust export failed: {result.stderr}")

            # Parse output to get count
            output = result.stdout.strip()
            if "Exported" in output:
                # Extract number from "Exported X sidecar files"
                import re

                match = re.search(r"Exported (\d+)", output)
                if match:
                    return int(match.group(1))

            return 0

        except Exception as e:
            self.logger.error(f"Rust sidecar export failed: {e}")
            raise

    def get_performance_info(self) -> Dict[str, Any]:
        """
        Get performance information about the Rust integration.

        Returns:
            Dictionary with performance information
        """
        return {
            "rust_available": self.rust_available,
            "rust_binary_path": str(self.config.rust_binary_path)
            if self.config.rust_binary_path
            else None,
            "max_workers": self.config.max_workers,
            "timeout": self.config.timeout,
        }

    def save_sidecar_data(
        self, 
        image_path: Path, 
        operation_type: str, 
        data: Dict[str, Any]
    ) -> bool:
        """
        Save sidecar data using Rust binary format.
        
        Args:
            image_path: Path to the image file
            operation_type: Type of operation (e.g., "yolov8", "face_detection")
            data: Data to save in the sidecar
            
        Returns:
            True if successful, False otherwise
            
        Raises:
            RuntimeError: If Rust is not available
        """
        if not self.rust_available:
            raise RuntimeError("Rust not available. Sidecar operations require Rust.")
        
        try:
            # Create a temporary JSON file with the data
            import tempfile
            import os
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
                import json
                json.dump(data, tmp_file, indent=2)
                temp_path = tmp_file.name
            
            try:
                # Get the target path
                base_path = image_path.parent
                
                # Use Rust binary to convert JSON to binary format
                cmd = [
                    str(self.config.rust_binary_path),
                    "write",
                    "--input", temp_path,
                    "--output", str(image_path.with_suffix(".bin")),
                    "--operation", operation_type
                ]
                
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                
                if result.returncode == 0:
                    self.logger.debug(f"Saved sidecar using Rust backend: {image_path.with_suffix('.bin')}")
                    return True
                else:
                    self.logger.warning(f"Rust backend failed: {result.stderr}")
                    return False
                    
            finally:
                # Clean up temporary file
                try:
                    os.unlink(temp_path)
                except Exception as e:
                    self.logger.warning(f"Failed to cleanup temp file {temp_path}: {e}")
                    
        except Exception as e:
            self.logger.error(f"save_sidecar_data failed: {e}")
            return False
