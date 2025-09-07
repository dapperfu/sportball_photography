"""
Logging utilities for soccer photo sorter.

This module provides centralized logging configuration and utilities
for consistent logging across the application.
"""

import sys
from pathlib import Path
from typing import Optional, Dict, Any
from loguru import logger
import json
from datetime import datetime


def setup_logging(log_level: str = "INFO",
                 log_file: Optional[Path] = None,
                 verbose: bool = False,
                 log_format: Optional[str] = None) -> None:
    """
    Setup logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path
        verbose: Whether to enable verbose output
        log_format: Optional custom log format
    """
    # Remove default handler
    logger.remove()
    
    # Set log level
    level = log_level.upper()
    if verbose and level == "INFO":
        level = "DEBUG"
    
    # Default format
    if log_format is None:
        log_format = (
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        )
    
    # Console handler
    logger.add(
        sys.stderr,
        format=log_format,
        level=level,
        colorize=True,
        backtrace=True,
        diagnose=True,
    )
    
    # File handler (if specified)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            log_file,
            format=log_format,
            level=level,
            rotation="10 MB",
            retention="7 days",
            compression="zip",
            backtrace=True,
            diagnose=True,
        )
    
    logger.info(f"Logging initialized with level: {level}")


def get_logger(name: Optional[str] = None):
    """
    Get logger instance.
    
    Args:
        name: Optional logger name
        
    Returns:
        Logger instance
    """
    if name:
        return logger.bind(name=name)
    return logger


def log_processing_start(operation: str, 
                        input_path: Path, 
                        output_path: Path,
                        file_count: int) -> None:
    """
    Log processing start information.
    
    Args:
        operation: Operation name
        input_path: Input directory path
        output_path: Output directory path
        file_count: Number of files to process
    """
    logger.info(f"Starting {operation}")
    logger.info(f"Input directory: {input_path}")
    logger.info(f"Output directory: {output_path}")
    logger.info(f"Files to process: {file_count}")


def log_processing_progress(current: int, 
                           total: int, 
                           operation: str = "Processing") -> None:
    """
    Log processing progress.
    
    Args:
        current: Current file number
        total: Total files
        operation: Operation name
    """
    percentage = (current / total) * 100 if total > 0 else 0
    logger.info(f"{operation}: {current}/{total} ({percentage:.1f}%)")


def log_processing_complete(operation: str,
                           stats: Dict[str, Any]) -> None:
    """
    Log processing completion.
    
    Args:
        operation: Operation name
        stats: Processing statistics
    """
    logger.info(f"{operation} completed")
    logger.info(f"Statistics: {json.dumps(stats, indent=2)}")


def log_error(error: Exception, 
             context: Optional[str] = None,
             file_path: Optional[Path] = None) -> None:
    """
    Log error with context.
    
    Args:
        error: Exception object
        context: Optional context information
        file_path: Optional file path where error occurred
    """
    message = f"Error: {str(error)}"
    
    if context:
        message = f"{context} - {message}"
    
    if file_path:
        message = f"{file_path} - {message}"
    
    logger.error(message)
    logger.exception(error)


def log_cuda_info(cuda_manager) -> None:
    """
    Log CUDA information.
    
    Args:
        cuda_manager: CudaManager instance
    """
    if not cuda_manager.is_available:
        logger.info("CUDA not available")
        return
    
    logger.info(f"CUDA available with {cuda_manager.device_count} device(s)")
    
    for device_id in range(cuda_manager.device_count):
        info = cuda_manager.get_device_info(device_id)
        logger.info(f"  Device {device_id}: {info['name']}")
        logger.info(f"    Memory: {info['memory_total'] / 1024**3:.1f} GB")
        logger.info(f"    Compute Capability: {info['compute_capability']}")


def log_performance_stats(stats: Dict[str, Any]) -> None:
    """
    Log performance statistics.
    
    Args:
        stats: Performance statistics dictionary
    """
    logger.info("Performance Statistics:")
    
    if 'processing_time' in stats:
        logger.info(f"  Processing time: {stats['processing_time']:.2f} seconds")
    
    if 'files_per_second' in stats:
        logger.info(f"  Processing rate: {stats['files_per_second']:.2f} files/second")
    
    if 'memory_usage' in stats:
        logger.info(f"  Memory usage: {stats['memory_usage']:.2f} MB")
    
    if 'gpu_memory_usage' in stats:
        logger.info(f"  GPU memory usage: {stats['gpu_memory_usage']:.2f} GB")


class ProcessingLogger:
    """Context manager for processing operations."""
    
    def __init__(self, operation: str, file_count: int):
        """
        Initialize processing logger.
        
        Args:
            operation: Operation name
            file_count: Number of files to process
        """
        self.operation = operation
        self.file_count = file_count
        self.start_time = None
        self.processed_count = 0
        self.error_count = 0
    
    def __enter__(self):
        """Enter context."""
        self.start_time = datetime.now()
        logger.info(f"Starting {self.operation} for {self.file_count} files")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context."""
        if exc_type:
            logger.error(f"{self.operation} failed: {exc_val}")
            return
        
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()
        
        logger.info(f"{self.operation} completed")
        logger.info(f"  Files processed: {self.processed_count}/{self.file_count}")
        logger.info(f"  Errors: {self.error_count}")
        logger.info(f"  Duration: {duration:.2f} seconds")
        
        if self.processed_count > 0:
            rate = self.processed_count / duration
            logger.info(f"  Processing rate: {rate:.2f} files/second")
    
    def log_progress(self, current: int) -> None:
        """Log progress."""
        self.processed_count = current
        percentage = (current / self.file_count) * 100 if self.file_count > 0 else 0
        
        if current % 10 == 0 or current == self.file_count:  # Log every 10 files or last file
            logger.info(f"{self.operation}: {current}/{self.file_count} ({percentage:.1f}%)")
    
    def log_error(self, error: Exception, file_path: Optional[Path] = None) -> None:
        """Log error."""
        self.error_count += 1
        log_error(error, self.operation, file_path)
