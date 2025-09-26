"""
Parallel JSON Validator

Massively parallel JSON validation for detection tools.

Author: Claude Sonnet 4 (claude-3-5-sonnet-20241022)
Generated via Cursor IDE (cursor.sh) with AI assistance
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count
from dataclasses import dataclass
from loguru import logger

from .base import DetectionResult


@dataclass
class ValidationResult:
    """Result of JSON validation operation."""
    file_path: Path
    is_valid: bool
    error: Optional[str] = None
    processing_time: float = 0.0
    file_size: int = 0
    detection_count: int = 0
    tool_name: Optional[str] = None
    
    def as_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'file_path': str(self.file_path),
            'is_valid': self.is_valid,
            'error': self.error,
            'processing_time': self.processing_time,
            'file_size': self.file_size,
            'detection_count': self.detection_count,
            'tool_name': self.tool_name
        }


class ParallelJSONValidator:
    """
    Massively parallel JSON validator for detection tools.
    
    This provides high-performance JSON validation and analysis
    for sidecar files and detection results.
    """
    
    def __init__(self, max_workers: Optional[int] = None, use_processes: bool = True):
        """
        Initialize parallel JSON validator.
        
        Args:
            max_workers: Maximum number of parallel workers
            use_processes: Whether to use ProcessPoolExecutor (True) or ThreadPoolExecutor (False)
        """
        self.max_workers = max_workers or min(cpu_count() * 2, 64)  # Cap at 64 to avoid overhead
        self.use_processes = use_processes
        self.logger = logger.bind(component="parallel_json_validator")
        
        # Validation statistics
        self.stats = {
            'total_files': 0,
            'valid_files': 0,
            'invalid_files': 0,
            'total_processing_time': 0.0,
            'total_file_size': 0
        }
    
    def validate_json_file(self, file_path: Path, 
                          validation_rules: Optional[List[Callable]] = None) -> ValidationResult:
        """
        Validate a single JSON file.
        
        Args:
            file_path: Path to the JSON file
            validation_rules: Optional list of validation functions
            
        Returns:
            ValidationResult object
        """
        start_time = time.time()
        
        try:
            # Check if file exists
            if not file_path.exists():
                return ValidationResult(
                    file_path=file_path,
                    is_valid=False,
                    error="File does not exist",
                    processing_time=time.time() - start_time
                )
            
            # Get file size
            file_size = file_path.stat().st_size
            
            # Try to parse JSON
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
            except json.JSONDecodeError as e:
                return ValidationResult(
                    file_path=file_path,
                    is_valid=False,
                    error=f"JSON decode error: {e}",
                    processing_time=time.time() - start_time,
                    file_size=file_size
                )
            except Exception as e:
                return ValidationResult(
                    file_path=file_path,
                    is_valid=False,
                    error=f"File read error: {e}",
                    processing_time=time.time() - start_time,
                    file_size=file_size
                )
            
            # Apply custom validation rules
            if validation_rules:
                for rule in validation_rules:
                    try:
                        if not rule(data):
                            return ValidationResult(
                                file_path=file_path,
                                is_valid=False,
                                error=f"Validation rule failed: {rule.__name__}",
                                processing_time=time.time() - start_time,
                                file_size=file_size
                            )
                    except Exception as e:
                        return ValidationResult(
                            file_path=file_path,
                            is_valid=False,
                            error=f"Validation rule error: {e}",
                            processing_time=time.time() - start_time,
                            file_size=file_size
                        )
            
            # Extract detection information
            detection_count = self._extract_detection_count(data)
            tool_name = self._extract_tool_name(data)
            
            return ValidationResult(
                file_path=file_path,
                is_valid=True,
                processing_time=time.time() - start_time,
                file_size=file_size,
                detection_count=detection_count,
                tool_name=tool_name
            )
            
        except Exception as e:
            return ValidationResult(
                file_path=file_path,
                is_valid=False,
                error=f"Unexpected error: {e}",
                processing_time=time.time() - start_time
            )
    
    def validate_json_files_parallel(self, file_paths: List[Path],
                                   validation_rules: Optional[List[Callable]] = None,
                                   show_progress: bool = True) -> List[ValidationResult]:
        """
        Validate multiple JSON files in parallel.
        
        Args:
            file_paths: List of JSON file paths
            validation_rules: Optional list of validation functions
            show_progress: Whether to show progress bar
            
        Returns:
            List of ValidationResult objects
        """
        if not file_paths:
            return []
        
        self.logger.info(f"Validating {len(file_paths)} JSON files with {self.max_workers} workers")
        
        # Choose executor based on configuration
        executor_class = ProcessPoolExecutor if self.use_processes else ThreadPoolExecutor
        
        results = []
        start_time = time.time()
        
        # Import tqdm for progress bar
        if show_progress:
            try:
                from tqdm import tqdm
            except ImportError:
                show_progress = False
        
        with executor_class(max_workers=self.max_workers) as executor:
            # Submit all validation tasks
            future_to_path = {
                executor.submit(self.validate_json_file, file_path, validation_rules): file_path
                for file_path in file_paths
            }
            
            # Collect results as they complete
            if show_progress:
                progress_bar = tqdm(
                    total=len(file_paths),
                    desc="Validating JSON files",
                    unit="files"
                )
            
            for future in as_completed(future_to_path):
                try:
                    result = future.result()
                    results.append(result)
                    
                    # Update statistics
                    self.stats['total_files'] += 1
                    self.stats['total_processing_time'] += result.processing_time
                    self.stats['total_file_size'] += result.file_size
                    
                    if result.is_valid:
                        self.stats['valid_files'] += 1
                    else:
                        self.stats['invalid_files'] += 1
                    
                except Exception as e:
                    # Handle task failure
                    file_path = future_to_path[future]
                    error_result = ValidationResult(
                        file_path=file_path,
                        is_valid=False,
                        error=f"Task failed: {e}",
                        processing_time=0.0
                    )
                    results.append(error_result)
                    self.stats['invalid_files'] += 1
                
                if show_progress:
                    progress_bar.update(1)
            
            if show_progress:
                progress_bar.close()
        
        total_time = time.time() - start_time
        self.logger.info(f"Validation completed in {total_time:.2f}s ({len(file_paths)/total_time:.1f} files/sec)")
        
        return results
    
    def validate_sidecar_files(self, directory: Path, 
                             operation_type: Optional[str] = None,
                             show_progress: bool = True) -> List[ValidationResult]:
        """
        Validate all sidecar files in a directory.
        
        Args:
            directory: Directory to search for sidecar files
            operation_type: Optional operation type filter
            show_progress: Whether to show progress bar
            
        Returns:
            List of ValidationResult objects
        """
        # Find all JSON files
        json_files = list(directory.glob('*.json'))
        
        if operation_type:
            # Filter by operation type
            filtered_files = []
            for json_file in json_files:
                try:
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                    
                    # Check if this file contains the specified operation type
                    if self._contains_operation_type(data, operation_type):
                        filtered_files.append(json_file)
                        
                except Exception:
                    # If we can't read the file, include it for validation
                    filtered_files.append(json_file)
            
            json_files = filtered_files
        
        return self.validate_json_files_parallel(json_files, show_progress=show_progress)
    
    def validate_detection_results(self, results: Dict[str, DetectionResult],
                                 show_progress: bool = True) -> List[ValidationResult]:
        """
        Validate detection results in parallel.
        
        Args:
            results: Dictionary of detection results
            show_progress: Whether to show progress bar
            
        Returns:
            List of ValidationResult objects
        """
        validation_results = []
        
        if show_progress:
            try:
                from tqdm import tqdm
                progress_bar = tqdm(total=len(results), desc="Validating detection results", unit="results")
            except ImportError:
                progress_bar = None
        else:
            progress_bar = None
        
        for image_path, result in results.items():
            try:
                # Validate the detection result
                is_valid = self._validate_detection_result(result)
                
                validation_result = ValidationResult(
                    file_path=Path(image_path),
                    is_valid=is_valid,
                    processing_time=result.processing_time,
                    detection_count=result.get_detection_count(),
                    tool_name=result.tool_name
                )
                
                validation_results.append(validation_result)
                
            except Exception as e:
                validation_result = ValidationResult(
                    file_path=Path(image_path),
                    is_valid=False,
                    error=f"Validation error: {e}",
                    processing_time=0.0
                )
                validation_results.append(validation_result)
            
            if progress_bar:
                progress_bar.update(1)
        
        if progress_bar:
            progress_bar.close()
        
        return validation_results
    
    def get_validation_statistics(self) -> Dict[str, Any]:
        """
        Get validation statistics.
        
        Returns:
            Dictionary with validation statistics
        """
        if self.stats['total_files'] == 0:
            return self.stats.copy()
        
        return {
            **self.stats,
            'valid_percentage': (self.stats['valid_files'] / self.stats['total_files']) * 100,
            'invalid_percentage': (self.stats['invalid_files'] / self.stats['total_files']) * 100,
            'avg_processing_time': self.stats['total_processing_time'] / self.stats['total_files'],
            'avg_file_size': self.stats['total_file_size'] / self.stats['total_files']
        }
    
    def reset_statistics(self) -> None:
        """Reset validation statistics."""
        self.stats = {
            'total_files': 0,
            'valid_files': 0,
            'invalid_files': 0,
            'total_processing_time': 0.0,
            'total_file_size': 0
        }
    
    def _extract_detection_count(self, data: Dict[str, Any]) -> int:
        """Extract detection count from JSON data."""
        # Try common detection count fields
        if 'count' in data:
            return int(data['count'])
        elif 'faces' in data and isinstance(data['faces'], list):
            return len(data['faces'])
        elif 'objects' in data and isinstance(data['objects'], list):
            return len(data['objects'])
        elif 'detections' in data and isinstance(data['detections'], list):
            return len(data['detections'])
        
        # Check nested structures
        for key in ['data', 'result', 'detection']:
            if key in data and isinstance(data[key], dict):
                nested_count = self._extract_detection_count(data[key])
                if nested_count > 0:
                    return nested_count
        
        return 0
    
    def _extract_tool_name(self, data: Dict[str, Any]) -> Optional[str]:
        """Extract tool name from JSON data."""
        # Try common tool name fields
        for key in ['tool_name', 'detector', 'model', 'algorithm']:
            if key in data:
                return str(data[key])
        
        # Check nested structures
        for key in ['data', 'result', 'metadata']:
            if key in data and isinstance(data[key], dict):
                nested_tool = self._extract_tool_name(data[key])
                if nested_tool:
                    return nested_tool
        
        return None
    
    def _contains_operation_type(self, data: Dict[str, Any], operation_type: str) -> bool:
        """Check if JSON data contains a specific operation type."""
        # Check direct keys
        if operation_type in data:
            return True
        
        # Check sidecar_info structure
        if 'sidecar_info' in data and isinstance(data['sidecar_info'], dict):
            if data['sidecar_info'].get('operation_type') == operation_type:
                return True
        
        # Check nested structures
        for key in ['data', 'result']:
            if key in data and isinstance(data[key], dict):
                if self._contains_operation_type(data[key], operation_type):
                    return True
        
        return False
    
    def _validate_detection_result(self, result: DetectionResult) -> bool:
        """Validate a DetectionResult object."""
        try:
            # Basic validation
            if not isinstance(result, DetectionResult):
                return False
            
            if not isinstance(result.success, bool):
                return False
            
            if not isinstance(result.data, dict):
                return False
            
            # Check for required fields
            if result.success and not result.data:
                return False
            
            return True
            
        except Exception:
            return False
