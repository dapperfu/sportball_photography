"""
Sidecar File Management

A comprehensive class for managing sidecar files associated with images,
including symlink resolution, operation detection, and read/write operations.

Author: Claude Sonnet 4 (claude-3-5-sonnet-20241022)
Generated via Cursor IDE (cursor.sh) with AI assistance
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class OperationType(Enum):
    """Supported operation types for sidecar files."""
    FACE_DETECTION = "face_detection"
    OBJECT_DETECTION = "object_detection"
    BALL_DETECTION = "ball_detection"
    QUALITY_ASSESSMENT = "quality_assessment"
    GAME_DETECTION = "game_detection"
    YOLOV8 = "yolov8"
    UNKNOWN = "unknown"


class SidecarInfo:
    """Information about a sidecar file and its associated image."""
    
    def __init__(self, image_path: Path, sidecar_path: Path, 
                 operation: OperationType, symlink_info: Optional[Dict] = None):
        self.image_path = image_path
        self.sidecar_path = sidecar_path
        self.operation = operation
        self.symlink_info = symlink_info or {}
        self.data: Optional[Dict] = None
        self._loaded = False
    
    def load(self) -> Dict[str, Any]:
        """Load sidecar data from file."""
        if not self._loaded:
            try:
                with open(self.sidecar_path, 'r') as f:
                    self.data = json.load(f)
                self._loaded = True
            except Exception as e:
                logger.error(f"Failed to load sidecar {self.sidecar_path}: {e}")
                self.data = {}
        return self.data or {}
    
    def save(self, data: Dict[str, Any]) -> bool:
        """Save data to sidecar file."""
        try:
            # Ensure directory exists
            self.sidecar_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Convert data to JSON-serializable format
            serializable_data = self._make_serializable(data)
            
            with open(self.sidecar_path, 'w') as f:
                json.dump(serializable_data, f, indent=2)
            
            self.data = serializable_data
            self._loaded = True
            return True
        except Exception as e:
            logger.error(f"Failed to save sidecar {self.sidecar_path}: {e}")
            return False
    
    def _make_serializable(self, obj):
        """Convert object to JSON-serializable format."""
        if hasattr(obj, 'as_dict'):
            return obj.as_dict()
        elif hasattr(obj, 'to_dict'):  # fallback for older convention
            return obj.to_dict()
        elif hasattr(obj, '__dict__'):  # fallback to instance dict
            return {key: self._make_serializable(value) for key, value in obj.__dict__.items()}
        elif isinstance(obj, dict):
            return {key: self._make_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_serializable(item) for item in obj]
        elif hasattr(obj, 'tolist'):  # numpy arrays
            return obj.tolist()
        else:
            # Try dataclasses.asdict as last resort
            try:
                from dataclasses import asdict
                if hasattr(obj, '__dataclass_fields__'):
                    return asdict(obj)
            except (ImportError, TypeError):
                pass
            return obj
    
    def get_processing_time(self) -> Optional[float]:
        """Extract processing time from sidecar data."""
        data = self.load()
        
        # Check various possible locations for processing time
        if 'data' in data and 'processing_time' in data['data']:
            return data['data']['processing_time']
        
        # Check metadata sections
        for key in data:
            if isinstance(data[key], dict) and 'metadata' in data[key]:
                metadata = data[key]['metadata']
                if 'processing_time' in metadata:
                    return metadata['processing_time']
                if 'extraction_timestamp' in metadata:
                    # Calculate processing time from timestamp if available
                    try:
                        timestamp = datetime.fromisoformat(metadata['extraction_timestamp'])
                        return (datetime.now() - timestamp).total_seconds()
                    except Exception:
                        pass
        
        return None
    
    def get_success_status(self) -> bool:
        """Extract success status from sidecar data."""
        data = self.load()
        
        # Check various possible locations for success status
        if 'data' in data and 'success' in data['data']:
            return data['data']['success']
        
        # Check metadata sections
        for key in data:
            if isinstance(data[key], dict) and 'metadata' in data[key]:
                metadata = data[key]['metadata']
                if 'success' in metadata:
                    return metadata['success']
        
        # Default to True if no explicit success/failure indicator
        return True
    
    def get_data_size(self) -> int:
        """Get the size of the sidecar data."""
        data = self.load()
        return len(json.dumps(data))


class Sidecar:
    """
    Comprehensive sidecar file management class.
    
    Handles symlink resolution, operation detection, and read/write operations
    for sidecar files associated with images.
    """
    
    # Supported image extensions
    IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.tiff', '.bmp', '.webp']
    
    # Operation type mapping from sidecar file keys
    OPERATION_MAPPING = {
        'Face_detector': OperationType.FACE_DETECTION,
        'Object_detector': OperationType.OBJECT_DETECTION,
        'Ball_detector': OperationType.BALL_DETECTION,
        'Quality_assessor': OperationType.QUALITY_ASSESSMENT,
        'Game_detector': OperationType.GAME_DETECTION,
        'yolov8': OperationType.YOLOV8,
    }
    
    def __init__(self, base_directory: Optional[Path] = None):
        """
        Initialize Sidecar manager.
        
        Args:
            base_directory: Base directory for operations (optional)
        """
        self.base_directory = base_directory
        self._cache: Dict[str, SidecarInfo] = {}
    
    def find_sidecar_for_image(self, image_path: Path) -> Optional[SidecarInfo]:
        """
        Find sidecar file for a given image path.
        
        Handles both regular files and symlinks by resolving symlinks
        and looking for sidecar files next to the target file.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            SidecarInfo object if sidecar found, None otherwise
        """
        if not image_path.exists():
            return None
        
        # Resolve symlink if needed
        if image_path.is_symlink():
            try:
                target_path = image_path.resolve()
                symlink_info = {
                    'symlink_path': str(image_path),
                    'target_path': str(target_path),
                    'is_symlink': True,
                    'broken': not target_path.exists()
                }
                actual_image_path = target_path
            except Exception as e:
                logger.error(f"Failed to resolve symlink {image_path}: {e}")
                return None
        else:
            actual_image_path = image_path
            symlink_info = {
                'symlink_path': str(image_path),
                'target_path': str(image_path),
                'is_symlink': False
            }
        
        # Look for sidecar file next to the actual image
        sidecar_path = actual_image_path.with_suffix('.json')
        
        if sidecar_path.exists():
            operation = self._detect_operation_type(sidecar_path)
            return SidecarInfo(
                image_path=image_path,
                sidecar_path=sidecar_path,
                operation=operation,
                symlink_info=symlink_info
            )
        
        return None
    
    def find_all_sidecars(self, directory: Path) -> List[SidecarInfo]:
        """
        Find all sidecar files in a directory.
        
        Args:
            directory: Directory to search for images and sidecars
            
        Returns:
            List of SidecarInfo objects
        """
        sidecars = []
        
        # Find all image files
        image_files = []
        for ext in self.IMAGE_EXTENSIONS:
            image_files.extend(directory.glob(f'*{ext}'))
            image_files.extend(directory.glob(f'*{ext.upper()}'))
        
        # Process each image file
        for image_file in image_files:
            sidecar_info = self.find_sidecar_for_image(image_file)
            if sidecar_info:
                sidecars.append(sidecar_info)
        
        # Also look for pattern-based sidecars
        pattern_sidecars = []
        pattern_sidecars.extend(directory.glob('*_*.json'))
        
        for sidecar_path in pattern_sidecars:
            # Try to find corresponding image
            image_name = sidecar_path.stem.rsplit('_', 1)[0]
            image_path = None
            
            for ext in self.IMAGE_EXTENSIONS:
                potential_image = directory / f"{image_name}{ext}"
                if potential_image.exists():
                    image_path = potential_image
                    break
            
            if image_path:
                operation = self._detect_operation_type(sidecar_path)
                sidecar_info = SidecarInfo(
                    image_path=image_path,
                    sidecar_path=sidecar_path,
                    operation=operation
                )
                sidecars.append(sidecar_info)
        
        return sidecars
    
    def create_sidecar(self, image_path: Path, operation: OperationType, 
                      data: Dict[str, Any]) -> Optional[SidecarInfo]:
        """
        Create a new sidecar file for an image.
        
        Args:
            image_path: Path to the image file
            operation: Type of operation
            data: Data to store in sidecar
            
        Returns:
            SidecarInfo object if successful, None otherwise
        """
        # Resolve symlink if needed
        if image_path.is_symlink():
            try:
                target_path = image_path.resolve()
                symlink_info = {
                    'symlink_path': str(image_path),
                    'target_path': str(target_path),
                    'is_symlink': True,
                    'broken': not target_path.exists()
                }
                actual_image_path = target_path
            except Exception as e:
                logger.error(f"Failed to resolve symlink {image_path}: {e}")
                return None
        else:
            actual_image_path = image_path
            symlink_info = {
                'symlink_path': str(image_path),
                'target_path': str(image_path),
                'is_symlink': False
            }
        
        # Create sidecar path next to actual image
        sidecar_path = actual_image_path.with_suffix('.json')
        
        # Add metadata to data
        enhanced_data = {
            'sidecar_info': {
                'operation_type': operation.value,
                'created_at': datetime.now().isoformat(),
                'image_path': str(actual_image_path),
                'symlink_path': str(image_path),
                'symlink_info': symlink_info
            },
            'data': data
        }
        
        sidecar_info = SidecarInfo(
            image_path=image_path,
            sidecar_path=sidecar_path,
            operation=operation,
            symlink_info=symlink_info
        )
        
        if sidecar_info.save(enhanced_data):
            return sidecar_info
        
        return None
    
    def get_statistics(self, directory: Path, 
                      operation_filter: Optional[OperationType] = None) -> Dict[str, Any]:
        """
        Get comprehensive statistics about sidecar files in a directory.
        
        Args:
            directory: Directory to analyze
            operation_filter: Optional filter by operation type
            
        Returns:
            Dictionary with comprehensive statistics
        """
        sidecars = self.find_all_sidecars(directory)
        
        # Apply filter if specified
        if operation_filter:
            sidecars = [s for s in sidecars if s.operation == operation_filter]
        
        # Count images (including symlinks)
        image_files = []
        symlink_count = 0
        broken_symlinks = 0
        
        for ext in self.IMAGE_EXTENSIONS:
            image_files.extend(directory.glob(f'*{ext}'))
            image_files.extend(directory.glob(f'*{ext.upper()}'))
        
        for image_file in image_files:
            if image_file.is_symlink():
                symlink_count += 1
                try:
                    if not image_file.resolve().exists():
                        broken_symlinks += 1
                except Exception:
                    broken_symlinks += 1
        
        # Analyze sidecars
        operation_counts = {}
        processing_times = {}
        success_rates = {}
        data_sizes = {}
        
        for sidecar in sidecars:
            operation = sidecar.operation.value
            
            # Count operations
            operation_counts[operation] = operation_counts.get(operation, 0) + 1
            
            # Collect processing times
            proc_time = sidecar.get_processing_time()
            if proc_time is not None:
                if operation not in processing_times:
                    processing_times[operation] = []
                processing_times[operation].append(proc_time)
            
            # Collect success rates
            success = sidecar.get_success_status()
            if operation not in success_rates:
                success_rates[operation] = {'success': 0, 'total': 0}
            success_rates[operation]['total'] += 1
            if success:
                success_rates[operation]['success'] += 1
            
            # Collect data sizes
            data_size = sidecar.get_data_size()
            if operation not in data_sizes:
                data_sizes[operation] = []
            data_sizes[operation].append(data_size)
        
        # Calculate averages
        avg_processing_times = {}
        for operation, times in processing_times.items():
            if times:
                avg_processing_times[operation] = sum(times) / len(times)
        
        success_rate_percentages = {}
        for operation, rates in success_rates.items():
            if rates['total'] > 0:
                success_rate_percentages[operation] = (rates['success'] / rates['total']) * 100
        
        avg_data_sizes = {}
        for operation, sizes in data_sizes.items():
            if sizes:
                avg_data_sizes[operation] = sum(sizes) / len(sizes)
        
        total_images = len(image_files)
        total_sidecars = len(sidecars)
        coverage_percentage = (total_sidecars / total_images * 100) if total_images > 0 else 0
        
        return {
            'directory': str(directory),
            'total_images': total_images,
            'symlink_count': symlink_count,
            'broken_symlinks': broken_symlinks,
            'total_sidecars': total_sidecars,
            'coverage_percentage': coverage_percentage,
            'operation_counts': operation_counts,
            'avg_processing_times': avg_processing_times,
            'success_rate_percentages': success_rate_percentages,
            'avg_data_sizes': avg_data_sizes,
            'filter_applied': operation_filter.value if operation_filter else None,
            'sidecars': [{
                'image_path': str(s.image_path),
                'sidecar_path': str(s.sidecar_path),
                'operation': s.operation.value,
                'symlink_info': s.symlink_info
            } for s in sidecars]
        }
    
    def _detect_operation_type(self, sidecar_path: Path) -> OperationType:
        """
        Detect operation type from sidecar file content.
        
        Args:
            sidecar_path: Path to sidecar file
            
        Returns:
            OperationType enum value
        """
        try:
            with open(sidecar_path, 'r') as f:
                data = json.load(f)
            
            # Check for sidecar_info structure
            if 'sidecar_info' in data:
                operation_str = data['sidecar_info'].get('operation_type', 'unknown')
                try:
                    return OperationType(operation_str)
                except ValueError:
                    pass
            
            # Check for detector-specific keys
            for key, operation_type in self.OPERATION_MAPPING.items():
                if key in data:
                    return operation_type
            
            return OperationType.UNKNOWN
            
        except Exception as e:
            logger.error(f"Failed to detect operation type for {sidecar_path}: {e}")
            return OperationType.UNKNOWN
    
    def _cleanup_orphaned_sidecars_impl(self, directory: Path) -> int:
        """
        Remove sidecar files that don't have corresponding image files.
        
        Args:
            directory: Directory to clean up
            
        Returns:
            Number of orphaned sidecar files removed
        """
        removed_count = 0
        
        # Find all sidecar files
        sidecar_files = []
        sidecar_files.extend(directory.glob('*.json'))
        sidecar_files.extend(directory.glob('*_*.json'))
        
        for sidecar_path in sidecar_files:
            # Check if corresponding image exists
            image_name = sidecar_path.stem.rsplit('_', 1)[0]
            image_exists = False
            
            for ext in self.IMAGE_EXTENSIONS:
                potential_image = directory / f"{image_name}{ext}"
                if potential_image.exists():
                    image_exists = True
                    break
            
            if not image_exists:
                try:
                    sidecar_path.unlink()
                    removed_count += 1
                    logger.info(f"Removed orphaned sidecar: {sidecar_path}")
                except Exception as e:
                    logger.error(f"Failed to remove orphaned sidecar {sidecar_path}: {e}")
        
        return removed_count
    
    # Backward compatibility methods for core module
    def load_data(self, image_path: Path, operation_type: str) -> Optional[Dict[str, Any]]:
        """Load data from sidecar file for backward compatibility."""
        sidecar_info = self.find_sidecar_for_image(image_path)
        if sidecar_info:
            data = sidecar_info.load()
            # Check if this sidecar matches the requested operation
            if sidecar_info.operation.value == operation_type:
                return data.get('data', data)
        return None
    
    def save_data(self, image_path: Path, operation_type: str, data: Dict[str, Any], 
                  metadata: Optional[Dict] = None) -> bool:
        """Save data to sidecar file for backward compatibility."""
        try:
            operation = OperationType(operation_type)
        except ValueError:
            operation = OperationType.UNKNOWN
        
        sidecar_info = self.create_sidecar(image_path, operation, data)
        return sidecar_info is not None
    
    def get_operation_summary(self, directory: Path) -> Dict[str, Any]:
        """Get operation summary for backward compatibility."""
        return self.get_statistics(directory)
    
    def clear_cache(self):
        """Clear internal cache for backward compatibility."""
        self._cache.clear()
    
    def cleanup_orphaned_sidecars(self, directory: Path) -> int:
        """Clean up orphaned sidecar files for backward compatibility."""
        return self._cleanup_orphaned_sidecars_impl(directory)


# Backward compatibility alias
SidecarManager = Sidecar