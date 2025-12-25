"""
CUDA utilities for soccer photo sorter.

This module provides CUDA detection, initialization, and management
for GPU acceleration of image processing operations.
"""

import os
from typing import Optional, List, Dict, Any
from pathlib import Path
from loguru import logger

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available - CUDA support disabled")

try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False
    logger.warning("OpenCV not available - CUDA support disabled")


class CudaManager:
    """Manager for CUDA operations and GPU detection."""
    
    def __init__(self, memory_limit_gb: Optional[int] = None):
        """
        Initialize CUDA manager.
        
        Args:
            memory_limit_gb: Optional GPU memory limit in GB
        """
        self.memory_limit_gb = memory_limit_gb
        self._cuda_available = False
        self._device_count = 0
        self._current_device = None
        self._device_info = {}
        
        self._initialize_cuda()
    
    def _initialize_cuda(self) -> None:
        """Initialize CUDA and detect available devices."""
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available - CUDA support disabled")
            return
        
        if not torch.cuda.is_available():
            logger.info("CUDA not available on this system")
            return
        
        self._cuda_available = True
        self._device_count = torch.cuda.device_count()
        
        # Get device information
        for i in range(self._device_count):
            device_info = {
                'name': torch.cuda.get_device_name(i),
                'memory_total': torch.cuda.get_device_properties(i).total_memory,
                'memory_reserved': 0,
                'memory_allocated': 0,
                'compute_capability': torch.cuda.get_device_properties(i).major,
            }
            self._device_info[i] = device_info
        
        # Set memory limit if specified
        if self.memory_limit_gb:
            self._set_memory_limit()
        
        logger.info(f"CUDA initialized with {self._device_count} device(s)")
        for i, info in self._device_info.items():
            logger.info(f"  Device {i}: {info['name']} ({info['memory_total'] / 1024**3:.1f} GB)")
    
    def _set_memory_limit(self) -> None:
        """Set GPU memory limit."""
        if not self._cuda_available:
            return
        
        memory_limit_bytes = self.memory_limit_gb * 1024**3
        
        for i in range(self._device_count):
            torch.cuda.set_per_process_memory_fraction(
                memory_limit_bytes / self._device_info[i]['memory_total'],
                device=i
            )
        
        logger.info(f"GPU memory limit set to {self.memory_limit_gb} GB per device")
    
    @property
    def is_available(self) -> bool:
        """Check if CUDA is available."""
        return self._cuda_available
    
    @property
    def device_count(self) -> int:
        """Get number of available CUDA devices."""
        return self._device_count
    
    @property
    def current_device(self) -> Optional[int]:
        """Get current CUDA device."""
        return self._current_device
    
    def get_device_info(self, device_id: int = 0) -> Dict[str, Any]:
        """
        Get information about a specific CUDA device.
        
        Args:
            device_id: Device ID
            
        Returns:
            Device information dictionary
        """
        if not self._cuda_available:
            return {}
        
        if device_id >= self._device_count:
            raise ValueError(f"Device {device_id} not available")
        
        info = self._device_info[device_id].copy()
        
        # Update memory usage
        if torch.cuda.is_available():
            info['memory_reserved'] = torch.cuda.memory_reserved(device_id)
            info['memory_allocated'] = torch.cuda.memory_allocated(device_id)
            info['memory_free'] = info['memory_total'] - info['memory_reserved']
        
        return info
    
    def get_all_device_info(self) -> Dict[int, Dict[str, Any]]:
        """
        Get information about all CUDA devices.
        
        Returns:
            Dictionary mapping device IDs to device information
        """
        if not self._cuda_available:
            return {}
        
        info = {}
        for i in range(self._device_count):
            info[i] = self.get_device_info(i)
        
        return info
    
    def set_device(self, device_id: int) -> None:
        """
        Set current CUDA device.
        
        Args:
            device_id: Device ID to set
        """
        if not self._cuda_available:
            raise RuntimeError("CUDA not available")
        
        if device_id >= self._device_count:
            raise ValueError(f"Device {device_id} not available")
        
        torch.cuda.set_device(device_id)
        self._current_device = device_id
        logger.info(f"CUDA device set to {device_id}")
    
    def get_best_device(self) -> int:
        """
        Get the best available CUDA device based on memory.
        
        Returns:
            Device ID of the best device
        """
        if not self._cuda_available:
            raise RuntimeError("CUDA not available")
        
        best_device = 0
        max_memory = 0
        
        for i in range(self._device_count):
            memory = self._device_info[i]['memory_total']
            if memory > max_memory:
                max_memory = memory
                best_device = i
        
        return best_device
    
    def clear_cache(self, device_id: Optional[int] = None) -> None:
        """
        Clear CUDA cache for a device.
        
        Args:
            device_id: Device ID to clear cache for (None for all devices)
        """
        if not self._cuda_available:
            return
        
        if device_id is None:
            torch.cuda.empty_cache()
            logger.info("CUDA cache cleared for all devices")
        else:
            if device_id >= self._device_count:
                raise ValueError(f"Device {device_id} not available")
            
            with torch.cuda.device(device_id):
                torch.cuda.empty_cache()
            logger.info(f"CUDA cache cleared for device {device_id}")
    
    def get_memory_usage(self, device_id: int = 0) -> Dict[str, float]:
        """
        Get memory usage for a CUDA device.
        
        Args:
            device_id: Device ID
            
        Returns:
            Memory usage dictionary with keys: total, reserved, allocated, free
        """
        if not self._cuda_available:
            return {'total': 0, 'reserved': 0, 'allocated': 0, 'free': 0}
        
        if device_id >= self._device_count:
            raise ValueError(f"Device {device_id} not available")
        
        info = self.get_device_info(device_id)
        
        return {
            'total': info['memory_total'] / 1024**3,  # Convert to GB
            'reserved': info['memory_reserved'] / 1024**3,
            'allocated': info['memory_allocated'] / 1024**3,
            'free': info['memory_free'] / 1024**3,
        }
    
    def check_opencv_cuda(self) -> bool:
        """
        Check if OpenCV was compiled with CUDA support.
        
        Returns:
            True if OpenCV CUDA is available
        """
        if not OPENCV_AVAILABLE:
            return False
        
        try:
            return cv2.cuda.getCudaEnabledDeviceCount() > 0
        except Exception:
            return False
    
    def get_opencv_cuda_devices(self) -> List[int]:
        """
        Get list of CUDA devices available to OpenCV.
        
        Returns:
            List of device IDs
        """
        if not self.check_opencv_cuda():
            return []
        
        try:
            device_count = cv2.cuda.getCudaEnabledDeviceCount()
            return list(range(device_count))
        except Exception:
            return []
    
    def optimize_for_inference(self) -> None:
        """Optimize CUDA settings for inference."""
        if not self._cuda_available:
            return
        
        # Enable cuDNN benchmarking for consistent performance
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        
        # Set memory allocation strategy
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
        
        logger.info("CUDA optimized for inference")
    
    def get_system_info(self) -> Dict[str, Any]:
        """
        Get comprehensive CUDA system information.
        
        Returns:
            System information dictionary
        """
        info = {
            'cuda_available': self._cuda_available,
            'device_count': self._device_count,
            'torch_version': torch.__version__ if TORCH_AVAILABLE else None,
            'opencv_cuda_available': self.check_opencv_cuda(),
            'opencv_cuda_devices': self.get_opencv_cuda_devices(),
            'memory_limit_gb': self.memory_limit_gb,
        }
        
        if self._cuda_available:
            info['devices'] = self.get_all_device_info()
            info['current_device'] = self._current_device
        
        return info
    
    def __repr__(self) -> str:
        """String representation of CUDA manager."""
        if not self._cuda_available:
            return "CudaManager(cuda_available=False)"
        
        return f"CudaManager(devices={self._device_count}, current={self._current_device})"
