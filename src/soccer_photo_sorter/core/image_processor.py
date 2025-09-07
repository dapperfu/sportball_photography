"""
Image processing core module.

This module provides the main image processing functionality
for soccer photo sorting operations.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
from PIL import Image
import torch
from loguru import logger

from ..utils.cuda_utils import CudaManager
from ..utils.file_utils import ImageValidator
from ..config.settings import Settings


class ImageProcessor:
    """Main image processor for soccer photo operations."""
    
    def __init__(self, settings: Settings, cuda_manager: Optional[CudaManager] = None):
        """
        Initialize image processor.
        
        Args:
            settings: Configuration settings
            cuda_manager: Optional CUDA manager for GPU acceleration
        """
        self.settings = settings
        self.cuda_manager = cuda_manager or CudaManager(
            memory_limit_gb=settings.processing.gpu_memory_limit
        )
        self.validator = ImageValidator(settings.detection.supported_formats)
        
        # Initialize OpenCV with CUDA if available
        self._opencv_cuda_available = self.cuda_manager.check_opencv_cuda()
        if self._opencv_cuda_available:
            logger.info("OpenCV CUDA support available")
        
        # Set processing parameters
        self.max_size = settings.detection.max_image_size
        self.min_size = settings.detection.min_image_size
    
    def load_image(self, image_path: Path) -> Optional[np.ndarray]:
        """
        Load image from file.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Image array or None if loading failed
        """
        if not self.validator.is_valid_image_file(image_path):
            logger.error(f"Invalid image file: {image_path}")
            return None
        
        try:
            # Load with OpenCV
            image = cv2.imread(str(image_path))
            if image is None:
                logger.error(f"Failed to load image: {image_path}")
                return None
            
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            logger.debug(f"Loaded image: {image_path} ({image.shape})")
            return image
            
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            return None
    
    def resize_image(self, 
                    image: np.ndarray, 
                    max_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        Resize image while maintaining aspect ratio.
        
        Args:
            image: Input image array
            max_size: Maximum size (width, height)
            
        Returns:
            Resized image array
        """
        if max_size is None:
            max_size = self.max_size
        
        height, width = image.shape[:2]
        max_width, max_height = max_size
        
        # Calculate scaling factor
        scale_w = max_width / width
        scale_h = max_height / height
        scale = min(scale_w, scale_h, 1.0)  # Don't upscale
        
        if scale < 1.0:
            new_width = int(width * scale)
            new_height = int(height * scale)
            
            # Use CUDA resize if available
            if self._opencv_cuda_available:
                try:
                    gpu_image = cv2.cuda_GpuMat()
                    gpu_image.upload(image)
                    gpu_resized = cv2.cuda.resize(gpu_image, (new_width, new_height))
                    resized = gpu_resized.download()
                    return resized
                except Exception:
                    pass  # Fall back to CPU
            
            # CPU resize
            resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
            return resized
        
        return image
    
    def preprocess_image(self, 
                        image: np.ndarray,
                        blur_kernel_size: Optional[int] = None) -> np.ndarray:
        """
        Preprocess image for analysis.
        
        Args:
            image: Input image array
            blur_kernel_size: Optional blur kernel size
            
        Returns:
            Preprocessed image array
        """
        if blur_kernel_size is None:
            blur_kernel_size = self.settings.color.blur_kernel_size
        
        # Apply Gaussian blur to reduce noise
        if blur_kernel_size > 1:
            if self._opencv_cuda_available:
                try:
                    gpu_image = cv2.cuda_GpuMat()
                    gpu_image.upload(image)
                    gpu_blurred = cv2.cuda.GaussianBlur(
                        gpu_image, 
                        (blur_kernel_size, blur_kernel_size), 
                        0
                    )
                    blurred = gpu_blurred.download()
                    return blurred
                except Exception:
                    pass  # Fall back to CPU
            
            # CPU blur
            blurred = cv2.GaussianBlur(image, (blur_kernel_size, blur_kernel_size), 0)
            return blurred
        
        return image
    
    def extract_color_histogram(self, 
                              image: np.ndarray,
                              bins: int = 256) -> Dict[str, np.ndarray]:
        """
        Extract color histogram from image.
        
        Args:
            image: Input image array
            bins: Number of histogram bins
            
        Returns:
            Dictionary with color channel histograms
        """
        histograms = {}
        
        # Extract histograms for each color channel
        for i, color in enumerate(['red', 'green', 'blue']):
            channel = image[:, :, i]
            hist = cv2.calcHist([channel], [0], None, [bins], [0, 256])
            histograms[color] = hist.flatten()
        
        return histograms
    
    def detect_dominant_colors(self, 
                             image: np.ndarray,
                             num_colors: int = 5) -> List[Tuple[int, int, int]]:
        """
        Detect dominant colors in image.
        
        Args:
            image: Input image array
            num_colors: Number of dominant colors to detect
            
        Returns:
            List of dominant colors as (R, G, B) tuples
        """
        # Reshape image to be a list of pixels
        pixels = image.reshape(-1, 3)
        
        # Convert to float32 for k-means
        pixels = np.float32(pixels)
        
        # Apply k-means clustering
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        _, labels, centers = cv2.kmeans(
            pixels, num_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
        )
        
        # Convert centers back to integers
        centers = np.uint8(centers)
        
        # Sort by frequency
        unique_labels, counts = np.unique(labels, return_counts=True)
        sorted_indices = np.argsort(counts)[::-1]
        
        dominant_colors = []
        for idx in sorted_indices:
            color = tuple(centers[unique_labels[idx]])
            dominant_colors.append(color)
        
        return dominant_colors
    
    def classify_color(self, 
                      color: Tuple[int, int, int],
                      color_config: Dict[str, Dict[str, Any]]) -> Optional[str]:
        """
        Classify color based on configuration.
        
        Args:
            color: Color tuple (R, G, B)
            color_config: Color configuration dictionary
            
        Returns:
            Color category name or None if no match
        """
        r, g, b = color
        best_match = None
        best_distance = float('inf')
        
        for color_name, config in color_config.items():
            rgb_range = config['rgb_range']
            tolerance = config['tolerance']
            
            # Calculate distance
            distance = np.sqrt(
                (r - rgb_range[0])**2 + 
                (g - rgb_range[1])**2 + 
                (b - rgb_range[2])**2
            )
            
            if distance <= tolerance and distance < best_distance:
                best_distance = distance
                best_match = color_name
        
        return best_match
    
    def extract_jersey_regions(self, image: np.ndarray) -> List[np.ndarray]:
        """
        Extract potential jersey regions from image.
        
        Args:
            image: Input image array
            
        Returns:
            List of jersey region arrays
        """
        # Convert to HSV for better color segmentation
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Define color ranges for common jersey colors
        color_ranges = [
            # Red
            ([0, 50, 50], [10, 255, 255]),
            ([170, 50, 50], [180, 255, 255]),
            # Blue
            ([100, 50, 50], [130, 255, 255]),
            # Green
            ([40, 50, 50], [80, 255, 255]),
            # Yellow
            ([20, 50, 50], [40, 255, 255]),
        ]
        
        jersey_regions = []
        
        for lower, upper in color_ranges:
            # Create mask
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter contours by area
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 1000:  # Minimum area threshold
                    # Extract region
                    x, y, w, h = cv2.boundingRect(contour)
                    region = image[y:y+h, x:x+w]
                    jersey_regions.append(region)
        
        return jersey_regions
    
    def enhance_for_ocr(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance image for OCR processing.
        
        Args:
            image: Input image array
            
        Returns:
            Enhanced image array
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Apply adaptive thresholding
        enhanced = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        # Apply morphological operations to clean up
        kernel = np.ones((2, 2), np.uint8)
        enhanced = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, kernel)
        
        return enhanced
    
    def get_image_metadata(self, image_path: Path) -> Dict[str, Any]:
        """
        Get comprehensive image metadata.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Image metadata dictionary
        """
        metadata = self.validator.get_image_info(image_path)
        if not metadata:
            return {}
        
        # Add additional metadata
        metadata.update({
            'file_path': image_path,
            'file_name': image_path.name,
            'file_stem': image_path.stem,
            'file_suffix': image_path.suffix,
            'parent_dir': image_path.parent.name,
        })
        
        return metadata
    
    def batch_process_images(self, 
                           image_paths: List[Path],
                           processor_func,
                           batch_size: Optional[int] = None) -> List[Any]:
        """
        Process images in batches.
        
        Args:
            image_paths: List of image file paths
            processor_func: Function to process each image
            batch_size: Optional batch size
            
        Returns:
            List of processing results
        """
        if batch_size is None:
            batch_size = self.settings.processing.batch_size
        
        results = []
        
        for i in range(0, len(image_paths), batch_size):
            batch = image_paths[i:i + batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}: {len(batch)} images")
            
            batch_results = []
            for image_path in batch:
                try:
                    result = processor_func(image_path)
                    batch_results.append(result)
                except Exception as e:
                    logger.error(f"Error processing {image_path}: {e}")
                    batch_results.append(None)
            
            results.extend(batch_results)
        
        return results
    
    def __repr__(self) -> str:
        """String representation of image processor."""
        cuda_status = "CUDA" if self.cuda_manager.is_available else "CPU"
        opencv_cuda = "OpenCV-CUDA" if self._opencv_cuda_available else "OpenCV-CPU"
        
        return f"ImageProcessor(device={cuda_status}, opencv={opencv_cuda})"
