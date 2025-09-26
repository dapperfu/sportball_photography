"""
Pythonic Decorators for Sportball

This module provides decorators for common operations like GPU acceleration,
parallel processing, progress tracking, and result caching.

Author: Claude Sonnet 4 (claude-3-5-sonnet-20241022)
Generated via Cursor IDE (cursor.sh) with AI assistance
"""

import functools
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union
import logging
from loguru import logger

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False


def gpu_accelerated(device: Optional[str] = None, fallback_cpu: bool = True):
    """
    Decorator to automatically handle GPU acceleration for functions.
    
    Args:
        device: Specific GPU device to use (e.g., 'cuda:0', 'cuda:1')
        fallback_cpu: Whether to fall back to CPU if GPU is not available
        
    Example:
        @gpu_accelerated(device='cuda:0')
        def detect_faces(image):
            # Function will automatically use GPU if available
            pass
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not TORCH_AVAILABLE:
                logger.warning("PyTorch not available, running on CPU")
                return func(*args, **kwargs)
            
            # Determine device
            if device:
                target_device = device
            else:
                target_device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
            # Set device context
            original_device = None
            if hasattr(func, '__self__') and hasattr(func.__self__, 'device'):
                original_device = func.__self__.device
                func.__self__.device = target_device
            
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                if fallback_cpu and target_device != 'cpu':
                    logger.warning(f"GPU operation failed, falling back to CPU: {e}")
                    target_device = 'cpu'
                    if hasattr(func, '__self__') and hasattr(func.__self__, 'device'):
                        func.__self__.device = target_device
                    return func(*args, **kwargs)
                else:
                    raise
            finally:
                # Restore original device
                if original_device is not None and hasattr(func, '__self__'):
                    func.__self__.device = original_device
        
        return wrapper
    return decorator


def parallel_processing(max_workers: Optional[int] = None, 
                      use_threads: bool = True,
                      chunk_size: int = 1):
    """
    Decorator to automatically parallelize processing of iterables.
    
    Args:
        max_workers: Maximum number of workers (defaults to CPU count)
        use_threads: Use ThreadPoolExecutor (True) or ProcessPoolExecutor (False)
        chunk_size: Number of items to process per worker
        
    Example:
        @parallel_processing(max_workers=4)
        def process_images(image_paths):
            # Function will automatically parallelize processing
            pass
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Check if first argument is iterable
            if not args or not hasattr(args[0], '__iter__'):
                return func(*args, **kwargs)
            
            items = args[0]
            if not items:
                return []
            
            # Determine number of workers
            if max_workers is None:
                import os
                max_workers = os.cpu_count() or 1
            
            # Choose executor type
            executor_class = ThreadPoolExecutor if use_threads else ProcessPoolExecutor
            
            results = []
            with executor_class(max_workers=max_workers) as executor:
                # Submit tasks in chunks
                futures = []
                for i in range(0, len(items), chunk_size):
                    chunk = items[i:i + chunk_size]
                    future = executor.submit(func, chunk, *args[1:], **kwargs)
                    futures.append(future)
                
                # Collect results
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        if isinstance(result, list):
                            results.extend(result)
                        else:
                            results.append(result)
                    except Exception as e:
                        logger.error(f"Parallel processing error: {e}")
                        raise
            
            return results
        
        return wrapper
    return decorator


def progress_tracked(description: str = "Processing", 
                    unit: str = "items",
                    show_eta: bool = True,
                    show_rate: bool = True,
                    verbose: bool = True):
    """
    Decorator to add progress tracking to functions that process iterables.
    
    Args:
        description: Description for the progress bar
        unit: Unit of measurement
        show_eta: Show estimated time remaining
        show_rate: Show processing rate
        verbose: Whether to show progress bar (False to suppress when not in verbose mode)
        
    Example:
        @progress_tracked(description="Detecting faces", unit="images")
        def detect_faces_in_images(images):
            # Progress bar will automatically track progress
            pass
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not TQDM_AVAILABLE:
                logger.warning("tqdm not available, running without progress tracking")
                return func(*args, **kwargs)
            
            # Check if first argument is iterable
            if not args or not hasattr(args[0], '__iter__'):
                return func(*args, **kwargs)
            
            items = args[0]
            if not items:
                return []
            
            # Check if we should show progress bar based on verbose setting
            # If verbose=False, check if we're in a non-verbose context
            show_progress = verbose
            if not verbose:
                # Check if we're in a verbose context by looking for verbose in kwargs or context
                show_progress = kwargs.get('verbose', False)
                # Also check if we're in a CLI context with verbose enabled
                import os
                if os.getenv('SPORTBALL_VERBOSE', '').lower() in ('true', '1', 'yes'):
                    show_progress = True
            
            # Create progress bar
            with tqdm(total=len(items), 
                    desc=description, 
                    unit=unit,
                    disable=not (TQDM_AVAILABLE and show_progress)) as pbar:
                
                # Wrap the function to update progress
                def progress_wrapper(*inner_args, **inner_kwargs):
                    result = func(*inner_args, **inner_kwargs)
                    pbar.update(1)
                    return result
                
                # Process items
                results = []
                for item in items:
                    result = progress_wrapper(item, *args[1:], **kwargs)
                    results.append(result)
                
                return results
        
        return wrapper
    return decorator


def cached_result(cache_dir: Optional[Union[str, Path]] = None,
                 cache_key_func: Optional[Callable] = None,
                 expire_seconds: Optional[int] = None):
    """
    Decorator to cache function results based on input parameters.
    
    Args:
        cache_dir: Directory to store cache files
        cache_key_func: Function to generate cache key from arguments
        expire_seconds: Cache expiration time in seconds
        
    Example:
        @cached_result(cache_dir="./cache", expire_seconds=3600)
        def expensive_computation(data):
            # Result will be cached for 1 hour
            pass
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            import hashlib
            import pickle
            import os
            
            # Determine cache directory
            if cache_dir:
                cache_path = Path(cache_dir)
            else:
                cache_path = Path.cwd() / ".sportball_cache"
            
            cache_path.mkdir(exist_ok=True)
            
            # Generate cache key
            if cache_key_func:
                cache_key = cache_key_func(*args, **kwargs)
            else:
                # Default: hash of function name and arguments
                key_data = f"{func.__name__}:{str(args)}:{str(sorted(kwargs.items()))}"
                cache_key = hashlib.md5(key_data.encode()).hexdigest()
            
            cache_file = cache_path / f"{cache_key}.pkl"
            
            # Check if cache exists and is valid
            if cache_file.exists():
                if expire_seconds:
                    file_age = time.time() - cache_file.stat().st_mtime
                    if file_age > expire_seconds:
                        cache_file.unlink()  # Remove expired cache
                    else:
                        # Load from cache
                        try:
                            with open(cache_file, 'rb') as f:
                                cached_result = pickle.load(f)
                            logger.debug(f"Cache hit for {func.__name__}")
                            return cached_result
                        except Exception as e:
                            logger.warning(f"Failed to load cache: {e}")
                            cache_file.unlink()  # Remove corrupted cache
            
            # Compute result
            result = func(*args, **kwargs)
            
            # Save to cache
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump(result, f)
                logger.debug(f"Cached result for {func.__name__}")
            except Exception as e:
                logger.warning(f"Failed to save cache: {e}")
            
            return result
        
        return wrapper
    return decorator


def retry_on_failure(max_retries: int = 3, 
                    delay: float = 1.0,
                    backoff_factor: float = 2.0,
                    exceptions: tuple = (Exception,)):
    """
    Decorator to retry function execution on failure.
    
    Args:
        max_retries: Maximum number of retry attempts
        delay: Initial delay between retries in seconds
        backoff_factor: Multiplier for delay after each retry
        exceptions: Tuple of exception types to retry on
        
    Example:
        @retry_on_failure(max_retries=3, delay=1.0)
        def unreliable_operation():
            # Will retry up to 3 times on failure
            pass
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            current_delay = delay
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_retries:
                        logger.error(f"Function {func.__name__} failed after {max_retries} retries: {e}")
                        raise
                    
                    logger.warning(f"Attempt {attempt + 1} failed for {func.__name__}: {e}")
                    if attempt < max_retries:
                        time.sleep(current_delay)
                        current_delay *= backoff_factor
            
            return None  # Should never reach here
        
        return wrapper
    return decorator


def timing_decorator(func: Callable) -> Callable:
    """
    Decorator to measure and log function execution time.
    
    Example:
        @timing_decorator
        def slow_function():
            # Execution time will be logged
            pass
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        execution_time = end_time - start_time
        logger.info(f"{func.__name__} executed in {execution_time:.3f} seconds")
        
        return result
    
    return wrapper


def validate_inputs(**validators):
    """
    Decorator to validate function inputs.
    
    Args:
        **validators: Dictionary mapping parameter names to validation functions
        
    Example:
        @validate_inputs(
            image_path=lambda x: Path(x).exists(),
            confidence=lambda x: 0 <= x <= 1
        )
        def process_image(image_path, confidence):
            pass
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get function signature
            import inspect
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            
            # Validate each parameter
            for param_name, validator in validators.items():
                if param_name in bound_args.arguments:
                    value = bound_args.arguments[param_name]
                    if not validator(value):
                        raise ValueError(f"Invalid value for {param_name}: {value}")
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator
