#!/usr/bin/env python3
"""
GPU Stress Test for InsightFace

This script processes images in parallel to really stress the GPU.

Author: Claude Sonnet 4 (claude-3-5-sonnet-20241022)
Generated via Cursor IDE (cursor.sh) with AI assistance
"""

import sys
import time
import cv2
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import threading
from queue import Queue
import insightface
import torch

def stress_gpu_insightface(image_dir: Path, max_images: int = 1000, batch_size: int = 32):
    """Stress test InsightFace with GPU acceleration."""
    
    print(f"🚀 Starting GPU stress test on {image_dir}")
    print(f"📊 Max images: {max_images}, Batch size: {batch_size}")
    
    # Initialize InsightFace with GPU
    print("🔧 Initializing InsightFace with GPU...")
    app = insightface.app.FaceAnalysis(
        name='buffalo_l',
        providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
    )
    app.prepare(ctx_id=0, det_size=(640, 640))
    
    print("✅ InsightFace initialized with GPU support")
    
    # Find images
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        image_files.extend(image_dir.glob(ext))
    
    image_files = image_files[:max_images]
    print(f"📁 Found {len(image_files)} images to process")
    
    # Process images in batches
    total_faces = 0
    total_time = 0
    
    for i in range(0, len(image_files), batch_size):
        batch_files = image_files[i:i + batch_size]
        print(f"🔄 Processing batch {i//batch_size + 1}: {len(batch_files)} images")
        
        batch_start = time.time()
        
        # Load all images in batch
        batch_images = []
        batch_paths = []
        
        for img_path in batch_files:
            try:
                image = cv2.imread(str(img_path))
                if image is not None:
                    batch_images.append(image)
                    batch_paths.append(img_path)
            except Exception as e:
                print(f"❌ Failed to load {img_path}: {e}")
        
        if not batch_images:
            continue
        
        # Process batch with InsightFace
        try:
            # Process each image individually but in parallel
            with ThreadPoolExecutor(max_workers=8) as executor:
                futures = []
                for img, path in zip(batch_images, batch_paths):
                    future = executor.submit(process_single_image, app, img, path)
                    futures.append(future)
                
                # Collect results
                batch_faces = 0
                for future in futures:
                    try:
                        faces = future.result(timeout=30)
                        batch_faces += faces
                    except Exception as e:
                        print(f"❌ Processing error: {e}")
                
                total_faces += batch_faces
                
        except Exception as e:
            print(f"❌ Batch processing failed: {e}")
            continue
        
        batch_time = time.time() - batch_start
        total_time += batch_time
        
        print(f"✅ Batch completed: {batch_faces} faces in {batch_time:.2f}s ({batch_time/len(batch_images):.3f}s per image)")
        
        # Check GPU usage
        if torch.cuda.is_available():
            memory_used = torch.cuda.memory_allocated() / 1024**2
            print(f"🎮 GPU Memory: {memory_used:.1f}MB")
    
    print(f"\n🏆 Final Results:")
    print(f"📊 Total images processed: {len(image_files)}")
    print(f"👥 Total faces found: {total_faces}")
    print(f"⏱️  Total time: {total_time:.2f}s")
    print(f"📈 Average time per image: {total_time/len(image_files):.3f}s")
    print(f"🎯 Faces per image: {total_faces/len(image_files):.1f}")

def process_single_image(app, image, path):
    """Process a single image with InsightFace."""
    try:
        faces = app.get(image)
        return len(faces)
    except Exception as e:
        print(f"❌ Error processing {path}: {e}")
        return 0

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python gpu_stress_test.py <image_directory> [max_images] [batch_size]")
        sys.exit(1)
    
    image_dir = Path(sys.argv[1])
    max_images = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    batch_size = int(sys.argv[3]) if len(sys.argv) > 3 else 32
    
    if not image_dir.exists():
        print(f"❌ Directory {image_dir} does not exist")
        sys.exit(1)
    
    stress_gpu_insightface(image_dir, max_images, batch_size)
