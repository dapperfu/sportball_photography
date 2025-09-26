#!/usr/bin/env python3
"""
Aggressive GPU Stress Test for InsightFace

This script creates multiple InsightFace instances to really stress the GPU.

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
import insightface
import torch
import multiprocessing as mp

def create_insightface_app():
    """Create a new InsightFace app instance."""
    app = insightface.app.FaceAnalysis(
        name='buffalo_l',
        providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
    )
    app.prepare(ctx_id=0, det_size=(640, 640))
    return app

def process_image_batch(app, image_paths, batch_id):
    """Process a batch of images with a single InsightFace app."""
    print(f"🔄 Batch {batch_id}: Processing {len(image_paths)} images")
    
    batch_faces = 0
    batch_start = time.time()
    
    for i, img_path in enumerate(image_paths):
        try:
            image = cv2.imread(str(img_path))
            if image is not None:
                faces = app.get(image)
                batch_faces += len(faces)
                
                # Force GPU memory allocation
                if torch.cuda.is_available():
                    # Create some GPU tensors to stress the GPU
                    dummy_tensor = torch.randn(1000, 1000).cuda()
                    del dummy_tensor
                    torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"❌ Error processing {img_path}: {e}")
    
    batch_time = time.time() - batch_start
    print(f"✅ Batch {batch_id}: {batch_faces} faces in {batch_time:.2f}s")
    
    return batch_faces, batch_time

def aggressive_gpu_stress(image_dir: Path, max_images: int = 1000, num_processes: int = 4):
    """Aggressive GPU stress test with multiple processes."""
    
    print(f"🚀 Starting AGGRESSIVE GPU stress test on {image_dir}")
    print(f"📊 Max images: {max_images}, Processes: {num_processes}")
    
    # Find images
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        image_files.extend(image_dir.glob(ext))
    
    image_files = image_files[:max_images]
    print(f"📁 Found {len(image_files)} images to process")
    
    # Split images into batches for each process
    batch_size = len(image_files) // num_processes
    batches = []
    for i in range(num_processes):
        start_idx = i * batch_size
        end_idx = start_idx + batch_size if i < num_processes - 1 else len(image_files)
        batches.append(image_files[start_idx:end_idx])
    
    print(f"🔄 Created {len(batches)} batches")
    
    # Process batches in parallel
    total_faces = 0
    total_time = 0
    
    with ThreadPoolExecutor(max_workers=num_processes) as executor:
        futures = []
        for i, batch in enumerate(batches):
            if batch:  # Only process non-empty batches
                app = create_insightface_app()
                future = executor.submit(process_image_batch, app, batch, i)
                futures.append(future)
        
        # Collect results
        for future in futures:
            try:
                faces, batch_time = future.result()
                total_faces += faces
                total_time += batch_time
            except Exception as e:
                print(f"❌ Batch processing error: {e}")
    
    print(f"\n🏆 Final Results:")
    print(f"📊 Total images processed: {len(image_files)}")
    print(f"👥 Total faces found: {total_faces}")
    print(f"⏱️  Total time: {total_time:.2f}s")
    print(f"📈 Average time per image: {total_time/len(image_files):.3f}s")
    print(f"🎯 Faces per image: {total_faces/len(image_files):.1f}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python aggressive_gpu_stress.py <image_directory> [max_images] [num_processes]")
        sys.exit(1)
    
    image_dir = Path(sys.argv[1])
    max_images = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    num_processes = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    
    if not image_dir.exists():
        print(f"❌ Directory {image_dir} does not exist")
        sys.exit(1)
    
    aggressive_gpu_stress(image_dir, max_images, num_processes)
