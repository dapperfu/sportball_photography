#!/usr/bin/env python3
"""
Simple Phase 1 Test

A simplified test of Phase 1 features without complex dependencies.

Author: Claude Sonnet 4 (claude-3-5-sonnet-20241022)
Generated via Cursor IDE (cursor.sh) with AI assistance
"""

import cv2
import numpy as np
from pathlib import Path
from loguru import logger
import click


def test_ball_detection(image_path: Path) -> int:
    """Simple ball detection test using Hough circles."""
    image = cv2.imread(str(image_path))
    if image is None:
        return 0
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detect circles
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=20,
        param1=50,
        param2=30,
        minRadius=5,
        maxRadius=100
    )
    
    return len(circles[0]) if circles is not None else 0


def test_action_detection(image_path: Path) -> int:
    """Simple action detection test using pose landmarks."""
    try:
        import mediapipe as mp
        
        image = cv2.imread(str(image_path))
        if image is None:
            return 0
        
        # Initialize pose detection
        mp_pose = mp.solutions.pose
        pose_detector = mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            min_detection_confidence=0.5
        )
        
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect pose
        results = pose_detector.process(rgb_image)
        
        return 1 if results.pose_landmarks else 0
        
    except ImportError:
        logger.warning("MediaPipe not available")
        return 0


def test_quality_assessment(image_path: Path) -> float:
    """Simple quality assessment using Laplacian variance."""
    image = cv2.imread(str(image_path))
    if image is None:
        return 0.0
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Calculate sharpness using Laplacian variance
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # Normalize to 0-1 scale
    if laplacian_var > 1000:
        return 1.0
    elif laplacian_var > 500:
        return 0.8
    elif laplacian_var > 200:
        return 0.6
    elif laplacian_var > 100:
        return 0.4
    else:
        return 0.2


@click.command()
@click.argument('input_pattern', type=str)
@click.option('--max-images', '-m', default=3, type=int, help='Maximum number of images to process')
def main(input_pattern: str, max_images: int):
    """Simple Phase 1 feature test."""
    
    # Find images
    if input_pattern.startswith('/'):
        parent_dir = Path(input_pattern).parent
        pattern = Path(input_pattern).name
        image_files = list(parent_dir.glob(pattern))
    else:
        image_files = list(Path('.').glob(input_pattern))
    
    if not image_files:
        logger.error(f"No images found matching pattern: {input_pattern}")
        return
    
    # Limit images
    image_files = image_files[:max_images]
    
    logger.info(f"Testing Phase 1 features on {len(image_files)} images")
    
    total_balls = 0
    total_actions = 0
    total_quality = 0.0
    
    for image_path in image_files:
        logger.info(f"Testing {image_path.name}")
        
        # Test ball detection
        balls = test_ball_detection(image_path)
        total_balls += balls
        logger.info(f"  Balls detected: {balls}")
        
        # Test action detection
        actions = test_action_detection(image_path)
        total_actions += actions
        logger.info(f"  Actions detected: {actions}")
        
        # Test quality assessment
        quality = test_quality_assessment(image_path)
        total_quality += quality
        logger.info(f"  Quality score: {quality:.2f}")
    
    # Summary
    logger.info("=== PHASE 1 TEST SUMMARY ===")
    logger.info(f"Images processed: {len(image_files)}")
    logger.info(f"Total balls detected: {total_balls}")
    logger.info(f"Total actions detected: {total_actions}")
    logger.info(f"Average quality score: {total_quality/len(image_files):.2f}")
    logger.info(f"Average balls per image: {total_balls/len(image_files):.1f}")
    logger.info(f"Average actions per image: {total_actions/len(image_files):.1f}")


if __name__ == '__main__':
    main()
