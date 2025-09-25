#!/usr/bin/env python3
"""
Face Extraction Tool

This tool reads JSON sidecar files created by face_detection.py and extracts
faces from images to directories. Can create both clean and annotated versions.

Author: Claude Sonnet 4 (claude-3-5-sonnet-20241022)
Generated via Cursor IDE (cursor.sh) with AI assistance
"""

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import click
import cv2
import numpy as np
from tqdm import tqdm
from loguru import logger


@dataclass
class FacialFeature:
    """Information about a detected facial feature."""
    feature_type: str
    x: int
    y: int
    width: int
    height: int
    confidence: float


@dataclass
class DetectedFace:
    """Information about a detected face from JSON."""
    face_id: int
    x: int
    y: int
    width: int
    height: int
    confidence: float
    crop_x: int
    crop_y: int
    crop_width: int
    crop_height: int
    detection_scale_factor: float
    face_encoding: Optional[List[float]]
    facial_features: List[FacialFeature]


@dataclass
class ExtractionResult:
    """Result of face extraction from an image."""
    image_path: str
    faces_extracted: int
    extraction_time: float
    extracted_faces: List[str]  # List of extracted filenames
    error: Optional[str] = None


class FaceExtractor:
    """Face extraction from JSON sidecar data."""
    
    def __init__(self, create_annotated: bool = True):
        """
        Initialize face extractor.
        
        Args:
            create_annotated: Whether to create annotated versions of faces
        """
        self.create_annotated = create_annotated
        logger.info(f"Initialized face extractor (annotated: {create_annotated})")
    
    def annotate_facial_features(self, face_region: np.ndarray, features: List[FacialFeature]) -> np.ndarray:
        """
        Draw annotations for facial features on the face region.
        
        Args:
            face_region: Face image to annotate
            features: List of detected facial features
            
        Returns:
            Annotated face image
        """
        annotated = face_region.copy()
        
        # Color mapping for different features
        colors = {
            'left_eye': (0, 255, 0),    # Green
            'right_eye': (0, 255, 0),   # Green
            'nose': (255, 0, 0),        # Blue
            'mouth': (0, 0, 255)        # Red
        }
        
        for feature in features:
            color = colors.get(feature.feature_type, (255, 255, 255))
            
            # Draw bounding box
            cv2.rectangle(annotated, 
                         (feature.x, feature.y), 
                         (feature.x + feature.width, feature.y + feature.height), 
                         color, 2)
            
            # Draw label
            label = feature.feature_type.replace('_', ' ').title()
            cv2.putText(annotated, label, 
                       (feature.x, feature.y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return annotated
    
    def load_detection_data(self, json_path: Path) -> Optional[Dict]:
        """
        Load face detection data from JSON sidecar file.
        
        Args:
            json_path: Path to the JSON sidecar file
            
        Returns:
            Detection data dictionary or None if failed
        """
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            if "Face_detector" not in data:
                logger.debug(f"No face_detector data in {json_path}")
                return None
            
            return data["Face_detector"]
            
        except Exception as e:
            logger.error(f"Failed to load JSON {json_path}: {e}")
            return None
    
    def extract_faces_from_image(self, image_path: Path, json_path: Path, output_dir: Path) -> ExtractionResult:
        """
        Extract faces from an image using JSON sidecar data.
        
        Args:
            image_path: Path to the original image
            json_path: Path to the JSON sidecar file
            output_dir: Directory to save extracted faces
            
        Returns:
            Extraction result
        """
        logger.info(f"Extracting faces from {image_path.name}")
        
        # Load detection data
        detection_data = self.load_detection_data(json_path)
        if not detection_data:
            return ExtractionResult(
                image_path=str(image_path),
                faces_extracted=0,
                extraction_time=0.0,
                extracted_faces=[],
                error="Failed to load detection data"
            )
        
        # Load original image
        image = cv2.imread(str(image_path))
        if image is None:
            return ExtractionResult(
                image_path=str(image_path),
                faces_extracted=0,
                extraction_time=0.0,
                extracted_faces=[],
                error="Failed to load image"
            )
        
        start_time = cv2.getTickCount()
        extracted_faces = []
        
        # Process each detected face
        for face_data in detection_data["faces"]:
            try:
                # Extract face region using crop coordinates
                crop_x = face_data["crop_area"]["x"]
                crop_y = face_data["crop_area"]["y"]
                crop_w = face_data["crop_area"]["width"]
                crop_h = face_data["crop_area"]["height"]
                
                # Extract face region from original image
                face_region = image[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
                
                if face_region.size == 0:
                    logger.warning(f"Empty face region for face {face_data['face_id']}")
                    continue
                
                # Generate filenames
                face_id = face_data["face_id"]
                face_filename = f"{image_path.stem}_face_{face_id:02d}.jpg"
                annotated_filename = f"{image_path.stem}_face_{face_id:02d}_annotated.jpg"
                face_path = output_dir / face_filename
                annotated_path = output_dir / annotated_filename
                
                # Save clean face
                cv2.imwrite(str(face_path), face_region)
                extracted_faces.append(face_filename)
                
                # Create and save annotated version if requested
                if self.create_annotated:
                    # Convert facial features
                    facial_features = []
                    for feature_data in face_data["facial_features"]:
                        feature = FacialFeature(
                            feature_type=feature_data["feature_type"],
                            x=feature_data["coordinates"]["x"],
                            y=feature_data["coordinates"]["y"],
                            width=feature_data["coordinates"]["width"],
                            height=feature_data["coordinates"]["height"],
                            confidence=feature_data["confidence"]
                        )
                        facial_features.append(feature)
                    
                    # Create annotated version
                    annotated_face = self.annotate_facial_features(face_region, facial_features)
                    cv2.imwrite(str(annotated_path), annotated_face)
                    extracted_faces.append(annotated_filename)
                
                logger.debug(f"Extracted face {face_id}: {face_filename}")
                
            except Exception as e:
                logger.error(f"Error extracting face {face_data.get('face_id', 'unknown')}: {e}")
                continue
        
        extraction_time = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()
        
        return ExtractionResult(
            image_path=str(image_path),
            faces_extracted=len(detection_data["faces"]),
            extraction_time=extraction_time,
            extracted_faces=extracted_faces
        )
    
    def extract_faces_from_directory(self, input_dir: Path, output_dir: Path) -> List[ExtractionResult]:
        """
        Extract faces from all images in a directory that have JSON sidecar files.
        
        Args:
            input_dir: Directory containing images and JSON sidecar files
            output_dir: Directory to save extracted faces
            
        Returns:
            List of extraction results
        """
        # Find all JSON sidecar files
        json_files = list(input_dir.glob("*.json"))
        
        if not json_files:
            logger.error(f"No JSON sidecar files found in {input_dir}")
            return []
        
        logger.info(f"Found {len(json_files)} JSON sidecar files")
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results = []
        
        # Process each JSON file
        for json_path in tqdm(json_files, desc="Extracting faces"):
            # Find corresponding image file
            image_stem = json_path.stem
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
            
            image_path = None
            for ext in image_extensions:
                potential_image = input_dir / f"{image_stem}{ext}"
                if potential_image.exists():
                    image_path = potential_image
                    break
            
            if not image_path:
                logger.warning(f"No image file found for {json_path}")
                continue
            
            # Extract faces
            result = self.extract_faces_from_image(image_path, json_path, output_dir)
            results.append(result)
        
        return results
    
    def create_extraction_summary(self, results: List[ExtractionResult], output_dir: Path) -> None:
        """Create a summary report of the extraction process."""
        total_images = len(results)
        total_faces_extracted = sum(result.faces_extracted for result in results)
        total_time = sum(result.extraction_time for result in results)
        
        # Create summary data
        summary_data = {
            'extraction_summary': {
                'total_images_processed': total_images,
                'total_faces_extracted': total_faces_extracted,
                'total_extraction_time_seconds': total_time,
                'average_time_per_image_seconds': total_time / total_images if total_images > 0 else 0,
                'create_annotated': self.create_annotated
            },
            'image_results': []
        }
        
        # Add per-image results
        for result in results:
            image_data = {
                'image_path': result.image_path,
                'faces_extracted': result.faces_extracted,
                'extraction_time_seconds': result.extraction_time,
                'extracted_files': result.extracted_faces,
                'error': result.error
            }
            summary_data['image_results'].append(image_data)
        
        # Save JSON summary
        json_file = output_dir / "face_extraction_summary.json"
        with open(json_file, 'w') as f:
            json.dump(summary_data, f, indent=2)
        
        # Create markdown summary
        md_file = output_dir / "face_extraction_report.md"
        with open(md_file, 'w') as f:
            f.write("# Face Extraction Report\n\n")
            f.write(f"**Create Annotated**: {self.create_annotated}\n")
            f.write(f"**Images Processed**: {total_images}\n")
            f.write(f"**Faces Extracted**: {total_faces_extracted}\n")
            f.write(f"**Total Extraction Time**: {total_time:.2f}s\n")
            f.write(f"**Average Time per Image**: {total_time/total_images:.2f}s\n\n")
            
            f.write("## Per-Image Results\n\n")
            f.write("| Image | Faces Extracted | Time (s) | Files Created |\n")
            f.write("|-------|-----------------|----------|---------------|\n")
            
            for result in results:
                files_count = len(result.extracted_faces)
                f.write(f"| {Path(result.image_path).name} | {result.faces_extracted} | {result.extraction_time:.3f} | {files_count} |\n")
        
        logger.info(f"Extraction summary saved to {json_file}")
        logger.info(f"Extraction report saved to {md_file}")


@click.command()
@click.argument('input_dir', type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.argument('output_dir', type=click.Path(path_type=Path))
@click.option('--annotated/--no-annotated', default=True, help='Create annotated versions of faces')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
def main(input_dir: Path, output_dir: Path, annotated: bool, verbose: bool):
    """Extract faces from images using JSON sidecar files created by face_detection.py."""
    
    # Setup logging
    if verbose:
        logger.add("face_extraction.log", level="DEBUG")
    
    # Initialize extractor
    extractor = FaceExtractor(create_annotated=annotated)
    
    # Extract faces
    logger.info(f"Starting face extraction (annotated: {annotated})")
    results = extractor.extract_faces_from_directory(input_dir, output_dir)
    
    if not results:
        logger.error("No extractions performed")
        return
    
    # Create summary
    extractor.create_extraction_summary(results, output_dir)
    
    # Calculate summary statistics
    total_images = len(results)
    total_faces_extracted = sum(result.faces_extracted for result in results)
    total_time = sum(result.extraction_time for result in results)
    
    logger.info(f"Face extraction complete!")
    logger.info(f"Processed {total_images} images")
    logger.info(f"Extracted {total_faces_extracted} faces")
    logger.info(f"Total extraction time: {total_time:.2f}s")
    logger.info(f"Average time per image: {total_time/total_images:.2f}s")
    logger.info(f"Faces saved to {output_dir}")


if __name__ == "__main__":
    main()
