#!/usr/bin/env python3
"""
Face Extraction Tool with Border Padding

This tool extracts faces from images with configurable border padding
to capture more of the head/face context, not just the detected face region.

Author: Claude Sonnet 4 (claude-3-5-sonnet-20241022)
Generated via Cursor IDE (cursor.sh) with AI assistance
"""

import json
import logging
import os
from dataclasses import dataclass, asdict
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
    feature_type: str  # 'left_eye', 'right_eye', 'nose', 'mouth'
    x: int
    y: int
    width: int
    height: int
    confidence: float


@dataclass
class ExtractedFace:
    """Information about an extracted face."""
    face_id: int
    original_image: str
    x: int
    y: int
    width: int
    height: int
    confidence: float
    extracted_filename: str
    annotated_filename: str
    border_padding: float
    facial_features: List[FacialFeature]


@dataclass
class ExtractionResult:
    """Result of face extraction from an image."""
    image_path: str
    faces_found: int
    faces_extracted: int
    extraction_time: float
    extracted_faces: List[ExtractedFace]
    error: Optional[str] = None


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle numpy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


class FaceExtractor:
    """Face extraction tool with configurable border padding."""
    
    def __init__(self, border_padding: float = 0.25):
        """
        Initialize the face extractor.
        
        Args:
            border_padding: Percentage of padding around detected face (0.25 = 25%)
        """
        self.border_padding = border_padding
        
        # Initialize OpenCV cascades for face and feature detection
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
        # Try to load nose and mouth cascades, but don't fail if they're not available
        # Note: These cascade files may not be available in all OpenCV installations
        self.nose_cascade = None
        self.mouth_cascade = None
        
        # Only try to load if the files exist
        nose_path = cv2.data.haarcascades + 'haarcascade_mcs_nose.xml'
        mouth_path = cv2.data.haarcascades + 'haarcascade_mcs_mouth.xml'
        
        if os.path.exists(nose_path):
            try:
                self.nose_cascade = cv2.CascadeClassifier(nose_path)
                if self.nose_cascade.empty():
                    self.nose_cascade = None
            except:
                self.nose_cascade = None
                
        if os.path.exists(mouth_path):
            try:
                self.mouth_cascade = cv2.CascadeClassifier(mouth_path)
                if self.mouth_cascade.empty():
                    self.mouth_cascade = None
            except:
                self.mouth_cascade = None
        
        logger.info(f"Initialized face extractor with {border_padding*100:.0f}% border padding")
    
    def detect_facial_features(self, face_region: np.ndarray) -> List[FacialFeature]:
        """
        Detect facial features (eyes, nose, mouth) in a face region.
        
        Args:
            face_region: Cropped face image
            
        Returns:
            List of detected facial features
        """
        features = []
        gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
        
        # Detect eyes
        eyes = self.eye_cascade.detectMultiScale(gray_face, 1.1, 3)
        for i, (ex, ey, ew, eh) in enumerate(eyes):
            # Determine if left or right eye based on position
            eye_type = 'left_eye' if ex < face_region.shape[1] // 2 else 'right_eye'
            features.append(FacialFeature(
                feature_type=eye_type,
                x=int(ex), y=int(ey), width=int(ew), height=int(eh),
                confidence=0.8
            ))
        
        # Detect nose (if cascade is available)
        if self.nose_cascade is not None:
            try:
                noses = self.nose_cascade.detectMultiScale(gray_face, 1.1, 3)
                for nx, ny, nw, nh in noses:
                    features.append(FacialFeature(
                        feature_type='nose',
                        x=int(nx), y=int(ny), width=int(nw), height=int(nh),
                        confidence=0.8
                    ))
            except:
                pass
        
        # Detect mouth (if cascade is available)
        if self.mouth_cascade is not None:
            try:
                mouths = self.mouth_cascade.detectMultiScale(gray_face, 1.1, 3)
                for mx, my, mw, mh in mouths:
                    features.append(FacialFeature(
                        feature_type='mouth',
                        x=int(mx), y=int(my), width=int(mw), height=int(mh),
                        confidence=0.8
                    ))
            except:
                pass
        
        return features
    
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
    
    def extract_faces_from_image(self, image_path: Path, output_dir: Path) -> ExtractionResult:
        """
        Extract faces from a single image with border padding.
        
        Args:
            image_path: Path to the input image
            output_dir: Directory to save extracted faces
            
        Returns:
            Extraction result with face information
        """
        logger.info(f"Extracting faces from {image_path.name}")
        
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            return ExtractionResult(
                image_path=str(image_path),
                faces_found=0,
                faces_extracted=0,
                extraction_time=0.0,
                extracted_faces=[],
                error="Failed to load image"
            )
        
        # Detect faces
        start_time = cv2.getTickCount()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        detection_time = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()
        
        extracted_faces = []
        
        # Extract each face with border padding
        for i, (x, y, w, h) in enumerate(faces):
            try:
                # Calculate padded coordinates
                padding_x = int(w * self.border_padding)
                padding_y = int(h * self.border_padding)
                
                # Calculate new coordinates with padding
                new_x = max(0, x - padding_x)
                new_y = max(0, y - padding_y)
                new_w = min(image.shape[1] - new_x, w + 2 * padding_x)
                new_h = min(image.shape[0] - new_y, h + 2 * padding_y)
                
                # Extract face region
                face_region = image[new_y:new_y+new_h, new_x:new_x+new_w]
                
                # Detect facial features
                facial_features = self.detect_facial_features(face_region)
                
                # Generate filenames
                face_filename = f"{image_path.stem}_face_{i+1:02d}.jpg"
                annotated_filename = f"{image_path.stem}_face_{i+1:02d}_annotated.jpg"
                face_path = output_dir / face_filename
                annotated_path = output_dir / annotated_filename
                
                # Save extracted face
                cv2.imwrite(str(face_path), face_region)
                
                # Create annotated version with facial features
                annotated_face = self.annotate_facial_features(face_region, facial_features)
                cv2.imwrite(str(annotated_path), annotated_face)
                
                # Create face info
                face_info = ExtractedFace(
                    face_id=i+1,
                    original_image=image_path.name,
                    x=new_x,
                    y=new_y,
                    width=new_w,
                    height=new_h,
                    confidence=0.8,  # OpenCV doesn't provide confidence
                    extracted_filename=face_filename,
                    annotated_filename=annotated_filename,
                    border_padding=self.border_padding,
                    facial_features=facial_features
                )
                
                extracted_faces.append(face_info)
                logger.debug(f"Extracted face {i+1}: {face_filename}")
                
            except Exception as e:
                logger.error(f"Error extracting face {i+1}: {e}")
                continue
        
        return ExtractionResult(
            image_path=str(image_path),
            faces_found=len(faces),
            faces_extracted=len(extracted_faces),
            extraction_time=detection_time,
            extracted_faces=extracted_faces
        )
    
    def extract_faces_from_images(self, image_pattern: str, output_dir: Path, 
                                max_images: Optional[int] = None) -> List[ExtractionResult]:
        """
        Extract faces from multiple images.
        
        Args:
            image_pattern: Pattern to match image files or directory path
            output_dir: Directory to save extracted faces
            max_images: Maximum number of images to process
            
        Returns:
            List of extraction results
        """
        # Check if input is a directory
        input_path = Path(image_pattern)
        if input_path.is_dir():
            # If it's a directory, find all image files in it
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
            image_files = []
            for file_path in input_path.iterdir():
                if file_path.is_file() and file_path.suffix.lower() in image_extensions:
                    image_files.append(file_path)
        else:
            # Find images using pattern
            if image_pattern.startswith('/'):
                # Absolute path
                parent_dir = Path(image_pattern).parent
                pattern = Path(image_pattern).name
                image_files = list(parent_dir.glob(pattern))
            else:
                # Relative path
                image_files = list(Path('.').glob(image_pattern))
        
        if not image_files:
            logger.error(f"No images found matching pattern: {image_pattern}")
            return []
        
        # Limit number of images if specified
        if max_images:
            image_files = image_files[:max_images]
        
        logger.info(f"Found {len(image_files)} images to process")
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Process images
        results = []
        for image_path in tqdm(image_files, desc="Extracting faces"):
            result = self.extract_faces_from_image(image_path, output_dir)
            results.append(result)
        
        return results
    
    def create_extraction_summary(self, results: List[ExtractionResult], output_dir: Path) -> None:
        """Create a summary report of the extraction process."""
        total_images = len(results)
        total_faces_found = sum(r.faces_found for r in results)
        total_faces_extracted = sum(r.faces_extracted for r in results)
        total_time = sum(r.extraction_time for r in results)
        
        # Create summary data
        summary_data = {
            'extraction_summary': {
                'total_images_processed': total_images,
                'total_faces_found': total_faces_found,
                'total_faces_extracted': total_faces_extracted,
                'extraction_success_rate': total_faces_extracted / total_faces_found if total_faces_found > 0 else 0,
                'average_faces_per_image': total_faces_found / total_images if total_images > 0 else 0,
                'total_extraction_time': total_time,
                'average_time_per_image': total_time / total_images if total_images > 0 else 0,
                'border_padding_percentage': self.border_padding * 100
            },
            'image_results': [asdict(result) for result in results]
        }
        
        # Save JSON summary
        json_file = output_dir / "face_extraction_summary.json"
        with open(json_file, 'w') as f:
            json.dump(summary_data, f, indent=2, cls=NumpyEncoder)
        
        # Create markdown summary
        md_file = output_dir / "face_extraction_report.md"
        with open(md_file, 'w') as f:
            f.write("# Face Extraction Report\n\n")
            f.write(f"**Border Padding**: {self.border_padding*100:.0f}%\n")
            f.write(f"**Images Processed**: {total_images}\n")
            f.write(f"**Faces Found**: {total_faces_found}\n")
            f.write(f"**Faces Extracted**: {total_faces_extracted}\n")
            success_rate = (total_faces_extracted/total_faces_found*100) if total_faces_found > 0 else 0.0
            avg_faces = (total_faces_found/total_images) if total_images > 0 else 0.0
            f.write(f"**Success Rate**: {success_rate:.1f}%\n")
            f.write(f"**Average Faces per Image**: {avg_faces:.1f}\n")
            f.write(f"**Total Processing Time**: {total_time:.2f}s\n")
            f.write(f"**Average Time per Image**: {total_time/total_images:.2f}s\n\n")
            
            f.write("## Per-Image Results\n\n")
            f.write("| Image | Faces Found | Faces Extracted | Time (s) |\n")
            f.write("|-------|-------------|-----------------|----------|\n")
            
            for result in results:
                f.write(f"| {Path(result.image_path).name} | {result.faces_found} | {result.faces_extracted} | {result.extraction_time:.3f} |\n")
            
            f.write("\n## Extracted Faces\n\n")
            f.write("| Original Image | Face ID | Clean File | Annotated File | Dimensions | Features |\n")
            f.write("|----------------|---------|------------|----------------|------------|----------|\n")
            
            for result in results:
                for face in result.extracted_faces:
                    features_str = ", ".join([f.feature_type for f in face.facial_features])
                    f.write(f"| {face.original_image} | {face.face_id} | {face.extracted_filename} | {face.annotated_filename} | {face.width}x{face.height} | {features_str} |\n")
            
            f.write("\n## Facial Feature Summary\n\n")
            feature_counts = {}
            for result in results:
                for face in result.extracted_faces:
                    for feature in face.facial_features:
                        feature_counts[feature.feature_type] = feature_counts.get(feature.feature_type, 0) + 1
            
            f.write("| Feature Type | Count |\n")
            f.write("|--------------|-------|\n")
            for feature_type, count in sorted(feature_counts.items()):
                f.write(f"| {feature_type.replace('_', ' ').title()} | {count} |\n")
        
        logger.info(f"Extraction summary saved to {json_file}")
        logger.info(f"Extraction report saved to {md_file}")


@click.command()
@click.argument('input_pattern', type=str)
@click.argument('output_dir', type=str)
@click.option('--border-padding', '-b', default=0.25, help='Border padding percentage (0.25 = 25%)')
@click.option('--max-images', '-m', default=None, type=int, help='Maximum number of images to process')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
def main(input_pattern: str, output_dir: str, border_padding: float, max_images: Optional[int], verbose: bool):
    """Extract faces from images with configurable border padding."""
    
    # Setup logging
    if verbose:
        logger.add("face_extraction.log", level="DEBUG")
    
    # Create output directory
    output_path = Path(output_dir)
    
    # Initialize extractor
    extractor = FaceExtractor(border_padding=border_padding)
    
    # Extract faces
    logger.info(f"Starting face extraction with {border_padding*100:.0f}% border padding")
    results = extractor.extract_faces_from_images(input_pattern, output_path, max_images)
    
    if not results:
        logger.error("No images processed")
        return
    
    # Create summary
    extractor.create_extraction_summary(results, output_path)
    
    # Print summary
    total_faces = sum(r.faces_extracted for r in results)
    logger.info(f"Face extraction complete!")
    logger.info(f"Processed {len(results)} images")
    logger.info(f"Extracted {total_faces} faces")
    logger.info(f"Faces saved to {output_path}")
    logger.info(f"Border padding: {border_padding*100:.0f}%")


if __name__ == '__main__':
    main()
