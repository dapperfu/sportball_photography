#!/usr/bin/env python3
"""
YOLOv8 Object Extractor with Annotated Images and Individual Object Extraction

This tool extracts detected objects from images using YOLOv8 detection data
stored in JSON sidecar files. Creates annotated images with bounding boxes
and individual object files with labeled names.

Author: Claude Sonnet 4 (claude-3-5-sonnet-20241022)
Generated via Cursor IDE (cursor.sh) with AI assistance
"""

import json
import logging
import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Set
import click
import cv2
import numpy as np
from tqdm import tqdm
from loguru import logger


@dataclass
class ExtractionResult:
    """Result of object extraction from an image."""
    image_path: str
    objects_extracted: int
    annotated_image_path: Optional[str] = None
    individual_objects: List[Dict[str, Any]] = None
    error: Optional[str] = None


class YOLOv8ObjectExtractor:
    """YOLOv8 object extraction with annotated images and individual object files."""
    
    def __init__(self, 
                 annotation_style: str = "box",
                 font_scale: float = 1.0,
                 thickness: int = 2,
                 include_confidence: bool = True):
        """
        Initialize YOLOv8 object extractor.
        
        Args:
            annotation_style: Style of annotation ("box", "filled_box", "circle")
            font_scale: Scale of text font
            thickness: Thickness of annotation lines
            include_confidence: Whether to include confidence in labels
        """
        self.annotation_style = annotation_style
        self.font_scale = font_scale
        self.thickness = thickness
        self.include_confidence = include_confidence
        
        # Color palette for different object classes
        self.colors = self._generate_color_palette()
        
        logger.info(f"Initialized YOLOv8 extractor with {annotation_style} annotation style")
    
    def _generate_color_palette(self) -> Dict[str, Tuple[int, int, int]]:
        """Generate a color palette for different object classes."""
        # Define colors for common object classes
        color_map = {
            'person': (0, 255, 0),        # Green
            'sports ball': (255, 0, 0),   # Red
            'car': (0, 0, 255),           # Blue
            'truck': (255, 255, 0),      # Cyan
            'bus': (255, 0, 255),         # Magenta
            'bicycle': (0, 255, 255),     # Yellow
            'motorcycle': (128, 0, 128),  # Purple
            'airplane': (255, 165, 0),    # Orange
            'boat': (0, 128, 255),        # Light Blue
            'train': (128, 128, 0),       # Olive
            'bird': (255, 192, 203),     # Pink
            'cat': (255, 20, 147),       # Deep Pink
            'dog': (255, 69, 0),         # Red Orange
            'horse': (139, 69, 19),      # Saddle Brown
            'cow': (160, 82, 45),       # Sienna
            'elephant': (105, 105, 105), # Dim Gray
            'bear': (139, 0, 0),         # Dark Red
            'zebra': (255, 255, 255),    # White
            'giraffe': (255, 215, 0),    # Gold
            'chair': (128, 0, 0),        # Maroon
            'couch': (0, 100, 0),        # Dark Green
            'bed': (70, 130, 180),       # Steel Blue
            'dining table': (72, 61, 139), # Dark Slate Blue
            'toilet': (47, 79, 79),      # Dark Slate Gray
            'tv': (25, 25, 112),         # Midnight Blue
            'laptop': (0, 0, 139),       # Dark Blue
            'mouse': (139, 0, 139),      # Dark Magenta
            'remote': (85, 107, 47),     # Dark Olive Green
            'keyboard': (139, 69, 19),   # Saddle Brown
            'cell phone': (72, 61, 139), # Dark Slate Blue
            'microwave': (47, 79, 79),   # Dark Slate Gray
            'oven': (25, 25, 112),       # Midnight Blue
            'toaster': (0, 0, 139),      # Dark Blue
            'bottle': (139, 0, 139),     # Dark Magenta
            'wine glass': (85, 107, 47), # Dark Olive Green
            'cup': (139, 69, 19),        # Saddle Brown
            'fork': (72, 61, 139),       # Dark Slate Blue
            'knife': (47, 79, 79),       # Dark Slate Gray
            'spoon': (25, 25, 112),      # Midnight Blue
            'bowl': (0, 0, 139),         # Dark Blue
            'banana': (255, 255, 0),     # Yellow
            'apple': (255, 0, 0),        # Red
            'sandwich': (255, 165, 0),   # Orange
            'orange': (255, 140, 0),     # Dark Orange
            'broccoli': (0, 255, 0),     # Green
            'carrot': (255, 69, 0),      # Red Orange
            'hot dog': (255, 20, 147),   # Deep Pink
            'pizza': (255, 192, 203),   # Pink
            'donut': (255, 215, 0),     # Gold
            'cake': (255, 105, 180),     # Hot Pink
            'potted plant': (0, 128, 0), # Green
            'sink': (70, 130, 180),      # Steel Blue
            'refrigerator': (72, 61, 139), # Dark Slate Blue
            'book': (25, 25, 112),      # Midnight Blue
            'clock': (0, 0, 139),        # Dark Blue
            'vase': (139, 0, 139),       # Dark Magenta
            'scissors': (85, 107, 47),   # Dark Olive Green
            'teddy bear': (139, 69, 19), # Saddle Brown
            'hair drier': (72, 61, 139), # Dark Slate Blue
            'toothbrush': (47, 79, 79),  # Dark Slate Gray
        }
        
        # Add default colors for any missing classes
        default_colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
            (0, 255, 255), (128, 0, 128), (255, 165, 0), (0, 128, 255), (128, 128, 0),
            (255, 192, 203), (255, 20, 147), (255, 69, 0), (139, 69, 19), (160, 82, 45),
            (105, 105, 105), (139, 0, 0), (255, 255, 255), (255, 215, 0), (128, 0, 0),
            (0, 100, 0), (70, 130, 180), (72, 61, 139), (47, 79, 79), (25, 25, 112),
            (0, 0, 139), (139, 0, 139), (85, 107, 47), (139, 69, 19), (72, 61, 139),
            (47, 79, 79), (25, 25, 112), (0, 0, 139), (139, 0, 139), (85, 107, 47),
            (139, 69, 19), (72, 61, 139), (47, 79, 79), (25, 25, 112), (0, 0, 139)
        ]
        
        # Fill in missing colors with defaults
        color_index = 0
        for class_name in color_map:
            if class_name not in color_map:
                color_map[class_name] = default_colors[color_index % len(default_colors)]
                color_index += 1
        
        return color_map
    
    def _get_color_for_class(self, class_name: str) -> Tuple[int, int, int]:
        """Get color for a specific object class."""
        return self.colors.get(class_name, (255, 255, 255))  # Default to white
    
    def _draw_annotation(self, image: np.ndarray, x: int, y: int, width: int, height: int, 
                        label: str, color: Tuple[int, int, int]) -> np.ndarray:
        """
        Draw annotation on image.
        
        Args:
            image: Input image
            x, y, width, height: Bounding box coordinates
            label: Text label to display
            color: Color for annotation
            
        Returns:
            Annotated image
        """
        annotated_image = image.copy()
        
        if self.annotation_style == "box":
            # Draw rectangle
            cv2.rectangle(annotated_image, (x, y), (x + width, y + height), color, self.thickness)
        elif self.annotation_style == "filled_box":
            # Draw filled rectangle with transparency
            overlay = annotated_image.copy()
            cv2.rectangle(overlay, (x, y), (x + width, y + height), color, -1)
            cv2.addWeighted(overlay, 0.3, annotated_image, 0.7, 0, annotated_image)
            cv2.rectangle(annotated_image, (x, y), (x + width, y + height), color, self.thickness)
        elif self.annotation_style == "circle":
            # Draw circle at center
            center_x = x + width // 2
            center_y = y + height // 2
            radius = min(width, height) // 4
            cv2.circle(annotated_image, (center_x, center_y), radius, color, self.thickness)
        
        # Draw label background
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, self.thickness
        )
        
        # Position label above the bounding box
        label_x = x
        label_y = y - 10 if y - 10 > text_height else y + height + text_height + 10
        
        # Draw label background rectangle
        cv2.rectangle(annotated_image, 
                     (label_x, label_y - text_height - baseline),
                     (label_x + text_width, label_y + baseline),
                     color, -1)
        
        # Draw label text
        cv2.putText(annotated_image, label, (label_x, label_y - baseline),
                   cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, (255, 255, 255), self.thickness)
        
        return annotated_image
    
    def _annotate_individual_object(self, object_image: np.ndarray, class_name: str, 
                                   confidence: float, no_confidence: bool = False) -> np.ndarray:
        """
        Add annotation to an individual extracted object.
        
        Args:
            object_image: The cropped object image
            class_name: Name of the object class
            confidence: Detection confidence score
            no_confidence: Whether to exclude confidence from label
            
        Returns:
            Annotated object image
        """
        # Create a copy to avoid modifying the original
        annotated_image = object_image.copy()
        
        # Create label text
        if no_confidence:
            label = class_name
        else:
            label = f"{class_name} {confidence:.2f}"
        
        # Get text size for positioning
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, self.thickness
        )
        
        # Position label at top-left corner of the object
        label_x = 10
        label_y = text_height + 10
        
        # Ensure label fits within image bounds
        img_height, img_width = annotated_image.shape[:2]
        if label_x + text_width > img_width:
            label_x = img_width - text_width - 10
        if label_y > img_height:
            label_y = img_height - 10
        
        # Draw background rectangle for label
        cv2.rectangle(annotated_image,
                     (label_x, label_y - text_height - baseline),
                     (label_x + text_width, label_y + baseline),
                     (0, 0, 0), -1)  # Black background
        
        # Draw label text
        cv2.putText(annotated_image, label, (label_x, label_y - baseline),
                   cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, (255, 255, 255), self.thickness)
        
        return annotated_image
    
    def _mark_objects_extracted(self, json_path: Path, objects_count: int, annotated_image_path: Optional[str]) -> None:
        """
        Mark objects as extracted in the JSON sidecar file.
        
        Args:
            json_path: Path to the JSON sidecar file
            objects_count: Number of objects extracted
            annotated_image_path: Path to the annotated image (if created)
        """
        try:
            # Load existing JSON data
            with open(json_path, 'r') as f:
                json_data = json.load(f)
            
            # Add extraction metadata to YOLOv8 section
            if 'yolov8' in json_data:
                from datetime import datetime
                json_data['yolov8']['objects_extracted'] = {
                    'extraction_timestamp': datetime.now().isoformat(),
                    'objects_extracted_count': objects_count,
                    'annotated_image_path': annotated_image_path,
                    'extractor_version': '1.0.0'
                }
                
                # Save updated JSON
                with open(json_path, 'w') as f:
                    json.dump(json_data, f, indent=2)
                    
                logger.debug(f"Marked {objects_count} objects as extracted in {json_path.name}")
                
        except Exception as e:
            logger.warning(f"Failed to mark objects as extracted in {json_path}: {e}")
    
    def extract_objects_from_image(self, image_path: Path, output_dir: Path, 
                                 create_annotated: bool = True, 
                                 create_individual: bool = True,
                                 annotate_individual: bool = False,
                                 force: bool = False) -> ExtractionResult:
        """
        Extract objects from a single image using JSON sidecar data.
        
        Args:
            image_path: Path to the input image
            output_dir: Output directory for extracted objects
            create_annotated: Whether to create annotated image
            create_individual: Whether to create individual object files
            annotate_individual: Whether to add labels to individual extracted objects
            force: Whether to force extraction even if objects already extracted
            
        Returns:
            Extraction result with object information
        """
        logger.info(f"Extracting objects from {image_path.name}")
        
        # Resolve symlink to get the original image path
        original_image_path = image_path.resolve() if image_path.is_symlink() else image_path
        
        # Check if JSON sidecar exists
        json_path = original_image_path.parent / f"{original_image_path.stem}.json"
        if not json_path.exists():
            return ExtractionResult(
                image_path=str(image_path),
                objects_extracted=0,
                error="No JSON sidecar file found"
            )
        
        # Load JSON data
        try:
            with open(json_path, 'r') as f:
                json_data = json.load(f)
        except Exception as e:
            return ExtractionResult(
                image_path=str(image_path),
                objects_extracted=0,
                error=f"Failed to load JSON: {e}"
            )
        
        # Check if YOLOv8 data exists
        if 'yolov8' not in json_data:
            return ExtractionResult(
                image_path=str(image_path),
                objects_extracted=0,
                error="No YOLOv8 data found in JSON sidecar"
            )
        
        # Check if objects have already been extracted (unless force is True)
        if not force:
            yolov8_data = json_data['yolov8']
            if 'objects_extracted' in yolov8_data:
                logger.info(f"Skipping {image_path.name} - objects already extracted (use --force to override)")
                return ExtractionResult(
                    image_path=str(image_path),
                    objects_extracted=yolov8_data.get('total_objects_found', 0),
                    error="Skipped - objects already extracted"
                )
        
        yolov8_data = json_data['yolov8']
        objects_data = yolov8_data.get('objects', [])
        
        if not objects_data:
            logger.info(f"No objects found in {image_path.name}")
            return ExtractionResult(
                image_path=str(image_path),
                objects_extracted=0
            )
        
        # Load original image
        image = cv2.imread(str(image_path))
        if image is None:
            return ExtractionResult(
                image_path=str(image_path),
                objects_extracted=0,
                error="Failed to load image"
            )
        
        original_height, original_width = image.shape[:2]
        
        # Create output subdirectories
        image_output_dir = output_dir / image_path.stem
        image_output_dir.mkdir(parents=True, exist_ok=True)
        
        annotated_image_path = None
        individual_objects = []
        
        # Track class-specific counters for better naming
        class_counters = {}
        
        # Process each detected object
        for i, obj_data in enumerate(objects_data):
            try:
                # Get object information
                class_name = obj_data['class_name']
                class_id = obj_data['class_id']
                confidence = obj_data['confidence']
                
                # Increment class-specific counter
                class_counters[class_name] = class_counters.get(class_name, 0) + 1
                class_counter = class_counters[class_name]
                
                # Get coordinates (use pixel coordinates for extraction)
                coords_pixels = obj_data['coordinates_pixels']
                x = int(coords_pixels['x'])
                y = int(coords_pixels['y'])
                width = int(coords_pixels['width'])
                height = int(coords_pixels['height'])
                
                # Get crop area for individual extraction
                crop_percent = obj_data['crop_area_percent']
                crop_x = int(crop_percent['x'] * original_width)
                crop_y = int(crop_percent['y'] * original_height)
                crop_width = int(crop_percent['width'] * original_width)
                crop_height = int(crop_percent['height'] * original_height)
                
                # Ensure coordinates are within image bounds
                crop_x = max(0, crop_x)
                crop_y = max(0, crop_y)
                crop_width = min(crop_width, original_width - crop_x)
                crop_height = min(crop_height, original_height - crop_y)
                
                # Create individual object filename using class-specific counter
                object_filename = f"{image_path.stem}_{class_name}_{class_counter:02d}.jpg"
                object_path = image_output_dir / object_filename
                
                # Extract individual object
                if create_individual and crop_width > 0 and crop_height > 0:
                    object_image = image[crop_y:crop_y+crop_height, crop_x:crop_x+crop_width]
                    
                    # Add annotation to individual object if requested
                    if annotate_individual:
                        object_image = self._annotate_individual_object(
                            object_image, class_name, confidence, no_confidence=False
                        )
                    
                    cv2.imwrite(str(object_path), object_image)
                    
                    individual_objects.append({
                        'filename': object_filename,
                        'path': str(object_path),
                        'class_name': class_name,
                        'class_id': class_id,
                        'confidence': confidence,
                        'crop_coordinates': {
                            'x': crop_x, 'y': crop_y, 
                            'width': crop_width, 'height': crop_height
                        },
                        'original_coordinates': {
                            'x': x, 'y': y, 'width': width, 'height': height
                        }
                    })
                
                # Create annotated image
                if create_annotated:
                    # Create label text
                    label = f"{class_name}_{i+1:02d}"
                    if self.include_confidence:
                        label += f" ({confidence:.2f})"
                    
                    # Get color for this class
                    color = self._get_color_for_class(class_name)
                    
                    # Draw annotation
                    image = self._draw_annotation(image, x, y, width, height, label, color)
                
            except Exception as e:
                logger.error(f"Error processing object {i+1}: {e}")
                continue
        
        # Save annotated image
        if create_annotated and objects_data:
            annotated_filename = f"{image_path.stem}_annotated.jpg"
            annotated_image_path = image_output_dir / annotated_filename
            cv2.imwrite(str(annotated_image_path), image)
        
        # Mark objects as extracted in JSON file
        self._mark_objects_extracted(json_path, len(objects_data), str(annotated_image_path) if annotated_image_path else None)
        
        return ExtractionResult(
            image_path=str(image_path),
            objects_extracted=len(objects_data),
            annotated_image_path=str(annotated_image_path) if annotated_image_path else None,
            individual_objects=individual_objects
        )
    
    def extract_objects_from_directory(self, input_dir: Path, output_dir: Path,
                                     create_annotated: bool = True,
                                     create_individual: bool = True,
                                     annotate_individual: bool = False,
                                     max_images: Optional[int] = None,
                                     force: bool = False,
                                     max_workers: Optional[int] = None) -> List[ExtractionResult]:
        """
        Extract objects from all images in a directory.
        
        Args:
            input_dir: Input directory containing images and JSON sidecars
            output_dir: Output directory for extracted objects
            create_annotated: Whether to create annotated images
            create_individual: Whether to create individual object files
            annotate_individual: Whether to add labels to individual extracted objects
            max_images: Maximum number of images to process
            force: Whether to force extraction even if objects already extracted
            max_workers: Maximum number of parallel workers (default: CPU count)
            
        Returns:
            List of extraction results
        """
        # Find all images with JSON sidecars
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        image_files = []
        
        for file_path in input_dir.iterdir():
            if (file_path.is_file() and 
                file_path.suffix.lower() in image_extensions):
                # Resolve symlink to get the original image path
                original_file_path = file_path.resolve() if file_path.is_symlink() else file_path
                
                # Check if JSON sidecar exists and contains YOLOv8 data
                json_path = original_file_path.parent / f"{original_file_path.stem}.json"
                if json_path.exists():
                    try:
                        with open(json_path, 'r') as f:
                            json_data = json.load(f)
                        if 'yolov8' in json_data:
                            image_files.append(file_path)
                    except:
                        # Skip files with invalid JSON
                        continue
        
        if not image_files:
            logger.error(f"No images with YOLOv8 JSON sidecars found in {input_dir}")
            return []
        
        # Limit number of images if specified
        if max_images:
            image_files = image_files[:max_images]
        
        logger.info(f"Found {len(image_files)} images with YOLOv8 JSON sidecars to process")
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Process images in parallel
        results = []
        if max_workers is None:
            max_workers = os.cpu_count() or 4  # Default to CPU count, fallback to 4
        max_workers = min(max_workers, len(image_files))  # Don't exceed number of images
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_image = {
                executor.submit(self.extract_objects_from_image, image_path, output_dir, 
                              create_annotated, create_individual, annotate_individual, force): image_path 
                for image_path in image_files
            }
            
            # Process completed tasks with progress bar
            with tqdm(as_completed(future_to_image), 
                     total=len(image_files), 
                     desc="Extracting objects") as pbar:
                for future in pbar:
                    try:
                        result = future.result()
                        results.append(result)
                        
                        # Update progress bar description with current status
                        if result.error:
                            if "Skipped" in result.error:
                                # For skipped items, show info below progress bar
                                pbar.write(f"ℹ️  {result.image_path.split('/')[-1]}: {result.error}")
                                pbar.set_description(f"Extracting objects ({result.objects_extracted} extracted)")
                            else:
                                # For real errors, show in description
                                pbar.set_description(f"Extracting objects (Error: {result.error})")
                        else:
                            pbar.set_description(f"Extracting objects ({result.objects_extracted} extracted)")
                            
                    except Exception as e:
                        image_path = future_to_image[future]
                        logger.error(f"Error processing {image_path}: {e}")
                        # Create error result
                        error_result = ExtractionResult(
                            image_path=str(image_path),
                            objects_extracted=0,
                            error=str(e)
                        )
                        results.append(error_result)
                        pbar.write(f"❌ {image_path.name}: Error processing - {e}")
                        pbar.set_description(f"Extracting objects (Error: {e})")
        
        return results
    
    def create_extraction_report(self, results: List[ExtractionResult], output_dir: Path) -> Path:
        """
        Create a comprehensive extraction report.
        
        Args:
            results: List of extraction results
            output_dir: Output directory for the report
            
        Returns:
            Path to the generated report file
        """
        from datetime import datetime
        
        # Calculate statistics
        total_images = len(results)
        total_objects = sum(result.objects_extracted for result in results)
        successful_extractions = len([r for r in results if r.objects_extracted > 0])
        failed_extractions = len([r for r in results if r.error])
        
        # Count objects by class
        class_counts = {}
        individual_objects_count = 0
        
        for result in results:
            if result.individual_objects:
                individual_objects_count += len(result.individual_objects)
                for obj in result.individual_objects:
                    class_name = obj['class_name']
                    class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        # Create report data
        report_data = {
            'extraction_summary': {
                'total_images_processed': total_images,
                'successful_extractions': successful_extractions,
                'failed_extractions': failed_extractions,
                'total_objects_extracted': total_objects,
                'individual_objects_created': individual_objects_count,
                'extraction_timestamp': datetime.now().isoformat(),
                'tool_version': '1.0.0'
            },
            'objects_by_class': class_counts,
            'detailed_results': []
        }
        
        # Add detailed results
        for result in results:
            result_data = {
                'image_path': result.image_path,
                'objects_extracted': result.objects_extracted,
                'annotated_image_path': result.annotated_image_path,
                'individual_objects_count': len(result.individual_objects) if result.individual_objects else 0,
                'error': result.error
            }
            
            if result.individual_objects:
                result_data['individual_objects'] = [
                    {
                        'filename': obj['filename'],
                        'class_name': obj['class_name'],
                        'confidence': obj['confidence']
                    }
                    for obj in result.individual_objects
                ]
            
            report_data['detailed_results'].append(result_data)
        
        # Save report
        report_path = output_dir / "extraction_report.json"
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(report_path, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        logger.info(f"Extraction report saved: {report_path}")
        return report_path


@click.command()
@click.argument('input_path', type=click.Path(exists=True, path_type=Path))
@click.argument('output_dir', type=click.Path(path_type=Path))
@click.option('--annotation-style', '-a', 
              type=click.Choice(['box', 'filled_box', 'circle']),
              default='box',
              help='Style of annotation on images')
@click.option('--font-scale', '-f', default=1.0, help='Scale of text font')
@click.option('--thickness', '-t', default=2, help='Thickness of annotation lines')
@click.option('--no-confidence', is_flag=True, help='Exclude confidence scores from labels')
@click.option('--no-annotated', is_flag=True, help='Skip creating annotated images')
@click.option('--no-individual', is_flag=True, help='Skip creating individual object files')
@click.option('--annotate-individual', is_flag=True, help='Add labels to individual extracted objects')
@click.option('--max-images', '-m', default=None, type=int, help='Maximum number of images to process')
@click.option('--force', is_flag=True, help='Force extraction even if objects already extracted')
@click.option('--workers', '-w', default=None, type=int, help=f'Number of parallel workers (default: {os.cpu_count()})')
@click.option('--verbose', '-v', count=True, help='Enable verbose logging (-v for info, -vv for debug)')
def main(input_path: Path, output_dir: Path, annotation_style: str, font_scale: float, 
         thickness: int, no_confidence: bool, no_annotated: bool, no_individual: bool,
         annotate_individual: bool, max_images: Optional[int], force: bool, workers: Optional[int], verbose: int):
    """Extract objects from images using YOLOv8 detection data and create annotated images."""
    
    # Setup logging based on verbosity level
    logger.remove()  # Remove default handler
    
    if verbose >= 2:
        logger.add("yolo_object_extraction.log", level="DEBUG")
        logger.add(sys.stderr, level="DEBUG")
    elif verbose >= 1:
        logger.add(sys.stderr, level="INFO")
    else:
        # Only show warnings and errors by default
        logger.add(sys.stderr, level="WARNING")
    
    # Initialize extractor
    extractor = YOLOv8ObjectExtractor(
        annotation_style=annotation_style,
        font_scale=font_scale,
        thickness=thickness,
        include_confidence=not no_confidence
    )
    
    # Extract objects
    logger.info(f"Starting object extraction from {input_path}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Annotation style: {annotation_style}")
    logger.info(f"Create annotated images: {not no_annotated}")
    logger.info(f"Create individual objects: {not no_individual}")
    
    # Handle single file vs directory
    if input_path.is_file():
        # Single file processing
        result = extractor.extract_objects_from_image(
            image_path=input_path,
            output_dir=output_dir,
            create_annotated=not no_annotated,
            create_individual=not no_individual,
            annotate_individual=annotate_individual,
            force=force
        )
        results = [result]
    else:
        # Directory processing
        results = extractor.extract_objects_from_directory(
            input_dir=input_path,
            output_dir=output_dir,
            create_annotated=not no_annotated,
            create_individual=not no_individual,
            annotate_individual=annotate_individual,
            max_images=max_images,
            force=force,
            max_workers=workers
        )
    
    if not results:
        logger.error("No images processed")
        return 1
    
    # Create extraction report
    report_path = extractor.create_extraction_report(results, output_dir)
    
    # Calculate summary statistics
    total_images = len(results)
    total_objects = sum(result.objects_extracted for result in results)
    successful_extractions = len([r for r in results if r.objects_extracted > 0])
    failed_extractions = len([r for r in results if r.error])
    
    # Count objects by class
    class_counts = {}
    individual_objects_count = 0
    
    for result in results:
        if result.individual_objects:
            individual_objects_count += len(result.individual_objects)
            for obj in result.individual_objects:
                class_name = obj['class_name']
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
    
    logger.info(f"Object extraction complete!")
    logger.info(f"Processed {total_images} images")
    logger.info(f"Successfully extracted from {successful_extractions} images")
    logger.info(f"Failed extractions: {failed_extractions}")
    logger.info(f"Total objects extracted: {total_objects}")
    logger.info(f"Individual object files created: {individual_objects_count}")
    logger.info(f"Report saved: {report_path}")
    
    if class_counts:
        logger.info("Objects extracted by class:")
        for class_name, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"  {class_name}: {count}")
    
    return 0


if __name__ == "__main__":
    main()
