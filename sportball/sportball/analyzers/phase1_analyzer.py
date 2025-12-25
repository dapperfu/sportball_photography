#!/usr/bin/env python3
"""
Phase 1 Sports Analysis Tool

This tool implements Phase 1 features for comprehensive sports photo analysis:
1. Ball detection and basic tracking
2. Action classification (running, jumping, kicking)
3. Field position analysis
4. Photo quality assessment
5. Enhanced player identification

Author: Claude Sonnet 4 (claude-3-5-sonnet-20241022)
Generated via Cursor IDE (cursor.sh) with AI assistance
"""

import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import click
import cv2
import numpy as np
from tqdm import tqdm
from loguru import logger

# Import our new Phase 1 modules
import sys
sys.path.append(str(Path(__file__).parent / "src"))
from ..detectors.ball import BallDetector, BallTrackingResult
from ..detectors.action import ActionClassifier, ActionClassificationResult
from ..detectors.field import FieldAnalyzer, FieldAnalysisResult
from ..detectors.quality import QualityAssessor, QualityAssessment
from ..detectors.face import FaceDetector
# Note: pose_detector not yet integrated - may need to be added


@dataclass
class Phase1AnalysisResult:
    """Complete Phase 1 analysis result for an image."""
    image_path: str
    ball_detection: BallTrackingResult
    action_classification: ActionClassificationResult
    field_analysis: FieldAnalysisResult
    quality_assessment: QualityAssessment
    face_count: int
    pose_count: int
    processing_time: float
    success: bool
    error: Optional[str] = None


@dataclass
class Phase1Summary:
    """Summary of Phase 1 analysis across multiple images."""
    total_images: int
    successful_analyses: int
    total_balls_detected: int
    total_actions_detected: int
    average_quality_score: float
    field_detection_rate: float
    processing_time: float
    ball_types: Dict[str, int]
    action_types: Dict[str, int]
    quality_grades: Dict[str, int]


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


class Phase1SportsAnalyzer:
    """Comprehensive Phase 1 sports analysis system."""
    
    def __init__(self, 
                 ball_confidence: float = 0.5,
                 action_confidence: float = 0.6,
                 enable_tracking: bool = True):
        """
        Initialize the Phase 1 analyzer.
        
        Args:
            ball_confidence: Ball detection confidence threshold
            action_confidence: Action classification confidence threshold
            enable_tracking: Enable motion tracking
        """
        # Initialize detectors
        self.ball_detector = BallDetector(
            confidence_threshold=ball_confidence,
            enable_tracking=enable_tracking
        )
        
        self.action_classifier = ActionClassifier(
            confidence_threshold=action_confidence,
            enable_speed_estimation=True
        )
        
        self.field_analyzer = FieldAnalyzer(
            field_type='soccer',
            enable_offside_detection=True
        )
        
        self.quality_assessor = QualityAssessor()
        
        # Initialize existing detectors
        self.face_detector = FaceDetector()
        self.pose_detector = PoseDetector()
        
        logger.info("Initialized Phase1SportsAnalyzer")
    
    def analyze_image(self, image_path: Path) -> Phase1AnalysisResult:
        """
        Perform comprehensive Phase 1 analysis on a single image.
        
        Args:
            image_path: Path to the input image
            
        Returns:
            Complete Phase 1 analysis result
        """
        start_time = cv2.getTickCount()
        
        try:
            # Load image
            image = cv2.imread(str(image_path))
            if image is None:
                return Phase1AnalysisResult(
                    image_path=str(image_path),
                    ball_detection=BallTrackingResult([], 0, False, "Failed to load image"),
                    action_classification=ActionClassificationResult([], 0, False, "Failed to load image"),
                    field_analysis=FieldAnalysisResult(False, [], [], (0, 0), 0, False, "Failed to load image"),
                    quality_assessment=QualityAssessment(QualityMetrics(0, 0, 0, 0, 0, 0, 0, 0), [], 'F', 0, False, "Failed to load image"),
                    face_count=0,
                    pose_count=0,
                    processing_time=0,
                    success=False,
                    error="Failed to load image"
                )
            
            # Detect faces and poses for player detection
            face_detections = self.face_detector.detect_faces(image)
            pose_results = self.pose_detector.detect_poses(image)
            
            # Extract player bounding boxes from face detections
            player_bboxes = []
            for face in face_detections:
                if 'bbox' in face:
                    x, y, w, h = face['bbox']
                    player_bboxes.append((x, y, w, h))
            
            # Perform Phase 1 analyses
            ball_detection = self.ball_detector.detect_balls(image)
            action_classification = self.action_classifier.classify_actions(image, player_bboxes)
            field_analysis = self.field_analyzer.analyze_field_positions(image, player_bboxes)
            quality_assessment = self.quality_assessor.assess_photo_quality(image)
            
            processing_time = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()
            
            return Phase1AnalysisResult(
                image_path=str(image_path),
                ball_detection=ball_detection,
                action_classification=action_classification,
                field_analysis=field_analysis,
                quality_assessment=quality_assessment,
                face_count=len(face_detections),
                pose_count=len(pose_results.get('poses', [])),
                processing_time=processing_time,
                success=True
            )
            
        except Exception as e:
            logger.error(f"Phase 1 analysis failed for {image_path}: {e}")
            processing_time = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()
            return Phase1AnalysisResult(
                image_path=str(image_path),
                ball_detection=BallTrackingResult([], processing_time, False, str(e)),
                action_classification=ActionClassificationResult([], processing_time, False, str(e)),
                field_analysis=FieldAnalysisResult(False, [], [], (0, 0), processing_time, False, str(e)),
                quality_assessment=QualityAssessment(QualityMetrics(0, 0, 0, 0, 0, 0, 0, 0), [], 'F', processing_time, False, str(e)),
                face_count=0,
                pose_count=0,
                processing_time=processing_time,
                success=False,
                error=str(e)
            )
    
    def analyze_images(self, image_pattern: str, output_dir: Path, max_images: Optional[int] = None) -> List[Phase1AnalysisResult]:
        """
        Analyze multiple images with Phase 1 features.
        
        Args:
            image_pattern: Pattern to match image files
            output_dir: Directory to save results
            max_images: Maximum number of images to process
            
        Returns:
            List of analysis results
        """
        # Find images
        if image_pattern.startswith('/'):
            parent_dir = Path(image_pattern).parent
            pattern = Path(image_pattern).name
            image_files = list(parent_dir.glob(pattern))
        else:
            image_files = list(Path('.').glob(image_pattern))
        
        if not image_files:
            logger.error(f"No images found matching pattern: {image_pattern}")
            return []
        
        # Limit number of images if specified
        if max_images:
            image_files = image_files[:max_images]
        
        logger.info(f"Found {len(image_files)} images to analyze")
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Analyze images
        results = []
        for image_path in tqdm(image_files, desc="Analyzing images"):
            result = self.analyze_image(image_path)
            results.append(result)
            
            # Save individual annotated images
            self._save_annotated_image(result, output_dir)
        
        return results
    
    def _save_annotated_image(self, result: Phase1AnalysisResult, output_dir: Path) -> None:
        """Save annotated image with all Phase 1 analysis results."""
        try:
            # Load original image
            image = cv2.imread(result.image_path)
            if image is None:
                return
            
            annotated = image.copy()
            
            # Draw ball detections
            if result.ball_detection.success:
                annotated = self.ball_detector.draw_ball_detections(annotated, result.ball_detection.balls_detected)
            
            # Draw action classifications
            if result.action_classification.success:
                annotated = self.action_classifier.draw_action_detections(annotated, result.action_classification.actions_detected)
            
            # Draw field analysis
            if result.field_analysis.success:
                annotated = self.field_analyzer.draw_field_analysis(annotated, result.field_analysis)
            
            # Draw quality assessment
            annotated = self.quality_assessor.draw_quality_assessment(annotated, result.quality_assessment)
            
            # Save annotated image
            image_name = Path(result.image_path).stem
            annotated_path = output_dir / f"{image_name}_phase1_analysis.jpg"
            cv2.imwrite(str(annotated_path), annotated)
            
        except Exception as e:
            logger.error(f"Failed to save annotated image: {e}")
    
    def create_summary(self, results: List[Phase1AnalysisResult], output_dir: Path) -> Phase1Summary:
        """Create comprehensive summary of Phase 1 analysis."""
        total_images = len(results)
        successful_analyses = sum(1 for r in results if r.success)
        
        # Count balls and actions
        total_balls = sum(len(r.ball_detection.balls_detected) for r in results if r.ball_detection.success)
        total_actions = sum(len(r.action_classification.actions_detected) for r in results if r.action_classification.success)
        
        # Calculate average quality score
        quality_scores = [r.quality_assessment.metrics.overall_score for r in results if r.quality_assessment.success]
        average_quality = np.mean(quality_scores) if quality_scores else 0.0
        
        # Calculate field detection rate
        field_detections = sum(1 for r in results if r.field_analysis.success and r.field_analysis.field_detected)
        field_detection_rate = field_detections / total_images if total_images > 0 else 0.0
        
        # Count ball types
        ball_types = {}
        for result in results:
            if result.ball_detection.success:
                for ball in result.ball_detection.balls_detected:
                    ball_types[ball.ball_type] = ball_types.get(ball.ball_type, 0) + 1
        
        # Count action types
        action_types = {}
        for result in results:
            if result.action_classification.success:
                for action in result.action_classification.actions_detected:
                    action_types[action.action_type] = action_types.get(action.action_type, 0) + 1
        
        # Count quality grades
        quality_grades = {}
        for result in results:
            if result.quality_assessment.success:
                grade = result.quality_assessment.quality_grade
                quality_grades[grade] = quality_grades.get(grade, 0) + 1
        
        # Calculate total processing time
        total_processing_time = sum(r.processing_time for r in results)
        
        summary = Phase1Summary(
            total_images=total_images,
            successful_analyses=successful_analyses,
            total_balls_detected=total_balls,
            total_actions_detected=total_actions,
            average_quality_score=average_quality,
            field_detection_rate=field_detection_rate,
            processing_time=total_processing_time,
            ball_types=ball_types,
            action_types=action_types,
            quality_grades=quality_grades
        )
        
        # Save summary to JSON
        summary_path = output_dir / "phase1_analysis_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(asdict(summary), f, indent=2, cls=NumpyEncoder)
        
        # Create markdown report
        self._create_markdown_report(summary, results, output_dir)
        
        return summary
    
    def _create_markdown_report(self, summary: Phase1Summary, results: List[Phase1AnalysisResult], output_dir: Path) -> None:
        """Create comprehensive markdown report."""
        report_path = output_dir / "phase1_analysis_report.md"
        
        with open(report_path, 'w') as f:
            f.write("# Phase 1 Sports Analysis Report\n\n")
            f.write(f"**Analysis Date**: {Path().cwd()}\n")
            f.write(f"**Total Images**: {summary.total_images}\n")
            f.write(f"**Successful Analyses**: {summary.successful_analyses}\n")
            f.write(f"**Success Rate**: {summary.successful_analyses/summary.total_images*100:.1f}%\n")
            f.write(f"**Total Processing Time**: {summary.processing_time:.2f}s\n")
            f.write(f"**Average Time per Image**: {summary.processing_time/summary.total_images:.2f}s\n\n")
            
            f.write("## Ball Detection Results\n\n")
            f.write(f"**Total Balls Detected**: {summary.total_balls_detected}\n")
            f.write(f"**Average Balls per Image**: {summary.total_balls_detected/summary.total_images:.1f}\n\n")
            
            f.write("### Ball Types Detected\n")
            f.write("| Ball Type | Count |\n")
            f.write("|-----------|-------|\n")
            for ball_type, count in summary.ball_types.items():
                f.write(f"| {ball_type.title()} | {count} |\n")
            f.write("\n")
            
            f.write("## Action Classification Results\n\n")
            f.write(f"**Total Actions Detected**: {summary.total_actions_detected}\n")
            f.write(f"**Average Actions per Image**: {summary.total_actions_detected/summary.total_images:.1f}\n\n")
            
            f.write("### Action Types Detected\n")
            f.write("| Action Type | Count |\n")
            f.write("|-------------|-------|\n")
            for action_type, count in summary.action_types.items():
                f.write(f"| {action_type.title()} | {count} |\n")
            f.write("\n")
            
            f.write("## Field Analysis Results\n\n")
            f.write(f"**Field Detection Rate**: {summary.field_detection_rate*100:.1f}%\n\n")
            
            f.write("## Photo Quality Assessment\n\n")
            f.write(f"**Average Quality Score**: {summary.average_quality_score:.2f}\n\n")
            
            f.write("### Quality Grade Distribution\n")
            f.write("| Grade | Count | Percentage |\n")
            f.write("|-------|-------|------------|\n")
            for grade, count in summary.quality_grades.items():
                percentage = count / summary.total_images * 100
                f.write(f"| {grade} | {count} | {percentage:.1f}% |\n")
            f.write("\n")
            
            f.write("## Per-Image Results\n\n")
            f.write("| Image | Balls | Actions | Quality | Grade | Time (s) |\n")
            f.write("|-------|-------|---------|---------|-------|----------|\n")
            
            for result in results:
                image_name = Path(result.image_path).name
                ball_count = len(result.ball_detection.balls_detected) if result.ball_detection.success else 0
                action_count = len(result.action_classification.actions_detected) if result.action_classification.success else 0
                quality_score = result.quality_assessment.metrics.overall_score if result.quality_assessment.success else 0.0
                quality_grade = result.quality_assessment.quality_grade if result.quality_assessment.success else 'F'
                
                f.write(f"| {image_name} | {ball_count} | {action_count} | {quality_score:.2f} | {quality_grade} | {result.processing_time:.2f} |\n")
        
        logger.info(f"Phase 1 analysis report saved to {report_path}")


@click.command()
@click.argument('input_pattern', type=str)
@click.argument('output_dir', type=str)
@click.option('--max-images', '-m', default=None, type=int, help='Maximum number of images to process')
@click.option('--ball-confidence', default=0.5, help='Ball detection confidence threshold')
@click.option('--action-confidence', default=0.6, help='Action classification confidence threshold')
@click.option('--enable-tracking', is_flag=True, default=True, help='Enable motion tracking')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
def main(input_pattern: str, output_dir: str, max_images: Optional[int], 
         ball_confidence: float, action_confidence: float, enable_tracking: bool, verbose: bool):
    """Perform comprehensive Phase 1 sports analysis."""
    
    # Setup logging
    if verbose:
        logger.add("phase1_analysis.log", level="DEBUG")
    
    # Create output directory
    output_path = Path(output_dir)
    
    # Initialize analyzer
    analyzer = Phase1SportsAnalyzer(
        ball_confidence=ball_confidence,
        action_confidence=action_confidence,
        enable_tracking=enable_tracking
    )
    
    # Analyze images
    logger.info("Starting Phase 1 sports analysis")
    results = analyzer.analyze_images(input_pattern, output_path, max_images)
    
    if not results:
        logger.error("No images processed")
        return
    
    # Create summary
    summary = analyzer.create_summary(results, output_path)
    
    # Print summary
    logger.info("Phase 1 analysis complete!")
    logger.info(f"Processed {summary.total_images} images")
    logger.info(f"Detected {summary.total_balls_detected} balls")
    logger.info(f"Classified {summary.total_actions_detected} actions")
    logger.info(f"Average quality score: {summary.average_quality_score:.2f}")
    logger.info(f"Field detection rate: {summary.field_detection_rate*100:.1f}%")
    logger.info(f"Results saved to {output_path}")


if __name__ == '__main__':
    main()
