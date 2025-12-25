"""
Jersey Color Game Splitting Module

Intelligent game splitting based on jersey color patterns and player distributions
for organizing photographs containing multiple games.

Author: Claude Sonnet 4 (claude-3-5-sonnet-20241022)
Generated via Cursor IDE (cursor.sh) with AI assistance
"""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple, Union, Set
import numpy as np
from loguru import logger

from .pose import PoseDetector, PoseDetectionResult
from .color_analysis import ColorAnalyzer, ColorAnalysisResult, ColorCluster


@dataclass
class JerseyColorGroup:
    """Represents a group of similar jersey colors."""
    
    colors: List[ColorCluster]
    dominant_color: ColorCluster
    team_name: str
    confidence: float
    photo_count: int
    photo_paths: List[Path]


@dataclass
class GameSplitDecision:
    """Represents a decision about splitting a photo into games."""
    
    photo_path: Path
    should_split: bool
    split_games: List[str]  # List of team names/game identifiers
    confidence: float
    reasoning: str
    jersey_groups: List[JerseyColorGroup]


@dataclass
class JerseySplittingResult:
    """Result of jersey-based game splitting."""
    
    success: bool
    split_decisions: List[GameSplitDecision]
    detected_teams: List[JerseyColorGroup]
    processing_time: float
    error: Optional[str] = None


class JerseyGameSplitter:
    """
    Intelligent game splitting based on jersey color patterns.
    
    This class provides game splitting capabilities by analyzing jersey colors
    in photographs to identify different teams and automatically organize
    photos into separate games based on team affiliations.
    """
    
    def __init__(
        self,
        pose_confidence_threshold: float = 0.7,
        color_similarity_threshold: float = 0.15,
        min_team_photos: int = 5,
        multi_team_threshold: float = 0.3,
        cache_enabled: bool = True,
    ):
        """
        Initialize the JerseyGameSplitter.
        
        Args:
            pose_confidence_threshold: Minimum confidence for pose detections
            color_similarity_threshold: Threshold for grouping similar jersey colors
            min_team_photos: Minimum photos required to form a team
            multi_team_threshold: Threshold for detecting multi-team photos
            cache_enabled: Whether to enable result caching
        """
        self.pose_confidence_threshold = pose_confidence_threshold
        self.color_similarity_threshold = color_similarity_threshold
        self.min_team_photos = min_team_photos
        self.multi_team_threshold = multi_team_threshold
        self.cache_enabled = cache_enabled
        
        self.logger = logger.bind(component="jersey_splitter")
        
        # Initialize pose detector and color analyzer
        self.pose_detector = PoseDetector(
            confidence_threshold=pose_confidence_threshold,
            cache_enabled=cache_enabled
        )
        
        self.color_analyzer = ColorAnalyzer(
            similarity_threshold=color_similarity_threshold,
            cache_enabled=cache_enabled
        )
        
        self.logger.info("Initialized JerseyGameSplitter")
    
    def split_games_by_jersey_color(
        self,
        image_paths: Union[Path, List[Path]],
        output_dir: Optional[Path] = None,
        save_sidecar: bool = True,
        **kwargs
    ) -> JerseySplittingResult:
        """
        Split games based on jersey colors.
        
        Args:
            image_paths: Single image path or list of image paths
            output_dir: Optional output directory for organized games
            save_sidecar: Whether to save results to sidecar files
            **kwargs: Additional arguments for splitting
            
        Returns:
            JerseySplittingResult containing splitting decisions
        """
        if isinstance(image_paths, Path):
            image_paths = [image_paths]
        
        self.logger.info(f"Splitting games by jersey color for {len(image_paths)} images")
        
        start_time = time.perf_counter()
        
        try:
            # Step 1: Detect poses in all images
            self.logger.info("Step 1: Detecting poses...")
            pose_results = self.pose_detector.detect_poses(
                image_paths, save_sidecar=save_sidecar
            )
            
            # Step 2: Analyze jersey colors
            self.logger.info("Step 2: Analyzing jersey colors...")
            color_results = self.color_analyzer.analyze_jersey_colors(
                image_paths, pose_results=pose_results, save_sidecar=save_sidecar
            )
            
            # Step 3: Identify jersey color groups (teams)
            self.logger.info("Step 3: Identifying jersey color groups...")
            jersey_groups = self._identify_jersey_groups(color_results)
            
            # Step 4: Make splitting decisions
            self.logger.info("Step 4: Making splitting decisions...")
            split_decisions = self._make_splitting_decisions(
                image_paths, color_results, jersey_groups
            )
            
            # Step 5: Organize output if requested
            if output_dir:
                self.logger.info("Step 5: Organizing output...")
                self._organize_output(split_decisions, jersey_groups, output_dir)
            
            processing_time = time.perf_counter() - start_time
            
            return JerseySplittingResult(
                success=True,
                split_decisions=split_decisions,
                detected_teams=jersey_groups,
                processing_time=processing_time
            )
            
        except Exception as e:
            processing_time = time.perf_counter() - start_time
            self.logger.error(f"Jersey splitting failed: {e}")
            return JerseySplittingResult(
                success=False,
                split_decisions=[],
                detected_teams=[],
                processing_time=processing_time,
                error=str(e)
            )
    
    def _identify_jersey_groups(
        self, 
        color_results: Dict[str, Any]
    ) -> List[JerseyColorGroup]:
        """Identify jersey color groups representing different teams."""
        all_jersey_colors = []
        photo_color_mapping = {}
        
        # Collect all jersey colors from all photos
        for photo_path_str, result in color_results.items():
            if not result.get("success", False):
                continue
            
            jersey_analysis = result.get("jersey_analysis")
            if not jersey_analysis:
                continue
            
            jersey_colors = jersey_analysis.get("jersey_colors", [])
            if jersey_colors:
                photo_color_mapping[photo_path_str] = jersey_colors
                all_jersey_colors.extend(jersey_colors)
        
        if not all_jersey_colors:
            return []
        
        # Group similar colors
        color_groups = self._group_similar_colors(all_jersey_colors)
        
        # Create JerseyColorGroup objects
        jersey_groups = []
        for i, color_group in enumerate(color_groups):
            if len(color_group) < self.min_team_photos:
                continue  # Skip groups with too few colors
            
            # Find dominant color (most frequent)
            dominant_color = max(color_group, key=lambda c: c["percentage"])
            
            # Count photos for this team
            team_photos = []
            for photo_path_str, photo_colors in photo_color_mapping.items():
                for photo_color in photo_colors:
                    if self._colors_match(photo_color, dominant_color):
                        team_photos.append(Path(photo_path_str))
                        break
            
            if len(team_photos) >= self.min_team_photos:
                team_name = f"Team_{i+1}_{self._color_to_name(dominant_color)}"
                
                jersey_group = JerseyColorGroup(
                    colors=color_group,
                    dominant_color=dominant_color,
                    team_name=team_name,
                    confidence=self._calculate_team_confidence(color_group),
                    photo_count=len(team_photos),
                    photo_paths=team_photos
                )
                jersey_groups.append(jersey_group)
        
        return jersey_groups
    
    def _group_similar_colors(self, colors: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """Group similar colors together."""
        if len(colors) <= 1:
            return [colors]
        
        groups = []
        used = set()
        
        for i, color1 in enumerate(colors):
            if i in used:
                continue
            
            group = [color1]
            used.add(i)
            
            for j, color2 in enumerate(colors[i+1:], i+1):
                if j in used:
                    continue
                
                # Calculate color similarity
                similarity = self._calculate_color_similarity(color1, color2)
                
                if similarity < self.color_similarity_threshold:
                    group.append(color2)
                    used.add(j)
            
            groups.append(group)
        
        return groups
    
    def _calculate_color_similarity(self, color1: Dict[str, Any], color2: Dict[str, Any]) -> float:
        """Calculate similarity between two colors using LAB color space."""
        lab1 = np.array(color1["lab_color"])
        lab2 = np.array(color2["lab_color"])
        
        # Euclidean distance in LAB space
        distance = np.linalg.norm(lab1 - lab2)
        
        # Normalize to 0-1 range
        normalized_distance = distance / 100.0
        
        return normalized_distance
    
    def _colors_match(self, color1: Dict[str, Any], color2: Dict[str, Any]) -> bool:
        """Check if two colors match within similarity threshold."""
        similarity = self._calculate_color_similarity(color1, color2)
        return similarity < self.color_similarity_threshold
    
    def _color_to_name(self, color: Dict[str, Any]) -> str:
        """Convert color to a human-readable name."""
        rgb = color["rgb_color"]
        r, g, b = rgb
        
        # Simple color naming based on RGB values
        if r > g and r > b:
            if r > 200 and g < 100 and b < 100:
                return "Red"
            elif r > 150 and g > 100 and b < 100:
                return "Orange"
        elif g > r and g > b:
            if g > 200 and r < 100 and b < 100:
                return "Green"
            elif g > 150 and r > 100 and b < 100:
                return "Yellow"
        elif b > r and b > g:
            if b > 200 and r < 100 and g < 100:
                return "Blue"
            elif b > 150 and r > 100 and g < 100:
                return "Purple"
        elif r > 200 and g > 200 and b > 200:
            return "White"
        elif r < 50 and g < 50 and b < 50:
            return "Black"
        
        # Fallback to RGB values
        return f"RGB_{r}_{g}_{b}"
    
    def _calculate_team_confidence(self, color_group: List[Dict[str, Any]]) -> float:
        """Calculate confidence for a team based on color consistency."""
        if not color_group:
            return 0.0
        
        # Confidence based on:
        # 1. Number of similar colors (more = more confident)
        # 2. Color consistency (lower variance = more confident)
        # 3. Dominance of the most common color
        
        num_colors = len(color_group)
        color_consistency = 1.0 - (num_colors - 1) / max(num_colors, 10)  # Penalty for too many colors
        
        # Calculate color variance
        lab_colors = [np.array(c["lab_color"]) for c in color_group]
        if len(lab_colors) > 1:
            mean_color = np.mean(lab_colors, axis=0)
            variance = np.mean([np.linalg.norm(c - mean_color) for c in lab_colors])
            color_variance_penalty = min(variance / 50.0, 1.0)  # Normalize variance
        else:
            color_variance_penalty = 0.0
        
        confidence = (color_consistency * 0.6) + ((1.0 - color_variance_penalty) * 0.4)
        
        return min(confidence, 1.0)
    
    def _make_splitting_decisions(
        self,
        image_paths: List[Path],
        color_results: Dict[str, Any],
        jersey_groups: List[JerseyColorGroup]
    ) -> List[GameSplitDecision]:
        """Make splitting decisions for each photo."""
        split_decisions = []
        
        for image_path in image_paths:
            photo_path_str = str(image_path)
            
            if photo_path_str not in color_results:
                # No color analysis available
                decision = GameSplitDecision(
                    photo_path=image_path,
                    should_split=False,
                    split_games=[],
                    confidence=0.0,
                    reasoning="No color analysis available",
                    jersey_groups=[]
                )
                split_decisions.append(decision)
                continue
            
            color_result = color_results[photo_path_str]
            if not color_result.get("success", False):
                # Color analysis failed
                decision = GameSplitDecision(
                    photo_path=image_path,
                    should_split=False,
                    split_games=[],
                    confidence=0.0,
                    reasoning="Color analysis failed",
                    jersey_groups=[]
                )
                split_decisions.append(decision)
                continue
            
            jersey_analysis = color_result.get("jersey_analysis")
            if not jersey_analysis:
                # No jersey analysis available
                decision = GameSplitDecision(
                    photo_path=image_path,
                    should_split=False,
                    split_games=[],
                    confidence=0.0,
                    reasoning="No jersey analysis available",
                    jersey_groups=[]
                )
                split_decisions.append(decision)
                continue
            
            # Analyze jersey colors in this photo
            photo_jersey_colors = jersey_analysis.get("jersey_colors", [])
            
            if not photo_jersey_colors:
                # No jersey colors detected
                decision = GameSplitDecision(
                    photo_path=image_path,
                    should_split=False,
                    split_games=[],
                    confidence=0.0,
                    reasoning="No jersey colors detected",
                    jersey_groups=[]
                )
                split_decisions.append(decision)
                continue
            
            # Match photo colors to team groups
            matched_teams = []
            photo_jersey_groups = []
            
            for jersey_group in jersey_groups:
                for photo_color in photo_jersey_colors:
                    if self._colors_match(photo_color, jersey_group.dominant_color):
                        matched_teams.append(jersey_group.team_name)
                        photo_jersey_groups.append(jersey_group)
                        break
            
            # Determine if photo should be split
            should_split = len(matched_teams) > 1
            confidence = self._calculate_split_confidence(
                photo_jersey_colors, matched_teams, jersey_groups
            )
            
            reasoning = self._generate_split_reasoning(
                photo_jersey_colors, matched_teams, should_split
            )
            
            decision = GameSplitDecision(
                photo_path=image_path,
                should_split=should_split,
                split_games=matched_teams,
                confidence=confidence,
                reasoning=reasoning,
                jersey_groups=photo_jersey_groups
            )
            split_decisions.append(decision)
        
        return split_decisions
    
    def _calculate_split_confidence(
        self,
        photo_jersey_colors: List[Dict[str, Any]],
        matched_teams: List[str],
        jersey_groups: List[JerseyColorGroup]
    ) -> float:
        """Calculate confidence in splitting decision."""
        if not photo_jersey_colors:
            return 0.0
        
        # Base confidence from color analysis quality
        base_confidence = 0.5
        
        # Bonus for multiple distinct teams
        if len(matched_teams) > 1:
            team_diversity_bonus = min(len(matched_teams) / 3.0, 0.3)
        else:
            team_diversity_bonus = 0.0
        
        # Bonus for strong color separation
        if len(photo_jersey_colors) > 1:
            color_separation = self._calculate_color_separation(photo_jersey_colors)
            separation_bonus = color_separation * 0.2
        else:
            separation_bonus = 0.0
        
        confidence = base_confidence + team_diversity_bonus + separation_bonus
        
        return min(confidence, 1.0)
    
    def _calculate_color_separation(self, colors: List[Dict[str, Any]]) -> float:
        """Calculate how well separated the colors are."""
        if len(colors) <= 1:
            return 0.0
        
        # Calculate pairwise distances between colors
        distances = []
        for i, color1 in enumerate(colors):
            for j, color2 in enumerate(colors[i+1:], i+1):
                distance = self._calculate_color_similarity(color1, color2)
                distances.append(distance)
        
        if not distances:
            return 0.0
        
        # Higher average distance = better separation
        avg_distance = sum(distances) / len(distances)
        
        # Normalize to 0-1 range
        normalized_separation = min(avg_distance / self.color_similarity_threshold, 1.0)
        
        return normalized_separation
    
    def _generate_split_reasoning(
        self,
        photo_jersey_colors: List[Dict[str, Any]],
        matched_teams: List[str],
        should_split: bool
    ) -> str:
        """Generate human-readable reasoning for splitting decision."""
        if not photo_jersey_colors:
            return "No jersey colors detected"
        
        if len(matched_teams) == 0:
            return "No teams matched to jersey colors"
        
        if len(matched_teams) == 1:
            return f"Single team detected: {matched_teams[0]}"
        
        if should_split:
            return f"Multiple teams detected: {', '.join(matched_teams)} - should split"
        else:
            return f"Multiple teams detected but confidence too low: {', '.join(matched_teams)}"
    
    def _organize_output(
        self,
        split_decisions: List[GameSplitDecision],
        jersey_groups: List[JerseyColorGroup],
        output_dir: Path
    ):
        """Organize photos into output directories based on splitting decisions."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create directories for each team
        team_dirs = {}
        for jersey_group in jersey_groups:
            team_dir = output_dir / jersey_group.team_name
            team_dir.mkdir(exist_ok=True)
            team_dirs[jersey_group.team_name] = team_dir
        
        # Create directory for multi-team photos
        multi_team_dir = output_dir / "MultiTeam"
        multi_team_dir.mkdir(exist_ok=True)
        
        # Create directory for single-team photos
        single_team_dir = output_dir / "SingleTeam"
        single_team_dir.mkdir(exist_ok=True)
        
        # Organize photos
        for decision in split_decisions:
            photo_path = decision.photo_path
            
            if not photo_path.exists():
                continue
            
            if decision.should_split and len(decision.split_games) > 1:
                # Multi-team photo - copy to multi-team directory
                dest_path = multi_team_dir / photo_path.name
                self._copy_photo(photo_path, dest_path)
            elif len(decision.split_games) == 1:
                # Single team photo - copy to team directory
                team_name = decision.split_games[0]
                if team_name in team_dirs:
                    dest_path = team_dirs[team_name] / photo_path.name
                    self._copy_photo(photo_path, dest_path)
            else:
                # No clear team - copy to single team directory
                dest_path = single_team_dir / photo_path.name
                self._copy_photo(photo_path, dest_path)
    
    def _copy_photo(self, src_path: Path, dest_path: Path):
        """Copy photo from source to destination."""
        try:
            import shutil
            shutil.copy2(src_path, dest_path)
        except Exception as e:
            self.logger.warning(f"Failed to copy {src_path} to {dest_path}: {e}")
    
    def get_split_summary(self, result: JerseySplittingResult) -> Dict[str, Any]:
        """Get summary of splitting results."""
        if not result.success:
            return {"error": result.error}
        
        total_photos = len(result.split_decisions)
        split_photos = sum(1 for d in result.split_decisions if d.should_split)
        single_team_photos = sum(1 for d in result.split_decisions if len(d.split_games) == 1)
        multi_team_photos = sum(1 for d in result.split_decisions if len(d.split_games) > 1)
        no_team_photos = sum(1 for d in result.split_decisions if len(d.split_games) == 0)
        
        avg_confidence = sum(d.confidence for d in result.split_decisions) / total_photos if total_photos > 0 else 0
        
        return {
            "total_photos": total_photos,
            "detected_teams": len(result.detected_teams),
            "teams": [
                {
                    "name": team.team_name,
                    "color": team.dominant_color["rgb_color"],
                    "photo_count": team.photo_count,
                    "confidence": team.confidence
                }
                for team in result.detected_teams
            ],
            "split_photos": split_photos,
            "single_team_photos": single_team_photos,
            "multi_team_photos": multi_team_photos,
            "no_team_photos": no_team_photos,
            "average_confidence": avg_confidence,
            "processing_time": result.processing_time
        }
