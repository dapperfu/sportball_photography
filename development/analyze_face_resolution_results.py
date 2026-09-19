#!/usr/bin/env python3
"""
Analyze Face Resolution Test Results

This script analyzes the results from the face resolution test and creates
visualizations showing the relationship between image resolution and detection performance.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import cv2
from typing import Dict, List, Tuple
import click
from tqdm import tqdm


def load_test_results(results_file: Path) -> Dict:
    """Load the test results from JSON file."""
    with open(results_file, 'r') as f:
        return json.load(f)


def create_performance_charts(results: Dict, output_dir: Path):
    """Create performance visualization charts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract data
    resolutions = []
    faces_per_image = []
    confidences = []
    times_per_image = []
    method_counts = {}
    
    for res_key, summary in results['resolution_summaries'].items():
        if summary['total_images'] > 0:
            resolutions.append(res_key)
            faces_per_image.append(summary.get('avg_faces_per_image', 0))
            confidences.append(summary.get('avg_high_confidence_faces', 0))
            times_per_image.append(summary.get('avg_time_per_image', 0))
            
            # Collect method counts
            for method, count in summary.get('method_counts', {}).items():
                if method not in method_counts:
                    method_counts[method] = []
                method_counts[method].append(count)
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Face Detection Resolution Performance Analysis', fontsize=16)
    
    # Plot 1: Faces per image vs Resolution
    ax1.plot(range(len(resolutions)), faces_per_image, 'bo-', linewidth=2, markersize=8)
    ax1.set_title('Average Faces Detected per Image')
    ax1.set_xlabel('Resolution')
    ax1.set_ylabel('Faces per Image')
    ax1.set_xticks(range(len(resolutions)))
    ax1.set_xticklabels(resolutions, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: High-confidence faces vs Resolution
    ax2.plot(range(len(resolutions)), confidences, 'go-', linewidth=2, markersize=8)
    ax2.set_title('Average High-Confidence Faces per Image')
    ax2.set_xlabel('Resolution')
    ax2.set_ylabel('High-Confidence Faces')
    ax2.set_xticks(range(len(resolutions)))
    ax2.set_xticklabels(resolutions, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Processing time vs Resolution
    ax3.plot(range(len(resolutions)), times_per_image, 'ro-', linewidth=2, markersize=8)
    ax3.set_title('Average Processing Time per Image')
    ax3.set_xlabel('Resolution')
    ax3.set_ylabel('Time (seconds)')
    ax3.set_xticks(range(len(resolutions)))
    ax3.set_xticklabels(resolutions, rotation=45, ha='right')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Detection methods breakdown
    if method_counts:
        x = np.arange(len(resolutions))
        width = 0.35
        colors = ['skyblue', 'lightcoral', 'lightgreen', 'gold', 'plum']
        
        # Ensure all method counts have the same length as resolutions
        for method, counts in method_counts.items():
            if len(counts) != len(resolutions):
                # Pad with zeros if needed
                padded_counts = [0] * len(resolutions)
                for i, count in enumerate(counts):
                    if i < len(resolutions):
                        padded_counts[i] = count
                method_counts[method] = padded_counts
        
        for i, (method, counts) in enumerate(method_counts.items()):
            ax4.bar(x + i * width, counts, width, label=method, color=colors[i % len(colors)])
        
        ax4.set_title('Detection Methods by Resolution')
        ax4.set_xlabel('Resolution')
        ax4.set_ylabel('Total Detections')
        ax4.set_xticks(x + width / 2)
        ax4.set_xticklabels(resolutions, rotation=45, ha='right')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'face_resolution_performance.png', dpi=300, bbox_inches='tight')
    plt.close()


def create_sample_visualizations(results: Dict, output_dir: Path, input_pattern: str, num_samples: int = 5):
    """Create sample visualizations showing face detection at different resolutions."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get sample images
    if input_pattern.startswith('/'):
        # Absolute path pattern
        image_paths = list(Path(input_pattern).parent.glob(Path(input_pattern).name))
    else:
        # Relative path pattern
        image_paths = list(Path().glob(input_pattern))
    
    if len(image_paths) > num_samples:
        image_paths = image_paths[:num_samples]
    
    # Get top 3 resolutions by face count
    sorted_resolutions = sorted(
        results['resolution_summaries'].items(),
        key=lambda x: x[1].get('avg_faces_per_image', 0),
        reverse=True
    )
    top_resolutions = [res[0] for res in sorted_resolutions[:3]]
    
    for image_path in tqdm(image_paths, desc="Creating sample visualizations"):
        image_name = image_path.stem
        
        # Find results for this image
        image_results = None
        for result in results['detailed_results']:
            if result['image_name'] == image_path.name:
                image_results = result
                break
        
        if not image_results:
            continue
        
        # Create comparison visualization
        fig, axes = plt.subplots(1, len(top_resolutions), figsize=(20, 6))
        if len(top_resolutions) == 1:
            axes = [axes]
        
        fig.suptitle(f'Face Detection Comparison: {image_name}', fontsize=16)
        
        for i, resolution in enumerate(top_resolutions):
            # Find result for this resolution
            res_result = None
            for result in image_results['results']:
                if result['resolution'] == resolution:
                    res_result = result
                    break
            
            if res_result:
                # Load annotated image
                annotated_path = output_dir.parent / 'annotated_images' / f"{image_name}_{resolution}_annotated.jpg"
                if annotated_path.exists():
                    img = cv2.imread(str(annotated_path))
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    
                    axes[i].imshow(img)
                    axes[i].set_title(f'{resolution}\n{res_result["total_faces"]} faces detected')
                    axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_dir / f'face_comparison_{image_name}.png', dpi=300, bbox_inches='tight')
        plt.close()


def generate_report(results: Dict, output_dir: Path):
    """Generate a comprehensive analysis report."""
    report_path = output_dir / 'face_resolution_analysis_report.md'
    
    # Sort resolutions by performance
    sorted_resolutions = sorted(
        results['resolution_summaries'].items(),
        key=lambda x: x[1].get('avg_faces_per_image', 0),
        reverse=True
    )
    
    with open(report_path, 'w') as f:
        f.write("# Face Detection Resolution Analysis Report\n\n")
        
        # Test info
        test_info = results['test_info']
        f.write("## Test Information\n\n")
        f.write(f"- **Input Pattern**: {test_info['input_pattern']}\n")
        f.write(f"- **Images Tested**: {test_info['num_images_tested']}\n")
        f.write(f"- **Resolutions Tested**: {test_info['resolutions_tested']}\n")
        f.write(f"- **CUDA Available**: {test_info['cuda_available']}\n")
        f.write(f"- **Test Timestamp**: {test_info['test_timestamp']}\n\n")
        
        # Top performing resolutions
        f.write("## Top Performing Resolutions\n\n")
        f.write("| Rank | Resolution | Avg Faces/Image | Avg High-Confidence | Avg Time (s) |\n")
        f.write("|------|------------|-----------------|---------------------|--------------|\n")
        
        for i, (res_key, summary) in enumerate(sorted_resolutions[:10], 1):
            avg_faces = summary.get('avg_faces_per_image', 0)
            avg_conf = summary.get('avg_high_confidence_faces', 0)
            avg_time = summary.get('avg_time_per_image', 0)
            f.write(f"| {i} | {res_key} | {avg_faces:.1f} | {avg_conf:.1f} | {avg_time:.3f} |\n")
        
        f.write("\n")
        
        # Key findings
        f.write("## Key Findings\n\n")
        
        if sorted_resolutions:
            best_res = sorted_resolutions[0]
            worst_res = sorted_resolutions[-1]
            
            f.write(f"### Best Resolution: {best_res[0]}\n")
            f.write(f"- Average faces per image: {best_res[1].get('avg_faces_per_image', 0):.1f}\n")
            f.write(f"- Average processing time: {best_res[1].get('avg_time_per_image', 0):.3f}s\n\n")
            
            f.write(f"### Worst Resolution: {worst_res[0]}\n")
            f.write(f"- Average faces per image: {worst_res[1].get('avg_faces_per_image', 0):.1f}\n")
            f.write(f"- Average processing time: {worst_res[1].get('avg_time_per_image', 0):.3f}s\n\n")
            
            # Performance improvement
            best_faces = best_res[1].get('avg_faces_per_image', 0)
            worst_faces = worst_res[1].get('avg_faces_per_image', 0)
            if worst_faces > 0:
                improvement = ((best_faces - worst_faces) / worst_faces) * 100
                f.write(f"### Performance Improvement\n")
                f.write(f"- **{improvement:.1f}%** improvement in face detection from worst to best resolution\n\n")
        
        # Detection methods analysis
        f.write("## Detection Methods Analysis\n\n")
        
        # Collect all methods used
        all_methods = set()
        for summary in results['resolution_summaries'].values():
            all_methods.update(summary.get('method_counts', {}).keys())
        
        if all_methods:
            f.write("| Method | Total Detections | Average per Resolution |\n")
            f.write("|--------|------------------|------------------------|\n")
            
            method_totals = {}
            for summary in results['resolution_summaries'].values():
                for method, count in summary.get('method_counts', {}).items():
                    method_totals[method] = method_totals.get(method, 0) + count
            
            for method in sorted(method_totals.keys()):
                total = method_totals[method]
                avg = total / len(results['resolution_summaries'])
                f.write(f"| {method} | {total} | {avg:.1f} |\n")
        
        f.write("\n")
        
        # Recommendations
        f.write("## Recommendations\n\n")
        
        if sorted_resolutions:
            top_3 = sorted_resolutions[:3]
            f.write("### Optimal Resolution Settings\n\n")
            f.write("Based on the test results, the following resolutions provide the best balance of detection accuracy and processing speed:\n\n")
            
            for i, (res_key, summary) in enumerate(top_3, 1):
                avg_faces = summary.get('avg_faces_per_image', 0)
                avg_time = summary.get('avg_time_per_image', 0)
                f.write(f"{i}. **{res_key}**: {avg_faces:.1f} faces/image, {avg_time:.3f}s processing time\n")
            
            f.write("\n### Usage Guidelines\n\n")
            f.write("- **Maximum Detection**: Use the highest resolution that meets your processing time requirements\n")
            f.write("- **Balanced Performance**: Use 4K (3840x2160) for good detection with reasonable processing time\n")
            f.write("- **Fast Processing**: Use 1080p (1920x1080) for quick processing with decent detection\n")
            f.write("- **Batch Processing**: Consider using multiple resolutions and combining results for maximum accuracy\n\n")
        
        f.write("## Files Generated\n\n")
        f.write("- `face_resolution_performance.png`: Performance charts\n")
        f.write("- `face_comparison_*.png`: Sample image comparisons\n")
        f.write("- `face_resolution_analysis_report.md`: This report\n")
        f.write("- `annotated_images/`: Directory containing all annotated test images\n")


@click.command()
@click.argument('results_file', type=click.Path(exists=True))
@click.argument('output_dir', type=click.Path())
@click.option('--input-pattern', default="*.jpg", help='Input pattern for sample visualizations')
@click.option('--num-samples', default=5, help='Number of sample images for comparison')
def main(results_file: str, output_dir: str, input_pattern: str, num_samples: int):
    """
    Analyze face resolution test results and generate visualizations.
    
    RESULTS_FILE: Path to face_resolution_test_results.json
    OUTPUT_DIR: Directory to save analysis results
    """
    results_path = Path(results_file)
    output_path = Path(output_dir)
    
    print("🔍 Loading test results...")
    results = load_test_results(results_path)
    
    print("📊 Creating performance charts...")
    create_performance_charts(results, output_path)
    
    print("🖼️ Creating sample visualizations...")
    create_sample_visualizations(results, output_path, input_pattern, num_samples)
    
    print("📝 Generating analysis report...")
    generate_report(results, output_path)
    
    print(f"\n✅ Analysis complete! Results saved to: {output_path}")
    print(f"📄 Report: {output_path}/face_resolution_analysis_report.md")
    print(f"📊 Charts: {output_path}/face_resolution_performance.png")
    print(f"🖼️ Samples: {output_path}/face_comparison_*.png")


if __name__ == '__main__':
    main()
