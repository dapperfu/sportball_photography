# AI-Based Face Clustering Tool

This tool reads face recognition JSON data and uses AI clustering algorithms to group similar faces into likely persons. It then extracts and crops faces with configurable padding and saves them to organized directories.

## Features

- **Multiple Clustering Algorithms**: DBSCAN, K-means, and Hierarchical clustering
- **Face Encoding Extraction**: Automatically extracts face encodings from JSON data
- **Configurable Padding**: Set custom padding percentage for face crops (default: 25%)
- **Organized Output**: Creates `person_01/`, `person_02/`, etc. directories
- **Parallel Processing**: High-performance parallel processing for large datasets
- **Comprehensive Reporting**: Detailed clustering reports with statistics
- **Rich CLI Interface**: Beautiful command-line interface with progress bars

## Prerequisites

1. **Face Recognition Data**: You need JSON files containing face detection data with encodings
2. **Dependencies**: Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Basic Usage

```bash
# Basic clustering with default settings
python face_clustering.py --input /path/to/json/files --output /path/to/output

# Specify custom output directory
python face_clustering.py --input /path/to/json/files --output ./clustered_faces
```

### Advanced Options

```bash
# Use K-means algorithm with custom parameters
python face_clustering.py --input /path/to/json/files --algorithm kmeans --max-clusters 20

# Use Hierarchical clustering
python face_clustering.py --input /path/to/json/files --algorithm hierarchical

# Custom padding and cluster size
python face_clustering.py --input /path/to/json/files --padding 0.3 --min-cluster-size 3

# High performance with 8 workers
python face_clustering.py --input /path/to/json/files --workers 8

# Verbose logging
python face_clustering.py --input /path/to/json/files --verbose
```

### Command Line Options

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--input` | `-i` | Required | Input directory containing JSON files |
| `--output` | `-o` | `./results/clustered_faces` | Output directory for organized faces |
| `--pattern` | `-p` | `*.json` | File pattern to match JSON files |
| `--algorithm` | `-a` | `dbscan` | Clustering algorithm (`dbscan`, `kmeans`, `hierarchical`) |
| `--padding` | | `0.25` | Padding percentage for face crops (0.25 = 25%) |
| `--min-cluster-size` | | `2` | Minimum faces per cluster |
| `--max-clusters` | | `50` | Maximum number of clusters (K-means/Hierarchical) |
| `--workers` | `-w` | `4` | Number of parallel workers |
| `--verbose` | `-v` | False | Enable verbose logging |
| `--quiet` | `-q` | False | Suppress output except errors |

## Input Data Format

The tool expects JSON files created by the face detection system. Each JSON file should contain:

```json
{
  "Face_detector": {
    "faces": [
      {
        "face_id": 1,
        "coordinates": {
          "x": 100,
          "y": 150,
          "width": 200,
          "height": 250
        },
        "confidence": 0.95,
        "face_encoding": [0.1, 0.2, ...],  // 128-dimensional encoding
        "crop_area": {
          "x": 75,
          "y": 125,
          "width": 250,
          "height": 300
        },
        "detection_scale_factor": 1.0
      }
    ]
  }
}
```

## Output Structure

The tool creates an organized directory structure:

```
output_directory/
├── person_01/
│   ├── photo_001_face_000.jpg
│   ├── photo_002_face_000.jpg
│   └── ...
├── person_02/
│   ├── photo_003_face_000.jpg
│   └── ...
├── person_03/
│   └── ...
└── face_clustering_report.json
```

## Clustering Algorithms

### DBSCAN (Default)
- **Best for**: Automatic cluster detection, handling noise
- **Pros**: No need to specify number of clusters, handles outliers
- **Cons**: Sensitive to parameter tuning

### K-means
- **Best for**: Known number of clusters, spherical clusters
- **Pros**: Fast, works well with spherical data
- **Cons**: Requires specifying number of clusters, sensitive to initialization

### Hierarchical
- **Best for**: Hierarchical relationships, small datasets
- **Pros**: Creates cluster hierarchy, deterministic
- **Cons**: Computationally expensive for large datasets

## Performance Tips

1. **Use Parallel Processing**: Increase `--workers` for faster processing
2. **Choose Right Algorithm**: DBSCAN for unknown clusters, K-means for known count
3. **Adjust Cluster Size**: Set `--min-cluster-size` based on your data
4. **Optimize Padding**: Smaller padding = faster processing, larger = better crops

## Example Workflow

1. **Run Face Detection**: First run face detection on your images to generate JSON files
2. **Cluster Faces**: Use this tool to group similar faces
3. **Review Results**: Check the generated report and organized directories
4. **Fine-tune**: Adjust parameters based on results

```bash
# Step 1: Detect faces (if not already done)
python face_detection.py --input /path/to/images --pattern "*.jpg"

# Step 2: Cluster detected faces
python face_clustering.py --input /path/to/json/files --output ./clustered_faces

# Step 3: Review results
ls ./clustered_faces/
cat ./clustered_faces/face_clustering_report.json
```

## Troubleshooting

### Common Issues

1. **No JSON files found**: Ensure JSON files are in the input directory
2. **No face encodings**: Check that JSON files contain `face_encoding` data
3. **No clusters created**: Try reducing `--min-cluster-size` or adjusting algorithm
4. **Poor clustering**: Try different algorithms or adjust parameters

### Debug Mode

Use `--verbose` flag for detailed logging:

```bash
python face_clustering.py --input /path/to/json/files --verbose
```

## Testing

Run the test script to verify functionality:

```bash
python test_face_clustering.py
```

## Integration

The face clustering tool integrates with the existing soccer photo sorter pipeline:

1. **Face Detection** → Generates JSON files with face encodings
2. **Face Clustering** → Groups similar faces into persons
3. **Face Extraction** → Crops and organizes faces by person

This creates a complete pipeline for organizing photos by detected faces.
