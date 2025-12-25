# Sportball 🏈⚽📸

**Unified Sports Photo Analysis Package**

A comprehensive Python package for analyzing and organizing sports photographs using computer vision, machine learning, and AI techniques.

## 🚀 Features

- **Face Detection & Recognition** - Detect and cluster faces in sports photos
- **Object Detection** - YOLOv8-powered object detection (players, balls, equipment)
- **Game Boundary Detection** - Automatically split photos into games based on timestamps
- **Photo Quality Assessment** - Multi-metric quality analysis and filtering
- **Jersey Color & Number Detection** - Identify team colors and player numbers
- **Recursive Processing** - Process directories recursively by default
- **Parallel Processing** - GPU-accelerated processing with multi-threading
- **Sidecar Data Management** - JSON sidecar files for metadata and caching
- **Comprehensive CLI** - Clean command-line interface with subcommands

## 📦 Installation

### Basic Installation

```bash
pip install -e .
```

Or from PyPI (when published):
```bash
pip install sportball
```

### With GPU Support (Recommended)

```bash
pip install -e .[cuda]
```

### Development Installation

```bash
git clone <repository-url>
cd soccer_photo_sorter
pip install -e .[dev]
```

## 🎯 Quick Start

### Face Detection

```bash
# Detect faces in images (recursive by default)
sportball face detect /path/to/images

# Detect faces with specific confidence threshold
sportball face detect /path/to/images --confidence 0.7

# Extract face encodings for clustering
sportball face extract /path/to/images --output /path/to/encodings
```

### Object Detection

```bash
# Detect all objects in images
sportball object detect /path/to/images

# Detect specific objects (e.g., sports balls)
sportball object detect /path/to/images --classes "sports ball"

# Extract detected objects to separate files
sportball object extract /path/to/images --output /path/to/extracted
```

### Game Organization

```bash
# Automatically detect and split games
sportball games split /path/to/photos --output /path/to/games

# Use manual split points
sportball games split /path/to/photos --split-file splits.txt

# Process with parallel workers
sportball games split /path/to/photos --workers 8
```

### Quality Assessment

```bash
# Assess photo quality
sportball quality assess /path/to/images

# Filter by quality threshold
sportball quality assess /path/to/images --min-quality 0.7
```

### Sidecar Management

```bash
# Get statistics about sidecar files
sportball sidecar stats /path/to/images

# Clean up orphaned sidecar files
sportball sidecar cleanup /path/to/images
```

## 📁 Project Structure

```
sportball/
├── sportball/                    # Main package
│   ├── cli/                      # CLI commands
│   ├── detectors/                # Detection modules
│   ├── analyzers/                # Analysis tools
│   ├── utils/                    # Utility modules
│   ├── config/                   # Configuration
│   └── core/                     # Core functionality
├── tests/                        # Test suite
├── docs/                         # Documentation
└── pyproject.toml                # Package configuration
```

## 🔧 Configuration

### Model Files

Model files (e.g., `yolov8n.pt`) are stored in `~/.config/sportball/models/` for cross-project sharing. The package automatically uses this location when model paths are not explicitly specified.

To manually place a model file:
```bash
mkdir -p ~/.config/sportball/models
cp yolov8n.pt ~/.config/sportball/models/
```

### Environment Variables

- `XDG_CONFIG_HOME` - Override default config directory location
- `CUDA_VISIBLE_DEVICES` - Control GPU device selection

## 🚀 Migration from Old Scripts

If you were using the old root-level scripts (e.g., `enhanced_game_organizer.py`, `face_detection.py`), here's how to migrate:

### Old: `enhanced_game_organizer.py`
```bash
python enhanced_game_organizer.py --input /path/to/photos
```

### New: Use CLI command
```bash
sportball games split /path/to/photos --output /path/to/games
```

### Old: `face_detection.py`
```bash
python face_detection.py /path/to/images
```

### New: Use CLI command
```bash
sportball face detect /path/to/images
```

All functionality from the old scripts is available through the unified CLI interface. See `sportball --help` for all available commands.

## 📚 Documentation

- [Face Clustering Guide](docs/FACE_CLUSTERING_README.md)
- [Game Organizer Guide](docs/UNIFIED_GAME_ORGANIZER_README.md)

## 🛠️ Development

### Running Tests

```bash
pytest tests/
```

### Code Formatting

```bash
black sportball/
isort sportball/
```

### Type Checking

```bash
mypy sportball/
```

## 📝 Requirements

- Python 3.8+
- OpenCV (opencv-contrib-python)
- PyTorch
- Ultralytics (for YOLOv8)
- face-recognition
- Click, Rich, Loguru (for CLI)

See `requirements.txt` or `pyproject.toml` for complete dependency list.

## 🤝 Contributing

Contributions are welcome! Please see our contributing guidelines for details.

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- OpenCV for computer vision capabilities
- Ultralytics for YOLOv8
- face_recognition library
- The sports photography community
