# Sportball 🏈⚽📸

**Unified Sports Photo Analysis Package**

A comprehensive Python package for analyzing and organizing sports photographs using computer vision, machine learning, and AI techniques.

## 🚀 Features

- **Face Detection & Recognition** - Detect and cluster faces in sports photos
- **Object Detection** - YOLOv8-powered object detection (players, balls, equipment)
- **Game Boundary Detection** - Automatically split photos into games based on timestamps
- **Jersey Color Splitting** - Split games based on jersey colors using pose detection and color analysis
- **Photo Quality Assessment** - Multi-metric quality analysis and filtering
- **Recursive Processing** - Process directories recursively by default
- **Parallel Processing** - GPU-accelerated processing with multi-threading
- **High-Performance Sidecar Management** - Rust-powered JSON sidecar operations (3-10x faster)
- **Comprehensive CLI** - Clean command-line interface with subcommands

## 📦 Installation

### Basic Installation

```bash
pip install sportball
```

### With GPU Support (Recommended)

```bash
pip install sportball[cuda]
```

### Development Installation

```bash
git clone https://github.com/sportball/sportball.git
cd sportball
pip install -e .[dev]
```

### High-Performance Rust Integration

The Rust sidecar tool is automatically included as a Git dependency from [https://github.com/dapperfu/image-sidecar-rust.git](https://github.com/dapperfu/image-sidecar-rust.git).

**Installation Requirements** (choose one):

#### Option 1: Automatic (Rust Python Package)
If you have Rust installed, the Python package will be built automatically during install:

```bash
# Install sportball (will automatically build Rust package if Rust is available)
pip install -e .  # or pip install sportball
```

#### Option 2: Binary CLI Approach
If you prefer using the pre-built binary:

```bash
# Build the Rust binary separately
cd /path/to/image-sidecar-rust
cargo build --release

# Make it available in PATH
export PATH=$PATH:/path/to/image-sidecar-rust/target/release
```

**Note**: The Rust tool is an **optional** dependency. If neither the Python package nor the binary is available, Sportball will automatically fall back to Python implementations. The Rust tool provides 3-10x performance improvements for sidecar operations.

## 🎯 Quick Start

### Face Detection

```bash
# Detect faces in images (recursive by default)
sportball face detect /path/to/images

# Detect faces with specific confidence threshold
sportball face detect /path/to/images --confidence 0.7

# Process only current directory (disable recursion)
sportball face detect /path/to/images --no-recursive
```

### Object Detection

```bash
# Detect objects in images (recursive by default)
sportball object detect /path/to/images

# Detect specific object classes (including balls)
sportball object detect /path/to/images --classes "person,sports ball"

# Detect only balls
sportball object detect /path/to/images --classes "sports ball"

# Extract detected objects
sportball object extract /path/to/images /path/to/output --object-types "person,sports ball"
```

### Game Splitting

```bash
# Automatically detect and split games
sportball games split /path/to/photos /path/to/games

# Use specific file pattern (e.g., September 2025 photos)
sportball games split /path/to/photos /path/to/games --pattern "202509*_*"

# Add manual split points
sportball games split /path/to/photos /path/to/games --split-file splits.txt

# Split games by jersey colors (requires pose detection)
sportball games split /path/to/photos /path/to/games --split-by-jersey

# Jersey splitting with custom parameters
sportball games split /path/to/photos /path/to/games --split-by-jersey --pose-confidence 0.8 --color-similarity 0.1
```

### Ball Detection (via Object Detection)

```bash
# Detect balls in images (recursive by default)
sportball object detect /path/to/images --classes "sports ball"

# Extract detected balls
sportball object extract /path/to/images /path/to/output --object-types "sports ball"

# Analyze ball detection results
sportball object analyze /path/to/images --classes "sports ball"
```

### Quality Assessment

```bash
# Assess photo quality (recursive by default)
sportball quality assess /path/to/images

# Filter low-quality images
sportball quality assess /path/to/images --filter-low-quality --min-score 0.6

# Process only current directory
sportball quality assess /path/to/images --no-recursive
```

## 🛠️ CLI Commands

### Main Commands

- `sportball face` - Face detection and recognition
- `sportball object` - Object detection and extraction (including balls)
- `sportball games` - Game boundary detection and splitting
- `sportball quality` - Photo quality assessment
- `sportball util` - Utility operations (cache, sidecar management)

### Command Aliases

You can use `sb` as a shorter alias for `sportball`:

```bash
sb face detect /path/to/images
sb object detect /path/to/images --classes "sports ball"
sb object extract /path/to/images /path/to/output
sb games split /path/to/photos /path/to/games
```

### Global Options

- `--gpu/--no-gpu` - Enable/disable GPU acceleration
- `--workers N` - Number of parallel workers
- `--cache/--no-cache` - Enable/disable result caching
- `--verbose` - Enable verbose logging
- `--quiet` - Suppress output except errors
- `--no-recursive` - Disable recursive directory processing (most commands)

## 🔧 Configuration

### GPU Acceleration

Sportball automatically detects and uses GPU acceleration when available. You can control this behavior:

```bash
# Force CPU usage
sportball --no-gpu face detect /path/to/images

# Specify number of workers
sportball --workers 8 face detect /path/to/images
```

### Caching

Results are automatically cached to avoid reprocessing. Cache management:

```bash
# Clear cache
sportball util clear-cache

# Show cache summary
sportball util sidecar-summary /path/to/images
```

## 📊 Sidecar Files

Sportball uses JSON sidecar files to store metadata and results:

- `image_face_detection.json` - Face detection results
- `image_object_detection.json` - Object detection results
- `image_ball_detection.json` - Ball detection results
- `image_quality_assessment.json` - Quality assessment results
- `game_detection.json` - Game boundary detection results

### Sidecar Management

```bash
# Show sidecar summary
sportball util sidecar-summary /path/to/images

# Clean up orphaned sidecar files
sportball util cleanup-sidecars /path/to/images

# Delete sidecar files for specific operation
sportball util delete-sidecars /path/to/images --operation face_detection
```

## 🐍 Python API

You can also use Sportball programmatically:

```python
from sportball import SportballCore
from pathlib import Path

# Initialize core
core = SportballCore(enable_gpu=True, max_workers=4)

# Detect faces
results = core.detect_faces(Path("/path/to/images"))

# Detect objects
results = core.detect_objects(Path("/path/to/images"))

# Detect games
results = core.detect_games(Path("/path/to/photos"))

# Assess quality
results = core.assess_quality(Path("/path/to/images"))
```

## 🎨 Decorators

Sportball provides Pythonic decorators for common operations:

```python
from sportball.decorators import gpu_accelerated, parallel_processing, progress_tracked

@gpu_accelerated(device='cuda:0')
@parallel_processing(max_workers=4)
@progress_tracked(description="Processing images")
def process_images(images):
    # Your processing code here
    pass
```

## 📈 Performance

- **GPU Acceleration** - Automatic CUDA detection and fallback to CPU
- **Parallel Processing** - Multi-threaded and multi-process support
- **Progress Tracking** - Real-time progress bars with tqdm
- **Result Caching** - Avoid reprocessing with intelligent caching
- **Memory Efficient** - Lazy loading and efficient memory management

## 🔍 Examples

### Complete Workflow

```bash
# 1. Detect faces in all photos (recursive)
sportball face detect /path/to/photos --confidence 0.6

# 2. Detect objects (players, balls) - recursive by default
sportball object detect /path/to/photos --classes "person,sports ball"

# 3. Split photos into games
sportball games split /path/to/photos /path/to/games

# 4. Assess photo quality (recursive)
sportball quality assess /path/to/photos --filter-low-quality

# 5. Generate comprehensive report
sportball util sidecar-summary /path/to/photos
```

### Batch Processing

```bash
# Process multiple directories
for dir in /path/to/games/*/; do
    sportball face detect "$dir" --confidence 0.7
    sportball object detect "$dir" --classes "person,sports ball"
done
```

## 🧪 Testing

```bash
# Run tests
pytest

# Run with coverage
pytest --cov=sportball

# Run specific test categories
pytest -m "not slow"  # Skip slow tests
pytest -m cuda        # Only CUDA tests
```

## 📚 Documentation

- [API Reference](https://sportball.readthedocs.io/api/)
- [CLI Reference](https://sportball.readthedocs.io/cli/)
- [Examples](https://sportball.readthedocs.io/examples/)
- [Contributing](https://sportball.readthedocs.io/contributing/)

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- OpenCV for computer vision capabilities
- PyTorch for deep learning framework
- YOLOv8 (Ultralytics) for object detection
- face_recognition for face detection and recognition
- Click for CLI framework
- Rich for beautiful terminal output
- Rust for high-performance sidecar operations
- Rayon for parallel processing

## 📞 Support

- [GitHub Issues](https://github.com/sportball/sportball/issues)
- [Discussions](https://github.com/sportball/sportball/discussions)
- [Email](mailto:support@sportball.ai)

## 🚀 Performance

Sportball now includes high-performance Rust integration for sidecar operations:

- **3-10x faster** JSON validation and processing
- **Massive parallelism** across all CPU cores
- **Automatic fallback** to Python when Rust unavailable
- **Zero code changes** required for existing code

See [RUST_SIDECAR_INTEGRATION.md](RUST_SIDECAR_INTEGRATION.md) for details.

---

**Sportball** - Making sports photo analysis simple, fast, and powerful! 🏈⚽📸
# Version bump to 1.1.0
