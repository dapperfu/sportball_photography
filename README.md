# Soccer Photo Sorter 🏈📸

An AI-powered system for automatically organizing soccer game photographs by detecting game boundaries and organizing photos chronologically.

## Overview

Tired of manually sorting through hundreds of soccer photos? This system automatically detects game boundaries based on photo timestamps and organizes them into chronological game folders, saving you hours of tedious organization work.

## 🎯 **Primary Tool: Enhanced Game Organizer**

The **`enhanced_game_organizer.py`** is the main tool that provides:
- **Multi-day photo processing** - handles photos from multiple dates
- **Automatic game detection** - finds game boundaries based on time gaps
- **Manual split support** - apply custom game boundaries from text files
- **Parallel processing** - fast processing with multiple workers
- **Comprehensive reporting** - detailed JSON reports with game statistics

## 🚀 **Quick Start**

```bash
# Process all photos in a directory
python enhanced_game_organizer.py --input /path/to/photos

# Process specific month (September 2025)
python enhanced_game_organizer.py --input /path/to/photos --pattern "202509*_*"

# With manual splits
python enhanced_game_organizer.py --input /path/to/photos --split-file splits.txt

# High performance with 8 workers
python enhanced_game_organizer.py --input /path/to/photos --workers 8
```

## Features

### 🎨 Jersey Color Detection
- Automatically detects dominant jersey colors in photographs
- Organizes photos into color-coded directories (Red, Blue, Green, etc.)
- Handles multiple colors in a single photo intelligently

### 🔢 Jersey Number Recognition
- Uses OCR technology to read jersey numbers
- Creates directories for each detected number (e.g., "Number_15", "Number_07")
- Perfect for finding all photos of a specific player

### 👤 Face Recognition & Player Grouping
- Detects and groups faces to identify individual players
- Creates player-specific directories (e.g., "Player_A", "Player_B")
- Great for tracking individual players throughout the game

## How It Works

### Basic Usage

```bash
# Sort photos by jersey colors
python jersey_color_sorter.py --input /path/to/photos --output /path/to/sorted

# Sort photos by jersey numbers
python jersey_number_sorter.py --input /path/to/photos --output /path/to/sorted

# Sort photos by player faces
python face_sorter.py --input /path/to/photos --output /path/to/sorted
```

### Advanced Usage

```bash
# Run all sorting methods with custom confidence thresholds
python soccer_photo_sorter.py \
    --input /path/to/photos \
    --output /path/to/sorted \
    --color-confidence 0.8 \
    --number-confidence 0.7 \
    --face-confidence 0.75 \
    --threads 4
```

## Installation

### Prerequisites
- Python 3.8 or higher
- Tesseract OCR engine
- OpenCV and computer vision libraries
- **CUDA Support (Optional but Recommended)**:
  - NVIDIA GPU with CUDA Compute Capability 6.0+
  - CUDA Toolkit 11.0 or higher
  - cuDNN library

### Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd soccer_photo_sorter
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install Tesseract OCR**
   - **Ubuntu/Debian**: `sudo apt-get install tesseract-ocr`
   - **macOS**: `brew install tesseract`
   - **Windows**: Download from [GitHub releases](https://github.com/UB-Mannheim/tesseract/wiki)

5. **Install CUDA Support (Optional)**
   - **Install CUDA Toolkit**: Download from [NVIDIA Developer](https://developer.nvidia.com/cuda-downloads)
   - **Install cuDNN**: Download from [NVIDIA cuDNN](https://developer.nvidia.com/cudnn)
   - **Verify Installation**:
     ```bash
     python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
     python -c "import cv2; print(f'OpenCV CUDA: {cv2.cuda.getCudaEnabledDeviceCount()}')"
     ```

## Directory Structure

After processing, your photos will be organized like this:

```
sorted_photos/
├── by_color/
│   ├── Red/
│   ├── Blue/
│   ├── Green/
│   └── Yellow/
├── by_number/
│   ├── Number_01/
│   ├── Number_07/
│   ├── Number_15/
│   └── Number_23/
└── by_player/
    ├── Player_A/
    ├── Player_B/
    └── Player_C/
```

## Configuration

### Custom Color Categories

Create a `config.json` file to customize color detection:

```json
{
    "colors": {
        "red": {"rgb_range": [200, 0, 0], "tolerance": 50},
        "blue": {"rgb_range": [0, 0, 200], "tolerance": 50},
        "green": {"rgb_range": [0, 200, 0], "tolerance": 50}
    },
    "confidence_thresholds": {
        "color": 0.8,
        "number": 0.7,
        "face": 0.75
    }
}
```

### Processing Options

- `--threads N`: Number of parallel processing threads
- `--confidence X`: Minimum confidence threshold (0.0-1.0)
- `--preserve-structure`: Keep original directory structure
- `--dry-run`: Preview changes without creating directories
- `--use-cuda`: Force CUDA acceleration (if available)
- `--cpu-only`: Disable CUDA and use CPU only
- `--gpu-memory-limit N`: Limit GPU memory usage (in GB)

## Performance

### CPU Processing
- **Processing Speed**: ~1000 photos in 20-30 minutes
- **Memory Usage**: Typically under 2GB RAM

### CUDA Acceleration (Recommended)
- **Processing Speed**: ~1000 photos in 5-10 minutes
- **Memory Usage**: Efficient GPU memory utilization
- **Speed Improvement**: 3-5x faster than CPU-only processing
- **GPU Requirements**: NVIDIA GPU with CUDA Compute Capability 6.0+

### General Specifications
- **Supported Formats**: JPEG, PNG, TIFF, RAW
- **Image Size**: 100x100 to 8000x8000 pixels

## Accuracy Expectations

- **Jersey Colors**: ~85% accuracy on clear photos
- **Jersey Numbers**: ~75% accuracy on readable numbers
- **Face Recognition**: ~70% accuracy for consistent lighting

## Troubleshooting

### Common Issues

**"No faces detected"**
- Ensure faces are clearly visible and well-lit
- Try lowering the face confidence threshold

**"Jersey numbers not recognized"**
- Check that numbers are clearly visible
- Ensure Tesseract OCR is properly installed
- Try different OCR preprocessing options

**"Colors misclassified"**
- Adjust color tolerance in configuration
- Check for unusual lighting conditions
- Verify jersey colors are distinct from background

**"CUDA not detected"**
- Verify CUDA Toolkit installation: `nvcc --version`
- Check GPU compatibility: `nvidia-smi`
- Ensure PyTorch/TensorFlow CUDA versions match your CUDA installation
- Try running with `--cpu-only` flag as fallback

### Getting Help

1. Check the logs in `processing_logs/` directory
2. Run with `--verbose` flag for detailed output
3. Use `--dry-run` to preview changes before processing

## Examples

### Example 1: Basic Color Sorting
```bash
python jersey_color_sorter.py \
    --input ./soccer_game_photos \
    --output ./organized_photos \
    --verbose
```

### Example 2: Find All Photos of Player #15
```bash
python jersey_number_sorter.py \
    --input ./soccer_game_photos \
    --output ./player_15_photos \
    --number-filter 15
```

### Example 3: Complete Processing Pipeline with CUDA
```bash
python soccer_photo_sorter.py \
    --input ./soccer_game_photos \
    --output ./fully_organized \
    --all-methods \
    --use-cuda \
    --threads 6 \
    --confidence 0.8
```

### Example 4: Multi-GPU Processing
```bash
python soccer_photo_sorter.py \
    --input ./soccer_game_photos \
    --output ./fully_organized \
    --all-methods \
    --use-cuda \
    --gpu-memory-limit 8 \
    --threads 8
```

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- OpenCV for computer vision capabilities
- Tesseract OCR for text recognition
- face_recognition library for face detection
- The soccer photography community for inspiration

---

**Note**: This system is designed for personal use and hobbyist photographers. For commercial applications, please ensure compliance with relevant privacy and data protection regulations.
