# Unified Game Organizer - Complete Soccer Photo Sorting System

## Overview
A comprehensive system that combines **automated game detection** with **manual splitting capabilities** to accurately organize soccer photos into game-specific folders.

## 🎯 **Key Features**

### 🤖 **Automated Detection**
- Detects game boundaries based on temporal gaps in photo timestamps
- Configurable parameters for game duration and gap thresholds
- Handles edge cases and invalid timestamps gracefully

### ✋ **Manual Splitting**
- Add manual split points at specific timestamps (e.g., "14:00:00")
- Perfect for cases where you ran from one game to another without gaps
- Interactive mode for adding/removing splits
- Save/load split configurations

### 🔗 **Unified Workflow**
- Combines both automated and manual approaches
- Creates organized folders with symlinks or file copies
- Generates comprehensive reports with both detection methods
- Interactive workflow for step-by-step organization

## 📊 **Results from September 20th, 2025**

### **Automated Detection Results**
- **4 games detected** from 7,234 total photos
- **6,457 photos organized** (777 excluded as short sessions)

### **With Manual Split Example**
- **1 manual split** added at 14:00:00
- **5 final games** created
- **Same 6,457 photos** organized

### **Final Game Organization**
1. **Game 1**: 10:21-11:37 (2,050 photos, 75.8 min)
2. **Game 2**: 12:12-13:19 (1,411 photos, 66.4 min)
3. **Game 3**: 13:34-14:00 (521 photos, 25.5 min) ← *Split from original Game 3*
4. **Game 4**: 14:00-14:41 (758 photos, 41.2 min) ← *Split from original Game 3*
5. **Game 5**: 15:32-17:09 (1,717 photos, 97.4 min)

## 🚀 **Usage Options**

### **1. Simple Automated Detection**
```bash
python run_game_detection.py
```

### **2. Manual Splitting Only**
```bash
python manual_game_splitter.py --input /keg/pictures/incoming/2025/09-Sep --interactive
```

### **3. Unified Workflow (Recommended)**
```bash
python run_unified_organizer.py
```

### **4. Interactive Mode**
```bash
python unified_game_organizer.py --input /keg/pictures/incoming/2025/09-Sep --output /path/to/output --interactive
```

## 🎮 **Interactive Commands**

### **Manual Splitting Commands**
- `add <timestamp>` - Add split at timestamp (e.g., "add 14:00:00")
- `remove <timestamp>` - Remove split at timestamp
- `list` - List current splits
- `apply` - Apply splits and show results
- `save` - Save splits to file
- `load <file>` - Load splits from file
- `quit` - Exit interactive mode

### **Timestamp Formats**
- `14:00:00` - Standard time format
- `140000` - Compact format (no colons)

## 📁 **File Structure**

```
/projects/soccer_photo_sorter/
├── game_detector.py              # Core automated detection
├── manual_game_splitter.py       # Manual splitting functionality
├── unified_game_organizer.py    # Complete unified workflow
├── run_game_detection.py         # Simple automated detection
├── run_manual_splitter.py        # Manual splitting demo
├── run_unified_organizer.py     # Unified workflow demo
└── results/
    ├── game_detection/           # Automated detection results
    └── unified_games/           # Unified workflow results
```

## ⚙️ **Configuration Options**

### **GameDetectionConfig Parameters**
- `min_game_duration_minutes`: Minimum game duration (default: 30)
- `min_gap_minutes`: Minimum gap to separate games (default: 10)
- `max_gap_minutes`: Maximum gap within a game (default: 5)
- `min_photos_per_game`: Minimum photos per game (default: 50)

### **Output Options**
- `create_symlinks`: Create symlinks (True) or copy files (False)
- `use_final_games`: Use final games with manual splits (True) or original games (False)

## 📊 **Reports Generated**

### **Automated Detection Report**
- `game_detection_report.json` - Basic automated results

### **Comprehensive Report**
- `comprehensive_game_report.json` - Combined automated + manual results
- Includes both original and final game information
- Manual splits applied and their effects

## 🔧 **Technical Details**

### **Algorithm**
1. **Temporal Analysis**: Analyzes photo timestamps to find natural breaks
2. **Gap Detection**: Identifies significant time gaps between photos
3. **Duration Filtering**: Filters out sessions shorter than minimum duration
4. **Manual Override**: Allows manual insertion of split points
5. **Final Organization**: Creates organized folders with symlinks or copies

### **Dependencies**
- Standard Python library only (pathlib, datetime, json, logging)
- No external dependencies required

### **Performance**
- Processes 7,000+ photos in seconds
- Memory efficient with streaming processing
- Handles large photo collections gracefully

## 💡 **Use Cases**

### **Perfect For**
- Multiple games shot on the same day
- Cases where you ran from one game to another
- Need for precise game boundary control
- Large photo collections requiring organization

### **Example Scenarios**
- **Scenario 1**: 4 distinct games with clear time gaps → Automated detection works perfectly
- **Scenario 2**: Ran from Game 3 to Game 4 without stopping → Add manual split at transition time
- **Scenario 3**: Mixed scenarios → Use unified workflow for best results

## 🎯 **Next Steps**

1. **Review Results**: Check the organized folders and reports
2. **Adjust Parameters**: Modify detection thresholds if needed
3. **Add Manual Splits**: Insert splits for any missed boundaries
4. **Run Additional Analysis**: Apply jersey color detection, face recognition, etc.
5. **Final Organization**: Copy files instead of symlinks when satisfied

## 📝 **Notes**

- **Symlinks**: Created by default for easy review without duplicating files
- **Backup**: Original photos remain untouched in source directory
- **Flexibility**: Can run automated-only, manual-only, or combined workflow
- **Extensibility**: Easy to add new detection methods or analysis features

The unified system provides the best of both worlds: the efficiency of automated detection with the precision of manual control when needed!
