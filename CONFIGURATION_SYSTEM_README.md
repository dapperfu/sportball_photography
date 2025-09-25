# Configuration File System for Game Splitting

## Overview
A comprehensive configuration file system that allows you to generate, edit, and apply manual game splits through JSON configuration files. This provides a user-friendly way to handle manual splitting without interactive commands.

## 🎯 **Key Features**

### 📋 **Configuration File Generation**
- Automatically generates configuration files from automated detection results
- Includes all detection parameters and automated games
- Ready for manual editing to add custom splits

### ✏️ **Manual Editing**
- Edit JSON configuration files with any text editor
- Add manual splits with descriptions, reasons, and confidence levels
- Validate configuration before applying

### 🔧 **Configuration Application**
- Load and apply configurations to create final game organization
- Validation and error checking
- Command-line interface for all operations

## 🚀 **Complete Workflow**

### **Step 1: Generate Configuration**
```bash
python game_split_config.py --input /keg/pictures/incoming/2025/09-Sep --generate --save my_config.json --notes "My custom configuration"
```

### **Step 2: Edit Configuration**
Edit the generated JSON file to add manual splits:
```json
{
  "manual_splits": [
    {
      "timestamp": "14:00:00",
      "description": "Game transition - ran from one game to another",
      "reason": "Photographer moved between games without stopping",
      "confidence": "high"
    },
    {
      "timestamp": "15:30:00",
      "description": "Halftime break",
      "reason": "Clear break in action for halftime",
      "confidence": "medium"
    }
  ]
}
```

### **Step 3: Apply Configuration**
```bash
python game_split_config.py --load my_config.json --apply --summary
```

## 📊 **Configuration File Format**

### **Generated Configuration Structure**
```json
{
  "created_at": "2025-09-24T20:59:33.058142",
  "created_by": "User",
  "source_directory": "/keg/pictures/incoming/2025/09-Sep",
  "file_pattern": "20250920_*",
  "total_photos": 6457,
  "detection_config": {
    "min_game_duration_minutes": 30,
    "max_gap_minutes": 5,
    "min_gap_minutes": 10,
    "min_photos_per_game": 50,
    "max_photos_per_game": 2000
  },
  "automated_games": [
    {
      "game_id": 1,
      "start_time": "10:21:19",
      "end_time": "11:37:04",
      "duration_minutes": 75.8,
      "photo_count": 2050,
      "gap_before_minutes": 17.3,
      "gap_after_minutes": 35.7
    }
  ],
  "manual_splits": [],
  "notes": "Configuration notes",
  "version": "1.0"
}
```

### **Manual Split Format**
```json
{
  "timestamp": "14:00:00",
  "description": "Human-readable description",
  "reason": "Why this split was added",
  "confidence": "high|medium|low"
}
```

## 🎮 **Command-Line Options**

### **Generate Configuration**
```bash
python game_split_config.py --input /path/to/photos --generate --save config.json
```

**Options:**
- `--input, -i`: Input directory containing photos
- `--pattern, -p`: File pattern (default: "20250920_*")
- `--generate, -g`: Generate configuration from automated detection
- `--save, -s`: Save configuration to file
- `--notes`: Add notes to configuration
- `--created-by`: Set creator name

### **Load and Apply Configuration**
```bash
python game_split_config.py --load config.json --apply
```

**Options:**
- `--load, -l`: Load configuration from file
- `--apply, -a`: Apply configuration to create final games
- `--validate, -v`: Validate configuration
- `--summary`: Print configuration summary

## 📁 **File Structure**

```
/projects/soccer_photo_sorter/
├── game_split_config.py          # Core configuration system
├── run_config_workflow.py        # Complete workflow demo
├── demo_config_commands.py       # Command-line examples
└── config_files/
    ├── my_config.json           # Your custom configuration
    └── config_demo.json         # Example configuration
```

## 🔧 **Usage Examples**

### **Example 1: Basic Workflow**
```bash
# Generate configuration
python game_split_config.py --input /keg/pictures/incoming/2025/09-Sep --generate --save config.json

# Edit config.json manually to add splits

# Apply configuration
python game_split_config.py --load config.json --apply --summary
```

### **Example 2: Custom Parameters**
```bash
# Generate with custom pattern and notes
python game_split_config.py --input /keg/pictures/incoming/2025/09-Sep --pattern "20250920_*" --generate --save config.json --notes "Custom configuration" --created-by "John Doe"
```

### **Example 3: Validation**
```bash
# Load and validate configuration
python game_split_config.py --load config.json --validate --summary
```

## 📊 **Real Results Example**

### **Generated Configuration**
- **4 automated games** detected
- **6,457 photos** organized
- **0 manual splits** (ready for editing)

### **After Manual Editing**
- **2 manual splits** added:
  - `14:00:00` - Game transition (high confidence)
  - `15:30:00` - Halftime break (medium confidence)

### **Final Results**
- **5 final games** created
- **Same 6,457 photos** organized
- **Game 3 split** into two parts at 14:00:00

## 💡 **Benefits**

### **User-Friendly**
- No need to remember interactive commands
- Edit configuration files with any text editor
- Clear structure and validation

### **Reproducible**
- Save configurations for reuse
- Version control friendly
- Share configurations with others

### **Flexible**
- Add multiple manual splits
- Include descriptions and reasoning
- Set confidence levels

### **Robust**
- Validation and error checking
- Clear error messages
- Graceful handling of edge cases

## 🎯 **Use Cases**

### **Perfect For**
- Multiple games shot on the same day
- Cases where you ran from one game to another
- Need for precise game boundary control
- Sharing configurations with team members

### **Example Scenarios**
- **Scenario 1**: 4 distinct games → Generate config, no manual splits needed
- **Scenario 2**: Ran from Game 3 to Game 4 → Add manual split at transition time
- **Scenario 3**: Multiple transitions → Add multiple manual splits

## 🚀 **Next Steps**

1. **Generate Configuration**: Run the generate command with your photo directory
2. **Edit Configuration**: Add manual splits to the JSON file
3. **Apply Configuration**: Load and apply the configuration
4. **Review Results**: Check the final game organization
5. **Create Folders**: Use the final games to create organized folders

## 📝 **Notes**

- **Configuration files** are JSON format for easy editing
- **Validation** ensures configuration integrity before application
- **Backup** original configurations before major edits
- **Version control** friendly for team collaboration
- **Extensible** format for future enhancements

The configuration file system provides a professional, user-friendly way to handle manual game splitting with full control and reproducibility!
