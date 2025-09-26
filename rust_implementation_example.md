# Rust Implementation Example

This document provides an example of how to implement the Rust performance module for sportball detection operations.

## Overview

The Rust implementation provides massively parallel JSON validation and file processing capabilities that can significantly outperform Python implementations for large-scale operations.

## Implementation Structure

```
sportball-rust/
├── Cargo.toml
├── src/
│   ├── main.rs
│   ├── lib.rs
│   ├── validation/
│   │   ├── mod.rs
│   │   └── json_validator.rs
│   ├── processing/
│   │   ├── mod.rs
│   │   └── file_processor.rs
│   └── utils/
│       ├── mod.rs
│       └── parallel.rs
└── README.md
```

## Cargo.toml

```toml
[package]
name = "sportball-rust"
version = "0.1.0"
edition = "2021"

[dependencies]
tokio = { version = "1.0", features = ["full"] }
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
rayon = "1.7"
clap = { version = "4.0", features = ["derive"] }
anyhow = "1.0"
thiserror = "1.0"
tracing = "0.1"
tracing-subscriber = "0.3"
```

## Main Implementation (src/main.rs)

```rust
use clap::{Parser, Subcommand};
use sportball_rust::validation::JsonValidator;
use sportball_rust::processing::FileProcessor;
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "sportball-rust")]
#[command(about = "High-performance Rust implementation for sportball detection operations")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Validate JSON files in parallel
    ValidateJson {
        /// Input file containing list of JSON files to validate
        #[arg(short, long)]
        input: PathBuf,
        
        /// Output file (use '-' for stdout)
        #[arg(short, long, default_value = "-")]
        output: String,
        
        /// Number of parallel workers
        #[arg(short, long, default_value = "16")]
        workers: usize,
        
        /// Operation type filter
        #[arg(long)]
        operation_type: Option<String>,
    },
    
    /// Process files in parallel
    ProcessFiles {
        /// Input file containing list of files to process
        #[arg(short, long)]
        input: PathBuf,
        
        /// Output file (use '-' for stdout)
        #[arg(short, long, default_value = "-")]
        output: String,
        
        /// Number of parallel workers
        #[arg(short, long, default_value = "16")]
        workers: usize,
        
        /// Processing operation
        #[arg(short, long)]
        operation: String,
    },
    
    /// Analyze images in parallel
    AnalyzeImages {
        /// Input file containing list of image files to analyze
        #[arg(short, long)]
        input: PathBuf,
        
        /// Output file (use '-' for stdout)
        #[arg(short, long, default_value = "-")]
        output: String,
        
        /// Analysis type
        #[arg(short, long)]
        analysis: String,
        
        /// Number of parallel workers
        #[arg(short, long, default_value = "16")]
        workers: usize,
    },
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();
    
    let cli = Cli::parse();
    
    match cli.command {
        Commands::ValidateJson { input, output, workers, operation_type } => {
            let validator = JsonValidator::new(workers);
            let results = validator.validate_files_parallel(&input, operation_type).await?;
            
            if output == "-" {
                for result in results {
                    println!("{}", serde_json::to_string(&result)?);
                }
            } else {
                let output_file = std::fs::File::create(&output)?;
                serde_json::to_writer(output_file, &results)?;
            }
        }
        
        Commands::ProcessFiles { input, output, workers, operation } => {
            let processor = FileProcessor::new(workers);
            let results = processor.process_files_parallel(&input, &operation).await?;
            
            if output == "-" {
                for result in results {
                    println!("{}", serde_json::to_string(&result)?);
                }
            } else {
                let output_file = std::fs::File::create(&output)?;
                serde_json::to_writer(output_file, &results)?;
            }
        }
        
        Commands::AnalyzeImages { input, output, analysis, workers } => {
            let processor = FileProcessor::new(workers);
            let results = processor.analyze_images_parallel(&input, &analysis).await?;
            
            if output == "-" {
                for result in results {
                    println!("{}", serde_json::to_string(&result)?);
                }
            } else {
                let output_file = std::fs::File::create(&output)?;
                serde_json::to_writer(output_file, &results)?;
            }
        }
    }
    
    Ok(())
}
```

## JSON Validator (src/validation/json_validator.rs)

```rust
use anyhow::Result;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use tokio::fs as async_fs;
use tokio::io::AsyncBufReadExt;
use tokio::io::BufReader;

#[derive(Debug, Serialize, Deserialize)]
pub struct ValidationResult {
    pub file_path: String,
    pub is_valid: bool,
    pub error: Option<String>,
    pub processing_time: f64,
    pub file_size: u64,
    pub detection_count: u32,
    pub tool_name: Option<String>,
}

pub struct JsonValidator {
    workers: usize,
}

impl JsonValidator {
    pub fn new(workers: usize) -> Self {
        Self { workers }
    }
    
    pub async fn validate_files_parallel(
        &self,
        input_file: &Path,
        operation_type: Option<String>,
    ) -> Result<Vec<ValidationResult>> {
        // Read file paths from input file
        let file_paths = self.read_file_paths(input_file).await?;
        
        // Filter by operation type if specified
        let filtered_paths = if let Some(op_type) = operation_type {
            self.filter_by_operation_type(&file_paths, &op_type).await?
        } else {
            file_paths
        };
        
        // Validate files in parallel using rayon
        let results: Vec<ValidationResult> = filtered_paths
            .par_iter()
            .map(|path| self.validate_single_file(path))
            .collect();
        
        Ok(results)
    }
    
    async fn read_file_paths(&self, input_file: &Path) -> Result<Vec<String>> {
        let file = async_fs::File::open(input_file).await?;
        let reader = BufReader::new(file);
        let mut lines = reader.lines();
        let mut paths = Vec::new();
        
        while let Some(line) = lines.next_line().await? {
            let path = line.trim();
            if !path.is_empty() {
                paths.push(path.to_string());
            }
        }
        
        Ok(paths)
    }
    
    async fn filter_by_operation_type(
        &self,
        file_paths: &[String],
        operation_type: &str,
    ) -> Result<Vec<String>> {
        let filtered: Vec<String> = file_paths
            .par_iter()
            .filter(|path| {
                if let Ok(contents) = fs::read_to_string(path) {
                    if let Ok(data) = serde_json::from_str::<serde_json::Value>(&contents) {
                        self.contains_operation_type(&data, operation_type)
                    } else {
                        true // Include files that can't be parsed for validation
                    }
                } else {
                    true // Include files that can't be read for validation
                }
            })
            .cloned()
            .collect();
        
        Ok(filtered)
    }
    
    fn validate_single_file(&self, file_path: &str) -> ValidationResult {
        let start_time = std::time::Instant::now();
        
        match fs::metadata(file_path) {
            Ok(metadata) => {
                let file_size = metadata.len();
                
                match fs::read_to_string(file_path) {
                    Ok(contents) => {
                        match serde_json::from_str::<serde_json::Value>(&contents) {
                            Ok(data) => {
                                let processing_time = start_time.elapsed().as_secs_f64();
                                let detection_count = self.extract_detection_count(&data);
                                let tool_name = self.extract_tool_name(&data);
                                
                                ValidationResult {
                                    file_path: file_path.to_string(),
                                    is_valid: true,
                                    error: None,
                                    processing_time,
                                    file_size,
                                    detection_count,
                                    tool_name,
                                }
                            }
                            Err(e) => ValidationResult {
                                file_path: file_path.to_string(),
                                is_valid: false,
                                error: Some(format!("JSON decode error: {}", e)),
                                processing_time: start_time.elapsed().as_secs_f64(),
                                file_size,
                                detection_count: 0,
                                tool_name: None,
                            },
                        }
                    }
                    Err(e) => ValidationResult {
                        file_path: file_path.to_string(),
                        is_valid: false,
                        error: Some(format!("File read error: {}", e)),
                        processing_time: start_time.elapsed().as_secs_f64(),
                        file_size: 0,
                        detection_count: 0,
                        tool_name: None,
                    },
                }
            }
            Err(_) => ValidationResult {
                file_path: file_path.to_string(),
                is_valid: false,
                error: Some("File does not exist".to_string()),
                processing_time: start_time.elapsed().as_secs_f64(),
                file_size: 0,
                detection_count: 0,
                tool_name: None,
            },
        }
    }
    
    fn extract_detection_count(&self, data: &serde_json::Value) -> u32 {
        // Try common detection count fields
        if let Some(count) = data.get("count").and_then(|v| v.as_u64()) {
            return count as u32;
        }
        
        // Check for arrays of detections
        for key in &["faces", "objects", "detections"] {
            if let Some(array) = data.get(key).and_then(|v| v.as_array()) {
                return array.len() as u32;
            }
        }
        
        // Check nested structures
        for key in &["data", "result", "detection"] {
            if let Some(nested) = data.get(key) {
                let nested_count = self.extract_detection_count(nested);
                if nested_count > 0 {
                    return nested_count;
                }
            }
        }
        
        0
    }
    
    fn extract_tool_name(&self, data: &serde_json::Value) -> Option<String> {
        // Try common tool name fields
        for key in &["tool_name", "detector", "model", "algorithm"] {
            if let Some(name) = data.get(key).and_then(|v| v.as_str()) {
                return Some(name.to_string());
            }
        }
        
        // Check nested structures
        for key in &["data", "result", "metadata"] {
            if let Some(nested) = data.get(key) {
                if let Some(name) = self.extract_tool_name(nested) {
                    return Some(name);
                }
            }
        }
        
        None
    }
    
    fn contains_operation_type(&self, data: &serde_json::Value, operation_type: &str) -> bool {
        // Check direct keys
        if data.get(operation_type).is_some() {
            return true;
        }
        
        // Check sidecar_info structure
        if let Some(sidecar_info) = data.get("sidecar_info") {
            if let Some(op_type) = sidecar_info.get("operation_type").and_then(|v| v.as_str()) {
                if op_type == operation_type {
                    return true;
                }
            }
        }
        
        // Check nested structures
        for key in &["data", "result"] {
            if let Some(nested) = data.get(key) {
                if self.contains_operation_type(nested, operation_type) {
                    return true;
                }
            }
        }
        
        false
    }
}
```

## Usage Examples

### Building the Rust Binary

```bash
cd sportball-rust
cargo build --release
```

### Using the Rust Implementation

```bash
# Validate JSON files in parallel
echo -e "/path/to/file1.json\n/path/to/file2.json" > files.txt
./target/release/sportball-rust validate-json --input files.txt --workers 32

# Process files with specific operation
./target/release/sportball-rust process-files --input files.txt --operation "face_detection" --workers 16

# Analyze images
./target/release/sportball-rust analyze-images --input images.txt --analysis "quality_assessment" --workers 8
```

## Performance Benefits

The Rust implementation provides significant performance improvements:

1. **Massive Parallelism**: Uses rayon for data parallelism across CPU cores
2. **Zero-Copy Operations**: Minimizes memory allocations and copying
3. **Efficient I/O**: Uses async I/O for better throughput
4. **Memory Safety**: Compile-time guarantees prevent common errors
5. **SIMD Optimizations**: Can leverage CPU vector instructions

## Integration with Python

The Python `RustPerformanceModule` automatically detects and uses the Rust binary when available, falling back to Python implementations when not. This provides the best of both worlds: high performance when available, reliability always.

## Future Enhancements

1. **GPU Acceleration**: Add CUDA/OpenCL support for even higher performance
2. **Streaming Processing**: Process files as streams for memory efficiency
3. **Custom Plugins**: Allow custom validation and processing plugins
4. **Distributed Processing**: Scale across multiple machines
5. **Real-time Processing**: Support for real-time detection pipelines
