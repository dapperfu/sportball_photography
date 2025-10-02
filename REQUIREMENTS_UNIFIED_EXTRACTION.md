# Sportball Unified Extraction Requirements Document

## Document Information
- **Document Type**: Requirements Specification
- **Version**: 1.0
- **Date**: 2025-01-27
- **Author**: Claude Sonnet 4 (claude-3-5-sonnet-20241022)
- **Generated via**: Cursor IDE (cursor.sh) with AI assistance
- **Context**: Shorten extraction commands and use flags to indicate what should be extracted

## Executive Summary

The current extraction commands (`sb face extract` and `sb object extract`) are verbose and require users to understand the internal separation between face and object extraction. This document outlines the requirements to create a unified `sb extract` command that uses flags to indicate what should be extracted, providing a more intuitive and streamlined user experience.

## Problem Statement

### Current State
- Users must use separate commands: `sb face extract` and `sb object extract`
- Commands are verbose and expose internal architecture
- Users must understand the difference between face and object extraction
- No unified extraction command exists
- Command names are not intuitive for end users

### Expected State
- Single `sb extract` command handles all extraction needs
- Flags indicate what should be extracted (`--faces`, `--objects`, `--both`)
- Command is intuitive and hides internal complexity
- Unified extraction provides better user experience
- Backward compatibility maintained for existing workflows

## Functional Requirements

### FR-001: Unified Extraction Command
**Priority**: Critical
**Description**: Create a single `sb extract` command that replaces both `sb face extract` and `sb object extract`.

**Current Commands**:
- `sb face extract <input_path> [--output] [--padding] [--workers] [--no-recursive] [--verbose] [--show-empty-results]`
- `sb object extract <input_path> <output_dir> [--object-types] [--min-size] [--max-size] [--padding] [--no-recursive] [--workers]`

**New Unified Command**:
- `sb extract <input_path> <output_dir> [--faces] [--objects] [--both] [--face-padding] [--object-padding] [--object-types] [--min-size] [--max-size] [--workers] [--no-recursive] [--verbose] [--show-empty-results]`

**Requirements**:
- FR-001.1: `sb extract` must be the primary extraction command
- FR-001.2: Command must accept both input path and output directory as arguments
- FR-001.3: Command must support extraction of faces, objects, or both
- FR-001.4: Command must maintain all existing functionality from individual commands
- FR-001.5: Command must provide clear help text and examples

### FR-002: Extraction Type Flags
**Priority**: Critical
**Description**: Use flags to indicate what should be extracted from images.

**Requirements**:
- FR-002.1: `--faces` flag must extract only faces from images
- FR-002.2: `--objects` flag must extract only objects from images
- FR-002.3: `--both` flag must extract both faces and objects from images
- FR-002.4: Default behavior must be `--both` when no flag is specified
- FR-002.5: Flags must be mutually exclusive (only one can be specified)
- FR-002.6: Invalid flag combinations must show clear error messages

### FR-003: Parameter Consolidation
**Priority**: High
**Description**: Consolidate parameters from both extraction commands into the unified command.

**Current Face Extraction Parameters**:
- `--output` (optional, defaults to `<input>_faces`)
- `--padding` (default: 20px)
- `--workers` (optional)
- `--no-recursive` (flag)
- `--verbose` (count)
- `--show-empty-results` (flag)

**Current Object Extraction Parameters**:
- `output_dir` (required)
- `--object-types` (optional, comma-separated)
- `--min-size` (default: 32px)
- `--max-size` (optional)
- `--padding` (default: 10px)
- `--workers` (optional)
- `--no-recursive` (flag)

**Requirements**:
- FR-003.1: Unified command must accept `output_dir` as required argument
- FR-003.2: Unified command must support `--face-padding` for face extraction (default: 20px)
- FR-003.3: Unified command must support `--object-padding` for object extraction (default: 10px)
- FR-003.4: Unified command must support `--object-types` for object filtering
- FR-003.5: Unified command must support `--min-size` and `--max-size` for object filtering
- FR-003.6: Unified command must support `--workers`, `--no-recursive`, `--verbose`, `--show-empty-results`
- FR-003.7: Parameter names must be clear and unambiguous

### FR-004: Output Directory Structure
**Priority**: High
**Description**: Define clear output directory structure for unified extraction.

**Requirements**:
- FR-004.1: When `--faces` is specified, faces must be saved to `<output_dir>/faces/`
- FR-004.2: When `--objects` is specified, objects must be saved to `<output_dir>/objects/`
- FR-004.3: When `--both` is specified, faces must be saved to `<output_dir>/faces/` and objects to `<output_dir>/objects/`
- FR-004.4: Directory structure must be created automatically
- FR-004.5: Existing files must not be overwritten without explicit confirmation
- FR-004.6: Directory structure must be documented in help text

### FR-005: Backward Compatibility
**Priority**: Medium
**Description**: Maintain backward compatibility with existing extraction commands.

**Requirements**:
- FR-005.1: `sb face extract` command must continue to work unchanged
- FR-005.2: `sb object extract` command must continue to work unchanged
- FR-005.3: Existing scripts and workflows must not break
- FR-005.4: Individual commands must be marked as deprecated in help text
- FR-005.5: Migration guide must be provided for users

### FR-006: Error Handling and Validation
**Priority**: High
**Description**: Provide clear error handling and parameter validation.

**Requirements**:
- FR-006.1: Command must validate that at least one extraction type is specified
- FR-006.2: Command must validate that input path exists and contains images
- FR-006.3: Command must validate that output directory is writable
- FR-006.4: Command must provide clear error messages for invalid parameters
- FR-006.5: Command must handle missing sidecar files gracefully
- FR-006.6: Command must provide helpful suggestions when errors occur

## Non-Functional Requirements

### NFR-001: Performance Consistency
**Priority**: High
**Description**: Unified extraction must maintain performance characteristics of individual commands.

**Requirements**:
- NFR-001.1: Face extraction performance must be identical to `sb face extract`
- NFR-001.2: Object extraction performance must be identical to `sb object extract`
- NFR-001.3: Combined extraction must be efficient and not significantly slower
- NFR-001.4: Memory usage must be similar to individual commands
- NFR-001.5: GPU utilization must be identical to individual commands

### NFR-002: User Experience
**Priority**: High
**Description**: Unified command must provide excellent user experience.

**Requirements**:
- NFR-002.1: Command must be intuitive for new users
- NFR-002.2: Help text must be comprehensive and clear
- NFR-002.3: Examples must be provided for common use cases
- NFR-002.4: Progress indicators must be clear and informative
- NFR-002.5: Output must be well-formatted and easy to understand

### NFR-003: Maintainability
**Priority**: Medium
**Description**: Code must be maintainable and follow DRY principles.

**Requirements**:
- NFR-003.1: Unified extraction must reuse existing extraction logic
- NFR-003.2: No code duplication between unified and individual commands
- NFR-003.3: Single source of truth for extraction logic
- NFR-003.4: Code must be well-documented and tested

## Technical Requirements

### TR-001: Command Implementation
**Priority**: High
**Description**: Implement the unified extraction command in the CLI.

**Requirements**:
- TR-001.1: Command must be implemented in `sportball/cli/commands/unified_commands.py`
- TR-001.2: Command must use Click framework for argument parsing
- TR-001.3: Command must integrate with existing core extraction methods
- TR-001.4: Command must support all existing global options
- TR-001.5: Command must provide comprehensive help text

### TR-002: Core Integration
**Priority**: High
**Description**: Integrate unified extraction with existing core methods.

**Requirements**:
- TR-002.1: Unified command must use `core.extract_faces()` for face extraction
- TR-002.2: Unified command must use `core.extract_objects()` for object extraction
- TR-002.3: Unified command must use `core.extract_unified()` for combined extraction
- TR-002.4: All existing core methods must remain unchanged
- TR-002.5: Core methods must support the unified extraction workflow

### TR-003: Parameter Handling
**Priority**: High
**Description**: Implement proper parameter handling and validation.

**Requirements**:
- TR-003.1: Parameter validation must be implemented in the command
- TR-003.2: Default values must match individual commands
- TR-003.3: Parameter precedence must be clearly defined
- TR-003.4: Invalid parameter combinations must be caught and reported

### TR-004: Output Management
**Priority**: High
**Description**: Implement proper output directory management.

**Requirements**:
- TR-004.1: Output directories must be created automatically
- TR-004.2: Directory structure must be consistent
- TR-004.3: File naming must be consistent with individual commands
- TR-004.4: Progress tracking must work across all extraction types

## Implementation Requirements

### IR-001: CLI Command Implementation
**Priority**: Critical
**Description**: Implement the unified extraction command.

**Implementation**:
- IR-001.1: Add `extract` command to `unified_commands.py`
- IR-001.2: Implement parameter parsing for all extraction options
- IR-001.3: Implement flag validation and mutual exclusivity
- IR-001.4: Implement output directory structure creation
- IR-001.5: Implement error handling and validation

### IR-002: Core Method Integration
**Priority**: Critical
**Description**: Integrate with existing core extraction methods.

**Implementation**:
- IR-002.1: Use `core.extract_faces()` when `--faces` flag is specified
- IR-002.2: Use `core.extract_objects()` when `--objects` flag is specified
- IR-002.3: Use `core.extract_unified()` when `--both` flag is specified
- IR-002.4: Implement proper parameter passing to core methods
- IR-002.5: Implement result aggregation and display

### IR-003: Parameter Consolidation
**Priority**: High
**Description**: Consolidate parameters from individual commands.

**Implementation**:
- IR-003.1: Map `--face-padding` to face extraction padding
- IR-003.2: Map `--object-padding` to object extraction padding
- IR-003.3: Map `--object-types` to object type filtering
- IR-003.4: Map `--min-size` and `--max-size` to object size filtering
- IR-003.5: Map common parameters (`--workers`, `--no-recursive`, etc.)

### IR-004: Output Directory Management
**Priority**: High
**Description**: Implement proper output directory structure.

**Implementation**:
- IR-004.1: Create `<output_dir>/faces/` for face extraction
- IR-004.2: Create `<output_dir>/objects/` for object extraction
- IR-004.3: Create both directories for combined extraction
- IR-004.4: Implement directory creation with proper error handling
- IR-004.5: Implement file conflict resolution

### IR-005: Help and Documentation
**Priority**: Medium
**Description**: Provide comprehensive help and documentation.

**Implementation**:
- IR-005.1: Write comprehensive help text for the command
- IR-005.2: Provide examples for common use cases
- IR-005.3: Document parameter relationships and constraints
- IR-005.4: Add deprecation notices to individual commands
- IR-005.5: Create migration guide for existing users

## Testing Requirements

### TEST-001: Functional Testing
**Priority**: Critical
**Description**: Verify unified extraction produces identical results as individual commands.

**Test Cases**:
- TEST-001.1: Test `sb extract --faces` produces identical results as `sb face extract`
- TEST-001.2: Test `sb extract --objects` produces identical results as `sb object extract`
- TEST-001.3: Test `sb extract --both` produces identical results as running both commands separately
- TEST-001.4: Test default behavior (no flags) extracts both faces and objects
- TEST-001.5: Test parameter validation and error handling
- TEST-001.6: Test output directory structure creation
- TEST-001.7: Test file naming consistency

### TEST-002: Performance Testing
**Priority**: High
**Description**: Verify unified extraction maintains performance characteristics.

**Test Cases**:
- TEST-002.1: Compare face extraction performance between unified and individual commands
- TEST-002.2: Compare object extraction performance between unified and individual commands
- TEST-002.3: Compare combined extraction performance
- TEST-002.4: Test memory usage consistency
- TEST-002.5: Test with large image sets

### TEST-003: Integration Testing
**Priority**: High
**Description**: Verify unified extraction works with other commands.

**Test Cases**:
- TEST-003.1: Test unified extraction with unified detection results
- TEST-003.2: Test unified extraction with individual detection results
- TEST-003.3: Test parameter consistency across commands
- TEST-003.4: Test error handling and recovery

### TEST-004: Backward Compatibility Testing
**Priority**: Medium
**Description**: Verify individual commands continue to work.

**Test Cases**:
- TEST-004.1: Test `sb face extract` still works unchanged
- TEST-004.2: Test `sb object extract` still works unchanged
- TEST-004.3: Test existing scripts continue to work
- TEST-004.4: Test parameter compatibility

## Acceptance Criteria

### AC-001: Command Functionality
- Unified `sb extract` command works for faces, objects, and both
- Command produces identical results as individual commands
- All parameters work correctly and consistently
- Error handling is comprehensive and helpful

### AC-002: User Experience
- Command is intuitive and easy to use
- Help text is comprehensive and clear
- Examples are provided for common use cases
- Progress indicators are informative

### AC-003: Performance
- Face extraction performance matches individual command
- Object extraction performance matches individual command
- Combined extraction is efficient
- Memory usage is consistent

### AC-004: Backward Compatibility
- Individual commands continue to work unchanged
- Existing scripts and workflows are not broken
- Migration path is clear and documented

### AC-005: Output Quality
- Output directory structure is consistent and logical
- File naming is consistent with individual commands
- Results are identical to individual commands
- Progress reporting is accurate

## Risk Assessment

### High Risk
- **Performance Degradation**: If unified extraction is significantly slower, users will avoid it
- **Result Inconsistency**: If results don't match individual commands, it breaks user workflows
- **Parameter Conflicts**: If parameter handling is incorrect, it could cause errors or unexpected behavior

### Medium Risk
- **User Confusion**: If the command interface is not intuitive, users will struggle to adopt it
- **Backward Compatibility**: If individual commands break, existing workflows will fail
- **Output Structure**: If output directory structure is confusing, users will have trouble finding results

### Low Risk
- **Code Complexity**: Internal implementation complexity doesn't affect end users
- **Documentation**: Poor documentation can be improved iteratively
- **Testing**: Comprehensive testing can catch most issues before release

## Success Metrics

1. **Functionality**: 100% feature parity with individual commands
2. **Performance**: Within 5% of individual command performance
3. **User Adoption**: Users prefer unified command over individual commands
4. **Error Rate**: Low error rate with clear error messages
5. **Documentation**: Comprehensive help text and examples

## Migration Strategy

### Phase 1: Implementation
- Implement unified `sb extract` command
- Add comprehensive testing
- Maintain individual commands unchanged

### Phase 2: Documentation
- Add help text and examples
- Create migration guide
- Update documentation

### Phase 3: Deprecation
- Mark individual commands as deprecated
- Provide migration recommendations
- Monitor usage patterns

### Phase 4: Removal (Future)
- Consider removing individual commands in future major version
- Provide clear migration timeline
- Maintain backward compatibility during transition

## Examples

### Basic Usage
```bash
# Extract both faces and objects (default)
sb extract /path/to/images /path/to/output

# Extract only faces
sb extract /path/to/images /path/to/output --faces

# Extract only objects
sb extract /path/to/images /path/to/output --objects

# Extract both explicitly
sb extract /path/to/images /path/to/output --both
```

### Advanced Usage
```bash
# Extract faces with custom padding
sb extract /path/to/images /path/to/output --faces --face-padding 30

# Extract specific object types
sb extract /path/to/images /path/to/output --objects --object-types "person,sports ball"

# Extract with size filtering
sb extract /path/to/images /path/to/output --objects --min-size 50 --max-size 500

# Extract with parallel processing
sb extract /path/to/images /path/to/output --both --workers 8
```

### Output Structure
```
/path/to/output/
├── faces/
│   ├── image1/
│   │   ├── face_001.jpg
│   │   └── face_002.jpg
│   └── image2/
│       └── face_001.jpg
└── objects/
    ├── image1/
    │   ├── person_001.jpg
    │   └── sports_ball_001.jpg
    └── image2/
        └── person_001.jpg
```

## Conclusion

The unified extraction command will provide a more intuitive and streamlined user experience while maintaining full backward compatibility. By using flags to indicate extraction types and consolidating parameters, users can perform all extraction tasks with a single command. The implementation will reuse existing core methods to ensure consistency and performance while providing a cleaner interface for end users.

This requirements document provides a comprehensive roadmap for implementing the unified extraction command, including technical specifications, testing requirements, and migration strategy. The implementation will significantly improve the user experience while maintaining the reliability and performance of the existing extraction functionality.
