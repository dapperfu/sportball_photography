# Sportball Unified Detection Requirements Document

## Document Information
- **Document Type**: Requirements Specification
- **Version**: 1.0
- **Date**: 2025-10-01
- **Author**: Claude Sonnet 4 (claude-3-5-sonnet-20241022)
- **Generated via**: Cursor IDE (cursor.sh) with AI assistance
- **Context**: Fix unified detection commands to work identically to individual detection commands

## Executive Summary

The unified detection commands (`sb detect`, `sb extract`, `sb annotate`) are currently broken and do not work the same as the individual detection commands (`sb face detect`, `sb object detect`). This document outlines the requirements to fix these commands so they provide identical functionality with identical backend workflows, GPU acceleration, parallel processing, and output formats.

## Problem Statement

### Current State
- `sb detect` shows "Faces detected: 0" and "Objects detected: 0" when individual commands find faces and objects
- `sb detect` creates no sidecar files while individual commands create proper JSON sidecar files
- `sb extract` and `sb annotate` likely have similar issues
- End users are confused by the separation of `sb face` and `sb object` commands

### Expected State
- `sb detect` should find identical faces and objects as `sb face detect` and `sb object detect`
- `sb detect` should create identical sidecar files with both face and object data
- All unified commands should use identical backend workflows, GPU acceleration, and parallel processing
- Commands should be intuitive and not require users to understand internal separation

## Functional Requirements

### FR-001: Face Detection Parameter Consistency
**Priority**: Critical
**Description**: The unified detection must use identical face detection parameters as the face detection command.

**Current Issue**: 
- Face detection command uses: `confidence=0.5`, `min_faces=0`, `face_size=64`
- Unified detection uses: `confidence=0.5`, `border_padding=0.25` (invalid parameter), missing `min_faces` and `face_size`

**Requirements**:
- FR-001.1: Unified detection must pass `min_faces=0` parameter to face detection
- FR-001.2: Unified detection must pass `face_size=64` parameter to face detection
- FR-001.3: Unified detection must remove invalid `border_padding` parameter from face detection calls
- FR-001.4: Unified detection must use identical confidence threshold (0.5) as face detection command

### FR-002: Object Detection Method Consistency
**Priority**: Critical
**Description**: The unified detection must use identical object detection methods and parameters as the object detection command.

**Current Issue**:
- Object detection command uses: `detect_objects()` batch method with `confidence`, `classes`, `save_sidecar`, `force` parameters
- Unified detection uses: `detect_objects_in_image()` single image method with only `image_path` and `force` parameters

**Requirements**:
- FR-002.1: Unified detection must use `detect_objects()` batch method instead of `detect_objects_in_image()`
- FR-002.2: Unified detection must pass `confidence` parameter to object detection
- FR-002.3: Unified detection must pass `classes` parameter to object detection
- FR-002.4: Unified detection must pass `save_sidecar` parameter to object detection
- FR-002.5: Unified detection must pass `force` parameter to object detection

### FR-003: Sidecar File Format Consistency
**Priority**: Critical
**Description**: The unified detection must create sidecar files with identical structure and content as individual detection commands.

**Current Issue**:
- Face detection creates files with `["face_detection", "sidecar_info"]` keys
- Object detection creates files with `["sidecar_info", "yolov8"]` keys
- Unified detection creates NO sidecar files

**Requirements**:
- FR-003.1: Unified detection must create sidecar files with both `face_detection` AND `yolov8` data
- FR-003.2: Face detection data must be stored under `face_detection` key with identical structure
- FR-003.3: Object detection data must be stored under `yolov8` key with identical structure
- FR-003.4: Sidecar files must include `sidecar_info` metadata
- FR-003.5: All metadata fields must match individual detection commands exactly

### FR-004: CLI Display Accuracy
**Priority**: High
**Description**: The unified detection CLI must display accurate counts and results.

**Current Issue**:
- Unified detection shows "Faces detected: 0" when faces are actually found
- Unified detection shows "Objects detected: 0" when objects are actually found
- CLI display logic incorrectly counts results

**Requirements**:
- FR-004.1: CLI must display correct face count from actual detection results
- FR-004.2: CLI must display correct object count from actual detection results
- FR-004.3: CLI must display correct processing statistics (time, speed, etc.)
- FR-004.4: CLI must show detailed results per image when verbose mode is enabled

### FR-005: Backend Workflow Consistency
**Priority**: High
**Description**: The unified detection must use identical backend workflows as individual detection commands.

**Current Issue**:
- Individual commands use optimized batch processing
- Unified detection processes images one by one in a loop
- Different GPU acceleration and parallel processing strategies

**Requirements**:
- FR-005.1: Unified detection must use identical batch processing as individual commands
- FR-005.2: Unified detection must use identical GPU acceleration strategies
- FR-005.3: Unified detection must use identical parallel processing strategies
- FR-005.4: Unified detection must use identical progress tracking and error handling
- FR-005.5: Unified detection must achieve similar processing speeds as individual commands

### FR-006: Command Interface Simplification
**Priority**: Medium
**Description**: The command interface should be simplified to hide internal complexity from end users.

**Current Issue**:
- Users must understand `sb face` vs `sb object` separation
- Commands are confusing and not intuitive
- Internal architecture is exposed to end users

**Requirements**:
- FR-006.1: `sb detect` must be the primary detection command
- FR-006.2: `sb face detect` and `sb object detect` must be deprecated or hidden
- FR-006.3: `sb extract` must work with unified detection results
- FR-006.4: `sb annotate` must work with unified detection results
- FR-006.5: All commands must have consistent parameter interfaces

## Non-Functional Requirements

### NFR-001: Performance Consistency
**Priority**: High
**Description**: Unified detection must achieve similar performance characteristics as individual commands.

**Requirements**:
- NFR-001.1: Processing speed must be within 10% of individual command speeds
- NFR-001.2: Memory usage must be similar to individual commands
- NFR-001.3: GPU utilization must be identical to individual commands

### NFR-002: Reliability
**Priority**: High
**Description**: Unified detection must be as reliable as individual commands.

**Requirements**:
- NFR-002.1: Error handling must be identical to individual commands
- NFR-002.2: Graceful shutdown must work identically
- NFR-002.3: Recovery from failures must be identical

### NFR-003: Maintainability
**Priority**: Medium
**Description**: Code must be maintainable and follow DRY principles.

**Requirements**:
- NFR-003.1: Unified detection must reuse identical code paths as individual commands
- NFR-003.2: No code duplication between unified and individual commands
- NFR-003.3: Single source of truth for detection logic

## Technical Requirements

### TR-001: Code Architecture
**Priority**: High
**Description**: Unified detection must use identical code architecture as individual commands.

**Requirements**:
- TR-001.1: Unified detection must call `core.detect_faces()` with identical parameters
- TR-001.2: Unified detection must call `core.detect_objects()` with identical parameters
- TR-001.3: Unified detection must use identical detector instances
- TR-001.4: Unified detection must use identical sidecar management

### TR-002: Parameter Handling
**Priority**: High
**Description**: Parameter handling must be consistent across all commands.

**Requirements**:
- TR-002.1: All commands must accept identical parameter sets
- TR-002.2: Parameter validation must be identical
- TR-002.3: Default values must be identical
- TR-002.4: Parameter precedence must be identical

### TR-003: Error Handling
**Priority**: High
**Description**: Error handling must be identical across all commands.

**Requirements**:
- TR-003.1: Error messages must be identical
- TR-003.2: Error recovery must be identical
- TR-003.3: Logging must be identical
- TR-003.4: Exit codes must be identical

## Implementation Requirements

### IR-001: Face Detection Fix
**Priority**: Critical
**Description**: Fix face detection in unified detection to use identical parameters.

**Implementation**:
- IR-001.1: Modify `_process_single_image_unified()` to pass `min_faces=0` and `face_size=64`
- IR-001.2: Remove invalid `border_padding` parameter from face detection calls
- IR-001.3: Use identical confidence threshold (0.5)
- IR-001.4: Ensure face detection uses identical detector instance

### IR-002: Object Detection Fix
**Priority**: Critical
**Description**: Fix object detection in unified detection to use identical methods.

**Implementation**:
- IR-002.1: Replace `detect_objects_in_image()` with `detect_objects()` batch method
- IR-002.2: Pass `confidence`, `classes`, `save_sidecar`, `force` parameters
- IR-002.3: Use identical detector instance as object detection command
- IR-002.4: Ensure identical parameter handling

### IR-003: Sidecar File Fix
**Priority**: Critical
**Description**: Fix sidecar file creation to match individual commands.

**Implementation**:
- IR-003.1: Ensure face detection data is saved under `face_detection` key
- IR-003.2: Ensure object detection data is saved under `yolov8` key
- IR-003.3: Include `sidecar_info` metadata
- IR-003.4: Use identical sidecar manager as individual commands

### IR-004: CLI Display Fix
**Priority**: High
**Description**: Fix CLI display to show accurate results.

**Implementation**:
- IR-004.1: Fix `display_unified_results()` to count actual results
- IR-004.2: Ensure face count matches actual detected faces
- IR-004.3: Ensure object count matches actual detected objects
- IR-004.4: Display detailed results per image

### IR-005: Command Interface Fix
**Priority**: Medium
**Description**: Simplify command interface for end users.

**Implementation**:
- IR-005.1: Make `sb detect` the primary command
- IR-005.2: Deprecate or hide `sb face detect` and `sb object detect`
- IR-005.3: Ensure `sb extract` works with unified results
- IR-005.4: Ensure `sb annotate` works with unified results

## Testing Requirements

### TEST-001: Functional Testing
**Priority**: Critical
**Description**: Verify unified detection produces identical results as individual commands.

**Test Cases**:
- TEST-001.1: Test with images containing faces only
- TEST-001.2: Test with images containing objects only
- TEST-001.3: Test with images containing both faces and objects
- TEST-001.4: Test with images containing neither faces nor objects
- TEST-001.5: Verify identical face counts between commands
- TEST-001.6: Verify identical object counts between commands
- TEST-001.7: Verify identical sidecar file structure
- TEST-001.8: Verify identical sidecar file content

### TEST-002: Performance Testing
**Priority**: High
**Description**: Verify unified detection achieves similar performance as individual commands.

**Test Cases**:
- TEST-002.1: Compare processing speeds
- TEST-002.2: Compare memory usage
- TEST-002.3: Compare GPU utilization
- TEST-002.4: Test with large image sets

### TEST-003: Integration Testing
**Priority**: High
**Description**: Verify unified detection works with other commands.

**Test Cases**:
- TEST-003.1: Test `sb extract` with unified detection results
- TEST-003.2: Test `sb annotate` with unified detection results
- TEST-003.3: Test parameter consistency across commands

## Acceptance Criteria

### AC-001: Detection Accuracy
- Unified detection must find identical faces as face detection command
- Unified detection must find identical objects as object detection command
- Detection results must be 100% consistent across commands

### AC-002: Sidecar File Consistency
- Unified detection must create sidecar files with identical structure
- Sidecar files must contain identical data as individual commands
- All metadata must match exactly

### AC-003: CLI Display Accuracy
- CLI must display correct face and object counts
- CLI must display accurate processing statistics
- CLI must show detailed results when requested

### AC-004: Performance Consistency
- Processing speed must be within 10% of individual commands
- Memory usage must be similar to individual commands
- GPU utilization must be identical

### AC-005: Command Interface
- `sb detect` must be the primary detection command
- All commands must have consistent parameter interfaces
- Commands must be intuitive for end users

## Risk Assessment

### High Risk
- **Detection Accuracy**: If unified detection doesn't find identical results, it breaks user workflows
- **Sidecar File Format**: If sidecar files don't match, downstream tools will fail
- **Performance**: If performance is significantly worse, users will avoid unified commands

### Medium Risk
- **CLI Display**: Incorrect display confuses users but doesn't break functionality
- **Command Interface**: Interface changes may confuse existing users

### Low Risk
- **Code Architecture**: Internal changes don't affect end users
- **Error Handling**: Error handling improvements are beneficial

## Success Metrics

1. **Detection Accuracy**: 100% consistency between unified and individual commands
2. **Sidecar File Consistency**: 100% identical structure and content
3. **Performance**: Within 10% of individual command performance
4. **User Satisfaction**: Commands are intuitive and work as expected
5. **Code Quality**: No duplication, single source of truth for detection logic

## Conclusion

The unified detection commands are currently broken and do not work the same as individual detection commands. This requirements document outlines the necessary fixes to ensure identical functionality, performance, and user experience. The primary issues are incorrect parameter passing, wrong method usage, broken sidecar file creation, and inaccurate CLI display. Fixing these issues will provide users with a unified, intuitive interface while maintaining the reliability and performance of the individual commands.
