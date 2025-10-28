# CRITICAL REQUIREMENTS FOR IMAGE-SIDECAR-RUST PROJECT

## Status: BLOCKING sportball functionality

The sportball project **CANNOT** read sidecar files until these requirements are implemented.

**Python fallback is PROHIBITED per requirements.**

---

## RUST-019: Python Bindings Read Method (CRITICAL - BLOCKING)

### What's Missing

The `ImageSidecar` Python bindings **DO NOT HAVE** a `read_data()` method.

### What's Required

```python
def read_data(image_path: str) -> dict:
    """
    Read sidecar data for an image file.
    
    Args:
        image_path: Path to image (e.g., "/path/to/image.jpg")
        
    Returns:
        dict: Full sidecar data including all operations
              Returns {} if no sidecar found (does NOT raise error)
    """
```

### Required Functionality

1. **Accept `image_path` as string** - e.g., "/path/to/image.jpg"

2. **Auto-detect sidecar format** - Check all extensions in priority order:
   - `.bin` (binary format - preferred)
   - `.rkyv` (Rkyv format)
   - `.json` (JSON format)

3. **Handle symlinks** - If image is a symlink:
   - Resolve symlink to actual image
   - Look for sidecar next to actual image
   - Return symlink info in metadata

4. **Return all operations** - Sidecar may contain multiple operations:
   ```python
   {
       "sidecar_info": {...},
       "data": {
           "face_detection": {...},
           "yolov8": {...},
           "object_detection": {...}
       }
   }
   ```

5. **Deserialize formats**:
   - Binary (`.bin`) → deserialize to Python dict
   - JSON (`.json`) → parse and return as dict
   - Rkyv (`.rkyv`) → deserialize to Python dict

6. **Return empty dict on missing sidecar** - Do NOT raise error if sidecar doesn't exist
   - This allows sportball to check if extraction is needed

### Why This Is Blocking

**sportball** uses this to:
- Extract detected faces from sidecar files
- Extract detected objects from sidecar files
- Read existing detection results for filtering/processing

**Without `read_data()`**:
- `sb extract --faces` **DOES NOT WORK**
- `sb extract --objects` **DOES NOT WORK**
- All extraction workflows are **BROKEN**

**Python fallback is PROHIBITED** per requirements TR-008.3.

---

## Other Missing Requirements

See `requirements/REQUIREMENTS_RUST_SIDECAR_IMPLEMENTATION.sdoc` for complete list:

- RUST-001: save_data() signature fix
- RUST-004: Binary format as default
- RUST-007: Data merge functionality
- RUST-010: Read existing JSON files
- RUST-011: Sidecar metadata preservation
- RUST-014: Thread safety
- RUST-015: Python integration tests
- RUST-016: Automatic format migration
- RUST-017: File discovery method
- RUST-018: Nested backend support

---

## Complete Requirements Document

See: `requirements/REQUIREMENTS_RUST_SIDECAR_IMPLEMENTATION.sdoc`

This document contains:
- 20 detailed requirements (RUST-001 through RUST-020)
- Exact Python API specifications
- Implementation requirements
- Testing requirements
- Priority levels (Critical/High/Medium)
- Status tracking

---

## What to Do

1. Read `requirements/REQUIREMENTS_RUST_SIDECAR_IMPLEMENTATION.sdoc`
2. Implement `read_data()` method per RUST-019
3. Implement other Critical requirements
4. Add Python integration tests
5. Publish new version to PyPI/GitHub

---

## Contact

For questions, contact the sportball team with requirements document.

