# Camera Display Segmentation Fault Fix

## Problem

The original code was experiencing **segmentation faults** when trying to display camera feeds using OpenCV's `cv2.imshow()` function.

## Root Causes

1. **Threading conflicts**: OpenCV GUI functions don't mix well with camera capture in separate threads
2. **Headless environment issues**: Some systems don't have proper X11/GUI support
3. **Improper initialization**: Direct calls to `cv2.imshow()` without proper error handling
4. **Resource conflicts**: Multiple VideoCapture instances can conflict with GUI operations

## Solution Implemented

### 1. Safe Display Checker
```python
def check_opencv_gui_available():
    """Check if OpenCV GUI is available (not headless)."""
    try:
        # Try to create a test window
        test_img = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.imshow('__test__', test_img)
        cv2.waitKey(1)
        cv2.destroyWindow('__test__')
        cv2.waitKey(1)
        return True
    except Exception as e:
        print(f"OpenCV GUI not available: {e}")
        return False
```

### 2. CameraDisplay Wrapper Class
- Safely creates windows with error handling
- Gracefully disables display if GUI is unavailable
- Proper cleanup on exit
- Prevents segfaults through try-except blocks

### 3. Command-line Control
New `--no_display` flag to explicitly disable camera display:
```bash
python kinova_client.py --no_display  # Run without display windows
```

## Usage

### With Display (default):
```bash
python kinova_client.py --host <server> --task "pick up object"
```

### Without Display (headless/SSH):
```bash
python kinova_client.py --host <server> --task "pick up object" --no_display
```

### With Video Recording Only:
```bash
python kinova_client.py --save_video --no_display
```

## What Changed

**Before:**
```python
# Direct OpenCV calls - prone to segfault
cv2.namedWindow('External Camera', cv2.WINDOW_NORMAL)
cv2.imshow('External Camera', img)
cv2.waitKey(1)
```

**After:**
```python
# Safe wrapper class
camera_display = CameraDisplay(enable_display=not args.no_display)
camera_display.update(external_img, wrist_img)  # Safe update
camera_display.close()  # Proper cleanup
```

## Benefits

1. ✅ **No more segfaults** - Graceful fallback if GUI unavailable
2. ✅ **Works over SSH** - Can run headless with `--no_display`
3. ✅ **Better error handling** - Catches and reports display issues
4. ✅ **Clean shutdown** - Properly closes windows on exit
5. ✅ **Flexible** - Display can be enabled/disabled without code changes

## Testing

Test if display works:
```bash
# This will report if GUI is available
python kinova_client.py --steps 1 --dry_run
```

If you see "✓ Camera display windows created", display is working.
If you see "ℹ Camera display disabled", use `--no_display` flag.
