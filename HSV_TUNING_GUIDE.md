# HSV Field Masking - Tuning Guide

## Overview

The two-stage detection approach first isolates the green playing field using HSV color space masking, then performs line detection only on that isolated area. This dramatically reduces noise from crowds, stadium structures, and other background elements.

## How HSV Masking Works

### What is HSV?

HSV (Hue, Saturation, Value) is a color space that separates:
- **Hue**: The color type (0-180 in OpenCV) - where green is ~35-85
- **Saturation**: The color intensity (0-255) - how "pure" the color is
- **Value**: The brightness (0-255) - how light or dark

### Why HSV for Field Detection?

HSV is superior to RGB for color-based segmentation because:
1. **Lighting invariance**: Shadows and highlights affect Value, not Hue
2. **Intuitive**: Green is a range of Hue values, regardless of brightness
3. **Robust**: Works across different lighting conditions and cameras

## The Two-Stage Process

### Stage 1: Field Segmentation
```python
# Convert to HSV
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

# Define green range
lower_green = (35, 40, 40)   # (H_min, S_min, V_min)
upper_green = (85, 255, 255) # (H_max, S_max, V_max)

# Create mask
mask = cv2.inRange(hsv, lower_green, upper_green)

# Clean up with morphological operations
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
```

### Stage 2: Masked Line Detection
```python
# Apply mask - zero out everything except field
masked_frame = cv2.bitwise_and(frame, frame, mask=mask)

# Now run Canny and Hough only on the field
edges = cv2.Canny(masked_frame, ...)
lines = cv2.HoughLinesP(edges, ...)
```

## Tuning HSV Parameters

### Quick Reference Table

| Field Condition | H_min | H_max | S_min | S_max | V_min | V_max |
|----------------|-------|-------|-------|-------|-------|-------|
| **Bright natural grass** | 35 | 85 | 40 | 255 | 40 | 255 |
| **Artificial turf (vivid)** | 35 | 75 | 60 | 255 | 50 | 255 |
| **Shaded/dark field** | 35 | 90 | 30 | 255 | 20 | 200 |
| **Overexposed/sunny** | 30 | 85 | 30 | 255 | 80 | 255 |
| **Night game (lights)** | 35 | 90 | 25 | 255 | 30 | 220 |
| **Worn natural grass** | 30 | 95 | 20 | 255 | 25 | 255 |

### Understanding Each Parameter

#### Hue (H) - The Color

- **Range**: 0-180 in OpenCV (360° compressed)
- **Green range**: ~35-85
- **Adjust if**: Field appears yellow (lower H_min) or blue-green (raise H_max)

**Common values:**
```python
# Standard green field
HSV_LOWER_GREEN = (35, 40, 40)
HSV_UPPER_GREEN = (85, 255, 255)

# Yellow-ish grass (warm)
HSV_LOWER_GREEN = (25, 40, 40)
HSV_UPPER_GREEN = (75, 255, 255)

# Blue-ish turf (cool)
HSV_LOWER_GREEN = (40, 40, 40)
HSV_UPPER_GREEN = (95, 255, 255)
```

#### Saturation (S) - Color Purity

- **Range**: 0-255
- **Low saturation**: Washed out, pale colors
- **High saturation**: Vivid, pure colors
- **Adjust if**: Mask includes non-green areas

**Common values:**
```python
# Vivid artificial turf
S_min = 60  # Only very saturated greens

# Natural grass (varies)
S_min = 30  # Allow less saturated greens

# Very worn field
S_min = 20  # Allow pale greens
```

#### Value (V) - Brightness

- **Range**: 0-255
- **Low value**: Dark areas
- **High value**: Bright areas
- **Adjust if**: Shadows excluded or highlights included

**Common values:**
```python
# Bright sunny day
V_min = 80   # Ignore dark shadows
V_max = 255

# Evening/overcast
V_min = 40   # Include darker greens
V_max = 220  # Ignore bright highlights

# Night game
V_min = 30   # Very dark acceptable
V_max = 200  # Bright stadium lights
```

## Step-by-Step Tuning Process

### Step 1: Run with Default Settings

```bash
python football_field_tracker.py
```

Look at panel 2 (Field Mask) in the output window.

### Step 2: Assess the Mask Quality

**Good mask:**
- Field area is mostly white
- Background (crowd, stadium) is black
- Coverage: 40-80% of frame (for zoomed footage)

**Poor mask:**
- Field has black holes
- Background areas are white
- Coverage < 20% or > 90%

### Step 3: Adjust Parameters

Edit `config.py`:

```python
# If field is NOT detected (too little white):
HSV_LOWER_GREEN = (30, 20, 30)   # Widen the range
HSV_UPPER_GREEN = (95, 255, 255)

# If too much background detected (too much white):
HSV_LOWER_GREEN = (40, 50, 50)   # Narrow the range
HSV_UPPER_GREEN = (80, 255, 220)
```

### Step 4: Adjust Morphological Operations

If the mask is noisy (small white specks) or has gaps:

```python
# Remove noise (small white specks)
MORPH_OPENING_ITERATIONS = 2  # Increase from 1

# Fill gaps (black holes in field)
MORPH_CLOSING_ITERATIONS = 3  # Increase from 2

# Bigger operations (slower but cleaner)
MORPH_KERNEL_SIZE = 21  # Increase from 15
```

## Common Issues and Solutions

### Issue 1: Field Not Detected

**Symptoms:**
- Panel 2 is mostly black
- Console shows "Field mask coverage: < 20%"

**Solutions:**
1. **Widen Hue range:**
   ```python
   HSV_LOWER_GREEN = (25, 40, 40)  # Lower H_min
   HSV_UPPER_GREEN = (95, 255, 255)  # Raise H_max
   ```

2. **Lower Saturation minimum:**
   ```python
   HSV_LOWER_GREEN = (35, 20, 40)  # Lower S_min
   ```

3. **Lower Value minimum:**
   ```python
   HSV_LOWER_GREEN = (35, 40, 20)  # Lower V_min
   ```

### Issue 2: Too Much Background Detected

**Symptoms:**
- Panel 2 shows crowd/stadium as white
- Unwanted lines still detected
- Coverage > 80%

**Solutions:**
1. **Narrow Hue range:**
   ```python
   HSV_LOWER_GREEN = (40, 40, 40)  # Raise H_min
   HSV_UPPER_GREEN = (80, 255, 255)  # Lower H_max
   ```

2. **Raise Saturation minimum:**
   ```python
   HSV_LOWER_GREEN = (35, 60, 40)  # Raise S_min (only pure greens)
   ```

3. **Increase morphological opening:**
   ```python
   MORPH_OPENING_ITERATIONS = 3
   ```

### Issue 3: Field Has Holes (Black Patches)

**Symptoms:**
- Panel 2 shows white field with black patches
- Patches are on yard lines, shadows, or players

**Solutions:**
1. **Increase closing iterations:**
   ```python
   MORPH_CLOSING_ITERATIONS = 4
   ```

2. **Use larger kernel:**
   ```python
   MORPH_KERNEL_SIZE = 25
   ```

3. **Widen Value range:**
   ```python
   HSV_LOWER_GREEN = (35, 40, 20)  # Lower V_min (accept shadows)
   ```

### Issue 4: Players and Lines Masked Out

**Symptoms:**
- Mask includes green jerseys
- White lines are holes in mask

**This is actually GOOD!** The mask is supposed to remove non-field elements. Line detection still works because:
- Line detection runs on the masked frame
- White lines show up in edge detection even if masked
- The mask removes background noise, not the lines themselves

### Issue 5: Different Lighting Across Field

**Symptoms:**
- Half the field detected, other half not
- Sunlight on one side, shadow on other

**Solutions:**
1. **Widen Value range dramatically:**
   ```python
   HSV_LOWER_GREEN = (35, 40, 15)   # Very low V_min
   HSV_UPPER_GREEN = (85, 255, 255) # Full V_max
   ```

2. **Reduce Saturation minimum:**
   ```python
   HSV_LOWER_GREEN = (35, 25, 20)  # Lower S_min
   ```

## Interactive HSV Tuning Tool

For advanced tuning, you can create a simple tool with trackbars:

```python
import cv2
import numpy as np

def nothing(x):
    pass

# Load a sample frame
frame = cv2.imread('sample_frame.jpg')
cv2.namedWindow('HSV Tuner')

# Create trackbars
cv2.createTrackbar('H_min', 'HSV Tuner', 35, 180, nothing)
cv2.createTrackbar('H_max', 'HSV Tuner', 85, 180, nothing)
cv2.createTrackbar('S_min', 'HSV Tuner', 40, 255, nothing)
cv2.createTrackbar('S_max', 'HSV Tuner', 255, 255, nothing)
cv2.createTrackbar('V_min', 'HSV Tuner', 40, 255, nothing)
cv2.createTrackbar('V_max', 'HSV Tuner', 255, 255, nothing)

while True:
    # Get current trackbar values
    h_min = cv2.getTrackbarPos('H_min', 'HSV Tuner')
    h_max = cv2.getTrackbarPos('H_max', 'HSV Tuner')
    s_min = cv2.getTrackbarPos('S_min', 'HSV Tuner')
    s_max = cv2.getTrackbarPos('S_max', 'HSV Tuner')
    v_min = cv2.getTrackbarPos('V_min', 'HSV Tuner')
    v_max = cv2.getTrackbarPos('V_max', 'HSV Tuner')
    
    # Create mask
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])
    mask = cv2.inRange(hsv, lower, upper)
    
    # Show result
    result = cv2.bitwise_and(frame, frame, mask=mask)
    combined = np.hstack([frame, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), result])
    cv2.imshow('HSV Tuner', combined)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    # Print current values when 'p' is pressed
    if cv2.waitKey(1) & 0xFF == ord('p'):
        print(f"HSV_LOWER_GREEN = ({h_min}, {s_min}, {v_min})")
        print(f"HSV_UPPER_GREEN = ({h_max}, {s_max}, {v_max})")

cv2.destroyAllWindows()
```

## Validation

### Good Field Mask Checklist

✅ **Coverage**: 30-80% of frame (for zoomed footage)  
✅ **Field shape**: Recognizable field boundary  
✅ **Background**: Crowd/stadium is black  
✅ **Consistency**: Mask stable across frames  
✅ **Line detection**: 3-10 lines detected (reasonable count)

### Warning Signs

⚠️ Coverage < 20%: Field not detected properly  
⚠️ Coverage > 90%: Background included  
⚠️ Flickering mask: Unstable parameters  
⚠️ No lines detected: Mask too restrictive  
⚠️ 50+ lines detected: Mask not filtering enough

## Performance Notes

### Speed vs. Accuracy

```python
# Fast (rough mask)
MORPH_KERNEL_SIZE = 9
MORPH_CLOSING_ITERATIONS = 1
MORPH_OPENING_ITERATIONS = 1

# Balanced (default)
MORPH_KERNEL_SIZE = 15
MORPH_CLOSING_ITERATIONS = 2
MORPH_OPENING_ITERATIONS = 1

# Accurate (slow but clean)
MORPH_KERNEL_SIZE = 25
MORPH_CLOSING_ITERATIONS = 3
MORPH_OPENING_ITERATIONS = 2
```

### Processing Time Impact

- HSV conversion: ~1ms per frame
- cv2.inRange: ~1-2ms per frame
- Morphological ops: ~5-20ms depending on kernel size

Total overhead: ~10-30ms per frame (still real-time at 30 FPS)

## Examples by Stadium Type

### Outdoor Natural Grass (Sunny)
```python
HSV_LOWER_GREEN = (35, 35, 60)
HSV_UPPER_GREEN = (85, 255, 255)
```

### Indoor/Dome Artificial Turf
```python
HSV_LOWER_GREEN = (40, 50, 50)
HSV_UPPER_GREEN = (80, 255, 240)
```

### College Field (Varied Condition)
```python
HSV_LOWER_GREEN = (30, 25, 30)
HSV_UPPER_GREEN = (90, 255, 255)
```

### Professional Stadium (Well-Maintained)
```python
HSV_LOWER_GREEN = (35, 45, 45)
HSV_UPPER_GREEN = (80, 255, 255)
```

## Debugging Tips

1. **Always check panel 2** (Field Mask) - this is your ground truth
2. **Start wide, then narrow** - easier to restrict than to expand
3. **Test on multiple frames** - lighting changes throughout game
4. **Look at histogram** - use cv2.calcHist on HSV channels to see actual value distributions
5. **Save working values** - keep a log of successful parameters for different stadiums

## Summary

The HSV masking approach is powerful but requires tuning for each video. Spend 5-10 minutes finding the right parameters, then they should work for the entire game. The payoff is dramatically cleaner line detection focused only on the playing field.

**Pro tip**: Create multiple config profiles for different stadiums/lighting conditions, then just swap them out:
```python
# config_sunny.py
HSV_LOWER_GREEN = (35, 40, 60)

# config_night.py  
HSV_LOWER_GREEN = (35, 30, 30)

# config_dome.py
HSV_LOWER_GREEN = (40, 50, 50)
```

