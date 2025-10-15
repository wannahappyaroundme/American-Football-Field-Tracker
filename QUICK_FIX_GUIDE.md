# Quick Fix Guide - Tracker Errors

## ✅ Fixed: initialize_tracker() Error

The error you saw:
```
TypeError: initialize_tracker() got an unexpected keyword argument 'max_age'
```

**Has been fixed!** The function signature has been updated to match the new system.

## Current Status

Your tracker should now run with the basic functionality. You'll see warnings about missing modules:

```
Warning: frame_analyzer not available, dynamic processing disabled
Warning: pose_estimator not available, pose estimation disabled
Warning: team_classifier not available, team classification disabled
Warning: field_homography not available, bird's eye view disabled
Warning: DeepSORT not available, falling back to SORT
```

**This is normal!** The basic tracker still works - you just don't have the advanced features yet.

## To Enable Advanced Features

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- `mediapipe` - For pose estimation
- `scikit-learn` - For team classification
- `torch` + `torchvision` - For DeepSORT
- `deep-sort-realtime` - For improved tracking

### Step 2: Verify Modules

Check if the new modules are working:

```python
# Test each module
python -c "from frame_analyzer import AdaptiveFrameProcessor; print('✓ Frame analyzer OK')"
python -c "from pose_estimator import PoseEstimator; print('✓ Pose estimator OK')"
python -c "from team_classifier import TeamClassifier; print('✓ Team classifier OK')"
python -c "from field_homography import FieldLineDetector; print('✓ Field homography OK')"
python -c "from deep_sort_realtime.deepsort_tracker import DeepSort; print('✓ DeepSORT OK')"
```

### Step 3: Run Again

```bash
python tracker.py
```

Now you should see:
```
Initializing DeepSORT tracker...
Initializing Advanced Modules...
Initializing dynamic frame processor...
Initializing pose estimator...
Initializing team classifier...
Initializing bird's eye view system...
```

## What Works Now (Even Without New Dependencies)

✅ **Basic tracking** - YOLO + SORT (original system)  
✅ **ROI masking** - Excludes top/bottom of frame  
✅ **Static frame skip** - Using `FRAME_SKIP` parameter  
✅ **Visualization** - Bounding boxes with IDs  

## What You'll Get After Installing Dependencies

🆕 **Dynamic frame processing** - Intelligent motion-based skipping  
🆕 **DeepSORT tracking** - More robust with appearance features  
🆕 **Pose estimation** - Player posture analysis  
🆕 **Team classification** - Automatic team detection  
🆕 **Bird's eye view** - Tactical map visualization  

## Troubleshooting

### If `pip install -r requirements.txt` fails:

Try installing packages one at a time:

```bash
# Core (should already have these)
pip install opencv-python numpy ultralytics

# Advanced features (install individually)
pip install mediapipe
pip install scikit-learn
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install deep-sort-realtime
```

### If torch installation is too slow:

You can skip DeepSORT and still use SORT:

```python
# In tracker_config.py
USE_DEEPSORT = False
```

### If MediaPipe has issues:

You can disable pose estimation:

```python
# In tracker_config.py
ENABLE_POSE_ESTIMATION = False
```

### If you want minimal overhead:

Disable all advanced features temporarily:

```python
# In tracker_config.py
ENABLE_DYNAMIC_PROCESSING = False
ENABLE_POSE_ESTIMATION = False
ENABLE_TEAM_CLASSIFICATION = False
ENABLE_BIRDS_EYE_VIEW = False
USE_DEEPSORT = False
```

This gives you the original basic tracker.

## Current Configuration

Your current settings (from previous edits):

```python
FRAME_SKIP = 30  # Very high - processes only every 30th frame
ROI_BOTTOM_PERCENT = 0.2  # Excludes bottom 20%
```

**Note**: With `FRAME_SKIP = 30`, you're only processing 3.3% of frames. Consider:
- Lower to `FRAME_SKIP = 5` for more data (still fast)
- Or install dependencies and use dynamic processing instead

## Recommended Next Steps

### Option A: Quick Test (Current Setup)
```bash
# Just run it as-is with basic features
python tracker.py
```

### Option B: Full Upgrade (Recommended)
```bash
# Install everything
pip install -r requirements.txt

# Set reasonable config
# In tracker_config.py:
FRAME_SKIP = 5  # Reduced from 30
ENABLE_DYNAMIC_PROCESSING = True  # Will override FRAME_SKIP

# Run with all features
python tracker.py
```

### Option C: Gradual Upgrade
```bash
# Install one feature at a time
pip install mediapipe
# Test pose estimation

pip install scikit-learn
# Test team classification

pip install torch torchvision deep-sort-realtime
# Test DeepSORT
```

## Summary

- ✅ **Your tracker now runs** (basic features)
- 📦 **Advanced features available** (install dependencies)
- ⚙️ **All features optional** (enable/disable in config)
- 🔧 **No breaking changes** (falls back gracefully)

You can use the basic tracker immediately or install dependencies for advanced features. Your choice!

---

**Current Status**: ✅ **Ready to Run**  
**Advanced Features**: ⏳ **Pending Installation**  
**Next Step**: Run `python tracker.py` or install dependencies

