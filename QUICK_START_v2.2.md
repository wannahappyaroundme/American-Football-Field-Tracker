# Quick Start Guide - v2.2 (Team Freeze Edition)

## 🚀 What's New in v2.2?

**Problem Fixed**: Teams were changing constantly frame-by-frame!

**Solution**: Aggressive team freezing that locks player teams after just 2 frames.

---

## Run Your Analysis

```bash
python tracker.py
```

That's it! The new aggressive freezing is enabled by default.

---

## What You'll See

### During Processing:
```
🔒 ID:5 팀 고정: Team A (신뢰도: 2프레임)
🔒 ID:12 팀 고정: Team B (신뢰도: 2프레임)
📊 Frame 30: 22 players | 18 frozen (🔒) | 20 fresh detections
📊 Frame 60: 22 players | 22 frozen (🔒) | 21 fresh detections
```

### In The Output Video:
- **Green borders** around players with frozen teams
- **🔒 emoji** in player labels
- **Stable colors** - no more flickering!

### Final Summary:
```
TEAM FREEZING STATISTICS
Maximum players tracked: 22
Maximum frozen players: 22
Freeze success rate: 100.0%
```

---

## Still Seeing Flickering?

### Option 1: Instant Freeze (Most Aggressive)
Edit [tracker_config.py](tracker_config.py):
```python
TEAM_ASSIGNMENT_CONFIDENCE = 1  # Freeze on first detection
```

### Option 2: Turn Off Indicators (If Distracting)
```python
SHOW_FROZEN_INDICATOR = False   # No green borders
PRINT_FREEZE_EVENTS = False     # Less console output
```

### Option 3: Fix Color Detection
Make sure your HSV ranges match your teams:
```python
TEAM_A_HSV_RANGE = ((90, 50, 50), (130, 255, 255))    # Blue
TEAM_B_HSV_RANGE = ((0, 0, 180), (180, 30, 255))      # White
```

---

## Current Settings (v2.2 Defaults)

```python
# Fast freezing
TEAM_ASSIGNMENT_CONFIDENCE = 2  # Was 5 in v2.1

# Better tracking
TRACKING_IOU_THRESHOLD = 0.20   # Was 0.25
MAX_TRACKING_FRAMES = 90        # Was 60

# Visual feedback
SHOW_FROZEN_INDICATOR = True
PRINT_FREEZE_EVENTS = True
PRINT_TRACKING_STATS = True
```

---

## Expected Results

✅ 95-100% freeze success rate
✅ Minimal flickering (only first 1-2 frames)
✅ Stable team colors throughout video
✅ Clear visual confirmation of frozen teams

---

## Need Help?

See [TEAM_FREEZE_IMPROVEMENTS.md](TEAM_FREEZE_IMPROVEMENTS.md) for:
- Detailed explanation of changes
- Troubleshooting guide
- Advanced configuration options
- Manual override system (if needed)

---

**Quick Test**: Run `python tracker.py` and watch the console for 🔒 notifications. If you see players freezing within the first few seconds, it's working perfectly!
