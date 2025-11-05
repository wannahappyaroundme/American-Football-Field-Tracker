"""
Test script to verify CLIP integration is working correctly.
Run this before processing videos to ensure all components are functional.
"""

import sys
import numpy as np
import cv2

print("="*60)
print("CLIP Integration Test Suite")
print("="*60)

# Test 1: Import all modules
print("\n[Test 1/5] Testing module imports...")
try:
    import torch
    import clip
    from ultralytics import YOLO
    from clip_classifier import CLIPEntityClassifier, CLIPTeamClassifier
    from detector_tracker import DetectorTracker
    from team_classifier import TeamClassifier
    print("✓ All modules imported successfully")
except ImportError as e:
    print(f"✗ Import failed: {e}")
    print("\nPlease run: bash install_clip.sh")
    sys.exit(1)

# Test 2: Check GPU availability
print("\n[Test 2/5] Checking GPU availability...")
if torch.cuda.is_available():
    print(f"✓ CUDA available (device: {torch.cuda.get_device_name(0)})")
    print(f"  - CUDA version: {torch.version.cuda}")
    print(f"  - GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    device = 'cuda'
else:
    print("⚠ CUDA not available, using CPU (slower)")
    device = 'cpu'

# Test 3: Load CLIP models
print("\n[Test 3/5] Loading CLIP models...")
try:
    print("  - Loading entity classifier...")
    entity_classifier = CLIPEntityClassifier()
    print("  ✓ Entity classifier loaded")

    print("  - Loading team classifier...")
    team_classifier = CLIPTeamClassifier()
    print("  ✓ Team classifier loaded")

    print("✓ CLIP models loaded successfully")
except Exception as e:
    print(f"✗ Failed to load CLIP models: {e}")
    sys.exit(1)

# Test 4: Test entity classification with dummy image
print("\n[Test 4/5] Testing entity classification...")
try:
    # Create a dummy image (blue rectangle simulating a player)
    dummy_image = np.zeros((200, 100, 3), dtype=np.uint8)
    dummy_image[:, :] = (255, 0, 0)  # Blue in BGR

    entity_type, confidence = entity_classifier.classify_entity(dummy_image)
    print(f"  - Entity type: {entity_type}")
    print(f"  - Confidence: {confidence:.2%}")
    print("✓ Entity classification working")
except Exception as e:
    print(f"✗ Entity classification failed: {e}")
    sys.exit(1)

# Test 5: Test team classification with dummy image
print("\n[Test 5/5] Testing team classification...")
try:
    team_label, team_color, confidence = team_classifier.classify_team(dummy_image)
    print(f"  - Team: {team_label}")
    print(f"  - Color (BGR): {team_color}")
    print(f"  - Confidence: {confidence:.2%}")
    print("✓ Team classification working")
except Exception as e:
    print(f"✗ Team classification failed: {e}")
    sys.exit(1)

# Test 6: Test batch processing
print("\n[Test 6/6] Testing batch processing...")
try:
    # Create multiple dummy images
    dummy_images = [dummy_image.copy() for _ in range(5)]

    results = entity_classifier.batch_classify_entities(dummy_images)
    print(f"  - Processed {len(results)} images in batch")
    print(f"  - Results: {[r[0] for r in results]}")
    print("✓ Batch processing working")
except Exception as e:
    print(f"✗ Batch processing failed: {e}")
    sys.exit(1)

# Summary
print("\n" + "="*60)
print("All Tests Passed!")
print("="*60)
print("\nSystem Configuration:")
print(f"  - Device: {device}")
print(f"  - CLIP model: {entity_classifier.model.__class__.__name__}")
print(f"  - Entity prompts: {len(entity_classifier.entity_prompts)}")
print(f"  - Team prompts: {len(team_classifier.team_prompts)}")

print("\nNext steps:")
print("1. Place your video in: input/video.mp4")
print("2. Run calibration: python calibrate_homography.py")
print("3. Customize prompts in config.py (optional)")
print("4. Run analysis: python main.py")

print("\nFor more information, see CLIP_INTEGRATION_README.md")
print("="*60)
