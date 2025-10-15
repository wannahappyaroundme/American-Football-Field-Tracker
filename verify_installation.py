"""
Installation Verification Script
=================================

Run this script to verify that all dependencies are installed correctly
and the tracking system is ready to use.
"""

import sys

def check_dependency(module_name, package_name=None):
    """Check if a Python module can be imported."""
    if package_name is None:
        package_name = module_name
    
    try:
        __import__(module_name)
        print(f"✅ {package_name:20} installed")
        return True
    except ImportError:
        print(f"❌ {package_name:20} NOT installed")
        print(f"   Install with: pip install {package_name}")
        return False


def check_yolo_model():
    """Check if YOLO model can be loaded."""
    try:
        from ultralytics import YOLO
        print(f"✅ {'YOLOv8':20} model loading...")
        model = YOLO('yolov8n.pt')  # This will download if not present
        print(f"✅ {'YOLOv8':20} model loaded successfully")
        return True
    except Exception as e:
        print(f"❌ {'YOLOv8':20} model loading failed")
        print(f"   Error: {e}")
        return False


def check_files():
    """Check if required files exist."""
    import os
    
    required_files = [
        'tracker.py',
        'sort.py',
        'tracker_config.py',
        'requirements.txt'
    ]
    
    all_exist = True
    for filename in required_files:
        if os.path.exists(filename):
            print(f"✅ {filename:20} found")
        else:
            print(f"❌ {filename:20} NOT found")
            all_exist = False
    
    return all_exist


def main():
    """Main verification function."""
    print("="*60)
    print("Football Tracker - Installation Verification")
    print("="*60)
    print()
    
    # Check Python version
    print("Python Version:")
    print(f"  {sys.version}")
    print()
    
    # Check dependencies
    print("Checking Dependencies:")
    print("-"*60)
    
    all_ok = True
    
    # Core dependencies
    all_ok &= check_dependency('cv2', 'opencv-python')
    all_ok &= check_dependency('numpy')
    all_ok &= check_dependency('ultralytics')
    all_ok &= check_dependency('filterpy')
    all_ok &= check_dependency('scipy')
    
    print()
    
    # Check required files
    print("Checking Required Files:")
    print("-"*60)
    all_ok &= check_files()
    print()
    
    # Check YOLO model
    if all_ok:
        print("Checking YOLO Model:")
        print("-"*60)
        all_ok &= check_yolo_model()
        print()
    
    # Final verdict
    print("="*60)
    if all_ok:
        print("✅ ALL CHECKS PASSED!")
        print()
        print("Your installation is complete and ready to use.")
        print()
        print("Next steps:")
        print("  1. Place your video as 'zoomed_game.mp4'")
        print("  2. Run: python tracker.py")
        print()
        print("Or edit 'tracker_config.py' to change the video path.")
    else:
        print("❌ INSTALLATION INCOMPLETE")
        print()
        print("Please install missing dependencies:")
        print("  pip install -r requirements.txt")
        print()
        print("Then run this script again to verify.")
    print("="*60)


if __name__ == "__main__":
    main()

