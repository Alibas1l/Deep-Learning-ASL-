#!/usr/bin/env python3
"""
Simple test script to verify camera functionality
"""

import cv2
import sys

def test_camera():
    """Test if camera is accessible"""
    print("Testing camera access...")
    
    # Try to open camera
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Error: Could not open camera")
        print("Make sure your camera is not being used by another application")
        return False
    
    print("✅ Camera opened successfully")
    
    # Try to read a frame
    ret, frame = cap.read()
    if not ret:
        print("❌ Error: Could not read from camera")
        cap.release()
        return False
    
    print("✅ Successfully read frame from camera")
    print(f"Frame shape: {frame.shape}")
    
    # Release camera
    cap.release()
    print("✅ Camera released successfully")
    
    return True

if __name__ == "__main__":
    print("=" * 50)
    print("Camera Test for Sign Language Detection App")
    print("=" * 50)
    
    success = test_camera()
    
    if success:
        print("\n🎉 Camera test passed! You can now run the Flask app.")
        print("Run: python3 app.py")
        print("Then open: http://localhost:5001")
    else:
        print("\n❌ Camera test failed. Please check your camera setup.")
        sys.exit(1)