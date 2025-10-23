#!/usr/bin/env python3
"""
Main script to run the Sign Language Detection App
Handles both demo mode and real camera mode
"""

import subprocess
import sys
import os
import time

def check_camera():
    """Check if camera is available"""
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()
            return ret
        return False
    except:
        return False

def check_dependencies():
    """Check if all required dependencies are installed"""
    required_packages = ['flask', 'opencv-python', 'tensorflow', 'numpy', 'Pillow']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    return missing_packages

def install_dependencies():
    """Install missing dependencies"""
    print("Installing missing dependencies...")
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
        print("✅ Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing dependencies: {e}")
        return False

def main():
    print("=" * 60)
    print("Sign Language Detection App - Setup & Run")
    print("=" * 60)
    
    # Check dependencies
    print("Checking dependencies...")
    missing = check_dependencies()
    if missing:
        print(f"Missing packages: {', '.join(missing)}")
        if not install_dependencies():
            print("❌ Failed to install dependencies. Please install manually:")
            print("pip install -r requirements.txt")
            return False
    else:
        print("✅ All dependencies are installed!")
    
    # Check for model file
    if not os.path.exists('LSignLD.h5'):
        print("⚠ Model file 'LSignLD.h5' not found.")
        print("Creating model structure...")
        try:
            subprocess.check_call([sys.executable, 'setup_model.py'])
            print("✅ Model structure created!")
        except subprocess.CalledProcessError:
            print("❌ Failed to create model structure")
            return False
    
    # Check camera availability
    print("Checking camera availability...")
    camera_available = check_camera()
    
    if camera_available:
        print("✅ Camera detected! Running with real camera...")
        app_file = 'app.py'
    else:
        print("⚠ No camera detected. Running in demo mode...")
        app_file = 'demo_app.py'
    
    print("\n" + "=" * 60)
    print("Starting the application...")
    print("Open your browser and go to: http://localhost:5001")
    print("Press Ctrl+C to stop the application")
    print("=" * 60)
    
    # Run the appropriate app
    try:
        subprocess.run([sys.executable, app_file])
    except KeyboardInterrupt:
        print("\n\nApplication stopped by user.")
    except Exception as e:
        print(f"\n❌ Error running application: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)