"""
Test Setup - Verify all dependencies and model are ready
"""

import sys

def test_imports():
    """Test if all required packages are installed"""
    print("Testing package imports...")
    
    packages = {
        'tensorflow': None,
        'cv2': 'opencv-python',
        'numpy': None
    }
    
    failed = []
    
    for package, pip_name in packages.items():
        try:
            __import__(package)
            print(f"  ✓ {package}")
        except ImportError:
            install_name = pip_name if pip_name else package
            print(f"  ✗ {package} - Install with: pip install {install_name}")
            failed.append(install_name)
    
    return len(failed) == 0

def test_model():
    """Test if model file exists and can be loaded"""
    print("\nTesting model...")
    import os
    
    model_path = 'LSignLD.h5'
    
    if not os.path.exists(model_path):
        print(f"  ✗ Model file not found: {model_path}")
        print(f"    Run 'python train_model.py' to create the model")
        return False
    
    try:
        import tensorflow as tf
        model = tf.keras.models.load_model(model_path)
        print(f"  ✓ Model loaded successfully")
        print(f"    Input shape: {model.input_shape}")
        print(f"    Output shape: {model.output_shape}")
        print(f"    Total parameters: {model.count_params():,}")
        return True
    except Exception as e:
        print(f"  ✗ Error loading model: {e}")
        return False

def test_camera():
    """Test if camera is accessible"""
    print("\nTesting camera...")
    
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print(f"  ✗ Camera not accessible")
            print(f"    Make sure your webcam is connected and not in use")
            return False
        
        ret, frame = cap.read()
        if not ret:
            print(f"  ✗ Could not read from camera")
            cap.release()
            return False
        
        print(f"  ✓ Camera accessible")
        print(f"    Resolution: {frame.shape[1]}x{frame.shape[0]}")
        
        cap.release()
        return True
        
    except Exception as e:
        print(f"  ✗ Error accessing camera: {e}")
        return False

def main():
    """Run all tests"""
    print("="*50)
    print("ASL Sign Language Detector - Setup Test")
    print("="*50)
    print()
    
    results = {
        'imports': test_imports(),
        'model': test_model(),
        'camera': test_camera()
    }
    
    print("\n" + "="*50)
    print("Test Summary")
    print("="*50)
    
    all_passed = all(results.values())
    
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    print()
    
    if all_passed:
        print("🎉 All tests passed! You're ready to run the detector.")
        print("\nRun: python real_time_sign_detector.py")
    else:
        print("⚠️  Some tests failed. Please fix the issues above.")
        
        if not results['imports']:
            print("\n1. Install missing packages:")
            print("   pip install -r requirements.txt")
        
        if not results['model']:
            print("\n2. Train or copy the model:")
            print("   python train_model.py")
        
        if not results['camera']:
            print("\n3. Check your webcam connection")
    
    print()
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
