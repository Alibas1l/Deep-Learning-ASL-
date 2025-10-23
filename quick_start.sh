#!/bin/bash

echo "=========================================="
echo "ASL Sign Language Detector - Quick Start"
echo "=========================================="
echo ""

# Check if model exists
if [ -f "LSignLD.h5" ]; then
    echo "✓ Model found: LSignLD.h5"
    echo ""
    echo "Starting real-time detector..."
    python real_time_sign_detector.py
else
    echo "✗ Model not found: LSignLD.h5"
    echo ""
    echo "You need to train the model first:"
    echo "1. Update the dataset path in train_model.py"
    echo "2. Run: python train_model.py"
    echo ""
    echo "Or if you have the model file elsewhere, copy it here:"
    echo "   cp /path/to/your/LSignLD.h5 ."
    echo ""
fi
