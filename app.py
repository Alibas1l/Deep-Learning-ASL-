from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import base64
import io
from PIL import Image
import json

app = Flask(__name__)

# Load the trained model
try:
    model = load_model('LSignLD.h5')
    print("Model loaded successfully!")
except:
    print("Model file not found. Please ensure LSignLD.h5 is in the current directory.")
    model = None

# ASL class names (29 classes: A-Z, del, nothing, space)
class_names = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    'del', 'nothing', 'space'
]

# Global variables for camera
camera = None
prediction_history = []
current_prediction = ""

def initialize_camera():
    global camera
    if camera is None:
        camera = cv2.VideoCapture(0)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return camera

def preprocess_frame(frame):
    """Preprocess frame for model prediction"""
    # Resize frame to model input size (64x64)
    frame_resized = cv2.resize(frame, (64, 64))
    
    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    
    # Normalize pixel values to [0, 1]
    frame_normalized = frame_rgb.astype(np.float32) / 255.0
    
    # Add batch dimension
    frame_batch = np.expand_dims(frame_normalized, axis=0)
    
    return frame_batch

def predict_sign(frame):
    """Predict sign language from frame"""
    if model is None:
        return "Model not loaded", 0.0
    
    try:
        # Preprocess frame
        processed_frame = preprocess_frame(frame)
        
        # Make prediction
        predictions = model.predict(processed_frame, verbose=0)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(np.max(predictions[0]))
        
        predicted_class = class_names[predicted_class_idx]
        
        return predicted_class, confidence
    except Exception as e:
        print(f"Prediction error: {e}")
        return "Error", 0.0

def generate_frames():
    """Generate video frames with predictions"""
    global current_prediction, prediction_history
    
    camera = initialize_camera()
    
    while True:
        success, frame = camera.read()
        if not success:
            break
        
        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)
        
        # Make prediction every 10 frames to reduce computational load
        if hasattr(generate_frames, 'frame_count'):
            generate_frames.frame_count += 1
        else:
            generate_frames.frame_count = 0
            
        if generate_frames.frame_count % 10 == 0:
            predicted_class, confidence = predict_sign(frame)
            current_prediction = predicted_class
            
            # Add to prediction history
            if confidence > 0.7:  # Only add confident predictions
                prediction_history.append(predicted_class)
                if len(prediction_history) > 20:  # Keep only last 20 predictions
                    prediction_history.pop(0)
        
        # Draw prediction on frame
        cv2.putText(frame, f"Prediction: {current_prediction}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Draw bounding box for hand detection area
        cv2.rectangle(frame, (50, 50), (300, 300), (255, 0, 0), 2)
        cv2.putText(frame, "Place hand here", (50, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_prediction')
def get_prediction():
    global current_prediction, prediction_history
    return jsonify({
        'current_prediction': current_prediction,
        'prediction_history': prediction_history[-10:],  # Last 10 predictions
        'model_loaded': model is not None
    })

@app.route('/clear_history')
def clear_history():
    global prediction_history
    prediction_history = []
    return jsonify({'status': 'cleared'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)