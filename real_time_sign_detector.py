"""
Real-time Sign Language Detection Application
Uses webcam to detect ASL alphabet signs in real-time
"""

import cv2
import numpy as np
import tensorflow as tf
from collections import deque
import time

# Define class names (ASL Alphabet - 29 classes)
CLASS_NAMES = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 
               'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 
               'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space']

class SignLanguageDetector:
    def __init__(self, model_path='LSignLD.h5', image_size=64):
        """
        Initialize the Sign Language Detector
        
        Args:
            model_path: Path to the trained model (.h5 file)
            image_size: Size of input images (default: 64x64)
        """
        self.image_size = image_size
        self.model = None
        self.prediction_history = deque(maxlen=10)  # Store last 10 predictions
        self.fps_history = deque(maxlen=30)
        
        # Load the model
        try:
            print(f"Loading model from {model_path}...")
            self.model = tf.keras.models.load_model(model_path)
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Please make sure 'LSignLD.h5' exists in the current directory.")
            raise
    
    def preprocess_frame(self, frame, roi=None):
        """
        Preprocess a frame for model prediction
        
        Args:
            frame: Input frame from camera
            roi: Region of Interest (x, y, w, h) or None for center region
        
        Returns:
            Preprocessed image ready for prediction
        """
        if roi is None:
            # Use center region of frame
            h, w = frame.shape[:2]
            size = min(h, w) // 2
            x = (w - size) // 2
            y = (h - size) // 2
            roi = (x, y, size, size)
        
        x, y, w, h = roi
        roi_frame = frame[y:y+h, x:x+w]
        
        # Resize to model input size
        resized = cv2.resize(roi_frame, (self.image_size, self.image_size))
        
        # Convert to RGB (OpenCV uses BGR)
        rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Normalize and add batch dimension
        normalized = rgb_frame.astype(np.float32)
        return np.expand_dims(normalized, axis=0)
    
    def predict(self, preprocessed_image):
        """
        Make a prediction on preprocessed image
        
        Returns:
            (predicted_class, confidence, all_probabilities)
        """
        predictions = self.model.predict(preprocessed_image, verbose=0)
        predicted_class = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class]
        
        return CLASS_NAMES[predicted_class], confidence, predictions[0]
    
    def draw_ui(self, frame, prediction, confidence, roi, fps=0):
        """
        Draw the UI elements on the frame
        
        Args:
            frame: Video frame
            prediction: Predicted class name
            confidence: Prediction confidence
            roi: Region of interest (x, y, w, h)
            fps: Frames per second
        """
        h, w = frame.shape[:2]
        overlay = frame.copy()
        
        # Draw ROI box with glow effect
        x, y, box_w, box_h = roi
        cv2.rectangle(overlay, (x-2, y-2), (x+box_w+2, y+box_h+2), (0, 255, 255), 3)
        cv2.rectangle(frame, (x, y), (x+box_w, y+box_h), (0, 200, 200), 2)
        
        # Add transparency
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        
        # Draw header bar
        cv2.rectangle(frame, (0, 0), (w, 80), (40, 40, 40), -1)
        cv2.putText(frame, "ASL Sign Language Detector", (20, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(frame, f"FPS: {fps:.1f}", (w - 150, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Draw prediction panel (right side)
        panel_w = 350
        panel_x = w - panel_w - 20
        panel_y = 100
        panel_h = 200
        
        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), 
                     (30, 30, 30), -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        
        # Border
        cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), 
                     (0, 255, 0), 2)
        
        # Prediction text
        cv2.putText(frame, "PREDICTION", (panel_x + 20, panel_y + 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Large predicted letter
        letter_color = (0, 255, 0) if confidence > 0.7 else (0, 255, 255) if confidence > 0.5 else (0, 128, 255)
        cv2.putText(frame, prediction, (panel_x + 140, panel_y + 110), 
                   cv2.FONT_HERSHEY_SIMPLEX, 3.0, letter_color, 4)
        
        # Confidence bar
        bar_y = panel_y + 140
        bar_w = int((panel_w - 40) * confidence)
        cv2.rectangle(frame, (panel_x + 20, bar_y), (panel_x + panel_w - 20, bar_y + 20), 
                     (60, 60, 60), -1)
        cv2.rectangle(frame, (panel_x + 20, bar_y), (panel_x + 20 + bar_w, bar_y + 20), 
                     letter_color, -1)
        
        # Confidence percentage
        cv2.putText(frame, f"Confidence: {confidence*100:.1f}%", (panel_x + 20, bar_y + 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Instructions panel (bottom)
        inst_h = 60
        inst_y = h - inst_h
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, inst_y), (w, h), (40, 40, 40), -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        
        cv2.putText(frame, "Instructions: Place your hand in the yellow box | Press 'Q' to quit | Press 'S' to screenshot", 
                   (20, inst_y + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Prediction history (left side)
        if len(self.prediction_history) > 0:
            hist_panel_x = 20
            hist_panel_y = 100
            hist_panel_w = 250
            hist_panel_h = min(300, 30 + len(self.prediction_history) * 25)
            
            overlay = frame.copy()
            cv2.rectangle(overlay, (hist_panel_x, hist_panel_y), 
                         (hist_panel_x + hist_panel_w, hist_panel_y + hist_panel_h), 
                         (30, 30, 30), -1)
            cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
            cv2.rectangle(frame, (hist_panel_x, hist_panel_y), 
                         (hist_panel_x + hist_panel_w, hist_panel_y + hist_panel_h), 
                         (100, 100, 100), 2)
            
            cv2.putText(frame, "HISTORY", (hist_panel_x + 10, hist_panel_y + 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            
            for i, (hist_pred, hist_conf) in enumerate(list(self.prediction_history)[-8:]):
                y_pos = hist_panel_y + 50 + i * 25
                text = f"{hist_pred} ({hist_conf*100:.0f}%)"
                cv2.putText(frame, text, (hist_panel_x + 10, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        return frame
    
    def run(self):
        """
        Run the real-time detection application
        """
        # Initialize camera
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open camera.")
            return
        
        # Set camera properties for better quality
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        print("Starting real-time detection...")
        print("Press 'Q' to quit")
        print("Press 'S' to save screenshot")
        
        screenshot_count = 0
        
        while True:
            start_time = time.time()
            
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame.")
                break
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Define ROI (center region)
            h, w = frame.shape[:2]
            roi_size = min(h, w) // 2
            roi_x = (w - roi_size) // 2
            roi_y = (h - roi_size) // 2
            roi = (roi_x, roi_y, roi_size, roi_size)
            
            # Preprocess and predict
            preprocessed = self.preprocess_frame(frame, roi)
            prediction, confidence, _ = self.predict(preprocessed)
            
            # Update history
            self.prediction_history.append((prediction, confidence))
            
            # Calculate FPS
            fps = 1.0 / (time.time() - start_time) if (time.time() - start_time) > 0 else 0
            self.fps_history.append(fps)
            avg_fps = np.mean(self.fps_history)
            
            # Draw UI
            frame = self.draw_ui(frame, prediction, confidence, roi, avg_fps)
            
            # Display frame
            cv2.imshow('ASL Sign Language Detector', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                break
            elif key == ord('s') or key == ord('S'):
                screenshot_name = f'screenshot_{screenshot_count}.png'
                cv2.imwrite(screenshot_name, frame)
                print(f"Screenshot saved as {screenshot_name}")
                screenshot_count += 1
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        print("Application closed.")


def main():
    """Main function to run the application"""
    try:
        detector = SignLanguageDetector(model_path='LSignLD.h5', image_size=64)
        detector.run()
    except Exception as e:
        print(f"Error: {e}")
        print("\nPlease ensure:")
        print("1. The trained model 'LSignLD.h5' exists in the current directory")
        print("2. You have a webcam connected")
        print("3. All required packages are installed (run: pip install -r requirements.txt)")


if __name__ == "__main__":
    main()
