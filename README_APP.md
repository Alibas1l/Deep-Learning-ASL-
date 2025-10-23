# Sign Language Detection App

A real-time sign language detection application that uses your laptop camera to recognize ASL (American Sign Language) alphabet letters.

## Features

- 🎥 Real-time camera feed with hand detection overlay
- 🤖 AI-powered sign language recognition
- 📊 Live prediction display with confidence scores
- 📝 Prediction history tracking
- 🎨 Modern, responsive web interface
- 📱 Mobile-friendly design

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Your Model

If you have your trained model file (`LSignLD.h5`), place it in the project directory.

If you don't have the model file, run the setup script to create the model structure:

```bash
python setup_model.py
```

**Note:** You'll need to train the model with your sign language dataset for it to work properly.

### 3. Run the Application

```bash
python app.py
```

### 4. Access the App

Open your web browser and go to: `http://localhost:5000`

## How to Use

1. **Start the Camera**: Click the "Start Camera" button
2. **Position Your Hand**: Place your hand in the green detection box
3. **Make Signs**: Perform ASL alphabet gestures clearly
4. **View Results**: See real-time predictions and confidence scores
5. **Track History**: Monitor recent predictions in the history panel

## Tips for Best Results

- Ensure good lighting on your hand
- Keep your hand within the green detection box
- Make clear, distinct gestures
- Hold each gesture for 1-2 seconds
- Avoid background clutter

## Supported Signs

The app recognizes 29 ASL classes:
- Letters A-Z (26 classes)
- Special signs: del, nothing, space (3 classes)

## Troubleshooting

### Camera Not Working
- Make sure your camera is not being used by another application
- Check camera permissions in your browser
- Try refreshing the page

### Model Not Loading
- Ensure `LSignLD.h5` is in the project directory
- Check that the model file is not corrupted
- Run `python setup_model.py` to recreate the model structure

### Poor Predictions
- Ensure good lighting conditions
- Make sure your hand is clearly visible in the detection box
- Try different hand positions and gestures
- Check that the model was trained properly

## Technical Details

- **Framework**: Flask web application
- **Computer Vision**: OpenCV for camera handling
- **AI Model**: TensorFlow/Keras CNN model
- **Frontend**: HTML5, CSS3, JavaScript
- **Camera**: Uses laptop's built-in webcam

## File Structure

```
/workspace/
├── app.py                 # Main Flask application
├── setup_model.py         # Model setup script
├── requirements.txt       # Python dependencies
├── LSignLD.h5            # Trained model (you need to provide this)
├── templates/
│   └── index.html        # Web interface template
└── README_APP.md         # This file
```

## Development

To modify the app:

1. **Backend**: Edit `app.py` for Flask routes and model logic
2. **Frontend**: Edit `templates/index.html` for UI changes
3. **Model**: Use `setup_model.py` to modify model architecture

## License

This project is for educational purposes. Make sure you have proper permissions for any datasets used for training.