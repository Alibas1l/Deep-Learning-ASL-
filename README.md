# Sign Language Detection App

A real-time sign language detection application that uses your laptop camera to recognize ASL (American Sign Language) alphabet letters. The app features a modern web interface similar to the design shown in your reference image.

## 🎯 Features

- 🎥 **Real-time Camera Feed**: Live video stream with hand detection overlay
- 🤖 **AI-Powered Recognition**: CNN model for accurate sign language detection
- 📊 **Live Predictions**: Real-time display of predicted letters with confidence scores
- 📝 **Prediction History**: Track recent predictions in an interactive grid
- 🎨 **Modern UI**: Beautiful, responsive web interface with gradient backgrounds
- 📱 **Mobile-Friendly**: Works on desktop and mobile devices
- 🔄 **Demo Mode**: Works without camera for testing and demonstration

## 🚀 Quick Start

### Option 1: Easy Setup (Recommended)
```bash
python3 run_app.py
```

### Option 2: Manual Setup
```bash
# Install dependencies
pip3 install -r requirements.txt

# Set up model (if needed)
python3 setup_model.py

# Run the app
python3 app.py  # For real camera
# OR
python3 demo_app.py  # For demo mode
```

Then open your browser and go to: **http://localhost:5001**

## 📁 Project Structure

```
/workspace/
├── app.py                 # Main Flask app with real camera
├── demo_app.py           # Demo version (no camera required)
├── run_app.py            # Smart launcher (auto-detects camera)
├── setup_model.py        # Model setup and creation
├── test_camera.py        # Camera functionality test
├── requirements.txt      # Python dependencies
├── LSignLD.h5           # Trained model (you need to provide this)
├── templates/
│   └── index.html       # Web interface template
├── README.md            # This file
└── README_APP.md        # Detailed app documentation
```

## 🎮 How to Use

1. **Start the App**: Run `python3 run_app.py` or `python3 app.py`
2. **Open Browser**: Go to `http://localhost:5001`
3. **Position Hand**: Place your hand in the green detection box
4. **Make Signs**: Perform clear ASL alphabet gestures
5. **View Results**: See real-time predictions and confidence scores
6. **Track History**: Monitor recent predictions in the history panel

## 🎯 Supported Signs

The app recognizes **29 ASL classes**:
- **Letters A-Z** (26 classes)
- **Special signs**: del, nothing, space (3 classes)

## 🛠️ Technical Details

### Backend
- **Framework**: Flask web application
- **Computer Vision**: OpenCV for camera handling
- **AI Model**: TensorFlow/Keras CNN model
- **Image Processing**: Real-time frame preprocessing and prediction

### Frontend
- **HTML5**: Modern semantic markup
- **CSS3**: Responsive design with gradients and animations
- **JavaScript**: Real-time updates and camera controls
- **Font Awesome**: Beautiful icons

### Model Architecture
- **Type**: Convolutional Neural Network (CNN)
- **Input**: 64x64x3 RGB images
- **Layers**: Conv2D + BatchNorm + ReLU + MaxPool blocks
- **Output**: 29 classes (A-Z + del, nothing, space)

## 🔧 Setup Instructions

### Prerequisites
- Python 3.7+
- Webcam (for real camera mode)
- Modern web browser

### Installation

1. **Clone/Download** the project files
2. **Install Dependencies**:
   ```bash
   pip3 install -r requirements.txt
   ```
3. **Set up Model**:
   ```bash
   python3 setup_model.py
   ```
4. **Run the App**:
   ```bash
   python3 run_app.py
   ```

### For Real Camera Usage

1. **Ensure your camera is not being used** by other applications
2. **Grant camera permissions** to your browser
3. **Run the app** and allow camera access when prompted

## 🎨 UI Features

### Design Elements
- **Gradient Backgrounds**: Beautiful purple-blue gradients
- **Card-based Layout**: Clean, modern card design
- **Responsive Grid**: Adapts to different screen sizes
- **Smooth Animations**: Hover effects and transitions
- **Status Indicators**: Real-time connection and model status

### Interactive Elements
- **Live Camera Feed**: Real-time video with detection overlay
- **Prediction Display**: Large, clear letter display
- **History Grid**: Interactive prediction history
- **Control Buttons**: Start/stop camera, clear history
- **Status Updates**: Connection and model status

## 🐛 Troubleshooting

### Camera Issues
- **No Camera Detected**: The app will automatically switch to demo mode
- **Permission Denied**: Grant camera permissions in your browser
- **Camera in Use**: Close other applications using the camera

### Model Issues
- **Model Not Loaded**: Ensure `LSignLD.h5` is in the project directory
- **Poor Predictions**: Make sure the model was trained properly
- **Slow Performance**: Try reducing the prediction frequency

### Browser Issues
- **Video Not Loading**: Check browser compatibility and permissions
- **Layout Problems**: Try refreshing the page or clearing cache

## 📊 Performance Tips

### For Better Recognition
- **Good Lighting**: Ensure your hand is well-lit
- **Clear Background**: Avoid cluttered backgrounds
- **Steady Hand**: Hold gestures for 1-2 seconds
- **Proper Positioning**: Keep hand within the detection box

### For Better Performance
- **Close Other Apps**: Free up system resources
- **Use Chrome/Firefox**: Better webcam support
- **Stable Internet**: For smooth video streaming

## 🔮 Future Enhancements

- **Multiple Hand Detection**: Support for both hands
- **Gesture Sequences**: Recognize word sequences
- **Custom Training**: Easy model retraining interface
- **Export Features**: Save prediction history
- **Mobile App**: Native mobile application
- **Voice Output**: Text-to-speech for predictions

## 📝 Notes

- **Demo Mode**: The app works without a camera for demonstration purposes
- **Model Training**: You need to train the model with your sign language dataset
- **Browser Compatibility**: Works best with Chrome, Firefox, and Safari
- **Performance**: Real-time prediction may vary based on system specs

## 🎉 Success!

Your sign language detection app is now ready! The interface matches the design you requested with:

- ✅ Real-time camera feed
- ✅ Live prediction display
- ✅ Prediction history tracking
- ✅ Modern, responsive UI
- ✅ Demo mode for testing
- ✅ Easy setup and deployment

Enjoy using your sign language detection app! 🤟

---

**Need Help?** Check the troubleshooting section or run `python3 test_camera.py` to diagnose camera issues.