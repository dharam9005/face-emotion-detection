# 🎭 Advanced Face Emotion Detection (Real-Time)

## 🚀 Overview

This project is a real-time Face Emotion Detection system using Deep Learning, OpenCV, and MTCNN.
It detects human emotions from live webcam feed with improved accuracy and smooth performance.

## 🧠 Features

* Real-time emotion detection via webcam
* Optimized smooth camera performance (low lag)
* Face detection using MTCNN (high accuracy)
* Emotion smoothing to reduce flickering
* Confidence score display
* TensorFlow/Keras deep learning model (.h5)
* Clean modular code structure

## 🛠️ Tech Stack

* Python
* OpenCV
* TensorFlow / Keras
* MTCNN
* NumPy

## 📂 Project Structure

```
face-emotion-detection/
│── app.py                # Main application (optimized)
│── emotion_utils.py      # Preprocessing & smoothing functions
│── emotion_model.h5      # Trained emotion model
│── requirements.txt      # Dependencies
│── README.md             # Documentation
```

## ⚡ Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/dharam9005/face-emotion-detection.git
cd face-emotion-detection
```

### 2. Create Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate   # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Project

```bash
python app.py
```

## 🎯 Model Details

* Input Shape: 64x64 grayscale
* Output: 7 Emotion Classes
  (Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral)

## 🔥 Recent Improvements (Latest Update)

* Fixed Keras model compatibility (compile=False)
* Optimized FPS for smoother webcam performance
* Added frame skipping optimization
* Improved preprocessing (64x64 input alignment)
* Modular utility functions (emotion_utils.py)
* Stable emotion prediction (anti-flicker)

## 📸 Output

Real-time webcam window showing:

* Detected face bounding box
* Emotion label with confidence %
* Smooth and stable predictions

## 👨‍💻 Author

Dharmender Singh Yadav (Dharam)
B.Tech AI & Data Science