import cv2
import numpy as np
from collections import deque

# Emotion smoothing queue (reduces flickering)
emotion_queue = deque(maxlen=10)

def preprocess_face(face_img):
    """
    Preprocess face for emotion model
    """
    try:
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (64, 64))
        normalized = resized / 255.0
        reshaped = normalized.reshape(1, 64, 64, 1)
        return reshaped
    except:
        return None

def get_stable_emotion(emotion):
    """
    Smooth emotion predictions to avoid flickering
    """
    emotion_queue.append(emotion)
    return max(set(emotion_queue), key=emotion_queue.count)