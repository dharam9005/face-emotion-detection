import cv2
import numpy as np
from tensorflow.keras.models import load_model
from mtcnn import MTCNN
from emotion_utils import preprocess_face, get_stable_emotion

# Load model (IMPORTANT FIX for old .h5 model)
model = load_model("emotion_model.h5", compile=False)

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Initialize face detector
detector = MTCNN()

# Initialize camera (Windows fix included)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Check if camera opened
if not cap.isOpened():
    print("ERROR: Camera not accessible")
    exit()

print("Starting Advanced Emotion Detection... Press Q to quit.")

# MAIN LOOP (continue must be inside this loop)
frame_count = 0
skip_frames = 2  # Predict every 3rd frame (BIG speed boost)

while True:
    ret, frame = cap.read()

    if not ret or frame is None:
        continue

    # 🔥 Resize frame for faster processing (MAJOR SPEED BOOST)
    frame = cv2.resize(frame, (640, 480))

    frame_count += 1

    # Only run heavy AI detection every few frames
    if frame_count % (skip_frames + 1) == 0:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect faces (heavy step)
        faces = detector.detect_faces(rgb_frame)

        detected_faces = faces  # store faces globally
    else:
        # Reuse last detected faces (very fast)
        faces = detected_faces if 'detected_faces' in globals() else []

    for face in faces:
        x, y, w, h = face['box']
        x, y = abs(x), abs(y)

        face_img = frame[y:y+h, x:x+w]
        processed_face = preprocess_face(face_img)

        if processed_face is not None and frame_count % (skip_frames + 1) == 0:
            try:
                predictions = model.predict(processed_face, verbose=0)
                max_index = int(np.argmax(predictions))
                confidence = predictions[0][max_index] * 100
                emotion = emotion_labels[max_index]
                stable_emotion = get_stable_emotion(emotion)
                current_label = f"{stable_emotion} ({confidence:.1f}%)"
            except:
                current_label = "Detecting..."

        # Draw rectangle + label (light operation)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        if 'current_label' in locals():
            cv2.putText(frame, current_label, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow("Advanced Face Emotion Detection (Optimized)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()