import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Load emotion model (inference only)
emotion_model = load_model("emotion_model.h5", compile=False)

emotion_labels = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Sad",
    "Surprise",
    "Neutral"
]

# Start webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    raise RuntimeError("Webcam not accessible")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.05,
        minNeighbors=3,
        minSize=(40, 40)
    )

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        if face.size == 0:
            continue

        face = cv2.resize(face, (64, 64))
        face = face.astype("float32") / 255.0
        face = np.expand_dims(face, axis=-1)
        face = np.expand_dims(face, axis=0)

        preds = emotion_model.predict(face, verbose=0)
        emotion_index = np.argmax(preds)
        confidence = preds[0][emotion_index] * 100
        emotion = emotion_labels[emotion_index]

        if confidence < 25:
            label = f"{emotion} (low confidence)"
        else:
            label = f"{emotion} ({confidence:.1f}%)"

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            frame,
            label,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2
        )

    cv2.putText(
        frame,
        "Press Q to quit",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.imshow("Face Emotion Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
