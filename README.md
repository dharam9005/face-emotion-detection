# Face Emotion Detection 😃😡😢

A real-time face emotion detection system built with **Python**, **OpenCV**, and **TensorFlow/Keras**.

This upgraded version improves practical usage with:
- stronger video stability through temporal smoothing,
- uncertainty handling to reduce overconfident mistakes,
- lightweight face tracking with persistent IDs,
- quality filtering for blurry/over-dark face crops,
- better preprocessing (CLAHE + padded crop),
- real-time metrics (FPS + number of active faces).

---

## 🚀 Features

- Real-time webcam inference
- Face detection (Haar Cascade)
- 7-class emotion classification:
  - Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral
- EMA-smoothed probability predictions per tracked face ID
- "Uncertain" output when confidence/margin are weak
- Top-2 emotion display (primary + alternative)
- Face quality gate (brightness + blur checks)
- Configurable runtime thresholds via CLI flags

---

## 🛠 Tech Stack

- Python 3.9+
- OpenCV
- NumPy
- TensorFlow / Keras

---

## 📦 Installation

```bash
pip install -r requirements.txt
```

---

## ▶️ Run

```bash
python app.py
```

Optional tuning:

```bash
python app.py --camera 0 --model emotion_model.h5 --min-confidence 0.45 --min-margin 0.15 --smooth-alpha 0.55
```

### CLI options

- `--camera`: webcam index (default `0`)
- `--model`: path to Keras model file (default `emotion_model.h5`)
- `--min-confidence`: minimum top-1 probability before accepting a class
- `--min-margin`: required probability gap between top-1 and top-2 class
- `--smooth-alpha`: smoothing factor for temporal EMA (`0..1`)
- `--frame-width`, `--frame-height`: camera capture resolution

---

## 📈 Why this is more accurate/effective

Compared to naive frame-by-frame inference, this version improves practical quality by:

1. **Temporal smoothing**: reduces label flicker in live video.
2. **Uncertainty logic**: avoids aggressive wrong predictions.
3. **Quality filtering**: skips unusable face crops (very blurry/underexposed).
4. **Contrast normalization**: CLAHE helps low-light robustness.
5. **Padded face ROI**: captures more context around face area.

---

## 🧭 Next upgrades for production

- Swap Haar detector with RetinaFace / MediaPipe for profile/occlusion robustness
- Retrain model with FER+ / AffectNet style data + calibration
- Add ONNX/TFLite export + quantization for edge devices
- Add privacy-safe event API and dashboard analytics

