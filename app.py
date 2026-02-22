import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from tensorflow.keras.models import load_model


EMOTION_LABELS = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Sad",
    "Surprise",
    "Neutral",
]


@dataclass
class TrackState:
    bbox: Tuple[int, int, int, int]
    smoothed_probs: np.ndarray
    missed_frames: int = 0


class FaceTracker:
    """Simple IoU-based face tracker that keeps temporal emotion smoothing per face ID."""

    def __init__(self, iou_threshold: float = 0.3, max_missed: int = 12):
        self.iou_threshold = iou_threshold
        self.max_missed = max_missed
        self.tracks: Dict[int, TrackState] = {}
        self._next_id = 1

    @staticmethod
    def _iou(box_a: Tuple[int, int, int, int], box_b: Tuple[int, int, int, int]) -> float:
        ax, ay, aw, ah = box_a
        bx, by, bw, bh = box_b
        x1 = max(ax, bx)
        y1 = max(ay, by)
        x2 = min(ax + aw, bx + bw)
        y2 = min(ay + ah, by + bh)

        inter_w = max(0, x2 - x1)
        inter_h = max(0, y2 - y1)
        inter = inter_w * inter_h
        if inter == 0:
            return 0.0

        union = aw * ah + bw * bh - inter
        return inter / (union + 1e-6)

    def match(self, detections: List[Tuple[int, int, int, int]]) -> Dict[int, Tuple[int, int, int, int]]:
        assignments: Dict[int, Tuple[int, int, int, int]] = {}
        used_detection_indexes = set()

        for track_id, track in sorted(self.tracks.items()):
            best_iou = 0.0
            best_idx = None
            for idx, det in enumerate(detections):
                if idx in used_detection_indexes:
                    continue
                iou = self._iou(track.bbox, det)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx

            if best_idx is not None and best_iou >= self.iou_threshold:
                det = detections[best_idx]
                assignments[track_id] = det
                used_detection_indexes.add(best_idx)
                track.bbox = det
                track.missed_frames = 0
            else:
                track.missed_frames += 1

        for idx, det in enumerate(detections):
            if idx in used_detection_indexes:
                continue
            self.tracks[self._next_id] = TrackState(
                bbox=det,
                smoothed_probs=np.zeros(len(EMOTION_LABELS), dtype=np.float32),
            )
            assignments[self._next_id] = det
            self._next_id += 1

        stale_tracks = [
            track_id for track_id, track in self.tracks.items() if track.missed_frames > self.max_missed
        ]
        for track_id in stale_tracks:
            self.tracks.pop(track_id, None)

        return assignments


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Advanced real-time face emotion detection")
    parser.add_argument("--camera", type=int, default=0, help="Webcam index")
    parser.add_argument("--model", default="emotion_model.h5", help="Path to emotion model (.h5)")
    parser.add_argument("--min-confidence", type=float, default=0.45, help="Top-1 confidence threshold")
    parser.add_argument(
        "--min-margin",
        type=float,
        default=0.15,
        help="Minimum top1-top2 probability margin to avoid uncertain labels",
    )
    parser.add_argument(
        "--smooth-alpha",
        type=float,
        default=0.55,
        help="EMA smoothing factor for probabilities (higher = more responsive)",
    )
    parser.add_argument("--frame-width", type=int, default=960)
    parser.add_argument("--frame-height", type=int, default=540)
    return parser.parse_args()


def load_face_detector() -> cv2.CascadeClassifier:
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    if face_cascade.empty():
        raise RuntimeError("Failed to load Haar cascade face detector")
    return face_cascade


def detect_faces(gray_frame: np.ndarray, detector: cv2.CascadeClassifier) -> List[Tuple[int, int, int, int]]:
    faces = detector.detectMultiScale(
        gray_frame,
        scaleFactor=1.05,
        minNeighbors=5,
        minSize=(48, 48),
        flags=cv2.CASCADE_SCALE_IMAGE,
    )
    return [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]


def preprocess_face(gray_frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
    x, y, w, h = bbox
    h_img, w_img = gray_frame.shape[:2]
    pad = int(0.12 * max(w, h))

    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(w_img, x + w + pad)
    y2 = min(h_img, y + h + pad)

    roi = gray_frame[y1:y2, x1:x2]
    if roi.size == 0:
        return None

    # Basic quality guardrails to suppress very poor face crops
    brightness = float(np.mean(roi))
    if brightness < 18 or brightness > 245:
        return None

    blur_score = cv2.Laplacian(roi, cv2.CV_64F).var()
    if blur_score < 25:
        return None

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    roi = clahe.apply(roi)
    roi = cv2.resize(roi, (64, 64), interpolation=cv2.INTER_AREA)
    roi = roi.astype("float32") / 255.0
    return np.expand_dims(np.expand_dims(roi, axis=-1), axis=0)


def classify_emotion(
    model,
    face_input: np.ndarray,
    prev_probs: np.ndarray,
    alpha: float,
) -> np.ndarray:
    probs = model.predict(face_input, verbose=0)[0].astype(np.float32)
    if prev_probs.sum() == 0:
        return probs
    return alpha * probs + (1 - alpha) * prev_probs


def format_prediction(
    probs: np.ndarray,
    min_confidence: float,
    min_margin: float,
) -> Tuple[str, float, int]:
    top_indices = np.argsort(probs)[::-1]
    top_idx, second_idx = int(top_indices[0]), int(top_indices[1])
    top_prob, second_prob = float(probs[top_idx]), float(probs[second_idx])

    if top_prob < min_confidence or (top_prob - second_prob) < min_margin:
        return (
            f"Uncertain ({EMOTION_LABELS[top_idx]} {top_prob * 100:.1f}%)",
            top_prob,
            top_idx,
        )

    return f"{EMOTION_LABELS[top_idx]} ({top_prob * 100:.1f}%)", top_prob, top_idx


def main() -> None:
    args = parse_args()

    detector = load_face_detector()
    emotion_model = load_model(args.model, compile=False)
    tracker = FaceTracker()

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.frame_height)

    if not cap.isOpened():
        raise RuntimeError("Webcam not accessible")

    fps_history: List[float] = []
    prev_tick = cv2.getTickCount()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detect_faces(gray, detector)
        assignments = tracker.match(faces)

        for track_id, bbox in assignments.items():
            face_input = preprocess_face(gray, bbox)
            if face_input is None:
                continue

            track = tracker.tracks[track_id]
            smoothed_probs = classify_emotion(
                emotion_model,
                face_input,
                track.smoothed_probs,
                alpha=args.smooth_alpha,
            )
            track.smoothed_probs = smoothed_probs

            label, confidence, top_idx = format_prediction(
                smoothed_probs,
                min_confidence=args.min_confidence,
                min_margin=args.min_margin,
            )

            x, y, w, h = bbox
            color = (0, 220, 0) if "Uncertain" not in label else (0, 165, 255)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(
                frame,
                f"ID {track_id}: {label}",
                (x, max(25, y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                color,
                2,
            )

            top2_indices = np.argsort(smoothed_probs)[::-1][:2]
            second_idx = int(top2_indices[1])
            second_prob = float(smoothed_probs[second_idx]) * 100
            cv2.putText(
                frame,
                f"Alt: {EMOTION_LABELS[second_idx]} {second_prob:.1f}%",
                (x, y + h + 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                (255, 255, 255),
                1,
            )

        current_tick = cv2.getTickCount()
        elapsed = (current_tick - prev_tick) / cv2.getTickFrequency()
        prev_tick = current_tick
        fps = 1.0 / max(elapsed, 1e-6)
        fps_history.append(fps)
        if len(fps_history) > 20:
            fps_history.pop(0)

        avg_fps = sum(fps_history) / len(fps_history)
        cv2.putText(
            frame,
            f"FPS: {avg_fps:.1f} | Faces: {len(assignments)} | Q: Quit",
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.68,
            (255, 255, 255),
            2,
        )

        cv2.imshow("Advanced Face Emotion Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
