"""Real-time webcam emotion detection (Phase 9-11). Entry: /app.py"""
import time
import cv2
import torch
import numpy as np

from src.config_loader import load_config
from src.data.transforms import get_eval_transforms
from src.inference.predict import load_trained_model
from src.inference.face_detector import FaceDetector
from src.inference.temporal_smoothing import TemporalSmoother
from src.backend.api import send_emotion_snapshot


def run_webcam(session_id="local-session", checkpoint_path=None):
    cfg = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = checkpoint_path or f"{cfg.paths.checkpoint_dir}/emotion_model_best.pth"

    model, classes = load_trained_model(cfg, checkpoint_path, device)
    transform = get_eval_transforms(cfg)
    detector = FaceDetector(min_detection_confidence=cfg.inference.face_detection_confidence)
    smoother = TemporalSmoother(classes, window_size=cfg.inference.smoothing_window)

    cap = cv2.VideoCapture(cfg.inference.webcam_index)
    frame_count = 0

    print("Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        if frame_count % cfg.inference.frame_skip == 0:
            face = detector.detect_and_crop(frame)
            if face is not None:
                rgb_face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                from PIL import Image
                pil_face = Image.fromarray(rgb_face)
                tensor = transform(pil_face).unsqueeze(0).to(device)

                with torch.no_grad():
                    logits = model(tensor)
                    probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

                label, smoothed_probs = smoother.update(probs)

                cv2.putText(frame, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                            1.0, (0, 255, 0), 2)

                send_emotion_snapshot(
                    session_id=session_id,
                    timestamp=time.time(),
                    probabilities={c: float(p) for c, p in zip(classes, smoothed_probs)},
                )

        cv2.imshow("Facial Emotion Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    detector.close()


if __name__ == "__main__":
    run_webcam()
