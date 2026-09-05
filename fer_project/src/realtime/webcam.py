"""
Real-Time FER2013 Emotion Recognition via Webcam

Model:
    EfficientNet-B0

Checkpoint:
    saved_models/emotion_model_best.pth

Classes:
    angry, disgust, fear, happy, neutral, sad, surprise

Pipeline:
    Webcam -> Face Detection -> Face Crop
           -> Resize -> Normalize -> EfficientNet-B0
           -> Emotion Prediction -> Smoothed Display

Controls:
    Q / ESC : Quit
"""

from pathlib import Path
from collections import deque

import cv2
import torch
import torch.nn.functional as F
from torchvision import transforms, models


# ============================================================
# PROJECT PATHS
# ============================================================

# Project root:
# fer_project/
# ├── configs/
# ├── saved_models/
# ├── dataset/
# └── src/
#     └── realtime/
#         └── webcam.py

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "saved_models"
    / "emotion_model_best.pth"
)


# ============================================================
# CONFIGURATION
# ============================================================

CLASSES = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise",
]

INPUT_SIZE = 224

# Number of recent predictions used for smoothing.
# Higher = more stable but slightly slower to react.
SMOOTHING_FRAMES = 10

# Minimum confidence before displaying the predicted emotion.
CONFIDENCE_THRESHOLD = 0.35

# Webcam settings
CAMERA_INDEX = 0
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print("REAL-TIME FER2013 EMOTION RECOGNITION")
print("=" * 70)

print(f"\nDevice     : {device}")
print(f"Checkpoint : {CHECKPOINT_PATH}")


# ============================================================
# CHECKPOINT
# ============================================================

if not CHECKPOINT_PATH.exists():
    raise FileNotFoundError(
        f"\nCheckpoint not found:\n{CHECKPOINT_PATH}"
    )


# ============================================================
# MODEL
# ============================================================

print("\nLoading EfficientNet-B0...")

model = models.efficientnet_b0(
    weights=None
)

in_features = model.classifier[1].in_features

model.classifier[1] = torch.nn.Linear(
    in_features,
    len(CLASSES)
)


# ============================================================
# LOAD TRAINED WEIGHTS
# ============================================================

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=device
)

if (
    isinstance(checkpoint, dict)
    and "model_state_dict" in checkpoint
):
    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    print(
        f"Checkpoint epoch: "
        f"{checkpoint.get('epoch', 'unknown')}"
    )

    if "val_accuracy" in checkpoint:
        print(
            f"Validation accuracy: "
            f"{checkpoint['val_accuracy'] * 100:.2f}%"
        )

else:
    model.load_state_dict(checkpoint)

model = model.to(device)
model.eval()

print("Model loaded successfully.")


# ============================================================
# IMAGE TRANSFORMATION
# ============================================================

transform = transforms.Compose([
    transforms.ToPILImage(),

    transforms.Resize(
        (INPUT_SIZE, INPUT_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


# ============================================================
# FACE DETECTOR
# ============================================================

# OpenCV's built-in Haar cascade is used here only for
# face detection. The emotion model itself remains EfficientNet-B0.

cascade_path = (
    cv2.data.haarcascades
    + "haarcascade_frontalface_default.xml"
)

face_detector = cv2.CascadeClassifier(
    cascade_path
)

if face_detector.empty():
    raise RuntimeError(
        "Could not load OpenCV face detector."
    )


# ============================================================
# WEBCAM
# ============================================================

cap = cv2.VideoCapture(
    CAMERA_INDEX
)

if not cap.isOpened():
    raise RuntimeError(
        "Could not open webcam.\n"
        "Check that your webcam is connected and "
        "not being used by another application."
    )

cap.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    CAMERA_WIDTH
)

cap.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    CAMERA_HEIGHT
)


# ============================================================
# PREDICTION SMOOTHING
# ============================================================

prediction_history = deque(
    maxlen=SMOOTHING_FRAMES
)


# ============================================================
# REAL-TIME LOOP
# ============================================================

print("\nWebcam started.")
print("Press Q or ESC to quit.\n")


with torch.no_grad():

    while True:

        ret, frame = cap.read()

        if not ret:
            print("Could not read webcam frame.")
            break


        # ----------------------------------------------------
        # Mirror the webcam for a natural interaction view
        # ----------------------------------------------------

        frame = cv2.flip(
            frame,
            1
        )


        # ----------------------------------------------------
        # Convert to grayscale for face detection
        # ----------------------------------------------------

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )


        # ----------------------------------------------------
        # Detect faces
        # ----------------------------------------------------

        faces = face_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(80, 80)
        )


        # ----------------------------------------------------
        # Process each detected face
        # ----------------------------------------------------

        for (
            x,
            y,
            w,
            h
        ) in faces:

            # ------------------------------------------------
            # Add a little padding around face
            # ------------------------------------------------

            padding = int(
                0.15 * max(w, h)
            )

            x1 = max(
                0,
                x - padding
            )

            y1 = max(
                0,
                y - padding
            )

            x2 = min(
                frame.shape[1],
                x + w + padding
            )

            y2 = min(
                frame.shape[0],
                y + h + padding
            )


            # ------------------------------------------------
            # Crop face
            # ------------------------------------------------

            face = frame[
                y1:y2,
                x1:x2
            ]

            if face.size == 0:
                continue


            # ------------------------------------------------
            # Prepare image
            # ------------------------------------------------

            input_tensor = transform(
                face
            )

            input_tensor = (
                input_tensor
                .unsqueeze(0)
                .to(device)
            )


            # ------------------------------------------------
            # Model prediction
            # ------------------------------------------------

            outputs = model(
                input_tensor
            )

            probabilities = F.softmax(
                outputs,
                dim=1
            )[0]


            confidence, prediction = (
                probabilities.max(dim=0)
            )

            confidence = confidence.item()

            predicted_class = (
                prediction.item()
            )


            # ------------------------------------------------
            # Add prediction to smoothing history
            # ------------------------------------------------

            prediction_history.append(
                predicted_class
            )


            # ------------------------------------------------
            # Smoothed prediction
            # ------------------------------------------------

            if len(prediction_history) > 0:

                counts = torch.bincount(
                    torch.tensor(
                        list(prediction_history)
                    ),
                    minlength=len(CLASSES)
                )

                smoothed_class = (
                    counts.argmax().item()
                )

            else:

                smoothed_class = predicted_class


            emotion = CLASSES[
                smoothed_class
            ]


            # ------------------------------------------------
            # Display confidence
            # ------------------------------------------------

            if confidence >= CONFIDENCE_THRESHOLD:

                label = (
                    f"{emotion.upper()} "
                    f"{confidence * 100:.1f}%"
                )

            else:

                label = "UNCERTAIN"


            # ------------------------------------------------
            # Face rectangle
            # ------------------------------------------------

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (255, 255, 255),
                2
            )


            # ------------------------------------------------
            # Label background
            # ------------------------------------------------

            text_size = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                2
            )[0]

            text_width = text_size[0]
            text_height = text_size[1]

            label_y1 = max(
                0,
                y1 - text_height - 15
            )

            label_y2 = y1

            cv2.rectangle(
                frame,
                (
                    x1,
                    label_y1
                ),
                (
                    x1 + text_width + 12,
                    label_y2
                ),
                (0, 0, 0),
                -1
            )


            # ------------------------------------------------
            # Emotion text
            # ------------------------------------------------

            cv2.putText(
                frame,
                label,
                (
                    x1 + 6,
                    label_y2 - 7
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )


        # ====================================================
        # UI
        # ====================================================

        cv2.putText(
            frame,
            "FER2013 Real-Time Emotion Recognition",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            frame,
            "Press Q or ESC to quit",
            (20, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (220, 220, 220),
            1,
            cv2.LINE_AA
        )


        # ====================================================
        # SHOW FRAME
        # ====================================================

        cv2.imshow(
            "Mindo - Real-Time Emotion Recognition",
            frame
        )


        # ====================================================
        # KEYBOARD CONTROL
        # ====================================================

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q") or key == 27:
            break


# ============================================================
# CLEANUP
# ============================================================

cap.release()
cv2.destroyAllWindows()

print("\nWebcam stopped.")
print("Real-time evaluation complete.")