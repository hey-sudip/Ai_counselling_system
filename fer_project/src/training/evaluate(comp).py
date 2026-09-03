"""
Comparison Evaluation Script for FER2013.

Evaluates the OLD 30-epoch baseline checkpoint
on the SAME untouched FER2013 test set used for
the final model evaluation.

Purpose:
    Compare the old baseline model against the new
    final model fairly using identical test data,
    preprocessing, class order, and metrics.

Old checkpoint:
    saved_models/baseline_70_30.pth

Classes:
    angry, disgust, fear, happy, neutral, sad, surprise

Important:
    - Test set is used ONLY for evaluation.
    - No training is performed.
    - No model selection is performed.
    - num_workers=0 is used for Windows compatibility.
"""

from pathlib import Path
import torch
import yaml

from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

from sklearn.metrics import (
    precision_recall_fscore_support,
    confusion_matrix,
)


# ============================================================
# PATHS
# ============================================================

# Project root:
# fer_project/
# ├── configs/
# ├── dataset/
# ├── saved_models/
# └── src/
#     └── training/
#         └── evaluate(comp).py

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"

OLD_CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "saved_models"
    / "baseline_70_30.pth"
)


# ============================================================
# LOAD CONFIG
# ============================================================

with open(CONFIG_PATH, "r") as f:
    cfg = yaml.safe_load(f)


# ============================================================
# DEVICE
# ============================================================

device_name = cfg["training"]["device"]

if device_name == "cuda" and not torch.cuda.is_available():
    print("CUDA requested but not available.")
    print("Falling back to CPU.")
    device = torch.device("cpu")
else:
    device = torch.device(device_name)


# ============================================================
# CLASSES
# ============================================================

classes = cfg["classes"]
num_classes = len(classes)


# ============================================================
# DATASET PATH
# ============================================================

test_dir = PROJECT_ROOT / cfg["paths"]["test_dir"]


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("FER2013 OLD BASELINE MODEL EVALUATION")
print("=" * 70)

print(f"\nDevice       : {device}")
print(f"Classes      : {classes}")
print(f"Test dataset : {test_dir}")
print(f"Checkpoint   : {OLD_CHECKPOINT_PATH}")


# ============================================================
# TEST TRANSFORM
# ============================================================

input_size = cfg["model"]["input_size"]

test_transform = transforms.Compose([
    transforms.Resize((input_size, input_size)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


# ============================================================
# TEST DATASET
# ============================================================

test_dataset = datasets.ImageFolder(
    root=test_dir,
    transform=test_transform
)

print(f"Test samples : {len(test_dataset)}")


# ============================================================
# VERIFY CLASS ORDER
# ============================================================

if test_dataset.classes != classes:
    raise ValueError(
        "\nClass order mismatch!\n"
        f"Config classes : {classes}\n"
        f"Dataset classes: {test_dataset.classes}\n"
        "\nMake sure the test folders match the training class order."
    )


# ============================================================
# TEST DATALOADER
# ============================================================

test_loader = DataLoader(
    test_dataset,
    batch_size=cfg["training"]["batch_size"],
    shuffle=False,
    num_workers=0,
    pin_memory=(device.type == "cuda")
)


# ============================================================
# CHECKPOINT
# ============================================================

if not OLD_CHECKPOINT_PATH.exists():
    raise FileNotFoundError(
        f"\nOld checkpoint not found:\n"
        f"{OLD_CHECKPOINT_PATH}"
    )


# ============================================================
# CREATE MODEL
# ============================================================

model_name = cfg["model"]["name"]

if model_name != "efficientnet_b0":
    raise ValueError(
        f"This evaluation script expects efficientnet_b0, "
        f"but config specifies: {model_name}"
    )


model = models.efficientnet_b0(
    weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1
)

in_features = model.classifier[1].in_features

model.classifier[1] = torch.nn.Linear(
    in_features,
    num_classes
)


# ============================================================
# LOAD OLD CHECKPOINT
# ============================================================

print("\nLoading old baseline checkpoint...")

checkpoint = torch.load(
    OLD_CHECKPOINT_PATH,
    map_location=device
)

if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    print(
        f"Checkpoint epoch: "
        f"{checkpoint.get('epoch', 'unknown')}"
    )

    print(
        f"Checkpoint val acc: "
        f"{checkpoint.get('val_accuracy', 'unknown')}"
    )

else:

    # Supports checkpoints that contain only
    # the model state dictionary.

    model.load_state_dict(checkpoint)

    print("Checkpoint format: raw model state_dict")


model = model.to(device)
model.eval()


# ============================================================
# CLASS WEIGHTS
# ============================================================
#
# This is kept identical to the final evaluation script.
#
# IMPORTANT:
# Class weights affect TEST LOSS only.
# Accuracy, precision, recall, F1 and confusion matrix
# are calculated directly from predictions and labels.
#

train_dir = PROJECT_ROOT / cfg["paths"]["train_dir"]

train_dataset_for_weights = datasets.ImageFolder(
    root=train_dir
)

train_targets = torch.tensor(
    train_dataset_for_weights.targets,
    dtype=torch.long
)

class_counts = torch.bincount(
    train_targets,
    minlength=num_classes
).float()

class_weights = 1.0 / torch.sqrt(class_counts)

class_weights = (
    class_weights /
    class_weights.mean()
)

criterion = torch.nn.CrossEntropyLoss(
    weight=class_weights.to(device),
    label_smoothing=0.1
)


# ============================================================
# EVALUATION FUNCTION
# ============================================================

@torch.no_grad()
def evaluate(
    model,
    dataloader,
    criterion,
    device,
    num_classes
):

    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    all_preds = []
    all_labels = []

    for images, labels in dataloader:

        images = images.to(
            device,
            non_blocking=True
        )

        labels = labels.to(
            device,
            non_blocking=True
        )

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        running_loss += (
            loss.item() *
            images.size(0)
        )

        preds = outputs.argmax(
            dim=1
        )

        correct += (
            (preds == labels)
            .sum()
            .item()
        )

        total += labels.size(0)

        all_preds.extend(
            preds.cpu()
            .numpy()
            .tolist()
        )

        all_labels.extend(
            labels.cpu()
            .numpy()
            .tolist()
        )


    avg_loss = running_loss / total

    accuracy = correct / total


    # --------------------------------------------------------
    # Per-class metrics
    # --------------------------------------------------------

    (
        precision,
        recall,
        f1,
        support
    ) = precision_recall_fscore_support(
        all_labels,
        all_preds,
        labels=list(range(num_classes)),
        average=None,
        zero_division=0
    )


    # --------------------------------------------------------
    # Macro metrics
    # --------------------------------------------------------

    (
        macro_precision,
        macro_recall,
        macro_f1,
        _
    ) = precision_recall_fscore_support(
        all_labels,
        all_preds,
        labels=list(range(num_classes)),
        average="macro",
        zero_division=0
    )


    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    cm = confusion_matrix(
        all_labels,
        all_preds,
        labels=list(range(num_classes))
    )


    return (
        avg_loss,
        accuracy,
        macro_precision,
        macro_recall,
        macro_f1,
        cm,
        precision,
        recall,
        f1,
        support
    )


# ============================================================
# RUN TEST EVALUATION
# ============================================================

print("\n" + "-" * 70)
print("Running evaluation on TEST SET...")
print("-" * 70)


(
    test_loss,
    test_accuracy,
    macro_precision,
    macro_recall,
    macro_f1,
    cm,
    precision,
    recall,
    f1,
    support
) = evaluate(
    model,
    test_loader,
    criterion,
    device,
    num_classes
)


# ============================================================
# FINAL RESULTS
# ============================================================

print("\n" + "=" * 70)
print("OLD BASELINE TEST RESULTS")
print("=" * 70)

print(f"\nTest Loss       : {test_loss:.4f}")
print(f"Test Accuracy   : {test_accuracy * 100:.2f}%")
print(f"Macro Precision : {macro_precision:.4f}")
print(f"Macro Recall    : {macro_recall:.4f}")
print(f"Macro F1        : {macro_f1:.4f}")


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

print(
    f"\n{'Class':<12}"
    f"{'Precision':>12}"
    f"{'Recall':>12}"
    f"{'F1':>12}"
    f"{'Support':>12}"
)

print("-" * 60)


for i, class_name in enumerate(classes):

    print(
        f"{class_name:<12}"
        f"{precision[i]:>12.4f}"
        f"{recall[i]:>12.4f}"
        f"{f1[i]:>12.4f}"
        f"{int(support[i]):>12}"
    )


# ============================================================
# CONFUSION MATRIX
# ============================================================

print("\n" + "=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

print(
    f"\n{'Actual ↓ / Predicted →':<24}"
    + "".join(
        f"{class_name[:8]:>10}"
        for class_name in classes
    )
)

print(
    "-" * (
        24 +
        10 * num_classes
    )
)


for i, row in enumerate(cm):

    print(
        f"{classes[i]:<24}"
        + "".join(
            f"{value:>10d}"
            for value in row
        )
    )


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("OLD BASELINE EVALUATION COMPLETE")
print("=" * 70)

print("\nImportant:")
print("- Test set was used only for evaluation.")
print("- No training was performed.")
print("- No model selection was performed.")
print("- Same test set and preprocessing as final evaluation.")
print(f"- Old checkpoint: {OLD_CHECKPOINT_PATH}")
print("\n")