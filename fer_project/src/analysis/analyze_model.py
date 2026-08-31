"""Analyze the best fine-tuned EfficientNet-B0 checkpoint on validation data."""

import torch
import torch.nn as nn
from sklearn.metrics import (
    precision_recall_fscore_support,
    confusion_matrix,
)

from src.config_loader import load_config
from src.data.dataset import build_dataloaders
from src.model.efficientnet_b0 import build_model


# =========================================================
# DEVICE
# =========================================================

def get_device(cfg):
    """Select CUDA when requested and available."""

    if (
        cfg.training.device == "cuda"
        and torch.cuda.is_available()
    ):
        return torch.device("cuda")

    return torch.device("cpu")


# =========================================================
# MAIN ANALYSIS
# =========================================================

@torch.no_grad()
def analyze():

    cfg = load_config()

    device = get_device(cfg)

    print("=" * 70)
    print("EFFICIENTNET-B0 VALIDATION ANALYSIS")
    print("=" * 70)

    print(f"\nUsing device: {device}")

    if device.type == "cuda":
        print(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )

    # -----------------------------------------------------
    # DATA
    # -----------------------------------------------------

    print("\n" + "=" * 70)
    print("LOADING VALIDATION DATASET")
    print("=" * 70)

    (
        _,
        val_loader,
        _,
        classes
    ) = build_dataloaders(cfg)

    print(f"\nClasses: {classes}")
    print(
        f"Validation batches: "
        f"{len(val_loader)}"
    )

    # -----------------------------------------------------
    # MODEL
    # -----------------------------------------------------

    print("\n" + "=" * 70)
    print("LOADING BEST MODEL")
    print("=" * 70)

    model = build_model(cfg).to(device)

    checkpoint_path = (
        "saved_models/emotion_model_finetuned.pth"
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device
    )

    # Handle our complete training checkpoint.
    if "model_state_dict" in checkpoint:
        model.load_state_dict(
            checkpoint["model_state_dict"]
        )
    else:
        # Also support a plain state_dict checkpoint.
        model.load_state_dict(checkpoint)

    model.eval()

    print(f"\nCheckpoint: {checkpoint_path}")

    if isinstance(checkpoint, dict):
        if "best_epoch" in checkpoint:
            print(
                f"Best epoch recorded: "
                f"{checkpoint['best_epoch']}"
            )

        if "val_acc" in checkpoint:
            print(
                f"Checkpoint validation accuracy: "
                f"{checkpoint['val_acc'] * 100:.2f}%"
            )

    # -----------------------------------------------------
    # STANDARD LOSS
    # -----------------------------------------------------

    # Do NOT use the training class weights here.
    # We want an unbiased evaluation loss.

    criterion = nn.CrossEntropyLoss()

    # -----------------------------------------------------
    # EVALUATION
    # -----------------------------------------------------

    print("\n" + "=" * 70)
    print("RUNNING VALIDATION ANALYSIS")
    print("=" * 70)

    running_loss = 0.0
    correct = 0
    total = 0

    all_preds = []
    all_labels = []

    for images, labels in val_loader:

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
            loss.item() * images.size(0)
        )

        predictions = outputs.argmax(
            dim=1
        )

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

        all_preds.extend(
            predictions.cpu().tolist()
        )

        all_labels.extend(
            labels.cpu().tolist()
        )

    # -----------------------------------------------------
    # OVERALL METRICS
    # -----------------------------------------------------

    val_loss = running_loss / total
    val_accuracy = correct / total

    (
        precision,
        recall,
        f1,
        support
    ) = precision_recall_fscore_support(
        all_labels,
        all_preds,
        labels=list(range(len(classes))),
        average=None,
        zero_division=0
    )

    (
        macro_precision,
        macro_recall,
        macro_f1,
        _
    ) = precision_recall_fscore_support(
        all_labels,
        all_preds,
        labels=list(range(len(classes))),
        average="macro",
        zero_division=0
    )

    cm = confusion_matrix(
        all_labels,
        all_preds,
        labels=list(range(len(classes)))
    )

    # -----------------------------------------------------
    # OVERALL RESULTS
    # -----------------------------------------------------

    print("\n" + "=" * 70)
    print("OVERALL VALIDATION RESULTS")
    print("=" * 70)

    print(
        f"\nValidation Loss      : "
        f"{val_loss:.4f}"
    )

    print(
        f"Validation Accuracy  : "
        f"{val_accuracy * 100:.2f}%"
    )

    print(
        f"Macro Precision      : "
        f"{macro_precision:.4f}"
    )

    print(
        f"Macro Recall         : "
        f"{macro_recall:.4f}"
    )

    print(
        f"Macro F1             : "
        f"{macro_f1:.4f}"
    )

    # -----------------------------------------------------
    # PER-CLASS RESULTS
    # -----------------------------------------------------

    print("\n" + "=" * 70)
    print("PER-CLASS RESULTS")
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

    # -----------------------------------------------------
    # PREDICTION DISTRIBUTION
    # -----------------------------------------------------

    print("\n" + "=" * 70)
    print("PREDICTION DISTRIBUTION")
    print("=" * 70)

    print(
        f"\n{'Class':<12}"
        f"{'Actual':>12}"
        f"{'Predicted':>12}"
        f"{'Difference':>12}"
    )

    print("-" * 48)

    for i, class_name in enumerate(classes):

        actual_count = sum(
            1 for label in all_labels
            if label == i
        )

        predicted_count = sum(
            1 for pred in all_preds
            if pred == i
        )

        difference = predicted_count - actual_count

        print(
            f"{class_name:<12}"
            f"{actual_count:>12}"
            f"{predicted_count:>12}"
            f"{difference:>12}"
        )

    # -----------------------------------------------------
    # CONFUSION MATRIX
    # -----------------------------------------------------

    print("\n" + "=" * 70)
    print("CONFUSION MATRIX")
    print("=" * 70)

    print(
        "\nRows = Actual"
        "\nColumns = Predicted\n"
    )

    print(
        f"{'':<12}"
        + "".join(
            f"{c[:8]:>10}"
            for c in classes
        )
    )

    for i, row in enumerate(cm):

        print(
            f"{classes[i]:<12}"
            + "".join(
                f"{value:>10d}"
                for value in row
            )
        )

    # -----------------------------------------------------
    # MOST CONFUSED PAIRS
    # -----------------------------------------------------

    print("\n" + "=" * 70)
    print("MOST CONFUSED EMOTION PAIRS")
    print("=" * 70)

    confusion_pairs = []

    for i in range(len(classes)):
        for j in range(len(classes)):

            if i == j:
                continue

            confusion_pairs.append(
                (
                    cm[i, j],
                    classes[i],
                    classes[j]
                )
            )

    confusion_pairs.sort(
        reverse=True
    )

    print()

    for count, actual, predicted in confusion_pairs[:10]:

        print(
            f"{actual:<10} → "
            f"{predicted:<10}: "
            f"{count} samples"
        )

    # -----------------------------------------------------
    # WORST CLASSES
    # -----------------------------------------------------

    print("\n" + "=" * 70)
    print("LOWEST F1 CLASSES")
    print("=" * 70)

    worst_classes = sorted(
        range(len(classes)),
        key=lambda i: f1[i]
    )

    print()

    for i in worst_classes:

        print(
            f"{classes[i]:<12}"
            f"F1={f1[i]:.4f}  "
            f"Recall={recall[i]:.4f}  "
            f"Precision={precision[i]:.4f}"
        )

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":
    analyze()