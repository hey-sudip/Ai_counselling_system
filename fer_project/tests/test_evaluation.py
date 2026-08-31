import torch
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
)

from src.config_loader import load_config
from src.data.dataset import build_dataloaders
from src.model.efficientnet_b0 import build_model


def main():
    cfg = load_config()

    device = torch.device(
        "cuda" if cfg.training.device == "cuda"
        and torch.cuda.is_available()
        else "cpu"
    )

    print(f"Using device: {device}")

    # Build dataloaders
    _, _, test_loader, classes = build_dataloaders(cfg)

    # Build model
    model = build_model(cfg).to(device)

    # Load best checkpoint
    checkpoint_path = "saved_models/emotion_model_finetuned.pth"

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    print(
        f"Loaded checkpoint from epoch: "
        f"{checkpoint.get('epoch', 'unknown')}"
    )

    print(
        f"Validation accuracy stored in checkpoint: "
        f"{checkpoint.get('val_acc', 'unknown')}"
    )

    model.eval()

    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, labels in test_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            predictions = outputs.argmax(dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())

    # Accuracy
    test_accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)

    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

    # Classification report
    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)

    print(
        classification_report(
            all_labels,
            all_predictions,
            target_names=classes,
            digits=4,
            zero_division=0
        )
    )

    # Confusion matrix
    cm = confusion_matrix(
        all_labels,
        all_predictions
    )

    print("\n" + "=" * 60)
    print("CONFUSION MATRIX")
    print("=" * 60)

    print("Classes:")
    print(classes)

    print()

    for i, row in enumerate(cm):
        print(
            f"{classes[i]:10s}: {row.tolist()}"
        )


if __name__ == "__main__":
    main()