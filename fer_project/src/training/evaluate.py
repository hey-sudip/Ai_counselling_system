"""Evaluation utilities for FER model analysis."""

import torch
from sklearn.metrics import (
    precision_recall_fscore_support,
    confusion_matrix,
)


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    """
    Evaluate model on a dataloader.

    Returns:
        avg_loss
        accuracy
        macro_precision
        macro_recall
        macro_f1
        confusion_matrix
        per_class_precision
        per_class_recall
        per_class_f1
        per_class_support
    """

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
            loss.item() * images.size(0)
        )

        preds = outputs.argmax(dim=1)

        correct += (
            preds == labels
        ).sum().item()

        total += labels.size(0)

        all_preds.extend(
            preds.cpu().numpy().tolist()
        )

        all_labels.extend(
            labels.cpu().numpy().tolist()
        )

    avg_loss = running_loss / total
    accuracy = correct / total

    (
        precision,
        recall,
        f1,
        support
    ) = precision_recall_fscore_support(
        all_labels,
        all_preds,
        labels=list(range(7)),
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
        labels=list(range(7)),
        average="macro",
        zero_division=0
    )

    cm = confusion_matrix(
        all_labels,
        all_preds,
        labels=list(range(7))
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
        support,
    )


def print_report(
    precision,
    recall,
    f1,
    cm,
    classes,
    support=None
):
    """Print detailed per-class evaluation report."""

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

        support_value = (
            int(support[i])
            if support is not None
            else "-"
        )

        print(
            f"{class_name:<12}"
            f"{precision[i]:>12.4f}"
            f"{recall[i]:>12.4f}"
            f"{f1[i]:>12.4f}"
            f"{support_value:>12}"
        )

    print("-" * 60)

    print("\nConfusion Matrix:")

    print(
        f"{'Actual ↓ / Predicted →':<24}"
        + "".join(
            f"{class_name[:8]:>10}"
            for class_name in classes
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