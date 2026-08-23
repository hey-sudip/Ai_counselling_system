"""Evaluation: loss, accuracy, precision/recall/F1, confusion matrix (Phase 6)."""
import torch
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        all_preds.extend(preds.cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())

    avg_loss = running_loss / total
    acc = correct / total
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="macro", zero_division=0
    )
    cm = confusion_matrix(all_labels, all_preds)

    return avg_loss, acc, precision, recall, cm


def print_report(precision, recall, f1, cm, classes):
    print(f"Precision: {precision:.4f}  Recall: {recall:.4f}  F1: {f1:.4f}")
    print("Confusion matrix:")
    print(f"{'':10s}" + "".join(f"{c[:6]:>7s}" for c in classes))
    for i, row in enumerate(cm):
        print(f"{classes[i][:9]:10s}" + "".join(f"{v:7d}" for v in row))
