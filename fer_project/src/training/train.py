"""Training loop (Phase 5). Entry point: /train.py at project root."""
import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from src.config_loader import load_config
from src.data.dataset import build_dataloaders
from src.model.efficientnet_b0 import build_model
from src.training.evaluate import evaluate


def get_device(cfg):
    if cfg.training.device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train():
    cfg = load_config()
    device = get_device(cfg)
    print(f"Using device: {device}")

    train_loader, val_loader, _, classes = build_dataloaders(cfg)
    print(f"Classes: {classes}")

    model = build_model(cfg).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(
        model.parameters(),
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.training.num_epochs)

    os.makedirs(cfg.paths.checkpoint_dir, exist_ok=True)
    best_val_acc = 0.0
    epochs_no_improve = 0

    for epoch in range(1, cfg.training.num_epochs + 1):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        loop = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.training.num_epochs}")
        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            loop.set_postfix(loss=loss.item())

        scheduler.step()
        train_loss = running_loss / total
        train_acc = correct / total

        val_loss, val_acc, _, _, _ = evaluate(model, val_loader, criterion, device)
        print(f"Epoch {epoch}: train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            ckpt_path = os.path.join(cfg.paths.checkpoint_dir, "emotion_model_best.pth")
            torch.save({
                "model_state_dict": model.state_dict(),
                "classes": classes,
                "val_acc": val_acc,
                "epoch": epoch,
            }, ckpt_path)
            print(f"  -> Saved new best model ({val_acc:.4f}) to {ckpt_path}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= cfg.training.early_stopping_patience:
                print("Early stopping triggered.")
                break

    last_path = os.path.join(cfg.paths.checkpoint_dir, "emotion_model_last.pth")
    torch.save({"model_state_dict": model.state_dict(), "classes": classes}, last_path)
    print(f"Training complete. Best val_acc={best_val_acc:.4f}")


if __name__ == "__main__":
    train()
