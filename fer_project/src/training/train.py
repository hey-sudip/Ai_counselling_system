"""
src/training/train.py

Facial Emotion Recognition (FER) - Version 2 Baseline Training
AI Mental Wellness / Counselling System

Model:      EfficientNet-B0 (torchvision, ImageNet pretrained)
Strategy:   Full fine-tuning (unfreeze_all() - no staged unfreezing yet)
Optimizer:  AdamW with separate backbone / classifier parameter groups
Scheduler:  CosineAnnealingLR
Loss:       CrossEntropyLoss with moderated sqrt inverse-frequency
class weighting

Configuration is owned entirely by config.yaml via
src/config_loader.py. This file does not define, override, or
argparse any hyperparameters - every value below is read from `cfg`.

Validation is delegated to src/training/evaluate.py::evaluate(), which
is called once per epoch. There is no second validation loop here.

The test set is loaded (by build_dataloaders) but is never referenced
past the point it's discarded below. It is only used later, standalone,
after training has finished.

--------------------------------------------------------------------
Expected config.yaml schema (read by this file):
--------------------------------------------------------------------
seed: 42

paths:
  train_dir: ...
  val_dir: ...
  test_dir: ...
  checkpoint_dir: ...

model:
  input_size: 224
  num_classes: 7
  pretrained: true

training:
  batch_size: ...
  num_workers: ...
  num_epochs: ...
  early_stopping_patience: ...
  grad_clip_norm: ...
  backbone_learning_rate: ...
  learning_rate: ...          # classifier LR
  weight_decay: ...
  device: "cuda" | "cpu" | "auto"

augmentation:
  horizontal_flip_prob: ...
  rotation_degrees: ...
  brightness: ...
  contrast: ...
--------------------------------------------------------------------
"""

import time
import random
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from src.config_loader import load_config
from src.data.dataset import build_dataloaders
from src.model.efficientnet_b0 import (
    build_model,
    unfreeze_all,
    count_trainable_parameters,
)
from src.training.evaluate import evaluate, print_report

# Console colour support (PowerShell / Windows Terminal friendly).
# colorama translates ANSI codes for legacy cmd.exe and is a no-op on
# terminals that already support ANSI, so it's safe cross-platform.
# Falls back to plain text if colorama isn't installed - purely
# cosmetic, never a hard dependency for training itself.
try:
    from colorama import init as _colorama_init, Fore, Style
    _colorama_init(autoreset=True)
except ImportError:
    class _NoColor:
        def __getattr__(self, _name):
            return ""
    Fore = _NoColor()
    Style = _NoColor()


# --------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("train")


# --------------------------------------------------------------------------
# Presentation helpers
# --------------------------------------------------------------------------
def pct(value: float) -> str:
    """0.4732 -> '47.32%' - used for end-of-epoch summaries."""
    return f"{value * 100:.2f}%"


def print_rule(char: str = "=", width: int = 78) -> None:
    print(Fore.BLUE + char * width + Style.RESET_ALL)


def print_banner(title: str) -> None:
    print_rule()
    print(Fore.CYAN + Style.BRIGHT + title.center(78) + Style.RESET_ALL)
    print_rule()


# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Reproducibility over raw speed, as before.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(device_setting: str) -> torch.device:
    if device_setting == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info("Auto-selected GPU: %s", torch.cuda.get_device_name(device))
        else:
            device = torch.device("cpu")
            logger.info("Auto-selected CPU (CUDA not available).")
        return device

    device = torch.device(device_setting)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            f"config.yaml requests device='{device_setting}' but CUDA is "
            f"not available on this machine."
        )
    if device.type == "cuda":
        logger.info("Using GPU: %s", torch.cuda.get_device_name(device))
    else:
        logger.info("Using CPU.")
    return device


# --------------------------------------------------------------------------
# Class weighting: moderated square-root inverse frequency
# --------------------------------------------------------------------------
def compute_class_weights(train_dataset, num_classes: int) -> tuple:
    """
    Moderated square-root inverse-frequency class weighting.

    weight[c] = 1 / sqrt(count[c])
    weights are then normalized so they sum to num_classes
    (keeps the effective loss scale comparable to unweighted CE).

    Returns:
        (weights, counts) - the normalized weight tensor and the raw
        per-class training sample counts, so callers can log both.
    """
    counts = np.zeros(num_classes, dtype=np.float64)

    if hasattr(train_dataset, "targets"):
        targets = train_dataset.targets
    elif hasattr(train_dataset, "samples"):
        targets = [label for _, label in train_dataset.samples]
    else:
        raise AttributeError(
            "train_dataset must expose either `.targets` or `.samples` "
            "(as produced by torchvision.datasets.ImageFolder) to compute "
            "class weights."
        )

    for label in targets:
        counts[label] += 1

    if np.any(counts == 0):
        zero_classes = np.where(counts == 0)[0].tolist()
        raise ValueError(
            f"Class(es) {zero_classes} have zero samples in the training "
            f"set. Cannot compute inverse-frequency weights."
        )

    inv_sqrt = 1.0 / np.sqrt(counts)
    weights = inv_sqrt / inv_sqrt.sum() * num_classes

    return torch.tensor(weights, dtype=torch.float32), counts

# --------------------------------------------------------------------------
# Optimizer parameter groups (differential LR)
# --------------------------------------------------------------------------
def build_optimizer(model: nn.Module, backbone_lr: float, classifier_lr: float,
                     weight_decay: float) -> AdamW:
    """
    Splits parameters into 'classifier' (model.classifier, i.e. the
    Dropout + Linear head) and 'backbone' (model.features, i.e.
    everything else) groups so each can use a different learning rate
    under full fine-tuning.
    """
    backbone_params = []
    classifier_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("classifier"):
            classifier_params.append(param)
        else:
            backbone_params.append(param)

    if not classifier_params:
        raise ValueError(
            "No parameters matched the 'classifier' name prefix. Check "
            "that build_model() exposes the final head as "
            "`model.classifier` (torchvision EfficientNet-B0 default)."
        )
    if not backbone_params:
        raise ValueError(
            "No backbone parameters found with requires_grad=True. "
            "This run expects full fine-tuning - make sure unfreeze_all() "
            "was applied."
        )

    param_groups = [
        {"params": backbone_params, "lr": backbone_lr, "name": "backbone"},
        {"params": classifier_params, "lr": classifier_lr, "name": "classifier"},
    ]

    return AdamW(param_groups, weight_decay=weight_decay)


def get_current_lrs(optimizer: AdamW) -> dict:
    return {
        group.get("name", f"group_{i}"): group["lr"]
        for i, group in enumerate(optimizer.param_groups)
    }


# --------------------------------------------------------------------------
# Train for one epoch (validation lives in evaluate.py, not here)
# --------------------------------------------------------------------------
def run_train_epoch(model, dataloader, criterion, optimizer, device,
                     grad_clip_norm: float, epoch: int, total_epochs: int) -> tuple:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    progress = tqdm(
        dataloader,
        desc=f"Epoch {epoch}/{total_epochs} [train]",
        leave=False,
    )

    for images, labels in progress:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)

        optimizer.step()

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += batch_size

        progress.set_postfix(
            loss=f"{running_loss / total:.4f}",
            acc=f"{correct / total:.4f}",
        )

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


# --------------------------------------------------------------------------
# Checkpointing
# --------------------------------------------------------------------------
def save_checkpoint(path, model, optimizer, scheduler, epoch: int,
                     best_epoch: int, val_acc: float, val_loss: float,
                     class_names: list, class_weights: torch.Tensor) -> None:
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "epoch": epoch,
        "best_epoch": best_epoch,
        "val_accuracy": val_acc,
        "val_loss": val_loss,
        "classes": class_names,
        "class_weights": class_weights.cpu(),
        "training_strategy": "full_fine_tuning",
        "dataset_version": "FER2013_CLEAN_v1",
    }
    torch.save(checkpoint, path)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main() -> None:
    cfg = load_config()

    set_seed(cfg.seed)
    logger.info("Random seed set to %d", cfg.seed)

    device = get_device(cfg.training.device)

    checkpoint_dir = Path(cfg.paths.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = checkpoint_dir / "emotion_model_best.pth"
    last_ckpt_path = checkpoint_dir / "emotion_model_last.pth"

    # ----------------------------------------------------------------
    # Data - dataset.py / transforms.py read cfg.paths, cfg.training,
    # cfg.model and cfg.augmentation directly, so cfg is passed through
    # unmodified.
    # ----------------------------------------------------------------
    logger.info("Building dataloaders from %s", cfg.paths.train_dir)
    train_loader, val_loader, test_loader, class_names = build_dataloaders(cfg)
    # The test loader is intentionally never used past this point.
    del test_loader

    num_classes = len(class_names)
    if num_classes != cfg.model.num_classes:
        logger.warning(
            "Number of classes found in dataset (%d) does not match "
            "cfg.model.num_classes (%d). The model head will use "
            "cfg.model.num_classes as configured - update config.yaml "
            "if this is unintended.",
            num_classes, cfg.model.num_classes,
        )
    logger.info("Classes (%d): %s", num_classes, class_names)

    # ----------------------------------------------------------------
    # Model - full fine-tuning, nothing frozen. Staged fine-tuning
    # helpers exist in efficientnet_b0.py but are not used yet.
    # ----------------------------------------------------------------
    model = build_model(cfg)
    unfreeze_all(model)
    model.to(device)

    trainable_params = count_trainable_parameters(model)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(
        "Parameters - total: %s | trainable: %s (full fine-tuning)",
        f"{total_params:,}", f"{trainable_params:,}",
    )

    # ----------------------------------------------------------------
    # Class weights (moderated sqrt inverse frequency)
    # ----------------------------------------------------------------
    class_weights, class_counts = compute_class_weights(train_loader.dataset, num_classes)
    class_weights = class_weights.to(device)
    logger.info(
        "Training class distribution: %s",
        {name: int(c) for name, c in zip(class_names, class_counts)},
    )
    logger.info(
        "Class weights (sqrt inverse-frequency): %s",
        {name: round(w, 4) for name, w in zip(class_names, class_weights.tolist())},
    )

    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=0.1
    )
    # ----------------------------------------------------------------
    # Optimizer + scheduler - every value from cfg.training
    # ----------------------------------------------------------------
    optimizer = build_optimizer(
        model,
        backbone_lr=cfg.training.backbone_learning_rate,
        classifier_lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
    )
    # No scheduler section in config.yaml - CosineAnnealingLR's default
    # eta_min=0 is used as-is rather than introducing a new config key.
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=cfg.training.num_epochs * 2
    )

    best_val_acc = 0.0
    best_epoch = 0
    epochs_without_improvement = 0

    # ----------------------------------------------------------------
    # Training loop
    # ----------------------------------------------------------------
    print_banner("TRAINING START - FER V2 Baseline (full fine-tuning)")
    print(
        f"{Fore.WHITE}Epochs: {cfg.training.num_epochs}  |  "
        f"Patience: {cfg.training.early_stopping_patience}  |  "
        f"Device: {device}{Style.RESET_ALL}\n"
    )
    training_start = time.time()

    for epoch in range(1, cfg.training.num_epochs + 1):
        epoch_start = time.time()

        train_loss, train_acc = run_train_epoch(
            model, train_loader, criterion, optimizer, device,
            grad_clip_norm=cfg.training.grad_clip_norm,
            epoch=epoch, total_epochs=cfg.training.num_epochs,
        )

        # Validation is fully delegated to evaluate.py - no second
        # validation loop lives in this file.
        (
            val_loss,
            val_acc,
            macro_precision,
            macro_recall,
            macro_f1,
            cm,
            per_class_precision,
            per_class_recall,
            per_class_f1,
            per_class_support,
        ) = evaluate(model, val_loader, criterion, device)

        scheduler.step()

        current_lrs = get_current_lrs(optimizer)
        epoch_time = time.time() - epoch_start
        is_best = val_acc > best_val_acc

        # End-of-epoch summary: percentages here (readable at a glance),
        # while the tqdm bar above still shows live decimals per batch.
        val_color = Fore.GREEN if is_best else Fore.YELLOW
        print_rule("-")
        print(
            f"{Fore.CYAN}{Style.BRIGHT}Epoch {epoch:>3}/{cfg.training.num_epochs}"
            f"{Style.RESET_ALL}  ({epoch_time:.1f}s)"
        )
        print(
            f"  train  -> loss: {train_loss:.4f}   acc: {pct(train_acc)}"
        )
        print(
            f"  val    -> loss: {val_loss:.4f}   "
            f"acc: {val_color}{pct(val_acc)}{Style.RESET_ALL}   "
            f"macro_f1: {pct(macro_f1)}"
        )
        print(
            f"  lr     -> backbone: {current_lrs.get('backbone', float('nan')):.2e}"
            f"   classifier: {current_lrs.get('classifier', float('nan')):.2e}"
        )

        # Always save "last" checkpoint
        save_checkpoint(
            last_ckpt_path, model, optimizer, scheduler, epoch,
            best_epoch, val_acc, val_loss, class_names, class_weights,
        )

        # Save "best" checkpoint + early stopping bookkeeping
        if is_best:
            best_val_acc = val_acc
            best_epoch = epoch
            epochs_without_improvement = 0
            save_checkpoint(
                best_ckpt_path, model, optimizer, scheduler, epoch,
                best_epoch, val_acc, val_loss, class_names, class_weights,
            )
            print(
                f"  {Fore.GREEN}{Style.BRIGHT}\u2713 New best model "
                f"(val_acc: {pct(val_acc)}) -> {best_ckpt_path.name}"
                f"{Style.RESET_ALL}"
            )
        else:
            epochs_without_improvement += 1
            print(
                f"  {Fore.YELLOW}No improvement: "
                f"{epochs_without_improvement}/{cfg.training.early_stopping_patience}"
                f"  (best: {pct(best_val_acc)} @ epoch {best_epoch})"
                f"{Style.RESET_ALL}"
            )

        if epochs_without_improvement >= cfg.training.early_stopping_patience:
            print_rule("-")
            print(
                f"{Fore.RED}{Style.BRIGHT}Early stopping triggered at epoch "
                f"{epoch} (no improvement for "
                f"{cfg.training.early_stopping_patience} epochs).{Style.RESET_ALL}"
            )
            break

    total_time = time.time() - training_start
    print_banner("TRAINING COMPLETE")
    print(
        f"  Duration     : {total_time / 60.0:.1f} min\n"
        f"  {Fore.GREEN}Best val acc : {pct(best_val_acc)} (epoch {best_epoch}){Style.RESET_ALL}\n"
        f"  Best model   : {best_ckpt_path}\n"
        f"  Last model   : {last_ckpt_path}\n"
    )

    # Final validation-set report from the last epoch run, using the
    # existing evaluate.py reporting utility (still not the test set).
    print_report(
        per_class_precision, per_class_recall, per_class_f1, cm,
        class_names, support=per_class_support,
    )

    logger.info(
        "Reminder: the test set was NOT used during training. "
        "Run src/training/evaluate.py against %s for final metrics.",
        best_ckpt_path,
    )


if __name__ == "__main__":
    main()