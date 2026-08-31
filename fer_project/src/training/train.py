"""EfficientNet-B0 fine-tuning training pipeline."""

import os
import random

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from src.config_loader import load_config
from src.data.dataset import build_dataloaders
from src.model.efficientnet_b0 import build_model
from src.training.evaluate import evaluate


# =========================================================
# REPRODUCIBILITY
# =========================================================

def set_seed(seed=42):
    """Set random seeds for reproducible training."""

    random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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
# CLASS WEIGHTS
# =========================================================

def calculate_class_weights(
    train_loader,
    num_classes,
    device
):
    """
    Calculate moderated class weights.

    Square-root inverse-frequency weighting is used
    to compensate for class imbalance without allowing
    the smallest classes to dominate the loss.
    """

    dataset = train_loader.dataset

    targets = torch.tensor(
        dataset.targets,
        dtype=torch.long
    )

    class_counts = torch.bincount(
        targets,
        minlength=num_classes
    ).float()

    if torch.any(class_counts == 0):
        raise RuntimeError(
            "At least one class has zero training samples."
        )

    total_samples = class_counts.sum()

    inverse_weights = (
        total_samples
        / (num_classes * class_counts)
    )

    class_weights = torch.sqrt(
        inverse_weights
    )

    # Normalize weights around 1.0.
    class_weights = (
        class_weights
        / class_weights.mean()
    )

    class_weights = class_weights.to(device)

    print("\nClass counts:")

    for index, count in enumerate(class_counts):
        print(
            f"  Class {index}: "
            f"{int(count.item())}"
        )

    print("\nModerated class weights:")

    for index, weight in enumerate(class_weights):
        print(
            f"  Class {index}: "
            f"{weight.item():.4f}"
        )

    return class_weights


# =========================================================
# OPTIMIZER
# =========================================================

def create_optimizer(model, cfg):
    """
    Create AdamW with separate learning rates.

    Backbone:
        cfg.training.backbone_learning_rate

    Classification head:
        cfg.training.learning_rate
    """

    backbone_parameters = []
    classifier_parameters = []

    for name, parameter in model.named_parameters():

        if not parameter.requires_grad:
            continue

        if name.startswith("classifier."):
            classifier_parameters.append(parameter)
        else:
            backbone_parameters.append(parameter)

    if not backbone_parameters:
        raise RuntimeError(
            "No trainable backbone parameters found."
        )

    if not classifier_parameters:
        raise RuntimeError(
            "No trainable classifier parameters found."
        )

    optimizer = AdamW(
        [
            {
                "params": backbone_parameters,
                "lr": cfg.training.backbone_learning_rate,
            },
            {
                "params": classifier_parameters,
                "lr": cfg.training.learning_rate,
            },
        ],
        weight_decay=cfg.training.weight_decay,
    )

    print("\nOptimizer learning rates:")

    print(
        f"  Backbone    : "
        f"{cfg.training.backbone_learning_rate:.8f}"
    )

    print(
        f"  Classifier  : "
        f"{cfg.training.learning_rate:.8f}"
    )

    print(
        f"  Weight decay: "
        f"{cfg.training.weight_decay:.8f}"
    )

    return optimizer


# =========================================================
# ONE TRAINING EPOCH
# =========================================================

def train_one_epoch(
    model,
    train_loader,
    criterion,
    optimizer,
    device,
    epoch,
    total_epochs
):
    """Train the model for one epoch."""

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    loop = tqdm(
        train_loader,
        desc=f"Epoch {epoch}/{total_epochs}"
    )

    for images, labels in loop:

        images = images.to(
            device,
            non_blocking=True
        )

        labels = labels.to(
            device,
            non_blocking=True
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        # Prevent unusually large gradients.
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        optimizer.step()

        batch_size = images.size(0)

        running_loss += (
            loss.item() * batch_size
        )

        predictions = outputs.argmax(
            dim=1
        )

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

        current_acc = correct / total

        loop.set_postfix(
            loss=f"{loss.item():.4f}",
            acc=f"{current_acc:.4f}"
        )

    train_loss = running_loss / total
    train_acc = correct / total

    return train_loss, train_acc


# =========================================================
# CHECKPOINT
# =========================================================

def save_checkpoint(
    path,
    model,
    classes,
    val_acc,
    val_loss,
    epoch,
    best_epoch,
    class_weights,
    optimizer,
    scheduler
):
    """Save a complete training checkpoint."""

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "classes": classes,

            "val_acc": val_acc,
            "val_loss": val_loss,

            "epoch": epoch,
            "best_epoch": best_epoch,

            "stage": "single_stage_finetuning",

            "class_weights": class_weights.cpu(),

            "optimizer_state_dict":
                optimizer.state_dict(),

            "scheduler_state_dict":
                scheduler.state_dict(),
        },
        path
    )


# =========================================================
# MAIN TRAINING FUNCTION
# =========================================================

def train():
    """Main EfficientNet-B0 fine-tuning function."""

    # -----------------------------------------------------
    # CONFIG
    # -----------------------------------------------------

    cfg = load_config()

    set_seed(42)

    device = get_device(cfg)

    print("=" * 70)
    print("EFFICIENTNET-B0 FINE-TUNING")
    print("=" * 70)

    print(
        f"\nUsing device: {device}"
    )

    if device.type == "cuda":
        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

    # -----------------------------------------------------
    # DATA
    # -----------------------------------------------------

    print("\n" + "=" * 70)
    print("LOADING DATASET")
    print("=" * 70)

    (
        train_loader,
        val_loader,
        _,
        classes
    ) = build_dataloaders(cfg)

    print(
        f"\nClasses: {classes}"
    )

    num_classes = len(classes)

    if num_classes != cfg.model.num_classes:
        raise RuntimeError(
            f"Class mismatch: "
            f"dataset has {num_classes} classes, "
            f"config expects {cfg.model.num_classes}."
        )

    print(
        f"Training batches   : "
        f"{len(train_loader)}"
    )

    print(
        f"Validation batches : "
        f"{len(val_loader)}"
    )

    print(
        "\n✓ Test loader was created "
        "but will NOT be used during training."
    )

    # -----------------------------------------------------
    # MODEL
    # -----------------------------------------------------

    print("\n" + "=" * 70)
    print("BUILDING MODEL")
    print("=" * 70)

    model = build_model(cfg).to(device)

    print(
        f"\nModel: {cfg.model.name}"
    )

    print(
        f"Pretrained: "
        f"{cfg.model.pretrained}"
    )

    print(
        f"Input size: "
        f"{cfg.model.input_size}x"
        f"{cfg.model.input_size}"
    )

    print(
        f"Number of classes: "
        f"{num_classes}"
    )

    # -----------------------------------------------------
    # CLASS WEIGHTS
    # -----------------------------------------------------

    class_weights = calculate_class_weights(
        train_loader,
        num_classes,
        device
    )

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    # -----------------------------------------------------
    # OPTIMIZER
    # -----------------------------------------------------

    optimizer = create_optimizer(
        model,
        cfg
    )

    # -----------------------------------------------------
    # SCHEDULER
    # -----------------------------------------------------

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=cfg.training.num_epochs,
        eta_min=1e-6
    )

    # -----------------------------------------------------
    # CHECKPOINT DIRECTORY
    # -----------------------------------------------------

    os.makedirs(
        cfg.paths.checkpoint_dir,
        exist_ok=True
    )

    # IMPORTANT:
    # This training run starts fresh.
    #
    # The previous 63.18% model should already have
    # been backed up separately.
    #
    # Example:
    # emotion_model_finetuned_63_18.pth

    best_path = os.path.join(
        cfg.paths.checkpoint_dir,
        "emotion_model_finetuned.pth"
    )

    last_path = os.path.join(
        cfg.paths.checkpoint_dir,
        "emotion_model_finetuned_last.pth"
    )

    # -----------------------------------------------------
    # TRAINING STATE
    # -----------------------------------------------------

    best_val_acc = 0.0
    best_epoch = 0

    epochs_without_improvement = 0

    patience = (
        cfg.training.early_stopping_patience
    )

    # -----------------------------------------------------
    # TRAINING
    # -----------------------------------------------------

    print("\n" + "=" * 70)

    print(
        f"TRAINING FOR "
        f"{cfg.training.num_epochs} EPOCHS"
    )

    print("=" * 70)

    print(
        f"\nEarly stopping patience: "
        f"{patience}"
    )

    print(
        "\nTraining mode: FRESH RUN"
    )

    print(
        "Previous checkpoint will NOT "
        "be loaded."
    )

    for epoch in range(
        1,
        cfg.training.num_epochs + 1
    ):

        # -------------------------------------------------
        # TRAIN
        # -------------------------------------------------

        train_loss, train_acc = train_one_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            total_epochs=cfg.training.num_epochs
        )

        # -------------------------------------------------
        # VALIDATION
        # -------------------------------------------------

        val_loss, val_acc, _, _, _, _, _, _, _, _ = evaluate(
            model,
            val_loader,
            criterion,
            device
        )

        current_backbone_lr = (
            optimizer.param_groups[0]["lr"]
        )

        current_classifier_lr = (
            optimizer.param_groups[1]["lr"]
        )

        print(
            f"\nEpoch {epoch}: "
            f"train_loss={train_loss:.4f} "
            f"train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} "
            f"val_acc={val_acc:.4f}"
        )

        print(
            f"Learning rates: "
            f"backbone={current_backbone_lr:.8f} "
            f"classifier={current_classifier_lr:.8f}"
        )

        # -------------------------------------------------
        # BEST MODEL
        # -------------------------------------------------

        if val_acc > best_val_acc:

            best_val_acc = val_acc
            best_epoch = epoch

            epochs_without_improvement = 0

            save_checkpoint(
                path=best_path,
                model=model,
                classes=classes,
                val_acc=val_acc,
                val_loss=val_loss,
                epoch=epoch,
                best_epoch=best_epoch,
                class_weights=class_weights,
                optimizer=optimizer,
                scheduler=scheduler
            )

            print(
                "\n✓ New best model saved!"
            )

            print(
                f"  Validation accuracy: "
                f"{val_acc * 100:.2f}%"
            )

        else:

            epochs_without_improvement += 1

            print(
                f"\nNo validation improvement "
                f"for "
                f"{epochs_without_improvement} "
                f"epoch(s)."
            )

        # -------------------------------------------------
        # SCHEDULER
        # -------------------------------------------------

        scheduler.step()

        # -------------------------------------------------
        # EARLY STOPPING
        # -------------------------------------------------

        if (
            epochs_without_improvement
            >= patience
        ):

            print(
                "\n" + "=" * 70
            )

            print(
                "EARLY STOPPING"
            )

            print(
                "=" * 70
            )

            print(
                f"\nNo validation improvement "
                f"for {patience} epochs."
            )

            break

    # -----------------------------------------------------
    # SAVE LAST MODEL
    # -----------------------------------------------------

    save_checkpoint(
        path=last_path,
        model=model,
        classes=classes,
        val_acc=val_acc,
        val_loss=val_loss,
        epoch=epoch,
        best_epoch=best_epoch,
        class_weights=class_weights,
        optimizer=optimizer,
        scheduler=scheduler
    )

    # -----------------------------------------------------
    # FINAL SUMMARY
    # -----------------------------------------------------

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    print(
        f"\nBest validation accuracy: "
        f"{best_val_acc:.4f}"
    )

    print(
        f"Best validation accuracy: "
        f"{best_val_acc * 100:.2f}%"
    )

    print(
        f"Best epoch: "
        f"{best_epoch}"
    )

    print(
        f"\nBest model saved to:"
    )

    print(best_path)

    print(
        f"\nLast model saved to:"
    )

    print(last_path)

    print(
        "\n✓ Test set was not used "
        "for model selection."
    )

    print(
        "✓ Best checkpoint is selected "
        "using validation accuracy."
    )

    print(
        "\n✓ Training completed using "
        "EfficientNet-B0."
    )


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":
    train()