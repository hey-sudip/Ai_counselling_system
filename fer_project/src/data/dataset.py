from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets

from src.data.transforms import (
    get_train_transforms,
    get_eval_transforms,
)


def build_dataloaders(cfg):
    """
    Build train, validation and test dataloaders.
    """

    train_transform = get_train_transforms(cfg)
    eval_transform = get_eval_transforms(cfg)

    train_dir = Path(cfg.paths.train_dir)
    val_dir = Path(cfg.paths.val_dir)
    test_dir = Path(cfg.paths.test_dir)

    train_dataset = datasets.ImageFolder(
        train_dir,
        transform=train_transform,
    )

    val_dataset = datasets.ImageFolder(
        val_dir,
        transform=eval_transform,
    )

    test_dataset = datasets.ImageFolder(
        test_dir,
        transform=eval_transform,
    )

    classes = train_dataset.classes

    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.num_workers,
        pin_memory=pin_memory,
        persistent_workers=cfg.training.num_workers > 0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers,
        pin_memory=pin_memory,
        persistent_workers=cfg.training.num_workers > 0,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers,
        pin_memory=pin_memory,
        persistent_workers=cfg.training.num_workers > 0,
    )

    return (
        train_loader,
        val_loader,
        test_loader,
        classes,
    )