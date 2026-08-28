from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def build_transforms(cfg):
    input_size = cfg.model.input_size

    train_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.RandomHorizontalFlip(
            p=cfg.augmentation.horizontal_flip_prob
        ),
        transforms.RandomRotation(
            cfg.augmentation.rotation_degrees
        ),
        transforms.ColorJitter(
            brightness=cfg.augmentation.brightness,
            contrast=cfg.augmentation.contrast
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    return train_transform, eval_transform


def build_dataloaders(cfg):
    train_transform, eval_transform = build_transforms(cfg)

    train_dir = Path(cfg.paths.train_dir)
    val_dir = Path(cfg.paths.val_dir)
    test_dir = Path(cfg.paths.test_dir)

    train_dataset = datasets.ImageFolder(
        train_dir,
        transform=train_transform
    )

    val_dataset = datasets.ImageFolder(
        val_dir,
        transform=eval_transform
    )

    test_dataset = datasets.ImageFolder(
        test_dir,
        transform=eval_transform
    )

    classes = train_dataset.classes

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.num_workers,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers,
        pin_memory=torch.cuda.is_available()
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers,
        pin_memory=torch.cuda.is_available()
    )

    return train_loader, val_loader, test_loader, classes