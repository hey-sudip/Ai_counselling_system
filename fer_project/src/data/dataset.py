"""Dataset / DataLoader construction (Phase 2-3)."""
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from src.data.transforms import get_train_transforms, get_eval_transforms


def build_dataloaders(cfg):
    train_ds = ImageFolder(cfg.paths.train_dir, transform=get_train_transforms(cfg))
    val_ds = ImageFolder(cfg.paths.val_dir, transform=get_eval_transforms(cfg))
    test_ds = ImageFolder(cfg.paths.test_dir, transform=get_eval_transforms(cfg))

    train_loader = DataLoader(
        train_ds, batch_size=cfg.training.batch_size, shuffle=True,
        num_workers=cfg.training.num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.training.batch_size, shuffle=False,
        num_workers=cfg.training.num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg.training.batch_size, shuffle=False,
        num_workers=cfg.training.num_workers, pin_memory=True,
    )
    return train_loader, val_loader, test_loader, train_ds.classes
