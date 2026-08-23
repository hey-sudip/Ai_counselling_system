"""Resize, normalize, and augmentation transforms (Phase 3)."""
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transforms(cfg):
    size = cfg.model.input_size
    aug = cfg.augmentation
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.RandomHorizontalFlip(p=aug.horizontal_flip_prob),
        transforms.RandomRotation(aug.rotation_degrees),
        transforms.ColorJitter(brightness=aug.brightness, contrast=aug.contrast),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def get_eval_transforms(cfg):
    size = cfg.model.input_size
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
