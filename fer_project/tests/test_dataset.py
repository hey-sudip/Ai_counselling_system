from src.config_loader import load_config
from src.data.dataset import build_dataloaders


def main():
    cfg = load_config()

    train_loader, val_loader, test_loader, classes = build_dataloaders(cfg)

    print("Classes:", classes)
    print("Train batches:", len(train_loader))
    print("Validation batches:", len(val_loader))
    print("Test batches:", len(test_loader))

    images, labels = next(iter(train_loader))

    print("Image shape:", images.shape)
    print("Label shape:", labels.shape)


if __name__ == "__main__":
    main()