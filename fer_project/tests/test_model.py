import torch

from src.config_loader import load_config
from src.model.efficientnet_b0 import build_model


def main():
    cfg = load_config()

    model = build_model(cfg)

    print("Model created successfully.")
    print("Classifier:", model.classifier)

    x = torch.randn(2, 3, 224, 224)

    with torch.no_grad():
        output = model(x)

    print("Input shape :", x.shape)
    print("Output shape:", output.shape)


if __name__ == "__main__":
    main()
