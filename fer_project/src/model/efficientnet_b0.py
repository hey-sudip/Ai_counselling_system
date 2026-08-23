"""EfficientNet-B0 with a replaced 7-class classification head (Phase 4)."""
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


def build_model(cfg):
    weights = EfficientNet_B0_Weights.DEFAULT if cfg.model.pretrained else None
    model = efficientnet_b0(weights=weights)

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, cfg.model.num_classes)

    return model
