"""Basic sanity test that the model builds and runs a forward pass."""
import torch
from src.config_loader import load_config
from src.model.efficientnet_b0 import build_model


def test_model_forward_pass():
    cfg = load_config()
    model = build_model(cfg)
    model.eval()
    dummy = torch.randn(1, 3, cfg.model.input_size, cfg.model.input_size)
    with torch.no_grad():
        out = model(dummy)
    assert out.shape == (1, cfg.model.num_classes)
