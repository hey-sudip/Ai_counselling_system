"""Basic sanity test for temporal smoothing logic."""
import numpy as np
from src.inference.temporal_smoothing import TemporalSmoother


def test_smoother_returns_valid_label():
    classes = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
    smoother = TemporalSmoother(classes, window_size=3)

    probs = np.array([0.1, 0.05, 0.05, 0.6, 0.1, 0.05, 0.05])
    label, avg = smoother.update(probs)

    assert label in classes
    assert avg.shape == (7,)
