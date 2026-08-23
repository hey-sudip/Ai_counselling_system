"""Basic sanity tests for dataset loading."""
import os


def test_processed_dirs_exist():
    for split in ["train", "validation", "test"]:
        path = os.path.join("dataset", "processed", split)
        assert os.path.isdir(path), f"Missing directory: {path}"


def test_class_subfolders_exist():
    classes = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
    for split in ["train", "validation", "test"]:
        for c in classes:
            path = os.path.join("dataset", "processed", split, c)
            assert os.path.isdir(path), f"Missing class folder: {path}"
