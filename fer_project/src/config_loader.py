"""Loads configs/config.yaml into a dot-accessible object."""
import yaml
from pathlib import Path


class DotDict(dict):
    def __getattr__(self, key):
        try:
            value = self[key]
        except KeyError as e:
            raise AttributeError(key) from e
        if isinstance(value, dict):
            value = DotDict(value)
        return value

    def __setattr__(self, key, value):
        self[key] = value


def load_config(config_path: str = None) -> DotDict:
    if config_path is None:
        root = Path(__file__).resolve().parent.parent
        config_path = root / "configs" / "config.yaml"
    with open(config_path, "r") as f:
        raw = yaml.safe_load(f)
    return DotDict(raw)
