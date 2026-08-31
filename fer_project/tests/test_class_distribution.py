from pathlib import Path
from collections import Counter

from src.config_loader import load_config


def count_classes(folder):
    folder = Path(folder)

    counts = Counter()

    for class_dir in folder.iterdir():

        if not class_dir.is_dir():
            continue

        count = 0

        for file in class_dir.rglob("*"):
            if file.suffix.lower() in {
                ".jpg",
                ".jpeg",
                ".png",
                ".bmp",
            }:
                count += 1

        counts[class_dir.name] = count

    return counts


def main():

    cfg = load_config()

    train_counts = count_classes(
        cfg.paths.train_dir
    )

    print("=" * 50)
    print("FER2013 TRAINING CLASS DISTRIBUTION")
    print("=" * 50)

    total = sum(train_counts.values())

    for class_name, count in sorted(
        train_counts.items()
    ):
        percentage = count / total * 100

        print(
            f"{class_name:10s}: "
            f"{count:6d} "
            f"({percentage:5.2f}%)"
        )

    print("-" * 50)
    print(f"Total images: {total}")


if __name__ == "__main__":
    main()