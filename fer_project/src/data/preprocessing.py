import random
import shutil
from pathlib import Path

from configs.config import FER_PATH, PROCESSED_DATASET


class DataPreprocessing:

    def __init__(self):
        self.raw_dataset = Path(FER_PATH)
        self.output_dataset = Path(PROCESSED_DATASET)

        random.seed(42)

    def copy_test_dataset(self):
        test_root = self.raw_dataset / "test"
        print("\nCopying test dataset...")

        for class_folder in sorted(test_root.iterdir()):

            if not class_folder.is_dir():
                continue

            destination = self.output_dataset / "test" / class_folder.name

            destination.mkdir(parents=True, exist_ok=True)

            images = list(class_folder.glob("*"))

            for image in images:
                shutil.copy2(image, destination / image.name)

            print(f"{class_folder.name:<10} Test: {len(images):5}")

        print("\n✓ Test dataset copied successfully")

    def create_directory_structure(self):

        (self.output_dataset / "train").mkdir(parents=True, exist_ok=True)
        (self.output_dataset / "validation").mkdir(parents=True, exist_ok=True)
        (self.output_dataset / "test").mkdir(parents=True, exist_ok=True)

        print("✓ Directory structure created")

    def split_training_dataset(self):

        train_root = self.raw_dataset / "train"

        print("\nSplitting training dataset...")

        for class_folder in sorted(train_root.iterdir()):

            if not class_folder.is_dir():
                continue

            images = list(class_folder.glob("*"))

            random.shuffle(images)

            split_index = int(len(images) * 0.8)

            train_images = images[:split_index]
            val_images = images[split_index:]

            train_destination = self.output_dataset / "train" / class_folder.name
            val_destination = self.output_dataset / "validation" / class_folder.name

            train_destination.mkdir(parents=True, exist_ok=True)
            val_destination.mkdir(parents=True, exist_ok=True)

            for image in train_images:
                shutil.copy2(image, train_destination / image.name)

            for image in val_images:
                shutil.copy2(image, val_destination / image.name)

            print(
                f"{class_folder.name:<10}"
                f" Train: {len(train_images):5}"
                f" Validation: {len(val_images):5}"
            )
    def preprocess(self):

        self.create_directory_structure()

        self.split_training_dataset()

        self.copy_test_dataset()