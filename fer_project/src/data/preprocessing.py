import random
import shutil
from collections import defaultdict
from pathlib import Path

from PIL import Image

from configs.config import FER_PATH, PROCESSED_DATASET


class DataPreprocessing:
    """
    Leakage-safe FER2013 preprocessing.

    Pipeline:
        1. Read raw FER2013 train/test data.
        2. Detect exact visual duplicates between train and test.
        3. Remove train copies of test images.
        4. Remove duplicate images within the training dataset.
        5. Perform deterministic 80/20 stratified split.
        6. Copy test data separately.
        7. Verify all processed splits contain no exact duplicates.
    """

    IMAGE_EXTENSIONS = {
        ".jpg", ".jpeg", ".png", ".bmp", ".webp"
    }

    SEED = 42
    TRAIN_RATIO = 0.80

    def __init__(self):
        self.raw_dataset = Path(FER_PATH)
        self.output_dataset = Path(PROCESSED_DATASET)
        self.rng = random.Random(self.SEED)

    # =========================================================
    # IMAGE UTILITIES
    # =========================================================

    def get_images(self, directory):
        """Return all supported image files recursively."""

        directory = Path(directory)

        if not directory.exists():
            raise FileNotFoundError(
                f"Directory not found:\n{directory}"
            )

        return sorted(
            path
            for path in directory.rglob("*")
            if (
                path.is_file()
                and path.suffix.lower() in self.IMAGE_EXTENSIONS
            )
        )

    def get_signature(self, image_path):
        """
        Create an exact visual signature from decoded RGB pixels.

        Filename differences do not matter.
        """

        try:
            with Image.open(image_path) as image:
                image = image.convert("RGB")
                return image.size, image.tobytes()

        except Exception as error:
            print(f"WARNING: Could not read {image_path}")
            print(f"         {error}")
            return None

    def build_signature_map(self, images):
        """Map image signatures to image paths."""

        signatures = defaultdict(list)

        for image in images:
            signature = self.get_signature(image)

            if signature is not None:
                signatures[signature].append(image)

        return signatures

    # =========================================================
    # DIRECTORY SETUP
    # =========================================================

    def create_fresh_output(self):
        """Delete old processed data and create a clean dataset."""

        if self.output_dataset.exists():
            print("\nRemoving old processed dataset...")
            shutil.rmtree(self.output_dataset)

        for split in ("train", "validation", "test"):
            (self.output_dataset / split).mkdir(
                parents=True,
                exist_ok=True
            )

        print("✓ Fresh processed dataset created")

    # =========================================================
    # TRAIN ↔ TEST LEAKAGE
    # =========================================================

    def find_train_test_duplicates(self):
        """Find raw training images that also exist in test."""

        train_root = self.raw_dataset / "train"
        test_root = self.raw_dataset / "test"

        train_images = self.get_images(train_root)
        test_images = self.get_images(test_root)

        print("\n" + "=" * 60)
        print("CHECKING TRAIN ↔ TEST DUPLICATES")
        print("=" * 60)

        print(f"Raw training images : {len(train_images)}")
        print(f"Raw test images     : {len(test_images)}")

        test_signatures = set(
            self.build_signature_map(test_images)
        )

        duplicates = set()

        for image in train_images:
            signature = self.get_signature(image)

            if signature is not None and signature in test_signatures:
                duplicates.add(image)

        print(
            f"\nTrain images duplicated in test: "
            f"{len(duplicates)}"
        )

        if duplicates:
            print(
                "✓ These training copies will be excluded."
            )
        else:
            print("✓ No train ↔ test duplicates found.")

        return duplicates

    # =========================================================
    # CLEAN TRAINING DATA
    # =========================================================

    def collect_clean_training_data(self, excluded):
        """
        Remove train/test duplicates and duplicate copies
        from the complete training dataset.

        Duplicate detection is GLOBAL across all classes.
        """

        train_root = self.raw_dataset / "train"

        print("\n" + "=" * 60)
        print("CLEANING TRAINING DATA")
        print("=" * 60)

        all_images = []

        for class_dir in sorted(train_root.iterdir()):

            if not class_dir.is_dir():
                continue

            for image in self.get_images(class_dir):
                all_images.append(
                    (class_dir.name, image)
                )

        original_count = len(all_images)

        # Remove copies of test images
        all_images = [
            item
            for item in all_images
            if item[1] not in excluded
        ]

        removed_test = original_count - len(all_images)

        # Group exact duplicate images globally
        groups = defaultdict(list)

        for class_name, image in all_images:
            signature = self.get_signature(image)

            if signature is not None:
                groups[signature].append(
                    (class_name, image)
                )

        clean_data = defaultdict(list)
        duplicate_count = 0
        conflict_count = 0

        for group in groups.values():

            # One exact image = one representative
            if len(group) > 1:
                duplicate_count += len(group) - 1

                labels = {
                    class_name
                    for class_name, _ in group
                }

                if len(labels) > 1:
                    conflict_count += 1

                    print(
                        "\nWARNING: Same image has "
                        "different labels:"
                    )

                    for class_name, image in group:
                        print(
                            f"  {class_name}: {image}"
                        )

            # Deterministic representative
            class_name, image = sorted(
                group,
                key=lambda item: str(item[1])
            )[0]

            clean_data[class_name].append(image)

        print("\nCleaning summary:")
        print(f"Original training images : {original_count}")
        print(f"Train/test duplicates    : {removed_test}")
        print(f"Duplicate copies removed : {duplicate_count}")
        print(
            f"Conflicting label groups : "
            f"{conflict_count}"
        )
        print(
            f"Final unique images      : "
            f"{sum(map(len, clean_data.values()))}"
        )

        print("\nClean class distribution:")

        for class_name in sorted(clean_data):
            print(
                f"{class_name:<10}: "
                f"{len(clean_data[class_name]):5}"
            )

        return clean_data

    # =========================================================
    # TRAIN / VALIDATION SPLIT
    # =========================================================

    def split_training_data(self, clean_data):
        """Perform deterministic 80/20 stratified split."""

        print("\n" + "=" * 60)
        print("CREATING TRAIN / VALIDATION SPLIT")
        print("=" * 60)

        train_total = 0
        validation_total = 0

        for class_name in sorted(clean_data):

            images = list(clean_data[class_name])

            self.rng.shuffle(images)

            split_index = int(
                len(images) * self.TRAIN_RATIO
            )

            train_images = images[:split_index]
            validation_images = images[split_index:]

            train_dir = (
                self.output_dataset
                / "train"
                / class_name
            )

            validation_dir = (
                self.output_dataset
                / "validation"
                / class_name
            )

            train_dir.mkdir(parents=True, exist_ok=True)
            validation_dir.mkdir(
                parents=True,
                exist_ok=True
            )

            for image in train_images:
                shutil.copy2(
                    image,
                    train_dir / image.name
                )

            for image in validation_images:
                shutil.copy2(
                    image,
                    validation_dir / image.name
                )

            train_total += len(train_images)
            validation_total += len(validation_images)

            print(
                f"{class_name:<10}"
                f" Train: {len(train_images):5}"
                f" Validation: {len(validation_images):5}"
            )

        print("\n" + "-" * 60)
        print(f"Processed training images   : {train_total}")
        print(f"Processed validation images : {validation_total}")
        print(f"Total                       : {train_total + validation_total}")

    # =========================================================
    # TEST DATA
    # =========================================================

    def copy_test_data(self):
        """Copy raw test data without modification."""

        test_root = self.raw_dataset / "test"

        print("\n" + "=" * 60)
        print("COPYING TEST DATASET")
        print("=" * 60)

        total = 0

        for class_dir in sorted(test_root.iterdir()):

            if not class_dir.is_dir():
                continue

            destination = (
                self.output_dataset
                / "test"
                / class_dir.name
            )

            destination.mkdir(
                parents=True,
                exist_ok=True
            )

            images = self.get_images(class_dir)

            for image in images:
                shutil.copy2(
                    image,
                    destination / image.name
                )

            total += len(images)

            print(
                f"{class_dir.name:<10}"
                f" Test: {len(images):5}"
            )

        print(f"\nTotal test images: {total}")

    # =========================================================
    # FINAL LEAKAGE VERIFICATION
    # =========================================================

    def verify_no_leakage(self):
        """
        Verify exact visual duplicates do not exist
        between any processed splits.
        """

        print("\n" + "=" * 60)
        print("FINAL LEAKAGE VERIFICATION")
        print("=" * 60)

        splits = {
            "train": self.get_images(
                self.output_dataset / "train"
            ),
            "validation": self.get_images(
                self.output_dataset / "validation"
            ),
            "test": self.get_images(
                self.output_dataset / "test"
            ),
        }

        for name, images in splits.items():
            print(f"{name.capitalize():<12}: {len(images)}")

        signatures = {
            name: set(
                self.build_signature_map(images)
            )
            for name, images in splits.items()
        }

        train_val = (
            signatures["train"]
            & signatures["validation"]
        )

        train_test = (
            signatures["train"]
            & signatures["test"]
        )

        validation_test = (
            signatures["validation"]
            & signatures["test"]
        )

        print("\nDuplicate visual images:")
        print(
            f"Train ↔ Validation : {len(train_val)}"
        )
        print(
            f"Train ↔ Test       : {len(train_test)}"
        )
        print(
            f"Validation ↔ Test  : {len(validation_test)}"
        )

        leakage_free = not any(
            (
                train_val,
                train_test,
                validation_test,
            )
        )

        print("\n" + "=" * 60)

        if leakage_free:
            print("✓ NO DATA LEAKAGE DETECTED")
            print("=" * 60)
            return True

        print("✗ DATA LEAKAGE DETECTED")
        print("=" * 60)

        return False

    # =========================================================
    # MAIN PIPELINE
    # =========================================================

    def preprocess(self):

        print("\n" + "=" * 60)
        print("FER2013 LEAKAGE-SAFE PREPROCESSING")
        print("=" * 60)

        print(f"\nRaw dataset:")
        print(self.raw_dataset)

        print(f"\nProcessed dataset:")
        print(self.output_dataset)

        # Safety checks
        if not self.raw_dataset.exists():
            raise FileNotFoundError(
                f"Raw dataset not found:\n"
                f"{self.raw_dataset}"
            )

        train_root = self.raw_dataset / "train"
        test_root = self.raw_dataset / "test"

        if not train_root.exists():
            raise FileNotFoundError(
                f"Training directory not found:\n"
                f"{train_root}"
            )

        if not test_root.exists():
            raise FileNotFoundError(
                f"Test directory not found:\n"
                f"{test_root}"
            )

        # Start completely fresh
        self.create_fresh_output()

        # Remove train copies of test images
        excluded = self.find_train_test_duplicates()

        # Globally remove duplicate training images
        clean_data = self.collect_clean_training_data(
            excluded
        )

        # Stratified 80/20 split
        self.split_training_data(clean_data)

        # Copy test separately
        self.copy_test_data()

        # Final safety verification
        if not self.verify_no_leakage():
            raise RuntimeError(
                "\nLeakage verification failed. "
                "DO NOT train the model."
            )

        print("\n" + "=" * 60)
        print("PREPROCESSING COMPLETE")
        print("=" * 60)

        print("\n✓ Raw dataset was not modified.")
        print("✓ Train/test duplicates were removed.")
        print("✓ Training duplicates were removed globally.")
        print("✓ Train/validation split was performed afterward.")
        print("✓ Test data remained separate.")
        print("✓ Final leakage verification passed.")
        print(
            f"\nProcessed dataset ready at:\n"
            f"{self.output_dataset}"
        )


if __name__ == "__main__":
    processor = DataPreprocessing()
    processor.preprocess()