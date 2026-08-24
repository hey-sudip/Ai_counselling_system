import os
from pathlib import Path

from configs.config import FER_PATH, CK_PATH


class DataIngestion:
    def __init__(self):
        self.fer_path = Path(FER_PATH)
        self.ck_path = Path(CK_PATH)

    def _count_images(self, folder):
        """Count all image files in a folder recursively."""
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}

        count = 0
        for _, _, files in os.walk(folder):
            count += sum(
                1
                for file in files
                if Path(file).suffix.lower() in image_extensions
            )

        return count

    def ingest(self):
        print("=" * 60)
        print("            DATA INGESTION")
        print("=" * 60)

        # ---------------- FER2013 ---------------- #
        if not self.fer_path.exists():
            raise FileNotFoundError(
                f"FER2013 dataset not found!\nPath: {self.fer_path}"
            )

        print("✓ FER2013 dataset found")

        train_path = self.fer_path / "train"
        test_path = self.fer_path / "test"

        train_images = self._count_images(train_path)
        test_images = self._count_images(test_path)

        print(f"Train Images : {train_images}")
        print(f"Test Images  : {test_images}")

        # ---------------- CK+ ---------------- #
        if not self.ck_path.exists():
            raise FileNotFoundError(
                f"CK+ dataset not found!\nPath: {self.ck_path}"
            )

        print("\n✓ CK+ dataset found")

        ck_images = self._count_images(self.ck_path)

        print(f"CK+ Images   : {ck_images}")

        print("\n✓ Data ingestion completed successfully!")

        return {
            "fer_train": train_path,
            "fer_test": test_path,
            "ck_dataset": self.ck_path,
        }