from pathlib import Path

from configs.config import FER_PATH, CK_PATH


EXPECTED_CLASSES = {
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise",
}


class DataValidation:

    def __init__(self):
        self.fer_path = Path(FER_PATH)
        self.ck_path = Path(CK_PATH)

    def _count_images(self, folder):
        exts = {".jpg", ".jpeg", ".png", ".bmp"}
        return sum(
            1
            for file in folder.rglob("*")
            if file.suffix.lower() in exts
        )

    def validate_fer(self):

        print("\n========== VALIDATING FER2013 ==========")

        for split in ["train", "test"]:

            split_path = self.fer_path / split

            if not split_path.exists():
                raise FileNotFoundError(f"{split_path} not found")

            folders = {
                folder.name
                for folder in split_path.iterdir()
                if folder.is_dir()
            }

            missing = EXPECTED_CLASSES - folders
            extra = folders - EXPECTED_CLASSES

            if missing:
                raise Exception(f"Missing classes in {split}: {missing}")

            if extra:
                print(f"Extra classes in {split}: {extra}")

            print(f"\n{split.upper()}")

            for cls in sorted(folders):

                count = self._count_images(split_path / cls)

                print(f"{cls:<10} : {count}")

    def validate_ck(self):

        print("\n========== VALIDATING CK+ ==========")

        folders = [
            folder
            for folder in self.ck_path.iterdir()
            if folder.is_dir()
        ]

        for folder in sorted(folders):

            count = self._count_images(folder)

            print(f"{folder.name:<10} : {count}")

    def validate(self):

        self.validate_fer()
        self.validate_ck()

        print("\n✓ Validation Successful")