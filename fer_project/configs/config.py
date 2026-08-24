from pathlib import Path

# Raw datasets
FER_PATH = Path(
    r"C:\Users\user\Desktop\LUCY\AI Unleashed 2026\datasets\FER2013"
)

CK_PATH = Path(
    r"C:\Users\user\Desktop\LUCY\AI Unleashed 2026\datasets\CK+"
)

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Output folder
PROCESSED_DATASET = PROJECT_ROOT / "dataset" / "processed"