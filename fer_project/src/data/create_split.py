"""
Creates the processed FER2013 dataset.

Official protocol:
- Training -> 90% Train + 10% Validation (stratified)
- PublicTest + PrivateTest -> Test
"""

from pathlib import Path
import shutil

import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from configs.config import (
    FER_PATH,
    PROCESSED_DATASET,
    RANDOM_SEED,
    CLASS_NAMES,
)

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------

CSV_PATH = Path(
    r"C:\Users\user\Desktop\LUCY\AI Unleashed 2026\datasets\fer2013_clean.csv"
)

# ------------------------------------------------------------------
# Load CSV
# ------------------------------------------------------------------

df = pd.read_csv(CSV_PATH)

emotion_map = {
    0: "angry",
    1: "disgust",
    2: "fear",
    3: "happy",
    4: "sad",
    5: "surprise",
    6: "neutral",
}

df["emotion_name"] = df["emotion"].map(emotion_map)

# ------------------------------------------------------------------
# Official FER split
# ------------------------------------------------------------------

training_df = df[df["Usage"] == "Training"].copy()

test_df = df[df["Usage"] != "Training"].copy()

train_df, val_df = train_test_split(
    training_df,
    test_size=0.10,
    stratify=training_df["emotion"],
    random_state=RANDOM_SEED,
)

print(f"Train      : {len(train_df)}")
print(f"Validation : {len(val_df)}")
print(f"Test       : {len(test_df)}")

# ------------------------------------------------------------------
# Recreate processed folder
# ------------------------------------------------------------------

if PROCESSED_DATASET.exists():
    shutil.rmtree(PROCESSED_DATASET)

for split in ["train", "validation", "test"]:
    for emotion in CLASS_NAMES:
        (PROCESSED_DATASET / split / emotion).mkdir(
            parents=True,
            exist_ok=True,
        )

# ------------------------------------------------------------------
# Copy helper
# ------------------------------------------------------------------

metadata = []


def copy_split(split_name, split_df):

    print(f"\nCreating {split_name} dataset...")

    for row_idx, row in tqdm(
        split_df.iterrows(),
        total=len(split_df),
    ):

        emotion = row["emotion_name"]

        filename = f"row_{row_idx:05d}.png"

        source = FER_PATH / emotion / filename

        destination = (
            PROCESSED_DATASET
            / split_name
            / emotion
            / filename
        )

        shutil.copy2(source, destination)

        metadata.append(
            {
                "row": row_idx,
                "emotion": emotion,
                "split": split_name,
                "usage": row["Usage"],
                "filename": filename,
            }
        )


copy_split("train", train_df)
copy_split("validation", val_df)
copy_split("test", test_df)

# ------------------------------------------------------------------
# Save metadata
# ------------------------------------------------------------------

metadata_df = pd.DataFrame(metadata)

metadata_df.to_csv(
    PROCESSED_DATASET / "split_metadata.csv",
    index=False,
)

print("\nDataset successfully created.")
print(f"\nLocation:\n{PROCESSED_DATASET}")