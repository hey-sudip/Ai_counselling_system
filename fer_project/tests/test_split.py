from pathlib import Path
import hashlib


def get_image_hashes(folder):
    hashes = {}

    for path in Path(folder).rglob("*"):
        if path.is_file():
            try:
                data = path.read_bytes()
                file_hash = hashlib.md5(data).hexdigest()

                if file_hash not in hashes:
                    hashes[file_hash] = []

                hashes[file_hash].append(path)

            except Exception:
                pass

    return hashes


train_dir = Path("dataset/processed/train")
val_dir = Path("dataset/processed/validation")
test_dir = Path("dataset/processed/test")


print("=" * 70)
print("DETAILED DATASET LEAKAGE CHECK")
print("=" * 70)


print("\nScanning datasets...")

train_hashes = get_image_hashes(train_dir)
val_hashes = get_image_hashes(val_dir)
test_hashes = get_image_hashes(test_dir)


train_val = set(train_hashes) & set(val_hashes)
train_test = set(train_hashes) & set(test_hashes)
val_test = set(val_hashes) & set(test_hashes)


print("\nDuplicate counts:")
print(f"Train ↔ Validation : {len(train_val)}")
print(f"Train ↔ Test       : {len(train_test)}")
print(f"Validation ↔ Test  : {len(val_test)}")


print("\n" + "=" * 70)
print("SAMPLE DUPLICATES")
print("=" * 70)


print("\n--- TRAIN ↔ VALIDATION ---")

for i, file_hash in enumerate(train_val):

    if i >= 10:
        break

    print("\nTrain:")
    for path in train_hashes[file_hash]:
        print(" ", path)

    print("Validation:")
    for path in val_hashes[file_hash]:
        print(" ", path)


print("\n--- TRAIN ↔ TEST ---")

for i, file_hash in enumerate(train_test):

    if i >= 10:
        break

    print("\nTrain:")
    for path in train_hashes[file_hash]:
        print(" ", path)

    print("Test:")
    for path in test_hashes[file_hash]:
        print(" ", path)


print("\n--- VALIDATION ↔ TEST ---")

for i, file_hash in enumerate(val_test):

    if i >= 10:
        break

    print("\nValidation:")
    for path in val_hashes[file_hash]:
        print(" ", path)

    print("Test:")
    for path in test_hashes[file_hash]:
        print(" ", path)


print("\n" + "=" * 70)
print("CHECK COMPLETE")
print("=" * 70)