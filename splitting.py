from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_ROOT = SCRIPT_DIR / "processed_data"
IMAGE_DIR = DATA_ROOT / "images"
MASK_DIR = DATA_ROOT / "masks"
SPLIT_PATH = DATA_ROOT / "case_splits.csv"

def numeric_case_sort(case_id: str) -> tuple[int, str]:
    return (int(case_id), case_id) if case_id.isdigit() else (10**12, case_id)

def collect_case_ids() -> list[str]:
    image_cases = {p.stem.removesuffix("_X") for p in IMAGE_DIR.glob("*_X.npy")}
    mask_cases = {p.stem.removesuffix("_Y") for p in MASK_DIR.glob("*_Y.npy")}

    if not image_cases:
        raise FileNotFoundError(f"No *_X.npy files found in {IMAGE_DIR}")

    case_ids = sorted(image_cases & mask_cases, key=numeric_case_sort)
    missing_masks = sorted(image_cases - mask_cases, key=numeric_case_sort)

    if missing_masks:
        print(
            f"Warning: skipping {len(missing_masks)} image case(s) without matching "
            f"*_Y.npy masks. First few: {missing_masks[:10]}"
        )

    if not case_ids:
        raise FileNotFoundError(
            f"No matched image/mask pairs found in {IMAGE_DIR} and {MASK_DIR}"
        )

    return case_ids


case_ids = collect_case_ids()

train_cases, temp_cases = train_test_split(
    case_ids,
    test_size=0.30,
    random_state=30,
)

val_cases, test_cases = train_test_split(
    temp_cases,
    test_size=0.50,
    random_state=30,
)

split_rows = []

for case_id in train_cases:
    split_rows.append({"case_id": case_id, "split": "train"})

for case_id in val_cases:
    split_rows.append({"case_id": case_id, "split": "val"})

for case_id in test_cases:
    split_rows.append({"case_id": case_id, "split": "test"})

splits_df = pd.DataFrame(split_rows)
splits_df.to_csv(SPLIT_PATH, index=False)

print(f"Wrote {SPLIT_PATH}")
print(splits_df["split"].value_counts())
print(splits_df.head())
