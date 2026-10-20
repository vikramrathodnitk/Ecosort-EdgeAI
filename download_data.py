"""Prepare the EcoSort dataset from a ZIP file already on your computer.

Run from your project folder:
    python download_data.py

This script does not download data from the internet. Put the archive at:
    data/ecosort_dataset_public.zip
"""

from pathlib import Path
import subprocess
import sys
import zipfile


# The folder containing this script is the project folder.
PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
ARCHIVE_PATH = DATA_DIR / "ecosort_dataset_public.zip"
DATASET_DIR = DATA_DIR / "public"

EXPECTED_CLASSES = [
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash",
]


def dataset_is_ready():
    """Check for train/val folders and all six class folders."""
    train_dir = DATASET_DIR / "train"
    val_dir = DATASET_DIR / "val"

    if not train_dir.is_dir() or not val_dir.is_dir():
        return False

    train_classes_exist = all(
        (train_dir / name).is_dir() for name in EXPECTED_CLASSES
    )
    val_classes_exist = all(
        (val_dir / name).is_dir() for name in EXPECTED_CLASSES
    )

    return train_classes_exist and val_classes_exist


def count_images(folder):
    """Count common image files inside a folder and its subfolders."""
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    return sum(
        1
        for path in folder.rglob("*")
        if path.suffix.lower() in image_extensions
    )


def print_dataset_summary():
    """Show the image count and class folders for each split."""
    print("\nDataset is ready:")

    for split_name in ("train", "val"):
        split_dir = DATASET_DIR / split_name
        class_names = sorted(
            folder.name for folder in split_dir.iterdir() if folder.is_dir()
        )
        image_count = count_images(split_dir)

        print(f"  {split_name}: {image_count} images in {len(class_names)} folders")
        print(f"    Classes: {', '.join(class_names)}")

    print("\nNext step:")
    print("  python train_template.py --data_dir data/public")


def extract_archive():
    """Extract the dataset ZIP into data/public/."""
    if not ARCHIVE_PATH.is_file():
        return False

    print(f"Found dataset archive: {ARCHIVE_PATH}")
    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    try:
        with zipfile.ZipFile(ARCHIVE_PATH, "r") as archive:
            archive.extractall(DATASET_DIR)
    except (zipfile.BadZipFile, OSError) as error:
        print(f"Could not extract the dataset archive: {error}")
        return False

    if not dataset_is_ready():
        print("The ZIP was extracted, but the folder layout was not expected.")
        print("Expected data/public/train/<class>/ and data/public/val/<class>/.")
        return False

    return True


def run_preparation_script():
    """Run an optional local dataset preparation script, if available."""
    prep_script = DATA_DIR / "prepare_dataset.py"

    if not prep_script.is_file():
        return False

    print(f"No ZIP found. Running: {prep_script}")

    try:
        subprocess.run([sys.executable, str(prep_script)], check=True)
    except subprocess.CalledProcessError as error:
        print(f"The preparation script returned an error: {error}")
        return False

    return dataset_is_ready()


def main():
    print("=" * 60)
    print("EcoSort Dataset Setup")
    print("=" * 60)
    print(f"Project folder: {PROJECT_DIR}")
    print(f"Dataset folder: {DATASET_DIR}")

    # If the data is already prepared, there is nothing to extract.
    if dataset_is_ready():
        print("Training and validation folders are already present.")
        print_dataset_summary()
        return 0

    # Try the ZIP first, then the optional preparation script.
    if extract_archive() or run_preparation_script():
        print_dataset_summary()
        return 0

    print("\nDataset setup could not be completed.")
    print("Get the dataset ZIP from your challenge source and put it here:")
    print(f"  {ARCHIVE_PATH}")
    print("The ZIP must contain train/ and val/ folders with the six class folders.")
    print("You can also add data/prepare_dataset.py and run this script again.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())