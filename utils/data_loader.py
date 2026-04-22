"""
utils/data_loader.py
====================
Handles dataset download (via kagglehub), directory scanning,
image loading, augmentation, and train/val/test splitting.
"""

import os
import json
import shutil
import numpy as np
import tensorflow as tf
from pathlib import Path
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array


# ──────────────────────────────────────────────
#  Constants
# ──────────────────────────────────────────────
IMG_SIZE   = (224, 224)
BATCH_SIZE = 32
SEED       = 42


# ──────────────────────────────────────────────
#  Dataset download
# ──────────────────────────────────────────────
def download_dataset(dest_dir: str = "data") -> str:
    """
    Download the Kaggle brain-tumor MRI dataset using kagglehub.
    Returns the directory that contains 'yes/' and 'no/' sub-folders.
    """
    try:
        import kagglehub
        print("[INFO] Downloading dataset via kagglehub …")
        raw_path = kagglehub.dataset_download(
            "navoneel/brain-mri-images-for-brain-tumor-detection"
        )
        print(f"[INFO] Downloaded to: {raw_path}")
    except Exception as exc:
        raise RuntimeError(
            f"Failed to download dataset: {exc}\n"
            "Make sure your Kaggle API credentials are configured."
        )

    # The dataset is already split into yes/ no/ folders – find them
    raw_path = Path(raw_path)
    for candidate in [raw_path, *raw_path.rglob("*")]:
        if candidate.is_dir() and (
            (candidate / "yes").exists() or (candidate / "tumor").exists()
        ):
            print(f"[INFO] Dataset root found at: {candidate}")
            return str(candidate)

    # Fallback – return root
    return str(raw_path)


# ──────────────────────────────────────────────
#  File listing
# ──────────────────────────────────────────────
def _collect_files(dataset_dir: str) -> tuple[list, list]:
    """
    Scan dataset_dir for yes/ (tumor) and no/ (no-tumor) images.
    Returns (file_paths, labels) where label 1=tumor, 0=no-tumor.
    """
    dataset_dir = Path(dataset_dir)

    # Support both naming conventions
    yes_candidates = ["yes", "tumor", "Yes", "Tumor", "YES", "TUMOR"]
    no_candidates  = ["no", "no_tumor", "No", "No_Tumor", "NO", "NO_TUMOR"]

    yes_dir = no_dir = None
    for name in yes_candidates:
        p = dataset_dir / name
        if p.exists():
            yes_dir = p
            break
    for name in no_candidates:
        p = dataset_dir / name
        if p.exists():
            no_dir = p
            break

    if yes_dir is None or no_dir is None:
        raise FileNotFoundError(
            f"Could not find yes/no sub-directories in {dataset_dir}. "
            f"Found: {[d.name for d in dataset_dir.iterdir() if d.is_dir()]}"
        )

    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

    file_paths, labels = [], []
    for f in yes_dir.iterdir():
        if f.suffix.lower() in extensions:
            file_paths.append(str(f))
            labels.append(1)

    for f in no_dir.iterdir():
        if f.suffix.lower() in extensions:
            file_paths.append(str(f))
            labels.append(0)

    print(f"[INFO] Found {labels.count(1)} tumor images, "
          f"{labels.count(0)} non-tumor images  (total={len(labels)})")
    return file_paths, labels


# ──────────────────────────────────────────────
#  Data generators
# ──────────────────────────────────────────────
def build_data_generators(
    dataset_dir: str,
    img_size: tuple = IMG_SIZE,
    batch_size: int = BATCH_SIZE,
    val_split: float = 0.15,
    test_split: float = 0.15,
):
    """
    Returns (train_gen, val_gen, test_gen, class_weights, stats_dict).
    """
    file_paths, labels = _collect_files(dataset_dir)

    # ── split ────────────────────────────────────────
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        file_paths, labels,
        test_size=(val_split + test_split),
        stratify=labels,
        random_state=SEED,
    )
    relative_test = test_split / (val_split + test_split)
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp,
        test_size=relative_test,
        stratify=y_tmp,
        random_state=SEED,
    )

    print(f"[INFO] Train={len(X_train)}  Val={len(X_val)}  Test={len(X_test)}")

    # ── class weights ────────────────────────────────
    n_tumor    = sum(y_train)
    n_no_tumor = len(y_train) - n_tumor
    total      = len(y_train)
    class_weight = {
        0: total / (2.0 * n_no_tumor),
        1: total / (2.0 * n_tumor),
    }

    # ── augmentation (train only) ────────────────────
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        vertical_flip=False,
        fill_mode="nearest",
    )
    val_datagen = ImageDataGenerator(rescale=1.0 / 255)

    def _make_df(paths, lbls):
        import pandas as pd
        return pd.DataFrame(
            {"filename": paths, "label": [str(l) for l in lbls]}
        )

    train_df = _make_df(X_train, y_train)
    val_df   = _make_df(X_val,   y_val)
    test_df  = _make_df(X_test,  y_test)

    train_gen = train_datagen.flow_from_dataframe(
        train_df, x_col="filename", y_col="label",
        target_size=img_size, batch_size=batch_size,
        class_mode="binary", shuffle=True, seed=SEED,
    )
    val_gen = val_datagen.flow_from_dataframe(
        val_df, x_col="filename", y_col="label",
        target_size=img_size, batch_size=batch_size,
        class_mode="binary", shuffle=False,
    )
    test_gen = val_datagen.flow_from_dataframe(
        test_df, x_col="filename", y_col="label",
        target_size=img_size, batch_size=batch_size,
        class_mode="binary", shuffle=False,
    )

    stats = {
        "total_images": len(file_paths),
        "train_size":   len(X_train),
        "val_size":     len(X_val),
        "test_size":    len(X_test),
        "tumor_count":  labels.count(1),
        "no_tumor_count": labels.count(0),
    }

    return train_gen, val_gen, test_gen, class_weight, stats


# ──────────────────────────────────────────────
#  Single-image preprocessor (for inference)
# ──────────────────────────────────────────────
def preprocess_image(image_path: str, img_size: tuple = IMG_SIZE) -> np.ndarray:
    """
    Load + preprocess a single image for inference.
    Returns shape (1, H, W, 3), values in [0, 1].
    """
    img    = load_img(image_path, target_size=img_size)
    arr    = img_to_array(img) / 255.0
    return np.expand_dims(arr, axis=0)
