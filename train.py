"""
train.py
========
End-to-end training script for the Brain Tumor Detection model.

Usage
-----
    python train.py [--epochs 30] [--batch-size 32] [--lr 1e-4]
                    [--data-dir data] [--model-dir models]

Steps
-----
1. Download dataset (kagglehub)
2. Build data generators with augmentation
3. Build VGG16 transfer learning model
4. Train with early stopping & learning rate reduction
5. Evaluate on test set
6. Save model, history plots, confusion matrix, and metrics JSON
"""

import os
import sys
import json
import argparse
import numpy as np
import tensorflow as tf
from pathlib import Path

# ── add project root to path ──────────────────
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from utils.data_loader   import download_dataset, build_data_generators
from utils.model_builder import build_vgg16_model, get_model_summary
from utils.metrics       import (
    plot_training_history,
    plot_confusion_matrix,
    print_and_save_report,
    save_performance_stats,
)


# ──────────────────────────────────────────────
#  CLI arguments
# ──────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Brain Tumor Detection model (VGG16 Transfer Learning)"
    )
    parser.add_argument("--epochs",     type=int,   default=30,
                        help="Maximum training epochs (default: 30)")
    parser.add_argument("--batch-size", type=int,   default=32,
                        help="Batch size (default: 32)")
    parser.add_argument("--lr",         type=float, default=1e-4,
                        help="Initial learning rate (default: 1e-4)")
    parser.add_argument("--data-dir",   type=str,   default="data",
                        help="Directory where dataset will be stored")
    parser.add_argument("--model-dir",  type=str,   default="models",
                        help="Directory where model + artefacts are saved")
    parser.add_argument("--fine-tune-at", type=int, default=15,
                        help="Freeze VGG16 layers before this index (default: 15)")
    parser.add_argument("--patience",   type=int,   default=7,
                        help="EarlyStopping patience (default: 7)")
    return parser.parse_args()


# ──────────────────────────────────────────────
#  Reproducibility
# ──────────────────────────────────────────────
def set_seeds(seed: int = 42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


# ──────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────
def main():
    args = parse_args()
    set_seeds()

    MODEL_DIR   = Path(args.model_dir)
    MODEL_PATH  = MODEL_DIR / "model.h5"
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # ── GPU memory growth ────────────────────────
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"[INFO] GPUs detected: {len(gpus)}")
    else:
        print("[INFO] No GPU detected – running on CPU")

    # ─────────────────────────────────────────────
    #  1. Data
    # ─────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("  STEP 1: Dataset download & preparation")
    print("=" * 55)

    dataset_dir = download_dataset(dest_dir=args.data_dir)

    train_gen, val_gen, test_gen, class_weight, dataset_stats = (
        build_data_generators(
            dataset_dir,
            batch_size=args.batch_size,
        )
    )

    # ─────────────────────────────────────────────
    #  2. Model
    # ─────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("  STEP 2: Building model")
    print("=" * 55)

    model = build_vgg16_model(
        learning_rate=args.lr,
        fine_tune_at=args.fine_tune_at,
    )
    print(get_model_summary(model))

    # ─────────────────────────────────────────────
    #  3. Callbacks
    # ─────────────────────────────────────────────
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=args.patience,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(MODEL_PATH),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1,
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=str(MODEL_DIR / "logs"),
            histogram_freq=1,
        ),
    ]

    # ─────────────────────────────────────────────
    #  4. Training
    # ─────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("  STEP 3: Training")
    print("=" * 55)

    history = model.fit(
        train_gen,
        epochs=args.epochs,
        validation_data=val_gen,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1,
    )
    history_dict = history.history

    # ─────────────────────────────────────────────
    #  5. Evaluation on test set
    # ─────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("  STEP 4: Evaluation on test set")
    print("=" * 55)

    test_loss, test_acc, *_ = model.evaluate(test_gen, verbose=1)
    print(f"\n  Test Loss     : {test_loss:.4f}")
    print(f"  Test Accuracy : {test_acc:.4f}")

    # Predictions for confusion matrix
    test_gen.reset()
    preds  = model.predict(test_gen, verbose=1)
    y_pred = (preds.squeeze() >= 0.5).astype(int)
    y_true = np.array(test_gen.labels).astype(int)

    # ─────────────────────────────────────────────
    #  6. Artefacts
    # ─────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("  STEP 5: Saving artefacts")
    print("=" * 55)

    # Training history plot
    plot_training_history(
        history_dict,
        save_path=str(MODEL_DIR / "training_history.png"),
    )

    # Confusion matrix
    plot_confusion_matrix(
        y_true, y_pred,
        save_path=str(MODEL_DIR / "confusion_matrix.png"),
    )

    # Classification report
    report_dict = print_and_save_report(
        y_true, y_pred,
        save_path=str(MODEL_DIR / "classification_report.txt"),
    )

    # Performance stats JSON (consumed by Streamlit)
    save_performance_stats(
        history=history_dict,
        report_dict=report_dict,
        test_loss=test_loss,
        test_accuracy=test_acc,
        dataset_stats=dataset_stats,
        save_path=str(MODEL_DIR / "performance_stats.json"),
    )

    # Save training history (for optional reload)
    with open(MODEL_DIR / "history.json", "w") as f:
        json.dump(
            {k: [float(v) for v in vals] for k, vals in history_dict.items()},
            f, indent=2,
        )

    # Also explicitly save (ModelCheckpoint saves best, but save final too)
    model.save(str(MODEL_DIR / "model_final.h5"))

    print("\n" + "=" * 55)
    print("  ✅ Training complete!")
    print(f"  Best model    : {MODEL_PATH}")
    print(f"  Test accuracy : {test_acc * 100:.2f}%")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    main()
