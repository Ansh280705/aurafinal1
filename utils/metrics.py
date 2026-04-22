"""
utils/metrics.py
================
Evaluation helpers:
  - Plot accuracy / loss training curves
  - Plot confusion matrix
  - Generate classification report
  - Save model performance stats to JSON
"""

import json
import itertools
import numpy as np
import matplotlib
matplotlib.use("Agg")           # non-interactive backend (safe for servers)
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
)


# ──────────────────────────────────────────────
#  Training curve plots
# ──────────────────────────────────────────────
def plot_training_history(history, save_path: str = "models/training_history.png"):
    """
    Plot accuracy and loss curves side-by-side and save to disk.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#0F1117")

    for ax in axes:
        ax.set_facecolor("#1A1D27")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#2E3250")

    # Accuracy
    axes[0].plot(history["accuracy"],     color="#4ECDC4", linewidth=2,   label="Train Acc")
    axes[0].plot(history["val_accuracy"], color="#FF6B6B", linewidth=2,   label="Val Acc",   linestyle="--")
    axes[0].set_title("Model Accuracy",  fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend(facecolor="#2E3250", labelcolor="white")
    axes[0].grid(alpha=0.2)

    # Loss
    axes[1].plot(history["loss"],     color="#4ECDC4", linewidth=2,   label="Train Loss")
    axes[1].plot(history["val_loss"], color="#FF6B6B", linewidth=2,   label="Val Loss",   linestyle="--")
    axes[1].set_title("Model Loss",   fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend(facecolor="#2E3250", labelcolor="white")
    axes[1].grid(alpha=0.2)

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[INFO] Training history saved → {save_path}")


# ──────────────────────────────────────────────
#  Confusion matrix
# ──────────────────────────────────────────────
def plot_confusion_matrix(
    y_true: list | np.ndarray,
    y_pred: list | np.ndarray,
    class_names: list = ["No Tumor", "Tumor"],
    save_path: str = "models/confusion_matrix.png",
):
    """
    Plot and save a confusion matrix heatmap.
    """
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    fig.patch.set_facecolor("#0F1117")
    ax.set_facecolor("#1A1D27")

    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.tick_params(colors="white")

    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
        title="Confusion Matrix",
    )

    ax.title.set_color("white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.tick_params(colors="white")

    # Annotate cells
    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(
            j, i, format(cm[i, j], "d"),
            ha="center", va="center",
            color="white" if cm[i, j] < thresh else "black",
            fontsize=14, fontweight="bold",
        )

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[INFO] Confusion matrix saved → {save_path}")
    return cm


# ──────────────────────────────────────────────
#  Classification report
# ──────────────────────────────────────────────
def print_and_save_report(
    y_true: list | np.ndarray,
    y_pred: list | np.ndarray,
    class_names: list = ["No Tumor", "Tumor"],
    save_path: str = "models/classification_report.txt",
) -> dict:
    """
    Print + save classification report, return as dict.
    """
    report_str  = classification_report(y_true, y_pred, target_names=class_names)
    report_dict = classification_report(y_true, y_pred, target_names=class_names,
                                        output_dict=True)

    print("\n" + "=" * 50)
    print("CLASSIFICATION REPORT")
    print("=" * 50)
    print(report_str)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        f.write(report_str)
    print(f"[INFO] Classification report saved → {save_path}")

    return report_dict


# ──────────────────────────────────────────────
#  Performance stats JSON (used by Streamlit)
# ──────────────────────────────────────────────
def save_performance_stats(
    history: dict,
    report_dict: dict,
    test_loss: float,
    test_accuracy: float,
    dataset_stats: dict,
    save_path: str = "models/performance_stats.json",
):
    """
    Serialise key metrics to JSON so the Streamlit app can load them.
    """
    stats = {
        "test_loss":     round(float(test_loss), 4),
        "test_accuracy": round(float(test_accuracy), 4),
        "best_val_accuracy": round(float(max(history["val_accuracy"])), 4),
        "best_val_loss":     round(float(min(history["val_loss"])),     4),
        "epochs_trained":    len(history["accuracy"]),
        "report":            report_dict,
        "dataset":           dataset_stats,
    }

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"[INFO] Performance stats saved → {save_path}")
    return stats
