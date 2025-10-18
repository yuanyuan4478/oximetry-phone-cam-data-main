from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_curve, auc

from train_linear_regression import _fit_standardizer, _standardize, collect_dataset

THRESHOLDS_DEFAULT = [92.0, 90.0, 88.0]
ALPHA_DEFAULT = 25.0
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c"]


def _collect_predictions(alpha: float) -> tuple[np.ndarray, np.ndarray]:
    datasets = collect_dataset()
    labels_all: list[np.ndarray] = []
    preds_all: list[np.ndarray] = []

    for held_out in datasets:
        train_features = np.vstack(
            [ds.features for ds in datasets if ds.subject_id != held_out.subject_id]
        )
        train_labels = np.concatenate(
            [ds.labels for ds in datasets if ds.subject_id != held_out.subject_id]
        )
        mean, std = _fit_standardizer(train_features)
        model = Ridge(alpha=alpha)
        model.fit(_standardize(train_features, mean, std), train_labels)
        preds = model.predict(_standardize(held_out.features, mean, std))
        labels_all.append(held_out.labels)
        preds_all.append(preds)

    return np.concatenate(labels_all), np.concatenate(preds_all)


def _plot_roc(labels: np.ndarray, preds: np.ndarray, thresholds: list[float], output: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6, 6))

    annotation_offsets = [0.9, 0.7, 0.5]
    for thr, color, annot_y in zip(thresholds, COLORS, annotation_offsets):
        y_true = (labels < thr).astype(int)
        fpr, tpr, _ = roc_curve(y_true, preds)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, linewidth=2, label=f"<{int(thr)}%")
        ax.text(
            0.55,
            annot_y,
            f"AUC({int(thr)}) = {roc_auc:.2f}",
            color=color,
            fontsize=9,
        )
        for frac in [0.2, 0.5, 0.8]:
            idx = int(frac * (len(fpr) - 1))
            ax.scatter(fpr[idx], tpr[idx], color=color, s=18, edgecolors="white", linewidths=0.5)
            ax.text(
                fpr[idx] + 0.02,
                tpr[idx],
                f"({fpr[idx]:.2f}, {tpr[idx]:.2f})",
                color=color,
                fontsize=8,
            )

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.4)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("1 - Specificity (FPR)")
    ax.set_ylabel("Sensitivity (TPR)")
    ax.set_title("SpO Classification ROC Curve")
    ax.legend(title="Classifying as Below:")

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot ROC curves for SpO₂ classification thresholds")
    parser.add_argument(
        "--alpha",
        type=float,
        default=ALPHA_DEFAULT,
        help="Ridge regression alpha (default: 25.0)",
    )
    parser.add_argument(
        "--thresholds",
        nargs="*",
        type=float,
        default=THRESHOLDS_DEFAULT,
        help="List of thresholds for classification (default: 92 90 88)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("figures/spo2_classification_roc.png"),
        help="Output figure path (default: figures/spo2_classification_roc.png)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    thresholds = args.thresholds if args.thresholds else THRESHOLDS_DEFAULT
    labels, preds = _collect_predictions(args.alpha)
    result_path = _plot_roc(labels, preds, thresholds, args.output)
    print(f"Saved ROC curve to {result_path}")


if __name__ == "__main__":
    main()
