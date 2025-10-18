from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Ridge

from train_linear_regression import (
    SUBJECT_IDS,
    _fit_standardizer,
    _standardize,
    collect_dataset,
)

CORRECT_COLOR = "#2ca02c"
INCORRECT_COLOR = "#d62728"
RIDGE_COLOR = "#ff7f0e"
THRESHOLD_DEFAULT = 90.0
ALPHA_DEFAULT = 25.0


def _loocv_predict(datasets, subject_id: str, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    held_out = next(ds for ds in datasets if ds.subject_id == subject_id)
    train_features = np.vstack(
        [ds.features for ds in datasets if ds.subject_id != subject_id]
    )
    train_labels = np.concatenate(
        [ds.labels for ds in datasets if ds.subject_id != subject_id]
    )

    mean, std = _fit_standardizer(train_features)
    model = Ridge(alpha=alpha)
    model.fit(_standardize(train_features, mean, std), train_labels)

    preds = model.predict(_standardize(held_out.features, mean, std))
    return held_out.labels, preds


def _plot_subject(
    subject_id: str,
    labels: np.ndarray,
    preds: np.ndarray,
    threshold: float,
    output_dir: Path,
) -> tuple[Path, float]:
    output_dir.mkdir(parents=True, exist_ok=True)
    time_axis = np.arange(labels.shape[0])

    truth = labels >= threshold
    prediction = preds >= threshold
    correct_mask = truth == prediction

    accuracy = float(np.mean(correct_mask))

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(time_axis, labels, label="Ground truth", linewidth=1.4, color="black")
    ax.plot(time_axis, preds, label="Ridge prediction", linewidth=1.4, color=RIDGE_COLOR)
    ax.axhline(threshold, color="#1f77b4", linestyle="--", linewidth=1.0, label=f"Threshold {threshold:.0f}%")

    if correct_mask.any():
        ax.scatter(
            time_axis[correct_mask],
            preds[correct_mask],
            c=CORRECT_COLOR,
            s=16,
            label="Correct",
        )
    if (~correct_mask).any():
        ax.scatter(
            time_axis[~correct_mask],
            preds[~correct_mask],
            c=INCORRECT_COLOR,
            s=20,
            label="Incorrect",
        )

    ax.set_xlabel("Seconds")
    ax.set_ylabel("SpO₂ (%)")
    ax.set_title(f"Subject {subject_id}: Ridge Regression vs Ground Truth")
    ax.set_ylim(60, 100)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()

    output_path = output_dir / f"ridge_correctness_subject_{subject_id}.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path, accuracy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Ridge prediction correctness versus ground truth")
    parser.add_argument(
        "--subjects",
        nargs="*",
        default=SUBJECT_IDS,
        help="Subject IDs to plot (default: all subjects)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=THRESHOLD_DEFAULT,
        help="Classification threshold (default: 90.0)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=ALPHA_DEFAULT,
        help="Ridge regression alpha (default: 25.0)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("figures"),
        help="Directory to save plots (default: figures/)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    subjects = args.subjects if args.subjects else SUBJECT_IDS
    datasets = collect_dataset()

    for subject_id in subjects:
        labels, preds = _loocv_predict(datasets, subject_id, args.alpha)
        _, accuracy = _plot_subject(subject_id, labels, preds, args.threshold, args.output_dir)
        print(f"Subject {subject_id}: accuracy {accuracy*100:.2f}%")


if __name__ == "__main__":
    main()
