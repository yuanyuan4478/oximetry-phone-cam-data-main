from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from train_linear_regression import (
    SUBJECT_IDS,
    _fit_standardizer,
    _standardize,
    collect_dataset,
)
from sklearn.linear_model import LinearRegression


def compute_predictions(subject_id: str):
    datasets = collect_dataset()
    held_out = next(ds for ds in datasets if ds.subject_id == subject_id)
    train_features = np.vstack(
        [ds.features for ds in datasets if ds.subject_id != subject_id]
    )
    train_labels = np.concatenate(
        [ds.labels for ds in datasets if ds.subject_id != subject_id]
    )

    mean, std = _fit_standardizer(train_features)
    train_scaled = _standardize(train_features, mean, std)

    model = LinearRegression()
    model.fit(train_scaled, train_labels)

    test_scaled = _standardize(held_out.features, mean, std)
    preds = model.predict(test_scaled)
    return held_out.labels, preds


def plot_subject(subject_id: str, output_path: Path, threshold: float = 90.0) -> Path:
    labels, preds = compute_predictions(subject_id)
    time_axis = np.arange(labels.shape[0])

    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(9, 4))
    plt.plot(time_axis, labels, label="Ground truth", linewidth=1.4, color="black")
    plt.plot(time_axis, preds, label="Linear regression", linewidth=1.4, color="#1f77b4")
    plt.axhline(threshold, color="#ff7f0e", linestyle="--", linewidth=1.0, label=f"Threshold {threshold:.0f}%")
    plt.xlabel("Seconds")
    plt.ylabel("SpO₂ (%)")
    plt.title(f"Subject {subject_id}: Linear Regression vs Ground Truth")
    plt.ylim(60, 100)
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.text(0.99, 0.02, "Models: LinearRegression", ha="right", va="bottom", transform=plt.gca().transAxes, fontsize=8, alpha=0.7)
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate linear regression plots for hypoxemia data.")
    parser.add_argument(
        "--subject",
        choices=SUBJECT_IDS,
        default="100001",
        help="Subject identifier to plot (default: 100001)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("figures/linear_regression_subject_100001.png"),
        help="Output image path (default: figures/linear_regression_subject_100001.png)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=90.0,
        help="SpO₂ threshold to display (default: 90.0)",
    )
    args = parser.parse_args()

    if args.output.is_dir():
        output_path = args.output / f"linear_regression_subject_{args.subject}.png"
    else:
        output_path = args.output

    result_path = plot_subject(args.subject, output_path, args.threshold)
    print(f"Saved linear regression figure to {result_path}")


if __name__ == "__main__":
    main()
