from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

from train_linear_regression import (
    SUBJECT_IDS,
    _fit_standardizer,
    _standardize,
    collect_dataset,
)


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

    accuracy = float(np.mean((preds >= threshold) == (labels >= threshold)))
    mae = float(np.mean(np.abs(preds - labels)))

    plt.figure(figsize=(9, 4))
    plt.plot(time_axis, labels, label="Ground truth", linewidth=1.4, color="black")
    plt.plot(time_axis, preds, label="Linear regression", linewidth=1.4, color="#1f77b4")
    plt.axhline(
        threshold,
        color="#ff7f0e",
        linestyle="--",
        linewidth=1.0,
        label=f"Threshold {threshold:.0f}%",
    )
    plt.xlabel("Seconds")
    plt.ylabel("SpO₂ (%)")
    plt.title(f"Subject {subject_id}: Linear Regression vs Ground Truth")
    plt.ylim(60, 100)
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    ax = plt.gca()
    ax.text(
        0.01,
        0.95,
        f"Accuracy: {accuracy*100:.1f}%\nMAE: {mae:.2f} %",
        transform=ax.transAxes,
        fontsize=8,
        bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
        verticalalignment="top",
    )
    ax.text(
        0.99,
        0.02,
        "Models: LinearRegression",
        ha="right",
        va="bottom",
        transform=ax.transAxes,
        fontsize=8,
        alpha=0.7,
    )
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path


def resolve_output_path(base: Path, subject_id: str) -> Path:
    if base.suffix:
        base.parent.mkdir(parents=True, exist_ok=True)
        return base
    base.mkdir(parents=True, exist_ok=True)
    return base / f"linear_regression_subject_{subject_id}.png"


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
        default=Path("figures"),
        help="Output file or directory (default: figures/)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=90.0,
        help="SpO₂ threshold to display (default: 90.0)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Generate plots for all subjects, ignoring --subject.",
    )
    args = parser.parse_args()

    if args.all:
        output_dir = args.output if not args.output.suffix else args.output.parent
        output_dir.mkdir(parents=True, exist_ok=True)
        for sid in SUBJECT_IDS:
            result_path = plot_subject(sid, resolve_output_path(output_dir, sid), args.threshold)
            print(f"Saved linear regression figure to {result_path}")
    else:
        result_path = plot_subject(
            args.subject,
            resolve_output_path(args.output, args.subject),
            args.threshold,
        )
        print(f"Saved linear regression figure to {result_path}")


if __name__ == "__main__":
    main()
