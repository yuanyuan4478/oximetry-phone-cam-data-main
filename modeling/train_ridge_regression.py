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


def _loocv_predict(datasets, subject_id: str, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    held_out = next(ds for ds in datasets if ds.subject_id == subject_id)
    train_features = np.vstack([ds.features for ds in datasets if ds.subject_id != subject_id])
    train_labels = np.concatenate([ds.labels for ds in datasets if ds.subject_id != subject_id])

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
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    time_axis = np.arange(labels.shape[0])

    accuracy = float(np.mean((preds >= threshold) == (labels >= threshold)))
    mae = float(np.mean(np.abs(preds - labels)))

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(time_axis, labels, label="Ground truth", linewidth=1.4, color="black")
    ax.plot(time_axis, preds, label="Ridge prediction", linewidth=1.4, color="#ff7f0e")
    ax.axhline(threshold, color="#1f77b4", linestyle="--", linewidth=1.0, label=f"Threshold {threshold:.0f}%")
    ax.set_xlabel("Seconds")
    ax.set_ylabel("SpO₂ (%)")
    ax.set_title(f"Subject {subject_id}: Ridge Regression vs Ground Truth")
    ax.set_ylim(60, 100)
    ax.legend(loc="lower right", fontsize=8)
    ax.text(
        0.01,
        0.95,
        f"Accuracy: {accuracy*100:.1f}%\nMAE: {mae:.2f} %",
        transform=ax.transAxes,
        fontsize=8,
        bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
        va="top",
    )
    ax.text(
        0.99,
        0.02,
        "Models: RidgeRegression",
        ha="right",
        va="bottom",
        transform=ax.transAxes,
        fontsize=8,
        alpha=0.7,
    )
    fig.tight_layout()

    output_path = output_dir / f"ridge_regression_subject_{subject_id}.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def run_training(alpha: float, threshold: float, output_dir: Path) -> None:
    datasets = collect_dataset()
    metrics = []
    for subject_id in SUBJECT_IDS:
        labels, preds = _loocv_predict(datasets, subject_id, alpha)
        accuracy = float(np.mean((preds >= threshold) == (labels >= threshold)))
        mae = float(np.mean(np.abs(preds - labels)))
        _plot_subject(subject_id, labels, preds, threshold, output_dir)
        metrics.append((subject_id, accuracy, mae))

    print("Ridge Regression Leave-One-Subject-Out Results")
    print("Subject  Accuracy(%)  MAE(% SpO2)")
    for subject_id, accuracy, mae in metrics:
        print(f"{subject_id}    {accuracy*100:8.2f}    {mae:8.3f}")

    avg_accuracy = float(np.mean([m[1] for m in metrics]))
    avg_mae = float(np.mean([m[2] for m in metrics]))
    print("\nOverall averages:")
    print(f"Accuracy: {avg_accuracy*100:.2f}%")
    print(f"MAE: {avg_mae:.3f} %")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Ridge Regression for SpO₂ prediction with LOSO evaluation.")
    parser.add_argument(
        "--alpha",
        type=float,
        default=25.0,
        help="Regularization strength for ridge regression (default: 25.0)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=90.0,
        help="SpO₂ threshold for hypoxemia accuracy (default: 90.0)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("figures"),
        help="Directory where subject plots will be saved (default: figures/)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output if args.output.suffix == "" else args.output.parent
    run_training(args.alpha, args.threshold, output_dir)


if __name__ == "__main__":
    main()
