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

THRESHOLDS = [92.0, 90.0, 88.0]


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


def _compute_metrics(labels: np.ndarray, preds: np.ndarray, threshold: float) -> tuple[float, float]:
    true_pos = labels < threshold
    pred_pos = preds < threshold

    tp = np.sum(pred_pos & true_pos)
    tn = np.sum(~pred_pos & ~true_pos)
    fp = np.sum(pred_pos & ~true_pos)
    fn = np.sum(~pred_pos & true_pos)

    sensitivity = tp / (tp + fn) if tp + fn else np.nan
    specificity = tn / (tn + fp) if tn + fp else np.nan
    return sensitivity, specificity


def build_summary(alpha: float) -> tuple[list[str], dict[float, list[tuple[float, float]]]]:
    datasets = collect_dataset()
    subject_metrics: dict[float, list[tuple[float, float]]] = {thr: [] for thr in THRESHOLDS}
    subjects: list[str] = []

    for subject_id in SUBJECT_IDS:
        labels, preds = _loocv_predict(datasets, subject_id, alpha)
        subjects.append(subject_id)
        for thr in THRESHOLDS:
            sens, spec = _compute_metrics(labels, preds, thr)
            subject_metrics[thr].append((sens, spec))
    return subjects, subject_metrics


def aggregate_metrics(subject_metrics: dict[float, list[tuple[float, float]]]) -> dict[float, tuple[float, float]]:
    overall: dict[float, tuple[float, float]] = {}
    for thr, metrics in subject_metrics.items():
        sensitivities = [m[0] for m in metrics]
        specificities = [m[1] for m in metrics]
        overall[thr] = (float(np.nanmean(sensitivities)), float(np.nanmean(specificities)))
    return overall


def plot_table(
    subjects: list[str],
    subject_metrics: dict[float, list[tuple[float, float]]],
    overall: dict[float, tuple[float, float]],
    output_path: Path,
) -> Path:
    rows = subjects + ["Overall"]
    cols = []
    data = []

    for thr in THRESHOLDS:
        cols.extend([f"<{int(thr)}%\nSensitivity", f"<{int(thr)}%\nSpecificity"])

    for sid in subjects:
        row_values = []
        for thr in THRESHOLDS:
            sens, spec = subject_metrics[thr][subjects.index(sid)]
            row_values.extend([sens * 100, spec * 100])
        data.append(row_values)

    overall_row = []
    for thr in THRESHOLDS:
        sens, spec = overall[thr]
        overall_row.extend([sens * 100, spec * 100])
    data.append(overall_row)

    data_array = np.array(data)
    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(data_array, cmap="YlGn", vmin=0, vmax=100)

    ax.set_xticks(np.arange(len(cols)))
    ax.set_xticklabels(cols, fontsize=9)
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels(rows, fontsize=10)

    for i in range(data_array.shape[0]):
        for j in range(data_array.shape[1]):
            ax.text(
                j,
                i,
                f"{data_array[i, j]:.1f}",
                ha="center",
                va="center",
                color="black",
                fontsize=9,
            )

    ax.set_title("Leave-One-Subject-Out Classification Performance", fontsize=14, pad=14)
    ax.set_xlabel("Threshold and Metric", fontsize=11)
    ax.set_ylabel("Subject", fontsize=11)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Percentage (%)", rotation=90)

    ax.text(
        0.99,
        -0.12,
        "Positive class: SpO₂ below threshold",
        transform=ax.transAxes,
        fontsize=9,
        ha="right",
        va="top",
        color="0.4",
    )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize LOSO classification performance for Ridge regression")
    parser.add_argument(
        "--alpha",
        type=float,
        default=25.0,
        help="Ridge regression alpha (default: 25.0)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("figures/classification_per_subject.png"),
        help="Output figure path (default: figures/classification_per_subject.png)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    subjects, subject_metrics = build_summary(args.alpha)
    overall = aggregate_metrics(subject_metrics)
    result_path = plot_table(subjects, subject_metrics, overall, args.output)
    print(f"Saved classification summary to {result_path}")


if __name__ == "__main__":
    main()
