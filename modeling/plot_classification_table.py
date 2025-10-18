from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.table import Table
from sklearn.linear_model import Ridge

from train_linear_regression import (
    SUBJECT_IDS,
    _fit_standardizer,
    _standardize,
    collect_dataset,
)

THRESHOLDS = [92.0, 90.0, 88.0]
ALPHA = 25.0
HEADER_COLOR = "#8A1C61"
HEADER_TEXT_COLOR = "white"
BODY_TEXT_COLOR = "black"


def _loocv_predict(datasets, subject_id: str) -> tuple[np.ndarray, np.ndarray]:
    held_out = next(ds for ds in datasets if ds.subject_id == subject_id)
    train_features = np.vstack(
        [ds.features for ds in datasets if ds.subject_id != subject_id]
    )
    train_labels = np.concatenate(
        [ds.labels for ds in datasets if ds.subject_id != subject_id]
    )
    mean, std = _fit_standardizer(train_features)
    model = Ridge(alpha=ALPHA)
    model.fit(_standardize(train_features, mean, std), train_labels)
    preds = model.predict(_standardize(held_out.features, mean, std))
    return held_out.labels, preds


def _sens_spec(labels: np.ndarray, preds: np.ndarray, threshold: float) -> tuple[float, float]:
    truth = labels < threshold
    pred = preds < threshold
    tp = np.sum(truth & pred)
    tn = np.sum(~truth & ~pred)
    fp = np.sum(~truth & pred)
    fn = np.sum(truth & ~pred)
    sens = tp / (tp + fn) if tp + fn else np.nan
    spec = tn / (tn + fp) if tn + fp else np.nan
    return sens, spec


def compute_metrics() -> tuple[list[str], dict[float, list[tuple[float, float]]]]:
    datasets = collect_dataset()
    metrics: dict[float, list[tuple[float, float]]] = {thr: [] for thr in THRESHOLDS}
    subject_ids: list[str] = []
    for sid in SUBJECT_IDS:
        labels, preds = _loocv_predict(datasets, sid)
        subject_ids.append(sid)
        for thr in THRESHOLDS:
            metrics[thr].append(_sens_spec(labels, preds, thr))
    return subject_ids, metrics


def add_overall(metrics: dict[float, list[tuple[float, float]]]) -> dict[float, list[tuple[float, float]]]:
    extended = {thr: values.copy() for thr, values in metrics.items()}
    for thr, values in metrics.items():
        sens = np.nanmean([v[0] for v in values])
        spec = np.nanmean([v[1] for v in values])
        extended[thr].append((sens, spec))
    return extended


def render_table(subjects: list[str], metrics: dict[float, list[tuple[float, float]]], output: Path) -> Path:
    rows = subjects + ["Overall"]
    cols = ["Subject"] + [f"<{int(thr)}%" for thr in THRESHOLDS]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.axis("off")
    table = Table(ax, bbox=[0, 0, 1, 1])

    n_rows = len(rows) + 1  # header + data
    n_cols = len(cols)
    width = 1.0 / n_cols
    height = 1.0 / n_rows

    for col, label in enumerate(cols):
        cell = table.add_cell(0, col, width, height, text=label, loc="center", facecolor=HEADER_COLOR)
        cell.get_text().set_color(HEADER_TEXT_COLOR)
        cell.set_edgecolor("black")

    for row_idx, row_label in enumerate(rows, start=1):
        cell = table.add_cell(row_idx, 0, width, height, text=row_label, loc="center", facecolor="white")
        cell.get_text().set_color(BODY_TEXT_COLOR)
        cell.set_edgecolor("black")

    for col_idx, thr in enumerate(THRESHOLDS, start=1):
        values = metrics[thr]
        for row_idx, (sens, spec) in enumerate(values, start=1):
            text = f"{sens:.2f} / {spec:.2f}"
            cell = table.add_cell(row_idx, col_idx, width, height, text=text, loc="center", facecolor="white")
            cell.get_text().set_color(BODY_TEXT_COLOR)
            cell.set_edgecolor("black")

    ax.add_table(table)
    ax.set_title("Classification: Per-Subject Sensitivity and Specificity", fontsize=14, pad=12, color=HEADER_COLOR)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return output


def main() -> None:
    subjects, metrics = compute_metrics()
    metrics_with_overall = add_overall(metrics)
    render_table(subjects, metrics_with_overall, Path("figures/classification_table.png"))
    print("Saved classification table to figures/classification_table.png")


if __name__ == "__main__":
    main()
