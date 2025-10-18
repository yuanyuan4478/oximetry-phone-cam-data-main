from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Ridge

from train_linear_regression import _fit_standardizer, _standardize, collect_dataset

SUBJECT_ID = "100001"
THRESHOLD = 90.0
ALPHA = 25.0
OUTPUT = Path("figures/ridge_prediction_correctness.png")


def main() -> None:
    datasets = collect_dataset()
    held_out = next(ds for ds in datasets if ds.subject_id == SUBJECT_ID)
    train_features = np.vstack([ds.features for ds in datasets if ds.subject_id != SUBJECT_ID])
    train_labels = np.concatenate([ds.labels for ds in datasets if ds.subject_id != SUBJECT_ID])

    mean, std = _fit_standardizer(train_features)
    model = Ridge(alpha=ALPHA)
    model.fit(_standardize(train_features, mean, std), train_labels)

    preds = model.predict(_standardize(held_out.features, mean, std))
    labels = held_out.labels
    time_axis = np.arange(labels.shape[0])

    truth = labels >= THRESHOLD
    predicted = preds >= THRESHOLD
    correct_mask = truth == predicted

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(time_axis, labels, label="Ground truth", linewidth=1.4, color="black")
    ax.plot(time_axis, preds, label="Ridge prediction", linewidth=1.4, color="#ff7f0e")
    ax.axhline(THRESHOLD, color="#1f77b4", linestyle="--", linewidth=1.0, label=f"Threshold {THRESHOLD:.0f}%")

    ax.scatter(time_axis[correct_mask], preds[correct_mask], c="#2ca02c", s=16, label="Correct")
    ax.scatter(time_axis[~correct_mask], preds[~correct_mask], c="#d62728", s=20, label="Incorrect")

    ax.set_xlabel("Seconds")
    ax.set_ylabel("SpO₂ (%)")
    ax.set_title("Subject 100001: Ridge Regression vs Ground Truth")
    ax.set_ylim(60, 100)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTPUT, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
