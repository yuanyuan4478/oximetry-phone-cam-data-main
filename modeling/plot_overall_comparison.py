from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LinearRegression, Ridge

from dataset_utils import build_classification_datasets
from train_linear_regression import _fit_standardizer, _standardize, collect_dataset

MODELS = (
    ("LinearRegression", lambda: LinearRegression()),
    ("RidgeRegression", lambda: Ridge(alpha=25.0)),
    ("RandomForest", lambda: RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42)),
    (
        "HistGradientBoosting",
        lambda: HistGradientBoostingClassifier(learning_rate=0.1, max_depth=6, class_weight="balanced", random_state=42),
    ),
)


def evaluate_regressor(factory, threshold: float) -> tuple[float, float]:
    datasets = collect_dataset()
    total_correct = 0
    total_samples = 0
    total_abs_error = 0.0

    for held_out in datasets:
        train_features = np.vstack([ds.features for ds in datasets if ds.subject_id != held_out.subject_id])
        train_labels = np.concatenate([ds.labels for ds in datasets if ds.subject_id != held_out.subject_id])

        mean, std = _fit_standardizer(train_features)
        train_scaled = _standardize(train_features, mean, std)

        model = factory()
        model.fit(train_scaled, train_labels)

        test_scaled = _standardize(held_out.features, mean, std)
        preds = model.predict(test_scaled)

        total_correct += np.sum((preds >= threshold) == (held_out.labels >= threshold))
        total_samples += held_out.labels.size
        total_abs_error += np.abs(preds - held_out.labels).sum()

    accuracy = total_correct / total_samples
    mae = total_abs_error / total_samples
    return accuracy, mae


def evaluate_classifier(factory, threshold: float) -> float:
    datasets = build_classification_datasets(threshold)
    total_correct = 0
    total_samples = 0

    for held_out in datasets:
        train_features = np.vstack([ds.features for ds in datasets if ds.subject_id != held_out.subject_id])
        train_labels = np.concatenate([ds.labels for ds in datasets if ds.subject_id != held_out.subject_id])

        model = factory()
        model.fit(train_features, train_labels)
        preds = model.predict(held_out.features)

        total_correct += np.sum(preds == held_out.labels)
        total_samples += held_out.labels.size

    return total_correct / total_samples


def build_metrics(threshold: float) -> tuple[list[str], list[float], dict[str, float]]:
    labels: list[str] = []
    accuracies: list[float] = []
    maes: dict[str, float] = {}

    for name, factory in MODELS:
        if name in {"LinearRegression", "RidgeRegression"}:
            acc, mae = evaluate_regressor(factory, threshold)
            accuracies.append(acc)
            maes[name] = mae
        else:
            acc = evaluate_classifier(factory, threshold)
            accuracies.append(acc)
        labels.append(name)

    return labels, accuracies, maes


COLORS = {
    "LinearRegression": "#1f77b4",
    "RidgeRegression": "#ff7f0e",
    "RandomForest": "#2ca02c",
    "HistGradientBoosting": "#d62728",
}


def plot_overall(labels: list[str], accuracies: list[float], maes: dict[str, float], threshold: float, output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)

    x = np.arange(len(labels))
    percentages = np.array(accuracies) * 100.0
    colors = [COLORS[label] for label in labels]

    plt.figure(figsize=(8, 4.5))
    bars = plt.bar(x, percentages, color=colors)
    plt.xticks(x, labels, rotation=15)
    plt.ylabel("Accuracy (%)")
    plt.title("Leave-One-Subject-Out Hypoxemia Accuracy")
    plt.ylim(0, 100)

    for bar, pct in zip(bars, percentages):
        plt.text(bar.get_x() + bar.get_width() / 2, pct + 1.0, f"{pct:.1f}%", ha="center", va="bottom", fontsize=9)

    if maes:
        mae_lines = [f"{name}: {maes[name]:.2f} %" for name in sorted(maes.keys())]
        plt.text(
            0.02,
            0.02,
            "MAE (SpO? %):\n" + "\n".join(mae_lines),
            transform=plt.gca().transAxes,
            fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
            va="bottom",
        )

    plt.text(
        0.99,
        0.02,
        f"Threshold: {threshold:.0f}%",
        ha="right",
        va="bottom",
        transform=plt.gca().transAxes,
        fontsize=8,
        alpha=0.7,
    )

    plt.tight_layout()
    plt.savefig(output, dpi=150)
    plt.close()
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Create an overall model comparison figure.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=90.0,
        help="SpO? threshold for hypoxemia classification (default: 90).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("figures/overall_comparison.png"),
        help="Output image path (default: figures/overall_comparison.png).",
    )
    args = parser.parse_args()

    labels, accuracies, maes = build_metrics(args.threshold)
    result_path = plot_overall(labels, accuracies, maes, args.threshold, args.output)
    print(f"Saved overall comparison to {result_path}")


if __name__ == "__main__":
    main()
