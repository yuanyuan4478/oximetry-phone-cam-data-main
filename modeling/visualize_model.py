"""Visualize regression and tree-based classifiers for hypoxemia detection using leave-one-subject-out predictions."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import Ridge

from dataset_utils import build_classification_datasets, fit_standardizer as fit_standardizer_cls, standardize as standardize_cls
from train_linear_regression import (
    FEATURE_NAMES,
    SubjectDataset,
    _fit_standardizer as fit_standardizer_reg,
    _standardize as standardize_reg,
    collect_dataset,
    tune_ridge_alpha,
    _rescale_coefficients,
)

NEG_COLOR = "#d62728"
RIDGE_COLOR = "#1f77b4"
CLASSIFIER_STYLES = {
    "RandomForest": {"color": "#9467bd", "marker": "s"},
    "HistGradientBoosting": {"color": "#8c564b", "marker": "^"},
}


def _ridge_loso_predictions(
    datasets: Sequence[SubjectDataset],
    alpha: float,
    threshold: float,
) -> Tuple[Dict[str, Dict[str, np.ndarray]], np.ndarray, np.ndarray]:
    per_subject: Dict[str, Dict[str, np.ndarray]] = {}
    all_labels: list[np.ndarray] = []
    all_preds: list[np.ndarray] = []

    for held_out in datasets:
        train_features = np.vstack(
            [d.features for d in datasets if d.subject_id != held_out.subject_id]
        )
        train_labels = np.concatenate(
            [d.labels for d in datasets if d.subject_id != held_out.subject_id]
        )

        mean, std = fit_standardizer_reg(train_features)
        train_scaled = standardize_reg(train_features, mean, std)
        model = Ridge(alpha=alpha)
        model.fit(train_scaled, train_labels)

        test_scaled = standardize_reg(held_out.features, mean, std)
        preds = model.predict(test_scaled)
        accuracy = float(
            ((preds >= threshold) == (held_out.labels >= threshold)).mean()
        )

        per_subject[held_out.subject_id] = {
            "labels": held_out.labels,
            "preds": preds,
            "accuracy": accuracy,
        }
        all_labels.append(held_out.labels)
        all_preds.append(preds)

    return per_subject, np.concatenate(all_labels), np.concatenate(all_preds)


def _classifier_loso_predictions(
    datasets,
    factory,
    scale: bool,
) -> Tuple[Dict[str, Dict[str, np.ndarray]], np.ndarray]:
    per_subject: Dict[str, Dict[str, np.ndarray]] = {}
    all_probs: list[np.ndarray] = []

    for held_out in datasets:
        train_features = np.vstack(
            [d.features for d in datasets if d.subject_id != held_out.subject_id]
        )
        train_labels = np.concatenate(
            [d.labels for d in datasets if d.subject_id != held_out.subject_id]
        )

        if scale:
            mean, std = fit_standardizer_cls(train_features)
            train_scaled = standardize_cls(train_features, mean, std)
            test_scaled = standardize_cls(held_out.features, mean, std)
        else:
            mean = None
            std = None
            train_scaled = train_features
            test_scaled = held_out.features

        model = factory()
        model.fit(train_scaled, train_labels)
        probs = model.predict_proba(test_scaled)[:, 1]
        accuracy = float(((probs >= 0.5) == held_out.labels).mean())

        per_subject[held_out.subject_id] = {
            "probs": probs,
            "accuracy": accuracy,
        }
        all_probs.append(probs)

    return per_subject, np.concatenate(all_probs)


def _plot_subject_timeseries(
    subject_id: str,
    labels: np.ndarray,
    ridge_preds: np.ndarray,
    classifier_probs: Dict[str, np.ndarray],
    threshold: float,
    output_dir: Path,
) -> Tuple[Path, Dict[str, float]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    time_axis = np.arange(labels.shape[0])

    ridge_binary = ridge_preds >= threshold
    ridge_correct = ridge_binary == (labels >= threshold)
    ridge_acc = float(ridge_correct.mean()) if ridge_binary.size else float("nan")

    fig, ax1 = plt.subplots(figsize=(11, 5))
    ax1.plot(time_axis, labels, label="Ground truth", linewidth=1.5, color="black")
    ax1.plot(time_axis, ridge_preds, label="Ridge regression", linewidth=1.2, color=RIDGE_COLOR)
    ax1.axhline(
        threshold,
        color="black",
        linestyle="--",
        linewidth=1,
        label=f"Threshold {threshold:.1f}%",
    )

    wrong_idx = np.where(~ridge_correct)[0]
    if wrong_idx.size:
        ax1.scatter(
            time_axis[wrong_idx],
            ridge_preds[wrong_idx],
            color=NEG_COLOR,
            marker="x",
            s=40,
            label="Ridge misclass",
        )

    ax2 = ax1.twinx()
    ax2.set_ylabel("P(SpO₂ < threshold)")
    ax2.set_ylim(0, 1)

    stats = {"ridge_accuracy": ridge_acc}
    gt_low = labels < threshold

    for model_name, probs in classifier_probs.items():
        style = CLASSIFIER_STYLES[model_name]
        ax2.plot(
            time_axis,
            probs,
            label=f"{model_name} P(low)",
            linewidth=1.1,
            color=style["color"],
            alpha=0.9,
        )

        decisions = probs >= 0.5
        correct = decisions == gt_low
        acc = float(correct.mean()) if decisions.size else float("nan")
        stats[f"{model_name}_accuracy"] = acc

        ax1.scatter(
            time_axis[decisions],
            np.full(np.count_nonzero(decisions), threshold),
            color=style["color"],
            marker=style["marker"],
            s=30,
            alpha=0.8,
            label=f"{model_name} decision",
        )
        ax1.scatter(
            time_axis[~decisions],
            np.full(np.count_nonzero(~decisions), threshold),
            facecolors="none",
            edgecolors=style["color"],
            marker=style["marker"],
            s=30,
            alpha=0.8,
        )

        wrong_decisions = np.where(~correct)[0]
        if wrong_decisions.size:
            ax1.scatter(
                time_axis[wrong_decisions],
                np.full(wrong_decisions.size, threshold),
                color=NEG_COLOR,
                marker=style["marker"],
                s=34,
                alpha=0.85,
            )

    ax1.set_xlabel("Seconds")
    ax1.set_ylabel("SpO₂ (%)")
    ax1.set_title(f"Subject {subject_id}: Ridge vs tree classifiers")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    combined_lines = lines1 + [line for line in lines2 if line not in lines1]
    combined_labels = labels1 + [label for label in labels2 if label not in labels1]
    ax1.legend(combined_lines, combined_labels, loc="upper right", fontsize=8)
    fig.tight_layout()

    text_lines = [f"Ridge acc ≥{threshold:.1f}%: {ridge_acc*100:.1f}%"]
    for model_name in classifier_probs:
        text_lines.append(
            f"{model_name} acc <{threshold:.1f}%: {stats[f'{model_name}_accuracy']*100:.1f}%"
        )
    ax1.text(
        0.01,
        0.05,
        "\n".join(text_lines),
        transform=ax1.transAxes,
        fontsize=8,
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
    )

    path = output_dir / f"subject_{subject_id}_comparison.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path, stats


def _plot_scatter_comparison(
    labels: np.ndarray,
    ridge_preds: np.ndarray,
    rf_probs: np.ndarray,
    threshold: float,
    output_dir: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(5.8, 5.8))
    scatter = plt.scatter(
        ridge_preds,
        rf_probs,
        c=labels >= threshold,
        cmap=plt.get_cmap("coolwarm"),
        s=8,
        alpha=0.6,
    )
    plt.axvline(threshold, color=RIDGE_COLOR, linestyle="--", linewidth=1, label="Ridge threshold")
    plt.axhline(0.5, color="#888", linestyle=":", linewidth=1, label="RF decision")
    plt.xlabel("Ridge predicted SpO₂ (%)")
    plt.ylabel("RandomForest P(low SpO₂)")
    plt.title("Ridge vs Random Forest")
    plt.legend(loc="best", fontsize=9)
    cbar = plt.colorbar(scatter, label="Ground truth ≥ threshold")
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(["Below", "Above"])
    plt.tight_layout()
    plt.text(0.99, 0.01, 'Models: Ridge | RandomForest | HistGradientBoosting', ha='right', va='bottom', transform=plt.gca().transAxes, fontsize=8, alpha=0.7)
    path = output_dir / "ridge_vs_randomforest_scatter.png"
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def _plot_accuracy_bar(
    averages: Dict[str, float],
    output_dir: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = list(averages.keys())
    values = [averages[label] * 100 for label in labels]
    colors = [RIDGE_COLOR] + [CLASSIFIER_STYLES[label]["color"] for label in labels[1:]]
    plt.figure(figsize=(5.5, 3.2))
    plt.bar(labels, values, color=colors)
    plt.ylabel("Accuracy (%)")
    plt.ylim(0, 100)
    plt.title("Average per-subject classification accuracy")
    plt.tight_layout()
    plt.text(0.99, 0.01, 'Models: Ridge | RandomForest | HistGradientBoosting', ha='right', va='bottom', transform=plt.gca().transAxes, fontsize=8, alpha=0.7)
    path = output_dir / "average_accuracy.png"
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/plots"),
        help="Directory to store generated plots (default: outputs/plots)",
    )
    parser.add_argument(
        "--alphas",
        nargs="*",
        type=float,
        default=[0.01, 0.1, 1.0, 5.0, 10.0, 25.0],
        help="Candidate ridge alphas for tuning (default: 0.01 0.1 1 5 10 25)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=90.0,
        help="SpO₂ threshold for classification visuals (default: 90)",
    )
    parser.add_argument(
        "--rf-estimators",
        type=int,
        default=300,
        help="Number of trees for random forest (default: 300)",
    )
    parser.add_argument(
        "--gb-learning-rate",
        type=float,
        default=0.1,
        help="Learning rate for hist gradient boosting (default: 0.1)",
    )
    args = parser.parse_args()

    ridge_datasets = collect_dataset()
    alphas = args.alphas if isinstance(args.alphas, Sequence) else [args.alphas]
    best_alpha, _ = tune_ridge_alpha(ridge_datasets, alphas)
    ridge_loso_map, labels_concat, ridge_concat = _ridge_loso_predictions(
        ridge_datasets, best_alpha, args.threshold
    )

    classification_datasets = build_classification_datasets(args.threshold)

    rf_factory = lambda: RandomForestClassifier(
        n_estimators=args.rf_estimators,
        max_depth=None,
        class_weight="balanced",
        random_state=42,
    )
    rf_loso_map, rf_concat = _classifier_loso_predictions(
        classification_datasets, rf_factory, False
    )

    gb_factory = lambda: HistGradientBoostingClassifier(
        learning_rate=args.gb_learning_rate,
        max_depth=6,
        class_weight="balanced",
        random_state=42,
    )
    gb_loso_map, _ = _classifier_loso_predictions(
        classification_datasets, gb_factory, False
    )

    subject_plots = []
    accuracy_tracker: Dict[str, Dict[str, float]] = {}

    for ds in ridge_datasets:
        sid = ds.subject_id
        ridge_info = ridge_loso_map[sid]
        classifier_probs = {
            "RandomForest": rf_loso_map[sid]["probs"],
            "HistGradientBoosting": gb_loso_map[sid]["probs"],
        }

        path, stats = _plot_subject_timeseries(
            sid,
            ridge_info["labels"],
            ridge_info["preds"],
            classifier_probs,
            args.threshold,
            args.output_dir,
        )
        subject_plots.append(path)
        accuracy_tracker[sid] = stats

    scatter_path = _plot_scatter_comparison(
        labels_concat,
        ridge_concat,
        rf_concat,
        args.threshold,
        args.output_dir,
    )

    averages = {
        "Ridge": float(np.mean([stats["ridge_accuracy"] for stats in accuracy_tracker.values()])),
        "RandomForest": float(np.mean([stats["RandomForest_accuracy"] for stats in accuracy_tracker.values()])),
        "HistGradientBoosting": float(
            np.mean([stats["HistGradientBoosting_accuracy"] for stats in accuracy_tracker.values()])
        ),
    }
    bar_path = _plot_accuracy_bar(averages, args.output_dir)

    ridge_features_full = np.vstack([ds.features for ds in ridge_datasets])
    ridge_labels_full = np.concatenate([ds.labels for ds in ridge_datasets])
    mean_reg_full, std_reg_full = fit_standardizer_reg(ridge_features_full)
    ridge_full = Ridge(alpha=best_alpha)
    ridge_full.fit(standardize_reg(ridge_features_full, mean_reg_full, std_reg_full), ridge_labels_full)

    intercept_ridge, coeffs_ridge = _rescale_coefficients(ridge_full, mean_reg_full, std_reg_full)

    print(f"Best ridge alpha: {best_alpha}")
    print("Generated plots:")
    cwd = Path.cwd()
    for path in subject_plots + [scatter_path, bar_path]:
        resolved = path.resolve()
        try:
            display_path = resolved.relative_to(cwd)
        except ValueError:
            display_path = resolved
        print(f"  {display_path}")

    print("\nAverage per-subject accuracy:")
    for sid in accuracy_tracker:
        stats = accuracy_tracker[sid]
        print(
            f"  Subject {sid}: ridge={stats['ridge_accuracy']*100:.2f}% "
            f"random_forest={stats['RandomForest_accuracy']*100:.2f}% "
            f"hist_gb={stats['HistGradientBoosting_accuracy']*100:.2f}%"
        )

    print("\nRidge coefficients (per feature):")
    for feature, coef_ridge in zip(FEATURE_NAMES, coeffs_ridge):
        print(f"  {feature}: ridge={coef_ridge:.4f}")
    print(f"Ridge intercept: {intercept_ridge:.4f}")


if __name__ == "__main__":
    main()
