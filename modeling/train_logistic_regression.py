"""Hypoxemia classification baseline using logistic regression."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from joblib import dump
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)

from train_linear_regression import (
    FEATURE_NAMES,
    collect_dataset,
)


@dataclass
class ClassificationDataset:
    subject_id: str
    features: np.ndarray
    labels: np.ndarray  # binary (1 = hypoxemia)
    continuous: np.ndarray  # original SpO2 values


def _fit_standardizer(features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return mean, std


def _standardize(features: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (features - mean) / std


def _rescale_logistic_coefficients(
    model: LogisticRegression,
    mean: np.ndarray,
    std: np.ndarray,
) -> Tuple[float, np.ndarray]:
    std_safe = np.where(std == 0, 1.0, std)
    coef_scaled = model.coef_.reshape(-1)
    intercept_scaled = float(model.intercept_.reshape(-1)[0])
    coef_original = coef_scaled / std_safe
    intercept_original = intercept_scaled - float(np.dot(mean / std_safe, coef_scaled))
    return intercept_original, coef_original


def build_classification_datasets(threshold: float) -> List[ClassificationDataset]:
    base_datasets = collect_dataset()
    classification_datasets = []
    for base in base_datasets:
        labels_binary = (base.labels < threshold).astype(np.int32)
        classification_datasets.append(
            ClassificationDataset(base.subject_id, base.features, labels_binary, base.labels)
        )
    return classification_datasets


def leave_one_subject_out_cv(
    datasets: Sequence[ClassificationDataset],
    model_factory,
) -> Tuple[List[Dict], np.ndarray]:
    results: List[Dict] = []
    confusion = np.zeros((2, 2), dtype=np.int64)

    for held_out in datasets:
        train_features = np.vstack(
            [d.features for d in datasets if d.subject_id != held_out.subject_id]
        )
        train_labels = np.concatenate(
            [d.labels for d in datasets if d.subject_id != held_out.subject_id]
        )

        mean, std = _fit_standardizer(train_features)
        train_scaled = _standardize(train_features, mean, std)
        test_scaled = _standardize(held_out.features, mean, std)

        model = model_factory()
        model.fit(train_scaled, train_labels)

        preds = model.predict(test_scaled)
        probs = model.predict_proba(test_scaled)[:, 1]

        acc = accuracy_score(held_out.labels, preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            held_out.labels,
            preds,
            average="binary",
            zero_division=0,
        )
        try:
            roc_auc = roc_auc_score(held_out.labels, probs)
        except ValueError:
            roc_auc = float("nan")

        cm = confusion_matrix(held_out.labels, preds, labels=[0, 1])
        confusion += cm

        results.append(
            {
                "subject": held_out.subject_id,
                "samples": len(held_out.labels),
                "accuracy": acc,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "roc_auc": roc_auc,
            }
        )

    return results, confusion


def weighted_scores(cv_results: Sequence[Dict]) -> Dict[str, float]:
    total = sum(res["samples"] for res in cv_results)
    weighted = {}
    for key in ["accuracy", "precision", "recall", "f1"]:
        weighted[key] = sum(res[key] * res["samples"] for res in cv_results) / total
    auc_values = [res["roc_auc"] for res in cv_results if not np.isnan(res["roc_auc"])]
    weighted["roc_auc"] = float(np.mean(auc_values)) if auc_values else float("nan")
    return weighted


def train_full_model(
    datasets: Sequence[ClassificationDataset],
    model_factory,
) -> Tuple[LogisticRegression, np.ndarray, np.ndarray]:
    features = np.vstack([d.features for d in datasets])
    labels = np.concatenate([d.labels for d in datasets])
    mean, std = _fit_standardizer(features)
    features_scaled = _standardize(features, mean, std)
    model = model_factory()
    model.fit(features_scaled, labels)
    return model, mean, std


def save_artifacts(
    output_dir: Path,
    model: LogisticRegression,
    mean: np.ndarray,
    std: np.ndarray,
    metadata: Dict,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    dump(model, output_dir / "logistic_model.joblib")
    np.savez(output_dir / "logistic_scaler.npz", mean=mean, std=std)
    (output_dir / "logistic_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved logistic regression artifacts to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--threshold",
        type=float,
        default=90.0,
        help="SpO2 threshold defining the positive class (default: 90)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Optional directory to save trained model, scaler, and metadata",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=1000,
        help="Maximum iterations for logistic regression solver (default: 1000)",
    )
    args = parser.parse_args()

    datasets = build_classification_datasets(args.threshold)

    def factory() -> LogisticRegression:
        return LogisticRegression(
            penalty="l2",
            C=1.0,
            solver="lbfgs",
            max_iter=args.max_iter,
            class_weight="balanced",
        )

    cv_results, confusion = leave_one_subject_out_cv(datasets, factory)
    weighted = weighted_scores(cv_results)

    print(f"Logistic regression hypoxemia classification @ {args.threshold:.1f}%")
    print("Per-subject results (leave-one-subject-out):")
    for res in cv_results:
        print(
            f"  Subject {res['subject']}: accuracy={res['accuracy']*100:.2f}% precision={res['precision']*100:.2f}% "
            f"recall={res['recall']*100:.2f}% f1={res['f1']*100:.2f}% roc_auc={res['roc_auc']:.3f}"
        )

    print("\nWeighted scores (by samples):")
    print(
        "  Accuracy={accuracy:.2f}% Precision={precision:.2f}% Recall={recall:.2f}% F1={f1:.2f}% ROC-AUC={roc_auc:.3f}".format(
            accuracy=weighted["accuracy"] * 100,
            precision=weighted["precision"] * 100,
            recall=weighted["recall"] * 100,
            f1=weighted["f1"] * 100,
            roc_auc=weighted["roc_auc"],
        )
    )

    tn, fp, fn, tp = confusion.ravel()
    print("\nAggregated confusion matrix (pred columns, GT rows):")
    print(confusion)
    print(f"  TN={tn}, FP={fp}, FN={fn}, TP={tp}")

    model, mean, std = train_full_model(datasets, factory)
    intercept, coefficients = _rescale_logistic_coefficients(model, mean, std)
    print("\nModel coefficients (log-odds per unit feature):")
    for feature, coef in zip(FEATURE_NAMES, coefficients):
        print(f"  {feature}: {coef:.4f}")
    print(f"Intercept: {intercept:.4f}")

    if args.output_dir:
        output_dir = args.output_dir if args.output_dir.is_absolute() else Path.cwd() / args.output_dir
        metadata = {
            "threshold": args.threshold,
            "weighted_scores": weighted,
            "confusion_matrix": confusion.tolist(),
            "intercept": intercept,
            "coefficients": coefficients.tolist(),
        }
        save_artifacts(output_dir, model, mean, std, metadata)


if __name__ == "__main__":
    main()
