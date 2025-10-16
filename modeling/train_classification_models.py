"""Evaluate simple classifiers for hypoxemia detection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
from joblib import dump
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)

from dataset_utils import ClassificationDataset, build_classification_datasets, fit_standardizer, standardize


ModelFactory = Callable[[], object]


def _evaluate_model(
    name: str,
    datasets: Sequence[ClassificationDataset],
    model_factory: ModelFactory,
    scale: bool,
) -> Tuple[List[Dict], np.ndarray, object, np.ndarray, np.ndarray]:
    results: List[Dict] = []
    confusion = np.zeros((2, 2), dtype=np.int64)

    model = model_factory()

    train_features = np.vstack([d.features for d in datasets])
    train_labels = np.concatenate([d.labels for d in datasets])

    if scale:
        mean, std = fit_standardizer(train_features)
        train_scaled = standardize(train_features, mean, std)
    else:
        mean = np.zeros(train_features.shape[1])
        std = np.ones(train_features.shape[1])
        train_scaled = train_features

    model.fit(train_scaled, train_labels)

    for held_out in datasets:
        train_subset = np.vstack(
            [d.features for d in datasets if d.subject_id != held_out.subject_id]
        )
        label_subset = np.concatenate(
            [d.labels for d in datasets if d.subject_id != held_out.subject_id]
        )

        if scale:
            mean, std = fit_standardizer(train_subset)
            train_scaled_cv = standardize(train_subset, mean, std)
            test_scaled = standardize(held_out.features, mean, std)
        else:
            mean = np.zeros(train_subset.shape[1])
            std = np.ones(train_subset.shape[1])
            train_scaled_cv = train_subset
            test_scaled = held_out.features

        cv_model = model_factory()
        cv_model.fit(train_scaled_cv, label_subset)

        preds = cv_model.predict(test_scaled)
        probs = cv_model.predict_proba(test_scaled)[:, 1]

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

    return results, confusion, model, mean, std


def _weighted_scores(cv_results: Sequence[Dict]) -> Dict[str, float]:
    total = sum(res["samples"] for res in cv_results)
    weighted = {}
    for key in ["accuracy", "precision", "recall", "f1"]:
        weighted[key] = sum(res[key] * res["samples"] for res in cv_results) / total
    auc_values = [res["roc_auc"] for res in cv_results if not np.isnan(res["roc_auc"])]
    weighted["roc_auc"] = float(np.mean(auc_values)) if auc_values else float("nan")
    return weighted


def _print_summary(name: str, cv_results: Sequence[Dict], confusion: np.ndarray) -> None:
    print(f"\n{name} (leave-one-subject-out):")
    for res in cv_results:
        print(
            f"  Subject {res['subject']}: accuracy={res['accuracy']*100:.2f}% precision={res['precision']*100:.2f}% "
            f"recall={res['recall']*100:.2f}% f1={res['f1']*100:.2f}% roc_auc={res['roc_auc']:.3f}"
        )
    weighted = _weighted_scores(cv_results)
    print(
        "  Weighted: accuracy={accuracy:.2f}% precision={precision:.2f}% recall={recall:.2f}% f1={f1:.2f}% roc_auc={roc_auc:.3f}".format(
            accuracy=weighted["accuracy"] * 100,
            precision=weighted["precision"] * 100,
            recall=weighted["recall"] * 100,
            f1=weighted["f1"] * 100,
            roc_auc=weighted["roc_auc"],
        )
    )
    tn, fp, fn, tp = confusion.ravel()
    print("  Confusion matrix:\n", confusion)
    print(f"  TN={tn}, FP={fp}, FN={fn}, TP={tp}")


def save_artifacts(
    name: str,
    output_dir: Path,
    model,
    mean: np.ndarray,
    std: np.ndarray,
    metadata: Dict,
) -> None:
    model_dir = output_dir / name.lower().replace(" ", "_")
    model_dir.mkdir(parents=True, exist_ok=True)
    dump(model, model_dir / "model.joblib")
    np.savez(model_dir / "scaler.npz", mean=mean, std=std)
    (model_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved {name} artifacts to {model_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--threshold",
        type=float,
        default=90.0,
        help="SpO₂ threshold defining the positive class (default: 90)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Optional directory for saving model artifacts",
    )
    parser.add_argument(
        "--rf-estimators",
        type=int,
        default=300,
        help="Number of trees for RandomForestClassifier (default: 300)",
    )
    parser.add_argument(
        "--gb-learning-rate",
        type=float,
        default=0.1,
        help="Learning rate for HistGradientBoostingClassifier (default: 0.1)",
    )
    args = parser.parse_args()

    datasets = build_classification_datasets(args.threshold)

    model_specs = [
        (
            "Random Forest",
            lambda: RandomForestClassifier(
                n_estimators=args.rf_estimators,
                max_depth=None,
                class_weight="balanced",
                random_state=42,
            ),
            False,
        ),
        (
            "Hist Gradient Boosting",
            lambda: HistGradientBoostingClassifier(
                learning_rate=args.gb_learning_rate,
                max_depth=6,
                class_weight="balanced",
                random_state=42,
            ),
            False,
        ),
    ]

    for name, factory, scale in model_specs:
        cv_results, confusion, model, mean, std = _evaluate_model(name, datasets, factory, scale)
        _print_summary(name, cv_results, confusion)

        if args.output_dir:
            output_dir = args.output_dir if args.output_dir.is_absolute() else Path.cwd() / args.output_dir
            weighted = _weighted_scores(cv_results)
            metadata = {
                "threshold": args.threshold,
                "weighted_scores": weighted,
                "confusion_matrix": confusion.tolist(),
            }
            save_artifacts(name, output_dir, model, mean, std, metadata)


if __name__ == "__main__":
    main()
