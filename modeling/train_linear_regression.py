"""Baselines for Hoffman et al. (2022) smartphone oximetry data."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, r2_score

FPS = 30
SUBJECT_IDS: Sequence[str] = [
    "100001",
    "100002",
    "100003",
    "100004",
    "100005",
    "100006",
]
FEATURE_NAMES: Sequence[str] = (
    "left_r_mean",
    "left_g_mean",
    "left_b_mean",
    "left_r_std",
    "left_g_std",
    "left_b_std",
    "right_r_mean",
    "right_g_mean",
    "right_b_mean",
    "right_r_std",
    "right_g_std",
    "right_b_std",
    "delta_r_mean",
    "delta_g_mean",
    "delta_b_mean",
)

DATA_ROOT = Path(__file__).resolve().parents[1] / "data"
_PPG_ROOT_CANDIDATES = [
    DATA_ROOT / "ppg-from-raw",
    DATA_ROOT / "ppg-csv",
]
for candidate in _PPG_ROOT_CANDIDATES:
    left_dir = candidate / "Left"
    right_dir = candidate / "Right"
    if left_dir.exists() and right_dir.exists():
        PPG_ROOT = candidate
        break
else:  # pragma: no cover - defensive branch
    raise FileNotFoundError(
        "Could not find Left/Right CSV directories in data/ppg-from-raw or data/ppg-csv"
    )

GT_DIR = DATA_ROOT / "gt"

ModelFactory = Callable[[], LinearRegression]


def _load_ppg(subject_id: str, side: str) -> np.ndarray:
    df = pd.read_csv(PPG_ROOT / side / f"{subject_id}.csv")
    df.columns = [c.strip() for c in df.columns]
    return df[["R", "G", "B"]].to_numpy(dtype=np.float32)


def _load_ground_truth_spo2(subject_id: str) -> np.ndarray:
    df = pd.read_csv(GT_DIR / f"{subject_id}.csv").dropna(how="all")
    df.columns = [c.strip() for c in df.columns]
    spo2_cols = [c for c in df.columns if c.lower().startswith("spo2")]
    if not spo2_cols:
        raise ValueError(f"No SpO2 columns found for subject {subject_id}")
    spo2 = df[spo2_cols].apply(pd.to_numeric, errors="coerce")
    spo2 = spo2.replace(0, math.nan)
    return spo2.mean(axis=1, skipna=True).to_numpy(dtype=np.float32)


def _fit_standardizer(features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return mean, std


def _standardize(features: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (features - mean) / std


def _rescale_coefficients(
    model: LinearRegression, mean: np.ndarray, std: np.ndarray
) -> Tuple[float, np.ndarray]:
    std_safe = np.where(std == 0, 1.0, std)
    coef_scaled = model.coef_.astype(np.float64)
    coef_original = coef_scaled / std_safe
    intercept_original = float(model.intercept_ - np.dot(mean / std_safe, coef_scaled))
    return intercept_original, coef_original


@dataclass
class SubjectDataset:
    subject_id: str
    features: np.ndarray
    labels: np.ndarray


def build_subject_dataset(subject_id: str) -> SubjectDataset:
    left_ppg = _load_ppg(subject_id, "Left")
    right_ppg = _load_ppg(subject_id, "Right")
    gt = _load_ground_truth_spo2(subject_id)

    usable_seconds = min(
        left_ppg.shape[0] // FPS,
        right_ppg.shape[0] // FPS,
        gt.shape[0],
    )
    if usable_seconds == 0:
        raise ValueError(f"Insufficient samples for subject {subject_id}")

    left_ppg = left_ppg[: usable_seconds * FPS]
    right_ppg = right_ppg[: usable_seconds * FPS]
    gt = gt[:usable_seconds]

    left_per_second = left_ppg.reshape(usable_seconds, FPS, 3)
    right_per_second = right_ppg.reshape(usable_seconds, FPS, 3)

    left_means = left_per_second.mean(axis=1)
    left_stds = left_per_second.std(axis=1)
    right_means = right_per_second.mean(axis=1)
    right_stds = right_per_second.std(axis=1)
    delta_means = left_means - right_means

    features = np.concatenate(
        [left_means, left_stds, right_means, right_stds, delta_means],
        axis=1,
    )

    mask = ~np.isnan(gt)
    return SubjectDataset(subject_id, features[mask], gt[mask])


def collect_dataset() -> List[SubjectDataset]:
    return [build_subject_dataset(subject_id) for subject_id in SUBJECT_IDS]


def leave_one_subject_out_cv(
    datasets: Sequence[SubjectDataset], model_factory: ModelFactory
) -> List[dict]:
    results = []
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

        results.append(
            {
                "subject": held_out.subject_id,
                "samples": len(held_out.labels),
                "mae": mean_absolute_error(held_out.labels, preds),
                "r2": r2_score(held_out.labels, preds),
            }
        )
    return results


def weighted_mae(cv_results: Sequence[dict]) -> float:
    total_samples = sum(res["samples"] for res in cv_results)
    return sum(res["mae"] * res["samples"] for res in cv_results) / total_samples


def train_full_model(
    datasets: Sequence[SubjectDataset], model_factory: ModelFactory
) -> Tuple[LinearRegression, np.ndarray, np.ndarray]:
    features = np.vstack([d.features for d in datasets])
    labels = np.concatenate([d.labels for d in datasets])
    mean, std = _fit_standardizer(features)
    features_scaled = _standardize(features, mean, std)
    model = model_factory()
    model.fit(features_scaled, labels)
    return model, mean, std


def display_cv_results(name: str, cv_results: Sequence[dict]) -> float:
    print(f"\n{name} (leave-one-subject-out):")
    for res in cv_results:
        print(
            f"  Subject {res['subject']}: MAE={res['mae']:.2f} %, R^2={res['r2']:.3f} over {res['samples']} samples"
        )
    w_mae = weighted_mae(cv_results)
    print(f"  Weighted MAE: {w_mae:.2f} %")
    return w_mae


def report_coefficients(
    name: str,
    model: LinearRegression,
    mean: np.ndarray,
    std: np.ndarray,
) -> None:
    intercept_original, coef_original = _rescale_coefficients(model, mean, std)
    coef_report = ", ".join(
        f"{feature}={coef:.4f}" for feature, coef in zip(FEATURE_NAMES, coef_original)
    )
    print(f"\n{name} coefficients (trained on all subjects):")
    print(f"  Intercept={intercept_original:.4f}")
    print(f"  {coef_report}")


def evaluate_model(
    name: str, datasets: Sequence[SubjectDataset], model_factory: ModelFactory
) -> Tuple[float, List[dict], Tuple[LinearRegression, np.ndarray, np.ndarray]]:
    cv_results = leave_one_subject_out_cv(datasets, model_factory)
    weighted = display_cv_results(name, cv_results)
    model, mean, std = train_full_model(datasets, model_factory)
    report_coefficients(name, model, mean, std)
    return weighted, cv_results, (model, mean, std)


def tune_ridge_alpha(
    datasets: Sequence[SubjectDataset], alphas: Sequence[float]
) -> Tuple[float, List[Tuple[float, float]]]:
    scores: List[Tuple[float, float]] = []
    best_alpha = alphas[0]
    best_score = float("inf")
    for alpha in alphas:
        cv_results = leave_one_subject_out_cv(
            datasets, lambda alpha=alpha: Ridge(alpha=alpha)
        )
        score = weighted_mae(cv_results)
        scores.append((alpha, score))
        if score < best_score:
            best_score = score
            best_alpha = alpha
    return best_alpha, scores


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Optional directory where the best ridge model, scaler, and metadata will be saved.",
    )
    return parser.parse_args()


def save_artifacts(
    output_dir: Path,
    model: LinearRegression,
    mean: np.ndarray,
    std: np.ndarray,
    metadata: dict,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    dump(model, output_dir / "ridge_model.joblib")
    np.savez(output_dir / "ridge_scaler.npz", mean=mean, std=std, feature_names=np.array(FEATURE_NAMES))
    (output_dir / "ridge_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved ridge model artifacts to {output_dir}")


def main() -> None:
    args = parse_args()
    datasets = collect_dataset()
    print("Synchronized samples per subject (seconds of data):")
    for ds in datasets:
        print(
            f"  {ds.subject_id}: {len(ds.labels)} samples, SpO2 range {ds.labels.min():.1f}-{ds.labels.max():.1f}"
        )

    linear_weighted, _, _ = evaluate_model(
        "Linear regression", datasets, lambda: LinearRegression()
    )

    ridge_alphas = [0.01, 0.1, 1.0, 5.0, 10.0, 25.0]
    best_alpha, alpha_scores = tune_ridge_alpha(datasets, ridge_alphas)
    print("\nRidge alpha tuning (weighted MAE):")
    for alpha, score in alpha_scores:
        print(f"  alpha={alpha:.2f}: {score:.2f} %")

    ridge_label = f"Ridge regression (alpha={best_alpha:.2f})"
    ridge_weighted, _, (ridge_model, ridge_mean, ridge_std) = evaluate_model(
        ridge_label,
        datasets,
        lambda: Ridge(alpha=best_alpha),
    )

    if args.output_dir:
        output_dir = args.output_dir if args.output_dir.is_absolute() else Path.cwd() / args.output_dir
        intercept, coefficients = _rescale_coefficients(ridge_model, ridge_mean, ridge_std)
        metadata = {
            "alpha": best_alpha,
            "weighted_mae": ridge_weighted,
            "linear_weighted_mae": linear_weighted,
            "feature_names": list(FEATURE_NAMES),
            "intercept": intercept,
            "coefficients": coefficients.tolist(),
        }
        save_artifacts(output_dir, ridge_model, ridge_mean, ridge_std, metadata)


if __name__ == "__main__":
    main()
