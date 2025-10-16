"""Utility functions for building classification datasets from RGB features."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

from train_linear_regression import collect_dataset


@dataclass
class ClassificationDataset:
    subject_id: str
    features: np.ndarray
    labels: np.ndarray  # binary (1 = SpO₂ below threshold)
    continuous: np.ndarray  # original SpO₂ values


def build_classification_datasets(threshold: float) -> List[ClassificationDataset]:
    base_datasets = collect_dataset()
    classification_datasets: List[ClassificationDataset] = []
    for base in base_datasets:
        labels_binary = (base.labels < threshold).astype(np.int32)
        classification_datasets.append(
            ClassificationDataset(base.subject_id, base.features, labels_binary, base.labels)
        )
    return classification_datasets


def fit_standardizer(features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return mean, std


def standardize(features: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (features - mean) / std
