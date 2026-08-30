"""Shared helper utilities for validation modules."""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    """L2-normalize rows of an embedding matrix."""
    arr = np.asarray(vectors, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.maximum(norms, 1e-10)


def has_sufficient_variance(
    embeddings: np.ndarray, min_spread: float = 1e-6
) -> bool:
    """Return True when embeddings are not nearly collapsed to one point."""
    if embeddings.shape[0] < 2:
        return False
    return float(np.max(np.std(embeddings, axis=0))) >= min_spread


def cosine_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """n x n cosine-similarity matrix for an embedding matrix."""
    normed = l2_normalize(embeddings)
    return np.clip(normed @ normed.T, -1.0, 1.0)


def vendi_score(
    embeddings: np.ndarray,
    *,
    normalize_by_n: bool = False,
) -> float:
    """Compute Vendi score from embeddings.

    If normalize_by_n is True, returns VS(X) / n.
    """
    n = embeddings.shape[0]
    if n <= 1:
        return 1.0 if normalize_by_n else float(n)

    K = cosine_similarity_matrix(embeddings) / n
    eigvals = np.linalg.eigvalsh(K)
    eigvals = eigvals[eigvals > 0]
    if len(eigvals) == 0:
        return 1.0

    p = eigvals / eigvals.sum()
    score = float(np.exp(-np.sum(p * np.log(p + 1e-15))))
    return score / n if normalize_by_n else score


def value_to_str(value: object) -> str:
    """Convert dataframe values to stable string labels."""
    return "NA" if pd.isna(value) else str(value)


def make_composite_labels(
    df: pd.DataFrame,
    label_cols: List[str],
) -> List[str]:
    """Build + joined key=value labels from selected dataframe columns."""
    labels: List[str] = []
    for _, row in df.iterrows():
        parts = [f"{col}={value_to_str(row[col])}" for col in label_cols]
        labels.append("+".join(parts))
    return labels
