"""
Diversity Validation

Diversity measures the amount of linguistic and structural variation in the evaluation
dataset, at both the overall dataset level and per subgroup of constituent values.

Three complementary metrics are implemented:

──────────────────────────────────────────────────────────────────────────────────────
1. DCScore  (primary — classification-based, O(n²) summarisation)
──────────────────────────────────────────────────────────────────────────────────────
DCScore (Zhu et al., ICML 2025 — arXiv:2502.08512) frames diversity as a sample-level
n-class classification task.  Given n samples:

  Step 1 – embed:      h_i = Φ(T̃_i),   H ∈ ℝ^{n×d}
  Step 2 – normalise:  H ← L2-normalise(H)
  Step 3 – kernel:     K[i,j] = Kernel(h_i, h_j)
  Step 4 – classify:   P = row-wise softmax(K_scaled)
  Step 5 – summarise:  DCScore = tr(P) = Σᵢ P[i,i]

Kernel / tau semantics  (matches official dcscore_function.py exactly):

  Kernel          scikit-learn call                         Softmax input
  ─────────────── ─────────────────────────────────────     ──────────────
  'cs'            H @ H.T  (cosine sim for unit vecs)       K / tau      ← tau is temperature
  'rbf'           rbf_kernel(H, H, gamma=tau)               K            ← tau is gamma
  'laplacian'     laplacian_kernel(H, H, gamma=tau)         K            ← tau is gamma
  'polynomial'    polynomial_kernel(H, H, degree=tau)       K            ← tau is degree

  For 'cs', tau is a softmax temperature: lower sharpens class boundaries.
  For all other kernels, tau is consumed by the kernel; softmax receives K directly.

Interpretation:
    - DCScore ∈ [1/n, 1] after normalising by sample count.
    - DCScore = 1/n when all samples are identical   (P[i,i] = 1/n for all i).
    - DCScore → 1 when all samples are mutually distinct (P[i,i] → 1 for all i).
  - O(n²) summarisation vs O(n³) eigendecomposition in VendiScore.
  - Satisfies: effective number, identical samples, symmetry, monotonicity axioms.
  - Paper default: 'cs' kernel, tau = 1.

Reference:
  Zhu, Y. et al. (2025). Measuring Diversity in Synthetic Datasets. ICML 2025.
  arXiv:2502.08512   GitHub: https://github.com/bluewhalelab/dcscore

──────────────────────────────────────────────────────────────────────────────────────
2. VendiScore  (secondary — eigenspectrum-based, O(n³))
──────────────────────────────────────────────────────────────────────────────────────
VS(X) = exp( H(e / Σe) ),  e = eigenvalues of cosine-sim-matrix / n.
We report VS(X) / n so it shares the [1/n, 1] range with DCScore.
Retained as a well-established cross-check.

Reference: Friedman & Dieng (2022). arXiv:2210.02410

──────────────────────────────────────────────────────────────────────────────────────
3. Mean Pairwise Similarity  (supplementary — surface-level redundancy)
──────────────────────────────────────────────────────────────────────────────────────
  surface_diversity = 1 − mean_pairwise_cosine_similarity
Kept for backward-compatibility with the coverage report.

Subgroup diversity: all three metrics are computed per constituent-value group.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import (
    laplacian_kernel as _sklearn_laplacian,
    polynomial_kernel as _sklearn_poly,
    rbf_kernel as _sklearn_rbf,
)
from sklearn.preprocessing import normalize as _sklearn_normalize

from operationalized_metrics.helpers import (
    cosine_similarity_matrix,
    l2_normalize,
    value_to_str,
    vendi_score,
)

# Reports are represented as plain dictionaries for simpler downstream usage.


# ---------------------------------------------------------------------------
# Standalone dc_score function  (mirrors official dcscore_function.py API)
# ---------------------------------------------------------------------------


def dc_score(
    texts: Optional[List[str]] = None,
    *,
    embeddings: Optional[np.ndarray] = None,
    embedding_model: str = "all-mpnet-base-v2",
    kernel: Literal["cs", "rbf", "laplacian", "polynomial"] = "cs",
    tau: float = 1.0,
) -> float:
    """
    Compute DCScore for a list of texts or a pre-computed embedding matrix.

    This faithfully reimplements ``DCScore.calculate_dcscore_by_embedding``
    from the official source (github.com/bluewhalelab/dcscore), replacing
    PyTorch/sklearn with pure NumPy for environment portability while
    producing identical results.

    Algorithm
    ─────────
    1.  L2-normalise embeddings   (mirrors ``preprocessing.normalize``).
    2.  Build n×n kernel matrix K.
    3.  Compute row-wise softmax of the (possibly scaled) K.
    4.  Return tr(P) = Σᵢ P[i,i].

    Kernel / tau semantics (matches official source exactly)
    ────────────────────────────────────────────────────────
    ``'cs'``           K = H @ H.T  (cosine similarity for L2-normalised vecs).
                       Softmax input = K / tau.  tau is a **softmax temperature**.
                       Lower tau → sharper class boundaries.  Paper default: 1.

    ``'rbf'``          K = rbf_kernel(H, H, gamma=tau)  [sklearn convention].
                       K[i,j] = exp(−tau · ‖hᵢ−hⱼ‖²).
                       Softmax applied directly to K.  tau is **gamma**.

    ``'laplacian'``    K = laplacian_kernel(H, H, gamma=tau).
                       K[i,j] = exp(−tau · ‖hᵢ−hⱼ‖₁).
                       Softmax applied directly to K.  tau is **gamma**.

    ``'polynomial'``   K = polynomial_kernel(H, H, degree=tau).
                       K[i,j] = (hᵢ·hⱼ + 1)^tau.
                       Softmax applied directly to K.  tau is **degree**.

    Args:
        texts:           Raw text strings.  Ignored when ``embeddings`` is given.
                         Must supply at least one of ``texts`` / ``embeddings``.
        embeddings:      Pre-computed (n, d) array.  L2-normalised internally.
                         Pass this to avoid re-embedding when embeddings are already
                         available (e.g. reuse from CoverageValidator).
        embedding_model: Sentence-transformer model name (used only for ``texts``).
        kernel:          Pairwise similarity kernel.
        tau:             Kernel-specific hyperparameter (see table above).

    Returns:
        DCScore ∈ [1/n, 1] (normalised by sample count).  Higher = more diverse.

    Examples::

        # Simplest: pass raw texts
        score = dc_score(["The cat sat.", "A dog ran.", "Birds fly high."])

        # Reuse embeddings already computed — avoids re-embedding
        score = dc_score(embeddings=my_arr)

        # RBF kernel (tau = gamma)
        score = dc_score(texts, kernel="rbf", tau=0.1)

        # Polynomial kernel (tau = degree)
        score = dc_score(texts, kernel="polynomial", tau=3)
    """
    # ---- 0. Obtain embeddings ----
    if embeddings is not None:
        H = np.array(embeddings, dtype=np.float64)
    elif texts:
        mdl = SentenceTransformer(embedding_model)
        H = mdl.encode(texts, convert_to_numpy=True).astype(np.float64)
    else:
        raise ValueError("Provide either 'texts' or 'embeddings'.")

    # L2-normalise rows — mirrors official `preprocessing.normalize(arr, axis=1)`
    H = _sklearn_normalize(H, norm="l2", axis=1)

    n = H.shape[0]
    if n == 1:
        return 1.0

    # ---- 1. Build kernel matrix and determine softmax input ----
    if kernel == "cs":
        # Official: (embeddings_arr @ embeddings_arr.T) / tau  → softmax
        K_input = (H @ H.T) / tau

    elif kernel == "rbf":
        # Official: rbf_kernel(arr, arr, tau)  → softmax   [tau = gamma]
        K_input = _sklearn_rbf(H, H, gamma=tau)

    elif kernel == "laplacian":
        # Official: laplacian_kernel(arr, arr, tau)  → softmax  [tau = gamma]
        K_input = _sklearn_laplacian(H, H, gamma=tau)

    elif kernel == "polynomial":
        # Official: polynomial_kernel(arr, arr, tau)  → softmax  [tau = degree]
        # sklearn signature: polynomial_kernel(X, Y, degree, gamma=None, coef0=1)
        K_input = _sklearn_poly(H, H, degree=tau)

    else:
        raise ValueError(
            f"Unknown kernel '{kernel}'. "
            "Choose from: 'cs', 'rbf', 'laplacian', 'polynomial'."
        )

    # ---- 2. Row-wise softmax (numerically stable via max subtraction) ----
    # Mirrors torch's .softmax(dim=-1) which does the same stabilisation.
    K_shifted = K_input - K_input.max(axis=1, keepdims=True)
    exp_K = np.exp(K_shifted)
    P = exp_K / exp_K.sum(axis=1, keepdims=True)  # shape (n, n)

    # ---- 3. tr(P) = Σᵢ P[i,i] ----
    return float(np.trace(P)) / n


# ---------------------------------------------------------------------------
# DiversityValidator
# ---------------------------------------------------------------------------


class DiversityValidator:
    """
    Validates linguistic and structural diversity of a synthetic evaluation dataset.

    Expects the same DataFrame format as ``CoverageValidator``:
    an ``'example'`` text column plus one or more constituent label columns.

    All three metrics (DCScore, VendiScore, surface diversity) are computed at the
    full-dataset level and for each constituent-value subgroup.

    Usage::

        validator = DiversityValidator()
        report = validator.validate(df, constituent_cols=["topic", "actor"])
        validator.print_report(report)
    """

    def __init__(
        self,
        embedding_model: str = "all-mpnet-base-v2",
        redundancy_threshold: float = 0.95,
        min_samples_for_stats: int = 2,
        n_lowest_to_report: int = 5,
        dc_kernel: Literal["cs", "rbf", "laplacian", "polynomial"] = "cs",
        dc_tau: float = 1.0,
    ):
        """
        Args:
            embedding_model:       Sentence-transformer model name.
            redundancy_threshold:  Cosine similarity above which two examples are
                                   flagged as near-duplicates.
            min_samples_for_stats: Minimum group size required to compute stats.
            n_lowest_to_report:    Number of lowest-diversity groups to highlight.
            dc_kernel:             DCScore kernel.  One of ``'cs'`` (default, fastest),
                                   ``'rbf'``, ``'laplacian'``, ``'polynomial'``.
            dc_tau:                DCScore tau.  For ``'cs'``: softmax temperature.
                                   For others: kernel-specific hyperparameter
                                   (gamma for rbf/laplacian, degree for polynomial).
        """
        self.model = SentenceTransformer(embedding_model)
        self.redundancy_threshold = redundancy_threshold
        self.min_samples_for_stats = min_samples_for_stats
        self.n_lowest_to_report = n_lowest_to_report
        self.dc_kernel = dc_kernel
        self.dc_tau = dc_tau

        # Populated during validate()
        self.embeddings: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _normalize(self, vectors: np.ndarray) -> np.ndarray:
        return l2_normalize(vectors)

    def _cosine_sim_matrix(self, embeddings: np.ndarray) -> np.ndarray:
        """n×n cosine-similarity matrix for L2-normalised embeddings."""
        return cosine_similarity_matrix(embeddings)

    def _dc_score(self, embeddings: np.ndarray) -> float:
        """Compute DCScore using self.dc_kernel / dc_tau via the standalone function."""
        return dc_score(
            embeddings=embeddings, kernel=self.dc_kernel, tau=self.dc_tau
        )

    def _vendi_score(self, embeddings: np.ndarray) -> float:
        """VendiScore normalised to [1/n, 1] via exp(entropy(eig(K / n))) / n."""
        return vendi_score(embeddings, normalize_by_n=True)

    def _mean_pairwise_sim(self, embeddings: np.ndarray) -> float:
        """Mean upper-triangle cosine similarity (excludes diagonal)."""
        n = embeddings.shape[0]
        if n < 2:
            return 1.0
        sim = self._cosine_sim_matrix(embeddings)
        return float(sim[np.triu_indices(n, k=1)].mean())

    def _redundant_pairs(
        self,
        embeddings: np.ndarray,
        global_indices: List[int],
    ) -> List[Tuple[int, int]]:
        """Return (i, j) global-index pairs with cosine similarity > threshold."""
        n = embeddings.shape[0]
        if n < 2:
            return []
        sim = self._cosine_sim_matrix(embeddings)
        return [
            (global_indices[i], global_indices[j])
            for i in range(n)
            for j in range(i + 1, n)
            if sim[i, j] > self.redundancy_threshold
        ]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_subgroup_diversity(
        self,
        label_to_indices: Dict[str, List[int]],
    ) -> Dict[str, Dict[str, Any]]:
        """
        Compute diversity stats for every label group in label_to_indices.

        Args:
            label_to_indices: {label: [global_row_index, ...]} mapping.
                              Uses ``self.embeddings``.

        Returns:
            Dict mapping label → subgroup stats dictionary.
        """
        assert (
            self.embeddings is not None
        ), "Embeddings not initialised. Call validate() first."
        stats: Dict[str, Dict[str, Any]] = {}

        for label, indices in label_to_indices.items():
            if len(indices) < self.min_samples_for_stats:
                continue

            grp = self.embeddings[indices]
            mps = self._mean_pairwise_sim(grp)

            stats[label] = {
                "label": label,
                "count": len(indices),
                "dc_score": self._dc_score(grp),
                "vendi_score": self._vendi_score(grp),
                "mean_pairwise_similarity": mps,
                "surface_diversity_score": 1.0 - mps,
                "redundant_pairs": self._redundant_pairs(grp, indices),
            }

        return stats

    def validate(
        self,
        df: pd.DataFrame,
        constituent_cols: List[str],
    ) -> Dict[str, Any]:
        """
        Run full diversity validation on a generated dataset.

        Args:
            df:               DataFrame with an ``'example'`` text column and all
                              columns listed in ``constituent_cols``.
            constituent_cols: Constituent label columns (e.g. ``["topic", "actor"]``).

        Returns:
            DiversityReport.
        """
        required = set(constituent_cols) | {"example"}
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        texts = df["example"].astype(str).tolist()
        label_cols = [c for c in constituent_cols if c in df.columns]
        labels: List[str] = [
            "+".join(f"{c}={value_to_str(row[c])}" for c in label_cols)
            for _, row in df.iterrows()
        ]

        # Compute and cache L2-normalised embeddings
        self.embeddings = None
        print("Computing embeddings...")
        self.embeddings = self._normalize(self.model.encode(texts))
        n = len(texts)

        # Overall metrics
        print("Computing overall diversity...")
        overall_dc = self._dc_score(self.embeddings)
        overall_vs = self._vendi_score(self.embeddings)
        overall_mps = self._mean_pairwise_sim(self.embeddings)

        # Build label → index maps for fine-grained and coarse groups
        label_to_indices: Dict[str, List[int]] = defaultdict(list)
        for i, label in enumerate(labels):
            label_to_indices[label].append(i)
            for part in label.split("+"):
                label_to_indices[part].append(i)

        # Subgroup metrics
        print("Computing subgroup diversity...")
        subgroup_stats = self.compute_subgroup_diversity(label_to_indices)

        # Near-duplicates across the full dataset
        all_redundant = self._redundant_pairs(self.embeddings, list(range(n)))

        # Lowest-diversity groups (primary collapse signal)
        lowest = [
            (lbl, st["dc_score"])
            for lbl, st in sorted(
                subgroup_stats.items(), key=lambda kv: kv[1]["dc_score"]
            )
        ][: self.n_lowest_to_report]

        return {
            "overall_dc_score": overall_dc,
            "overall_vendi_score": overall_vs,
            "overall_mean_pairwise_similarity": overall_mps,
            "subgroup_stats": subgroup_stats,
            "all_redundant_pairs": all_redundant,
            "lowest_diversity_groups": lowest,
        }

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def print_report(self, report: Dict[str, Any]) -> None:
        print("\n" + "=" * 75)
        print("DIVERSITY VALIDATION REPORT")
        print("=" * 75)

        print("\n--- Overall Dataset ---")
        print(
            f"  DCScore   (classification-based, O(n²), normalised):  {report['overall_dc_score']:.3f}"
            f"  [kernel='{self.dc_kernel}', τ={self.dc_tau}]"
        )
        print(
            f"  VendiScore (eigenspectrum-based,  O(n³), normalised):  {report['overall_vendi_score']:.3f}"
        )
        # Surface diversity reporting removed as requested
        print(
            f"  Near-duplicate pairs (>{self.redundancy_threshold:.0%} sim):  "
            f"{len(report['all_redundant_pairs'])}"
        )

        print("\n--- Per-Subgroup Diversity ---")
        header = (
            f"{'Group':<38} {'n':>5} {'DCScore':>9}"
            f" {'Vendi':>8} {'SurfDiv':>8} {'Dups':>6}"
        )
        print(header)
        print("-" * len(header))
        for label, st in sorted(report["subgroup_stats"].items()):
            print(
                f"{label:<38} {st['count']:>5} {st['dc_score']:>9.3f}"
                f" {st['vendi_score']:>8.3f} {st['surface_diversity_score']:>8.3f}"
                f" {len(st['redundant_pairs']):>6}"
            )

        print(
            f"\n--- {self.n_lowest_to_report} Lowest-Diversity Groups (by DCScore) ---"
        )
        for label, dc in report["lowest_diversity_groups"]:
            collapse_flag = ""
            subgroup = report["subgroup_stats"].get(label)
            if subgroup is not None:
                threshold = min(1.0, 2.0 / max(subgroup["count"], 1))
                if dc < threshold:
                    collapse_flag = (
                        f"  ⚠  possible mode collapse (DC<{threshold:.3f})"
                    )
            print(f"  {label:<48}  DC={dc:.3f}{collapse_flag}")

        if report["all_redundant_pairs"]:
            print("\n--- Near-Duplicate Pairs (first 5) ---")
            for i, j in report["all_redundant_pairs"][:5]:
                print(f"  row {i} ↔ row {j}")

        print("\n" + "=" * 75)
