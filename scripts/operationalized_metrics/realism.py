"""
This module computes both stylistic and content realism. It also does so at
both the instance- and distribution-level.

The validator compares generated examples with schema seed examples using the
embedding representations and constituent labels available to the pipeline.
It reports per-example, per-category, and dataset-level diagnostics, including
novelty, scope, seed validity, containment, extrapolation, style distance, and
semantic distributional alignment. The resulting dictionary is serialized by
``serialize_realism_report`` for the metrics JSON output.

Realism is reported at two complementary levels:

Instance-level realism
----------------------
These diagnostics describe individual generated examples and their constituent
categories. They include novelty relative to seed examples, scope or
containment, log likelihood, seed validity, and extrapolation quality. They
help identify whether particular examples are plausible and how each example
contributes to the aggregate realism results.

Each generated example should be *plausible* — i.e. it could have been written by
a member of the intended deployment population.  We operationalise this as semantic
proximity to the seed examples provided by the domain expert.

For every generated example g we compute:

  novelty_min(g)  = min_{s ∈ seeds} cosine_distance(g, s)
                  = 1 − max_{s ∈ seeds} cosine_similarity(g, s)

  novelty_avg(g)  = mean_{s ∈ seeds} cosine_distance(g, s)

  in_scope(g)     = log_likelihood(g | vMF fitted to seeds + generated)
                    ≥ percentile_threshold(seed log-likelihoods)

Additionally (outdated) metrics:
1. Seed Validity Score  – mean log-likelihood (LL) of seeds under their category vMF. (NOTE: this is only done if the seed examples have labeled constituent values, if not, ignore)
2. Novelty (min)        – mean of min-seed distances across generated examples.
3. Novelty (avg)        – mean of avg-seed distances across generated examples.
4. Containment Rate     – % of generated with LL ≥ seed-LL percentile threshold.
5. Extrapolation Quality – novelty_min × containment_rate.


Distribution-level realism
---------------------------
The validator computes calibrated Sinkhorn optimal-transport alignment between
the seed and generated embedding distributions. The resulting
``calibrated_sinkhorn_distance`` and its uncertainty interval summarize how
closely the generated dataset matches the seed distribution at the dataset
level.

The implementation also computes a separate stylistic signal:
``mean_style_distance_avg`` and ``style_realism_score`` summarize the average
style distance between generated examples and the seed style reference. Thus,
the output preserves the instance/distribution distinction while exposing both
content/distributional and stylistic realism signals.
The generated distribution should align with the seed distribution in both location
and spread. We measure this with calibrated Sinkhorn distance (entropic OT) over
embedding-space cosine distances.

We compute:

    1) raw_sinkhorn(seed, generated)
    2) calibration anchors:
         - near: seed-vs-seed bootstrap split distances
         - far:  seed-vs-random-unit-vector distances
    3) calibrated alignment score in [0, 1]

Additionally, any constituent-value combination that the domain expert has declared
*invalid* must have zero coverage in the generated set.
"""

import warnings
import importlib
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
from scipy.stats import vonmises_fisher
from sentence_transformers import SentenceTransformer

try:
    ot = importlib.import_module("ot")
except Exception:  # pragma: no cover
    ot = None

from operationalized_metrics.helpers import (
    has_sufficient_variance,
    l2_normalize,
    make_composite_labels,
    value_to_str,
)

# Reports are represented as plain dictionaries for simpler downstream usage.


def serialize_realism_report(report: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a realism report into JSON-friendly primitive types."""
    instance_stats = []
    for item in report.get("instance_stats", []):
        instance_stats.append(
            {
                "idx": int(item["idx"]),
                "label": item["label"],
                "novelty_min": float(item["novelty_min"]),
                "novelty_avg": float(item["novelty_avg"]),
                "log_likelihood": float(item["log_likelihood"]),
                "is_in_scope": bool(item["is_in_scope"]),
            }
        )

    category_stats: Dict[str, Dict[str, Any]] = {}
    for label, stats in report.get("category_stats", {}).items():
        seed_validity = stats.get("seed_validity_score")
        category_stats[label] = {
            "n_generated": int(stats.get("n_generated", 0)),
            "n_seeds": int(stats.get("n_seeds", 0)),
            "seed_validity_score": (
                float(seed_validity)
                if seed_validity is not None and np.isfinite(seed_validity)
                else None
            ),
            "novelty_min": float(stats.get("novelty_min", float("nan"))),
            "novelty_avg": float(stats.get("novelty_avg", float("nan"))),
            "containment_rate": float(
                stats.get("containment_rate", float("nan"))
            ),
            "extrapolation_quality": float(
                stats.get("extrapolation_quality", float("nan"))
            ),
        }

    invalid_combo_coverage = {}
    for combo, count in report.get("invalid_combo_coverage", {}).items():
        invalid_combo_coverage[" | ".join(combo)] = int(count)

    seed_validity_score = report.get("seed_validity_score")

    return {
        "instance_stats": instance_stats,
        "category_stats": category_stats,
        "raw_sinkhorn_distance": float(
            report.get("raw_sinkhorn_distance", float("nan"))
        ),
        "calibrated_sinkhorn_distance": float(
            report.get("calibrated_sinkhorn_distance", float("nan"))
        ),
        "calibrated_sinkhorn_distance_ci_005": float(
            report.get("calibrated_sinkhorn_distance_ci_005", float("nan"))
        ),
        "calibrated_sinkhorn_distance_ci_095": float(
            report.get("calibrated_sinkhorn_distance_ci_095", float("nan"))
        ),
        "invalid_combo_coverage": invalid_combo_coverage,
        "dataset_containment_rate": float(
            report.get("dataset_containment_rate", float("nan"))
        ),
        "dataset_extrapolation_quality": float(
            report.get("dataset_extrapolation_quality", float("nan"))
        ),
        "out_of_scope_indices": [
            int(idx) for idx in report.get("out_of_scope_indices", [])
        ],
        "seed_has_labels": bool(report.get("seed_has_labels", False)),
        "seed_validity_score": (
            float(seed_validity_score)
            if seed_validity_score is not None
            else None
        ),
        "mean_dataset_novelty_min": float(
            report.get("mean_dataset_novelty_min", float("nan"))
        ),
        "mean_dataset_novelty_avg": float(
            report.get("mean_dataset_novelty_avg", float("nan"))
        ),
        "mean_style_distance_avg": float(
            report.get("mean_style_distance_avg", float("nan"))
        ),
        "style_realism_score": float(
            report.get("style_realism_score", float("nan"))
        ),
    }


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------


class RealismValidator:
    """
    Validates the realism of a generated evaluation dataset relative to
    domain-expert seed examples.

        The validator requires generated examples and either:
            - a seed DataFrame with the same constituent columns, or
            - a plain list of seed example texts.

        When only seed texts are provided, instance-level realism still runs, but
        group-level seed validity scores are skipped because no seed labels are
        available.

    Usage::

        validator = RealismValidator()
        report = validator.validate(
            generated_df=gen_df,
            seed_df=seed_df,
            constituent_cols=["topic", "actor"],
            invalid_combinations=[("corruption", "actor_c")],
        )
        validator.print_report(report)
    """

    def __init__(
        self,
        embedding_model: str = "all-mpnet-base-v2",
        style_embedding_model: str = "AnnaWegmann/Style-Embedding",
        seed_percentile_threshold: float = 10.0,
        min_samples_for_vmf: int = 3,
        sinkhorn_reg: float = 0.01,
        sinkhorn_calibration_bootstraps: int = 25,
        sinkhorn_seed: int = 42,
    ):
        """
        Args:
            embedding_model:           Sentence-transformer model name.
            style_embedding_model:     Sentence-transformer model name used for
                                       style embeddings.
            seed_percentile_threshold: Log-likelihood percentile of seed examples used
                                       as the containment boundary.  Generated examples
                                       whose LL falls below this percentile of seed LLs
                                       are flagged as out-of-scope.
            min_samples_for_vmf:       Minimum examples needed to fit a vMF distribution.
            sinkhorn_reg:              Entropic regularization for Sinkhorn OT.
            sinkhorn_calibration_bootstraps: Bootstrap count for calibration anchors.
            sinkhorn_seed:             RNG seed for reproducible calibration.
        """
        self.model = SentenceTransformer(embedding_model)
        self.style_model = SentenceTransformer(style_embedding_model)
        self.seed_percentile_threshold = seed_percentile_threshold
        self.min_samples_for_vmf = min_samples_for_vmf
        self.sinkhorn_reg = sinkhorn_reg
        self.sinkhorn_calibration_bootstraps = sinkhorn_calibration_bootstraps
        self.sinkhorn_seed = sinkhorn_seed

        # Populated during validate()
        self._seed_embeddings: Optional[np.ndarray] = None
        self._gen_embeddings: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Internal helpers  (mirrors CoverageValidator conventions)
    # ------------------------------------------------------------------

    def _normalize(self, vectors: np.ndarray) -> np.ndarray:
        return l2_normalize(vectors)

    def _safe_unit_vector(self, vector: np.ndarray) -> np.ndarray:
        """Return a finite unit vector, falling back to a basis vector if needed."""
        arr = np.asarray(vector, dtype=np.float64).reshape(-1)
        if arr.size == 0 or not np.all(np.isfinite(arr)):
            arr = np.zeros(1, dtype=np.float64)
            arr[0] = 1.0
            return arr

        norm = float(np.linalg.norm(arr))
        if norm <= 1e-12:
            fallback = np.zeros_like(arr)
            fallback[0] = 1.0
            return fallback

        return arr / norm

    def _has_sufficient_variance(self, embeddings: np.ndarray) -> bool:
        return has_sufficient_variance(embeddings)

    def _fit_vmf(self, embeddings: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Fit a vMF distribution, falling back to closed-form if scipy diverges.

        Returns:
            mu    – mean direction (unit vector)
            kappa – concentration parameter
        """
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("error", category=RuntimeWarning)
                mu, kappa = vonmises_fisher.fit(embeddings)
        except (RuntimeWarning, Exception):
            mu, kappa = self._fit_vmf_approx(embeddings)

        if not np.isfinite(kappa) or not np.all(np.isfinite(mu)):
            mu, kappa = self._fit_vmf_approx(embeddings)

        mu = self._safe_unit_vector(mu)
        return mu, float(kappa)

    def _fit_vmf_approx(
        self, embeddings: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """Closed-form fallback (same as CoverageValidator._approximate_vmf_parameters)."""
        mean_vec = embeddings.mean(axis=0)
        mu = self._safe_unit_vector(mean_vec)
        norm = float(np.linalg.norm(mean_vec))
        r_bar = float(np.clip(norm, 1e-6, 1 - 1e-6))
        d = embeddings.shape[1]
        kappa = (r_bar * (d - r_bar**2)) / (1 - r_bar**2)
        return mu, float(max(kappa, 1e-3))

    def _vmf_log_likelihood(
        self, embeddings: np.ndarray, mu: np.ndarray, kappa: float
    ) -> np.ndarray:
        return vonmises_fisher.logpdf(embeddings, mu=mu, kappa=kappa)

    def _cosine_distance_matrix(
        self, A: np.ndarray, B: np.ndarray
    ) -> np.ndarray:
        """
        Compute cosine distances between every row in A and every row in B.

        Returns:
            (|A|, |B|) array where entry [i, j] = 1 − cos_sim(A[i], B[j]).
        """
        A_normed = self._normalize(A)
        B_normed = self._normalize(B)
        cos_sim = A_normed @ B_normed.T  # shape (|A|, |B|)
        return 1.0 - np.clip(cos_sim, -1.0, 1.0)

    def _sinkhorn_distance(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        *,
        debiased: bool = True,
    ) -> float:
        """Compute Sinkhorn OT distance between two embedding sets."""
        if ot is None:
            raise ImportError(
                "Sinkhorn realism requires POT. Install with: pip install POT"
            )

        Xn = self._normalize(X)
        Yn = self._normalize(Y)

        a = np.ones(Xn.shape[0], dtype=np.float64) / max(Xn.shape[0], 1)
        b = np.ones(Yn.shape[0], dtype=np.float64) / max(Yn.shape[0], 1)

        M_xy = 1.0 - np.clip(Xn @ Yn.T, -1.0, 1.0)
        ot_xy = float(ot.sinkhorn2(a, b, M_xy, reg=self.sinkhorn_reg))
        if not debiased:
            return max(ot_xy, 0.0)

        M_xx = 1.0 - np.clip(Xn @ Xn.T, -1.0, 1.0)
        M_yy = 1.0 - np.clip(Yn @ Yn.T, -1.0, 1.0)
        ot_xx = float(ot.sinkhorn2(a, a, M_xx, reg=self.sinkhorn_reg))
        ot_yy = float(ot.sinkhorn2(b, b, M_yy, reg=self.sinkhorn_reg))
        return max(float(ot_xy - 0.5 * ot_xx - 0.5 * ot_yy), 0.0)

    def _calibrated_sinkhorn_alignment(
        self,
        seed_embeddings: np.ndarray,
        gen_embeddings: np.ndarray,
    ) -> Dict[str, float]:
        """Calibrate Sinkhorn distance into [0,1] distributional realism."""
        rng = np.random.default_rng(self.sinkhorn_seed)
        raw_sinkhorn = self._sinkhorn_distance(seed_embeddings, gen_embeddings)

        n_seed, d = seed_embeddings.shape
        n_gen = gen_embeddings.shape[0]
        boot = max(int(self.sinkhorn_calibration_bootstraps), 5)

        near_vals: List[float] = []
        far_vals: List[float] = []

        # Near anchor: two bootstrap resamples from seed set.
        # This remains stable even for very small seed sets (e.g., 3-10 points).
        if n_seed >= 2:
            sample_n = max(2, n_seed)
            for _ in range(boot):
                a_idx = rng.integers(0, n_seed, size=sample_n)
                b_idx = rng.integers(0, n_seed, size=sample_n)
                near_vals.append(
                    self._sinkhorn_distance(
                        seed_embeddings[a_idx], seed_embeddings[b_idx]
                    )
                )

        rand_n = max(2, n_gen)
        for _ in range(boot):
            random_vecs = rng.normal(size=(rand_n, d)).astype(np.float64)
            random_vecs = self._normalize(random_vecs)
            far_vals.append(
                self._sinkhorn_distance(seed_embeddings, random_vecs)
            )

        near_anchor = float(np.median(near_vals)) if near_vals else 0.0
        far_anchor = (
            float(np.median(far_vals)) if far_vals else max(raw_sinkhorn, 1e-6)
        )

        denom = max(far_anchor - near_anchor, 1e-8)
        alignment_score = float(
            np.clip(1.0 - (raw_sinkhorn - near_anchor) / denom, 0.0, 1.0)
        )

        # Bootstrap CI for calibrated alignment (resample both seed and generated sets).
        align_samples: List[float] = []
        if n_seed >= 2 and n_gen >= 2:
            seed_n = max(2, n_seed)
            gen_n = max(2, n_gen)
            for _ in range(boot):
                s_idx = rng.integers(0, n_seed, size=seed_n)
                g_idx = rng.integers(0, n_gen, size=gen_n)
                sampled_raw = self._sinkhorn_distance(
                    seed_embeddings[s_idx], gen_embeddings[g_idx]
                )
                sampled_align = float(
                    np.clip(
                        1.0 - (sampled_raw - near_anchor) / denom,
                        0.0,
                        1.0,
                    )
                )
                align_samples.append(sampled_align)

        if align_samples:
            alignment_ci_low = float(np.percentile(align_samples, 5))
            alignment_ci_high = float(np.percentile(align_samples, 95))
        else:
            alignment_ci_low = alignment_score
            alignment_ci_high = alignment_score

        return {
            "raw_sinkhorn_distance": raw_sinkhorn,
            "calibrated_sinkhorn_distance": alignment_score,
            "calibrated_sinkhorn_distance_ci_005": alignment_ci_low,
            "calibrated_sinkhorn_distance_ci_095": alignment_ci_high,
        }

    # ------------------------------------------------------------------
    # Core computations
    # ------------------------------------------------------------------

    def _instance_novelty(
        self,
        gen_embeddings: np.ndarray,
        seed_embeddings: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        For each generated example compute novelty_min and novelty_avg.

        Returns:
            novelty_min: (n_gen,) array – distance to nearest seed
            novelty_avg: (n_gen,) array – mean distance across all seeds
        """
        dist = self._cosine_distance_matrix(gen_embeddings, seed_embeddings)
        # dist shape: (n_gen, n_seeds)
        novelty_min = dist.min(axis=1)
        novelty_avg = dist.mean(axis=1)
        return novelty_min, novelty_avg

    def _compute_category_stats(
        self,
        label: str,
        gen_indices: List[int],
        seed_indices: List[int],
        gen_embeddings: np.ndarray,
        seed_embeddings: np.ndarray,
    ) -> Optional[Dict[str, Any]]:
        """
        Compute per-category realism stats.

        Args:
            label:          Composite constituent label.
            gen_indices:    Indices into gen_embeddings for this category.
            seed_indices:   Indices into seed_embeddings for this category.
            gen_embeddings: All generated embeddings.
            seed_embeddings: All seed embeddings.
        """
        cat_gen = gen_embeddings[gen_indices]
        cat_seed = seed_embeddings[seed_indices] if seed_indices else None

        n_gen = len(gen_indices)
        n_seeds = len(seed_indices) if seed_indices else 0

        # ---- Fit vMF to all examples in this category (seeds + generated) ----
        all_emb = (
            np.vstack([cat_seed, cat_gen])
            if cat_seed is not None and len(cat_seed) > 0
            else cat_gen
        )

        if len(
            all_emb
        ) < self.min_samples_for_vmf or not self._has_sufficient_variance(
            all_emb
        ):
            # Not enough data or variance to fit vMF for this category
            return None

        mu, kappa = self._fit_vmf(all_emb)

        # ---- Seed validity ----
        # Only compute seed validity score if seeds have labeled constituent values (i.e., cat_seed is not None and has at least one row)
        if cat_seed is not None and len(cat_seed) >= 1:
            seed_lls = self._vmf_log_likelihood(cat_seed, mu, kappa)
            seed_validity = float(seed_lls.mean())
            ll_threshold = float(
                np.percentile(seed_lls, self.seed_percentile_threshold)
            )
        else:
            # No labeled seeds for this category; skip seed validity score
            seed_validity = float("nan")
            ll_threshold = float("-inf")

        # ---- Generated log-likelihoods ----
        gen_lls = self._vmf_log_likelihood(cat_gen, mu, kappa)
        containment = float((gen_lls >= ll_threshold).mean())

        # ---- Novelty ----
        if cat_seed is not None and len(cat_seed) >= 1:
            novelty_min, novelty_avg = self._instance_novelty(cat_gen, cat_seed)
        else:
            novelty_min = np.full(n_gen, float("nan"))
            novelty_avg = np.full(n_gen, float("nan"))

        mean_novelty_min = float(np.nanmean(novelty_min))
        mean_novelty_avg = float(np.nanmean(novelty_avg))

        return {
            "label": label,
            "n_generated": n_gen,
            "n_seeds": n_seeds,
            "seed_validity_score": seed_validity,
            "novelty_min": mean_novelty_min,
            "novelty_avg": mean_novelty_avg,
            "containment_rate": containment,
            "extrapolation_quality": mean_novelty_min * containment,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate(
        self,
        generated_df: pd.DataFrame,
        constituent_cols: List[str],
        seed_df: Optional[pd.DataFrame] = None,
        seed_examples: Optional[Sequence[str]] = None,
        invalid_combinations: Optional[List[tuple]] = None,
        per_category: bool = False,
    ) -> Dict[str, Any]:
        """
        Run full realism validation.

        Args:
            generated_df:         DataFrame of synthetically generated examples.
                                  Must contain 'example' and each column in
                                  constituent_cols.
            seed_df:              Optional DataFrame of expert-authored seed
                                  examples. Must have the same schema as
                                  generated_df when provided.
            seed_examples:        Optional plain list/sequence of seed texts.
                                  Use this when seed labels are unavailable.
            constituent_cols:     Constituent label columns (e.g. ["topic", "actor"]).
            invalid_combinations: Optional list of constituent-value tuples (one value
                                  per column, in the same order as constituent_cols)
                                  that the domain expert has declared impossible/invalid.
                                  Any generated example matching one of these is flagged.

        Returns:
            RealismReport
        """
        required = set(constituent_cols) | {"example"}
        if seed_df is None and seed_examples is None:
            raise ValueError("Provide either seed_df or seed_examples.")

        for name, df in [("generated_df", generated_df)]:
            missing = [c for c in required if c not in df.columns]
            if missing:
                raise ValueError(f"{name} is missing columns: {missing}")

        seed_has_labels = seed_df is not None
        if seed_df is not None:
            missing = [c for c in required if c not in seed_df.columns]
            if missing:
                raise ValueError(f"seed_df is missing columns: {missing}")

        gen_texts = generated_df["example"].astype(str).tolist()
        seed_texts = (
            seed_df["example"].astype(str).tolist()
            if seed_df is not None
            else [
                str(text) for text in (seed_examples or []) if str(text).strip()
            ]
        )
        if not seed_texts:
            raise ValueError("At least one non-empty seed example is required.")

        gen_labels = make_composite_labels(generated_df, constituent_cols)
        seed_labels = (
            make_composite_labels(seed_df, constituent_cols)
            if seed_has_labels and seed_df is not None
            else None
        )

        # Reset cached embeddings
        self._gen_embeddings = None
        self._seed_embeddings = None

        all_texts = gen_texts + seed_texts
        print("Computing embeddings...")
        all_emb = self._normalize(self.model.encode(all_texts))
        n_gen = len(gen_texts)
        self._gen_embeddings = all_emb[:n_gen]
        self._seed_embeddings = all_emb[n_gen:]

        print("Computing style embeddings...")
        style_all_emb = self._normalize(self.style_model.encode(all_texts))
        style_gen_embeddings = style_all_emb[:n_gen]
        style_seed_embeddings = style_all_emb[n_gen:]

        # ----------------------------------------------------------------
        # Dataset vMF (all examples together)
        # ----------------------------------------------------------------
        print("Fitting dataset vMF...")
        if (
            self._has_sufficient_variance(all_emb)
            and len(all_emb) >= self.min_samples_for_vmf
        ):
            dataset_mu, dataset_kappa = self._fit_vmf(all_emb)
            seed_dataset_lls = self._vmf_log_likelihood(
                self._seed_embeddings, dataset_mu, dataset_kappa
            )
            dataset_ll_threshold = float(
                np.percentile(seed_dataset_lls, self.seed_percentile_threshold)
            )
            gen_dataset_lls = self._vmf_log_likelihood(
                self._gen_embeddings, dataset_mu, dataset_kappa
            )
        else:
            dataset_mu = None
            seed_dataset_lls = None
            dataset_ll_threshold = float("nan")
            gen_dataset_lls = np.full(n_gen, float("nan"))

        # ----------------------------------------------------------------
        # Instance-level stats
        # ----------------------------------------------------------------
        print("Computing instance-level novelty...")
        gen_novelty_min, gen_novelty_avg = self._instance_novelty(
            self._gen_embeddings, self._seed_embeddings
        )

        instance_stats: List[Dict[str, Any]] = []
        out_of_scope: List[int] = []

        for i in range(n_gen):
            in_scope = (
                bool(gen_dataset_lls[i] >= dataset_ll_threshold)
                if np.isfinite(gen_dataset_lls[i])
                and np.isfinite(dataset_ll_threshold)
                else False
            )
            if not in_scope:
                out_of_scope.append(i)

            instance_stats.append(
                {
                    "idx": i,
                    "label": gen_labels[i],
                    "novelty_min": float(gen_novelty_min[i]),
                    "novelty_avg": float(gen_novelty_avg[i]),
                    "log_likelihood": float(gen_dataset_lls[i]),
                    "is_in_scope": in_scope,
                }
            )

        # ----------------------------------------------------------------
        # Per-category stats (optional)
        # ----------------------------------------------------------------
        category_stats: Dict[str, Dict[str, Any]] = {}
        if per_category and seed_has_labels and seed_labels is not None:
            print("Computing per-category realism...")
            # Build label → index maps (fine-grained only for realism)
            gen_label_to_idx: Dict[str, List[int]] = defaultdict(list)
            for i, lbl in enumerate(gen_labels):
                gen_label_to_idx[lbl].append(i)

            seed_label_to_idx: Dict[str, List[int]] = defaultdict(list)
            for i, lbl in enumerate(seed_labels):
                seed_label_to_idx[lbl].append(i)

            all_category_labels = set(gen_label_to_idx) | set(seed_label_to_idx)

            for label in all_category_labels:
                gen_idx = gen_label_to_idx.get(label, [])
                seed_idx = seed_label_to_idx.get(label, [])
                if not gen_idx:
                    continue  # nothing generated for this label
                stats = self._compute_category_stats(
                    label,
                    gen_idx,
                    seed_idx,
                    self._gen_embeddings,
                    self._seed_embeddings,
                )
                if stats:
                    category_stats[label] = stats

        # ----------------------------------------------------------------
        # Distribution-level: calibrated Sinkhorn alignment
        # ----------------------------------------------------------------
        print("Computing distribution-level realism...")
        alignment = self._calibrated_sinkhorn_alignment(
            self._seed_embeddings, self._gen_embeddings
        )

        # ----------------------------------------------------------------
        # Invalid-combination audit
        # ----------------------------------------------------------------
        invalid_combo_coverage: Dict[tuple, int] = {}
        if invalid_combinations:
            for combo in invalid_combinations:
                count = 0
                for _, row in generated_df.iterrows():
                    row_combo = tuple(
                        value_to_str(row[c]) for c in constituent_cols
                    )
                    if row_combo == combo:
                        count += 1
                invalid_combo_coverage[combo] = count

        # ----------------------------------------------------------------
        # Dataset summary scores
        # ----------------------------------------------------------------
        dataset_containment = float(
            np.mean([s["is_in_scope"] for s in instance_stats])
        )
        mean_dataset_novelty_min = float(np.nanmean(gen_novelty_min))
        mean_dataset_novelty_avg = float(np.nanmean(gen_novelty_avg))
        dataset_extrapolation_quality = (
            mean_dataset_novelty_min * dataset_containment
        )
        style_dist = self._cosine_distance_matrix(
            style_gen_embeddings, style_seed_embeddings
        )
        mean_style_distance_avg = float(np.nanmean(style_dist.mean(axis=1)))
        style_realism_score = 1.0 / (1.0 + mean_style_distance_avg)

        # Attach dataset novelty metrics to the report object for printing
        self._dataset_novelty_min = mean_dataset_novelty_min
        self._dataset_novelty_avg = mean_dataset_novelty_avg
        self._seed_validity_score = (
            float(np.nanmean(seed_dataset_lls))
            if seed_dataset_lls is not None
            else None
        )

        return {
            "instance_stats": instance_stats,
            "category_stats": category_stats,
            "raw_sinkhorn_distance": alignment["raw_sinkhorn_distance"],
            "calibrated_sinkhorn_distance": alignment[
                "calibrated_sinkhorn_distance"
            ],
            "calibrated_sinkhorn_distance_ci_005": alignment[
                "calibrated_sinkhorn_distance_ci_005"
            ],
            "calibrated_sinkhorn_distance_ci_095": alignment[
                "calibrated_sinkhorn_distance_ci_095"
            ],
            "invalid_combo_coverage": invalid_combo_coverage,
            "dataset_containment_rate": dataset_containment,
            "dataset_extrapolation_quality": dataset_extrapolation_quality,
            "out_of_scope_indices": out_of_scope,
            "seed_has_labels": seed_has_labels,
            "seed_validity_score": self._seed_validity_score,
            "mean_dataset_novelty_min": mean_dataset_novelty_min,
            "mean_dataset_novelty_avg": mean_dataset_novelty_avg,
            "mean_style_distance_avg": mean_style_distance_avg,
            "style_realism_score": style_realism_score,
        }

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def print_report(self, report: Dict[str, Any]) -> None:
        print("\n" + "=" * 70)
        print("REALISM VALIDATION REPORT")
        print("=" * 70)

        # ---- Linguistic Realism ----
        novelty_min = getattr(self, "_dataset_novelty_min", float("nan"))
        novelty_avg = getattr(self, "_dataset_novelty_avg", float("nan"))
        containment = report["dataset_containment_rate"]
        # Combine: mean of normalized metrics (all in [0,1])
        # Novelty is a distance, so we invert it for realism (1-novelty)
        linguistic_realism = np.nanmean(
            [1 - novelty_min, 1 - novelty_avg, containment]
        )
        print("\n--- Linguistic Realism ---")
        print(f"  Containment rate:            {containment:.2%}")
        print(f"  Novelty (min):               {novelty_min:.4f}")
        print(f"  Novelty (avg):               {novelty_avg:.4f}")
        print(f"  Linguistic realism score:    {linguistic_realism:.4f}")

        if report.get("seed_has_labels"):
            seed_validity = report.get("seed_validity_score")
            print("\n--- Seed Validity ---")
            if seed_validity is None or not np.isfinite(seed_validity):
                print("  Seed validity score:         N/A")
            else:
                print(f"  Seed validity score:         {seed_validity:.4f}")

        # ---- Distributional Realism ----
        sinkhorn_raw = report["raw_sinkhorn_distance"]
        alignment_score = report["calibrated_sinkhorn_distance"]
        align_ci_low = report["calibrated_sinkhorn_distance_ci_005"]
        align_ci_high = report["calibrated_sinkhorn_distance_ci_095"]
        distributional_realism = alignment_score
        print("\n--- Distributional Realism ---")
        print(f"  Sinkhorn distance:           {sinkhorn_raw:.6f}")
        print(f"  Calibrated Sinkhorn distance: {alignment_score:.4f}")
        print(
            f"  ci_005 / ci_095:             [{align_ci_low:.4f}, {align_ci_high:.4f}]"
        )
        print(f"  Distributional realism score:{distributional_realism:.4f}")

        # ---- Extrapolation Quality ----
        extrapolation_quality = report["dataset_extrapolation_quality"]
        print("\n--- Extrapolation Quality ---")
        print(f"  Extrapolation quality:       {extrapolation_quality:.4f}")

        # ---- Combined Summary ----
        combined_score = np.nanmean(
            [linguistic_realism, distributional_realism, extrapolation_quality]
        )
        print("\n--- Overall Realism Summary ---")
        print(f"  Combined realism score:      {combined_score:.4f}")

        # ---- Out-of-scope examples ----
        print(
            f"\n  Out-of-scope examples:       {len(report['out_of_scope_indices'])}"
        )
        if report["out_of_scope_indices"]:
            print(
                f"  OOS indices (first 10):      {report['out_of_scope_indices'][:10]}"
            )

        # ---- Per-category ----
        if report["category_stats"]:
            print("\n--- Per-Category Extrapolation Scores ---")
            header = (
                f"{'Category':<35} {'Gen':>5} {'Seeds':>6} "
                f"{'SeedVal':>8} {'NovMin':>8} {'NovAvg':>8} "
                f"{'Contain':>8} {'ExtQual':>8}"
            )
            print(header)
            print("-" * len(header))
            for label, st in sorted(report["category_stats"].items()):
                sv = (
                    f"{st['seed_validity_score']:.2f}"
                    if np.isfinite(st["seed_validity_score"])
                    else "  N/A"
                )
                print(
                    f"{label:<35} {st['n_generated']:>5} {st['n_seeds']:>6} "
                    f"{sv:>8} {st['novelty_min']:>8.4f} {st['novelty_avg']:>8.4f} "
                    f"{st['containment_rate']:>8.2%} {st['extrapolation_quality']:>8.4f}"
                )

        # ---- Invalid combination audit ----
        if report["invalid_combo_coverage"] is not None:
            print("\n--- Invalid Combination Audit ---")
            if not report["invalid_combo_coverage"]:
                print("  No invalid combinations declared.")
            else:
                all_clean = all(
                    v == 0 for v in report["invalid_combo_coverage"].values()
                )
                for combo, cnt in report["invalid_combo_coverage"].items():
                    flag = "" if cnt == 0 else f"  ✗ VIOLATION ({cnt} examples)"
                    print(f"  {combo}: {cnt} generated examples{flag}")
                if all_clean:
                    print(
                        "  ✓ No invalid combinations found in generated data."
                    )

        print("\n" + "=" * 70)
