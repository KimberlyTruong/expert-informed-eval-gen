"""
Coverage Validation

Coverage as a concept is easy to measure at a superficial level by assuming the model is 100% correct.
This is however not something we can trust without validation, so we must ensure items said to contain
each constituent group are most similar to other items within that group. Otherwise, checking combinatorial coverage
would not be valid.

We validate this by fitting a distribution to the items claimed to belong to each subgroup and confirming the items are sufficiently close to each other.
We can do so by determining the center which most items in the subgroup are most similar with, then get the distribution representing items from the center.
We fit a von Mises-Fisher (vMF) distribution to model the density of words over a unit sphere distribution. This considers semantic consistency over
directional metrics (not considered with Gaussian distributions) which then allows us to look at items at the tails and center for validation. If there is
too much spread for a certain group relative to others, it is likely that group is too broad or the model is incorrectly generating examples for that group.

ref for vMF: https://pmc.ncbi.nlm.nih.gov/articles/PMC6327958/
             https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.vonmises_fisher.html

Components:
1. Combinatorial completeness (i.e., are all single-value combinations from the schema represented even at a superficial level?)
2. vMF-based internal coherence per category (i.e., how similar/well-concentrated the examples are to others in the same category)
3. Hierarchical coherence for multi-value examples (i.e., samples which cover more than one group from a constituent part)
4. Per-subgroup diversity
"""

import warnings

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Tuple, Optional, Set
from collections import defaultdict
from itertools import product

from sentence_transformers import SentenceTransformer
from scipy.stats import vonmises_fisher

from operationalized_metrics.helpers import (
    cosine_similarity_matrix,
    has_sufficient_variance,
    l2_normalize,
    value_to_str,
)


def compute_row_distance_metrics(
    df: pd.DataFrame,
    constituent_cols: List[str],
    embeddings: np.ndarray,
    category_centers: Dict[str, np.ndarray],
) -> pd.DataFrame:
    """Compute per-row centroid and dataset-center cosine distances.

    Centroid selection per row:
    1. Use the composite-label centroid if available in category_centers.
    2. Otherwise, average constituent-level centroids.
       For multi-value cells (semicolon-delimited), average value centroids
       first within each constituent, then average across constituents.
    """

    def _split_multi_value_cell(value: object) -> List[str]:
        text = value_to_str(value).strip()
        if not text or text == "NA":
            return ["NA"]
        parts = [part.strip() for part in text.split(";")]
        cleaned = [part for part in parts if part]
        return cleaned or ["NA"]

    def _make_composite_label(row: pd.Series) -> str:
        return "+".join(
            f"{col}={value_to_str(row[col])}" for col in constituent_cols
        )

    def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
        a_norm = l2_normalize(np.asarray(a, dtype=np.float64).reshape(1, -1))[0]
        b_norm = l2_normalize(np.asarray(b, dtype=np.float64).reshape(1, -1))[0]
        return float(1.0 - np.clip(np.dot(a_norm, b_norm), -1.0, 1.0))

    if embeddings.ndim != 2:
        raise ValueError("embeddings must be a 2D array")

    # Build constituent-value (atomic) centroids for fallback behavior.
    atomic_buckets: Dict[Tuple[str, str], List[int]] = {}
    for idx, row in df.iterrows():
        for col in constituent_cols:
            for atomic_value in _split_multi_value_cell(row[col]):
                atomic_buckets.setdefault((col, atomic_value), []).append(idx)

    atomic_centroids: Dict[Tuple[str, str], np.ndarray] = {}
    for key, indices in atomic_buckets.items():
        centroid = embeddings[indices].mean(axis=0)
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid = centroid / norm
        atomic_centroids[key] = centroid

    dataset_center = embeddings.mean(axis=0)
    ds_norm = np.linalg.norm(dataset_center)
    if ds_norm > 0:
        dataset_center = dataset_center / ds_norm

    centroid_dists: List[float] = []
    dataset_center_dists: List[float] = []

    for idx, row in df.iterrows():
        row_embedding = embeddings[idx]
        composite_label = _make_composite_label(row)

        if composite_label in category_centers:
            row_centroid = category_centers[composite_label]
        else:
            per_constituent_centroids: List[np.ndarray] = []
            for col in constituent_cols:
                value_centroids = [
                    atomic_centroids[(col, atomic_value)]
                    for atomic_value in _split_multi_value_cell(row[col])
                    if (col, atomic_value) in atomic_centroids
                ]
                if value_centroids:
                    c = np.mean(value_centroids, axis=0)
                    c_norm = np.linalg.norm(c)
                    if c_norm > 0:
                        c = c / c_norm
                    per_constituent_centroids.append(c)

            if per_constituent_centroids:
                row_centroid = np.mean(per_constituent_centroids, axis=0)
                rc_norm = np.linalg.norm(row_centroid)
                if rc_norm > 0:
                    row_centroid = row_centroid / rc_norm
            else:
                row_centroid = l2_normalize(
                    np.asarray(row_embedding, dtype=np.float64).reshape(1, -1)
                )[0]

        centroid_dists.append(_cosine_distance(row_embedding, row_centroid))
        dataset_center_dists.append(
            _cosine_distance(row_embedding, dataset_center)
        )

    return pd.DataFrame(
        {
            "centroid_dist": centroid_dists,
            "dataset_center_dist": dataset_center_dists,
        },
        index=df.index,
    )


def _is_placeholder_value(value: object) -> bool:
    if pd.isna(value):
        return True
    text = str(value).strip().lower()
    return text in {"optional", "none", "", "na"}


class CoverageValidator:
    """
    Validates coverage of synthetic evaluation datasets using vMF distributions
    and hierarchical membership checks.
    """

    def __init__(
        self,
        embedding_model: str = "all-mpnet-base-v2",
        similarity_threshold: float = 0.7,
        redundancy_threshold: float = 0.95,
        min_samples_for_stats: int = 3,
    ):
        """
        Initialize the validator.

        Args:
            embedding_model: Sentence transformer model name
            similarity_threshold: Minimum cosine similarity to category center
                                  for an example to be considered valid
            redundancy_threshold: Similarity above which two examples are
                                  considered redundant
            min_samples_for_stats: Minimum examples needed to compute category stats
        """
        self.model = SentenceTransformer(embedding_model)
        self.similarity_threshold = similarity_threshold
        self.redundancy_threshold = redundancy_threshold
        self.min_samples_for_stats = min_samples_for_stats

        # Will be populated during validation
        self.embeddings: Optional[np.ndarray] = None
        self.category_centers: Dict[str, np.ndarray] = {}
        self.category_concentrations: Dict[str, float] = {}

    def _normalize(self, vectors: np.ndarray) -> np.ndarray:
        """L2 normalize vectors to unit hypersphere."""
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
        """Return True when embeddings are not collapsed to an almost single point. This is to prevent kappa from blowing up from embeddings being near-identical"""
        return has_sufficient_variance(embeddings)

    def _estimate_vmf_parameters(
        self, embeddings: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """
        Estimate vMF parameters (mean direction μ, concentration κ) from samples
        using scipy's vonmises_fisher.fit().

        Args:
            embeddings: (n, d) array of L2-normalized embeddings

        Returns:
            mean_direction: (d,) unit vector
            kappa: concentration parameter (higher = tighter cluster)
        """
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("error", category=RuntimeWarning)
                mu, kappa = vonmises_fisher.fit(embeddings)
        except RuntimeWarning:
            mu, kappa = self._approximate_vmf_parameters(embeddings)
        if not np.isfinite(kappa) or not np.all(np.isfinite(mu)):
            mu, kappa = self._approximate_vmf_parameters(embeddings)
        mu = self._safe_unit_vector(mu)
        return mu, float(kappa)

    def _approximate_vmf_parameters(
        self, embeddings: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """Fallback closed-form approximation when SciPy fit becomes unstable."""
        mean_vec = embeddings.mean(axis=0)
        mean_direction = self._safe_unit_vector(mean_vec)
        norm = float(np.linalg.norm(mean_vec))
        r_bar = norm
        # Avoid hitting the singularities at 0 or 1
        r_bar = float(np.clip(r_bar, 1e-6, 1 - 1e-6))
        dim = embeddings.shape[1]
        kappa = (r_bar * (dim - r_bar**2)) / (1 - r_bar**2)
        return mean_direction, float(max(kappa, 1e-3))

    def _vmf_log_likelihood(
        self, embeddings: np.ndarray, mu: np.ndarray, kappa: float
    ) -> np.ndarray:
        """
        Compute log-likelihood of embeddings under a vMF distribution.

        Args:
            embeddings: (n, d) array of L2-normalized embeddings
            mu: mean direction
            kappa: concentration parameter

        Returns:
            (n,) array of log-likelihoods
        """
        return vonmises_fisher.logpdf(embeddings, mu=mu, kappa=kappa)

    def _pairwise_similarities(self, embeddings: np.ndarray) -> np.ndarray:
        """Compute pairwise cosine similarities for a set of embeddings."""
        return cosine_similarity_matrix(embeddings)

    def check_combinatorial_completeness(
        self,
        examples: List[Dict],
        schema_dimensions: Dict[str, List[str]],
        min_count: int = 1,
    ) -> Tuple[Set[tuple], Set[tuple], Set[tuple], float]:
        """
        Check if all single-value (atomic) combinations are represented.

        Semicolon-delimited multi-value cells are expanded into atomic values.
        A row that has a multi-value cell contributes to multiple observed
        combinations via a Cartesian product across its atomic values.

        Args:
            examples: List of example dicts, each with keys matching schema dimensions
            schema_dimensions: Dict mapping dimension names to lists of possible values
                              e.g., {"topic": ["corruption", "policy"],
                                     "actor": ["politician_a", "politician_b"]}

            min_count: Minimum number of times a combination must appear to be considered covered

        Returns:
            expected: Set of all expected combinations
            observed: Set of combinations found in examples at least min_count times
            missing: Set of missing combinations (not meeting min_count)
            ratio: Completeness ratio (observed/expected)
        """

        def _atomic_values(value: Any) -> List[str]:
            if pd.isna(value):
                return []
            text = value_to_str(value).strip()
            if not text or text == "NA":
                return []
            parts = [part.strip() for part in text.split(";")]
            cleaned = [part for part in parts if part]
            return cleaned

        # Generate all expected combinations from atomic schema values.
        dimension_names = list(schema_dimensions.keys())
        dimension_values = []
        for dim in dimension_names:
            atomic_dim_values = sorted(
                {
                    atomic
                    for raw_value in schema_dimensions[dim]
                    for atomic in _atomic_values(raw_value)
                }
            )
            dimension_values.append(atomic_dim_values)
        expected = set(product(*dimension_values))

        # Count occurrences of each combination
        combo_counts = {}
        for ex in examples:
            per_dim_atomic_values = [
                _atomic_values(ex.get(dim)) for dim in dimension_names
            ]
            if any(not values for values in per_dim_atomic_values):
                continue
            for combo in product(*per_dim_atomic_values):
                combo_counts[tuple(combo)] = (
                    combo_counts.get(tuple(combo), 0) + 1
                )

        observed = {
            combo for combo, count in combo_counts.items() if count >= min_count
        }
        missing = expected - observed
        ratio = len(observed) / len(expected) if expected else 1.0

        return expected, observed, missing, ratio

    def compute_category_stats(
        self, texts: List[str], labels: List[str], category_name: str
    ) -> Optional[Dict[str, Any]]:
        """
        Compute vMF-based statistics for a single category.

        Args:
            texts: All texts in the dataset
            labels: Category label for each text (can be multi-value separated by "+")
            category_name: The category to analyze

        Returns:
            CategoryStats or None if insufficient samples
        """
        # Composite labels (A+B+...) must match exactly; constituents match by membership.
        if "+" in category_name:
            indices = [
                i for i, label in enumerate(labels) if label == category_name
            ]
        else:
            indices = [
                i
                for i, label in enumerate(labels)
                if category_name in label.split("+")
            ]

        if len(indices) < self.min_samples_for_stats:
            return None

        # Get embeddings for these examples
        if self.embeddings is None:
            self.embeddings = self._normalize(self.model.encode(texts))

        category_embeddings = self.embeddings[indices]
        if not np.all(np.isfinite(category_embeddings)):
            warnings.warn(
                f"Skipping category '{category_name}' due to non-finite embeddings.",
                RuntimeWarning,
            )
            return None
        if not self._has_sufficient_variance(category_embeddings):
            warnings.warn(
                f"Skipping category '{category_name}' due to insufficient variance.",
                RuntimeWarning,
            )
            return None

        # Estimate vMF parameters using scipy
        mean_direction, kappa = self._estimate_vmf_parameters(
            category_embeddings
        )

        # Store for later use
        self.category_centers[category_name] = mean_direction
        self.category_concentrations[category_name] = kappa

        # Compute log-likelihoods for outlier detection
        log_likelihoods = self._vmf_log_likelihood(
            category_embeddings, mean_direction, kappa
        )

        # Find examples with low log-likelihood (potential outliers)
        # Use percentile-based threshold
        ll_threshold = np.percentile(log_likelihoods, 10)  # bottom 10%
        outlier_indices = [
            indices[i]
            for i, ll in enumerate(log_likelihoods)
            if ll < ll_threshold
        ]

        # Compute pairwise diversity
        pairwise_sim = self._pairwise_similarities(category_embeddings)
        # Get upper triangle (excluding diagonal)
        upper_tri = pairwise_sim[np.triu_indices(len(indices), k=1)]

        return {
            "name": category_name,
            "count": len(indices),
            "mean_direction": mean_direction,
            "concentration_kappa": kappa,
            "mean_pairwise_similarity": (
                float(upper_tri.mean()) if len(upper_tri) > 0 else 1.0
            ),
            "min_pairwise_similarity": (
                float(upper_tri.min()) if len(upper_tri) > 0 else 1.0
            ),
            "examples_below_threshold": outlier_indices,
        }

    def check_hierarchical_coherence(
        self, texts: List[str], labels: List[str], idx: int
    ) -> Dict[str, Any]:
        """
        Check hierarchical coherence for a multi-value example.

        Verifies that:
        1. Fine-grained: Example has high likelihood under its A+B distribution
        2. Coarse-grained: Example has acceptable likelihood under each constituent (A and B)

        Args:
            texts: All texts
            labels: All labels
            idx: Index of the example to check

        Returns:
            HierarchicalCheckResult with likelihood scores and flags
        """
        if self.embeddings is None:
            self.embeddings = self._normalize(self.model.encode(texts))

        example_embedding = self.embeddings[
            idx : idx + 1
        ]  # keep 2D for vonmises_fisher
        label = labels[idx]
        constituents = label.split("+")

        flags = []

        # Fine-grained check: log-likelihood under own combination's distribution
        fine_grained_ll = float("-inf")
        if (
            label in self.category_centers
            and label in self.category_concentrations
        ):
            fine_grained_ll = float(
                self._vmf_log_likelihood(
                    example_embedding,
                    self.category_centers[label],
                    self.category_concentrations[label],
                )[0]
            )

        # Coarse-grained checks: log-likelihood under each constituent's distribution
        coarse_log_likelihoods = {}
        for constituent in constituents:
            if (
                constituent in self.category_centers
                and constituent in self.category_concentrations
            ):
                ll = float(
                    self._vmf_log_likelihood(
                        example_embedding,
                        self.category_centers[constituent],
                        self.category_concentrations[constituent],
                    )[0]
                )
                coarse_log_likelihoods[constituent] = ll

                # Flag if log-likelihood is very low (use a relative threshold)
                # Compare to typical log-likelihood for this category
                kappa = self.category_concentrations[constituent]
                # Expected log-likelihood at the mean is roughly kappa (ignoring normalization)
                # Flag if much lower than expected
                if ll < -kappa:  # heuristic threshold
                    flags.append(
                        f"Low likelihood under constituent '{constituent}': {ll:.2f}"
                    )

        return {
            "example_idx": idx,
            "fine_grained_label": label,
            "fine_grained_similarity": fine_grained_ll,
            "coarse_similarities": coarse_log_likelihoods,
            "flags": flags,
        }

    def check_diversity(  # TODO: change diversity to topic diversity. Right now it is using cosine similarity
        self, texts: List[str], labels: List[str]
    ) -> Tuple[float, List[Tuple[int, int]]]:
        """
        Check within-category diversity and identify redundant pairs.

        Args:
            texts: All texts
            labels: All labels

        Returns:
            diversity_score: Overall diversity (1 - mean pairwise similarity)
            redundant_pairs: List of (idx1, idx2) pairs with similarity > threshold
        """
        if self.embeddings is None:
            self.embeddings = self._normalize(self.model.encode(texts))

        # Group by label
        label_to_indices = defaultdict(list)
        for i, label in enumerate(labels):
            label_to_indices[label].append(i)

        all_pairwise_sims = []
        redundant_pairs = []

        for label, indices in label_to_indices.items():
            if len(indices) < 2:
                continue

            category_embeddings = self.embeddings[indices]
            pairwise_sim = self._pairwise_similarities(category_embeddings)

            # Check each pair
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    sim = pairwise_sim[i, j]
                    all_pairwise_sims.append(sim)

                    if sim > self.redundancy_threshold:
                        redundant_pairs.append((indices[i], indices[j]))

        diversity_score = (
            1 - np.mean(all_pairwise_sims) if all_pairwise_sims else 0.0
        )

        return diversity_score, redundant_pairs

    def validate(
        self,
        df: pd.DataFrame,
        constituent_cols: List[str],
        required_constituent_cols: Optional[List[str]] = None,
        optional_constituent_cols: Optional[List[str]] = None,
        schema_dimensions: Optional[Dict[str, List[str]]] = None,
    ) -> Dict[str, Any]:
        """
        Run full coverage validation from a dataframe of synthetic examples.

        Args:
            df: pandas DataFrame containing the prevalidated "example" text
                column and constituent columns. The runner is responsible for
                ensuring any schema-derived columns are present or filled.
            constituent_cols: List of column names describing constituent labels
                (e.g., topic, actor). Coverage is evaluated across combinations
                of these columns plus the intended deployment population.

        Returns:
            CoverageReport with all validation results
        """
        texts = df["example"].astype(str).tolist()
        label_source_cols = constituent_cols + [
            "intended_deployment_population"
        ]
        label_source_cols = [
            col for col in label_source_cols if col in df.columns
        ]
        required_set = set(required_constituent_cols or constituent_cols)
        optional_set = set(optional_constituent_cols or [])

        coverage_cols = [
            col for col in label_source_cols if col in required_set
        ]

        schema_dimensions = schema_dimensions or {}
        examples_metadata: List[Dict[str, str]] = []
        if coverage_cols:
            examples_metadata = [
                {
                    col: value_to_str(row[col])
                    for col in coverage_cols
                    if not _is_placeholder_value(row[col])
                }
                for _, row in df.iterrows()
            ]

        if not schema_dimensions and coverage_cols:
            schema_dimensions = {
                col: sorted(
                    {
                        value_to_str(value)
                        for value in df[col].unique()
                        if not _is_placeholder_value(value)
                    }
                )
                for col in coverage_cols
            }

        labels: List[str] = []
        label_frame = df[label_source_cols]
        for row in label_frame.itertuples(index=False, name=None):
            parts = [
                f"{col}={value_to_str(value)}"
                for col, value in zip(label_source_cols, row)
            ]
            labels.append("+".join(parts))

        # Reset cached embeddings
        self.embeddings = None
        self.category_centers = {}
        self.category_concentrations = {}

        # Compute embeddings once
        print("Computing embeddings...")
        self.embeddings = self._normalize(self.model.encode(texts))

        # 1. Combinatorial completeness (if schema provided)
        if schema_dimensions and examples_metadata:
            expected, observed, missing, ratio = (
                self.check_combinatorial_completeness(
                    examples_metadata,
                    schema_dimensions,
                    self.min_samples_for_stats,
                )
            )
        else:
            expected, observed, missing, ratio = set(), set(), set(), 1.0

        # 2. Get all unique categories (both fine-grained and coarse)
        all_categories = set()
        for label in labels:
            all_categories.add(label)  # fine-grained (e.g., "A+B")
            for constituent in label.split("+"):
                all_categories.add(constituent)  # coarse (e.g., "A", "B")

        # 3. Compute stats for each category
        print("Computing category statistics...")
        category_stats = {}
        for category in all_categories:
            stats = self.compute_category_stats(texts, labels, category)
            if stats:
                category_stats[category] = stats

        # 4. Hierarchical coherence for multi-value examples
        print("Checking hierarchical coherence...")
        hierarchical_results = []
        for i, label in enumerate(labels):
            if "+" in label:  # multi-value example
                result = self.check_hierarchical_coherence(texts, labels, i)
                hierarchical_results.append(result)

        # 5. Diversity check
        print("Checking diversity...")
        diversity_score, redundant_pairs = self.check_diversity(texts, labels)

        # 6. Optional column coverage, summarized as a single holistic score.
        optional_column_scores = []
        for col in optional_set:
            if col not in df.columns:
                continue
            optional_column_scores.append(
                float((~df[col].map(_is_placeholder_value)).mean())
            )
        holistic_completeness_score = (
            float(np.mean(optional_column_scores))
            if optional_column_scores
            else 1.0
        )

        return {
            "expected_combinations": expected,
            "observed_combinations": observed,
            "missing_combinations": missing,
            "completeness_ratio": ratio,
            "holistic_completeness_score": holistic_completeness_score,
            "category_stats": category_stats,
            "hierarchical_results": hierarchical_results,
            "overall_diversity_score": diversity_score,
            "redundant_pairs": redundant_pairs,
        }

    def print_report(self, report: Dict[str, Any]):
        print("\n" + "=" * 60)
        print("COVERAGE VALIDATION REPORT")
        print("=" * 60)

        # Combinatorial completeness
        print("\n--- Combinatorial Completeness ---")
        print(f"Expected combinations: {len(report['expected_combinations'])}")
        print(f"Observed combinations: {len(report['observed_combinations'])}")
        print(f"Missing combinations: {len(report['missing_combinations'])}")
        print(f"Completeness ratio: {report['completeness_ratio']:.2%}")
        if report["missing_combinations"]:
            print(f"Missing: {list(report['missing_combinations'])[:5]}...")

        # Category coherence
        print("\n--- Group Coherence (vMF) ---")
        print(
            f"{'Group':<30} {'Count':>6} {'κ':>10} {'Mean Sim':>10} {'Outliers':>10}"
        )
        print("-" * 70)
        for name, stats in sorted(report["category_stats"].items()):
            kappa_str = (
                f"{stats['concentration_kappa']:.2f}"
                if stats["concentration_kappa"] < float("inf")
                else "∞"
            )
            print(
                f"{name:<30} {stats['count']:>6} {kappa_str:>10} "
                f"{stats['mean_pairwise_similarity']:>10.3f} {len(stats['examples_below_threshold']):>10}"
            )

        print(
            "\n--- Coherence Issues ---"
        )  # how coherent/well-concentrated are the examples within this group?
        flagged = [r for r in report["hierarchical_results"] if r["flags"]]
        if flagged:
            for result in flagged[:10]:  # show first 10
                print(
                    f"Example {result['example_idx']} ({result['fine_grained_label']}):"
                )
                print(
                    f"  Fine-grained log-likelihood: {result['fine_grained_similarity']:.2f}"
                )
                for flag in result["flags"]:
                    print(f"  - {flag}")
        else:
            print("No coherence issues detected.")

        # Diversity
        print("\n--- Diversity ---")
        print(
            f"Overall diversity score: {report['overall_diversity_score']:.3f}"
        )
        print(
            f"Redundant pairs (>{self.redundancy_threshold:.0%} similarity): {len(report['redundant_pairs'])}"
        )
        if report["redundant_pairs"]:
            print(f"First few redundant pairs: {report['redundant_pairs'][:5]}")

        print("\n" + "=" * 60)
