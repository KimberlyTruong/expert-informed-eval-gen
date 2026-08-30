#!/usr/bin/env python3
"""
Run dataset quality metrics from a schema JSON and one or more generated CSV files
with example_text and columns for each constituent.

This script runs the operationalized coverage, diversity, and realism
validators, then adds row-level metrics used for dataset inspection and
downstream curation. All metric defintions are under /operationalized_metrics/metric_definitions.md.

The validator implementations and shared metric helpers are in
``scripts/operationalized_metrics/``.

Evaluating a single dataset requires:
    - ``--schema``: schema JSON path
    - ``--dataset``: generated dataset CSV path

Evaluating all .csv files in a directory requires:
    - ``--input-dir``: directory containing the CSV files
    - ``--pattern``: pattern to match the CSV files
    - ``--recursive``: whether to search recursively in subdirectories

Use ``--not-realistic`` with a JSON path or JSON list to exclude invalid or
unrealistic constituent combinations from the constrained coverage metrics.

Optional outputs default to files beside the dataset:
  - <dataset_stem>_metrics.json
  - <dataset_stem>_metrics.csv
  - <dataset_stem>_coverage_vary.pdf

Examples:
    python scripts/metrics.py --schema path/to/schema.json \
            --dataset path/to/generated.csv
    python scripts/metrics.py --schema path/to/schema.json \
            --input-dir path/to/generated --pattern '*_labeled.csv' --recursive
"""

from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import re

from operationalized_metrics.coverage import (
    CoverageValidator,
    compute_row_distance_metrics,
)
from operationalized_metrics.diversity import DiversityValidator
from operationalized_metrics.realism import (
    RealismValidator,
    serialize_realism_report,
)
from operationalized_metrics.helpers import value_to_str

TEXT_COLUMN_CANDIDATES = [
    "example",
    "example text",
    "example_text",
    "text",
    "prompt",
]


def _parse_seed_subset_indices(raw: Optional[str]) -> Optional[List[int]]:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    parts = [part.strip() for part in text.split(",")]
    indices: List[int] = []
    for part in parts:
        if not part:
            continue
        if not part.lstrip("-").isdigit():
            raise ValueError(
                "Seed subset indices must be integers. " f"Got: {part!r}"
            )
        idx = int(part)
        if idx < 0:
            raise ValueError(
                "Seed subset indices must be non-negative. " f"Got: {idx}"
            )
        indices.append(idx)
    return indices or None


def _parse_not_realistic(raw: Optional[str]) -> Optional[List[List[str]]]:
    if raw is None:
        return None

    text = str(raw).strip()
    if not text:
        return None

    maybe_path = Path(text).expanduser()
    if maybe_path.exists() and maybe_path.is_file():
        parsed = json.loads(maybe_path.read_text(encoding="utf-8"))
    else:
        parsed = json.loads(text)

    if not isinstance(parsed, list):
        raise ValueError("not_realistic must be a JSON list of lists.")

    combos: List[List[str]] = []
    for item in parsed:
        if not isinstance(item, list):
            raise ValueError(
                "Each not_realistic entry must be a list of category values."
            )
        combos.append(
            [str(value).strip() for value in item if str(value).strip()]
        )

    return combos or None


# Helper to round to n significant figures
def round_sigfigs(x, sig=2):
    if x == 0 or not np.isfinite(x):
        return float(x)
    return float(f"{x:.{sig}g}")


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return round_sigfigs(float(value), 2)
    if isinstance(value, (set, frozenset, tuple)):
        return list(value)
    raise TypeError(
        f"Object of type {type(value).__name__} is not JSON serializable"
    )


# Recursively round all floats in a data structure to 2 significant figures
def _round_floats(obj):
    if isinstance(obj, dict):
        return {k: _round_floats(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_round_floats(v) for v in obj]
    elif isinstance(obj, float):
        return round_sigfigs(obj, 2)
    elif isinstance(obj, np.floating):
        return round_sigfigs(float(obj), 2)
    else:
        return obj


def _safe_json_dump(path: Path, data: Dict[str, Any]) -> None:
    rounded_data = _round_floats(data)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(
            rounded_data,
            handle,
            indent=2,
            ensure_ascii=False,
            default=_json_default,
        )


def _normalize_column_name(name: str) -> str:
    return str(name).strip()


def _find_text_column(columns: Sequence[str]) -> str:
    normalized = {_normalize_column_name(col): col for col in columns}
    for candidate in TEXT_COLUMN_CANDIDATES:
        if candidate in normalized:
            return normalized[candidate]
    raise ValueError(
        f"Could not find a text column. Expected one of: {TEXT_COLUMN_CANDIDATES}"
    )


def _flatten_constituent_columns(constituents: Dict[str, Any]) -> List[str]:
    columns: List[str] = []
    for name, spec in constituents.items():
        if isinstance(spec, dict) and "subConstituents" in spec:
            columns.extend(
                _flatten_constituent_columns(spec["subConstituents"])
            )
        else:
            columns.append(str(name))
    return columns


def _schema_constituent_columns_by_requirement(
    schema: Dict[str, Any],
) -> Tuple[List[str], List[str]]:
    """Return leaf constituent columns split by schema required flag."""

    def _collect(
        constituents: Dict[str, Any],
    ) -> Tuple[List[str], List[str]]:
        required_cols: List[str] = []
        optional_cols: List[str] = []
        for name, spec in constituents.items():
            if isinstance(spec, dict) and "subConstituents" in spec:
                nested_required, nested_optional = _collect(
                    spec["subConstituents"]
                )
                required_cols.extend(nested_required)
                optional_cols.extend(nested_optional)
                continue

            if not isinstance(spec, dict):
                continue

            if spec.get("required", False):
                required_cols.append(str(name))
            else:
                optional_cols.append(str(name))

        return required_cols, optional_cols

    return _collect(schema.get("constituents", {}))


def _schema_deployment_population_value(schema: Dict[str, Any]) -> Any:
    """Return a single deployment population value when schema is singleton.

    If the schema encodes multiple deployment populations, return None so the
    caller can treat the CSV column as required instead of auto-filling it.
    """

    deployment_population = schema.get("fixedFields", {}).get(
        "deploymentPopulation"
    )
    if isinstance(deployment_population, list):
        if len(deployment_population) == 1:
            return deployment_population[0]
        return None
    return deployment_population


def _atomic_values(value: Any) -> List[str]:
    text = value_to_str(value).strip()
    if not text or text == "NA":
        return ["NA"]
    parts = [part.strip() for part in text.split(";")]
    cleaned = [part for part in parts if part]
    return cleaned or ["NA"]


def _collect_expected_values(
    constituents: Dict[str, Any],
) -> Dict[str, List[str]]:
    expected_values: Dict[str, List[str]] = {}
    for name, spec in constituents.items():
        if isinstance(spec, dict) and "subConstituents" in spec:
            expected_values.update(
                _collect_expected_values(spec["subConstituents"])
            )
            continue

        if not isinstance(spec, dict):
            continue

        values = spec.get("value")
        if isinstance(values, list):
            raw_values = values
        elif values is None:
            raw_values = []
        else:
            raw_values = [values]

        expected_values[str(name)] = [
            atomic
            for raw_value in raw_values
            for atomic in _atomic_values(raw_value)
        ]

    return expected_values


def _coverage_atomic_values(value: Any) -> List[str]:
    """Return atomic values for coverage counting, excluding placeholders."""

    if pd.isna(value):
        return []
    text = value_to_str(value).strip()
    if not text or text == "NA":
        return []
    parts = [part.strip() for part in text.split(";")]
    return [part for part in parts if part]


def _canonical_category(value: Any) -> str:
    """Normalize a category string for matching:

    - Convert smart quotes to ASCII equivalents
    - Strip any parenthetical content (keep text before '(')
    - Trim whitespace
    """
    text = value_to_str(value).strip()
    if not text:
        return text
    # Normalize curly/smart quotes to ASCII
    text = text.replace("’", "'").replace("‘", "'")
    text = text.replace("“", '"').replace("”", '"')
    # If there are broken double quotes inside parentheticals (e.g. "(i.e., bright, ""loud)"),
    # remove the stray double-quotes so later repair can fix them. Only remove consecutive
    # double-quote characters inside parentheses.
    try:

        def _clean_paren(match: re.Match) -> str:
            inner = match.group(0)
            # Remove any ASCII double-quote characters inside parentheses so
            # broken quoting (e.g. ("loud", "bright") or (i.e., ""loud))
            # doesn't interfere with later processing. Keep other characters.
            if '"' in inner:
                inner = inner.replace('"', "")
            return inner

        text = re.sub(r"\([^)]*\)", _clean_paren, text)
    except Exception:
        # If regex processing fails for any reason, fall back to original text
        pass

    # Strip parenthetical suffixes
    if "(" in text:
        text = text.split("(", 1)[0].strip()
    return text


def _coverage_vary_penalty(count: int, k: int) -> float:
    """Monotone decreasing miscoverage penalty with no reward beyond k."""

    if count <= 0:
        return 1.0
    if count >= k:
        return 0.0
    return float(1.0 - (count / k))


def _normalize_combo_for_match(combo: Sequence[Any]) -> Tuple[str, ...]:
    return tuple(_canonical_category(value) for value in combo)


def _combo_matches_not_realistic(
    combo: Sequence[Any], not_realistic_combo: Sequence[Any]
) -> bool:
    combo_counter = Counter(_normalize_combo_for_match(combo))
    not_realistic_counter = Counter(
        _normalize_combo_for_match(not_realistic_combo)
    )
    for value, count in not_realistic_counter.items():
        if combo_counter[value] < count:
            return False
    return True


def _is_excluded_by_not_realistic(
    combo: Sequence[Any], not_realistic: Optional[Sequence[Sequence[Any]]]
) -> bool:
    if not not_realistic:
        return False
    return any(
        _combo_matches_not_realistic(combo, blocked_combo)
        for blocked_combo in not_realistic
    )


def _compute_coverage_constrained_summary(
    coverage_report: Dict[str, Any],
    not_realistic: Optional[Sequence[Sequence[Any]]],
) -> Optional[Dict[str, Any]]:
    if not not_realistic:
        return None

    expected_combos = list(coverage_report.get("expected_combinations", []))
    observed_combos = list(coverage_report.get("observed_combinations", []))
    missing_combos = list(coverage_report.get("missing_combinations", []))

    if not expected_combos:
        return {
            "coverage_constrained": 1.0,
            "expected_combinations_count": 0,
            "observed_combinations_count": 0,
            "excluded_combinations_count": 0,
            "non_penalized_missing_count": 0,
        }

    excluded_expected = [
        combo
        for combo in expected_combos
        if _is_excluded_by_not_realistic(combo, not_realistic)
    ]
    non_penalized_missing = [
        combo
        for combo in missing_combos
        if _is_excluded_by_not_realistic(combo, not_realistic)
    ]

    constrained_expected = max(0, len(expected_combos) - len(excluded_expected))
    constrained_observed = min(
        constrained_expected,
        len(observed_combos),
    )
    constrained_ratio = (
        float(constrained_observed / constrained_expected)
        if constrained_expected > 0
        else 1.0
    )

    return {
        "coverage_constrained": constrained_ratio,
        "expected_combinations_count": int(constrained_expected),
        "observed_combinations_count": int(constrained_observed),
        "excluded_combinations_count": int(len(excluded_expected)),
        "non_penalized_missing_count": int(len(non_penalized_missing)),
    }


def _compute_coverage_vary_summary(
    schema: Dict[str, Any],
    working_df: pd.DataFrame,
    coverage_cols: Sequence[str],
    default_k: int = 1,
    expected_by_column: Optional[Dict[str, List[str]]] = None,
    not_realistic: Optional[Sequence[Sequence[Any]]] = None,
) -> Dict[str, Any]:
    """Compute the smooth coverage penalty curve across k values.

    The score is the unweighted average of per-group miscoverage penalties.
    Each group is defined by the cartesian product of atomic values defined in
    the schema for the coverage columns. Groups that don't appear in the data
    are counted with 0 instances.
    """

    if default_k < 1:
        raise ValueError("default_k must be at least 1")

    empty_result = {
        "default_k": int(default_k),
        "default_score": float("nan"),
        "max_k": 0,
        "k_values": [],
        "scores": [],
        "group_count": 0,
        "excluded_group_count": 0,
        "group_count_mean": float("nan"),
        "group_count_median": float("nan"),
        "penalty_definition": "1 if n=0; 1 - n/k if 0<n<k; 0 if n>=k",
    }

    if not coverage_cols:
        return empty_result

    # Get expected atomic values from schema (handles nested subConstituents)
    if expected_by_column is None:
        expected_by_column = _collect_expected_values(
            schema.get("constituents", {})
        )

    # Include deploymentPopulation as intended_deployment_population when present
    # (merge into provided expected_by_column if caller supplied one)
    deployment_population = schema.get("fixedFields", {}).get(
        "deploymentPopulation"
    )
    if deployment_population is not None:
        if isinstance(deployment_population, list):
            expected_by_column["intended_deployment_population"] = [
                atomic
                for raw_value in deployment_population
                for atomic in _atomic_values(raw_value)
            ]
        else:
            expected_by_column["intended_deployment_population"] = (
                _atomic_values(deployment_population)
            )

    atomic_values_by_col: Dict[str, List[str]] = {}
    for col in coverage_cols:
        values = expected_by_column.get(col, [])
        if not values:
            return empty_result
        atomic_values_by_col[col] = sorted([value_to_str(v) for v in values])

    # Count observed instances of each combinatorial group
    combo_counts: Dict[Tuple[str, ...], int] = {}
    for _, row in working_df.iterrows():
        per_col_atomic_values = [
            _coverage_atomic_values(row[col]) for col in coverage_cols
        ]
        if any(not values for values in per_col_atomic_values):
            continue
        for combo in product(*per_col_atomic_values):
            combo_counts[tuple(combo)] = combo_counts.get(tuple(combo), 0) + 1

    # Build all expected combinations from schema values
    all_expected_combos = list(product(*atomic_values_by_col.values()))
    expected_combos = [
        combo
        for combo in all_expected_combos
        if not _is_excluded_by_not_realistic(combo, not_realistic)
    ]
    group_counts = [combo_counts.get(combo, 0) for combo in expected_combos]
    max_k = int(max(group_counts, default=0))

    if max_k < 1:
        return {
            **empty_result,
            "group_count": int(len(group_counts)),
            "excluded_group_count": int(
                len(all_expected_combos) - len(expected_combos)
            ),
            "group_count_mean": (
                float(np.mean(group_counts)) if group_counts else float("nan")
            ),
            "group_count_median": (
                float(np.median(group_counts)) if group_counts else float("nan")
            ),
        }

    k_values = list(range(1, max_k + 1))
    scores = [
        float(
            np.mean(
                [_coverage_vary_penalty(count, k) for count in group_counts]
            )
        )
        for k in k_values
    ]

    default_score = float(
        np.mean(
            [_coverage_vary_penalty(count, default_k) for count in group_counts]
        )
    )
    return {
        "default_k": int(default_k),
        "default_score": default_score,
        "max_k": int(max_k),
        "k_values": [int(k) for k in k_values],
        "scores": [float(score) for score in scores],
        "group_count": int(len(group_counts)),
        "excluded_group_count": int(
            len(all_expected_combos) - len(expected_combos)
        ),
        "group_count_mean": float(np.mean(group_counts)),
        "group_count_median": float(np.median(group_counts)),
        "penalty_definition": "1 if n=0; 1 - n/k if 0<n<k; 0 if n>=k",
    }


def _save_coverage_vary_plot(
    coverage_vary_summary: Dict[str, Any], output_pdf_path: Path
) -> None:
    """Save the coverage_vary k sweep as a PDF plot."""

    k_values = coverage_vary_summary.get("k_values", [])
    scores = coverage_vary_summary.get("scores", [])
    if not k_values or not scores:
        return

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required to save the coverage_vary PDF plot."
        ) from exc

    fig, ax = plt.subplots(figsize=(7, 4.5))
    # Invert scores from penalty to coverage
    coverage_scores = [1.0 - p for p in scores]
    default_penalty = coverage_vary_summary.get("default_score", float("nan"))
    default_coverage = 1.0 - default_penalty

    ax.plot(k_values, coverage_scores, color="#1f77b4", linewidth=2)
    ax.scatter(
        [coverage_vary_summary.get("default_k", 1)],
        [default_coverage],
        color="#d62728",
        zorder=3,
        label=f"k={coverage_vary_summary.get('default_k', 1)}",
    )
    ax.set_title("Coverage Vary Sweep")
    ax.set_xlabel("k")
    ax.set_ylabel("Average coverage")
    ax.set_xlim(min(k_values), max(k_values))
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_pdf_path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def _compute_marginal_coverage_summary(
    schema: Dict[str, Any],
    working_df: pd.DataFrame,
    coverage_cols: Sequence[str],
) -> Dict[str, Any]:
    expected_by_column = _collect_expected_values(
        schema.get("constituents", {})
    )
    deployment_population = schema.get("fixedFields", {}).get(
        "deploymentPopulation"
    )
    if deployment_population is not None:
        if isinstance(deployment_population, list):
            expected_by_column["intended_deployment_population"] = [
                atomic
                for raw_value in deployment_population
                for atomic in _atomic_values(raw_value)
            ]
        else:
            expected_by_column["intended_deployment_population"] = (
                _atomic_values(deployment_population)
            )

    marginal_coverage: Dict[str, Any] = {}
    for column in coverage_cols:
        expected_values = {
            value_to_str(value)
            for value in expected_by_column.get(column, [])
            if value_to_str(value).strip()
        }
        if not expected_values:
            continue

        observed_values = {
            atomic_value
            for value in working_df[column].dropna().unique()
            for atomic_value in _atomic_values(value)
            if atomic_value.strip()
        }
        matched_values = expected_values & observed_values
        missing_values = sorted(expected_values - observed_values)
        new_values = sorted(observed_values - expected_values)
        marginal_coverage[column] = {
            "expected_values_count": int(len(expected_values)),
            "observed_values_count": int(len(matched_values)),
            "missing_values_count": int(len(missing_values)),
            "coverage_ratio": (
                float(len(matched_values) / len(expected_values))
                if expected_values
                else 1.0
            ),
            "missing_values_sample": missing_values[:20],
            "new_values": new_values,
        }

    return {
        "marginal_coverage": marginal_coverage,
        "overall_marginal_coverage": (
            float(
                np.mean(
                    [
                        stats["coverage_ratio"]
                        for stats in marginal_coverage.values()
                    ]
                )
            )
            if marginal_coverage
            else 1.0
        ),
    }


def _extract_invalid_combinations(
    schema: Dict[str, Any], constituent_cols: Sequence[str]
) -> List[Tuple[str, ...]]:
    raw = schema.get("invalidCombinations") or schema.get(
        "invalid_combinations"
    )
    if raw is None:
        raw = schema.get("fixedFields", {}).get("invalidCombinations")
    if not raw:
        return []

    combos: List[Tuple[str, ...]] = []
    for item in raw:
        if isinstance(item, dict):
            combo = tuple(
                value_to_str(item.get(col)) for col in constituent_cols
            )
        elif isinstance(item, (list, tuple)):
            combo = tuple(value_to_str(v) for v in item)
        else:
            continue

        if len(combo) == len(constituent_cols):
            combos.append(combo)

    return combos


def _prepare_dataframe(df: pd.DataFrame, text_column: str) -> pd.DataFrame:
    prepared = df.copy()
    prepared.columns = [_normalize_column_name(col) for col in prepared.columns]
    if text_column != "example":
        prepared = prepared.rename(columns={text_column: "example"})
    return prepared


def _prepare_upfront_dataframe(
    raw_df: pd.DataFrame, schema: Dict[str, Any]
) -> Tuple[pd.DataFrame, List[str], List[str], List[str], Any, str]:
    """Validate the CSV once up front and prepare schema-derived columns.

    Returns:
        working_df: normalized dataframe with text column renamed to example
        constituent_cols: all constituent columns defined by the schema
        required_constituent_cols: constituent columns marked required
        optional_constituent_cols: constituent columns marked optional
        deployment_population_value: singleton deployment population value,
            or None when the schema implies multiple values.
    """

    raw_df = raw_df.copy()
    raw_df.columns = [_normalize_column_name(col) for col in raw_df.columns]

    text_column = _find_text_column(raw_df.columns)
    working_df = _prepare_dataframe(raw_df, text_column)

    constituent_cols = _flatten_constituent_columns(
        schema.get("constituents", {})
    )
    required_constituent_cols, optional_constituent_cols = (
        _schema_constituent_columns_by_requirement(schema)
    )
    missing_constituents = [
        col
        for col in required_constituent_cols
        if col not in working_df.columns
    ]
    if missing_constituents:
        raise ValueError(
            "Dataset is missing constituent columns required by the schema: "
            f"{missing_constituents}"
        )

    deployment_population_value = _schema_deployment_population_value(schema)
    if (
        "intended_deployment_population" not in working_df.columns
        and deployment_population_value is None
    ):
        raise ValueError(
            "Dataset is missing intended_deployment_population, and the schema "
            "does not define a singleton deploymentPopulation value that can be "
            "filled automatically."
        )

    if "intended_deployment_population" not in working_df.columns:
        working_df["intended_deployment_population"] = (
            deployment_population_value
        )

    return (
        working_df,
        constituent_cols,
        required_constituent_cols,
        optional_constituent_cols,
        deployment_population_value,
        text_column,
    )


def _serializable_coverage_summary(report: Dict[str, Any]) -> Dict[str, Any]:
    category_stats = {}
    for name, stats in report.get("category_stats", {}).items():
        category_stats[name] = {
            "count": int(stats["count"]),
            "concentration_kappa": float(stats["concentration_kappa"]),
            "mean_pairwise_similarity": float(
                stats["mean_pairwise_similarity"]
            ),
            "min_pairwise_similarity": float(stats["min_pairwise_similarity"]),
            "n_outliers": int(len(stats.get("examples_below_threshold", []))),
            "outlier_indices": [
                int(i) for i in stats.get("examples_below_threshold", [])
            ],
        }

    hierarchical = []
    for item in report.get("hierarchical_results", []):
        hierarchical.append(
            {
                "example_idx": int(item["example_idx"]),
                "fine_grained_label": item["fine_grained_label"],
                "fine_grained_similarity": float(
                    item["fine_grained_similarity"]
                ),
                "coarse_similarities": {
                    key: float(value)
                    for key, value in item.get(
                        "coarse_similarities", {}
                    ).items()
                },
                "flags": list(item.get("flags", [])),
            }
        )

    return {
        "expected_combinations_count": int(
            len(report.get("expected_combinations", []))
        ),
        "observed_combinations_count": int(
            len(report.get("observed_combinations", []))
        ),
        "missing_combinations_count": int(
            len(report.get("missing_combinations", []))
        ),
        "missing_combinations_sample": [
            list(combo)
            for combo in list(report.get("missing_combinations", []))[:20]
        ],
        "completeness_ratio": float(
            report.get("completeness_ratio", float("nan"))
        ),
        "holistic_completeness_score": float(
            report.get("holistic_completeness_score", float("nan"))
        ),
        "overall_diversity_score": float(
            report.get("overall_diversity_score", float("nan"))
        ),
        "redundant_pairs_count": int(len(report.get("redundant_pairs", []))),
        "redundant_pairs_sample": [
            list(pair) for pair in list(report.get("redundant_pairs", []))[:50]
        ],
        "category_stats": category_stats,
        "hierarchical_results": hierarchical,
    }


def _serializable_diversity_summary(report: Dict[str, Any]) -> Dict[str, Any]:
    subgroup_stats = {}
    for label, stats in report.get("subgroup_stats", {}).items():
        subgroup_stats[label] = {
            "count": int(stats["count"]),
            "dc_score": float(stats["dc_score"]),
            "vendi_score": float(stats["vendi_score"]),
            "mean_pairwise_similarity": float(
                stats["mean_pairwise_similarity"]
            ),
            "surface_diversity_score": float(stats["surface_diversity_score"]),
            "redundant_pairs": [
                list(pair) for pair in stats.get("redundant_pairs", [])
            ],
        }

    return {
        "overall_dc_score": float(report.get("overall_dc_score", float("nan"))),
        "overall_vendi_score": float(
            report.get("overall_vendi_score", float("nan"))
        ),
        "overall_mean_pairwise_similarity": float(
            report.get("overall_mean_pairwise_similarity", float("nan"))
        ),
        "subgroup_stats": subgroup_stats,
        "all_redundant_pairs_count": int(
            len(report.get("all_redundant_pairs", []))
        ),
        "all_redundant_pairs_sample": [
            list(pair)
            for pair in list(report.get("all_redundant_pairs", []))[:50]
        ],
        "lowest_diversity_groups": [
            [label, float(score)]
            for label, score in report.get("lowest_diversity_groups", [])
        ],
    }


def _compute_marginal_diversity(
    diversity_report: Dict[str, Any], min_group_size: int = 6
) -> float:
    subgroup_stats = diversity_report.get("subgroup_stats", {})
    scores: List[float] = []
    for label, stats in subgroup_stats.items():
        if "+" in label or "=" not in label:
            continue
        if int(stats.get("count", 0)) < min_group_size:
            continue
        score = stats.get("dc_score")
        if score is not None and np.isfinite(score):
            scores.append(float(score))
    return float(np.mean(scores)) if scores else float("nan")


def _build_metric_summary(
    coverage_summary: Dict[str, Any],
    coverage_vary_summary: Dict[str, Any],
    coverage_constrained_summary: Optional[Dict[str, Any]],
    coverage_vary_constrained_summary: Optional[Dict[str, Any]],
    coverage_uncertain_summary: Dict[str, Any],
    diversity_report: Dict[str, Any],
    marginal_diversity: float,
    realism_summary_full: Dict[str, Any],
    realism_summary_subset: Optional[Dict[str, Any]],
) -> Dict[str, float]:
    """Build compact top-level summary scores used for quick comparison."""
    subset = realism_summary_subset or {}
    return {
        "coverage": float(
            coverage_summary.get("completeness_ratio", float("nan"))
        ),
        "coverage_vary": float(
            1.0 - coverage_vary_summary.get("default_score", float("nan"))
        ),
        "coverage_constrained": float(
            (coverage_constrained_summary or {}).get(
                "coverage_constrained", float("nan")
            )
        ),
        "coverage_vary_constrained": float(
            1.0
            - (coverage_vary_constrained_summary or {}).get(
                "default_score", float("nan")
            )
        ),
        "coverage_uncertain": float(
            1.0 - coverage_uncertain_summary.get("default_score", float("nan"))
        ),
        "dist_diversity": float(
            diversity_report.get("overall_dc_score", float("nan"))
        ),
        "marginal_diversity": float(marginal_diversity),
        "full_stylistic_realism": float(
            realism_summary_full.get("style_realism_score", float("nan"))
        ),
        "subset_stylistic_realism": float(
            subset.get("style_realism_score", float("nan"))
        ),
        # Calibrated Sinkhorn already reports higher as better after calibration.
        "full_distributional_realism": float(
            realism_summary_full.get(
                "calibrated_sinkhorn_distance", float("nan")
            )
        ),
        "subset_distributional_realism": float(
            subset.get("calibrated_sinkhorn_distance", float("nan"))
        ),
    }


def _mark_outliers(
    df: pd.DataFrame,
    coverage_report: Dict[str, Any],
    invalid_combinations: Sequence[Tuple[str, ...]],
    constituent_cols: Sequence[str],
) -> Tuple[np.ndarray, List[int], List[int]]:
    coverage_outliers: set[int] = set()
    for stats in coverage_report.get("category_stats", {}).values():
        coverage_outliers.update(
            int(idx) for idx in stats.get("outlier_indices", [])
        )

    invalid_combo_set = set(invalid_combinations)
    invalid_indices: List[int] = []
    for idx, row in df.iterrows():
        combo = tuple(value_to_str(row[col]) for col in constituent_cols)
        if combo in invalid_combo_set:
            invalid_indices.append(int(idx))

    codes = np.zeros(len(df), dtype=int)
    for idx in coverage_outliers:
        if 0 <= idx < len(codes):
            codes[idx] = 1
    for idx in invalid_indices:
        if 0 <= idx < len(codes):
            codes[idx] = 2 if codes[idx] == 0 else 3

    return codes, sorted(coverage_outliers), sorted(invalid_indices)


def _build_row_metrics(
    working_df: pd.DataFrame,
    coverage_validator: CoverageValidator,
    constituent_cols: Sequence[str],
    realism_report: Dict[str, Any],
) -> pd.DataFrame:
    generated_embeddings = np.asarray(
        coverage_validator.embeddings, dtype=np.float64
    )
    if generated_embeddings.ndim != 2:
        raise ValueError(
            "Coverage validator did not cache a 2D embedding matrix."
        )

    row_metrics = working_df.copy()
    dist_df = compute_row_distance_metrics(
        df=working_df,
        constituent_cols=list(constituent_cols),
        embeddings=generated_embeddings,
        category_centers=coverage_validator.category_centers,
    )
    row_metrics["centroid_dist"] = dist_df["centroid_dist"].values
    row_metrics["dataset_center_dist"] = dist_df["dataset_center_dist"].values

    min_seed_dist = np.full(len(working_df), np.nan, dtype=np.float64)
    for item in realism_report.get("instance_stats", []):
        idx = int(item["idx"])
        if 0 <= idx < len(min_seed_dist):
            min_seed_dist[idx] = float(item["novelty_min"])
    row_metrics["min_seed_dist"] = min_seed_dist

    return row_metrics


def run_pipeline(
    schema_path: Path,
    dataset_path: Path,
    output_json_path: Path,
    output_csv_path: Path,
    output_pdf_path: Path,
    embedding_model: str,
    coverage_k: int,
    seed_subset_indices: Optional[List[int]],
    not_realistic: Optional[List[List[str]]] = None,
) -> Dict[str, Any]:
    with schema_path.open("r", encoding="utf-8") as handle:
        schema = json.load(handle)

    raw_df = pd.read_csv(dataset_path)
    (
        working_df,
        constituent_cols,
        required_constituent_cols,
        optional_constituent_cols,
        deployment_population_value,
        text_column,
    ) = _prepare_upfront_dataframe(raw_df, schema)

    invalid_combinations = _extract_invalid_combinations(
        schema, constituent_cols
    )
    seed_texts = schema.get("seedExamples", []) or []
    full_seed_texts = [str(text) for text in seed_texts if str(text).strip()]

    print("Running coverage metrics...")
    dataset_size = len(working_df)
    # Count number of unique categories (composite labels) in the dataset
    # Use constituent_cols to get all possible categories from schema
    # For coverage, categories are usually all unique combinations of constituent values present
    # Here, we use the number of unique composite labels in the data
    if constituent_cols:
        # Build composite label for each row
        def _composite_label(row):
            return "+".join(str(row[col]) for col in constituent_cols)

        unique_categories = working_df.apply(_composite_label, axis=1).nunique()
    else:
        unique_categories = 1
    min_samples_for_stats = max(3, dataset_size // max(1, unique_categories))
    coverage_validator = CoverageValidator(
        embedding_model=embedding_model,
        min_samples_for_stats=min_samples_for_stats,
    )
    expected_by_column = _collect_expected_values(
        schema.get("constituents", {})
    )
    coverage_schema_dimensions = {
        col: expected_by_column[col]
        for col in required_constituent_cols
        if col in expected_by_column
    }
    coverage_report = coverage_validator.validate(
        working_df,
        constituent_cols,
        required_constituent_cols=required_constituent_cols,
        optional_constituent_cols=optional_constituent_cols,
        schema_dimensions=coverage_schema_dimensions,
    )

    print("Running diversity metrics...")
    diversity_validator = DiversityValidator(embedding_model=embedding_model)
    diversity_report = diversity_validator.validate(
        working_df, constituent_cols
    )

    print("Running realism metrics...")
    realism_validator = RealismValidator(embedding_model=embedding_model)
    realism_report_full = realism_validator.validate(
        generated_df=working_df,
        constituent_cols=constituent_cols,
        seed_examples=full_seed_texts,
        invalid_combinations=invalid_combinations or None,
        per_category=False,
    )

    realism_report_subset = None
    if seed_subset_indices is not None:
        if not full_seed_texts:
            raise ValueError(
                "Seed subset indices were provided, but there are no seed examples."
            )
        subset_seed_texts = []
        for idx in seed_subset_indices:
            if idx < 0 or idx >= len(full_seed_texts):
                raise ValueError(
                    "Seed subset index out of range: "
                    f"{idx} (seed count={len(full_seed_texts)})"
                )
            subset_seed_texts.append(full_seed_texts[idx])
        if not subset_seed_texts:
            raise ValueError(
                "Seed subset indices did not resolve to any seed examples."
            )
        realism_report_subset = realism_validator.validate(
            generated_df=working_df,
            constituent_cols=constituent_cols,
            seed_examples=subset_seed_texts,
            invalid_combinations=invalid_combinations or None,
            per_category=False,
        )

    print("Computing row-level metrics...")
    row_metrics = _build_row_metrics(
        working_df=working_df,
        coverage_validator=coverage_validator,
        constituent_cols=constituent_cols,
        realism_report=realism_report_full,
    )

    outlier_codes, coverage_outlier_indices, invalid_group_indices = (
        _mark_outliers(
            working_df,
            coverage_report,
            invalid_combinations,
            constituent_cols,
        )
    )
    row_metrics["outlier"] = outlier_codes

    # Keep the CSV output close to the source format, but with the new metrics appended.
    output_df = raw_df.copy()
    for col in [
        "centroid_dist",
        "dataset_center_dist",
        "outlier",
        "min_seed_dist",
    ]:
        output_df[col] = row_metrics[col].values

    output_df.to_csv(output_csv_path, index=False)

    coverage_summary = _serializable_coverage_summary(coverage_report)
    coverage_cols = [
        col for col in required_constituent_cols if col in working_df.columns
    ]
    coverage_vary_summary = _compute_coverage_vary_summary(
        schema=schema,
        working_df=working_df,
        coverage_cols=coverage_cols,
        default_k=coverage_k,
        expected_by_column=expected_by_column,
    )
    coverage_constrained_summary = _compute_coverage_constrained_summary(
        coverage_report=coverage_report,
        not_realistic=not_realistic,
    )
    coverage_vary_constrained_summary = (
        _compute_coverage_vary_summary(
            schema=schema,
            working_df=working_df,
            coverage_cols=coverage_cols,
            default_k=coverage_k,
            expected_by_column=expected_by_column,
            not_realistic=not_realistic,
        )
        if not_realistic
        else None
    )
    # For coverage_uncertain we only consider required constituent columns
    # (same behavior as coverage_vary) so that uncertain counts aren't
    # exploded by optional or contextual fields.
    coverage_uncertain_cols = coverage_cols
    coverage_uncertain_summary = _compute_coverage_vary_summary(
        schema=schema,
        working_df=working_df,
        coverage_cols=coverage_uncertain_cols,
        default_k=coverage_k,
        expected_by_column=expected_by_column,
    )
    # Also compute coverage_vary_all which includes optional constituent columns
    coverage_all_cols = [
        col
        for col in (required_constituent_cols + optional_constituent_cols)
        if col in working_df.columns
    ]
    coverage_vary_all_summary = _compute_coverage_vary_summary(
        schema=schema,
        working_df=working_df,
        coverage_cols=coverage_all_cols,
        default_k=coverage_k,
        expected_by_column=expected_by_column,
    )
    coverage_summary.update(
        _compute_marginal_coverage_summary(
            schema=schema,
            working_df=working_df,
            coverage_cols=coverage_cols,
        )
    )
    diversity_summary = _serializable_diversity_summary(diversity_report)
    marginal_diversity = _compute_marginal_diversity(diversity_report)
    realism_summary_full = serialize_realism_report(realism_report_full)
    realism_summary_subset = (
        serialize_realism_report(realism_report_subset)
        if realism_report_subset is not None
        else None
    )

    row_level_summary = {
        "n_rows": int(len(output_df)),
        "outlier_counts": {
            str(code): int((output_df["outlier"] == code).sum())
            for code in [0, 1, 2, 3]
        },
        "centroid_dist_mean": float(np.nanmean(output_df["centroid_dist"])),
        "centroid_dist_median": float(np.nanmedian(output_df["centroid_dist"])),
        "dataset_center_dist_mean": float(
            np.nanmean(output_df["dataset_center_dist"])
        ),
        "min_seed_dist_mean": float(np.nanmean(output_df["min_seed_dist"])),
        "coverage_outlier_indices": coverage_outlier_indices,
        "invalid_group_indices": invalid_group_indices,
    }

    regeneration_candidates = output_df.index[
        output_df["outlier"] != 0
    ].tolist()

    # Compute marginal value distributions for each constituent column
    cross_distribution = {}
    for col in coverage_cols:
        value_counts = working_df[col].value_counts(dropna=False)
        total = value_counts.sum()
        col_dist = {}
        for val, count in value_counts.items():
            key = value_to_str(val)
            percent = 100.0 * count / total if total > 0 else 0.0
            col_dist[key] = percent
        cross_distribution[col] = col_dist

    expected_by_column = _collect_expected_values(
        schema.get("constituents", {})
    )
    total_rows = len(working_df)
    marginal_distribution: Dict[str, Dict[str, float]] = {}
    for col in constituent_cols:
        if col not in working_df.columns:
            continue
        # Canonicalize schema-defined expected values (strip parentheticals, normalize quotes)
        expected_values = [
            _canonical_category(value)
            for value in expected_by_column.get(col, [])
            if _canonical_category(value).strip()
        ]
        # Initialize counts for schema-defined canonical categories only
        counts: Dict[str, int] = {
            value: 0 for value in sorted(set(expected_values))
        }

        # Count number of examples (rows) that include each atomic value (canonicalized)
        for value in working_df[col].tolist():
            per_row_atomic = {
                _canonical_category(atomic)
                for atomic in _atomic_values(value)
                if _canonical_category(atomic).strip()
            }
            for atomic_value in per_row_atomic:
                # Only count atomic values defined in the schema (canonical form)
                if atomic_value in counts:
                    counts[atomic_value] += 1

        col_dist = {
            key: (100.0 * count / total_rows if total_rows > 0 else 0.0)
            for key, count in sorted(counts.items())
        }
        marginal_distribution[col] = col_dist

    summary = {
        "summary": {
            **_build_metric_summary(
                coverage_summary=coverage_summary,
                coverage_vary_summary=coverage_vary_summary,
                coverage_constrained_summary=coverage_constrained_summary,
                coverage_vary_constrained_summary=coverage_vary_constrained_summary,
                coverage_uncertain_summary=coverage_uncertain_summary,
                diversity_report=diversity_report,
                marginal_diversity=marginal_diversity,
                realism_summary_full=realism_summary_full,
                realism_summary_subset=realism_summary_subset,
            ),
            "cross_distribution": cross_distribution,
            "marginal_distribution": marginal_distribution,
        },
        "inputs": {
            "schema_path": str(schema_path),
            "dataset_path": str(dataset_path),
            "text_column": text_column,
            "constituent_columns": constituent_cols,
            "invalid_combinations_count": int(len(invalid_combinations)),
            "seed_example_count": int(
                len([text for text in seed_texts if str(text).strip()])
            ),
            "seed_subset_indices": seed_subset_indices,
            "not_realistic": not_realistic,
        },
        "coverage": coverage_summary,
        "coverage_vary": coverage_vary_summary,
        "coverage_constrained": coverage_constrained_summary,
        "coverage_vary_constrained": coverage_vary_constrained_summary,
        "coverage_vary_all": coverage_vary_all_summary,
        "coverage_uncertain": coverage_uncertain_summary,
        "diversity": diversity_summary,
        "realism_full": realism_summary_full,
        "realism_subset": realism_summary_subset,
        "row_level": row_level_summary,
        "regeneration_subset": {
            "candidate_count": int(len(regeneration_candidates)),
            "candidate_indices": regeneration_candidates,
            "ranking": None,
            "note": "Ranking is a placeholder and is intentionally not implemented yet.",
        },
        "schema": {
            "capability": schema.get("fixedFields", {}).get("capability"),
            "deployment_population": schema.get("fixedFields", {}).get(
                "deploymentPopulation"
            ),
            "contextual_constraints": schema.get("fixedFields", {}).get(
                "contextualConstraints"
            ),
        },
    }

    _safe_json_dump(output_json_path, summary)
    # _save_coverage_vary_plot(coverage_vary_summary, output_pdf_path)
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run coverage, diversity, and seed-distance realism metrics on a generated dataset."
    )
    parser.add_argument(
        "--schema", required=True, help="Path to the schema JSON file."
    )
    parser.add_argument("--dataset", help="Path to one dataset CSV file.")
    parser.add_argument(
        "--input-dir",
        help="Directory containing datasets to evaluate instead of --dataset.",
    )
    parser.add_argument(
        "--pattern",
        default="*_labeled.csv",
        help="Glob pattern used with --input-dir.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively search --input-dir.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for multiple datasets.",
    )
    parser.add_argument(
        "--output-json", default=None, help="Output JSON path for one dataset."
    )
    parser.add_argument(
        "--output-csv", default=None, help="Output CSV path for one dataset."
    )
    parser.add_argument(
        "--output-pdf", default=None, help="Output PDF path for one dataset."
    )
    parser.add_argument("--embedding-model", default="all-mpnet-base-v2")
    parser.add_argument("--coverage-k", type=int, default=1)
    parser.add_argument("--seed-subset-indices", default=None)
    parser.add_argument("--not-realistic", default=None)
    return parser


def _run_one_dataset(args: argparse.Namespace, dataset_path: Path) -> None:
    output_json_path = (
        Path(args.output_json).expanduser().resolve()
        if args.output_json and not args.input_dir
        else dataset_path.with_name(f"{dataset_path.stem}_metrics.json")
    )
    output_csv_path = (
        Path(args.output_csv).expanduser().resolve()
        if args.output_csv and not args.input_dir
        else dataset_path.with_name(f"{dataset_path.stem}_metrics.csv")
    )
    output_pdf_path = (
        Path(args.output_pdf).expanduser().resolve()
        if args.output_pdf and not args.input_dir
        else dataset_path.with_name(f"{dataset_path.stem}_coverage_vary.pdf")
    )
    if args.output_dir:
        output_dir = Path(args.output_dir).expanduser().resolve()
        output_json_path, output_csv_path, output_pdf_path = (
            output_dir / output_json_path.name,
            output_dir / output_csv_path.name,
            output_dir / output_pdf_path.name,
        )
    for path in (output_json_path, output_csv_path, output_pdf_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    summary = run_pipeline(
        schema_path=Path(args.schema).expanduser().resolve(),
        dataset_path=dataset_path,
        output_json_path=output_json_path,
        output_csv_path=output_csv_path,
        output_pdf_path=output_pdf_path,
        embedding_model=args.embedding_model,
        coverage_k=args.coverage_k,
        seed_subset_indices=_parse_seed_subset_indices(
            args.seed_subset_indices
        ),
        not_realistic=_parse_not_realistic(args.not_realistic),
    )
    print(f"Processed {dataset_path}: {summary['row_level']['n_rows']} rows.")


def main() -> None:
    args = build_arg_parser().parse_args()
    if bool(args.dataset) == bool(args.input_dir):
        raise SystemExit("Provide exactly one of --dataset or --input-dir.")
    if args.dataset:
        _run_one_dataset(args, Path(args.dataset).expanduser().resolve())
        return
    input_dir = Path(args.input_dir).expanduser().resolve()
    if not input_dir.exists():
        raise FileNotFoundError(f"Directory does not exist: {input_dir}")
    paths = (
        input_dir.rglob(args.pattern)
        if args.recursive
        else input_dir.glob(args.pattern)
    )
    datasets = [path for path in sorted(paths) if path.is_file()]
    for dataset_path in datasets:
        if (
            "_metrics" not in dataset_path.stem
            and "_labeled" in dataset_path.stem
        ):
            _run_one_dataset(args, dataset_path)


if __name__ == "__main__":
    main()
