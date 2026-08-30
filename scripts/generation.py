"""
Generates synthetic evaluation datasets using schema-based AND baseline prompts,
then labels the outputs using Claude Haiku (DEFAULT_LABEL_MODEL). Change DEFAULT_LABEL_MODEL
in the helpers/project_config.py file to use a different labeling model.

Parameters
----------
models             : list of (provider, model_name) tuples
schema_file        : path to the schema JSON file
n                  : number of examples to generate per call
runs               : number of independent generation runs per (model, seed_combo)
max_seed_reshuffle : max number of seed-example combinations sampled per value of s
max_s              : maximum number of seed examples included in any single prompt
output_dir         : root output directory (default: "./data/final/")
                     Outputs are written to <output_dir>/schema/ and <output_dir>/baseline/
do_labeling        : whether to label the generated outputs as a follow-up step (default: False)
                     
Other parameters exist but are specific to ablations and not relevant for general use of Truong et al.'s (2026) framework
"""

import argparse
import copy
import csv
import io
import itertools
import json
import random
from pathlib import Path
from typing import Optional

import pandas as pd

from helpers.model_api_client import ModelAPIClient
from helpers.project_config import (
    DEFAULT_GENERATION_MODELS,
    DEFAULT_LABEL_MODEL,
    DEFAULT_LABEL_PROVIDER,
    DEFAULT_OUTPUT_DIR,
)

# ── Labeling model defaults ───────────────────────────────

_LABEL_PROVIDER = DEFAULT_LABEL_PROVIDER
_LABEL_MODEL = DEFAULT_LABEL_MODEL
_ACTIVE_LABEL_CONFIG: Optional[dict] = None


def load_labeling_config(path: str) -> dict:
    with Path(path).expanduser().open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    columns = config.get("columns")
    if not isinstance(columns, dict) or not columns:
        raise ValueError(
            "Labeling config must contain a non-empty 'columns' object."
        )
    return config


def _label_columns(config: dict) -> list[str]:
    return list(config["columns"])


def _build_labeling_prompt(text: str, config: dict) -> str:
    options = json.dumps(
        {
            column: {"possibleValues": "multiple", "value": values}
            for column, values in config["columns"].items()
        },
        ensure_ascii=False,
    )
    columns = ", ".join(config["columns"])
    instructions = (
        "I am giving you a chunk of text with each example separated by a new line. "
        "The text will start with the tag <START> and end with <END>. Do not include "
        "these tags in your response. Output a CSV with only these label columns: "
        f"{columns}. Fill each column based on what the example addresses. "
        "Separate multiple values with a semi-colon and leave inapplicable columns blank.\n\n"
        f"Here are your options: {options}\n\n"
        "Return only raw CSV text with one header row and the same number of data rows "
        "as the input, in the same order. Do not include markdown or commentary."
    )
    return instructions + "\n\n<START>\n" + text + "\n<END>"


# ── Prompt builders ────────────────────────────────────────────────────────────


def _build_schema_system_prompt(n: int, schema: Optional[dict] = None) -> str:
    schema = schema or {}
    fixed_fields = schema.get("fixedFields", {})
    constituents = schema.get("constituents", {})

    # If this schema is an ablation variant, prepare a short prefix that
    # instructs the model to produce plain-text single-line outputs. We do
    # NOT replace the schema descriptions — keep the full schema guidance
    # (capability, constituents, deploymentPopulation, etc.) after the
    # prefix so the model still knows which fields remain in the variant.

    lines = [
        f"You are a helpful assistant that generates evaluation datasets. You will receive a schema and use the information it provides to generate {n} benchmark examples measuring the specified capability.",
        "",
        "Each field of the schema is defined below:",
    ]

    if "capability" in fixed_fields:
        lines.append(" * Capability: the capability being measured")
    if "systematizedConcept" in fixed_fields:
        lines.append(
            " * Systematized concept: The high-level definition of the capability you would like to measure, written to be context-agnostic."
        )
    if constituents:
        lines.append(
            " * Constituents of the systematized concept: Constituents are their own fields. Each of these constituents represent the conceptual components the capability depends on. Each constituent specifies:"
        )
        lines.append(
            "      * Whether it is required (must appear in every example) or optional (should appear in a meaningful but variable subset)"
        )
        lines.append(
            "      * How many possible values: there can be one one value (if present) OR one or more values (multiple)"
        )
    if "deploymentPopulation" in fixed_fields:
        lines.append(
            " * Intended deployment population: The real-world users who would write these prompts. Generated examples must authentically reflect how this population writes including their vocabulary, expected knowledge, level of formality, writing style, etc."
        )
    if "contextualConstraints" in fixed_fields:
        lines.append(
            " * Contextual constraints: The real-world setting that scopes all examples. Note that the context does not need to be explicitly stated."
        )
    if "systematizedInstance" in fixed_fields:
        lines.append(
            " * Systematized instance: The structural requirements for each example. Items marked as always-present are required in every example. Items marked as sometimes-present should appear in a meaningful but variable subset of examples — do not include them uniformly. Note that unless explicitly stated, the ordering of the items in the systematized instance does not matter and should be varied for diversity."
        )
    if schema.get("seedExamples"):
        lines.append(
            " * Seed examples: examples written by real people in the intended deployment population. These represent the ideal level of detail and format desired from the generated data. The generated examples should not just copy these examples, but use these as a reference."
        )

    guidance_paragraph = "You should aim to maximize the diversity of generated examples while maintaining coverage. Vary the underlying scenario, the sequence of ideas, the level of specificity, the writing style, and the opening sentence structure the way someone from the intended deployment population would. Do not use any punctuation (e.g., em dashes) that are not also present in the seed examples. Treat each row as a different prompt archetype written by a different member of the intended deployment population. Use the seed examples as a calibration reference for authentic voice, domain language, and appropriate detail, but do not reproduce their specific scenarios. Introduce new situations and perspectives. All examples must be structurally and stylistically distinct. Revise as needed."

    csv_output_requirements = "Output format requirements: Return only raw CSV text (no markdown code fences, no prose before/after). Include a single header row followed by exactly {n} data rows. Columns are: intended deployment population, constituent item addressed (one column per constituent), and example text. If any columns have multiple values, separate with a semicolon. Text can contain new lines. If any field contains a comma or newline, wrap the entire field in double quotes, and escape any internal double quotes by doubling them."

    txt_output_requirements = "Output format requirements: Return only raw text (no markdown code fences, no prose before/after). Output exactly {n} data rows, each on a new line. If an example contains internal newlines, represent them with the two-character sequence '\\n' so the entire example remains on a single line."

    lines.extend(["", guidance_paragraph, ""])

    # If we built an ablation prefix, prepend it to the normal schema description
    # but DO NOT append the CSV output requirements (they would contradict
    # the ablation-mode plain-text instructions). For non-ablation runs, append
    # the normal CSV output requirement.
    if schema.get("_is_ablation_variant"):
        lines.append(txt_output_requirements)
    else:
        lines.append(csv_output_requirements)

    return "\n".join(lines)


def _build_schema_generation_prompt(
    schema: dict, seed_combo: Optional[list], n: int
) -> str:
    schema_copy = copy.deepcopy(schema)
    if seed_combo is not None:
        schema_copy["seedExamples"] = seed_combo
    # If ablation variant, request plain-text single-line examples
    if schema_copy.get("_is_ablation_variant"):
        return (
            f"Generate {n} examples as plain text, one example per line. that meet the requirements "
            "specified in the schema. All examples must be structuraly and stylistically distinct. For example, don't always start with the same word or phrase. Notice how the seed examples below are all different in this way, and use them as a reference for the level of diversity desired. Write as if you are different people from the intended deployment population who could have also written some of the seed examples.\n\n"
            "Return only the generated examples, one per line. If an example contains internal newlines, represent them with the two-character sequence '\\n' so each example occupies exactly one line.\n\n"
            "Do not include markdown fences or commentary.\n\n"
            f"Here is the schema: {schema_copy}"
        )
    return (
        f"Please generate CSV content with {n} examples that meet the requirements "
        "specified in the schema. All examples must be structuraly and stylistically distinct. For example, don't always start with the same word or phrase. Notice how the seed examples below are all different in this way, and use them as a reference for the level of diversity desired. Write as if you are different people from the intended deployment population who could have also written some of the seed examples.\n\n"
        "Return only raw CSV text that can be saved directly to a .csv file. "
        "Do not include markdown fences or commentary.\n\n"
        "If any field contains a comma or newline, wrap the entire field in double quotes, and escape any internal double quotes by doubling them.\n\n"
        f"Here is the schema: {schema_copy}"
    )


def _build_baseline_system_prompt() -> str:
    return (
        "You are a helpful assistant that generates evaluation datasets.\n\n"
        "Do not use any punctuation (e.g., em dashes) that are not also present in the seed examples. "
        "Use the seed examples as a calibration reference for authentic voice, domain language, and appropriate detail, but do not reproduce their specific scenarios. "
        "Introduce new situations and perspectives. All examples must be structurally and stylistically distinct. Revise as needed."
    )


def _build_baseline_prompt(seed_combo: list, n: int) -> str:
    return (
        "You are a mental health consultant or educational facilitator writing a brief, "
        "first-person account of a classroom observation. Your account will be used as "
        "input for an AI tool that generates reflective questions for an upcoming meeting "
        "with the teachers involved.\n\n"
        f"Write {n} realistic, conversational prompts from your perspective, as though "
        "you have just observed a classroom scenario and want reflective questions to bring "
        "to a future meeting with the teachers.\n\n"
        "Here are examples written by real mental health consultants and educational facilitators:\n"
        f"{seed_combo}\n\n"
        "Output format requirements: Return only raw text (no markdown code fences, no prose "
        f"before/after). Output exactly {n} data rows, each on a new line. "
        "If any row contains a comma or newline, wrap the entire row in double quotes, and escape any internal double quotes by doubling them."
    )


# ── Schema ablation helpers ──────────────────────────────────────────────────

FIXED_FIELDS_FOR_ABLATION = [
    "capability",
    "systematizedConcept",
    "deploymentPopulation",
    "systematizedInstance",
]

CONSTITUENTS_FOR_ABLATION = [
    "actors",
    "observations",
    "assumptions",
    "early-conclusions",
    "(expected) explanation type",
]


def _remove_constituent(constituents: dict, target: str) -> bool:
    if target in constituents:
        del constituents[target]
        return True
    for key, spec in list(constituents.items()):
        if isinstance(spec, dict) and "subConstituents" in spec:
            removed = _remove_constituent(spec["subConstituents"], target)
            if removed:
                if not spec["subConstituents"]:
                    del constituents[key]
                return True
    return False


def _blank_constituent(constituents: dict, target: str) -> bool:
    """Find `target` in constituents (recursively) and replace its spec with an
    empty-valued placeholder so the name remains but has no values.
    Returns True if target found and blanked.
    """
    if target in constituents:
        spec = constituents[target]
        if isinstance(spec, dict):
            spec.pop("subConstituents", None)
            spec["value"] = []
            constituents[target] = spec
        else:
            constituents[target] = {"value": []}
        return True
    for key, spec in constituents.items():
        if isinstance(spec, dict) and "subConstituents" in spec:
            if _blank_constituent(spec["subConstituents"], target):
                return True
    return False


def _apply_schema_ablation(
    schema: dict,
    missing_fixed: list,
    missing_constituents: list,
    missing_seed_examples: bool,
) -> dict:
    schema_copy = copy.deepcopy(schema)

    fixed_fields = schema_copy.get("fixedFields", {})
    for field in missing_fixed:
        fixed_fields.pop(field, None)

    constituents = schema_copy.get("constituents", {})
    # Instead of removing constituent names entirely, blank their values so
    # the names remain present but carry no values.
    for name in missing_constituents:
        _blank_constituent(constituents, name)
    # Ensure the `constituents` key persists (possibly with blanked values)
    if constituents is not None:
        schema_copy["constituents"] = constituents

    if missing_seed_examples:
        schema_copy.pop("seedExamples", None)

    # Mark this schema as an ablation variant so prompt builders can adapt
    # output instructions (plain text) when appropriate.
    schema_copy["_is_ablation_variant"] = True

    return schema_copy


def build_schema_variants(
    schema: dict, include_seedless_variant: bool
) -> list[dict]:
    """Enumerate schema ablation variants.

    Each variant is a dict with keys: id, missing_fixed, missing_constituents,
    missing_seed_examples, schema.
    """
    fields = [("fixed", name) for name in FIXED_FIELDS_FOR_ABLATION] + [
        ("constituent", name) for name in CONSTITUENTS_FOR_ABLATION
    ]
    field_flags = list(itertools.product([0, 1], repeat=len(fields)))
    seedless_flags = [False, True] if include_seedless_variant else [False]

    variants: list[dict] = []
    variant_id = 0
    for flags in field_flags:
        missing_fixed = [
            name
            for (kind, name), flag in zip(fields, flags)
            if flag and kind == "fixed"
        ]
        missing_constituents = [
            name
            for (kind, name), flag in zip(fields, flags)
            if flag and kind == "constituent"
        ]

        # Tie rule: if more than one item is removed AND either capability or
        # systematizedConcept is selected for removal, remove BOTH together.
        total_missing = len(missing_fixed) + len(missing_constituents)
        if total_missing > 1:
            if ("capability" in missing_fixed) or (
                "systematizedConcept" in missing_fixed
            ):
                if "capability" not in missing_fixed:
                    missing_fixed.append("capability")
                if "systematizedConcept" not in missing_fixed:
                    missing_fixed.append("systematizedConcept")

        # Skip any variant that would remove ALL fixed fields AND ALL
        # constituents, leaving no schema fields at all.
        if len(missing_fixed) == len(FIXED_FIELDS_FOR_ABLATION) and len(
            missing_constituents
        ) == len(CONSTITUENTS_FOR_ABLATION):
            continue

        for seedless in seedless_flags:
            # If nothing is removed (no fixed, no constituents) and seedless is
            # False, skip the trivial no-op variant. If seedless is True, keep
            # the seedless-only variant.
            if not missing_fixed and not missing_constituents and not seedless:
                continue

            schema_variant = _apply_schema_ablation(
                schema,
                missing_fixed=missing_fixed,
                missing_constituents=missing_constituents,
                missing_seed_examples=seedless,
            )

            fixed_remaining = bool(schema_variant.get("fixedFields", {}))
            constituents_remaining = bool(
                schema_variant.get("constituents", {})
            )
            seed_present = bool(schema_variant.get("seedExamples"))

            # Ensure at least one schema field remains
            if not (fixed_remaining or constituents_remaining or seed_present):
                continue

            variants.append(
                {
                    "id": variant_id,
                    "missing_fixed": missing_fixed,
                    "missing_constituents": missing_constituents,
                    "missing_seed_examples": seedless,
                    "schema": schema_variant,
                }
            )
            variant_id += 1

    return variants


# ── Seed combination generation ────────────────────────────────────────────────


def build_seed_combinations(
    seed_examples: list,
    max_s: Optional[int],
    max_seed_reshuffle: int,
    inc_s: int = 1,
    exclude_s: Optional[list] = None,
) -> list[list]:
    """
    Returns a flat list of seed-example subsets for s = 0 .. max_s.
    s=0 contributes exactly one entry: the empty list.
    For each s >= 1, up to max_seed_reshuffle combinations are sampled.
    """
    all_combos: list[list] = []
    if exclude_s is None or 0 not in exclude_s:
        all_combos.append([])  # s=0

    effective_max_s = (
        len(seed_examples) if max_s is None else min(max_s, len(seed_examples))
    )
    s_values = list(range(inc_s, effective_max_s + 1, inc_s))
    if exclude_s is not None:
        s_values = [s for s in s_values if s not in exclude_s]
    for s in s_values:
        combos = list(itertools.combinations(seed_examples, s))
        if len(combos) > max_seed_reshuffle:
            combos = random.sample(combos, max_seed_reshuffle)
        all_combos.extend([list(c) for c in combos])

    return all_combos


# ── API helpers ────────────────────────────────────────────────────────────────


def _call_api(
    user_prompt: str,
    system_prompt: Optional[str],
    provider: str,
    model: str,
    n: int,
    temperature: float = 0.7,
    mult=False,
) -> Optional[str]:
    """Calls ModelAPIClient and returns the response string, or None on failure."""
    max_tokens = min(40000, 200 + 350 * n)

    if mult:
        max_tokens = int(max_tokens * 1.2)

    try:
        result = ModelAPIClient.call_api(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            provider=provider,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            n=1,
            mock=False,
        )
        return result if isinstance(result, str) and result.strip() else None
    except Exception as exc:
        print(f"[ERROR] API call failed ({provider}/{model}): {exc}")
        return None


def _label_text(text: str) -> Optional[str]:
    """Send text to the configured labeling model and return labeled CSV text."""
    if _ACTIVE_LABEL_CONFIG is None:
        raise ValueError(
            "A labeling config is required when labeling is enabled."
        )
    prompt = _build_labeling_prompt(text, _ACTIVE_LABEL_CONFIG)
    # Rough token budget: base overhead + one token per character of input
    max_tokens = min(30000, 500 + len(text) // 3)
    try:
        result = ModelAPIClient.call_api(
            user_prompt=prompt,
            system_prompt=None,
            provider=_LABEL_PROVIDER,
            model=_LABEL_MODEL,
            max_tokens=max_tokens,
            temperature=0.0,
            n=1,
            mock=False,
        )
        return result if isinstance(result, str) and result.strip() else None
    except Exception as exc:
        print(f"[ERROR] Labeling call failed: {exc}")
        return None


def _normalize_header(name: str) -> str:
    return name.strip().lower()


def _parse_label_csv_text(text: str, expected_rows: int) -> pd.DataFrame:
    rows = list(csv.reader(io.StringIO(text)))
    rows = [row for row in rows if any(cell.strip() for cell in row)]
    if not rows:
        raise ValueError("Empty label CSV output")

    header = rows[0]
    header_norm = [_normalize_header(h) for h in header]
    if _ACTIVE_LABEL_CONFIG is None:
        raise ValueError("A labeling config is required when parsing labels.")
    label_columns = _label_columns(_ACTIVE_LABEL_CONFIG)
    expected_norm = [_normalize_header(c) for c in label_columns]

    parsed_rows = []

    if all(name in header_norm for name in expected_norm):
        index_map = {name: header_norm.index(name) for name in expected_norm}
        for row in rows[1:]:
            parsed_rows.append(
                [
                    (
                        row[index_map[name]].strip()
                        if index_map[name] < len(row)
                        else ""
                    )
                    for name in expected_norm
                ]
            )
    else:
        data_rows = rows
        expected_cols = len(label_columns)
        for row in data_rows:
            if len(row) > expected_cols:
                row = row[: expected_cols - 1] + [
                    ",".join(row[expected_cols - 1 :])
                ]
            if len(row) < expected_cols:
                row = row + [""] * (expected_cols - len(row))
            parsed_rows.append([cell.strip() for cell in row])

    df = pd.DataFrame(parsed_rows, columns=label_columns)
    non_empty_mask = df.apply(
        lambda r: any(str(v).strip() for v in r.values), axis=1
    )
    df = df[non_empty_mask].reset_index(drop=True)

    if len(df) > expected_rows:
        df = df.iloc[:expected_rows].reset_index(drop=True)
    elif len(df) < expected_rows:
        pad = pd.DataFrame(
            [[""] * len(label_columns)] * (expected_rows - len(df)),
            columns=label_columns,
        )
        df = pd.concat([df, pad], ignore_index=True)

    return df


# ── File helpers ───────────────────────────────────────────────────────────────


def _ensure_dirs(output_dir: str) -> tuple[Path, Path]:
    schema_dir = Path(output_dir) / "schema"
    baseline_dir = Path(output_dir) / "baseline"
    schema_dir.mkdir(parents=True, exist_ok=True)
    baseline_dir.mkdir(parents=True, exist_ok=True)
    return schema_dir, baseline_dir


def _model_tag(provider: str, model: str) -> str:
    """Filesystem-safe tag combining provider and model name."""
    return f"{provider}_{model}".replace("/", "-").replace(":", "-")


def _extract_example_text(csv_text: str) -> Optional[str]:
    """
    Parses a CSV string and returns the example_text column as plain newline-
    delimited text with no header, suitable for writing to a .txt file.
    """
    try:
        df = pd.read_csv(io.StringIO(csv_text))
        for col in ("example_text", "example text", "text", "prompt"):
            if col in df.columns:
                return "\n".join(df[col].dropna().astype(str).tolist())
        # Fallback: use the column with the longest average value
        if len(df.columns) > 0:
            col_lengths = {
                col: df[col].dropna().astype(str).map(len).mean()
                for col in df.columns
            }
            if col_lengths:
                best_col = max(col_lengths, key=col_lengths.get)
                return "\n".join(df[best_col].dropna().astype(str).tolist())
        return None
    except Exception as exc:
        print(f"[WARN] Could not extract example_text column: {exc}")
        return None


def _label_directory(directory: Path) -> None:
    """Labels every .txt file in *directory*, saving results as *_labeled.csv."""
    txt_files = sorted(directory.glob("*.txt"))
    print(f"  Labeling {len(txt_files)} file(s) in {directory} ...")
    for txt_path in txt_files:
        out_path = directory / f"{txt_path.stem}_labeled.csv"
        if out_path.exists():
            print(f"    [SKIP] {txt_path.name} — already labeled")
            continue
        text = txt_path.read_text(encoding="utf-8").strip()
        if not text:
            print(f"    [SKIP] {txt_path.name} — empty file")
            continue
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            print(f"    [SKIP] {txt_path.name} — no non-empty lines")
            continue
        print(f"    {txt_path.name} ... ", end="", flush=True)
        labeled = _label_text("\n".join(lines))
        if labeled:
            try:
                labels_df = _parse_label_csv_text(labeled, len(lines))
                labels_df.insert(0, "example_text", lines)
                with open(out_path, "w", encoding="utf-8", newline="") as f:
                    labels_df.to_csv(f, index=False, quoting=csv.QUOTE_ALL)
                print("✓")
            except Exception as exc:
                print(f"✗ (CSV post-process error: {exc})")
        else:
            print("✗ (no output returned)")


# ── Core pipeline ──────────────────────────────────────────────────────────────


def run_generation(
    models: list[tuple[str, str]],
    schema_file: str,
    n: int,
    runs: int,
    max_seed_reshuffle: int,
    max_s: Optional[int],
    inc_s: int = 1,
    output_dir: str = "./data/final/",
    do_labeling: bool = False,
    random_seed: int = 42,
    exclude_s: Optional[list] = None,
    schema_ablation: bool = False,
    ablation_seedless: bool = True,
    fixed_seed_count: int = 3,
    label_config_path: Optional[str] = None,
) -> None:
    """
    See module docstring for parameter descriptions.
    """
    global _ACTIVE_LABEL_CONFIG
    _ACTIVE_LABEL_CONFIG = (
        load_labeling_config(label_config_path)
        if do_labeling and label_config_path
        else None
    )
    if do_labeling and _ACTIVE_LABEL_CONFIG is None:
        raise ValueError("label_config_path is required when do_labeling=True.")

    # Set global random seed for reproducibility
    random.seed(random_seed)

    schema: dict = json.load(open(schema_file, "r", encoding="utf-8"))
    seed_examples: list = schema.get("seedExamples", [])

    schema_dir, baseline_dir = _ensure_dirs(output_dir)

    # ── Step 1: Build seed combinations + schema variants ────────────────────
    print(f"Building seed combinations with random_seed={random_seed} ...")
    if schema_ablation:
        variants = build_schema_variants(
            schema, include_seedless_variant=ablation_seedless
        )
        print(f"  {len(variants)} schema variant(s) from ablation\n")
    else:
        variants = [
            {
                "id": 0,
                "missing_fixed": [],
                "missing_constituents": [],
                "missing_seed_examples": False,
                "schema": schema,
            }
        ]

    def _seed_combos_for_variant(
        variant_schema: dict, missing_seed_examples: bool
    ) -> list[list]:
        variant_seed_examples = variant_schema.get("seedExamples", [])
        if not schema_ablation:
            return build_seed_combinations(
                variant_seed_examples,
                max_s,
                max_seed_reshuffle,
                inc_s=inc_s,
                exclude_s=exclude_s,
            )
        # For ablation runs: if seedless variant requested, return empty combo.
        if missing_seed_examples:
            return [[]]
        if not variant_seed_examples:
            return [[]]
        # Fix the same first-k seeds every time for ablation (deterministic).
        k = min(fixed_seed_count, len(variant_seed_examples))
        return [variant_seed_examples[:k]]

    # ── Step 2: Generate for each model ───────────────────────────────────────
    exp_summary = []
    for provider, model in models:
        tag = _model_tag(provider, model)
        print("=" * 60)
        print(f"Model: {provider} / {model}")
        print("=" * 60)

        schema_system_prompt = _build_schema_system_prompt(n, schema)

        for run in range(1, runs + 1):
            print(f"\n  Run {run}/{runs}")

            for variant in variants:
                variant_schema = variant["schema"]
                missing_fixed = variant["missing_fixed"]
                missing_constituents = variant["missing_constituents"]
                missing_seed_examples = variant["missing_seed_examples"]

                variant_seed_examples = variant_schema.get("seedExamples", [])
                seed_example_to_idx = {
                    ex: i for i, ex in enumerate(variant_seed_examples)
                }

                combos = _seed_combos_for_variant(
                    variant_schema, missing_seed_examples
                )
                combo_seed_indices = []
                for combo in combos:
                    combo_seed_indices.append(
                        [seed_example_to_idx[ex] for ex in combo]
                    )

                for combo_idx, seed_combo in enumerate(combos):
                    s = len(seed_combo)
                    stem = (
                        f"{tag}_v{variant['id']:03d}_s{s}_"
                        f"combo{combo_idx:03d}_run{run}"
                    )
                    print(f"    [{stem}] ", end="", flush=True)

                    seed_indices = combo_seed_indices[combo_idx]

                # ── Schema prompt ──────────────────────────────────────────
                schema_filename = f"{stem}.csv"
                schema_path = schema_dir / schema_filename
                if schema_path.exists():
                    print("schema[skip] ", end="", flush=True)
                    exp_summary.append(
                        {
                            "filename": str(schema_path),
                            "seed_idx": seed_indices,
                            "missing_fixed": missing_fixed,
                            "missing_constituents": missing_constituents,
                            "missing_seed_examples": missing_seed_examples,
                            "variant_id": variant["id"],
                        }
                    )
                else:
                    schema_csv = None
                    total_rows = 0
                    csv_parts = []
                    log_written = False
                    # Try to get n rows, retrying as needed
                    while total_rows < n:
                        needed = n - total_rows
                        batch_size = min(needed, 25)
                        prompt = _build_schema_generation_prompt(
                            variant_schema, seed_combo, batch_size
                        )
                        result = _call_api(
                            prompt,
                            schema_system_prompt,
                            provider,
                            model,
                            batch_size,
                            mult=True,
                        )
                        if not result:
                            # Write failed result to log file
                            log_path = schema_dir / f"{stem}.log"
                            log_path.write_text("", encoding="utf-8")
                            log_written = True
                            break
                        # Parse and count rows
                        try:
                            df = pd.read_csv(io.StringIO(result))
                            rows = len(df)
                            if rows == 0:
                                # Write failed result to log file
                                log_path = schema_dir / f"{stem}.log"
                                log_path.write_text(result, encoding="utf-8")
                                log_written = True
                                break
                            csv_parts.append(df)
                            total_rows += rows
                        except Exception as exc:
                            # Write failed result to log file
                            log_path = schema_dir / f"{stem}.log"
                            log_path.write_text(result, encoding="utf-8")
                            log_written = True
                            print(f"[WARN] Could not parse schema CSV: {exc}")
                            break
                    if csv_parts:
                        full_df = pd.concat(csv_parts, ignore_index=True)
                        schema_csv = full_df.to_csv(index=False)
                    if schema_csv:
                        schema_path.write_text(schema_csv, encoding="utf-8")
                        txt_content = _extract_example_text(schema_csv)
                        if txt_content:
                            (schema_dir / f"{stem}.txt").write_text(
                                txt_content, encoding="utf-8"
                            )
                        print("schema✓ ", end="", flush=True)
                        exp_summary.append(
                            {
                                "filename": str(schema_path),
                                "seed_idx": seed_indices,
                                "missing_fixed": missing_fixed,
                                "missing_constituents": missing_constituents,
                                "missing_seed_examples": missing_seed_examples,
                                "variant_id": variant["id"],
                            }
                        )
                    else:
                        if not log_written:
                            log_path = schema_dir / f"{stem}.log"
                            log_path.write_text("", encoding="utf-8")
                        print("schema✗ ", end="", flush=True)

                # ── Baseline prompt ────────────────────────────────────────
                baseline_filename = f"{stem}.txt"
                baseline_path = baseline_dir / baseline_filename
                if baseline_path.exists():
                    print("baseline[skip]", flush=True)
                    exp_summary.append(
                        {
                            "filename": str(baseline_path),
                            "seed_idx": seed_indices,
                            "missing_fixed": missing_fixed,
                            "missing_constituents": missing_constituents,
                            "missing_seed_examples": missing_seed_examples,
                            "variant_id": variant["id"],
                        }
                    )
                else:
                    baseline_txt = None
                    total_rows = 0
                    txt_lines = []
                    log_written = False
                    while total_rows < n:
                        needed = n - total_rows
                        batch_size = min(needed, 25)
                        prompt = _build_baseline_prompt(seed_combo, batch_size)
                        result = _call_api(
                            prompt,
                            _build_baseline_system_prompt(),
                            provider,
                            model,
                            batch_size,
                        )
                        if not result:
                            log_path = baseline_dir / f"{stem}.log"
                            log_path.write_text("", encoding="utf-8")
                            log_written = True
                            break
                        # Parse lines, handling quoted rows with newlines/commas
                        import csv
                        from io import StringIO

                        try:
                            reader = csv.reader(
                                StringIO(result), skipinitialspace=True
                            )
                            for row in reader:
                                # Join all columns (should be one per row, but just in case)
                                line = ",".join(row).strip()
                                if line:
                                    txt_lines.append(line)
                            total_rows = len(txt_lines)
                        except Exception as exc:
                            log_path = baseline_dir / f"{stem}.log"
                            log_path.write_text(result, encoding="utf-8")
                            log_written = True
                            print(
                                f"[WARN] Could not parse baseline text as CSV: {exc}"
                            )
                            # Fallback: split by lines
                            lines = [
                                line
                                for line in result.strip().splitlines()
                                if line.strip()
                            ]
                            txt_lines.extend(lines)
                            total_rows = len(txt_lines)
                    if txt_lines:
                        # Truncate to n lines if over
                        baseline_txt = "\n".join(txt_lines)
                    if baseline_txt:
                        baseline_path.write_text(baseline_txt, encoding="utf-8")
                        print("baseline✓", flush=True)
                        exp_summary.append(
                            {
                                "filename": str(baseline_path),
                                "seed_idx": seed_indices,
                                "missing_fixed": missing_fixed,
                                "missing_constituents": missing_constituents,
                                "missing_seed_examples": missing_seed_examples,
                                "variant_id": variant["id"],
                            }
                        )
                    else:
                        if not log_written:
                            log_path = baseline_dir / f"{stem}.log"
                            log_path.write_text("", encoding="utf-8")
                        print("baseline✗", flush=True)

    # ── Step 3: Label all generated .txt files (optional) ─────────────────────
    if do_labeling:
        print(f"\n{'=' * 60}")
        print("Labeling generated outputs with Claude Haiku ...")
        print(f"{'=' * 60}\n")

        _label_directory(schema_dir)
        _label_directory(baseline_dir)

        # Label the original schema seed examples
        if seed_examples:
            seed_out = Path(output_dir) / "seed_examples_labeled.csv"
            print(
                f"\n  Labeling {len(seed_examples)} schema seed example(s) ... ",
                end="",
                flush=True,
            )
            labeled_seeds = _label_text("\n".join(seed_examples))
            if labeled_seeds:
                try:
                    labels_df = _parse_label_csv_text(
                        labeled_seeds, len(seed_examples)
                    )
                    labels_df.insert(0, "example_text", seed_examples)
                    with open(seed_out, "w", encoding="utf-8", newline="") as f:
                        labels_df.to_csv(f, index=False, quoting=csv.QUOTE_ALL)
                    print(f"✓  →  {seed_out}")
                except Exception as exc:
                    print(f"✗ (CSV post-process error: {exc})")
            else:
                print("✗ (no output returned)")

    # ── Step 4: Output exp_summary.csv ───────────────────────────────────────
    import csv

    exp_summary_name = (
        "exp_summary_schema_ablation.csv"
        if schema_ablation
        else "exp_summary.csv"
    )
    exp_summary_path = Path(output_dir) / exp_summary_name
    # Append to summary file if it exists, otherwise write header
    write_header = not exp_summary_path.exists()
    with open(exp_summary_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            if schema_ablation:
                writer.writerow(
                    [
                        "filename",
                        "seed_idx",
                        "missing_fixed",
                        "missing_constituents",
                        "missing_seed_examples",
                        "variant_id",
                    ]
                )
            else:
                writer.writerow(["filename", "seed_idx"])
        for entry in exp_summary:
            if schema_ablation:
                writer.writerow(
                    [
                        entry["filename"],
                        str(entry["seed_idx"]),
                        str(entry.get("missing_fixed", [])),
                        str(entry.get("missing_constituents", [])),
                        str(entry.get("missing_seed_examples", False)),
                        str(entry.get("variant_id", 0)),
                    ]
                )
            else:
                writer.writerow([entry["filename"], str(entry["seed_idx"])])

    print(f"\nWrote experiment summary to {exp_summary_path}")
    print("\nDone.")


# ── CLI entry point ───────────────────────────────────────────────────────────


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate schema-driven benchmark examples and optional labeled outputs."
    )
    parser.add_argument(
        "--schema-file",
        required=True,
        help="Path to the schema JSON file.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Root output directory for schema/baseline outputs.",
    )
    parser.add_argument(
        "--provider",
        default="anthropic",
        choices=["anthropic", "openai"],
        help="API provider for generation.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_GENERATION_MODELS[0][1],
        help="Model name for the selected provider.",
    )
    parser.add_argument(
        "--n", type=int, default=100, help="Examples to generate per batch."
    )
    parser.add_argument(
        "--runs", type=int, default=1, help="Number of generation runs."
    )
    parser.add_argument(
        "--max-seed-reshuffle",
        type=int,
        default=1,
        help="Maximum number of seed reshuffles to sample.",
    )
    parser.add_argument(
        "--max-s",
        type=int,
        default=None,
        help="Maximum number of seed examples in a prompt. Use None for all available examples.",
    )
    parser.add_argument(
        "--inc-s", type=int, default=2, help="Seed count increment."
    )
    parser.add_argument(
        "--exclude-s",
        nargs="*",
        default=[0],
        help="Seed counts to exclude (e.g. --exclude-s 0 1).",
    )
    parser.add_argument(
        "--do-labeling",
        action="store_true",
        help="Run labeling on the generated schema and baseline outputs.",
    )
    parser.add_argument(
        "--label-config",
        help="Path to a JSON labeling configuration. Required with --do-labeling.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for reproducible prompt generation.",
    )
    parser.add_argument(
        "--schema-ablation",
        action="store_true",
        help="Enable schema ablation runs.",
    )
    parser.add_argument(
        "--no-ablation-seedless",
        action="store_false",
        dest="ablation_seedless",
        help="Disable the seedless ablation variant when ablating.",
    )
    parser.add_argument(
        "--fixed-seed-count",
        type=int,
        default=3,
        help="Seed count for schema ablation runs.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    models = [(args.provider, args.model)]
    exclude_s = [int(value) for value in args.exclude_s]

    run_generation(
        models=models,
        schema_file=args.schema_file,
        n=args.n,
        runs=args.runs,
        max_seed_reshuffle=args.max_seed_reshuffle,
        max_s=args.max_s,
        inc_s=args.inc_s,
        output_dir=args.output_dir,
        do_labeling=args.do_labeling,
        random_seed=args.random_seed,
        exclude_s=exclude_s,
        schema_ablation=args.schema_ablation,
        ablation_seedless=getattr(args, "ablation_seedless", True),
        fixed_seed_count=args.fixed_seed_count,
        label_config_path=args.label_config,
    )


if __name__ == "__main__":
    main()
