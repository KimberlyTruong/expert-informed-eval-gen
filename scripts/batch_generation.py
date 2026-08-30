"""
Batch dataset generation using the OpenAI or Anthropic Batch API.

This script is intentionally separate from the synchronous generator.
It uses a two-step flow:

1) submit: build chunked requests (for example, 25 rows each), upload JSONL,
   and create a batch job.
2) collect: once the batch is complete, download results and combine chunk
   outputs into final files under schema/ and baseline/.

No retry loop is used. Failed/malformed chunks are logged for manual reruns.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import random
import shutil
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from openai import OpenAI

from anthropic import Anthropic
from anthropic.types.message_create_params import (
    MessageCreateParamsNonStreaming,
)
from anthropic.types.messages.batch_create_params import (
    Request as AnthropicRequest,
)

from generation import (
    _LABEL_MODEL,
    _LABEL_PROVIDER,
    ModelAPIClient,
    _build_baseline_prompt,
    _build_labeling_prompt,
    _build_schema_generation_prompt,
    _build_schema_system_prompt,
    _ensure_dirs,
    _extract_example_text,
    _model_tag,
    build_schema_variants,
    build_seed_combinations,
    load_labeling_config,
)
from helpers.project_config import DEFAULT_OUTPUT_DIR, DEFAULT_PARENT_DIRS
import hashlib
from csv_repair import repair_csv_last_column_overflow


def _make_custom_id(kind: str, stem: str, chunk_index: int) -> str:
    """Create a custom_id no longer than 64 characters (Anthropic limit).

    Format: "{kind}__{stem}__chunk{chunk_index:03d}" but if that exceeds
    64 chars we truncate `stem` and append a short hash to keep it unique.
    """
    suffix = f"__chunk{chunk_index:03d}"
    prefix = f"{kind}__"
    max_total = 64
    # Reserve space for prefix + suffix
    max_stem = max_total - (len(prefix) + len(suffix))
    if max_stem <= 0:
        # Edge case: fall back to hashed id
        h = hashlib.sha1(stem.encode("utf-8")).hexdigest()[:8]
        return f"{kind}__{h}{suffix}"

    if len(stem) <= max_stem:
        return f"{prefix}{stem}{suffix}"

    # Truncate stem and append short hash to retain uniqueness
    h = hashlib.sha1(stem.encode("utf-8")).hexdigest()[:8]
    # Reserve 1 char for separator between truncated stem and hash
    trunc_target = max_stem - 1 - len(h)
    if trunc_target <= 0:
        return f"{kind}__{h}{suffix}"
    short_stem = stem[:trunc_target]
    return f"{prefix}{short_stem}_{h}{suffix}"


@dataclass
class RequestMeta:
    custom_id: str
    kind: str  # schema | baseline
    provider: str
    model: str
    stem: str
    chunk_index: int
    chunk_size: int
    seed_idx: List[int]
    missing_fixed: Optional[List[str]] = None
    missing_constituents: Optional[List[str]] = None
    missing_seed_examples: Optional[bool] = None
    variant_id: Optional[int] = None


@dataclass
class LabelRequestMeta:
    custom_id: str
    provider: str
    model: str
    source_txt: str
    target_csv: str


def _parse_exclude_s(exclude_s_raw: str) -> List[int]:
    if not exclude_s_raw.strip():
        return []
    values = []
    for token in exclude_s_raw.split(","):
        token = token.strip()
        if token:
            values.append(int(token))
    return values


def _max_tokens_for_chunk(kind: str, chunk_size: int) -> int:
    base = min(40000, 200 + 350 * chunk_size)
    if kind == "schema":
        return int(base * 1.2)
    return base


def _make_messages(
    system_prompt: Optional[str], user_prompt: str
) -> List[Dict[str, str]]:
    messages: List[Dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})
    return messages


def _make_anthropic_params(
    system_prompt: Optional[str], user_prompt: str, model: str, max_tokens: int
) -> Dict[str, Any]:
    """Build Anthropic Message API params dict."""
    params: Dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [
            {
                "role": "user",
                "content": user_prompt,
            }
        ],
    }

    if system_prompt:
        params["system"] = system_prompt

    return params


def _extract_response_text(response_body: Dict[str, Any]) -> Optional[str]:
    """
    Extract text from Responses API output body in a defensive way.
    """
    if not isinstance(response_body, dict):
        return None

    output_text = response_body.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text

    output_items = response_body.get("output", [])
    if isinstance(output_items, list):
        texts: List[str] = []
        for item in output_items:
            if not isinstance(item, dict):
                continue
            content = item.get("content", [])
            if not isinstance(content, list):
                continue
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") in {"output_text", "text"}:
                    text_val = block.get("text") or block.get("value")
                    if isinstance(text_val, str):
                        texts.append(text_val)
        joined = "".join(texts).strip()
        if joined:
            return joined

    choices = response_body.get("choices")
    if isinstance(choices, list) and choices:
        message = (
            choices[0].get("message", {})
            if isinstance(choices[0], dict)
            else {}
        )
        content = message.get("content") if isinstance(message, dict) else None
        if isinstance(content, str) and content.strip():
            return content

    return None


def _read_batch_output_text(client: OpenAI, output_file_id: str) -> str:
    data = client.files.content(output_file_id)

    text_attr = getattr(data, "text", None)
    if isinstance(text_attr, str):
        return text_attr

    content_attr = getattr(data, "content", None)
    if isinstance(content_attr, (bytes, bytearray)):
        return bytes(content_attr).decode("utf-8")

    read_fn = getattr(data, "read", None)
    if callable(read_fn):
        raw = read_fn()
        if isinstance(raw, (bytes, bytearray)):
            return bytes(raw).decode("utf-8")
        if isinstance(raw, str):
            return raw

    return str(data)


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def _chunk_sizes(total_n: int, chunk_size: int) -> List[int]:
    chunks = []
    remaining = total_n
    while remaining > 0:
        c = min(chunk_size, remaining)
        chunks.append(c)
        remaining -= c
    return chunks


def _label_output_path(txt_path: Path) -> Path:
    return txt_path.with_name(f"{txt_path.stem}_labeled.csv")


def _safe_label_custom_id(txt_path: Path) -> str:
    path_str = str(txt_path.resolve())
    sanitized = re.sub(r"[^a-zA-Z0-9_-]+", "_", txt_path.stem)
    short_hash = hashlib.sha1(path_str.encode("utf-8")).hexdigest()[:10]
    max_stem_len = 64 - len("label__") - len("__") - len(short_hash)
    if max_stem_len < 1:
        return f"label__{short_hash}"
    if len(sanitized) > max_stem_len:
        sanitized = sanitized[:max_stem_len]
    custom_id = f"label__{sanitized}__{short_hash}"
    return custom_id[:64]


def _discover_unlabeled_txt_files(
    parent_dirs: List[str], recursive: bool = True
) -> List[Tuple[Path, Path]]:
    files: List[Tuple[Path, Path]] = []
    for txt_path, labeled_path, label_exists in _discover_label_targets(
        parent_dirs, recursive=recursive
    ):
        if not label_exists:
            files.append((txt_path, labeled_path))
    return files


def _discover_label_targets(
    parent_dirs: List[str], recursive: bool = True
) -> List[Tuple[Path, Path, bool]]:
    files: List[Tuple[Path, Path, bool]] = []
    for parent_dir in parent_dirs:
        root = Path(parent_dir)
        if not root.exists():
            print(f"[WARN] Skipping missing directory: {root}")
            continue
        if recursive:
            iterator = root.rglob("*.txt")
        else:
            iterator = root.glob("*.txt")
        for txt_path in sorted(iterator):
            labeled_path = _label_output_path(txt_path)
            files.append((txt_path, labeled_path, labeled_path.exists()))
    return files


def _archive_existing_label_file(labeled_path: Path) -> Optional[Path]:
    if not labeled_path.exists():
        return None

    old_dir = labeled_path.parent / "old"
    old_dir.mkdir(parents=True, exist_ok=True)

    archive_path = old_dir / labeled_path.name
    if archive_path.exists():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_path = (
            old_dir / f"{labeled_path.stem}__{timestamp}{labeled_path.suffix}"
        )
        counter = 1
        while archive_path.exists():
            archive_path = old_dir / (
                f"{labeled_path.stem}__{timestamp}_{counter}{labeled_path.suffix}"
            )
            counter += 1

    shutil.move(str(labeled_path), str(archive_path))
    return archive_path


def scan_missing_label_files(
    parent_dirs: List[str], recursive: bool = True
) -> List[Path]:
    return [
        txt_path
        for txt_path, _ in _discover_unlabeled_txt_files(
            parent_dirs, recursive=recursive
        )
    ]


def _max_tokens_for_label_text(text: str) -> int:
    # Remove the artificial cap; let caller/model/provider limits apply.
    return 500 + len(text) // 3


def _write_labeled_csv(
    text: str, out_path: Path, source_txt: Optional[str] = None
) -> None:
    try:
        df = pd.read_csv(io.StringIO(text))
        for col in ("example_text", "example text", "text", "prompt"):
            if col in df.columns:
                with open(out_path, "w", encoding="utf-8", newline="") as f:
                    df.to_csv(f, index=False, quoting=csv.QUOTE_ALL)
                return

        # If the labeling model returned label columns only, but we have the
        # original source text file, merge the example text as the first column
        # and write a fully quoted CSV so the example_text is preserved.
        if source_txt:
            try:
                src_lines = (
                    Path(source_txt).read_text(encoding="utf-8").splitlines()
                )
                src_lines = [l for l in src_lines if l.strip()]
            except Exception:
                src_lines = []

            # Ensure row counts align: pad or truncate as needed
            n_labels = len(df)
            if len(src_lines) < n_labels:
                src_lines.extend([""] * (n_labels - len(src_lines)))
            elif len(src_lines) > n_labels:
                src_lines = src_lines[:n_labels]

            if n_labels > 0:
                df.insert(0, "example_text", src_lines)
                with open(out_path, "w", encoding="utf-8", newline="") as f:
                    df.to_csv(f, index=False, quoting=csv.QUOTE_ALL)
                return

        out_path.write_text(text, encoding="utf-8")
    except Exception:
        out_path.write_text(text, encoding="utf-8")


def _build_label_requests(
    parent_dirs: List[str],
    recursive: bool = True,
    include_existing: bool = False,
    labeling_config: Optional[dict] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    requests: List[Dict[str, Any]] = []
    manifest_entries: Dict[str, Dict[str, Any]] = {}

    for txt_path, labeled_path, label_exists in _discover_label_targets(
        parent_dirs, recursive=recursive
    ):
        if label_exists and not include_existing:
            continue

        text = txt_path.read_text(encoding="utf-8").strip()
        if not text:
            print(f"[SKIP] empty file: {txt_path}")
            continue

        custom_id = _safe_label_custom_id(txt_path)
        if labeling_config is None:
            raise ValueError(
                "A labeling config is required for label requests."
            )
        prompt = _build_labeling_prompt(text, labeling_config)
        requests.append(
            {
                "custom_id": custom_id,
                "params": _make_anthropic_params(
                    None,
                    prompt,
                    _LABEL_MODEL,
                    _max_tokens_for_label_text(text),
                ),
            }
        )
        manifest_entries[custom_id] = LabelRequestMeta(
            custom_id=custom_id,
            provider=_LABEL_PROVIDER,
            model=_LABEL_MODEL,
            source_txt=str(txt_path.resolve()),
            target_csv=str(labeled_path.resolve()),
        ).__dict__

    return requests, manifest_entries


def _submit_label_batch(
    args: argparse.Namespace,
    include_existing: bool,
    archive_existing: bool,
    kind: str,
) -> None:
    requests, manifest_entries = _build_label_requests(
        args.parent_dirs,
        recursive=getattr(args, "recursive", True),
        include_existing=include_existing,
        labeling_config=load_labeling_config(args.label_config),
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_dir) / "batch_runs" / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = run_dir / "requests.jsonl"
    manifest_path = run_dir / "manifest.json"

    _write_jsonl(jsonl_path, requests)

    manifest = {
        "created_at": timestamp,
        "output_dir": str(Path(args.output_dir).resolve()),
        "parent_dirs": [str(Path(path).resolve()) for path in args.parent_dirs],
        "kind": kind,
        "provider": _LABEL_PROVIDER,
        "model": _LABEL_MODEL,
        "batch_endpoint": "/v1/messages",
        "request_count": len(requests),
        "entries": manifest_entries,
    }

    if not requests:
        manifest["batch_status"] = "no_requests"
        manifest_path.write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        print("No txt files found. Nothing to submit.")
        print(f"Run directory: {run_dir}")
        return

    if args.dry_run:
        manifest["dry_run"] = True
        manifest_path.write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        print(f"Dry run complete. Created:\n  {jsonl_path}\n  {manifest_path}")
        return

    if archive_existing:
        for txt_path, labeled_path, label_exists in _discover_label_targets(
            args.parent_dirs, recursive=getattr(args, "recursive", True)
        ):
            if not label_exists:
                continue
            archived_path = _archive_existing_label_file(labeled_path)
            if archived_path:
                print(f"[ARCHIVE] {labeled_path} -> {archived_path}")

    client = Anthropic()
    batch = client.messages.batches.create(requests=requests)

    manifest["batch_id"] = batch.id
    manifest["batch_status"] = batch.processing_status
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("Label batch submitted to Anthropic Batch API.")
    print(f"Run directory: {run_dir}")
    print(f"Batch ID: {batch.id}")
    print("Use the label-collect command to retrieve results once complete.")


def _build_requests(
    models: List[Tuple[str, str]],
    schema: Dict[str, Any],
    n: int,
    runs: int,
    max_seed_reshuffle: int,
    max_s: int,
    inc_s: int,
    exclude_s: Optional[List[int]],
    random_seed: int,
    chunk_size: int,
    output_dir: str,
    schema_ablation: bool,
    ablation_seedless: bool,
    fixed_seed_count: int,
    append_system_prompt: Optional[str] = None,
    append_user_prompt: Optional[str] = None,
    force: bool = False,
    rerun_filenames: Optional[List[str]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    random.seed(random_seed)

    if schema_ablation:
        variants = build_schema_variants(
            schema, include_seedless_variant=ablation_seedless
        )
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

    schema_dir, baseline_dir = _ensure_dirs(output_dir)

    requests: List[Dict[str, Any]] = []
    manifest_entries: Dict[str, Dict[str, Any]] = {}

    for provider, model in models:
        tag = _model_tag(provider, model)
        for run in range(1, runs + 1):
            for variant in variants:
                variant_schema = variant["schema"]
                missing_fixed = variant["missing_fixed"]
                missing_constituents = variant["missing_constituents"]
                missing_seed_examples = variant["missing_seed_examples"]

                variant_seed_examples = variant_schema.get("seedExamples", [])
                if schema_ablation:
                    if missing_seed_examples:
                        combos = [[]]
                    elif not variant_seed_examples:
                        combos = [[]]
                    else:
                        k = min(fixed_seed_count, len(variant_seed_examples))
                        combos = [variant_seed_examples[:k]]
                else:
                    combos = build_seed_combinations(
                        variant_seed_examples,
                        max_s,
                        max_seed_reshuffle,
                        inc_s=inc_s,
                        exclude_s=exclude_s,
                    )

                seed_example_to_idx = {
                    ex: i for i, ex in enumerate(variant_seed_examples)
                }
                combo_seed_indices: List[List[int]] = []
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
                    seed_indices = combo_seed_indices[combo_idx]

                    # Check once if ANY file with matching s+run exists (ignore combo number)
                    s_run_pattern = f"{tag}_s{s}_*_run{run}.csv"
                    schema_exists = bool(list(schema_dir.glob(s_run_pattern)))
                    baseline_pattern = f"{tag}_s{s}_*_run{run}.txt"
                    baseline_exists = bool(
                        list(baseline_dir.glob(baseline_pattern))
                    )

                    # If a rerun file was provided, only generate stems
                    # that appear in that list. This helps target only the
                    # failed outputs the user wants to re-run.
                    if rerun_filenames is not None:
                        expected_txt = f"{stem}.txt"
                        expected_csv = f"{stem}.csv"
                        found = False
                        for p in rerun_filenames:
                            if p.endswith(expected_txt) or p.endswith(
                                expected_csv
                            ):
                                found = True
                                break
                            # also allow matching by bare filename
                            if Path(p).name in (expected_txt, expected_csv):
                                found = True
                                break
                        if not found:
                            continue

                    for chunk_index, this_chunk_n in enumerate(
                        _chunk_sizes(n, chunk_size)
                    ):
                        schema_system_prompt = _build_schema_system_prompt(
                            this_chunk_n, variant_schema
                        )
                        schema_user_prompt = _build_schema_generation_prompt(
                            schema=variant_schema,
                            seed_combo=seed_combo,
                            n=this_chunk_n,
                        )
                        # Append any supplied prompt additions
                        if append_system_prompt:
                            schema_system_prompt = (
                                schema_system_prompt
                                + "\n"
                                + append_system_prompt
                            )
                        if append_user_prompt:
                            schema_user_prompt = (
                                schema_user_prompt + "\n" + append_user_prompt
                            )
                        schema_id = _make_custom_id("schema", stem, chunk_index)
                        # If `force` is set, treat existing files as missing so
                        # we re-generate even when a (partial) output exists.
                        should_generate = (not schema_exists) or force
                        if not should_generate:
                            if chunk_index == 0:
                                print(f"[SKIP] schema s={s} run={run} exists")
                        else:
                            if provider == "openai":
                                requests.append(
                                    {
                                        "custom_id": schema_id,
                                        "method": "POST",
                                        "url": "/v1/responses",
                                        "body": {
                                            "model": model,
                                            "input": _make_messages(
                                                schema_system_prompt,
                                                schema_user_prompt,
                                            ),
                                            "max_output_tokens": _max_tokens_for_chunk(
                                                "schema", this_chunk_n
                                            ),
                                            "reasoning": {"effort": "low"},
                                        },
                                    }
                                )
                            else:  # anthropic
                                requests.append(
                                    {
                                        "custom_id": schema_id,
                                        "params": _make_anthropic_params(
                                            schema_system_prompt,
                                            schema_user_prompt,
                                            model,
                                            _max_tokens_for_chunk(
                                                "schema", this_chunk_n
                                            ),
                                        ),
                                    }
                                )
                            manifest_entries[schema_id] = RequestMeta(
                                custom_id=schema_id,
                                kind="schema",
                                provider=provider,
                                model=model,
                                stem=stem,
                                chunk_index=chunk_index,
                                chunk_size=this_chunk_n,
                                seed_idx=seed_indices,
                                missing_fixed=missing_fixed,
                                missing_constituents=missing_constituents,
                                missing_seed_examples=missing_seed_examples,
                                variant_id=variant["id"],
                            ).__dict__

                        # If schema ablation is enabled, skip generating baseline
                        # requests — baselines are unrelated to ablation experiments.
                        if not schema_ablation:
                            baseline_user_prompt = _build_baseline_prompt(
                                seed_combo, this_chunk_n
                            )
                            baseline_id = _make_custom_id(
                                "baseline", stem, chunk_index
                            )
                            if baseline_exists:
                                if chunk_index == 0:
                                    print(
                                        f"[SKIP] baseline s={s} run={run} exists"
                                    )
                            else:
                                if provider == "openai":
                                    requests.append(
                                        {
                                            "custom_id": baseline_id,
                                            "method": "POST",
                                            "url": "/v1/responses",
                                            "body": {
                                                "model": model,
                                                "input": _make_messages(
                                                    "You are a helpful assistant.",
                                                    baseline_user_prompt,
                                                ),
                                                "max_output_tokens": _max_tokens_for_chunk(
                                                    "baseline", this_chunk_n
                                                ),
                                                "reasoning": {"effort": "low"},
                                            },
                                        }
                                    )
                                else:  # anthropic
                                    requests.append(
                                        {
                                            "custom_id": baseline_id,
                                            "params": _make_anthropic_params(
                                                "You are a helpful assistant.",
                                                baseline_user_prompt,
                                                model,
                                                _max_tokens_for_chunk(
                                                    "baseline", this_chunk_n
                                                ),
                                            ),
                                        }
                                    )
                                manifest_entries[baseline_id] = RequestMeta(
                                    custom_id=baseline_id,
                                    kind="baseline",
                                    provider=provider,
                                    model=model,
                                    stem=stem,
                                    chunk_index=chunk_index,
                                    chunk_size=this_chunk_n,
                                    seed_idx=seed_indices,
                                    missing_fixed=missing_fixed,
                                    missing_constituents=missing_constituents,
                                    missing_seed_examples=missing_seed_examples,
                                    variant_id=variant["id"],
                                ).__dict__

    return requests, manifest_entries


def submit_label_batch(args: argparse.Namespace) -> None:
    _submit_label_batch(
        args,
        include_existing=False,
        archive_existing=False,
        kind="label",
    )


def relabel_all_label_batch(args: argparse.Namespace) -> None:
    _submit_label_batch(
        args,
        include_existing=True,
        archive_existing=True,
        kind="relabel",
    )


def print_missing_label_files(args: argparse.Namespace) -> None:
    missing_files = scan_missing_label_files(
        args.parent_dirs, recursive=getattr(args, "recursive", True)
    )
    if not missing_files:
        print("No unlabeled txt files found.")
        return

    for txt_path in missing_files:
        print(txt_path)


def collect_label_batch(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir)
    manifest = _load_manifest(run_dir)

    batch_id = manifest.get("batch_id")
    if not batch_id:
        raise ValueError("manifest.json does not contain batch_id")

    client = Anthropic()
    batch = client.messages.batches.retrieve(batch_id)
    manifest["batch_status"] = batch.processing_status

    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    status = getattr(batch, "processing_status", None)
    print(f"Batch status: {status}")
    if status != "ended":
        if str(status).lower() in ("failed", "errored", "canceled", "expired"):
            print("Batch failed/errored. Inspect batch details for errors.")
            print(f"Result count: {getattr(batch, 'result_count', None)}")
            print(f"Error count: {getattr(batch, 'error_count', None)}")
            return
        print("Batch is not complete yet. Re-run collect later.")
        return

    parsed_results: Dict[str, str] = {}
    failed: List[Dict[str, Any]] = []

    for result in client.messages.batches.results(batch_id):
        custom_id = result.custom_id
        if result.result.type == "succeeded":
            message = result.result.message
            text = ""
            if message.content:
                for block in message.content:
                    if hasattr(block, "text"):
                        text += block.text
            if text:
                parsed_results[custom_id] = text
            else:
                failed.append(
                    {
                        "custom_id": custom_id,
                        "status": "succeeded",
                        "error": "No text extracted from message",
                    }
                )
        elif result.result.type == "errored":
            failed.append(
                {
                    "custom_id": custom_id,
                    "status": "errored",
                    "error": str(result.result.error),
                }
            )
        elif result.result.type == "expired":
            failed.append(
                {
                    "custom_id": custom_id,
                    "status": "expired",
                    "error": "Request expired",
                }
            )
        elif result.result.type == "canceled":
            failed.append(
                {
                    "custom_id": custom_id,
                    "status": "canceled",
                    "error": "Request was canceled",
                }
            )

    exp_summary: List[Dict[str, Any]] = []
    for custom_id, text in parsed_results.items():
        meta = manifest["entries"].get(custom_id)
        if not meta:
            failed.append(
                {
                    "custom_id": custom_id,
                    "error": "Missing metadata for custom_id",
                }
            )
            continue

        out_path = Path(meta["target_csv"])
        try:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            _write_labeled_csv(text, out_path, meta.get("source_txt"))
            exp_summary.append(
                {"filename": str(out_path), "source_txt": meta["source_txt"]}
            )
        except Exception as exc:
            failed.append(
                {
                    "custom_id": custom_id,
                    "error": f"Failed to write labeled CSV: {exc}",
                }
            )

    exp_summary_path = Path(manifest["output_dir"]) / "label_exp_summary.csv"
    write_header = not exp_summary_path.exists()
    with exp_summary_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["filename", "source_txt"])
        for entry in exp_summary:
            writer.writerow([entry["filename"], entry["source_txt"]])

    failures_path = run_dir / "label_failures.json"
    failures_path.write_text(json.dumps(failed, indent=2), encoding="utf-8")

    print(f"Wrote {len(exp_summary)} labeled file(s).")
    print(f"Updated summary: {exp_summary_path}")
    if failed:
        print(
            f"Encountered {len(failed)} failed request(s). See: {failures_path}"
        )
    else:
        print("No failed requests.")


def submit_batch(args: argparse.Namespace) -> None:
    schema = json.loads(Path(args.schema_file).read_text(encoding="utf-8"))
    rerun_filenames: Optional[List[str]] = None
    if getattr(args, "rerun_file", ""):
        try:
            rerun_filenames = [
                l.strip()
                for l in Path(args.rerun_file)
                .read_text(encoding="utf-8")
                .splitlines()
                if l.strip()
            ]
        except Exception:
            rerun_filenames = None

    requests, manifest_entries = _build_requests(
        models=[(args.provider, args.model)],
        schema=schema,
        n=args.n,
        runs=args.runs,
        max_seed_reshuffle=args.max_seed_reshuffle,
        max_s=args.max_s,
        inc_s=args.inc_s,
        exclude_s=_parse_exclude_s(args.exclude_s),
        random_seed=args.random_seed,
        chunk_size=args.chunk_size,
        output_dir=args.output_dir,
        schema_ablation=args.schema_ablation,
        ablation_seedless=args.ablation_seedless,
        fixed_seed_count=args.fixed_seed_count,
        append_system_prompt=(args.append_system_prompt or None),
        append_user_prompt=(args.append_user_prompt or None),
        force=args.force,
        rerun_filenames=rerun_filenames,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_dir) / "batch_runs" / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = run_dir / "requests.jsonl"
    manifest_path = run_dir / "manifest.json"

    _write_jsonl(jsonl_path, requests)

    manifest = {
        "created_at": timestamp,
        "schema_file": args.schema_file,
        "output_dir": str(Path(args.output_dir).resolve()),
        "model": args.model,
        "batch_endpoint": "/v1/responses",
        "completion_window": args.completion_window,
        "n": args.n,
        "chunk_size": args.chunk_size,
        "runs": args.runs,
        "max_seed_reshuffle": args.max_seed_reshuffle,
        "max_s": args.max_s,
        "inc_s": args.inc_s,
        "exclude_s": _parse_exclude_s(args.exclude_s),
        "random_seed": args.random_seed,
        "schema_ablation": args.schema_ablation,
        "ablation_seedless": args.ablation_seedless,
        "fixed_seed_count": args.fixed_seed_count,
        "request_count": len(requests),
        "entries": manifest_entries,
    }

    if args.dry_run:
        manifest["dry_run"] = True
        manifest_path.write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        print(f"Dry run complete. Created:\n  {jsonl_path}\n  {manifest_path}")
        return

    if args.provider == "openai":
        client = OpenAI()
        with jsonl_path.open("rb") as f:
            upload = client.files.create(file=f, purpose="batch")

        batch = client.batches.create(
            input_file_id=upload.id,
            endpoint="/v1/responses",
            completion_window=args.completion_window,
        )

        manifest["input_file_id"] = upload.id
        manifest["batch_id"] = batch.id
        manifest["batch_status"] = batch.status
        manifest_path.write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )

        print("Batch submitted.")
        print(f"Run directory: {run_dir}")
        print(f"Batch ID: {batch.id}")
        print("Use the collect command later to materialize outputs.")
    elif args.provider == "anthropic":
        client = Anthropic()

        # Convert OpenAI Batch API format to Anthropic Batch API format
        # OpenAI format has "body" dict, Anthropic format uses AnthropicRequest with params
        anthropic_requests = []
        for req in requests:
            if "body" in req and "url" in req:
                # This is OpenAI format, skip (shouldn't happen for Anthropic provider)
                continue
            # Anthropic format request
            anthropic_requests.append(req)

        # Submit to Anthropic Batch API
        batch = client.messages.batches.create(requests=anthropic_requests)

        manifest["local_batch"] = True
        manifest["batch_id"] = batch.id
        manifest["batch_status"] = batch.processing_status
        manifest["provider"] = "anthropic"
        manifest_path.write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )

        print("Anthropic batch submitted to Batch API.")
        print(f"Run directory: {run_dir}")
        print(f"Batch ID: {batch.id}")
        print("Use the collect command to retrieve results once complete.")
    else:
        raise ValueError(f"Unsupported provider: {args.provider}")


def _load_manifest(run_dir: Path) -> Dict[str, Any]:
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _combine_schema_chunks(chunk_texts: List[str]) -> Optional[str]:
    """Combine schema CSV chunks, with intelligent handling of last-column overflow.

    If the last column is 'example_text' or 'example text' and contains commas/newlines,
    those may parse as extra fields. This function detects and repairs such cases by
    re-escaping and re-quoting the last column.
    """
    frames: List[pd.DataFrame] = []

    for text in chunk_texts:
        # Try normal parse first
        try:
            df = pd.read_csv(io.StringIO(text))
            if len(df) == 0:
                continue
            frames.append(df)
        except Exception as parse_err:
            # If parse fails, try to repair last-column overflow
            repaired = repair_csv_last_column_overflow(text)
            if repaired:
                try:
                    df = pd.read_csv(io.StringIO(repaired))
                    if len(df) == 0:
                        continue
                    frames.append(df)
                except Exception:
                    # Repair didn't work, re-raise original error
                    raise parse_err
            else:
                # No repair possible, re-raise original error
                raise

    if not frames:
        return None
    return pd.concat(frames, ignore_index=True).to_csv(index=False)


def _combine_baseline_chunks(chunk_texts: List[str]) -> Optional[str]:
    lines: List[str] = []
    for text in chunk_texts:
        reader = csv.reader(io.StringIO(text), skipinitialspace=True)
        for row in reader:
            line = ",".join(row).strip()
            if line:
                lines.append(line)
    if not lines:
        return None
    return "\n".join(lines)


def collect_batch(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir)
    manifest = _load_manifest(run_dir)

    batch_id = manifest.get("batch_id")
    if not batch_id:
        raise ValueError("manifest.json does not contain batch_id")

    client = OpenAI()
    batch = client.batches.retrieve(batch_id)
    manifest["batch_status"] = batch.status

    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Batch status: {batch.status}")
    if batch.status != "completed":
        # Report failed/errored batches distinctly from in-progress ones
        if str(batch.status).lower() in (
            "failed",
            "errored",
            "cancelled",
            "canceled",
        ):
            print(
                "Batch failed or errored. Inspect error file / batch details below."
            )
            error_file_id = getattr(batch, "error_file_id", None)
            if error_file_id:
                print(f"Error file ID: {error_file_id}")
                try:
                    err_text = _read_batch_output_text(client, error_file_id)
                    print("Error file contents (truncated):")
                    print(err_text[:2000])
                except Exception as exc:
                    print(f"Could not read error file: {exc}")
            else:
                print("No error file available in batch metadata.")
            return
        else:
            print("Batch is not complete yet. Re-run collect later.")
            return

    output_file_id = getattr(batch, "output_file_id", None)
    if not output_file_id:
        raise ValueError("Completed batch has no output_file_id")

    output_jsonl = _read_batch_output_text(client, output_file_id)
    output_lines = [line for line in output_jsonl.splitlines() if line.strip()]

    parsed_results: Dict[str, str] = {}
    failed: List[Dict[str, Any]] = []

    for line in output_lines:
        obj = json.loads(line)
        custom_id = obj.get("custom_id")
        response = obj.get("response", {})
        status_code = response.get("status_code")
        body = response.get("body")

        if status_code != 200:
            failed.append(
                {
                    "custom_id": custom_id,
                    "status_code": status_code,
                    "error": response.get("error") or body,
                }
            )
            continue

        text = _extract_response_text(body if isinstance(body, dict) else {})
        if not text:
            failed.append(
                {
                    "custom_id": custom_id,
                    "status_code": status_code,
                    "error": "No text extracted from response body",
                }
            )
            continue
        parsed_results[custom_id] = text

    output_dir = Path(manifest["output_dir"])
    schema_dir, baseline_dir = _ensure_dirs(str(output_dir))

    by_stem_kind: Dict[
        Tuple[str, str], List[Tuple[int, str, List[int], dict]]
    ] = defaultdict(list)
    for custom_id, text in parsed_results.items():
        meta = manifest["entries"].get(custom_id)
        if not meta:
            failed.append(
                {
                    "custom_id": custom_id,
                    "status_code": 200,
                    "error": "Missing metadata for custom_id",
                }
            )
            continue
        by_stem_kind[(meta["stem"], meta["kind"])].append(
            (meta["chunk_index"], text, meta["seed_idx"], meta)
        )

    exp_summary: List[Dict[str, Any]] = []

    # Create directory for raw responses of failed parses
    raw_responses_dir = run_dir / "raw_responses"

    for (stem, kind), chunk_records in by_stem_kind.items():
        chunk_records = sorted(chunk_records, key=lambda x: x[0])
        chunk_texts = [x[1] for x in chunk_records]
        seed_idx = chunk_records[0][2]
        meta = chunk_records[0][3]

        if kind == "schema":
            try:
                # If this run used schema ablation, the responses are plain text
                # (one example per line). Write them directly to a .txt file.
                if manifest.get("schema_ablation"):
                    combined_text = "\n".join(chunk_texts)
                    schema_txt_path = schema_dir / f"{stem}.txt"
                    schema_txt_path.write_text(combined_text, encoding="utf-8")
                    exp_summary.append(
                        {
                            "filename": str(schema_txt_path),
                            "seed_idx": seed_idx,
                            "missing_fixed": meta.get("missing_fixed"),
                            "missing_constituents": meta.get(
                                "missing_constituents"
                            ),
                            "missing_seed_examples": meta.get(
                                "missing_seed_examples"
                            ),
                            "variant_id": meta.get("variant_id"),
                        }
                    )
                else:
                    combined_csv = _combine_schema_chunks(chunk_texts)
                    if not combined_csv:
                        raise ValueError("No valid schema chunk rows")

                    schema_path = schema_dir / f"{stem}.csv"
                    schema_path.write_text(combined_csv, encoding="utf-8")

                    txt_content = _extract_example_text(combined_csv)
                    if txt_content:
                        (schema_dir / f"{stem}.txt").write_text(
                            txt_content, encoding="utf-8"
                        )

                    exp_summary.append(
                        {
                            "filename": str(schema_path),
                            "seed_idx": seed_idx,
                            "missing_fixed": meta.get("missing_fixed"),
                            "missing_constituents": meta.get(
                                "missing_constituents"
                            ),
                            "missing_seed_examples": meta.get(
                                "missing_seed_examples"
                            ),
                            "variant_id": meta.get("variant_id"),
                        }
                    )
            except Exception as exc:
                # Save raw text for debugging
                raw_responses_dir.mkdir(parents=True, exist_ok=True)
                raw_file = raw_responses_dir / f"{stem}__schema__raw.txt"
                combined_text = "\n\n--- CHUNK SEPARATOR ---\n\n".join(
                    chunk_texts
                )
                raw_file.write_text(combined_text, encoding="utf-8")

                failed.append(
                    {
                        "custom_id": f"schema__{stem}",
                        "status_code": 200,
                        "error": f"Schema combine/write failed: {exc}",
                        "raw_response_file": str(raw_file.relative_to(run_dir)),
                    }
                )

        elif kind == "baseline":
            try:
                combined_txt = _combine_baseline_chunks(chunk_texts)
                if not combined_txt:
                    raise ValueError("No valid baseline chunk rows")

                baseline_path = baseline_dir / f"{stem}.txt"
                baseline_path.write_text(combined_txt, encoding="utf-8")
                exp_summary.append(
                    {
                        "filename": str(baseline_path),
                        "seed_idx": seed_idx,
                        "missing_fixed": meta.get("missing_fixed"),
                        "missing_constituents": meta.get(
                            "missing_constituents"
                        ),
                        "missing_seed_examples": meta.get(
                            "missing_seed_examples"
                        ),
                        "variant_id": meta.get("variant_id"),
                    }
                )
            except Exception as exc:
                # Save raw text for debugging
                raw_responses_dir.mkdir(parents=True, exist_ok=True)
                raw_file = raw_responses_dir / f"{stem}__baseline__raw.txt"
                combined_text = "\n\n--- CHUNK SEPARATOR ---\n\n".join(
                    chunk_texts
                )
                raw_file.write_text(combined_text, encoding="utf-8")

                failed.append(
                    {
                        "custom_id": f"baseline__{stem}",
                        "status_code": 200,
                        "error": f"Baseline combine/write failed: {exc}",
                        "raw_response_file": str(raw_file.relative_to(run_dir)),
                    }
                )

    summary_name = (
        "exp_summary_schema_ablation.csv"
        if manifest.get("schema_ablation")
        else "exp_summary.csv"
    )
    exp_summary_path = output_dir / summary_name
    write_header = not exp_summary_path.exists()
    with exp_summary_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            if manifest.get("schema_ablation"):
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
            if manifest.get("schema_ablation"):
                writer.writerow(
                    [
                        entry["filename"],
                        str(entry["seed_idx"]),
                        str(entry.get("missing_fixed")),
                        str(entry.get("missing_constituents")),
                        str(entry.get("missing_seed_examples")),
                        str(entry.get("variant_id")),
                    ]
                )
            else:
                writer.writerow([entry["filename"], str(entry["seed_idx"])])

    failures_path = run_dir / "failures.json"
    failures_path.write_text(json.dumps(failed, indent=2), encoding="utf-8")

    print(f"Wrote {len(exp_summary)} combined output file entries.")
    print(f"Updated summary: {exp_summary_path}")
    if failed:
        print(
            f"Encountered {len(failed)} failed request(s). See: {failures_path}"
        )
    else:
        print("No failed requests.")


def status_batch(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir)
    manifest = _load_manifest(run_dir)
    batch_id = manifest.get("batch_id")
    if not batch_id:
        raise ValueError("manifest.json does not contain batch_id")

    # Try to detect provider from manifest (explicit or from entries)
    provider = manifest.get("provider")
    if not provider and "entries" in manifest:
        entries = manifest["entries"]
        if entries:
            first_entry = next(iter(entries.values()))
            provider = first_entry.get("provider", "openai")

    if provider == "anthropic":
        client = Anthropic()
        batch = client.messages.batches.retrieve(batch_id)
        print(f"Batch ID: {getattr(batch, 'id', batch_id)}")
        print(f"Status: {getattr(batch, 'processing_status', None)}")
        print(f"Result count: {getattr(batch, 'result_count', None)}")
        print(f"Errors: {getattr(batch, 'error_count', None)}")
    else:
        client = OpenAI()
        batch = client.batches.retrieve(batch_id)
        print(f"Batch ID: {batch.id}")
        print(f"Status: {batch.status}")
        print(f"Output file ID: {getattr(batch, 'output_file_id', None)}")
        print(f"Error file ID: {getattr(batch, 'error_file_id', None)}")


def collect_anthropic_batch(args: argparse.Namespace) -> None:
    """Collect results from an Anthropic Batch API batch."""
    run_dir = Path(args.run_dir)
    manifest = _load_manifest(run_dir)

    batch_id = manifest.get("batch_id")
    if not batch_id:
        raise ValueError("manifest.json does not contain batch_id")

    client = Anthropic()
    batch = client.messages.batches.retrieve(batch_id)
    manifest["batch_status"] = batch.processing_status

    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    status = getattr(batch, "processing_status", None)
    print(f"Batch status: {status}")
    if status != "ended":
        if str(status).lower() in ("failed", "errored", "canceled", "expired"):
            print("Batch failed/errored. Inspect batch details for errors.")
            # Try to show any available error/result counts
            print(f"Result count: {getattr(batch, 'result_count', None)}")
            print(f"Error count: {getattr(batch, 'error_count', None)}")
            return
        else:
            print("Batch is not complete yet. Re-run collect later.")
            return

    parsed_results: Dict[str, str] = {}
    failed: List[Dict[str, Any]] = []

    # Stream results from Anthropic batch
    for result in client.messages.batches.results(batch_id):
        custom_id = result.custom_id
        if result.result.type == "succeeded":
            message = result.result.message
            text = ""
            if message.content:
                for block in message.content:
                    if hasattr(block, "text"):
                        text += block.text
            if text:
                parsed_results[custom_id] = text
            else:
                failed.append(
                    {
                        "custom_id": custom_id,
                        "status": "succeeded",
                        "error": "No text extracted from message",
                    }
                )
        elif result.result.type == "errored":
            failed.append(
                {
                    "custom_id": custom_id,
                    "status": "errored",
                    "error": str(result.result.error),
                }
            )
        elif result.result.type == "expired":
            failed.append(
                {
                    "custom_id": custom_id,
                    "status": "expired",
                    "error": "Request expired",
                }
            )
        elif result.result.type == "canceled":
            failed.append(
                {
                    "custom_id": custom_id,
                    "status": "canceled",
                    "error": "Request was canceled",
                }
            )

    # Materialize outputs (same as OpenAI path)
    output_dir = Path(manifest["output_dir"])
    schema_dir, baseline_dir = _ensure_dirs(str(output_dir))

    by_stem_kind: Dict[Tuple[str, str], List[Tuple[int, str, List[int]]]] = (
        defaultdict(list)
    )
    for custom_id, text in parsed_results.items():
        meta = manifest["entries"].get(custom_id)
        if not meta:
            failed.append(
                {
                    "custom_id": custom_id,
                    "error": "Missing metadata for custom_id",
                }
            )
            continue
        by_stem_kind[(meta["stem"], meta["kind"])].append(
            (meta["chunk_index"], text, meta["seed_idx"], meta)
        )

    exp_summary: List[Dict[str, Any]] = []

    # Create directory for raw responses of failed parses
    raw_responses_dir = run_dir / "raw_responses"

    for (stem, kind), chunk_records in by_stem_kind.items():
        chunk_records = sorted(chunk_records, key=lambda x: x[0])
        chunk_texts = [x[1] for x in chunk_records]
        seed_idx = chunk_records[0][2]
        meta = chunk_records[0][3]

        if kind == "schema":
            try:
                # If this run used schema ablation, the responses are plain text
                # (one example per line). Write them directly to a .txt file.
                if manifest.get("schema_ablation"):
                    combined_text = "\n".join(chunk_texts)
                    schema_txt_path = schema_dir / f"{stem}.txt"
                    schema_txt_path.write_text(combined_text, encoding="utf-8")
                    exp_summary.append(
                        {
                            "filename": str(schema_txt_path),
                            "seed_idx": seed_idx,
                            "missing_fixed": meta.get("missing_fixed"),
                            "missing_constituents": meta.get(
                                "missing_constituents"
                            ),
                            "missing_seed_examples": meta.get(
                                "missing_seed_examples"
                            ),
                            "variant_id": meta.get("variant_id"),
                        }
                    )
                else:
                    combined_csv = _combine_schema_chunks(chunk_texts)
                    if not combined_csv:
                        raise ValueError("No valid schema chunk rows")

                    schema_path = schema_dir / f"{stem}.csv"
                    schema_path.write_text(combined_csv, encoding="utf-8")

                    txt_content = _extract_example_text(combined_csv)
                    if txt_content:
                        (schema_dir / f"{stem}.txt").write_text(
                            txt_content, encoding="utf-8"
                        )

                    exp_summary.append(
                        {
                            "filename": str(schema_path),
                            "seed_idx": seed_idx,
                            "missing_fixed": meta.get("missing_fixed"),
                            "missing_constituents": meta.get(
                                "missing_constituents"
                            ),
                            "missing_seed_examples": meta.get(
                                "missing_seed_examples"
                            ),
                            "variant_id": meta.get("variant_id"),
                        }
                    )
            except Exception as exc:
                # Save raw text for debugging
                raw_responses_dir.mkdir(parents=True, exist_ok=True)
                raw_file = raw_responses_dir / f"{stem}__schema__raw.txt"
                combined_text = "\n\n--- CHUNK SEPARATOR ---\n\n".join(
                    chunk_texts
                )
                raw_file.write_text(combined_text, encoding="utf-8")

                failed.append(
                    {
                        "custom_id": f"schema__{stem}",
                        "error": f"Schema combine/write failed: {exc}",
                        "raw_response_file": str(raw_file.relative_to(run_dir)),
                    }
                )

        elif kind == "baseline":
            try:
                combined_txt = _combine_baseline_chunks(chunk_texts)
                if not combined_txt:
                    raise ValueError("No valid baseline chunk rows")

                baseline_path = baseline_dir / f"{stem}.txt"
                baseline_path.write_text(combined_txt, encoding="utf-8")
                exp_summary.append(
                    {
                        "filename": str(baseline_path),
                        "seed_idx": seed_idx,
                        "missing_fixed": meta.get("missing_fixed"),
                        "missing_constituents": meta.get(
                            "missing_constituents"
                        ),
                        "missing_seed_examples": meta.get(
                            "missing_seed_examples"
                        ),
                        "variant_id": meta.get("variant_id"),
                    }
                )
            except Exception as exc:
                # Save raw text for debugging
                raw_responses_dir.mkdir(parents=True, exist_ok=True)
                raw_file = raw_responses_dir / f"{stem}__baseline__raw.txt"
                combined_text = "\n\n--- CHUNK SEPARATOR ---\n\n".join(
                    chunk_texts
                )
                raw_file.write_text(combined_text, encoding="utf-8")

                failed.append(
                    {
                        "custom_id": f"baseline__{stem}",
                        "error": f"Baseline combine/write failed: {exc}",
                        "raw_response_file": str(raw_file.relative_to(run_dir)),
                    }
                )

    summary_name = (
        "exp_summary_schema_ablation.csv"
        if manifest.get("schema_ablation")
        else "exp_summary.csv"
    )
    exp_summary_path = output_dir / summary_name
    write_header = not exp_summary_path.exists()
    with exp_summary_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            if manifest.get("schema_ablation"):
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
            if manifest.get("schema_ablation"):
                writer.writerow(
                    [
                        entry["filename"],
                        str(entry["seed_idx"]),
                        str(entry.get("missing_fixed")),
                        str(entry.get("missing_constituents")),
                        str(entry.get("missing_seed_examples")),
                        str(entry.get("variant_id")),
                    ]
                )
            else:
                writer.writerow([entry["filename"], str(entry["seed_idx"])])

    failures_path = run_dir / "failures.json"
    failures_path.write_text(json.dumps(failed, indent=2), encoding="utf-8")

    print(f"Wrote {len(exp_summary)} combined output file entries.")
    print(f"Updated summary: {exp_summary_path}")
    if failed:
        print(
            f"Encountered {len(failed)} failed request(s). See: {failures_path}"
        )
    else:
        print("No failed requests.")


def collect_local(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir)
    manifest = _load_manifest(run_dir)

    jsonl_path = run_dir / "requests.jsonl"
    if not jsonl_path.exists():
        raise FileNotFoundError(f"Missing requests file: {jsonl_path}")

    parsed_results: Dict[str, str] = {}
    failed: List[Dict[str, Any]] = []

    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            custom_id = obj.get("custom_id")
            body = obj.get("body", {})
            model = body.get("model")
            input_field = body.get("input")

            # Normalize messages into system_prompt and user_prompt
            system_prompt = None
            user_prompt = None
            if isinstance(input_field, list):
                for msg in input_field:
                    role = msg.get("role")
                    content = msg.get("content")
                    if role == "system":
                        system_prompt = content
                    elif role == "user":
                        user_prompt = content
            elif isinstance(input_field, str):
                user_prompt = input_field

            # Find provider from manifest entries if present, else default to 'anthropic'
            entry_meta = manifest.get("entries", {}).get(custom_id, {})
            provider = entry_meta.get("provider", "anthropic")

            try:
                text = ModelAPIClient.call_api(
                    user_prompt=user_prompt or "",
                    system_prompt=system_prompt,
                    provider=provider,
                    model=model,
                    max_tokens=body.get("max_output_tokens", 40000),
                    temperature=0.7,
                    n=1,
                    mock=False,
                )
                if text:
                    parsed_results[custom_id] = text
                else:
                    failed.append(
                        {"custom_id": custom_id, "error": "Empty response"}
                    )
            except Exception as exc:
                failed.append({"custom_id": custom_id, "error": str(exc)})

    # Materialize outputs similarly to collect_batch
    output_dir = Path(manifest["output_dir"])
    schema_dir, baseline_dir = _ensure_dirs(str(output_dir))

    by_stem_kind: Dict[Tuple[str, str], List[Tuple[int, str, List[int]]]] = (
        defaultdict(list)
    )
    for custom_id, text in parsed_results.items():
        meta = manifest["entries"].get(custom_id)
        if not meta:
            failed.append(
                {
                    "custom_id": custom_id,
                    "error": "Missing metadata for custom_id",
                }
            )
            continue
        by_stem_kind[(meta["stem"], meta["kind"])].append(
            (meta["chunk_index"], text, meta["seed_idx"])
        )

    exp_summary: List[Dict[str, Any]] = []
    for (stem, kind), chunk_records in by_stem_kind.items():
        chunk_records = sorted(chunk_records, key=lambda x: x[0])
        chunk_texts = [x[1] for x in chunk_records]
        seed_idx = chunk_records[0][2]

        if kind == "schema":
            try:
                combined_csv = _combine_schema_chunks(chunk_texts)
                if not combined_csv:
                    raise ValueError("No valid schema chunk rows")

                schema_path = schema_dir / f"{stem}.csv"
                schema_path.write_text(combined_csv, encoding="utf-8")

                txt_content = _extract_example_text(combined_csv)
                if txt_content:
                    (schema_dir / f"{stem}.txt").write_text(
                        txt_content, encoding="utf-8"
                    )

                exp_summary.append(
                    {"filename": str(schema_path), "seed_idx": seed_idx}
                )
            except Exception as exc:
                failed.append(
                    {
                        "custom_id": f"schema__{stem}",
                        "error": f"Schema combine/write failed: {exc}",
                    }
                )
        elif kind == "baseline":
            try:
                combined_txt = _combine_baseline_chunks(chunk_texts)
                if not combined_txt:
                    raise ValueError("No valid baseline chunk rows")

                baseline_path = baseline_dir / f"{stem}.txt"
                baseline_path.write_text(combined_txt, encoding="utf-8")
                exp_summary.append(
                    {"filename": str(baseline_path), "seed_idx": seed_idx}
                )
            except Exception as exc:
                failed.append(
                    {
                        "custom_id": f"baseline__{stem}",
                        "error": f"Baseline combine/write failed: {exc}",
                    }
                )

    exp_summary_path = output_dir / "exp_summary.csv"
    write_header = not exp_summary_path.exists()
    with exp_summary_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["filename", "seed_idx"])
        for entry in exp_summary:
            writer.writerow([entry["filename"], str(entry["seed_idx"])])

    failures_path = run_dir / "failures.json"
    failures_path.write_text(json.dumps(failed, indent=2), encoding="utf-8")

    print(f"Wrote {len(exp_summary)} combined output file entries.")
    print(f"Updated summary: {exp_summary_path}")
    if failed:
        print(
            f"Encountered {len(failed)} failed request(s). See: {failures_path}"
        )
    else:
        print("No failed requests.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate datasets through the OpenAI or Anthropic Batch API."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    submit = sub.add_parser("submit", help="Build requests and submit batch")
    submit.add_argument(
        "--schema-file", required=True, help="Path to schema JSON"
    )
    submit.add_argument(
        "--output-dir", required=True, help="Output root directory"
    )
    submit.add_argument(
        "--model",
        required=True,
        help="Model name for the selected provider",
    )
    submit.add_argument(
        "--provider",
        required=True,
        choices=["openai", "anthropic"],
        help="API provider to target (openai or anthropic)",
    )
    submit.add_argument(
        "--n", type=int, required=True, help="Total rows per prompt family"
    )
    submit.add_argument(
        "--chunk-size", type=int, default=25, help="Rows per batched request"
    )
    submit.add_argument("--runs", type=int, default=1)
    submit.add_argument("--max-seed-reshuffle", type=int, default=1)
    submit.add_argument("--max-s", type=int, default=3)
    submit.add_argument("--inc-s", type=int, default=1)
    submit.add_argument(
        "--exclude-s", default="", help="Comma-separated s values to exclude"
    )
    submit.add_argument(
        "--schema-ablation",
        action="store_true",
        help="Enable schema field ablation across all subsets",
    )
    submit.add_argument(
        "--no-ablation-seedless",
        action="store_false",
        dest="ablation_seedless",
        help="Disable the seedless schema variant when ablating",
    )
    submit.add_argument(
        "--fixed-seed-count",
        type=int,
        default=3,
        help="Seed count for schema ablation (ignored if not ablating)",
    )
    submit.add_argument("--random-seed", type=int, default=42)
    submit.add_argument("--completion-window", default="24h", choices=["24h"])
    submit.add_argument("--dry-run", action="store_true")
    submit.add_argument(
        "--append-system-prompt",
        default="",
        help="Text to append to the generated system prompt for each request",
    )
    submit.add_argument(
        "--append-user-prompt",
        default="",
        help="Text to append to the generated user prompt for each request",
    )
    submit.add_argument(
        "--force",
        action="store_true",
        help="Force re-generation even if output files already exist",
    )
    submit.add_argument(
        "--rerun-file",
        default="",
        help="Path to a file listing output filenames to re-run (one per line)",
    )
    submit.set_defaults(func=submit_batch)

    collect = sub.add_parser(
        "collect", help="Download completed batch output and write final files"
    )
    collect.add_argument("--run-dir", required=True, help="Batch run directory")

    def collect_wrapper(args):
        """Wrapper that auto-detects provider from manifest."""
        manifest = _load_manifest(Path(args.run_dir))

        if manifest.get("kind") == "label":
            collect_label_batch(args)
            return

        # Try to detect provider from manifest
        provider = None

        # First check if provider is explicitly stored
        if "provider" in manifest:
            provider = manifest["provider"]

        # Otherwise, try to infer from entries
        if not provider and "entries" in manifest:
            entries = manifest["entries"]
            if entries:
                first_entry = next(iter(entries.values()))
                provider = first_entry.get("provider", "openai")

        # Default to openai if not determined
        if not provider:
            provider = "openai"

        if provider == "anthropic":
            collect_anthropic_batch(args)
        else:
            collect_batch(args)

    collect.set_defaults(func=collect_wrapper)

    status = sub.add_parser("status", help="Check batch status")
    status.add_argument("--run-dir", required=True, help="Batch run directory")
    status.set_defaults(func=status_batch)

    collect_local_parser = sub.add_parser(
        "collect-local",
        help="Execute staged local requests and write final files",
    )
    collect_local_parser.add_argument(
        "--run-dir", required=True, help="Batch run directory"
    )
    collect_local_parser.set_defaults(func=collect_local)

    label_submit = sub.add_parser(
        "submit-labels",
        help="Scan txt files and submit batch requests for missing *_labeled.csv outputs",
    )
    label_submit.add_argument(
        "--parent-dirs",
        nargs="+",
        default=DEFAULT_PARENT_DIRS,
        help="Directories to scan recursively for txt files",
    )
    label_submit.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Output root directory",
    )
    label_submit.add_argument(
        "--label-config",
        required=True,
        help="Path to a JSON labeling configuration",
    )
    label_submit.add_argument(
        "--no-subdirs",
        action="store_false",
        dest="recursive",
        help="Do not scan subdirectories; only consider files directly under each parent dir",
    )
    label_submit.add_argument("--dry-run", action="store_true")
    label_submit.set_defaults(func=submit_label_batch)

    relabel_submit = sub.add_parser(
        "relabel-labels",
        help="Archive existing *_labeled.csv files into old/ and relabel every txt file",
    )
    relabel_submit.add_argument(
        "--parent-dirs",
        nargs="+",
        default=DEFAULT_PARENT_DIRS,
        help="Directories to scan recursively for txt files",
    )
    relabel_submit.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Output root directory",
    )
    relabel_submit.add_argument(
        "--label-config",
        required=True,
        help="Path to a JSON labeling configuration",
    )
    relabel_submit.add_argument(
        "--no-subdirs",
        action="store_false",
        dest="recursive",
        help="Do not scan subdirectories; only consider files directly under each parent dir",
    )
    relabel_submit.add_argument("--dry-run", action="store_true")
    relabel_submit.set_defaults(func=relabel_all_label_batch)

    label_collect = sub.add_parser(
        "collect-labels",
        help="Download completed label batch output and write *_labeled.csv files",
    )
    label_collect.add_argument(
        "--run-dir", required=True, help="Label batch run directory"
    )
    label_collect.set_defaults(func=collect_label_batch)

    label_status = sub.add_parser(
        "status-labels", help="Check label batch status"
    )
    label_status.add_argument(
        "--run-dir", required=True, help="Label batch run directory"
    )
    label_status.set_defaults(func=status_batch)

    label_scan = sub.add_parser(
        "scan-labels",
        help="Print txt files that are missing matching *_labeled.csv outputs",
    )
    label_scan.add_argument(
        "--parent-dirs",
        nargs="+",
        default=DEFAULT_PARENT_DIRS,
        help="Directories to scan recursively for txt files",
    )
    label_scan.set_defaults(func=print_missing_label_files)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
