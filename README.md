# A Framework for Generating Valid Context-Specific Benchmarks through Expert Guidance
Kimberly Le Truong, Nari Johnson, Anna Kawakami, Hoda Heidari

## Overview
This repository provides implementations for schema, dataset quality assessment metrics, and synthetic data generation scripts described in Truong et al. (2026)'s "A Framework for Generating Valid Context-Specific Benchmarks through Expert Guidance."

## Prerequisites

- Python 3.10+
- API credentials for the provider used during generation or labeling. The scripts have been tested for: `ANTHROPIC_API_KEY` and `OPENAI_API_KEY`. Put these in `.env` in the root of the repository.

## Quick Start

From the repository root, create and activate a virtual environment:

```bash
python3 -m venv .venv
source ./venv/bin/activate
python -m pip install -r requirements.txt          # Install all requirements
```

Choose a schema under `configs/schemas/`, then use the commands below.

## How To

### 1. Define a schema

Store a schema JSON under `configs/schemas/<use_case>/`. Reference the completed schema in `configs/schemas/social-work/schema_data-5-7.json` whcih was used in our Truong et al.'s (2026) case study.

There are two ways to populate the schema. You can fill out the json file directly (See `configs/schemas/empty_schema.json` for a template). Or create a schema interactively. For the latter, open [tools/schema_form.html](tools/schema_form.html) in a browser. Fill out the form, choose **Finalize Schema**, and select **Download Schema**. The form downloads both `schema_data.json` and `schema_data.md`; store the JSON as the input used by the generation scripts and keep the Markdown as a human-readable record beside it.

You can also, optionally, define an exclusion list (e.g., `configs/schemas/social-work/not_realistic.json` for the social work case study). This should be kept in the same directory as the schema. These exclusion lists contain constituent groups that are not realistic and should not be included in the coverage calculation.

### 2. Generate a dataset of n examples

For synchronous generation:

```bash
python scripts/generation.py \
  --schema-file configs/schemas/social-work/schema_data-5-7.json \
  --output-dir data/generated/run_1 \
  --provider anthropic \
  --model claude-sonnet-4-6 \
  --n 100 \
  --runs 1 \
  --max-seed-reshuffle 1 \
  --max-s 3 \
  --inc-s 2 \
  --exclude-s 0 \
  --random-seed 42
```

Main arguments are `--schema-file`, `--output-dir`, `--provider`, `--model`, `--n`, `--runs`, `--max-seed-reshuffle`, `--max-s`, `--inc-s`, `--exclude-s`, and `--random-seed`. Optional arguments are `--do-labeling`, `--label-config`, `--schema-ablation`, `--no-ablation-seedless`, and `--fixed-seed-count`.

For asynchronous batch generation, select either the OpenAI or Anthropic provider explicitly. The batch workflow supports both providers and any models from those providers.

```bash
python scripts/batch_generation.py submit \
  --schema-file configs/schemas/social-work/schema_data-5-7.json \
  --output-dir data/generated/run_1 \
  --provider anthropic \
  --model claude-sonnet-4-6 \
  --n 100 \
  --chunk-size 25
```

Additional `submit` arguments include `--completion-window`, `--dry-run`, `--append-system-prompt`, `--append-user-prompt`, `--force`, and `--rerun-file`. 

Always, collect a completed job with:

```bash
python scripts/batch_generation.py collect \
  --run-dir data/generated/run_1/batch_runs/<timestamp>
```

### 3. Annotate generated examples

Get labels for constituent values for existing `.txt` files. These are files where each quote or row represents a new instance of the dataset. You can skip this step if the previous step was run synchronously with `--do-labeling`.

```bash
python scripts/batch_generation.py submit-labels \
  --parent-dirs data/generated/run_1/schema data/generated/run_1/baseline \
  --output-dir data/generated/run_1 \
  --label-config configs/labeling/social_work.json
```

The labeling arguments are `--parent-dirs`, `--output-dir`, `--label-config`, `--no-subdirs`, and `--dry-run`. 

Collect labels with:

```bash
python scripts/batch_generation.py collect-labels \
  --run-dir data/generated/run_1/batch_runs/<label_timestamp>
```

Use `relabel-labels` instead of `submit-labels` to archive existing labels and relabel every discovered text file. Use `scan-labels` to find text files missing a matching labeled CSV.

### 4. Run all quality assessment metrics

Evaluate one labeled CSV:

```bash
python scripts/metrics.py \
  --schema configs/schemas/social-work/schema_data-5-7.json \
  --dataset data/generated/run_1/schema/example_labeled.csv \
  --coverage-k 1 \
  --embedding-model all-mpnet-base-v2
```

Evaluate all matching labeled CSVs under a directory:

```bash
python scripts/metrics.py \
  --schema configs/schemas/social-work/schema_data-5-7.json \
  --input-dir data/generated/run_1 \
  --pattern '*_labeled.csv' \
  --recursive
```

Use exactly one of `--dataset` or `--input-dir`. Other metric arguments include `--pattern`, `--recursive`, `--output-dir`, `--output-json`, `--output-csv`, `--output-pdf`, `--embedding-model`, `--coverage-k`, `--seed-subset-indices`, and optional `--not-realistic`.

## Repository Structure

```text
scripts/                         # executable generation and metrics commands
scripts/helpers/                 # API client and generic execution defaults
scripts/operationalized_metrics/ # coverage, diversity, and realism implementations
configs/schemas/                 # schema, template, and exclusion inputs
configs/labeling/                # use-case-specific labeling configurations
data/                            # generated and intermediate data
outputs/                         # reports and plots
tools/schema_form.html           # browser-based schema authoring form
requirements.txt                 # Python dependencies
```

## Citation

Accepted to EMNLP 2026 Findings. Citation and preprint to come.
