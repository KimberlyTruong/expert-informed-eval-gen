"""Generic execution defaults for the benchmark workflow.

Use-case-specific schemas, labeling taxonomies, and exclusions belong under
``configs/`` and are supplied to the command-line tools explicitly.
"""

from __future__ import annotations

DEFAULT_OUTPUT_DIR = "./data/generated"
DEFAULT_PARENT_DIRS = ["./data/generated", "./data/annotations"]

DEFAULT_GENERATION_MODELS = [
    ("anthropic", "claude-haiku-4-5-20251001"),
    ("anthropic", "claude-sonnet-4-6"),
]

DEFAULT_LABEL_PROVIDER = "anthropic"
DEFAULT_LABEL_MODEL = "claude-haiku-4-5-20251001"
