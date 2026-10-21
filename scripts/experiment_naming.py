from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any


DEFAULT_DATA_OUTPUT_ROOT = Path("data") / "datasets"
DEFAULT_DNN_OUTPUT_ROOT = Path("runs") / "dnn"
DEFAULT_DNN_TRAINING_OUTPUT_ROOT = DEFAULT_DNN_OUTPUT_ROOT / "training"
DEFAULT_DNN_TUNING_OUTPUT_ROOT = DEFAULT_DNN_OUTPUT_ROOT / "tuning"
DEFAULT_EXPERIMENT_OUTPUT_ROOT = DEFAULT_DNN_TRAINING_OUTPUT_ROOT
DEFAULT_TABULAR_OUTPUT_ROOT = Path("runs") / "tabular"
DEFAULT_DATASET_EXPERIMENT_NAME = "dataset"
DEFAULT_STRATIFIED_BUCKET_BUDGETS = {
    "short": 0.00,
    "medium": 0.60,
    "long": 0.40,
}
DEFAULT_DEDUPE_SCOPE = "within_layout_bucket"


def slugify_experiment_name(
    name: str,
) -> str:
    slug = re.sub(
        r"[^A-Za-z0-9]+",
        "_",
        name.strip().lower(),
    ).strip("_")

    if not slug:
        raise ValueError(
            "Experiment name must contain at least one letter or digit."
        )

    return slug


def dataset_size_suffix(
    *,
    train_mazes: int,
    validation_mazes: int,
    test_mazes: int,
    tasks_per_maze: int,
    seed: int,
) -> str:
    return (
        f"{train_mazes}x{validation_mazes}x{test_mazes}"
        f"_t{tasks_per_maze}_seed{seed}"
    )


def _dataset_name_with_prefix(
    prefix: str,
    *,
    train_mazes: int,
    validation_mazes: int,
    test_mazes: int,
    tasks_per_maze: int,
    seed: int,
    stratify: bool,
    dedupe_threshold: float,
    bucket_budgets: dict[str, float] | None,
    dedupe_scope: str,
) -> str:
    dataset_name = (
        f"{prefix}_"
        + dataset_size_suffix(
            train_mazes=train_mazes,
            validation_mazes=validation_mazes,
            test_mazes=test_mazes,
            tasks_per_maze=tasks_per_maze,
            seed=seed,
        )
    )
    if not stratify:
        return dataset_name

    budgets = bucket_budgets or DEFAULT_STRATIFIED_BUCKET_BUDGETS
    budget_suffix = "_".join(
        f"{bucket[0]}{int(round(float(budgets[bucket]) * 100)):02d}"
        for bucket in ("short", "medium", "long")
    )
    threshold_suffix = f"{int(round(dedupe_threshold * 100)):03d}"
    return (
        f"{dataset_name}_stratified_{budget_suffix}"
        f"_dedupe{threshold_suffix}"
    )


def canonical_dataset_name(
    *,
    train_mazes: int,
    validation_mazes: int,
    test_mazes: int,
    tasks_per_maze: int,
    seed: int,
    stratify: bool = False,
    dedupe_threshold: float = 0.7,
    bucket_budgets: dict[str, float] | None = None,
    dedupe_scope: str = DEFAULT_DEDUPE_SCOPE,
) -> str:
    """Return the stable directory name for a generated dataset."""
    return _dataset_name_with_prefix(
        "dataset",
        train_mazes=train_mazes,
        validation_mazes=validation_mazes,
        test_mazes=test_mazes,
        tasks_per_maze=tasks_per_maze,
        seed=seed,
        stratify=stratify,
        dedupe_threshold=dedupe_threshold,
        bucket_budgets=bucket_budgets,
        dedupe_scope=dedupe_scope,
    )


def experiment_dataset_name(
    experiment_name: str | None,
    *,
    train_mazes: int,
    validation_mazes: int,
    test_mazes: int,
    tasks_per_maze: int,
    seed: int,
    stratify: bool = False,
    dedupe_threshold: float = 0.7,
    bucket_budgets: dict[str, float] | None = None,
    dedupe_scope: str = DEFAULT_DEDUPE_SCOPE,
) -> str:
    """Return the canonical name for a generated dataset.

    ``experiment_name`` remains accepted for compatibility, but generated
    dataset directories are intentionally independent of experiment labels.
    """
    del experiment_name
    return canonical_dataset_name(
        train_mazes=train_mazes,
        validation_mazes=validation_mazes,
        test_mazes=test_mazes,
        tasks_per_maze=tasks_per_maze,
        seed=seed,
        stratify=stratify,
        dedupe_threshold=dedupe_threshold,
        bucket_budgets=bucket_budgets,
        dedupe_scope=dedupe_scope,
    )


def experiment_output_name(
    experiment_name: str | None,
    *,
    train_mazes: int,
    validation_mazes: int,
    test_mazes: int,
    tasks_per_maze: int,
    seed: int,
    stratify: bool = False,
    dedupe_threshold: float = 0.7,
    bucket_budgets: dict[str, float] | None = None,
    dedupe_scope: str = DEFAULT_DEDUPE_SCOPE,
) -> str:
    """Return the experiment-labeled name used by training/tuning outputs."""
    name = (
        DEFAULT_DATASET_EXPERIMENT_NAME
        if experiment_name is None
        else experiment_name
    )
    return _dataset_name_with_prefix(
        slugify_experiment_name(name),
        train_mazes=train_mazes,
        validation_mazes=validation_mazes,
        test_mazes=test_mazes,
        tasks_per_maze=tasks_per_maze,
        seed=seed,
        stratify=stratify,
        dedupe_threshold=dedupe_threshold,
        bucket_budgets=bucket_budgets,
        dedupe_scope=dedupe_scope,
    )


def load_dataset_metadata(
    dataset_path: Path,
) -> dict[str, Any]:
    metadata_path = dataset_path.parent / "metadata.json"

    with metadata_path.open() as file:
        metadata = json.load(file)

    if not isinstance(
        metadata,
        dict,
    ):
        raise ValueError(
            f"{metadata_path} must contain a JSON object."
        )

    return metadata


def experiment_dataset_name_from_metadata(
    experiment_name: str,
    metadata: dict[str, Any],
) -> str:
    required_fields = [
        "train_mazes",
        "validation_mazes",
        "test_mazes",
        "tasks_per_maze",
        "master_seed",
    ]
    missing_fields = [
        field
        for field in required_fields
        if field not in metadata
    ]

    if missing_fields:
        raise ValueError(
            "Dataset metadata is missing required field(s): "
            + ", ".join(missing_fields)
        )

    return experiment_output_name(
        experiment_name,
        train_mazes=int(metadata["train_mazes"]),
        validation_mazes=int(metadata["validation_mazes"]),
        test_mazes=int(metadata["test_mazes"]),
        tasks_per_maze=int(metadata["tasks_per_maze"]),
        seed=int(metadata["master_seed"]),
        stratify=bool(metadata.get("stratify", False)),
        dedupe_threshold=float(
            metadata.get("dedupe_threshold", 0.7)
        ),
        bucket_budgets=metadata.get("bucket_budgets"),
        dedupe_scope=(
            str(
                metadata.get(
                    "dedupe_scope",
                    "within_layout_cross_bucket",
                )
            )
        ),
    )


def experiment_dataset_name_for_path(
    experiment_name: str,
    dataset_path: Path,
) -> str:
    return experiment_dataset_name_from_metadata(
        experiment_name,
        load_dataset_metadata(dataset_path),
    )
