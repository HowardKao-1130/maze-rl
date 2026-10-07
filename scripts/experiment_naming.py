from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any


DEFAULT_DATA_OUTPUT_ROOT = Path("data") / "datasets"
DEFAULT_EXPERIMENT_OUTPUT_ROOT = Path("runs") / "dnn"
DEFAULT_TABULAR_OUTPUT_ROOT = Path("runs") / "tabular"
DEFAULT_DATASET_EXPERIMENT_NAME = "dataset"


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


def experiment_dataset_name(
    experiment_name: str | None,
    *,
    train_mazes: int,
    validation_mazes: int,
    test_mazes: int,
    tasks_per_maze: int,
    seed: int,
) -> str:
    name = (
        DEFAULT_DATASET_EXPERIMENT_NAME
        if experiment_name is None
        else experiment_name
    )
    return (
        f"{slugify_experiment_name(name)}_"
        + dataset_size_suffix(
            train_mazes=train_mazes,
            validation_mazes=validation_mazes,
            test_mazes=test_mazes,
            tasks_per_maze=tasks_per_maze,
            seed=seed,
        )
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

    return experiment_dataset_name(
        experiment_name,
        train_mazes=int(metadata["train_mazes"]),
        validation_mazes=int(metadata["validation_mazes"]),
        test_mazes=int(metadata["test_mazes"]),
        tasks_per_maze=int(metadata["tasks_per_maze"]),
        seed=int(metadata["master_seed"]),
    )


def experiment_dataset_name_for_path(
    experiment_name: str,
    dataset_path: Path,
) -> str:
    return experiment_dataset_name_from_metadata(
        experiment_name,
        load_dataset_metadata(dataset_path),
    )
