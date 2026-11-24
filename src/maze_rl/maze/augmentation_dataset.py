"""Build split datasets from interactive maze failure-log tasks."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

from maze_rl.maze.augmentation import (
    MazeTask,
    TaskVariant,
    augment_task,
    canonical_layout_key,
)
from maze_rl.maze.pathfinding import Position, shortest_path_length


@dataclass(frozen=True)
class SourceTask:
    record_index: int
    run_id: str
    task: MazeTask
    optimal_path_length: int
    family_key: tuple[tuple[int, int], bytes]


@dataclass(frozen=True)
class SourceFamily:
    family_id: int
    key: tuple[tuple[int, int], bytes]
    tasks: tuple[SourceTask, ...]


VariantRecord = tuple[TaskVariant, int]
TaskIdentity = tuple[tuple[int, int], bytes, Position, Position]


def load_failure_log(path: str | Path) -> list[SourceTask]:
    """Load and validate tasks from a JSONL interactive failure log."""
    path = Path(path)
    tasks: list[SourceTask] = []
    for record_index, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        record = json.loads(line)
        maze = record["maze"]
        layout = np.asarray(maze["layout"], dtype=np.uint8)
        task = MazeTask(
            layout=layout,
            start=tuple(int(value) for value in maze["start"]),
            goal=tuple(int(value) for value in maze["goal"]),
        )
        actual_length = shortest_path_length(
            task.layout,
            task.start,
            task.goal,
        )
        if actual_length is None:
            raise ValueError(
                f"Failure-log row {record_index} has an unreachable goal."
            )
        recorded_length = maze.get("optimal_steps")
        if (
            recorded_length is not None
            and int(recorded_length) != int(actual_length)
        ):
            raise ValueError(
                f"Failure-log row {record_index} has an incorrect "
                "optimal_steps value."
            )
        tasks.append(
            SourceTask(
                record_index=record_index,
                run_id=str(record.get("run_id", record_index)),
                task=task,
                optimal_path_length=int(actual_length),
                family_key=canonical_layout_key(layout),
            )
        )
    return tasks


def group_source_tasks(tasks: list[SourceTask]) -> list[SourceFamily]:
    """Group all endpoint tasks from one D4-equivalent layout together."""
    grouped: dict[tuple[tuple[int, int], bytes], list[SourceTask]] = {}
    for task in tasks:
        grouped.setdefault(task.family_key, []).append(task)
    return [
        SourceFamily(
            family_id=family_id,
            key=key,
            tasks=tuple(grouped[key]),
        )
        for family_id, key in enumerate(sorted(grouped), 1)
    ]


def _difficulty_buckets(
    tasks: list[SourceTask],
    bucket_count: int,
) -> dict[int, int]:
    if bucket_count <= 0:
        raise ValueError("bucket_count must be positive.")
    lengths = np.asarray(
        [task.optimal_path_length for task in tasks],
        dtype=np.int64,
    )
    quantiles = np.quantile(
        lengths,
        np.linspace(0, 1, bucket_count + 1)[1:-1],
    )
    boundaries = np.unique(quantiles)
    return {
        id(task): int(np.searchsorted(boundaries, task.optimal_path_length))
        for task in tasks
    }


def split_source_families(
    families: list[SourceFamily],
    *,
    train_fraction: float = 0.8,
    validation_fraction: float = 0.1,
    test_fraction: float = 0.1,
    difficulty_buckets: int = 5,
    seed: int = 42,
) -> dict[str, list[SourceFamily]]:
    """Assign whole layout families to stratified train/val/test splits."""
    fractions = np.asarray(
        [train_fraction, validation_fraction, test_fraction],
        dtype=float,
    )
    if np.any(fractions <= 0) or not np.isclose(fractions.sum(), 1):
        raise ValueError("Split fractions must be positive and sum to one.")
    if len(families) < 3:
        raise ValueError("At least three source families are required.")

    tasks = [task for family in families for task in family.tasks]
    buckets_by_task = _difficulty_buckets(tasks, difficulty_buckets)
    bucket_count = max(buckets_by_task.values(), default=0) + 1
    total_histogram = np.zeros(bucket_count, dtype=float)
    family_histograms: dict[int, np.ndarray] = {}
    for family in families:
        histogram = np.zeros(bucket_count, dtype=float)
        for task in family.tasks:
            histogram[buckets_by_task[id(task)]] += 1
        family_histograms[family.family_id] = histogram
        total_histogram += histogram

    target_histograms = fractions[:, None] * total_histogram[None, :]
    current_histograms = np.zeros_like(target_histograms)
    current_counts = np.zeros(3, dtype=float)
    target_counts = fractions * len(tasks)

    rng = np.random.default_rng(seed)
    ordered_families = list(families)
    rng.shuffle(ordered_families)
    ordered_families.sort(key=lambda family: len(family.tasks), reverse=True)
    assignments: list[list[SourceFamily]] = [[], [], []]

    for family in ordered_families:
        histogram = family_histograms[family.family_id]
        scores = []
        for split_index in range(3):
            before = current_histograms[split_index]
            after = before + histogram
            target = target_histograms[split_index]
            score = np.sum(
                ((before - target) ** 2 - (after - target) ** 2)
                / np.maximum(target, 1)
            )
            count_overfill = max(
                0.0,
                current_counts[split_index]
                + len(family.tasks)
                - target_counts[split_index],
            )
            scores.append(score - 10 * count_overfill)
        split_index = int(np.argmax(scores))
        assignments[split_index].append(family)
        current_histograms[split_index] += histogram
        current_counts[split_index] += len(family.tasks)

    return {
        "train": assignments[0],
        "validation": assignments[1],
        "test": assignments[2],
    }


def _layout_seed(layout: np.ndarray) -> int:
    digest = hashlib.sha256(
        layout.shape.__repr__().encode("utf-8") + layout.tobytes()
    ).digest()
    return int.from_bytes(digest[:8], "little") % 1_000_000_000


def _select_family_variants(
    variants: list[tuple[TaskVariant, int]],
    *,
    family_id: int,
    max_variants: int | None,
    rng: np.random.Generator,
) -> list[tuple[TaskVariant, int]]:
    if max_variants is None or len(variants) <= max_variants:
        return variants
    if max_variants <= 0:
        raise ValueError("max_variants must be positive when provided.")

    original_indices = [
        index
        for index, (variant, _source_record_index) in enumerate(variants)
        if variant.perturbation_index is None
        and variant.transform == "rotate_0"
        and not variant.reversed_endpoints
    ]
    selected_indices = original_indices[:max_variants]
    selected_index_set = set(selected_indices)
    remaining_indices = [
        index
        for index in range(len(variants))
        if index not in selected_index_set
    ]
    slots = max(0, max_variants - len(selected_indices))
    if slots:
        order = rng.permutation(len(remaining_indices))[:slots]
        selected_indices.extend(
            remaining_indices[int(index)] for index in order
        )
    return [variants[index] for index in selected_indices]


def _generate_selected_variants(
    families: list[SourceFamily],
    *,
    seed: int,
    perturbations_per_task: int,
    max_variants_per_family: int | None,
    max_edits: int,
    max_attempts: int,
    add_probability: float,
    max_path_increase: int | None,
    excluded_task_identities: set[TaskIdentity] | None = None,
) -> dict[int, list[VariantRecord]]:
    rng = np.random.default_rng(seed)
    selected_by_family: dict[int, list[VariantRecord]] = {}

    for family in families:
        candidates: list[VariantRecord] = []
        seen_task_identities = set()
        for source_task in family.tasks:
            variants = augment_task(
                source_task.task,
                perturbation_count=perturbations_per_task,
                rng=rng,
                max_edits=max_edits,
                max_attempts=max_attempts,
                add_probability=add_probability,
                max_path_increase=max_path_increase,
            )
            for variant in variants:
                identity = _task_identity(variant)
                if identity in seen_task_identities:
                    continue
                if (
                    excluded_task_identities is not None
                    and identity in excluded_task_identities
                ):
                    continue
                seen_task_identities.add(identity)
                candidates.append((variant, source_task.record_index))
        selected_by_family[family.family_id] = _select_family_variants(
            candidates,
            family_id=family.family_id,
            max_variants=max_variants_per_family,
            rng=rng,
        )

    globally_seen_task_identities = set()
    for family in families:
        unique_variants = []
        for variant, source_record_index in selected_by_family[family.family_id]:
            identity = _task_identity(variant)
            if identity in globally_seen_task_identities:
                continue
            globally_seen_task_identities.add(identity)
            unique_variants.append((variant, source_record_index))
        selected_by_family[family.family_id] = unique_variants

    return selected_by_family


def _task_identity(variant: TaskVariant) -> TaskIdentity:
    return (
        variant.layout.shape,
        variant.layout.tobytes(),
        variant.start,
        variant.goal,
    )


def _variant_group_key(
    variant: TaskVariant,
    source_record_index: int,
) -> tuple[int, int | None]:
    """Keep a source task's spatial/reversal orbit together."""
    return source_record_index, variant.perturbation_index


def _split_family_variants_for_es(
    variants: list[VariantRecord],
    *,
    validation_fraction: float,
    rng: np.random.Generator,
) -> tuple[list[VariantRecord], list[VariantRecord]]:
    """Split variants without separating an augmentation group."""
    grouped: dict[tuple[int, int | None], list[VariantRecord]] = {}
    for variant, source_record_index in variants:
        grouped.setdefault(
            _variant_group_key(variant, source_record_index),
            [],
        ).append((variant, source_record_index))

    groups = list(grouped.values())
    if len(groups) < 2:
        return variants, []

    rng.shuffle(groups)
    target_validation_count = int(round(len(variants) * validation_fraction))
    target_validation_count = max(
        1,
        min(len(variants) - 1, target_validation_count),
    )

    validation_groups: list[list[VariantRecord]] = []
    validation_count = 0
    for group in sorted(groups, key=len, reverse=True):
        if not validation_groups:
            validation_groups.append(group)
            validation_count += len(group)
            continue
        if len(validation_groups) == len(groups) - 1:
            break
        current_error = abs(validation_count - target_validation_count)
        proposed_error = abs(
            validation_count + len(group) - target_validation_count
        )
        if proposed_error < current_error:
            validation_groups.append(group)
            validation_count += len(group)

    validation_group_ids = {id(group) for group in validation_groups}
    validation = [
        record
        for group in validation_groups
        for record in group
    ]
    training = [
        record
        for group in groups
        if id(group) not in validation_group_ids
        for record in group
    ]
    return training, validation


def _arrays_from_variant_records(
    records_by_family: dict[int, list[VariantRecord]],
) -> dict[str, np.ndarray]:
    layouts: list[np.ndarray] = []
    layout_indices: list[int] = []
    starts: list[Position] = []
    goals: list[Position] = []
    family_indices: list[int] = []
    source_record_indices: list[int] = []
    optimal_path_lengths: list[int] = []
    layout_by_identity: dict[tuple[tuple[int, int], bytes], int] = {}

    family_variant_counts = {
        family_id: len(records)
        for family_id, records in records_by_family.items()
    }
    for family_id, records in records_by_family.items():
        for variant, source_record_index in records:
            identity = (variant.layout.shape, variant.layout.tobytes())
            layout_index = layout_by_identity.get(identity)
            if layout_index is None:
                layout_index = len(layouts)
                layout_by_identity[identity] = layout_index
                layouts.append(np.array(variant.layout, copy=True))
            layout_indices.append(layout_index)
            starts.append(variant.start)
            goals.append(variant.goal)
            family_indices.append(family_id)
            source_record_indices.append(source_record_index)
            optimal_path_lengths.append(variant.optimal_path_length)

    task_count = len(layout_indices)
    return {
        "layouts": np.asarray(layouts, dtype=np.uint8),
        "layout_indices": np.asarray(layout_indices, dtype=np.int64),
        "seeds": np.asarray(
            [_layout_seed(layout) for layout in layouts],
            dtype=np.int64,
        ),
        "starts": np.asarray(starts, dtype=np.int64).reshape(task_count, 2),
        "goals": np.asarray(goals, dtype=np.int64).reshape(task_count, 2),
        "family_indices": np.asarray(family_indices, dtype=np.int64),
        "source_record_indices": np.asarray(
            source_record_indices,
            dtype=np.int64,
        ),
        "optimal_path_lengths": np.asarray(
            optimal_path_lengths,
            dtype=np.int64,
        ),
        "variant_weights": np.asarray(
            [
                1.0 / family_variant_counts[family_id]
                for family_id in family_indices
            ],
            dtype=np.float64,
        ),
    }


def build_mixed_family_es_arrays(
    families: list[SourceFamily],
    *,
    seed: int = 42,
    validation_fraction: float = 0.2,
    perturbations_per_task: int = 1,
    max_variants_per_family: int | None = 64,
    max_edits: int = 4,
    max_attempts: int = 100,
    add_probability: float = 0.5,
    max_path_increase: int | None = None,
    excluded_task_identities: set[TaskIdentity] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Build train/ES arrays with shared families and disjoint groups."""
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between zero and one.")
    if perturbations_per_task < 1 and all(
        len(family.tasks) < 2 for family in families
    ):
        raise ValueError(
            "mixed-family-es requires perturbations or multiple source tasks "
            "per family so that an ES group can be held out."
        )

    selected_by_family = _generate_selected_variants(
        families,
        seed=seed,
        perturbations_per_task=perturbations_per_task,
        max_variants_per_family=max_variants_per_family,
        max_edits=max_edits,
        max_attempts=max_attempts,
        add_probability=add_probability,
        max_path_increase=max_path_increase,
        excluded_task_identities=excluded_task_identities,
    )
    rng = np.random.default_rng(seed + 1)
    training_by_family: dict[int, list[VariantRecord]] = {}
    validation_by_family: dict[int, list[VariantRecord]] = {}
    for family_id, variants in selected_by_family.items():
        training, validation = _split_family_variants_for_es(
            variants,
            validation_fraction=validation_fraction,
            rng=rng,
        )
        training_by_family[family_id] = training
        if validation:
            validation_by_family[family_id] = validation

    if not validation_by_family:
        raise ValueError(
            "mixed-family-es could not produce a non-empty ES split; "
            "increase perturbations_per_task or provide multiple source "
            "tasks per family."
        )

    return (
        _arrays_from_variant_records(training_by_family),
        _arrays_from_variant_records(validation_by_family),
    )


def build_split_arrays(
    families: list[SourceFamily],
    *,
    seed: int = 42,
    perturbations_per_task: int = 0,
    max_variants_per_family: int | None = 64,
    max_edits: int = 4,
    max_attempts: int = 100,
    add_probability: float = 0.5,
    max_path_increase: int | None = None,
) -> dict[str, np.ndarray]:
    """Build the existing NPZ task schema plus lineage and weights."""
    selected_by_family = _generate_selected_variants(
        families,
        seed=seed,
        perturbations_per_task=perturbations_per_task,
        max_variants_per_family=max_variants_per_family,
        max_edits=max_edits,
        max_attempts=max_attempts,
        add_probability=add_probability,
        max_path_increase=max_path_increase,
    )
    return _arrays_from_variant_records(selected_by_family)
