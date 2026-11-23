from __future__ import annotations

import numpy as np

from maze_rl.maze.augmentation import (
    MazeTask,
    augment_task,
    generate_perturbed_variants,
    generate_spatial_variants,
)
from maze_rl.maze.pathfinding import shortest_path_length


def asymmetric_task() -> MazeTask:
    layout = np.zeros((5, 5), dtype=np.uint8)
    layout[0, 1] = 1
    layout[1, 4] = 1
    layout[3, 0] = 1
    layout[4, 2] = 1
    return MazeTask(layout, (0, 0), (4, 4))


def test_spatial_variants_generate_the_d4_reversal_orbit():
    task = asymmetric_task()

    variants = generate_spatial_variants(task)

    assert len(variants) == 16
    assert sum(variant.reversed_endpoints for variant in variants) == 8
    assert {
        variant.transform
        for variant in variants
    } == {
        "rotate_0",
        "rotate_0_reflect",
        "rotate_90",
        "rotate_90_reflect",
        "rotate_180",
        "rotate_180_reflect",
        "rotate_270",
        "rotate_270_reflect",
    }
    for variant in variants:
        assert shortest_path_length(
            variant.layout,
            variant.start,
            variant.goal,
        ) == variant.optimal_path_length


def test_perturbations_never_decrease_source_path_length():
    task = asymmetric_task()
    source_length = shortest_path_length(
        task.layout,
        task.start,
        task.goal,
    )

    variants = generate_perturbed_variants(
        task,
        4,
        rng=np.random.default_rng(42),
        max_edits=3,
        max_attempts=100,
    )

    assert variants
    for variant in variants:
        assert variant.optimal_path_length >= source_length
        assert variant.layout[variant.start] == 0
        assert variant.layout[variant.goal] == 0


def test_augmentation_applies_spatial_variants_to_each_perturbation():
    variants = augment_task(
        asymmetric_task(),
        perturbation_count=2,
        rng=np.random.default_rng(7),
        max_edits=2,
    )

    assert len(variants) >= 16
    assert any(variant.perturbation_index is None for variant in variants)
    assert any(variant.perturbation_index is not None for variant in variants)
    assert len({
        (variant.layout.tobytes(), variant.start, variant.goal)
        for variant in variants
    }) == len(variants)
