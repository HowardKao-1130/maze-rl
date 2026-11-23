from __future__ import annotations

import json

import numpy as np

from maze_rl.maze.augmentation_dataset import (
    build_split_arrays,
    group_source_tasks,
    load_failure_log,
    split_source_families,
)


def write_failure_log(path, count=12):
    records = []
    wall_positions = (
        (1, 0),
        (1, 1),
        (1, 2),
        (1, 3),
        (2, 0),
        (2, 1),
        (2, 2),
        (2, 3),
        (3, 0),
        (3, 1),
        (3, 2),
        (3, 3),
    )
    for index in range(count):
        layout = np.zeros((5, 5), dtype=np.uint8)
        for bit, position in enumerate(wall_positions):
            if index & (1 << bit):
                layout[position] = 1
        start = [0, 0]
        goal = [4, 4]
        records.append(
            {
                "run_id": f"run-{index}",
                "maze": {
                    "layout": layout.tolist(),
                    "start": start,
                    "goal": goal,
                    "optimal_steps": 8,
                },
            }
        )
    path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )


def test_split_keeps_layout_families_atomic(tmp_path):
    path = tmp_path / "failure_runs.jsonl"
    write_failure_log(path)
    families = group_source_tasks(load_failure_log(path))

    splits = split_source_families(families, seed=11)

    assigned = {
        family.family_id: split_name
        for split_name, split_families in splits.items()
        for family in split_families
    }
    assert len(assigned) == len(families)
    assert all(
        assigned[family.family_id] == split_name
        for split_name, split_families in splits.items()
        for family in split_families
    )
    assert all(splits.values())


def test_split_arrays_include_inverse_family_weights(tmp_path):
    path = tmp_path / "failure_runs.jsonl"
    write_failure_log(path)
    families = group_source_tasks(load_failure_log(path))

    arrays = build_split_arrays(
        families[:2],
        seed=3,
        max_variants_per_family=4,
    )

    assert len(arrays["starts"]) == len(arrays["variant_weights"])
    for family_id in np.unique(arrays["family_indices"]):
        mask = arrays["family_indices"] == family_id
        assert np.allclose(
            arrays["variant_weights"][mask],
            1 / mask.sum(),
        )
