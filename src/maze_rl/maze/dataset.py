from pathlib import Path

import numpy as np

from maze_rl.maze.generator import (
    generate_valid_maze,
    generate_valid_tasks_for_maze,
)


def generate_mazes(
    seeds: np.ndarray,
    height: int,
    width: int,
    wall_probability: float,
    min_path_length: int,
    tasks_per_maze: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    layouts = np.empty(
        (len(seeds), height, width),
        dtype=np.uint8,
    )

    task_count = len(seeds) * tasks_per_maze

    layout_indices = np.empty(
        task_count,
        dtype=np.int64,
    )

    starts = np.empty(
        (task_count, 2),
        dtype=np.int64,
    )

    goals = np.empty(
        (task_count, 2),
        dtype=np.int64,
    )

    for i, seed in enumerate(seeds):
        rng = np.random.default_rng(int(seed))

        for _ in range(1000):
            maze, _, _ = generate_valid_maze(
                height=height,
                width=width,
                wall_probability=wall_probability,
                rng=rng,
                min_path_length=min_path_length,
            )

            try:
                task_starts, task_goals = (
                    generate_valid_tasks_for_maze(
                        maze=maze,
                        task_count=tasks_per_maze,
                        rng=rng,
                        min_path_length=min_path_length,
                    )
                )
            except RuntimeError:
                continue

            break
        else:
            raise RuntimeError(
                "Could not generate a maze with enough valid tasks."
            )

        task_start = i * tasks_per_maze
        task_end = task_start + tasks_per_maze

        layouts[i] = maze
        layout_indices[task_start:task_end] = i
        starts[task_start:task_end] = task_starts
        goals[task_start:task_end] = task_goals

    return layouts, layout_indices, starts, goals


def generate_tasks_for_layouts(
    layouts: np.ndarray,
    seeds: np.ndarray,
    tasks_per_maze: int,
    min_path_length: int,
    excluded_tasks_by_layout: dict[
        int,
        set[tuple[tuple[int, int], tuple[int, int]]],
    ] | None = None,
    seed_offset: int = 1_000_000_000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    task_count = len(layouts) * tasks_per_maze

    layout_indices = np.empty(
        task_count,
        dtype=np.int64,
    )

    starts = np.empty(
        (task_count, 2),
        dtype=np.int64,
    )

    goals = np.empty(
        (task_count, 2),
        dtype=np.int64,
    )

    excluded_tasks_by_layout = (
        excluded_tasks_by_layout or {}
    )

    for layout_index, maze in enumerate(layouts):
        rng = np.random.default_rng(
            int(seeds[layout_index]) + seed_offset
        )
        task_start = layout_index * tasks_per_maze
        task_end = task_start + tasks_per_maze

        task_starts, task_goals = (
            generate_valid_tasks_for_maze(
                maze=maze,
                task_count=tasks_per_maze,
                rng=rng,
                min_path_length=min_path_length,
                excluded_tasks=excluded_tasks_by_layout.get(
                    layout_index
                ),
            )
        )

        layout_indices[task_start:task_end] = (
            layout_index
        )
        starts[task_start:task_end] = task_starts
        goals[task_start:task_end] = task_goals

    return layout_indices, starts, goals

def generate_split_seeds(
    train_size: int,
    validation_size: int,
    test_size: int,
    master_seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate unique seeds for train, validation, and test sets.
    """
    total_size = (
        train_size
        + validation_size
        + test_size
    )

    rng = np.random.default_rng(master_seed)

    seeds = rng.choice(
        1_000_000_000,
        size=total_size,
        replace=False,
    )

    train_end = train_size
    validation_end = train_size + validation_size

    train_seeds = seeds[:train_end]

    validation_seeds = seeds[
        train_end:validation_end
    ]

    test_seeds = seeds[validation_end:]

    return (
        train_seeds,
        validation_seeds,
        test_seeds,
    )


def save_dataset(
    path: str | Path,
    layouts: np.ndarray,
    seeds: np.ndarray,
    layout_indices: np.ndarray,
    starts: np.ndarray,
    goals: np.ndarray,
) -> None:
    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez_compressed(
        path,
        layouts=layouts,
        layout_indices=layout_indices,
        seeds=seeds,
        starts=starts,
        goals=goals,
    )


def load_dataset(
    path: str | Path,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load maze layouts and task metadata.
    """
    with np.load(path) as data:
        layouts = (
            data["layouts"]
            if "layouts" in data
            else data["mazes"]
        )
        seeds = data["seeds"]

    return layouts, seeds
