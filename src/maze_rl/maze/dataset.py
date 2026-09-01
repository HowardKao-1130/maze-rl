from pathlib import Path

import numpy as np

from maze_rl.maze.generator import generate_valid_maze


def generate_mazes(
    seeds: np.ndarray,
    height: int,
    width: int,
    wall_probability: float,
    min_path_length: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mazes = np.empty(
        (len(seeds), height, width),
        dtype=np.uint8,
    )

    starts = np.empty(
        (len(seeds), 2),
        dtype=np.int64,
    )

    goals = np.empty(
        (len(seeds), 2),
        dtype=np.int64,
    )

    for i, seed in enumerate(seeds):
        rng = np.random.default_rng(int(seed))

        maze, start, goal = generate_valid_maze(
            height=height,
            width=width,
            wall_probability=wall_probability,
            rng=rng,
            min_path_length=min_path_length,
        )

        mazes[i] = maze
        starts[i] = start
        goals[i] = goal

    return mazes, starts, goals

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
    mazes: np.ndarray,
    seeds: np.ndarray,
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
        mazes=mazes,
        seeds=seeds,
        starts=starts,
        goals=goals,
    )


def load_dataset(
    path: str | Path,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load maze layouts and seeds.
    """
    with np.load(path) as data:
        mazes = data["mazes"]
        seeds = data["seeds"]

    return mazes, seeds