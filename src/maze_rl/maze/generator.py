import numpy as np

from maze_rl.maze.pathfinding import (
    shortest_distances,
    shortest_path_length,
    traversable_positions,
)


def generate_obstacle_maze(
    height: int,
    width: int,
    wall_probability: float,
    rng: np.random.Generator,
) -> np.ndarray:
    maze = (
        rng.random((height, width)) < wall_probability
    ).astype(np.uint8)

    return maze

def generate_valid_maze(
    height: int,
    width: int,
    wall_probability: float,
    rng: np.random.Generator,
    min_path_length: int = 10,
    max_attempts: int = 1000,
):
    for _ in range(max_attempts):
        maze = generate_obstacle_maze(
            height=height,
            width=width,
            wall_probability=wall_probability,
            rng=rng,
        )

        free_positions = list(
            zip(*np.where(maze == 0))
        )

        if len(free_positions) < 2:
            continue

        start_index, goal_index = rng.choice(
            len(free_positions),
            size=2,
            replace=False,
        )

        start = free_positions[start_index]
        goal = free_positions[goal_index]

        distance = shortest_path_length(
            maze,
            start,
            goal,
        )

        if distance is None:
            continue

        if distance < min_path_length:
            continue

        return maze, start, goal

    raise RuntimeError(
        "Could not generate a valid maze."
    )


def generate_valid_tasks_for_maze(
    maze: np.ndarray,
    task_count: int,
    rng: np.random.Generator,
    min_path_length: int = 10,
    excluded_tasks: set[
        tuple[tuple[int, int], tuple[int, int]]
    ] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate unique valid start/goal tasks for one maze layout.
    """
    candidates = []
    excluded_tasks = excluded_tasks or set()

    for start in traversable_positions(maze):
        distances = shortest_distances(
            maze,
            start,
        )

        for goal, distance in distances.items():
            if goal == start:
                continue

            if distance < min_path_length:
                continue

            candidate = (start, goal)

            if candidate in excluded_tasks:
                continue

            candidates.append(candidate)

    if len(candidates) < task_count:
        raise RuntimeError(
            "Could not generate enough valid tasks for maze."
        )

    task_indices = rng.choice(
        len(candidates),
        size=task_count,
        replace=False,
    )

    starts = np.empty(
        (task_count, 2),
        dtype=np.int64,
    )
    goals = np.empty(
        (task_count, 2),
        dtype=np.int64,
    )

    for i, candidate_index in enumerate(task_indices):
        start, goal = candidates[int(candidate_index)]
        starts[i] = start
        goals[i] = goal

    return starts, goals
