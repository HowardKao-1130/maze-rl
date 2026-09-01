import numpy as np

from maze_rl.maze.pathfinding import shortest_path_length


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