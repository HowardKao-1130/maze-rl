from collections import deque

import numpy as np


Position = tuple[int, int]


def shortest_distances(
    maze: np.ndarray,
    start: Position,
) -> dict[Position, int]:
    """
    Return shortest-path distances from start to every reachable cell.
    """
    height, width = maze.shape

    distances = {start: 0}
    queue = deque([start])

    directions = [
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1),
    ]

    while queue:
        row, col = queue.popleft()

        for d_row, d_col in directions:
            next_row = row + d_row
            next_col = col + d_col
            next_position = (next_row, next_col)

            if not (
                0 <= next_row < height
                and 0 <= next_col < width
            ):
                continue

            if maze[next_row, next_col] == 1:
                continue

            if next_position in distances:
                continue

            distances[next_position] = (
                distances[(row, col)] + 1
            )

            queue.append(next_position)

    return distances


def shortest_path_length(
    maze: np.ndarray,
    start: Position,
    goal: Position,
) -> int | None:
    """
    Return the shortest path length between start and goal.

    Returns None if goal is unreachable.
    """
    distances = shortest_distances(maze, start)
    return distances.get(goal)


def traversable_positions(
    maze: np.ndarray,
) -> list[Position]:
    """
    Return all non-wall positions.
    """
    rows, cols = np.where(maze == 0)

    return [
        (int(row), int(col))
        for row, col in zip(rows, cols)
    ]