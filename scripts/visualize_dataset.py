import argparse

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import numpy as np

from maze_rl.maze.pathfinding import shortest_path_length


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize a generated maze."
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="data/train.npz",
    )

    parser.add_argument(
        "--index",
        type=int,
        default=0,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    data = np.load(args.dataset)

    maze = data["mazes"][args.index]
    start = data["starts"][args.index]
    goal = data["goals"][args.index]

    distance = shortest_path_length(
        maze,
        tuple(start),
        tuple(goal),
    )

    # 0 = free cell
    # 1 = wall
    cmap = ListedColormap([
        "white",
        "black",
    ])

    plt.imshow(
        maze,
        cmap=cmap,
        interpolation="nearest",
        vmin=0,
        vmax=1,
    )

    start_row, start_col = start
    goal_row, goal_col = goal

    # Color the entire start cell red.
    plt.gca().add_patch(
        plt.Rectangle(
            (start_col - 0.5, start_row - 0.5),
            1,
            1,
            facecolor="red",
        )
    )

    # Color the entire goal cell blue.
    plt.gca().add_patch(
        plt.Rectangle(
            (goal_col - 0.5, goal_row - 0.5),
            1,
            1,
            facecolor="blue",
        )
    )

    # Add labels inside the cells.
    plt.text(
        start_col,
        start_row,
        "S",
        ha="center",
        va="center",
        color="white",
        fontweight="bold",
    )

    plt.text(
        goal_col,
        goal_row,
        "G",
        ha="center",
        va="center",
        color="white",
        fontweight="bold",
    )

    height, width = maze.shape

    # Grid lines between cells.
    plt.xticks(
        np.arange(-0.5, width, 1),
        [],
    )

    plt.yticks(
        np.arange(-0.5, height, 1),
        [],
    )

    plt.grid(True)

    # Legend.
    legend_elements = [
        Patch(
            facecolor="white",
            edgecolor="black",
            label="Free cell",
        ),
        Patch(
            facecolor="black",
            edgecolor="black",
            label="Wall",
        ),
        Patch(
            facecolor="red",
            edgecolor="black",
            label="Start",
        ),
        Patch(
            facecolor="blue",
            edgecolor="black",
            label="Goal",
        ),
    ]

    plt.legend(
        handles=legend_elements,
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
    )

    plt.title(
        f"Maze {args.index} | Shortest path: {distance}"
    )

    plt.tight_layout()
    plt.show()

    print(f"Maze index: {args.index}")
    print(f"Start: {start}")
    print(f"Goal: {goal}")
    print(f"Shortest path length: {distance}")


if __name__ == "__main__":
    main()