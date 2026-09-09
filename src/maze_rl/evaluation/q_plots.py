from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    "/tmp/maze-rl-matplotlib",
)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import numpy as np
import torch
from tqdm import tqdm


ACTION_LABELS = [
    "U",
    "D",
    "L",
    "R",
]

ACTION_ARROWS = {
    0: ((0.28, -0.06), (0.28, -0.38)),
    1: ((0.28, 0.06), (0.28, 0.38)),
    2: ((-0.06, 0.28), (-0.38, 0.28)),
    3: ((0.06, 0.28), (0.38, 0.28)),
}


@dataclass
class QSnapshotPlotResult:
    image_count: int
    video_count: int
    skipped_video_reason: str | None = None


def task_state_key(
    layout_index: int,
    row: int,
    col: int,
    goal: np.ndarray,
) -> tuple[int, int, int, int, int]:
    goal_row, goal_col = (
        int(goal[0]),
        int(goal[1]),
    )

    return (
        layout_index,
        row,
        col,
        goal_row,
        goal_col,
    )


def q_values_for_cell(
    q_table: dict,
    layout_index: int,
    row: int,
    col: int,
    goal: np.ndarray,
) -> np.ndarray | None:
    state = task_state_key(
        layout_index,
        row,
        col,
        goal,
    )
    values = q_table.get(state)

    if values is not None:
        return values

    legacy_state = (
        layout_index,
        row,
        col,
    )

    return q_table.get(legacy_state)


def draw_q_values(
    axis,
    col: int,
    row: int,
    values: np.ndarray | None,
) -> None:
    if values is None:
        axis.text(
            col,
            row,
            "unseen",
            ha="center",
            va="center",
            color="#6b7280",
            fontsize=6,
        )
        return

    best_value = np.max(values)
    best_actions = np.flatnonzero(
        np.isclose(values, best_value)
    )

    for action in best_actions:
        start_offset, end_offset = ACTION_ARROWS[
            int(action)
        ]
        axis.annotate(
            "",
            xy=(
                col + end_offset[0],
                row + end_offset[1],
            ),
            xytext=(
                col + start_offset[0],
                row + start_offset[1],
            ),
            arrowprops={
                "arrowstyle": "-|>",
                "color": "#d62728",
                "linewidth": 1.4,
                "mutation_scale": 9,
                "shrinkA": 0,
                "shrinkB": 0,
            },
            zorder=3,
        )

    y_offsets = [
        -0.27,
        -0.09,
        0.09,
        0.27,
    ]

    for label, value, y_offset in zip(
        ACTION_LABELS,
        values,
        y_offsets,
    ):
        color = (
            "#d62728"
            if np.isclose(value, best_value)
            else "black"
        )

        axis.text(
            col,
            row + y_offset,
            f"{label}:{value:.2f}",
            ha="center",
            va="center",
            color=color,
            fontsize=6,
            zorder=4,
        )


def draw_goal_cell(
    axis,
    col: int,
    row: int,
    reached: bool,
) -> None:
    status = (
        "reached"
        if reached
        else "goal"
    )

    axis.text(
        col,
        row,
        f"G\n{status}",
        ha="center",
        va="center",
        color="black",
        fontweight="bold",
        fontsize=7,
        linespacing=1.05,
        zorder=5,
    )


def plot_task_q_values(
    env,
    q_table: dict,
    task_index: int,
    epoch: int,
    output_path: Path,
    goal_reached: bool | None = None,
) -> None:
    layout_index = int(
        env.layout_indices[task_index]
    )
    maze = env.layouts[layout_index]
    start = env.starts[task_index]
    goal = env.goals[task_index]

    cmap = ListedColormap(
        [
            "white",
            "black",
        ]
    )

    height, width = maze.shape
    figure, axis = plt.subplots(
        figsize=(max(7, width * 0.7), max(7, height * 0.7))
    )

    axis.imshow(
        maze,
        cmap=cmap,
        interpolation="nearest",
        vmin=0,
        vmax=1,
    )

    start_row, start_col = (
        int(start[0]),
        int(start[1]),
    )
    goal_row, goal_col = (
        int(goal[0]),
        int(goal[1]),
    )

    axis.add_patch(
        plt.Rectangle(
            (start_col - 0.5, start_row - 0.5),
            1,
            1,
            facecolor="#ffb3b3",
            edgecolor="black",
        )
    )

    axis.add_patch(
        plt.Rectangle(
            (goal_col - 0.5, goal_row - 0.5),
            1,
            1,
            facecolor="#99c2ff",
            edgecolor="black",
        )
    )

    for row in range(height):
        for col in range(width):
            if maze[row, col] == 1:
                continue

            if row == goal_row and col == goal_col:
                continue

            draw_q_values(
                axis=axis,
                col=col,
                row=row,
                values=q_values_for_cell(
                    q_table=q_table,
                    layout_index=layout_index,
                    row=row,
                    col=col,
                    goal=goal,
                ),
            )

    goal_values = q_values_for_cell(
        q_table=q_table,
        layout_index=layout_index,
        row=goal_row,
        col=goal_col,
        goal=goal,
    )
    reached = (
        goal_values is not None
        if goal_reached is None
        else goal_reached
    )

    axis.text(
        start_col,
        start_row - 0.35,
        "S",
        ha="center",
        va="center",
        color="black",
        fontweight="bold",
        fontsize=8,
    )

    draw_goal_cell(
        axis=axis,
        col=goal_col,
        row=goal_row,
        reached=reached,
    )

    axis.set_xticks(
        np.arange(-0.5, width, 1),
        [],
    )
    axis.set_yticks(
        np.arange(-0.5, height, 1),
        [],
    )
    axis.grid(True)

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
            facecolor="#ffb3b3",
            edgecolor="black",
            label="Start",
        ),
        Patch(
            facecolor="#99c2ff",
            edgecolor="black",
            label="Goal",
        ),
    ]

    axis.legend(
        handles=legend_elements,
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
    )

    axis.set_title(
        f"Task {task_index} | Layout {layout_index} | Epoch {epoch}"
    )
    figure.tight_layout()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    figure.savefig(
        output_path,
        dpi=150,
    )
    plt.close(figure)


def task_observation_for_cell(
    maze: np.ndarray,
    row: int,
    col: int,
    goal: np.ndarray,
) -> np.ndarray:
    observation = np.zeros(
        (
            3,
            maze.shape[0],
            maze.shape[1],
        ),
        dtype=np.float32,
    )

    observation[0] = maze
    observation[1, row, col] = 1.0
    observation[
        2,
        int(goal[0]),
        int(goal[1]),
    ] = 1.0

    return observation


@torch.no_grad()
def dqn_q_values_for_cell(
    agent,
    maze: np.ndarray,
    row: int,
    col: int,
    goal: np.ndarray,
) -> np.ndarray:
    observation = task_observation_for_cell(
        maze=maze,
        row=row,
        col=col,
        goal=goal,
    )
    observation_tensor = torch.as_tensor(
        observation,
        dtype=torch.float32,
        device=agent.device,
    ).unsqueeze(0)

    return (
        agent.online_network(
            observation_tensor
        )
        .squeeze(0)
        .detach()
        .cpu()
        .numpy()
    )


@torch.no_grad()
def state_value_for_cell(
    agent,
    maze: np.ndarray,
    row: int,
    col: int,
    goal: np.ndarray,
) -> float:
    observation = task_observation_for_cell(
        maze=maze,
        row=row,
        col=col,
        goal=goal,
    )
    observation_tensor = torch.as_tensor(
        observation,
        dtype=torch.float32,
        device=agent.device,
    ).unsqueeze(0)

    _, value = agent.network(
        observation_tensor
    )

    return float(value.item())


def draw_state_value(
    axis,
    col: int,
    row: int,
    value: float,
) -> None:
    axis.text(
        col,
        row,
        f"V\n{value:.2f}",
        ha="center",
        va="center",
        color="black",
        fontsize=7,
        linespacing=1.05,
        zorder=4,
    )


def _plot_task_neural_values(
    env,
    agent,
    algorithm: str,
    task_index: int,
    epoch: int,
    output_path: Path,
    goal_reached: bool | None = None,
) -> None:
    layout_index = int(
        env.layout_indices[task_index]
    )
    maze = env.layouts[layout_index]
    start = env.starts[task_index]
    goal = env.goals[task_index]

    cmap = ListedColormap(
        [
            "white",
            "black",
        ]
    )

    height, width = maze.shape
    figure, axis = plt.subplots(
        figsize=(max(7, width * 0.7), max(7, height * 0.7))
    )

    axis.imshow(
        maze,
        cmap=cmap,
        interpolation="nearest",
        vmin=0,
        vmax=1,
    )

    start_row, start_col = (
        int(start[0]),
        int(start[1]),
    )
    goal_row, goal_col = (
        int(goal[0]),
        int(goal[1]),
    )

    axis.add_patch(
        plt.Rectangle(
            (start_col - 0.5, start_row - 0.5),
            1,
            1,
            facecolor="#ffb3b3",
            edgecolor="black",
        )
    )

    axis.add_patch(
        plt.Rectangle(
            (goal_col - 0.5, goal_row - 0.5),
            1,
            1,
            facecolor="#99c2ff",
            edgecolor="black",
        )
    )

    for row in range(height):
        for col in range(width):
            if maze[row, col] == 1:
                continue

            if row == goal_row and col == goal_col:
                continue

            if algorithm == "dqn":
                draw_q_values(
                    axis=axis,
                    col=col,
                    row=row,
                    values=dqn_q_values_for_cell(
                        agent=agent,
                        maze=maze,
                        row=row,
                        col=col,
                        goal=goal,
                    ),
                )
            else:
                draw_state_value(
                    axis=axis,
                    col=col,
                    row=row,
                    value=state_value_for_cell(
                        agent=agent,
                        maze=maze,
                        row=row,
                        col=col,
                        goal=goal,
                    ),
                )

    axis.text(
        start_col,
        start_row - 0.35,
        "S",
        ha="center",
        va="center",
        color="black",
        fontweight="bold",
        fontsize=8,
    )

    draw_goal_cell(
        axis=axis,
        col=goal_col,
        row=goal_row,
        reached=(
            False
            if goal_reached is None
            else goal_reached
        ),
    )

    axis.set_xticks(
        np.arange(-0.5, width, 1),
        [],
    )
    axis.set_yticks(
        np.arange(-0.5, height, 1),
        [],
    )
    axis.grid(True)

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
            facecolor="#ffb3b3",
            edgecolor="black",
            label="Start",
        ),
        Patch(
            facecolor="#99c2ff",
            edgecolor="black",
            label="Goal",
        ),
    ]

    axis.legend(
        handles=legend_elements,
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
    )

    value_name = (
        "Q values"
        if algorithm == "dqn"
        else "State values"
    )
    axis.set_title(
        f"{value_name} | Task {task_index} | Layout {layout_index} | Epoch {epoch}"
    )
    figure.tight_layout()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    figure.savefig(
        output_path,
        dpi=150,
    )
    plt.close(figure)


def plot_neural_snapshots(
    env,
    agent,
    algorithm: str,
    model_snapshots: list[dict],
    task_indices: list[int],
    output_dir: Path,
    write_videos: bool = True,
    video_fps: float = 8.0,
    show_progress: bool = True,
) -> QSnapshotPlotResult:
    if algorithm not in {
        "dqn",
        "a2c",
        "ppo",
    }:
        return QSnapshotPlotResult(
            image_count=0,
            video_count=0,
            skipped_video_reason=None,
        )

    network = (
        agent.online_network
        if algorithm == "dqn"
        else agent.network
    )
    original_state_dict = {
        key: value.detach().cpu().clone()
        for key, value in network.state_dict().items()
    }

    plot_count = 0
    video_count = 0
    skipped_video_reason = None
    unique_task_indices = sorted(
        set(task_indices)
    )
    valid_snapshots = [
        (fallback_index, snapshot)
        for fallback_index, snapshot in enumerate(
            model_snapshots,
            start=1,
        )
        if int(snapshot["epoch"]) != 0
    ]

    progress = tqdm(
        total=(
            len(unique_task_indices)
            * len(valid_snapshots)
        ),
        desc=(
            "Q-value plots"
            if algorithm == "dqn"
            else "State-value plots"
        ),
        unit="plot",
        disable=not show_progress,
    )

    value_slug = (
        "q_values"
        if algorithm == "dqn"
        else "state_values"
    )

    try:
        for task_index in unique_task_indices:
            task_dir = output_dir / f"task_{task_index:05d}"
            frame_dir = task_dir / "frames"
            frame_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            for old_plot in task_dir.glob("*.png"):
                old_plot.unlink()

            for old_plot in frame_dir.glob("*.png"):
                old_plot.unlink()

            for old_video in task_dir.glob("*.mp4"):
                old_video.unlink()

            frame_paths = []

            for fallback_index, snapshot in valid_snapshots:
                network.load_state_dict(
                    snapshot[
                        "model_state_dict"
                    ]
                )
                epoch = int(snapshot["epoch"])
                reached_task_indices = snapshot.get(
                    "reached_task_indices"
                )
                goal_reached = (
                    None
                    if reached_task_indices is None
                    else task_index
                    in {
                        int(reached_task_index)
                        for reached_task_index in reached_task_indices
                    }
                )

                snapshot_index = int(
                    snapshot.get(
                        "snapshot_index",
                        fallback_index,
                    )
                )

                output_path = (
                    frame_dir
                    / f"snapshot_{snapshot_index:03d}.png"
                )

                _plot_task_neural_values(
                    env=env,
                    agent=agent,
                    algorithm=algorithm,
                    task_index=task_index,
                    epoch=epoch,
                    output_path=output_path,
                    goal_reached=goal_reached,
                )
                frame_paths.append(output_path)
                plot_count += 1
                progress.update()

            if write_videos and frame_paths:
                video_path = (
                    task_dir
                    / f"task_{task_index:05d}_{value_slug}.mp4"
                )
                try:
                    write_q_snapshot_video(
                        frame_paths=frame_paths,
                        output_path=video_path,
                        fps=video_fps,
                    )
                    video_count += 1
                except RuntimeError as error:
                    skipped_video_reason = str(error)
    finally:
        progress.close()
        network.load_state_dict(
            original_state_dict
        )

    return QSnapshotPlotResult(
        image_count=plot_count,
        video_count=video_count,
        skipped_video_reason=skipped_video_reason,
    )


def plot_q_snapshots(
    env,
    q_snapshots: list[dict],
    task_indices: list[int],
    output_dir: Path,
    write_videos: bool = True,
    video_fps: float = 8.0,
    show_progress: bool = True,
) -> QSnapshotPlotResult:
    plot_count = 0
    video_count = 0
    skipped_video_reason = None
    unique_task_indices = sorted(
        set(task_indices)
    )
    valid_snapshots = [
        (fallback_index, snapshot)
        for fallback_index, snapshot in enumerate(
            q_snapshots,
            start=1,
        )
        if int(snapshot["epoch"]) != 0
    ]

    progress = tqdm(
        total=(
            len(unique_task_indices)
            * len(valid_snapshots)
        ),
        desc="Q-value plots",
        unit="plot",
        disable=not show_progress,
    )

    try:
        for task_index in unique_task_indices:
            task_dir = output_dir / f"task_{task_index:05d}"
            frame_dir = task_dir / "frames"
            frame_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            for old_plot in task_dir.glob("*.png"):
                old_plot.unlink()

            for old_plot in frame_dir.glob("*.png"):
                old_plot.unlink()

            for old_video in task_dir.glob("*.mp4"):
                old_video.unlink()

            frame_paths = []

            for fallback_index, snapshot in valid_snapshots:
                epoch = int(snapshot["epoch"])
                reached_task_indices = snapshot.get(
                    "reached_task_indices"
                )
                goal_reached = (
                    None
                    if reached_task_indices is None
                    else task_index
                    in {
                        int(reached_task_index)
                        for reached_task_index in reached_task_indices
                    }
                )

                snapshot_index = int(
                    snapshot.get(
                        "snapshot_index",
                        fallback_index,
                    )
                )

                output_path = (
                    frame_dir
                    / f"snapshot_{snapshot_index:03d}.png"
                )

                plot_task_q_values(
                    env=env,
                    q_table=snapshot["q"],
                    task_index=task_index,
                    epoch=epoch,
                    output_path=output_path,
                    goal_reached=goal_reached,
                )
                frame_paths.append(output_path)
                plot_count += 1
                progress.update()

            if write_videos and frame_paths:
                video_path = (
                    task_dir
                    / f"task_{task_index:05d}_q_values.mp4"
                )
                try:
                    write_q_snapshot_video(
                        frame_paths=frame_paths,
                        output_path=video_path,
                        fps=video_fps,
                    )
                    video_count += 1
                except RuntimeError as error:
                    skipped_video_reason = str(error)
    finally:
        progress.close()

    return QSnapshotPlotResult(
        image_count=plot_count,
        video_count=video_count,
        skipped_video_reason=skipped_video_reason,
    )


def write_q_snapshot_video(
    frame_paths: list[Path],
    output_path: Path,
    fps: float,
) -> None:
    try:
        import imageio.v2 as imageio
    except ImportError as error:
        raise RuntimeError(
            "install imageio[ffmpeg] to enable MP4 export"
        ) from error

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    try:
        with imageio.get_writer(
            output_path,
            fps=fps,
            codec="libx264",
            macro_block_size=1,
        ) as writer:
            for frame_path in frame_paths:
                writer.append_data(
                    pad_frame_to_even_dimensions(
                        imageio.imread(frame_path)
                    )
                )
    except Exception as error:
        raise RuntimeError(
            f"could not write {output_path}: {error}"
        ) from error


def pad_frame_to_even_dimensions(
    frame: np.ndarray,
) -> np.ndarray:
    height, width = frame.shape[:2]
    pad_height = height % 2
    pad_width = width % 2

    if not pad_height and not pad_width:
        return frame

    padding = [
        (0, pad_height),
        (0, pad_width),
    ]
    padding.extend(
        [(0, 0)] * (frame.ndim - 2)
    )

    return np.pad(
        frame,
        padding,
        mode="edge",
    )
