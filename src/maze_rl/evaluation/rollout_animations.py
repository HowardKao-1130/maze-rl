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

from matplotlib import animation
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

from maze_rl.evaluation.q_plots import (
    draw_q_values,
    draw_state_value,
    dqn_q_values_for_cell,
    q_values_for_cell,
    state_value_for_cell,
)


@dataclass
class RolloutAnimationResult:
    animation_count: int
    skipped_animation_reason: str | None = None


def select_rollout_traces(
    results: list[dict],
) -> list[dict]:
    traces = [
        result
        for result in results
        if "rollout_trace" in result
    ]

    selected = []
    seen_task_indices = set()

    for result in traces:
        task_index = int(result["task_index"])

        if task_index in seen_task_indices:
            continue

        selected.append(result)
        seen_task_indices.add(task_index)

    return selected


def rollout_animation_path(
    output_dir: Path,
    split_label: str,
) -> Path:
    return output_dir / f"{split_label}_rollouts.mp4"


def plot_rollout_animations(
    env,
    results: list[dict],
    output_dir: Path,
    split_label: str,
    fps: float = 8.0,
    progress_callback=None,
    algorithm: str | None = None,
    agent=None,
    q_table: dict | None = None,
) -> RolloutAnimationResult:
    selected_results = select_rollout_traces(results)

    if not selected_results:
        return RolloutAnimationResult(
            animation_count=0,
            skipped_animation_reason=None,
        )

    output_path = rollout_animation_path(
        output_dir=output_dir,
        split_label=split_label,
    )

    try:
        write_rollout_animation(
            env=env,
            results=selected_results,
            split_label=split_label,
            output_path=output_path,
            fps=fps,
            progress_callback=progress_callback,
            algorithm=algorithm,
            agent=agent,
            q_table=q_table,
        )
    except RuntimeError as error:
        return RolloutAnimationResult(
            animation_count=0,
            skipped_animation_reason=str(error),
        )

    return RolloutAnimationResult(
        animation_count=1,
        skipped_animation_reason=None,
    )


def write_rollout_animation(
    env,
    results: list[dict],
    split_label: str,
    output_path: Path,
    fps: float,
    progress_callback=None,
    algorithm: str | None = None,
    agent=None,
    q_table: dict | None = None,
) -> None:
    if not animation.writers.is_available(
        "ffmpeg"
    ):
        raise RuntimeError(
            "Matplotlib ffmpeg writer is not available; "
            "install ffmpeg to enable MP4 rollout animations."
        )

    frames = []
    frame_completed_tasks = []
    completed_tasks = 0

    for result in results:
        rollout_trace = result["rollout_trace"]
        positions = np.asarray(
            rollout_trace["positions"],
            dtype=int,
        )

        if positions.size == 0:
            continue

        for position_index in range(len(positions)):
            frames.append(
                (
                    result,
                    positions,
                    position_index,
                )
            )
            frame_completed_tasks.append(
                completed_tasks
                + int(
                    position_index
                    == len(positions) - 1
                )
            )

        completed_tasks += 1

    if not frames:
        raise RuntimeError(
            "rollout trace does not contain positions"
        )
    rollout_task_count = completed_tasks

    cmap = ListedColormap(
        [
            "white",
            "black",
        ]
    )

    height = int(
        getattr(
            env,
            "height",
            env.layouts.shape[1],
        )
    )
    width = int(
        getattr(
            env,
            "width",
            env.layouts.shape[2],
        )
    )
    task_count = int(
        getattr(
            env,
            "num_tasks",
            len(env.starts),
        )
    )
    figure, axis = plt.subplots(
        figsize=(max(5, width * 0.65), max(5, height * 0.65))
    )
    value_overlay_cache = {}

    def draw_value_overlay(
        maze,
        goal,
        task_index: int,
        layout_index: int,
        goal_row: int,
        goal_col: int,
    ) -> None:
        if algorithm is None:
            return

        if task_index not in value_overlay_cache:
            overlay_values = []

            for row in range(height):
                for col in range(width):
                    if maze[row, col] == 1:
                        continue

                    if row == goal_row and col == goal_col:
                        continue

                    if q_table is not None:
                        overlay_values.append(
                            (
                                "q",
                                col,
                                row,
                                q_values_for_cell(
                                    q_table=q_table,
                                    layout_index=layout_index,
                                    row=row,
                                    col=col,
                                    goal=goal,
                                ),
                            )
                        )
                    elif algorithm == "dqn" and agent is not None:
                        overlay_values.append(
                            (
                                "q",
                                col,
                                row,
                                dqn_q_values_for_cell(
                                    agent=agent,
                                    maze=maze,
                                    row=row,
                                    col=col,
                                    goal=goal,
                                ),
                            )
                        )
                    elif (
                        algorithm in {
                            "a2c",
                            "ppo",
                        }
                        and agent is not None
                    ):
                        overlay_values.append(
                            (
                                "v",
                                col,
                                row,
                                state_value_for_cell(
                                    agent=agent,
                                    maze=maze,
                                    row=row,
                                    col=col,
                                    goal=goal,
                                ),
                            )
                        )

            value_overlay_cache[task_index] = overlay_values

        for value_type, col, row, value in value_overlay_cache[
            task_index
        ]:
            if value_type == "q":
                draw_q_values(
                    axis=axis,
                    col=col,
                    row=row,
                    values=value,
                )
            else:
                draw_state_value(
                    axis=axis,
                    col=col,
                    row=row,
                    value=value,
                )

    def update(frame_index: int):
        result, positions, position_index = frames[
            frame_index
        ]
        task_index = int(result["task_index"])
        layout_index = int(result["layout_index"])
        maze = env.layouts[layout_index]
        start = env.starts[task_index]
        goal = env.goals[task_index]
        path = positions[: position_index + 1]
        rows = path[:, 0]
        cols = path[:, 1]

        axis.clear()
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

        draw_value_overlay(
            maze=maze,
            goal=goal,
            task_index=task_index,
            layout_index=layout_index,
            goal_row=goal_row,
            goal_col=goal_col,
        )

        axis.add_patch(
            plt.Rectangle(
                (start_col - 0.5, start_row - 0.5),
                1,
                1,
                facecolor="#fca5a5",
                edgecolor="black",
            )
        )
        axis.add_patch(
            plt.Rectangle(
                (goal_col - 0.5, goal_row - 0.5),
                1,
                1,
                facecolor="#93c5fd",
                edgecolor="black",
            )
        )
        axis.text(
            start_col,
            start_row,
            "S",
            ha="center",
            va="center",
            color="black",
            fontweight="bold",
            zorder=4,
        )
        axis.text(
            goal_col,
            goal_row,
            "G",
            ha="center",
            va="center",
            color="black",
            fontweight="bold",
            zorder=4,
        )
        path_line, = axis.plot(
            cols,
            rows,
            color="#2563eb",
            linewidth=2.0,
            zorder=5,
        )
        agent_marker, = axis.plot(
            [cols[-1]],
            [rows[-1]],
            marker="o",
            markersize=10,
            markerfacecolor="#f97316",
            markeredgecolor="black",
            linestyle="",
            zorder=6,
        )
        axis.text(
            0.02,
            1.02,
            (
                f"{split_label} | task {task_index + 1}/{task_count} | "
                f"step {position_index}/{len(positions) - 1}"
            ),
            transform=axis.transAxes,
            ha="left",
            va="bottom",
            fontsize=10,
            fontweight="bold",
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
        axis.set_title(
            (
                f"{split_label} rollout | Task {task_index + 1} | "
                f"Maze {layout_index + 1}"
            )
        )
        figure.tight_layout()

        return (
            path_line,
            agent_marker,
        )

    rollout_animation = animation.FuncAnimation(
        figure,
        update,
        frames=len(frames),
        interval=1000 / fps,
        blit=False,
        repeat=False,
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    def task_progress_callback(
        current_frame: int,
        total_frames: int,
    ) -> None:
        if progress_callback is None:
            return

        frame_index = max(
            0,
            min(
                int(current_frame),
                len(frame_completed_tasks) - 1,
            ),
        )
        progress_callback(
            frame_completed_tasks[frame_index],
            rollout_task_count,
        )

    if progress_callback is not None:
        progress_callback(
            0,
            rollout_task_count,
        )

    try:
        rollout_animation.save(
            output_path,
            writer="ffmpeg",
            fps=fps,
            dpi=150,
            progress_callback=task_progress_callback,
        )
    except Exception as error:
        raise RuntimeError(
            f"could not write {output_path}: {error}"
        ) from error
    finally:
        plt.close(figure)
