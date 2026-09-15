from __future__ import annotations

import numpy as np
import torch

from maze_rl.evaluation import q_plots
from maze_rl.evaluation import rollout_animations
from maze_rl.evaluation.q_plots import (
    draw_goal_cell,
    draw_q_values,
    pad_frame_to_even_dimensions,
)
from maze_rl.evaluation.rollout_animations import (
    plot_rollout_animations,
    rollout_animation_path,
    select_rollout_traces,
)


class RecordingAxis:
    def __init__(self) -> None:
        self.annotations = []
        self.texts = []

    def annotate(self, *args, **kwargs):
        self.annotations.append(
            (args, kwargs)
        )

    def text(self, *args, **kwargs):
        self.texts.append(
            (args, kwargs)
        )


def test_draw_q_values_highlights_all_tied_best_actions():
    axis = RecordingAxis()

    draw_q_values(
        axis=axis,
        col=2,
        row=3,
        values=np.array(
            [1.0, 0.5, 1.0, 1.0]
        ),
    )

    assert len(axis.annotations) == 3
    assert [
        text_args[2]
        for text_args, text_kwargs in axis.texts
        if text_kwargs["color"] == "#d62728"
    ] == [
        "U:1.00",
        "L:1.00",
        "R:1.00",
    ]


def test_draw_goal_cell_marks_reached_without_arrows():
    axis = RecordingAxis()

    draw_goal_cell(
        axis=axis,
        col=2,
        row=3,
        reached=True,
    )

    assert axis.annotations == []
    assert axis.texts[0][0][2] == "G\nreached"


def test_plot_q_snapshots_writes_frames_beside_video(
    tmp_path,
    monkeypatch,
):
    plotted_goal_statuses = []

    def fake_plot_task_q_values(
        env,
        q_table,
        task_index,
        epoch,
        output_path,
        goal_reached=None,
    ):
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        output_path.write_bytes(b"frame")
        plotted_goal_statuses.append(
            goal_reached
        )

    def fake_write_q_snapshot_video(
        frame_paths,
        output_path,
        fps,
    ):
        output_path.write_bytes(b"video")

    monkeypatch.setattr(
        q_plots,
        "plot_task_q_values",
        fake_plot_task_q_values,
    )
    monkeypatch.setattr(
        q_plots,
        "write_q_snapshot_video",
        fake_write_q_snapshot_video,
    )

    result = q_plots.plot_q_snapshots(
        env=object(),
        q_snapshots=[
            {
                "snapshot_index": 1,
                "epoch": 1,
                "q": {},
                "reached_task_indices": [2],
            }
        ],
        task_indices=[2],
        output_dir=tmp_path,
        show_progress=False,
    )

    assert result.image_count == 1
    assert result.video_count == 1
    assert plotted_goal_statuses == [True]
    assert (
        tmp_path
        / "task_00002"
        / "frames"
        / "snapshot_001.png"
    ).exists()
    assert (
        tmp_path
        / "task_00002"
        / "task_00002_q_values.mp4"
    ).exists()


class FakeNeuralPlotEnv:
    layout_indices = np.array([0])
    layouts = np.zeros(
        (1, 1, 1),
        dtype=np.int64,
    )
    starts = np.array([[0, 0]])
    goals = np.array([[0, 0]])


class FakeRolloutAnimationEnv:
    height = 3
    width = 3
    num_tasks = 2
    layout_indices = np.array([0, 0])
    layouts = np.array(
        [
            [
                [0, 1, 0],
                [0, 0, 0],
                [0, 1, 0],
            ]
        ],
        dtype=np.int64,
    )
    starts = np.array(
        [
            [0, 0],
            [2, 0],
        ]
    )
    goals = np.array(
        [
            [1, 2],
            [2, 2],
        ]
    )


def rollout_result(
    episode: int,
    task_index: int,
) -> dict:
    return {
        "episode": episode,
        "task_index": task_index,
        "layout_index": 0,
        "rollout_trace": {
            "positions": [
                [0, 0],
                [1, 0],
                [1, 1],
            ],
            "actions": [
                1,
                3,
            ],
        },
    }


def test_select_rollout_traces_defaults_to_each_task_once():
    results = [
        rollout_result(
            episode=1,
            task_index=0,
        ),
        rollout_result(
            episode=2,
            task_index=1,
        ),
    ]

    assert select_rollout_traces(results) == [
        results[0],
        results[1],
    ]


def test_select_rollout_traces_keeps_first_rollout_per_task():
    results = [
        rollout_result(
            episode=1,
            task_index=0,
        ),
        rollout_result(
            episode=2,
            task_index=0,
        ),
        rollout_result(
            episode=3,
            task_index=1,
        ),
    ]

    assert select_rollout_traces(results) == [
        results[0],
        results[2],
    ]


def test_plot_rollout_animations_writes_labeled_mp4s(
    tmp_path,
    monkeypatch,
):
    written_paths = []

    def fake_write_rollout_animation(
        env,
        results,
        split_label,
        output_path,
        fps,
        progress_callback=None,
        algorithm=None,
        agent=None,
        q_table=None,
    ):
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        output_path.write_bytes(b"video")
        written_paths.append(
            (
                split_label,
                output_path,
                fps,
                [
                    result["task_index"]
                    for result in results
                ],
                algorithm,
                q_table,
            )
        )

    monkeypatch.setattr(
        rollout_animations,
        "write_rollout_animation",
        fake_write_rollout_animation,
    )

    result = plot_rollout_animations(
        env=FakeRolloutAnimationEnv(),
        results=[
            rollout_result(
                episode=1,
                task_index=0,
            ),
            rollout_result(
                episode=2,
                task_index=1,
            ),
        ],
        output_dir=tmp_path,
        split_label="val_with_same_layout",
        fps=12.0,
        algorithm="q_learning",
        q_table={
            (
                0,
                0,
                0,
                1,
                2,
            ): np.array(
                [1.0, 0.0, 0.0, 0.0]
            ).tolist()
        },
    )

    expected_path = (
        tmp_path
        / "val_with_same_layout_rollouts.mp4"
    )

    assert result.animation_count == 1
    assert result.skipped_animation_reason is None
    assert written_paths == [
        (
            "val_with_same_layout",
            expected_path,
            12.0,
            [0, 1],
            "q_learning",
            {
                (
                    0,
                    0,
                    0,
                    1,
                    2,
                ): np.array(
                    [1.0, 0.0, 0.0, 0.0]
                ).tolist()
            },
        ),
    ]
    assert expected_path.exists()


def test_plot_rollout_animations_reports_skipped_video_reason(
    tmp_path,
    monkeypatch,
):
    def fake_write_rollout_animation(
        env,
        results,
        split_label,
        output_path,
        fps,
        progress_callback=None,
        algorithm=None,
        agent=None,
        q_table=None,
    ):
        raise RuntimeError(
            "Matplotlib ffmpeg writer is not available"
        )

    monkeypatch.setattr(
        rollout_animations,
        "write_rollout_animation",
        fake_write_rollout_animation,
    )

    result = plot_rollout_animations(
        env=FakeRolloutAnimationEnv(),
        results=[
            rollout_result(
                episode=1,
                task_index=0,
            )
        ],
        output_dir=tmp_path,
        split_label="val",
    )

    assert result.animation_count == 0
    assert (
        result.skipped_animation_reason
        == "Matplotlib ffmpeg writer is not available"
    )


def test_rollout_animation_path_uses_split_video_name(tmp_path):
    assert rollout_animation_path(
        output_dir=tmp_path,
        split_label="val",
    ) == (
        tmp_path
        / "val_rollouts.mp4"
    )


def test_write_rollout_animation_reports_task_progress(
    tmp_path,
    monkeypatch,
):
    progress_updates = []
    saved_callback = None

    class FakeAnimation:
        def __init__(
            self,
            figure,
            update,
            frames,
            interval,
            blit,
            repeat,
        ):
            self.frames = frames

        def save(
            self,
            output_path,
            writer,
            fps,
            dpi,
            progress_callback=None,
        ):
            nonlocal saved_callback
            saved_callback = progress_callback

            for frame_index in range(self.frames):
                progress_callback(
                    frame_index,
                    self.frames,
                )

            output_path.write_bytes(b"video")

    monkeypatch.setattr(
        rollout_animations.animation,
        "FuncAnimation",
        FakeAnimation,
    )
    monkeypatch.setattr(
        rollout_animations.animation.writers,
        "is_available",
        lambda writer: True,
    )

    rollout_animations.write_rollout_animation(
        env=FakeRolloutAnimationEnv(),
        results=[
            rollout_result(
                episode=1,
                task_index=0,
            ),
            rollout_result(
                episode=2,
                task_index=1,
            ),
        ],
        split_label="val",
        output_path=tmp_path / "val_rollouts.mp4",
        fps=8.0,
        progress_callback=lambda current, total: progress_updates.append(
            (
                current,
                total,
            )
        ),
    )

    assert saved_callback is not None
    assert progress_updates[0] == (0, 2)
    assert progress_updates[-1] == (2, 2)
    assert {
        total
        for _, total in progress_updates
    } == {2}


class FakeDQNPlotAgent:
    device = torch.device("cpu")

    def __init__(self):
        self.online_network = torch.nn.Linear(
            1,
            1,
        )


def test_plot_neural_snapshots_writes_frames_and_restores_model(
    tmp_path,
    monkeypatch,
):
    plotted_epochs = []

    def fake_plot_task_neural_values(
        env,
        agent,
        algorithm,
        task_index,
        epoch,
        output_path,
        goal_reached=None,
    ):
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        output_path.write_bytes(b"frame")
        plotted_epochs.append(epoch)

    def fake_write_q_snapshot_video(
        frame_paths,
        output_path,
        fps,
    ):
        output_path.write_bytes(b"video")

    monkeypatch.setattr(
        q_plots,
        "_plot_task_neural_values",
        fake_plot_task_neural_values,
    )
    monkeypatch.setattr(
        q_plots,
        "write_q_snapshot_video",
        fake_write_q_snapshot_video,
    )

    agent = FakeDQNPlotAgent()
    original_weight = (
        agent.online_network.weight
        .detach()
        .clone()
    )
    snapshot_state_dict = {
        key: value.detach().clone()
        for key, value in agent.online_network.state_dict().items()
    }
    snapshot_state_dict["weight"] = torch.ones_like(
        snapshot_state_dict["weight"]
    )

    result = q_plots.plot_neural_snapshots(
        env=FakeNeuralPlotEnv(),
        agent=agent,
        algorithm="dqn",
        model_snapshots=[
            {
                "snapshot_index": 1,
                "epoch": 3,
                "model_state_dict": snapshot_state_dict,
                "reached_task_indices": [0],
            }
        ],
        task_indices=[0],
        output_dir=tmp_path,
        show_progress=False,
    )

    assert result.image_count == 1
    assert result.video_count == 1
    assert plotted_epochs == [3]
    assert (
        tmp_path
        / "task_00000"
        / "frames"
        / "snapshot_001.png"
    ).exists()
    assert (
        tmp_path
        / "task_00000"
        / "task_00000_q_values.mp4"
    ).exists()
    torch.testing.assert_close(
        agent.online_network.weight,
        original_weight,
    )


def test_pad_frame_to_even_dimensions_pads_odd_axes():
    frame = np.zeros(
        (3, 5, 4),
        dtype=np.uint8,
    )
    frame[:, -1, :] = 7
    frame[-1, :, :] = 9

    padded = pad_frame_to_even_dimensions(frame)

    assert padded.shape == (4, 6, 4)
    np.testing.assert_array_equal(
        padded[-1, :, :],
        padded[-2, :, :],
    )
    np.testing.assert_array_equal(
        padded[:, -1, :],
        padded[:, -2, :],
    )


def test_pad_frame_to_even_dimensions_reuses_even_frame():
    frame = np.zeros(
        (4, 6, 3),
        dtype=np.uint8,
    )

    assert pad_frame_to_even_dimensions(frame) is frame
