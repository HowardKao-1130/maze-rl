from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

import torch

from maze_rl.training.metrics import EpisodeMetrics
from scripts.generate_dataset import parse_args as parse_dataset_args
from scripts.evaluate import default_checkpoint_path
from scripts.open_q_video import q_video_path
from scripts.open_tensorboard import (
    build_tensorboard_command,
    default_tensorboard_dir,
    latest_event_file,
    latest_logdir,
)
from scripts.train import (
    HierarchicalTaskSampler,
    configure_reproducibility,
    log_tensorboard_episode,
    log_tensorboard_epoch,
    log_tensorboard_optimization_epoch,
    log_tensorboard_training_round,
    neural_hyperparameters_from_args,
    optimization_epochs_for_round,
    rollout_episodes_for_algorithm,
    should_validate_epoch,
    tensorboard_default_enabled,
    validate_neural_hyperparameters,
)


class RecordingWriter:
    def __init__(self) -> None:
        self.scalars = []
        self.flush_count = 0

    def add_scalar(
        self,
        tag,
        value,
        step,
    ):
        self.scalars.append(
            (
                tag,
                value,
                step,
            )
        )

    def flush(self):
        self.flush_count += 1


class EpsilonAgent:
    epsilon = 0.25


class EntropyAgent:
    entropy_coefficient = 0.05


def make_metrics(
    episode: int = 3,
    epoch: int = 2,
) -> EpisodeMetrics:
    return EpisodeMetrics(
        episode=episode,
        epoch=epoch,
        task_index=0,
        layout_index=0,
        episode_return=1.5,
        steps=7,
        internal_updates=4,
        success=True,
        wall_collisions=1,
        optimal_path_length=5,
        path_efficiency=5 / 7,
        loss=0.3,
        policy_loss=0.2,
        value_loss=0.1,
        entropy=0.9,
        mean_advantage=0.05,
    )


def test_default_checkpoint_path_uses_algorithm_specific_suffix():
    assert default_checkpoint_path(
        "q_learning"
    ) == Path("runs/q_learning/checkpoint.pkl")
    assert default_checkpoint_path(
        "dqn"
    ) == Path("runs/dqn/checkpoint.pt")


def test_dataset_split_aliases_name_layout_counts(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "generate_dataset.py",
            "--train-mazes",
            "3",
            "--validation-mazes",
            "2",
            "--test-mazes",
            "1",
            "--tasks-per-maze",
            "4",
        ],
    )

    args = parse_dataset_args()

    assert args.train_mazes == 3
    assert args.validation_mazes == 2
    assert args.test_mazes == 1
    assert args.tasks_per_maze == 4


def test_legacy_dataset_split_flags_still_parse(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "generate_dataset.py",
            "--train",
            "3",
            "--validation",
            "2",
            "--test",
            "1",
        ],
    )

    args = parse_dataset_args()

    assert args.train_mazes == 3
    assert args.validation_mazes == 2
    assert args.test_mazes == 1


def test_neural_hyperparameters_filter_unset_values():
    args = SimpleNamespace(
        batch_size=32,
        clip_epsilon=None,
        entropy_coefficient=None,
        entropy_coefficient_decay=None,
        entropy_coefficient_min=None,
        epsilon_decay=None,
        epsilon_min=None,
        epsilon_start=None,
        gae_lambda=None,
        gamma=0.97,
        learning_rate=0.0001,
        min_replay_size=None,
        minibatch_size=None,
        replay_capacity=None,
        target_update_interval=None,
        update_epochs=None,
        value_coefficient=None,
    )

    assert neural_hyperparameters_from_args(
        args
    ) == {
        "batch_size": 32,
        "gamma": 0.97,
        "learning_rate": 0.0001,
    }


def test_neural_hyperparameters_reject_tabular_agents():
    with pytest.raises(
        ValueError,
        match="only supported for DNN agents",
    ):
        validate_neural_hyperparameters(
            "q_learning",
            {
                "learning_rate": 0.001,
            },
        )


def test_configure_reproducibility_enables_deterministic_torch():
    configure_reproducibility(123)

    assert os.environ[
        "CUBLAS_WORKSPACE_CONFIG"
    ] == ":4096:8"
    assert torch.are_deterministic_algorithms_enabled()
    assert not torch.backends.cudnn.benchmark
    assert torch.backends.cudnn.deterministic
    assert not torch.backends.cuda.matmul.allow_tf32
    assert not torch.backends.cudnn.allow_tf32


def test_q_video_path_uses_task_directory_layout():
    assert q_video_path(
        algorithm="sarsa",
        task_index=7,
        runs_dir=Path("runs"),
    ) == Path(
        "runs/sarsa/q_value_plots/task_00007/task_00007_q_values.mp4"
    )


def test_default_tensorboard_dir_uses_algorithm_run_directory():
    assert default_tensorboard_dir(
        "ppo"
    ) == Path("runs/ppo/tensorboard")


def test_latest_event_file_uses_newest_tensorboard_file(tmp_path):
    logdir = tmp_path / "tensorboard"
    logdir.mkdir()

    older = logdir / "events.out.tfevents.1.host.1.0"
    newer = logdir / "events.out.tfevents.2.host.2.0"
    ignored = logdir / "metrics.csv"

    older.write_text("")
    newer.write_text("")
    ignored.write_text("")

    older_mtime = 100.0
    newer_mtime = 200.0
    older.touch()
    newer.touch()
    ignored.touch()
    import os

    os.utime(
        older,
        (older_mtime, older_mtime),
    )
    os.utime(
        newer,
        (newer_mtime, newer_mtime),
    )

    assert latest_event_file(logdir) == newer


def test_latest_logdir_links_single_event_file(tmp_path):
    event_file = tmp_path / "events.out.tfevents.1.host.1.0"
    event_file.write_text("event")

    logdir = latest_logdir(
        event_file=event_file,
        link_root=tmp_path / "links",
    )

    links = list(logdir.iterdir())
    assert len(links) == 1
    assert links[0].is_symlink()
    assert links[0].resolve() == event_file


def test_tensorboard_command_passes_through_extra_args():
    assert build_tensorboard_command(
        Path("/tmp/tb"),
        [
            "--port",
            "6007",
        ],
    ) == [
        "tensorboard",
        "--logdir",
        "/tmp/tb",
        "--port",
        "6007",
    ]


def test_tensorboard_defaults_only_for_neural_agents():
    assert not tensorboard_default_enabled(
        "sarsa"
    )
    assert tensorboard_default_enabled(
        "dqn"
    )


def test_tensorboard_episode_and_transition_tags_use_distinct_steps():
    writer = RecordingWriter()

    log_tensorboard_episode(
        writer=writer,
        metrics=make_metrics(),
        agent=EpsilonAgent(),
        transition_step=42,
    )

    tags = {
        tag: step
        for tag, _, step in writer.scalars
    }

    assert tags["by_rollout/return"] == 3
    assert tags["by_transition/return"] == 42
    assert tags["by_rollout/loss/total"] == 3
    assert tags["by_transition/policy/entropy"] == 42


def test_tensorboard_logs_entropy_coefficient_for_policy_agents():
    writer = RecordingWriter()

    log_tensorboard_episode(
        writer=writer,
        metrics=make_metrics(),
        agent=EntropyAgent(),
        transition_step=42,
    )

    assert (
        "by_rollout/agent/entropy_coefficient",
        0.05,
        3,
    ) in writer.scalars
    assert (
        "by_transition/agent/entropy_coefficient",
        0.05,
        42,
    ) in writer.scalars


def test_tensorboard_epoch_training_round_and_optimization_tags():
    writer = RecordingWriter()
    metrics = [
        make_metrics(
            episode=1,
            epoch=5,
        ),
        make_metrics(
            episode=2,
            epoch=5,
        ),
    ]

    log_tensorboard_epoch(
        writer=writer,
        epoch=5,
        epoch_metrics=metrics,
        agent=EpsilonAgent(),
    )
    log_tensorboard_training_round(
        writer=writer,
        training_round=6,
        round_metrics=metrics,
        agent=EpsilonAgent(),
    )
    log_tensorboard_optimization_epoch(
        writer=writer,
        optimization_epoch=8,
        round_metrics=metrics,
        agent=EpsilonAgent(),
    )

    tags = {
        tag: step
        for tag, _, step in writer.scalars
    }

    assert tags["by_dataset_epoch/mean_return"] == 5
    assert tags["by_training_round/mean_return"] == 6
    assert tags["by_optimization_epoch/mean_return"] == 8
    assert tags["by_optimization_epoch/loss/mean_policy"] == 8


def test_optimization_epoch_counts_match_algorithm_meaning():
    metrics = [
        make_metrics(),
    ]

    class PPOAgent:
        update_epochs = 4

    assert optimization_epochs_for_round(
        "a2c",
        object(),
        metrics,
    ) == 1
    assert optimization_epochs_for_round(
        "ppo",
        PPOAgent(),
        metrics,
    ) == 4
    assert optimization_epochs_for_round(
        "dqn",
        object(),
        metrics,
    ) == 4


def test_rollout_episodes_can_override_policy_defaults():
    assert rollout_episodes_for_algorithm(
        "ppo",
        None,
    ) == 64
    assert rollout_episodes_for_algorithm(
        "ppo",
        32,
    ) == 32


def test_rollout_episodes_rejects_non_policy_agents():
    with pytest.raises(
        ValueError,
        match="policy-gradient",
    ):
        rollout_episodes_for_algorithm(
            "dqn",
            32,
        )


def test_validation_schedule_includes_interval_and_final_epoch():
    assert should_validate_epoch(
        epoch=4,
        final_epoch=10,
        validation_interval=2,
    )
    assert should_validate_epoch(
        epoch=10,
        final_epoch=10,
        validation_interval=3,
    )
    assert not should_validate_epoch(
        epoch=5,
        final_epoch=10,
        validation_interval=2,
    )


def test_hierarchical_sampler_deduplicates_tasks_within_dataset_epoch():
    sampler = HierarchicalTaskSampler(
        task_indices=[0, 1, 2, 3],
        layout_indices=[0, 0, 1, 1],
        seed=1,
    )

    specs = sampler.sample_collection(
        collection_size=4,
        target_dataset_epochs=1,
    )
    task_indices = [
        spec.task_index
        for spec in specs
    ]
    layout_indices = [
        [0, 0, 1, 1][task_index]
        for task_index in task_indices
    ]

    assert len(task_indices) == 4
    assert len(set(task_indices)) == 4
    assert len(layout_indices) > len(
        set(layout_indices)
    )
    assert sampler.completed_dataset_epochs == 1


def test_hierarchical_sampler_continues_epoch_without_replacement_across_collections():
    sampler = HierarchicalTaskSampler(
        task_indices=[0, 1, 2, 3],
        layout_indices=[0, 0, 1, 1],
        seed=1,
    )

    first_specs = sampler.sample_collection(
        collection_size=3,
        target_dataset_epochs=2,
    )
    second_specs = sampler.sample_collection(
        collection_size=3,
        target_dataset_epochs=2,
    )

    epoch_one_tasks = [
        spec.task_index
        for spec in first_specs + second_specs
        if spec.dataset_epoch == 1
    ]
    epoch_two_tasks = [
        spec.task_index
        for spec in second_specs
        if spec.dataset_epoch == 2
    ]

    assert len(epoch_one_tasks) == 4
    assert set(epoch_one_tasks) == {
        0,
        1,
        2,
        3,
    }
    assert len(epoch_two_tasks) == 2
