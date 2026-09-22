from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import torch

from maze_rl.training.metrics import EpisodeMetrics
from maze_rl.training.plots import plot_validation_metrics
from scripts.generate_dataset import (
    main as generate_dataset_main,
    parse_args as parse_dataset_args,
)
from scripts.evaluate import default_checkpoint_path
from scripts.evaluate import (
    best_trial_checkpoint_path,
    default_rollout_animation_output_dir,
    default_tuning_results_path,
    evaluation_episode_plan,
    evaluation_csv_rows,
    infer_dataset_label,
    parse_args as parse_evaluate_args,
    same_layout_dataset_for,
)
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
    format_validation_progress_message,
    log_tensorboard_episode,
    log_tensorboard_epoch,
    log_tensorboard_optimization_epoch,
    log_tensorboard_training_round,
    neural_hyperparameters_from_args,
    optimization_epochs_for_round,
    parse_args as parse_train_args,
    rollout_group_size_for_algorithm,
    should_validate_epoch,
    should_stop_early,
    task_batch_size_for_algorithm,
    task_selection_passes_for_budget,
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


def test_evaluate_rollout_animation_defaults(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "evaluate.py",
            "--algorithm",
            "q_learning",
            "--dataset",
            "data/validation.npz",
        ],
    )

    args = parse_evaluate_args()

    assert args.checkpoint == Path(
        "runs/q_learning/checkpoint.pkl"
    )
    assert args.dataset_label == "val"
    assert not args.no_rollout_animations
    assert args.rollout_animation_fps == 8.0
    assert args.rollout_animation_output_dir is None
    assert default_rollout_animation_output_dir(
        args.checkpoint
    ) == Path(
        "runs/q_learning/rollout_animations"
    )


def test_evaluate_rollout_animation_cli_overrides(monkeypatch, tmp_path):
    output_dir = tmp_path / "animations"

    monkeypatch.setattr(
        "sys.argv",
        [
            "evaluate.py",
            "--algorithm",
            "dqn",
            "--dataset",
            "data/same_layout_new_goals.npz",
            "--dataset-label",
            "heldout_goals",
            "--rollout-animation-output-dir",
            str(output_dir),
            "--rollout-animation-fps",
            "12",
            "--no-rollout-animations",
        ],
    )

    args = parse_evaluate_args()

    assert args.dataset_label == "heldout_goals"
    assert args.rollout_animation_output_dir == output_dir
    assert args.rollout_animation_fps == 12.0
    assert args.no_rollout_animations


def test_evaluate_accepts_legacy_sampling_flags(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "evaluate.py",
            "--algorithm",
            "a2c",
            "--episodes",
            "7",
            "--fixed-index",
            "2",
        ],
    )

    args = parse_evaluate_args()

    assert args.episodes == 7
    assert args.fixed_index == 2
    assert not args.all_tasks


def test_evaluate_accepts_legacy_all_tasks_flag(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "evaluate.py",
            "--algorithm",
            "a2c",
            "--all-tasks",
        ],
    )

    args = parse_evaluate_args()

    assert args.all_tasks
    assert args.episodes is None
    assert args.fixed_index is None


def test_evaluate_rejects_all_tasks_with_fixed_index(
    monkeypatch,
):
    monkeypatch.setattr(
        "sys.argv",
        [
            "evaluate.py",
            "--algorithm",
            "a2c",
            "--all-tasks",
            "--fixed-index",
            "1",
        ],
    )

    with pytest.raises(SystemExit):
        parse_evaluate_args()


def test_evaluation_episode_plan_defaults_to_every_task_once():
    env = SimpleNamespace(num_tasks=4)

    episode_count, task_indices = evaluation_episode_plan(
        env,
        episodes=None,
        fixed_index=None,
        all_tasks=False,
    )

    assert episode_count == 4
    assert task_indices == [0, 1, 2, 3]


def test_evaluation_episode_plan_supports_legacy_sampling_modes():
    env = SimpleNamespace(num_tasks=4)

    assert evaluation_episode_plan(
        env,
        episodes=3,
        fixed_index=None,
        all_tasks=False,
    ) == (3, None)
    assert evaluation_episode_plan(
        env,
        episodes=3,
        fixed_index=2,
        all_tasks=False,
    ) == (3, [2, 2, 2])
    assert evaluation_episode_plan(
        env,
        episodes=None,
        fixed_index=2,
        all_tasks=False,
    ) == (200, [2] * 200)


def test_evaluate_rejects_nonpositive_episodes(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "evaluate.py",
            "--algorithm",
            "a2c",
            "--episodes",
            "0",
        ],
    )

    with pytest.raises(SystemExit):
        parse_evaluate_args()


def test_best_trial_checkpoint_path_selects_algorithm_best(tmp_path):
    tuning_results_path = (
        tmp_path / "tuning_results.csv"
    )
    tuning_results_path.write_text(
        "trial,algorithm,objective,checkpoint_path\n"
        "1,a2c,0.3,runs/tuning/trials/trial_001/a2c/best_checkpoint.pt\n"
        "2,dqn,0.8,runs/tuning/trials/trial_002/dqn/best_checkpoint.pt\n"
        "3,a2c,0.5,runs/tuning/trials/trial_003/a2c/best_checkpoint.pt\n"
    )

    assert best_trial_checkpoint_path(
        "a2c",
        tuning_results_path,
    ) == Path(
        "runs/tuning/trials/trial_003/a2c/best_checkpoint.pt"
    )


def test_best_trial_checkpoint_path_rejects_missing_algorithm(tmp_path):
    tuning_results_path = (
        tmp_path / "tuning_results.csv"
    )
    tuning_results_path.write_text(
        "trial,algorithm,objective,checkpoint_path\n"
        "1,dqn,0.8,runs/tuning/trials/trial_001/dqn/best_checkpoint.pt\n"
    )

    with pytest.raises(
        ValueError,
        match="No tuning_results.csv rows",
    ):
        best_trial_checkpoint_path(
            "a2c",
            tuning_results_path,
        )


def test_evaluate_best_trial_uses_tuning_results_csv(
    monkeypatch,
    tmp_path,
):
    tuning_results_path = (
        tmp_path / "tuning_results.csv"
    )
    tuning_results_path.write_text(
        "trial,algorithm,objective,checkpoint_path\n"
        "1,a2c,0.2,runs/tuning/trials/trial_001/a2c/best_checkpoint.pt\n"
        "2,a2c,0.7,runs/tuning/trials/trial_002/a2c/best_checkpoint.pt\n"
    )

    monkeypatch.setattr(
        "sys.argv",
        [
            "evaluate.py",
            "--algorithm",
            "a2c",
            "--best-trial",
            "--tuning-results",
            str(tuning_results_path),
        ],
    )

    args = parse_evaluate_args()

    assert args.best_trial
    assert args.tuning_results == tuning_results_path
    assert args.checkpoint == Path(
        "runs/tuning/trials/trial_002/a2c/best_checkpoint.pt"
    )


def test_evaluate_best_trial_rejects_explicit_checkpoint(
    monkeypatch,
):
    monkeypatch.setattr(
        "sys.argv",
        [
            "evaluate.py",
            "--algorithm",
            "a2c",
            "--best-trial",
            "--checkpoint",
            "runs/a2c/checkpoint.pt",
        ],
    )

    with pytest.raises(SystemExit):
        parse_evaluate_args()


def test_default_tuning_results_path_uses_tuning_run_csv():
    assert default_tuning_results_path() == Path(
        "runs/tuning/tuning_results.csv"
    )


def test_infer_dataset_label_recognizes_validation_splits():
    assert infer_dataset_label(
        Path("data/validation.npz")
    ) == "val"
    assert infer_dataset_label(
        Path("data/same_layout_new_goals.npz")
    ) == "val_with_same_layout"
    assert infer_dataset_label(
        Path("data/test.npz")
    ) == "test"


def test_same_layout_dataset_for_validation_sibling(tmp_path):
    validation_path = tmp_path / "validation.npz"
    same_layout_path = (
        tmp_path / "same_layout_new_goals.npz"
    )
    validation_path.write_bytes(b"")
    same_layout_path.write_bytes(b"")

    assert same_layout_dataset_for(
        validation_path
    ) == same_layout_path


def test_same_layout_dataset_for_non_validation_dataset(tmp_path):
    test_path = tmp_path / "test.npz"
    test_path.write_bytes(b"")

    assert same_layout_dataset_for(test_path) is None


def test_evaluation_csv_rows_strip_rollout_trace():
    rows = evaluation_csv_rows(
        [
            {
                "episode": 1,
                "task_index": 2,
                "rollout_trace": {
                    "positions": [
                        [0, 0],
                    ],
                },
            }
        ]
    )

    assert rows == [
        {
            "episode": 1,
            "task_index": 2,
        }
    ]


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


def test_training_progress_messages_enabled_by_default(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "train.py",
            "--algorithm",
            "dqn",
        ],
    )

    args = parse_train_args()

    assert args.no_progress is False


def test_training_accepts_no_progress(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "train.py",
            "--algorithm",
            "dqn",
            "--no-progress",
        ],
    )

    args = parse_train_args()

    assert args.no_progress is True


def test_validation_progress_formats_best_in_parentheses():
    assert (
        format_validation_progress_message(
            "validation",
            dataset_epoch=103,
            task_selection_pass_budget=200,
            validation_score=0.419,
            best_score=0.439,
        )
        == "validation task_selection_pass=  103/200 "
        "(52%) mean_path_efficiency=0.419 (best=0.439)"
    )


def test_same_layout_dataset_matches_validation_size(monkeypatch, tmp_path):
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
            "1",
            "--height",
            "5",
            "--width",
            "5",
            "--min-path-length",
            "2",
            "--output-dir",
            str(tmp_path),
        ],
    )

    generate_dataset_main()

    with np.load(
        tmp_path / "validation.npz"
    ) as validation_data, np.load(
        tmp_path / "same_layout_new_goals.npz"
    ) as same_layout_data:
        assert validation_data[
            "layouts"
        ].shape == same_layout_data[
            "layouts"
        ].shape
        assert len(
            validation_data["starts"]
        ) == len(
            same_layout_data["starts"]
        )


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
        train_frequency=None,
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


def test_dqn_accepts_train_frequency_hyperparameter():
    validate_neural_hyperparameters(
        "dqn",
        {
            "train_frequency": 64,
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


def test_rollouts_per_task_converts_to_task_selection_passes():
    assert rollout_group_size_for_algorithm(
        "grpo",
        None,
    ) == 8
    assert rollout_group_size_for_algorithm(
        "ppo",
        None,
    ) == 1
    assert task_selection_passes_for_budget(
        rollouts_per_task=100,
        rollout_group_size=8,
    ) == 13
    assert task_selection_passes_for_budget(
        rollouts_per_task=100,
        rollout_group_size=1,
    ) == 100


def test_rollout_group_size_rejects_non_positive_values():
    with pytest.raises(
        ValueError,
        match="positive",
    ):
        rollout_group_size_for_algorithm(
            "ppo",
            0,
        )


def test_task_batch_size_can_override_dnn_defaults():
    assert task_batch_size_for_algorithm(
        "ppo",
        None,
    ) == 64
    assert task_batch_size_for_algorithm(
        "grpo",
        None,
    ) == 16
    assert task_batch_size_for_algorithm(
        "dqn",
        None,
    ) == 1
    assert task_batch_size_for_algorithm(
        "grpo",
        32,
    ) == 32


def test_task_batch_size_rejects_non_positive_values():
    with pytest.raises(
        ValueError,
        match="positive",
    ):
        task_batch_size_for_algorithm(
            "dqn",
            0,
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


def test_early_stopping_ignores_all_zero_startup_plateau():
    assert not should_stop_early(
        patience=4,
        checks_without_improvement=4,
        has_positive_validation_score=False,
    )
    assert not should_stop_early(
        patience=4,
        checks_without_improvement=5,
        has_positive_validation_score=False,
    )


def test_early_stopping_still_applies_after_positive_validation():
    assert should_stop_early(
        patience=4,
        checks_without_improvement=4,
        has_positive_validation_score=True,
    )
    assert not should_stop_early(
        patience=4,
        checks_without_improvement=3,
        has_positive_validation_score=True,
    )
    assert not should_stop_early(
        patience=None,
        checks_without_improvement=4,
        has_positive_validation_score=True,
    )


def test_plot_validation_metrics_writes_split_curve(tmp_path):
    metrics_path = tmp_path / "validation_metrics.csv"
    metrics_path.write_text(
        "split,dataset_epoch,episodes,success_rate,"
        "average_episode_return,mean_path_efficiency,"
        "average_successful_path_efficiency\n"
        "validation,1,2,0.5,1.0,0.4,0.8\n"
        "same_layout,1,2,1.0,2.0,0.9,0.9\n"
    )
    output_path = tmp_path / "validation_metrics.png"

    plot_validation_metrics(
        metrics_path=metrics_path,
        output_path=output_path,
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0


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
