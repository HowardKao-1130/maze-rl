from __future__ import annotations

import csv
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.tune_dnn import (
    build_evaluation_command,
    build_training_command,
    load_search_space,
    parse_args,
    sample_hyperparameters,
    summarize_evaluation,
)


def test_sample_hyperparameters_uses_space_distributions():
    rng = np.random.default_rng(123)

    sample = sample_hyperparameters(
        {
            "learning_rate": {
                "type": "loguniform",
                "low": 1e-5,
                "high": 1e-3,
            },
            "batch_size": {
                "type": "choice",
                "values": [32, 64],
            },
            "update_epochs": {
                "type": "int",
                "low": 2,
                "high": 4,
            },
        },
        rng,
    )

    assert 1e-5 <= sample[
        "learning_rate"
    ] <= 1e-3
    assert sample["batch_size"] in {
        32,
        64,
    }
    assert 2 <= sample[
        "update_epochs"
    ] <= 4


def test_parse_args_uses_tuning_run_defaults(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "tune_dnn.py",
            "--algorithm",
            "ppo",
        ],
    )

    args = parse_args()

    assert args.trials == 60
    assert args.rollouts_per_task == 200
    assert args.eval_all_tasks is True
    assert args.eval_episodes == 200
    assert args.early_stopping_patience == 35
    assert args.tensorboard is True


def test_training_command_includes_trial_hyperparameters(tmp_path):
    args = SimpleNamespace(
        algorithm="ppo",
        rollouts_per_task=3,
        max_steps=7,
        validation_dataset=tmp_path / "validation.npz",
        same_layout_dataset=tmp_path
        / "same_layout_new_goals.npz",
        validation_interval=2,
        eval_all_tasks=False,
        eval_episodes=5,
        early_stopping_patience=4,
        early_stopping_min_delta=0.01,
        tensorboard=False,
        keep_plots=False,
    )

    command = build_training_command(
        args=args,
        trial_dir=tmp_path / "trial",
        train_dataset=tmp_path / "train.npz",
        trial_seed=11,
        hyperparameters={
            "learning_rate": 0.0003,
            "minibatch_size": 32,
            "task_batch_size": 64,
            "rollout_group_size": 1,
        },
    )

    assert "--no-tensorboard" in command
    assert "--no-plot" in command
    assert command[
        command.index("--minibatch-size") + 1
    ] == "32"
    assert command[
        command.index("--learning-rate") + 1
    ] == "0.0003"
    assert command[
        command.index("--validation-dataset") + 1
    ] == str(tmp_path / "validation.npz")
    assert command[
        command.index("--same-layout-dataset") + 1
    ] == str(
        tmp_path / "same_layout_new_goals.npz"
    )
    assert command[
        command.index("--validation-interval") + 1
    ] == "2"
    assert "--no-validation-all-tasks" in command
    assert command[
        command.index("--validation-episodes") + 1
    ] == "5"
    assert command[
        command.index("--early-stopping-patience") + 1
    ] == "4"
    assert command[
        command.index("--early-stopping-min-delta") + 1
    ] == "0.01"
    assert command[
        command.index("--rollouts-per-task") + 1
    ] == "3"
    assert command[
        command.index("--task-batch-size") + 1
    ] == "64"
    assert command[
        command.index("--rollout-group-size") + 1
    ] == "1"


def test_training_command_passes_grpo_task_and_group_sizes(tmp_path):
    args = SimpleNamespace(
        algorithm="grpo",
        rollouts_per_task=16,
        max_steps=7,
        validation_dataset=tmp_path / "validation.npz",
        same_layout_dataset=tmp_path
        / "same_layout_new_goals.npz",
        validation_interval=2,
        eval_all_tasks=True,
        eval_episodes=5,
        early_stopping_patience=None,
        early_stopping_min_delta=0.0,
        tensorboard=False,
        keep_plots=False,
    )

    command = build_training_command(
        args=args,
        trial_dir=tmp_path / "trial",
        train_dataset=tmp_path / "train.npz",
        trial_seed=11,
        hyperparameters={
            "learning_rate": 0.0003,
            "task_batch_size": 16,
            "rollout_group_size": 8,
        },
    )

    assert command[
        command.index("--learning-rate") + 1
    ] == "0.0003"
    assert command[
        command.index("--task-batch-size") + 1
    ] == "16"
    assert command[
        command.index("--rollout-group-size") + 1
    ] == "8"


def test_evaluation_command_defaults_to_all_validation_tasks(tmp_path):
    args = SimpleNamespace(
        algorithm="ppo",
        max_steps=9,
        eval_all_tasks=True,
        eval_episodes=5,
        keep_plots=False,
    )

    command = build_evaluation_command(
        args=args,
        checkpoint_path=tmp_path / "checkpoint.pt",
        validation_dataset=tmp_path / "validation.npz",
        evaluation_path=tmp_path / "evaluation.csv",
        trial_seed=17,
    )

    assert "--all-tasks" in command
    assert "--episodes" not in command
    assert "--evaluation-output" in command
    assert "--no-plot" in command


def test_summarize_evaluation_uses_mean_path_efficiency(tmp_path):
    path = tmp_path / "evaluation.csv"

    with path.open(
        "w",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "episode_return",
                "success",
                "path_efficiency",
            ],
        )
        writer.writeheader()
        writer.writerows(
            [
                {
                    "episode_return": "1.0",
                    "success": "True",
                    "path_efficiency": "1.0",
                },
                {
                    "episode_return": "0.0",
                    "success": "False",
                    "path_efficiency": "0.0",
                },
            ]
        )

    summary = summarize_evaluation(path)

    assert summary["mean_path_efficiency"] == 0.5
    assert (
        summary[
            "average_successful_path_efficiency"
        ]
        == 1.0
    )


def test_load_search_space_allows_training_loop_parameters(tmp_path):
    path = tmp_path / "space.json"
    path.write_text(
        '{"learning_rate": {"type": "uniform", "low": 0.1, "high": 0.2}, '
        '"task_batch_size": {"type": "choice", "values": [8]}, '
        '"rollout_group_size": {"type": "choice", "values": [1]}}'
    )

    search_space = load_search_space(
        "dqn",
        path,
    )

    assert "task_batch_size" in search_space
    assert "rollout_group_size" in search_space


def test_load_search_space_allows_dqn_train_frequency(tmp_path):
    path = tmp_path / "space.json"
    path.write_text(
        '{"learning_rate": {"type": "uniform", "low": 0.1, "high": 0.2}, '
        '"train_frequency": {"type": "choice", "values": [64]}}'
    )

    search_space = load_search_space(
        "dqn",
        path,
    )

    assert "train_frequency" in search_space


def test_load_search_space_normalizes_rollout_episodes_alias(tmp_path):
    path = tmp_path / "space.json"
    path.write_text(
        '{"learning_rate": {"type": "uniform", "low": 0.1, "high": 0.2}, '
        '"rollout_episodes": {"type": "choice", "values": [64]}}'
    )

    search_space = load_search_space(
        "ppo",
        path,
    )

    assert "task_batch_size" in search_space
    assert "rollout_episodes" not in search_space


def test_load_search_space_normalizes_group_size_alias(tmp_path):
    path = tmp_path / "space.json"
    path.write_text(
        '{"learning_rate": {"type": "uniform", "low": 0.1, "high": 0.2}, '
        '"group_size": {"type": "choice", "values": [8]}}'
    )

    search_space = load_search_space(
        "ppo",
        path,
    )

    assert "rollout_group_size" in search_space
    assert "group_size" not in search_space


def test_load_search_space_allows_grpo_rollout_group_size(tmp_path):
    path = tmp_path / "space.json"
    path.write_text(
        '{"learning_rate": {"type": "uniform", "low": 0.1, "high": 0.2}, '
        '"rollout_group_size": {"type": "choice", "values": [8]}}'
    )

    search_space = load_search_space(
        "grpo",
        path,
    )

    assert "rollout_group_size" in search_space


def test_load_search_space_allows_grpo_task_batch_size(tmp_path):
    path = tmp_path / "space.json"
    path.write_text(
        '{"learning_rate": {"type": "uniform", "low": 0.1, "high": 0.2}, '
        '"task_batch_size": {"type": "choice", "values": [16]}}'
    )

    search_space = load_search_space(
        "grpo",
        path,
    )

    assert "task_batch_size" in search_space
