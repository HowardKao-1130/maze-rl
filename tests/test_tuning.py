from __future__ import annotations

import csv
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.tune_dnn import (
    build_evaluation_command,
    build_training_command,
    load_search_space,
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


def test_training_command_includes_trial_hyperparameters(tmp_path):
    args = SimpleNamespace(
        algorithm="ppo",
        dataset_epochs=3,
        max_steps=7,
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
            "rollout_episodes": 64,
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
        command.index("--rollout-episodes") + 1
    ] == "64"


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


def test_load_search_space_rejects_unsupported_parameters(tmp_path):
    path = tmp_path / "space.json"
    path.write_text(
        '{"learning_rate": {"type": "uniform", "low": 0.1, "high": 0.2}, '
        '"rollout_episodes": {"type": "choice", "values": [8]}}'
    )

    with pytest.raises(
        ValueError,
        match="unsupported",
    ):
        load_search_space(
            "dqn",
            path,
        )
