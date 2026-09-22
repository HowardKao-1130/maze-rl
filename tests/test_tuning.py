from __future__ import annotations

import csv
import json
import os
import signal
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.tune_dnn import (
    DEFAULT_SEARCH_SPACES,
    REPO_ROOT,
    _terminate_child_process,
    apply_tuning_config,
    best_config_path_for_algorithm,
    build_evaluation_command,
    build_result_from_artifacts,
    build_training_command,
    collect_heatmap_cells,
    discover_validation_metric_artifacts,
    load_search_space,
    main,
    ordered_search_space,
    parse_args,
    read_completed_results,
    result_matches_trial,
    run_command,
    sample_hyperparameters,
    summarize_tuning,
    summarize_evaluation,
    write_best_config_index,
    write_result_row,
    write_tuning_config,
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


def test_ordered_search_space_restores_default_order_from_sorted_config():
    sorted_space = {
        name: DEFAULT_SEARCH_SPACES["dqn"][name]
        for name in sorted(DEFAULT_SEARCH_SPACES["dqn"])
    }
    default_rng = np.random.default_rng(42)
    sorted_rng = np.random.default_rng(42)
    default_rng.integers(1_000_000_000)
    sorted_rng.integers(1_000_000_000)

    expected = sample_hyperparameters(
        DEFAULT_SEARCH_SPACES["dqn"],
        default_rng,
    )
    actual = sample_hyperparameters(
        ordered_search_space(
            "dqn",
            sorted_space,
        ),
        sorted_rng,
    )

    assert actual == expected


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

    assert args.hyperparameter_combinations == 100
    assert args.rollouts_per_task == 200
    assert args.early_stopping_patience == 35
    assert args.q_snapshot_count == 11
    assert args.tensorboard is False
    assert args.tensorboard_from_cli is False
    assert args.resume is False


def test_parse_args_accepts_tensorboard_opt_in(
    monkeypatch,
):
    monkeypatch.setattr(
        "sys.argv",
        [
            "tune_dnn.py",
            "--algorithm",
            "ppo",
            "--tensorboard",
        ],
    )

    args = parse_args()

    assert args.tensorboard is True
    assert args.tensorboard_from_cli is True


def test_parse_args_accepts_tensorboard_opt_out(
    monkeypatch,
):
    monkeypatch.setattr(
        "sys.argv",
        [
            "tune_dnn.py",
            "--algorithm",
            "ppo",
            "--no-tensorboard",
        ],
    )

    args = parse_args()

    assert args.tensorboard is False
    assert args.tensorboard_from_cli is True


def test_parse_args_accepts_summarize_command(
    monkeypatch,
):
    monkeypatch.setattr(
        "sys.argv",
        [
            "tune_dnn.py",
            "summarize",
            "--output-dir",
            "runs/tuning",
        ],
    )

    args = parse_args()

    assert args.command == "summarize"
    assert args.output_dir.name == "tuning"
    assert args.metric == "mean_path_efficiency"


def test_resume_config_allows_tensorboard_cli_override():
    config = {
        "arguments": {
            "algorithm": "ppo",
            "hyperparameter_combinations": 100,
            "rollouts_per_task": 200,
            "max_steps": 200,
            "seed": 42,
            "train_dataset": "data/train.npz",
            "validation_dataset": "data/validation.npz",
            "same_layout_dataset": (
                "data/same_layout_new_goals.npz"
            ),
            "test_dataset": "data/test.npz",
            "evaluate_best_on_test": False,
            "search_space": None,
            "validation_interval": 1,
            "early_stopping_patience": 35,
            "early_stopping_min_delta": 0.0,
            "tensorboard": True,
            "keep_plots": False,
        },
        "search_space": {},
    }
    args = SimpleNamespace(
        hyperparameter_combinations=20,
        hyperparameter_combinations_from_cli=True,
        tensorboard=False,
        tensorboard_from_cli=True,
    )

    apply_tuning_config(
        args,
        config,
    )

    assert args.hyperparameter_combinations == 20
    assert args.tensorboard is False


def test_resume_config_uses_stored_tensorboard_without_cli_override():
    config = {
        "arguments": {
            "algorithm": "ppo",
            "hyperparameter_combinations": 100,
            "rollouts_per_task": 200,
            "max_steps": 200,
            "seed": 42,
            "train_dataset": "data/train.npz",
            "validation_dataset": "data/validation.npz",
            "same_layout_dataset": (
                "data/same_layout_new_goals.npz"
            ),
            "test_dataset": "data/test.npz",
            "evaluate_best_on_test": False,
            "search_space": None,
            "validation_interval": 1,
            "early_stopping_patience": 35,
            "early_stopping_min_delta": 0.0,
            "tensorboard": True,
            "keep_plots": False,
        },
        "search_space": {},
    }
    args = SimpleNamespace(
        hyperparameter_combinations=20,
        hyperparameter_combinations_from_cli=True,
        tensorboard=False,
        tensorboard_from_cli=False,
    )

    apply_tuning_config(
        args,
        config,
    )

    assert args.hyperparameter_combinations == 20
    assert args.tensorboard is True


def test_parse_args_accepts_resume(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "tune_dnn.py",
            "--algorithm",
            "ppo",
            "--resume",
        ],
    )

    args = parse_args()

    assert args.resume is True


def test_parse_args_accepts_trials_alias(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "tune_dnn.py",
            "--algorithm",
            "ppo",
            "--trials",
            "7",
        ],
    )

    args = parse_args()

    assert args.hyperparameter_combinations == 7


def test_resume_uses_stored_tuning_arguments(
    tmp_path,
    monkeypatch,
    capsys,
    ):
    write_tuning_config(
        tmp_path / "tuning_configs" / "ppo.json",
        SimpleNamespace(
            algorithm="ppo",
            hyperparameter_combinations=1,
            rollouts_per_task=3,
            max_steps=7,
            seed=123,
            train_dataset=tmp_path / "old_train.npz",
            validation_dataset=tmp_path
            / "old_validation.npz",
            same_layout_dataset=tmp_path
            / "old_same_layout.npz",
            test_dataset=tmp_path / "old_test.npz",
            evaluate_best_on_test=False,
            search_space=None,
            validation_interval=2,
            early_stopping_patience=4,
            early_stopping_min_delta=0.01,
            tensorboard=False,
            keep_plots=True,
        ),
        {
            "learning_rate": {
                "type": "choice",
                "values": [0.001],
            },
            "task_batch_size": {
                "type": "choice",
                "values": [64],
            },
            "rollout_group_size": {
                "type": "choice",
                "values": [1],
            },
        },
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "tune_dnn.py",
            "--algorithm",
            "ppo",
            "--resume",
            "--dry-run",
            "--output-dir",
            str(tmp_path),
            "--rollouts-per-task",
            "99",
            "--seed",
            "999",
        ],
    )

    main()

    output = capsys.readouterr().out

    assert "Loaded tuning arguments from" in output
    assert "Hyperparameter combination 1/1" in output
    assert "Hyperparameter combination 2/" not in output
    assert "--algorithm ppo" in output
    assert "--rollouts-per-task 3" in output
    assert "--rollouts-per-task 99" not in output
    assert f"--dataset {tmp_path / 'old_train.npz'}" in output
    assert "--max-steps 7" in output
    assert "--no-tensorboard" in output


def test_resume_allows_hyperparameter_combination_override(
    tmp_path,
    monkeypatch,
    capsys,
):
    write_tuning_config(
        tmp_path / "tuning_configs" / "ppo.json",
        SimpleNamespace(
            algorithm="ppo",
            hyperparameter_combinations=1,
            rollouts_per_task=3,
            max_steps=7,
            seed=123,
            train_dataset=tmp_path / "old_train.npz",
            validation_dataset=tmp_path
            / "old_validation.npz",
            same_layout_dataset=tmp_path
            / "old_same_layout.npz",
            test_dataset=tmp_path / "old_test.npz",
            evaluate_best_on_test=False,
            search_space=None,
            validation_interval=2,
            early_stopping_patience=4,
            early_stopping_min_delta=0.01,
            tensorboard=False,
            keep_plots=True,
        ),
        {
            "learning_rate": {
                "type": "choice",
                "values": [0.001],
            },
            "task_batch_size": {
                "type": "choice",
                "values": [64],
            },
            "rollout_group_size": {
                "type": "choice",
                "values": [1],
            },
        },
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "tune_dnn.py",
            "--algorithm",
            "ppo",
            "--resume",
            "--dry-run",
            "--output-dir",
            str(tmp_path),
            "--hyperparameter-combinations",
            "3",
            "--rollouts-per-task",
            "99",
        ],
    )

    main()

    output = capsys.readouterr().out

    assert "Loaded tuning arguments from" in output
    assert "Hyperparameter combination 1/3" in output
    assert "Hyperparameter combination 3/3" in output
    assert "Hyperparameter combination 4/" not in output
    assert "--rollouts-per-task 3" in output
    assert "--rollouts-per-task 99" not in output


def test_other_algorithm_history_does_not_block_new_sweep(
    tmp_path,
    monkeypatch,
    capsys,
):
    write_tuning_config(
        tmp_path / "tuning_configs" / "reinforce.json",
        SimpleNamespace(
            algorithm="reinforce",
            hyperparameter_combinations=1,
            rollouts_per_task=3,
            max_steps=7,
            seed=123,
            train_dataset=tmp_path / "old_train.npz",
            validation_dataset=tmp_path
            / "old_validation.npz",
            same_layout_dataset=tmp_path
            / "old_same_layout.npz",
            test_dataset=tmp_path / "old_test.npz",
            evaluate_best_on_test=False,
            search_space=None,
            validation_interval=2,
            early_stopping_patience=4,
            early_stopping_min_delta=0.01,
            tensorboard=False,
            keep_plots=True,
        ),
        {
            "learning_rate": {
                "type": "choice",
                "values": [0.001],
            },
            "task_batch_size": {
                "type": "choice",
                "values": [16],
            },
            "rollout_group_size": {
                "type": "choice",
                "values": [1],
            },
        },
    )
    (
        tmp_path
        / "trials"
        / "trial_001"
        / "reinforce"
    ).mkdir(
        parents=True,
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "tune_dnn.py",
            "--algorithm",
            "dqn",
            "--dry-run",
            "--output-dir",
            str(tmp_path),
            "--hyperparameter-combinations",
            "1",
        ],
    )

    main()

    output = capsys.readouterr().out

    assert "Loaded tuning arguments from" not in output
    assert "Hyperparameter combination 1/1" in output
    assert "--algorithm dqn" in output
    assert "--algorithm reinforce" not in output


def test_other_algorithm_legacy_config_does_not_block_new_sweep(
    tmp_path,
    monkeypatch,
    capsys,
):
    write_tuning_config(
        tmp_path / "tuning_config.json",
        SimpleNamespace(
            algorithm="reinforce",
            hyperparameter_combinations=1,
            rollouts_per_task=3,
            max_steps=7,
            seed=123,
            train_dataset=tmp_path / "old_train.npz",
            validation_dataset=tmp_path
            / "old_validation.npz",
            same_layout_dataset=tmp_path
            / "old_same_layout.npz",
            test_dataset=tmp_path / "old_test.npz",
            evaluate_best_on_test=False,
            search_space=None,
            validation_interval=2,
            early_stopping_patience=4,
            early_stopping_min_delta=0.01,
            tensorboard=False,
            keep_plots=True,
        ),
        {
            "learning_rate": {
                "type": "choice",
                "values": [0.001],
            },
            "task_batch_size": {
                "type": "choice",
                "values": [16],
            },
            "rollout_group_size": {
                "type": "choice",
                "values": [1],
            },
        },
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "tune_dnn.py",
            "--algorithm",
            "dqn",
            "--dry-run",
            "--output-dir",
            str(tmp_path),
            "--hyperparameter-combinations",
            "1",
        ],
    )

    main()

    output = capsys.readouterr().out

    assert "Loaded tuning arguments from" not in output
    assert "--algorithm dqn" in output


def test_resume_rejects_history_without_tuning_config(
    tmp_path,
    monkeypatch,
):
    (
        tmp_path / "trials" / "trial_001" / "ppo"
    ).mkdir(
        parents=True,
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "tune_dnn.py",
            "--algorithm",
            "ppo",
            "--resume",
            "--dry-run",
            "--output-dir",
            str(tmp_path),
        ],
    )

    with pytest.raises(
        RuntimeError,
        match="tuning_configs/ppo.json is missing",
    ):
        main()


def test_resume_keeps_completed_result_on_sample_mismatch(
    tmp_path,
    monkeypatch,
    capsys,
):
    checkpoint_path = tmp_path / "checkpoint.pt"
    evaluation_path = tmp_path / "evaluation.csv"
    training_summary_path = (
        tmp_path / "training_summary.json"
    )
    checkpoint_path.write_bytes(b"checkpoint")
    evaluation_path.write_text(
        "episode_return,success,path_efficiency\n"
    )
    training_summary_path.write_text(
        (
            '{"hyperparameters": {"learning_rate": 0.002},'
            '"task_batch_size": 64,'
            '"rollout_group_size": 1,'
            '"best_validation": {'
            '"mean_path_efficiency": 0.5'
            "}}"
        )
    )
    write_tuning_config(
        tmp_path / "tuning_configs" / "ppo.json",
        SimpleNamespace(
            algorithm="ppo",
            hyperparameter_combinations=1,
            rollouts_per_task=3,
            max_steps=7,
            seed=42,
            train_dataset=tmp_path / "train.npz",
            validation_dataset=tmp_path
            / "validation.npz",
            same_layout_dataset=tmp_path
            / "same_layout_new_goals.npz",
            test_dataset=tmp_path / "test.npz",
            evaluate_best_on_test=False,
            search_space=None,
            validation_interval=2,
            early_stopping_patience=4,
            early_stopping_min_delta=0.01,
            tensorboard=False,
            keep_plots=True,
        ),
        {
            "learning_rate": {
                "type": "choice",
                "values": [0.001],
            },
            "task_batch_size": {
                "type": "choice",
                "values": [64],
            },
            "rollout_group_size": {
                "type": "choice",
                "values": [1],
            },
        },
    )
    write_result_row(
        tmp_path / "tuning_results.csv",
        {
            "trial": 1,
            "algorithm": "ppo",
            "seed": 123,
            "objective": 0.5,
            "mean_path_efficiency": 0.5,
            "success_rate": 1.0,
            "average_episode_return": 2.0,
            "average_successful_path_efficiency": 0.5,
            "rollouts_per_task_requested": 3,
            "task_batch_size": 64,
            "rollout_group_size": 1,
            "task_selection_passes_requested": 3,
            "task_selection_passes_completed": 3,
            "dataset_epochs_completed": 3,
            "stopped_early": False,
            "checkpoint_path": str(checkpoint_path),
            "evaluation_path": str(evaluation_path),
            "hyperparameters": {
                "learning_rate": 0.002,
                "task_batch_size": 64,
                "rollout_group_size": 1,
            },
        },
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "tune_dnn.py",
            "--algorithm",
            "ppo",
            "--resume",
            "--dry-run",
            "--output-dir",
            str(tmp_path),
        ],
    )

    main()

    output = capsys.readouterr().out

    assert "keeping the completed recorded result" in output
    assert "Skipping completed hyperparameter combination 1" in output
    assert "scripts/train.py" not in output


def test_best_config_path_is_algorithm_specific(tmp_path):
    assert best_config_path_for_algorithm(
        tmp_path,
        "a2c",
    ) == tmp_path / "best_config_a2c.json"


def test_write_best_config_index_preserves_other_algorithms(tmp_path):
    path = tmp_path / "best_config.json"
    path.write_text(
        json.dumps(
            {
                "algorithm": "dqn",
                "trial": 3,
                "objective": 0.4,
            }
        )
    )

    write_best_config_index(
        path,
        "a2c",
        {
            "algorithm": "a2c",
            "trial": 2,
            "objective": 0.5,
        },
    )

    with path.open() as file:
        configs = json.load(file)

    assert configs["dqn"]["trial"] == 3
    assert configs["a2c"]["trial"] == 2


def test_training_command_includes_trial_hyperparameters(tmp_path):
    args = SimpleNamespace(
        algorithm="ppo",
        rollouts_per_task=3,
        max_steps=7,
        validation_dataset=tmp_path / "validation.npz",
        same_layout_dataset=tmp_path
        / "same_layout_new_goals.npz",
        validation_interval=2,
        early_stopping_patience=4,
        early_stopping_min_delta=0.01,
        q_snapshot_count=7,
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
    assert "--no-progress" in command
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
    assert "--validation-all-tasks" in command
    assert "--no-validation-all-tasks" not in command
    assert "--validation-episodes" not in command
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
    assert command[
        command.index("--q-snapshot-count") + 1
    ] == "7"


def test_training_command_passes_grpo_task_and_group_sizes(tmp_path):
    args = SimpleNamespace(
        algorithm="grpo",
        rollouts_per_task=16,
        max_steps=7,
        validation_dataset=tmp_path / "validation.npz",
        same_layout_dataset=tmp_path
        / "same_layout_new_goals.npz",
        validation_interval=2,
        early_stopping_patience=None,
        early_stopping_min_delta=0.0,
        q_snapshot_count=11,
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
    assert "--no-rollout-animations" in command


def test_evaluation_command_keeps_plots_when_requested(tmp_path):
    args = SimpleNamespace(
        algorithm="ppo",
        max_steps=9,
        keep_plots=True,
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
    assert "--no-plot" not in command
    assert "--no-rollout-animations" in command


def test_run_command_starts_child_in_process_group(
    monkeypatch,
):
    calls = []

    class FakeProcess:
        pid = 12345

        def wait(self):
            return 0

        def poll(self):
            return 0

    def fake_popen(
        command,
        *,
        cwd,
        env,
        start_new_session,
    ):
        calls.append(
            {
                "command": command,
                "cwd": cwd,
                "env": env,
                "start_new_session": start_new_session,
            }
        )
        return FakeProcess()

    monkeypatch.setattr(
        subprocess,
        "Popen",
        fake_popen,
    )

    run_command(
        ["python", "script.py"],
        dry_run=False,
    )

    assert len(calls) == 1
    assert calls[0]["command"] == [
        "python",
        "script.py",
    ]
    assert calls[0]["cwd"] == REPO_ROOT
    assert calls[0]["start_new_session"] is (
        os.name == "posix"
    )
    assert str(REPO_ROOT / "src") in calls[0][
        "env"
    ]["PYTHONPATH"]


def test_run_command_exits_cleanly_after_keyboard_interrupt(
    monkeypatch,
):
    terminated = []

    class FakeProcess:
        pid = 12345

        def wait(self):
            raise KeyboardInterrupt

        def poll(self):
            return None

    def fake_popen(
        command,
        *,
        cwd,
        env,
        start_new_session,
    ):
        return FakeProcess()

    def fake_terminate(
        process,
        *,
        initial_signal=signal.SIGTERM,
        timeout=10.0,
    ):
        terminated.append(
            (
                process.pid,
                initial_signal,
            )
        )

    monkeypatch.setattr(
        subprocess,
        "Popen",
        fake_popen,
    )
    monkeypatch.setattr(
        "scripts.tune_dnn._terminate_child_process",
        fake_terminate,
    )

    with pytest.raises(SystemExit) as exc_info:
        run_command(
            ["python", "script.py"],
            dry_run=False,
        )

    assert exc_info.value.code == 128 + signal.SIGINT
    assert terminated == [
        (
            12345,
            signal.SIGINT,
        )
    ]


@pytest.mark.skipif(
    os.name != "posix",
    reason="process-group termination is POSIX-specific",
)
def test_terminate_child_process_escalates_process_group(
    monkeypatch,
):
    sent_signals = []

    class FakeProcess:
        pid = 12345
        returncode = None

        def poll(self):
            return self.returncode

        def wait(self, timeout=None):
            if timeout is not None:
                raise subprocess.TimeoutExpired(
                    "python script.py",
                    timeout,
                )

            self.returncode = -signal.SIGKILL
            return self.returncode

    def fake_killpg(
        pid,
        signum,
    ):
        sent_signals.append(
            (
                pid,
                signum,
            )
        )

    monkeypatch.setattr(
        os,
        "killpg",
        fake_killpg,
    )

    _terminate_child_process(
        FakeProcess(),
        timeout=0.01,
    )

    assert sent_signals == [
        (
            12345,
            signal.SIGTERM,
        ),
        (
            12345,
            signal.SIGKILL,
        ),
    ]


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


def write_fake_validation_artifacts(
    run_dir,
    *,
    validation_score,
    same_layout_score,
):
    run_dir.mkdir(
        parents=True,
    )
    metrics_path = (
        run_dir / "validation_metrics.csv"
    )
    metrics_path.write_text(
        (
            "split,dataset_epoch,episodes,success_rate,"
            "average_episode_return,mean_path_efficiency,"
            "average_successful_path_efficiency\n"
            "validation,1,2,1.0,2.0,"
            f"{validation_score},"
            f"{validation_score}\n"
            "same_layout,1,2,1.0,2.0,"
            f"{same_layout_score},"
            f"{same_layout_score}\n"
        )
    )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(
        figsize=(1.0, 1.0)
    )
    axis.plot(
        [0, 1],
        [
            0,
            validation_score,
        ],
    )
    axis.axis("off")
    figure.savefig(
        run_dir / "validation_metrics.png"
    )
    plt.close(figure)


def test_summarize_tuning_writes_heatmap_and_trial_grids(tmp_path):
    output_dir = tmp_path / "tuning"
    write_fake_validation_artifacts(
        output_dir
        / "trials"
        / "trial_001"
        / "dqn",
        validation_score=0.25,
        same_layout_score=0.2,
    )
    write_fake_validation_artifacts(
        output_dir
        / "trials"
        / "trial_002"
        / "dqn",
        validation_score=0.5,
        same_layout_score=0.4,
    )
    write_fake_validation_artifacts(
        output_dir
        / "trials"
        / "trial_001"
        / "ppo",
        validation_score=0.75,
        same_layout_score=0.7,
    )
    summary_output_dir = (
        tmp_path / "summary"
    )

    summarize_tuning(
        SimpleNamespace(
            output_dir=output_dir,
            summary_output_dir=summary_output_dir,
            metric="mean_path_efficiency",
        )
    )

    assert (
        summary_output_dir
        / "mean_path_efficiency_heatmap.png"
    ).exists()
    assert (
        summary_output_dir
        / "dqn_validation_metrics_trials.png"
    ).exists()
    assert (
        summary_output_dir
        / "ppo_validation_metrics_trials.png"
    ).exists()

    artifacts = discover_validation_metric_artifacts(
        output_dir
    )
    cells = collect_heatmap_cells(
        artifacts,
        metric="mean_path_efficiency",
    )

    assert {
        (
            cell["algorithm"],
            cell["trial"],
            cell["split"],
            cell["value"],
        )
        for cell in cells
    } == {
        (
            "dqn",
            1,
            "validation",
            0.25,
        ),
        (
            "dqn",
            1,
            "same_layout",
            0.2,
        ),
        (
            "dqn",
            2,
            "validation",
            0.5,
        ),
        (
            "dqn",
            2,
            "same_layout",
            0.4,
        ),
        (
            "ppo",
            1,
            "validation",
            0.75,
        ),
        (
            "ppo",
            1,
            "same_layout",
            0.7,
        ),
    }


def test_read_completed_results_requires_artifacts(tmp_path):
    results_path = tmp_path / "tuning_results.csv"
    run_dir = tmp_path / "trial_001" / "ppo"
    run_dir.mkdir(
        parents=True,
    )
    checkpoint_path = run_dir / "checkpoint.pt"
    evaluation_path = run_dir / "evaluation.csv"
    training_summary_path = (
        run_dir / "training_summary.json"
    )
    checkpoint_path.write_bytes(b"checkpoint")
    evaluation_path.write_text("episode_return,success,path_efficiency\n")
    training_summary_path.write_text(
        (
            '{"hyperparameters": {"learning_rate": 0.001},'
            '"task_batch_size": 4,'
            '"rollout_group_size": 1,'
            '"best_validation": {'
            '"mean_path_efficiency": 0.5'
            "}}"
        )
    )

    write_result_row(
        results_path,
        {
            "trial": 1,
            "algorithm": "ppo",
            "seed": 11,
            "objective": 0.5,
            "mean_path_efficiency": 0.5,
            "success_rate": 1.0,
            "average_episode_return": 2.0,
            "average_successful_path_efficiency": 0.5,
            "rollouts_per_task_requested": 3,
            "task_batch_size": 4,
            "rollout_group_size": 1,
            "task_selection_passes_requested": 3,
            "task_selection_passes_completed": 3,
            "dataset_epochs_completed": 3,
            "stopped_early": False,
            "checkpoint_path": str(checkpoint_path),
            "evaluation_path": str(evaluation_path),
            "hyperparameters": {
                "learning_rate": 0.001,
            },
        },
    )
    write_result_row(
        results_path,
        {
            "trial": 2,
            "algorithm": "ppo",
            "seed": 12,
            "objective": 0.6,
            "mean_path_efficiency": 0.6,
            "success_rate": 1.0,
            "average_episode_return": 2.0,
            "average_successful_path_efficiency": 0.6,
            "rollouts_per_task_requested": 3,
            "task_batch_size": 4,
            "rollout_group_size": 1,
            "task_selection_passes_requested": 3,
            "task_selection_passes_completed": 3,
            "dataset_epochs_completed": 3,
            "stopped_early": False,
            "checkpoint_path": str(
                tmp_path / "missing.pt"
            ),
            "evaluation_path": str(evaluation_path),
            "hyperparameters": {
                "learning_rate": 0.002,
            },
        },
    )

    completed = read_completed_results(
        results_path,
        "ppo",
    )

    assert list(completed) == [1]
    assert completed[1]["objective"] == 0.5
    assert completed[1]["hyperparameters"] == {
        "learning_rate": 0.001,
    }


def test_read_completed_results_rejects_mismatched_artifacts(tmp_path):
    results_path = tmp_path / "tuning_results.csv"
    run_dir = tmp_path / "trial_001" / "ppo"
    run_dir.mkdir(
        parents=True,
    )
    checkpoint_path = run_dir / "checkpoint.pt"
    evaluation_path = run_dir / "evaluation.csv"
    training_summary_path = (
        run_dir / "training_summary.json"
    )
    checkpoint_path.write_bytes(b"checkpoint")
    evaluation_path.write_text("episode_return,success,path_efficiency\n")
    training_summary_path.write_text(
        (
            '{"hyperparameters": {"learning_rate": 0.002},'
            '"task_batch_size": 4,'
            '"rollout_group_size": 1,'
            '"best_validation": {'
            '"mean_path_efficiency": 0.5'
            "}}"
        )
    )
    write_result_row(
        results_path,
        {
            "trial": 1,
            "algorithm": "ppo",
            "seed": 11,
            "objective": 0.5,
            "mean_path_efficiency": 0.5,
            "success_rate": 1.0,
            "average_episode_return": 2.0,
            "average_successful_path_efficiency": 0.5,
            "rollouts_per_task_requested": 3,
            "task_batch_size": 4,
            "rollout_group_size": 1,
            "task_selection_passes_requested": 3,
            "task_selection_passes_completed": 3,
            "dataset_epochs_completed": 3,
            "stopped_early": False,
            "checkpoint_path": str(checkpoint_path),
            "evaluation_path": str(evaluation_path),
            "hyperparameters": {
                "learning_rate": 0.001,
            },
        },
    )

    completed = read_completed_results(
        results_path,
        "ppo",
    )

    assert completed == {}


def test_read_completed_results_rejects_stale_completion_marker(tmp_path):
    results_path = tmp_path / "tuning_results.csv"
    run_dir = tmp_path / "trial_001" / "dqn"
    tensorboard_dir = run_dir / "tensorboard"
    tensorboard_dir.mkdir(
        parents=True,
    )
    checkpoint_path = run_dir / "checkpoint.pt"
    evaluation_path = (
        run_dir / "validation_evaluation.csv"
    )
    training_summary_path = (
        run_dir / "training_summary.json"
    )
    metrics_path = run_dir / "metrics.csv"
    tensorboard_path = (
        tensorboard_dir / "events.out.tfevents.test"
    )
    checkpoint_path.write_bytes(b"checkpoint")
    evaluation_path.write_text("episode_return,success,path_efficiency\n")
    training_summary_path.write_text(
        (
            '{"hyperparameters": {"learning_rate": 0.001},'
            '"task_batch_size": 1,'
            '"rollout_group_size": 1,'
            '"best_validation": {'
            '"mean_path_efficiency": 0.5'
            "}}"
        )
    )
    metrics_path.write_text("episode_return\n")
    tensorboard_path.write_text("event")
    os.utime(
        checkpoint_path,
        ns=(1_000, 1_000),
    )
    os.utime(
        training_summary_path,
        ns=(2_000, 2_000),
    )
    os.utime(
        evaluation_path,
        ns=(3_000, 3_000),
    )
    os.utime(
        metrics_path,
        ns=(4_000, 4_000),
    )
    os.utime(
        tensorboard_path,
        ns=(4_000, 4_000),
    )
    write_result_row(
        results_path,
        {
            "trial": 1,
            "algorithm": "dqn",
            "seed": 11,
            "objective": 0.5,
            "mean_path_efficiency": 0.5,
            "success_rate": 1.0,
            "average_episode_return": 2.0,
            "average_successful_path_efficiency": 0.5,
            "rollouts_per_task_requested": 3,
            "task_batch_size": 1,
            "rollout_group_size": 1,
            "task_selection_passes_requested": 3,
            "task_selection_passes_completed": 3,
            "dataset_epochs_completed": 3,
            "stopped_early": False,
            "checkpoint_path": str(checkpoint_path),
            "evaluation_path": str(evaluation_path),
            "hyperparameters": {
                "learning_rate": 0.001,
                "task_batch_size": 1,
                "rollout_group_size": 1,
            },
        },
    )

    completed = read_completed_results(
        results_path,
        "dqn",
    )

    assert completed == {}


def test_build_result_from_artifacts_recovers_trial(tmp_path):
    run_dir = tmp_path / "trial_001" / "ppo"
    run_dir.mkdir(
        parents=True,
    )
    checkpoint_path = run_dir / "checkpoint.pt"
    best_checkpoint_path = run_dir / "best_checkpoint.pt"
    training_summary_path = (
        run_dir / "training_summary.json"
    )
    evaluation_path = (
        run_dir / "validation_evaluation.csv"
    )
    checkpoint_path.write_bytes(b"checkpoint")
    best_checkpoint_path.write_bytes(
        b"best checkpoint"
    )
    training_summary_path.write_text(
        (
            '{"algorithm": "ppo",'
            '"rollouts_per_task_requested": 3,'
            '"task_batch_size": 4,'
            '"rollout_group_size": 1,'
            '"task_selection_passes_requested": 3,'
            '"task_selection_passes_completed": 2,'
            '"dataset_epochs_completed": 2,'
            '"stopped_early": true,'
            '"hyperparameters": {"learning_rate": 0.001},'
            '"best_validation": {'
            '"mean_path_efficiency": 0.75,'
            '"success_rate": 1.0,'
            '"average_episode_return": 2.0,'
            '"average_successful_path_efficiency": 0.75'
            "}}"
        )
    )
    evaluation_path.write_text(
        (
            "episode_return,success,path_efficiency\n"
            "2.0,True,0.75\n"
        )
    )
    args = SimpleNamespace(
        algorithm="ppo",
    )

    result = build_result_from_artifacts(
        args=args,
        trial=1,
        trial_seed=11,
        checkpoint_path=checkpoint_path,
        best_checkpoint_path=best_checkpoint_path,
        training_summary_path=training_summary_path,
        evaluation_path=evaluation_path,
    )

    assert result is not None
    assert result["objective"] == 0.75
    assert result["seed"] == 11
    assert result["checkpoint_path"] == str(
        best_checkpoint_path
    )
    assert result["hyperparameters"] == {
        "learning_rate": 0.001,
        "task_batch_size": 4,
        "rollout_group_size": 1,
    }


def test_build_result_from_artifacts_treats_truncated_summary_as_incomplete(
    tmp_path,
):
    run_dir = tmp_path / "trial_001" / "ppo"
    run_dir.mkdir(
        parents=True,
    )
    checkpoint_path = run_dir / "checkpoint.pt"
    best_checkpoint_path = run_dir / "best_checkpoint.pt"
    training_summary_path = (
        run_dir / "training_summary.json"
    )
    evaluation_path = (
        run_dir / "validation_evaluation.csv"
    )
    checkpoint_path.write_bytes(b"checkpoint")
    best_checkpoint_path.write_bytes(
        b"best checkpoint"
    )
    training_summary_path.write_text(
        '{"algorithm": "ppo",'
    )
    evaluation_path.write_text(
        (
            "episode_return,success,path_efficiency\n"
            "2.0,True,0.75\n"
        )
    )
    args = SimpleNamespace(
        algorithm="ppo",
    )

    result = build_result_from_artifacts(
        args=args,
        trial=1,
        trial_seed=11,
        checkpoint_path=checkpoint_path,
        best_checkpoint_path=best_checkpoint_path,
        training_summary_path=training_summary_path,
        evaluation_path=evaluation_path,
    )

    assert result is None


def test_build_result_from_artifacts_treats_empty_evaluation_as_incomplete(
    tmp_path,
):
    run_dir = tmp_path / "trial_001" / "ppo"
    run_dir.mkdir(
        parents=True,
    )
    checkpoint_path = run_dir / "checkpoint.pt"
    best_checkpoint_path = run_dir / "best_checkpoint.pt"
    training_summary_path = (
        run_dir / "training_summary.json"
    )
    evaluation_path = (
        run_dir / "validation_evaluation.csv"
    )
    checkpoint_path.write_bytes(b"checkpoint")
    best_checkpoint_path.write_bytes(
        b"best checkpoint"
    )
    training_summary_path.write_text(
        (
            '{"algorithm": "ppo",'
            '"task_batch_size": 4,'
            '"rollout_group_size": 1,'
            '"dataset_epochs_completed": 2,'
            '"best_validation": {'
            '"mean_path_efficiency": 0.75,'
            '"success_rate": 1.0,'
            '"average_episode_return": 2.0,'
            '"average_successful_path_efficiency": 0.75'
            "}}"
        )
    )
    evaluation_path.write_text(
        "episode_return,success,path_efficiency\n"
    )
    args = SimpleNamespace(
        algorithm="ppo",
    )

    result = build_result_from_artifacts(
        args=args,
        trial=1,
        trial_seed=11,
        checkpoint_path=checkpoint_path,
        best_checkpoint_path=best_checkpoint_path,
        training_summary_path=training_summary_path,
        evaluation_path=evaluation_path,
    )

    assert result is None


@pytest.mark.parametrize(
    "missing_field",
    [
        "dataset_epochs_completed",
        "stopped_early",
    ],
)
def test_build_result_from_artifacts_treats_partial_summary_as_incomplete(
    tmp_path,
    missing_field,
):
    run_dir = tmp_path / "trial_001" / "ppo"
    run_dir.mkdir(
        parents=True,
    )
    checkpoint_path = run_dir / "checkpoint.pt"
    best_checkpoint_path = run_dir / "best_checkpoint.pt"
    training_summary_path = (
        run_dir / "training_summary.json"
    )
    evaluation_path = (
        run_dir / "validation_evaluation.csv"
    )
    checkpoint_path.write_bytes(b"checkpoint")
    best_checkpoint_path.write_bytes(
        b"best checkpoint"
    )
    training_summary = {
        "algorithm": "ppo",
        "task_batch_size": 4,
        "rollout_group_size": 1,
        "dataset_epochs_completed": 2,
        "stopped_early": False,
        "best_validation": {
            "mean_path_efficiency": 0.75,
            "success_rate": 1.0,
            "average_episode_return": 2.0,
            "average_successful_path_efficiency": 0.75,
        },
    }
    training_summary.pop(
        missing_field,
    )
    training_summary_path.write_text(
        json.dumps(
            training_summary,
        )
    )
    evaluation_path.write_text(
        (
            "episode_return,success,path_efficiency\n"
            "2.0,True,0.75\n"
        )
    )
    args = SimpleNamespace(
        algorithm="ppo",
    )

    result = build_result_from_artifacts(
        args=args,
        trial=1,
        trial_seed=11,
        checkpoint_path=checkpoint_path,
        best_checkpoint_path=best_checkpoint_path,
        training_summary_path=training_summary_path,
        evaluation_path=evaluation_path,
    )

    assert result is None


def test_build_result_from_artifacts_rejects_stale_completion_marker(tmp_path):
    run_dir = tmp_path / "trial_001" / "dqn"
    run_dir.mkdir(
        parents=True,
    )
    checkpoint_path = run_dir / "checkpoint.pt"
    best_checkpoint_path = run_dir / "best_checkpoint.pt"
    training_summary_path = (
        run_dir / "training_summary.json"
    )
    evaluation_path = (
        run_dir / "validation_evaluation.csv"
    )
    checkpoint_path.write_bytes(b"checkpoint")
    best_checkpoint_path.write_bytes(
        b"best checkpoint"
    )
    training_summary_path.write_text(
        (
            '{"algorithm": "dqn",'
            '"task_batch_size": 1,'
            '"rollout_group_size": 1,'
            '"dataset_epochs_completed": 2,'
            '"stopped_early": true,'
            '"hyperparameters": {"learning_rate": 0.001},'
            '"best_validation": {'
            '"mean_path_efficiency": 0.75,'
            '"success_rate": 1.0,'
            '"average_episode_return": 2.0,'
            '"average_successful_path_efficiency": 0.75'
            "}}"
        )
    )
    evaluation_path.write_text(
        (
            "episode_return,success,path_efficiency\n"
            "2.0,True,0.75\n"
        )
    )
    os.utime(
        training_summary_path,
        ns=(2_000, 2_000),
    )
    os.utime(
        evaluation_path,
        ns=(3_000, 3_000),
    )
    os.utime(
        checkpoint_path,
        ns=(4_000, 4_000),
    )
    args = SimpleNamespace(
        algorithm="dqn",
    )

    result = build_result_from_artifacts(
        args=args,
        trial=1,
        trial_seed=11,
        checkpoint_path=checkpoint_path,
        best_checkpoint_path=best_checkpoint_path,
        training_summary_path=training_summary_path,
        evaluation_path=evaluation_path,
    )

    assert result is None


def test_result_matches_trial_uses_command_precision():
    assert result_matches_trial(
        {
            "seed": 11,
            "hyperparameters": {
                "learning_rate": 0.000123456789123,
                "task_batch_size": 4,
            },
        },
        trial_seed=11,
        hyperparameters={
            "learning_rate": 0.0001234567891234,
            "task_batch_size": 4,
        },
    )

    assert not result_matches_trial(
        {
            "seed": 12,
            "hyperparameters": {
                "learning_rate": 0.000123456789123,
                "task_batch_size": 4,
            },
        },
        trial_seed=11,
        hyperparameters={
            "learning_rate": 0.0001234567891234,
            "task_batch_size": 4,
        },
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
