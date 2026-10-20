from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
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
    dataset_bundle_dir_for_reference,
    dataset_bundle_tuning_output_dir,
    default_experiment_tuning_output_dir,
    load_search_space,
    main,
    ordered_search_space,
    parse_args,
    read_completed_results,
    result_matches_trial,
    resolve_dataset_bundle_args,
    run_command,
    sample_hyperparameters,
    sample_hyperparameters_with_optuna,
    summarize_evaluation,
    write_best_config_index,
    write_result_row,
    write_tuning_config,
)
from scripts.summarize_tuning_runs import (
    aligned_training_metric_series,
    artifact_is_incomplete,
    artifact_title_score_text,
    best_performing_trials,
    collect_heatmap_cells,
    collect_parallel_coordinate_trials,
    discover_validation_metric_artifacts,
    format_saved_file_message as format_tuning_saved_file_message,
    metric_series_by_split,
    parse_args as parse_summary_args,
    plot_all_agent_training_progress,
    plot_best_trial_training_summary,
    plot_path_length_summary,
    plot_validation_metric_montage,
    summarize_tuning_runs,
    training_metric_series,
)


def write_dataset_bundle(
    root: Path,
    name: str = "dataset_3x2x1_t1_seed42",
) -> Path:
    dataset_dir = root / name
    dataset_dir.mkdir(parents=True)
    for split in [
        "train.npz",
        "validation.npz",
        "same_layout_new_goals.npz",
        "validation_natural.npz",
        "test.npz",
        "test_natural.npz",
    ]:
        (dataset_dir / split).write_bytes(b"")
    return dataset_dir


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


def test_sample_hyperparameters_with_optuna_uses_trial_suggestions():
    class FakeTrial:
        def suggest_float(
            self,
            name,
            low,
            high,
            *,
            log=False,
        ):
            if log:
                return high

            return low

        def suggest_categorical(
            self,
            name,
            values,
        ):
            return values[-1]

        def suggest_int(
            self,
            name,
            low,
            high,
        ):
            return high

    sample = sample_hyperparameters_with_optuna(
        {
            "learning_rate": {
                "type": "loguniform",
                "low": 1e-5,
                "high": 1e-3,
            },
            "gamma": {
                "type": "uniform",
                "low": 0.95,
                "high": 0.999,
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
        FakeTrial(),
    )

    assert sample == {
        "learning_rate": 1e-3,
        "gamma": 0.95,
        "batch_size": 64,
        "update_epochs": 4,
    }


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


def test_parse_args_has_no_default_read_or_save_dirs(monkeypatch):
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
    assert args.validation_all_tasks is True
    assert args.tensorboard is False
    assert args.tensorboard_from_cli is False
    assert args.sampler == "random"
    assert args.resume is False
    assert args.dataset is None
    assert args.train_dataset is None
    assert args.validation_dataset is None
    assert args.same_layout_dataset is None
    assert args.test_dataset is None
    assert args.output_dir is None


def test_parse_args_accepts_validation_sampling_opt_out(
    monkeypatch,
):
    monkeypatch.setattr(
        "sys.argv",
        [
            "tune_dnn.py",
            "--algorithm",
            "ppo",
            "--no-validation-all-tasks",
        ],
    )

    args = parse_args()

    assert args.validation_all_tasks is False


def test_tuning_experiment_output_dir_uses_train_metadata(
    tmp_path,
):
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    train_dataset = dataset_dir / "train.npz"
    train_dataset.write_bytes(b"")
    (
        dataset_dir / "metadata.json"
    ).write_text(
        json.dumps(
            {
                "train_mazes": 5000,
                "validation_mazes": 500,
                "test_mazes": 500,
                "tasks_per_maze": 5,
                "master_seed": 42,
            }
        )
    )

    assert default_experiment_tuning_output_dir(
        experiment_name="generalization_followup",
        train_dataset=train_dataset,
        experiment_root=tmp_path / "runs",
    ) == (
        tmp_path
        / "runs"
        / "generalization_followup_5000x500x500_t5_seed42"
    )


def test_dataset_bundle_resolution_infers_splits_and_output(
    tmp_path,
    monkeypatch,
):
    dataset_dir = write_dataset_bundle(
        tmp_path / "data" / "datasets",
        "tuning_200x50x50_t5_seed42",
    )
    monkeypatch.chdir(tmp_path)
    args = SimpleNamespace(
        dataset=dataset_dir.name,
        train_dataset=None,
        validation_dataset=None,
        same_layout_dataset=None,
        test_dataset=None,
        output_dir=None,
        experiment_name=None,
        experiment_root=tmp_path / "runs",
    )

    resolve_dataset_bundle_args(args)

    resolved_dataset_dir = (
        Path("data")
        / "datasets"
        / "tuning_200x50x50_t5_seed42"
    )
    assert args.train_dataset == (
        resolved_dataset_dir / "train.npz"
    )
    assert args.validation_dataset == (
        resolved_dataset_dir / "validation.npz"
    )
    assert args.same_layout_dataset == (
        resolved_dataset_dir / "same_layout_new_goals.npz"
    )
    assert args.natural_validation_dataset == (
        resolved_dataset_dir / "validation_natural.npz"
    )
    assert args.test_dataset == (
        resolved_dataset_dir / "test.npz"
    )
    assert args.natural_test_dataset == (
        resolved_dataset_dir / "test_natural.npz"
    )
    assert args.output_dir == (
        tmp_path
        / "runs"
        / "tuning_200x50x50_t5_seed42"
    )


def test_dataset_bundle_resolution_prefers_validation_tune_for_tuning(
    tmp_path,
    monkeypatch,
):
    dataset_dir = write_dataset_bundle(
        tmp_path / "data" / "datasets",
        "tuning_200x50x50_t5_seed42",
    )
    (dataset_dir / "validation_tune.npz").write_bytes(b"")
    monkeypatch.chdir(tmp_path)
    args = SimpleNamespace(
        dataset=dataset_dir.name,
        train_dataset=None,
        validation_dataset=None,
        same_layout_dataset=None,
        test_dataset=None,
        output_dir=None,
        experiment_name=None,
        experiment_root=tmp_path / "runs",
    )

    resolve_dataset_bundle_args(args)

    assert args.validation_dataset == (
        Path("data")
        / "datasets"
        / "tuning_200x50x50_t5_seed42"
        / "validation_tune.npz"
    )


def test_dataset_bundle_dir_for_reference_expands_dataset_name():
    assert dataset_bundle_dir_for_reference(
        Path("tuning_200x50x50_t5_seed42")
    ) == (
        Path("data")
        / "datasets"
        / "tuning_200x50x50_t5_seed42"
    )


def test_dataset_bundle_tuning_output_dir_uses_dataset_name(
    tmp_path,
):
    assert dataset_bundle_tuning_output_dir(
        dataset_dir=tmp_path / "tuning_200x50x50_t5_seed42",
        experiment_root=tmp_path / "runs",
    ) == (
        tmp_path
        / "runs"
        / "tuning_200x50x50_t5_seed42"
    )


def test_parse_args_accepts_optuna_sampler(
    monkeypatch,
):
    monkeypatch.setattr(
        "sys.argv",
        [
            "tune_dnn.py",
            "--algorithm",
            "ppo",
            "--sampler",
            "optuna",
        ],
    )

    args = parse_args()

    assert args.sampler == "optuna"


def test_parse_args_accepts_optuna_shortcut(
    monkeypatch,
):
    monkeypatch.setattr(
        "sys.argv",
        [
            "tune_dnn.py",
            "--algorithm",
            "ppo",
            "--optuna",
        ],
    )

    args = parse_args()

    assert args.sampler == "optuna"


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


def test_parse_args_rejects_removed_validation_interval_flag(
    monkeypatch,
):
    monkeypatch.setattr(
        "sys.argv",
        [
            "tune_dnn.py",
            "--algorithm",
            "ppo",
            "--validation-interval",
            "5",
        ],
    )

    with pytest.raises(SystemExit):
        parse_args()


def test_summarize_parse_args_derives_paths_from_dataset(
    monkeypatch,
    tmp_path,
):
    dataset_dir = tmp_path / "dataset_alpha"
    dataset_dir.mkdir()
    monkeypatch.setattr(
        "sys.argv",
        [
            "summarize_tuning_runs.py",
            "--dataset",
            str(dataset_dir),
        ],
    )

    args = parse_summary_args()

    assert args.runs_dir == (
        Path("runs") / "dnn" / "tuning" / "dataset_alpha"
    )
    assert args.output_dir == args.runs_dir / "summary_plots"
    assert args.metric == "mean_path_efficiency"


def test_tuning_saved_file_message_includes_clickable_url(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv(
        "WSL_DISTRO_NAME",
        "Ubuntu-24.04",
    )
    output_path = tmp_path / "heatmap image.png"

    assert format_tuning_saved_file_message(
        "Saved heatmap to ",
        output_path,
    ) == (
        f"Saved heatmap to {output_path}\n"
        "Open: file://wsl.localhost/Ubuntu-24.04"
        f"{output_path.resolve().as_posix().replace(' ', '%20')}"
    )


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
    assert args.sampler == "random"
    assert args.validation_all_tasks is True


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


def test_parse_args_rejects_removed_resume_training_state_flag(
    monkeypatch,
):
    monkeypatch.setattr(
        "sys.argv",
        [
            "tune_dnn.py",
            "--algorithm",
            "ppo",
            "--resume-training-state",
        ],
    )

    with pytest.raises(SystemExit):
        parse_args()


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
    dataset_dir = write_dataset_bundle(tmp_path)
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
            "--dataset",
            str(dataset_dir),
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
    dataset_dir = write_dataset_bundle(tmp_path)
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
            "--dataset",
            str(dataset_dir),
            "--hyperparameter-combinations",
            "1",
        ],
    )

    main()

    output = capsys.readouterr().out

    assert "Loaded tuning arguments from" not in output
    assert "--algorithm dqn" in output


def test_optuna_sampler_dry_run_uses_optional_study(
    tmp_path,
    monkeypatch,
    capsys,
):
    dataset_dir = write_dataset_bundle(tmp_path)
    class FakeSampler:
        def __init__(
            self,
            *,
            seed,
        ):
            self.seed = seed

    class FakeTrial:
        def __init__(
            self,
            number,
        ):
            self.number = number

        def suggest_float(
            self,
            name,
            low,
            high,
            *,
            log=False,
        ):
            return low

        def suggest_categorical(
            self,
            name,
            values,
        ):
            return values[0]

        def suggest_int(
            self,
            name,
            low,
            high,
        ):
            return low

    class FakeStudy:
        def __init__(self):
            self.trials = []
            self.told = []

        def ask(self):
            trial = FakeTrial(
                len(self.trials)
            )
            self.trials.append(trial)
            return trial

        def tell(
            self,
            trial,
            value,
        ):
            self.told.append(
                (
                    trial.number,
                    value,
                )
            )

    study = FakeStudy()
    created = []

    def create_study(
        *,
        direction,
        sampler,
    ):
        created.append(
            (
                direction,
                sampler,
            )
        )
        return study

    fake_optuna = SimpleNamespace(
        create_study=create_study,
        samplers=SimpleNamespace(
            TPESampler=FakeSampler,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "optuna",
        fake_optuna,
    )
    search_space_path = tmp_path / "space.json"
    search_space_path.write_text(
        json.dumps(
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
            }
        )
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "tune_dnn.py",
            "--algorithm",
            "ppo",
            "--sampler",
            "optuna",
            "--dry-run",
            "--output-dir",
            str(tmp_path),
            "--dataset",
            str(dataset_dir),
            "--search-space",
            str(search_space_path),
            "--hyperparameter-combinations",
            "2",
        ],
    )

    main()

    output = capsys.readouterr().out

    assert created[0][0] == "maximize"
    assert created[0][1].seed == 42
    assert study.told == [
        (
            0,
            0.0,
        ),
        (
            1,
            0.0,
        ),
    ]
    assert "Hyperparameter combination 1/2" in output
    assert "Hyperparameter combination 2/2" in output
    assert '"learning_rate": 0.001' in output


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
            '"task_selection_passes_requested": 3,'
            '"task_selection_passes_completed": 3,'
            '"dataset_epochs_completed": 3,'
            '"stopped_early": false,'
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
    assert "--no-progress" not in command
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
    assert "--validation-interval" not in command
    assert "--validation-all-tasks" not in command
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


def test_training_command_can_opt_out_of_all_validation_tasks(
    tmp_path,
):
    args = SimpleNamespace(
        algorithm="ppo",
        rollouts_per_task=3,
        max_steps=7,
        validation_dataset=tmp_path / "validation.npz",
        same_layout_dataset=None,
        validation_all_tasks=False,
        early_stopping_patience=4,
        early_stopping_min_delta=0.0,
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
        },
    )

    assert "--no-validation-all-tasks" in command
    assert "--validation-all-tasks" not in command


def test_training_command_keeps_plots_when_requested(tmp_path):
    args = SimpleNamespace(
        algorithm="ppo",
        rollouts_per_task=3,
        max_steps=7,
        validation_dataset=tmp_path / "validation.npz",
        same_layout_dataset=None,
        early_stopping_patience=4,
        early_stopping_min_delta=0.0,
        q_snapshot_count=7,
        tensorboard=False,
        keep_plots=True,
    )

    command = build_training_command(
        args=args,
        trial_dir=tmp_path / "trial",
        train_dataset=tmp_path / "train.npz",
        trial_seed=11,
        hyperparameters={
            "learning_rate": 0.0003,
        },
    )

    assert "--plot" in command
    assert "--no-plot" not in command


def test_training_command_can_request_resume_when_state_exists(
    tmp_path,
):
    state_dir = tmp_path / "trial" / "ppo"
    state_dir.mkdir(
        parents=True,
    )
    (
        state_dir / "training_state.pt"
    ).write_bytes(b"state")
    args = SimpleNamespace(
        algorithm="ppo",
        rollouts_per_task=3,
        max_steps=7,
        validation_dataset=tmp_path / "validation.npz",
        same_layout_dataset=tmp_path
        / "same_layout_new_goals.npz",
        early_stopping_patience=4,
        early_stopping_min_delta=0.01,
        q_snapshot_count=7,
        tensorboard=False,
        keep_plots=False,
        resume=True,
    )

    command = build_training_command(
        args=args,
        trial_dir=tmp_path / "trial",
        train_dataset=tmp_path / "train.npz",
        trial_seed=11,
        hyperparameters={
            "learning_rate": 0.0003,
        },
    )

    assert "--resume" in command


def test_training_command_skips_resume_without_state(
    tmp_path,
):
    args = SimpleNamespace(
        algorithm="ppo",
        rollouts_per_task=3,
        max_steps=7,
        validation_dataset=tmp_path / "validation.npz",
        same_layout_dataset=tmp_path
        / "same_layout_new_goals.npz",
        early_stopping_patience=4,
        early_stopping_min_delta=0.01,
        q_snapshot_count=7,
        tensorboard=False,
        keep_plots=False,
        resume=True,
    )

    command = build_training_command(
        args=args,
        trial_dir=tmp_path / "trial",
        train_dataset=tmp_path / "train.npz",
        trial_seed=11,
        hyperparameters={
            "learning_rate": 0.0003,
        },
    )

    assert "--resume" not in command


def test_training_command_passes_grpo_task_and_group_sizes(tmp_path):
    args = SimpleNamespace(
        algorithm="grpo",
        rollouts_per_task=16,
        max_steps=7,
        validation_dataset=tmp_path / "validation.npz",
        same_layout_dataset=tmp_path
        / "same_layout_new_goals.npz",
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


def test_summarize_evaluation_computes_weighted_path_metrics(tmp_path):
    path = tmp_path / "evaluation.csv"

    with path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "episode_return",
                "success",
                "path_efficiency",
                "optimal_path_length",
            ],
        )
        writer.writeheader()
        writer.writerows(
            [
                {
                    "episode_return": "1.0",
                    "success": "True",
                    "path_efficiency": "0.8",
                    "optimal_path_length": "10",
                },
                {
                    "episode_return": "0.0",
                    "success": "False",
                    "path_efficiency": "0.0",
                    "optimal_path_length": "20",
                },
            ]
        )

    summary = summarize_evaluation(path)

    assert summary["weighted_path_success"] == 1 / 3
    assert summary["weighted_path_efficiency"] == 8 / 30


def write_fake_validation_artifacts(
    run_dir,
    *,
    validation_score,
    same_layout_score,
    train_greedy_score=None,
    training_score=None,
    hyperparameters=None,
    task_batch_size=4,
    rollout_group_size=2,
    write_training_plot=True,
    completed=None,
    stopped_early=False,
    status=None,
):
    if train_greedy_score is None:
        train_greedy_score = validation_score

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
            "train_greedy,1,2,1.0,2.0,"
            f"{train_greedy_score},"
            f"{train_greedy_score}\n"
            "train_greedy,200,2,1.0,2.0,"
            f"{train_greedy_score},"
            f"{train_greedy_score}\n"
            "validation,1,2,1.0,2.0,"
            f"{validation_score},"
            f"{validation_score}\n"
            "validation,200,2,1.0,2.0,"
            f"{validation_score},"
            f"{validation_score}\n"
            "same_layout,1,2,1.0,2.0,"
            f"{same_layout_score},"
            f"{same_layout_score}\n"
            "same_layout,200,2,1.0,2.0,"
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

    training_summary = None
    if (
        hyperparameters is not None
        or completed is not None
        or status is not None
    ):
        training_summary = {
            "hyperparameters": hyperparameters or {},
            "task_batch_size": task_batch_size,
            "rollout_group_size": rollout_group_size,
        }

        if status is not None:
            training_summary["status"] = status

        if completed is not None:
            training_summary.update(
                {
                    "task_selection_passes_requested": 3,
                    "task_selection_passes_completed": (
                        3
                        if completed
                        else 2
                    ),
                    "dataset_epochs_completed": (
                        3
                        if completed
                        else 2
                    ),
                    "stopped_early": stopped_early,
                }
            )

    if training_score is None:
        if training_summary is not None:
            (
                run_dir / "training_summary.json"
            ).write_text(
                json.dumps(
                    training_summary
                )
            )
        return

    (
        run_dir / "metrics.csv"
    ).write_text(
        (
            "episode,epoch,task_index,layout_index,"
            "episode_return,steps,internal_updates,success,"
            "wall_collisions,optimal_path_length,"
            "path_efficiency\n"
            "1,1,0,0,1.0,4,,True,0,4,"
            f"{training_score - 0.1}\n"
            "2,1,1,0,1.0,4,,True,0,4,"
            f"{training_score}\n"
            "3,2,0,0,1.0,4,,True,0,4,"
            f"{training_score - 0.2}\n"
        )
    )

    if not write_training_plot:
        if training_summary is not None:
            (
                run_dir / "training_summary.json"
            ).write_text(
                json.dumps(
                    training_summary
                )
            )
        return

    figure, axis = plt.subplots(
        figsize=(1.0, 1.0)
    )
    axis.plot(
        [0, 1],
        [
            0,
            training_score,
        ],
    )
    axis.axis("off")
    figure.savefig(
        run_dir / "training_metrics.png"
    )
    plt.close(figure)

    if training_summary is not None:
        (
            run_dir / "training_summary.json"
        ).write_text(
            json.dumps(
                training_summary
            )
        )


def test_summarize_tuning_writes_heatmap_trial_grids_and_parallel_coordinates(
    tmp_path,
    capsys,
):
    output_dir = tmp_path / "tuning"
    write_fake_validation_artifacts(
        output_dir
        / "trials"
        / "trial_001"
        / "dqn",
        validation_score=0.25,
        same_layout_score=0.2,
        training_score=0.45,
        hyperparameters={
            "learning_rate": 0.001,
            "gamma": 0.95,
            "batch_size": 32,
        },
        write_training_plot=False,
    )
    write_fake_validation_artifacts(
        output_dir
        / "trials"
        / "trial_002"
        / "dqn",
        validation_score=0.5,
        same_layout_score=0.4,
        training_score=0.55,
        hyperparameters={
            "learning_rate": 0.0001,
            "gamma": 0.99,
            "batch_size": 64,
        },
    )
    write_fake_validation_artifacts(
        output_dir
        / "trials"
        / "trial_001"
        / "ppo",
        validation_score=0.75,
        same_layout_score=0.7,
        training_score=0.8,
        hyperparameters={
            "learning_rate": 0.0003,
            "gamma": 0.98,
            "clip_ratio": 0.2,
        },
    )
    summary_output_dir = (
        tmp_path / "summary"
    )

    summarize_tuning_runs(
        SimpleNamespace(
            runs_dir=output_dir,
            output_dir=summary_output_dir,
            metric="mean_path_efficiency",
        )
    )
    summary_output = capsys.readouterr().out

    assert "=== Summary: tuning heatmap ===" in summary_output
    assert (
        "=== Summary: all-agent training progress ==="
        in summary_output
    )
    assert (
        "=== Summary: best-trial training history by agent ==="
        in summary_output
    )
    assert "=== Agent: dqn ===" in summary_output
    assert "=== Agent: ppo ===" in summary_output
    assert summary_output.index(
        "=== Agent: dqn ==="
    ) < summary_output.index(
        "=== Agent: ppo ==="
    )

    assert (
        summary_output_dir
        / "mean_path_efficiency_heatmap.png"
    ).exists()
    assert (
        summary_output_dir
        / "mean_path_efficiency_training_progress_by_agent.png"
    ).exists()
    assert (
        summary_output_dir
        / "mean_path_efficiency_best_trial_training_history_summary.png"
    ).exists()
    assert (
        summary_output_dir
        / "success_rate_heatmap.png"
    ).exists()
    assert (
        summary_output_dir
        / "success_rate_training_progress_by_agent.png"
    ).exists()
    assert (
        summary_output_dir
        / "success_rate_best_trial_training_history_summary.png"
    ).exists()
    assert (
        summary_output_dir
        / "optimal_path_length_summary.png"
    ).exists()
    assert (
        summary_output_dir
        / "dqn_validation_metrics_trials.png"
    ).exists()
    assert (
        summary_output_dir
        / "ppo_validation_metrics_trials.png"
    ).exists()
    import imageio.v2 as imageio

    assert imageio.imread(
        summary_output_dir
        / "ppo_validation_metrics_trials.png"
    ).shape[-1] == 3
    assert (
        summary_output_dir
        / "dqn_mean_path_efficiency_parallel_coordinates.png"
    ).exists()
    assert (
        summary_output_dir
        / "ppo_mean_path_efficiency_parallel_coordinates.png"
    ).exists()
    assert not (
        output_dir
        / "trials"
        / "trial_001"
        / "dqn"
        / "training_metrics.png"
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

    dqn_trial = next(
        artifact
        for artifact in artifacts
        if artifact["algorithm"] == "dqn"
        and artifact["trial"] == 1
    )

    assert artifact_title_score_text(
        dqn_trial,
        metric="mean_path_efficiency",
    ) == "v 0.25  vs 0.20  t 0.40"

    parallel_trials = collect_parallel_coordinate_trials(
        artifacts,
        metric="mean_path_efficiency",
    )

    assert {
        (
            trial["algorithm"],
            trial["trial"],
            trial["score"],
            trial["hyperparameters"][
                "task_batch_size"
            ],
            trial["hyperparameters"][
                "rollout_group_size"
            ],
        )
        for trial in parallel_trials
    } == {
        (
            "dqn",
            1,
            0.25,
            4.0,
            2.0,
        ),
        (
            "dqn",
            2,
            0.5,
            4.0,
            2.0,
        ),
        (
            "ppo",
            1,
            0.75,
            4.0,
            2.0,
        ),
    }


def test_aligned_training_metric_series_uses_validation_epochs(tmp_path):
    run_dir = (
        tmp_path
        / "tuning"
        / "trials"
        / "trial_001"
        / "dqn"
    )
    write_fake_validation_artifacts(
        run_dir,
        validation_score=0.25,
        same_layout_score=0.2,
        training_score=0.45,
        write_training_plot=False,
    )

    assert aligned_training_metric_series(
        run_dir / "metrics.csv",
        metric="mean_path_efficiency",
        aligned_epochs={1},
    ) == [
        (
            1,
            0.4,
        )
    ]


def test_training_metric_series_uses_full_training_progress(tmp_path):
    run_dir = (
        tmp_path
        / "tuning"
        / "trials"
        / "trial_001"
        / "dqn"
    )
    write_fake_validation_artifacts(
        run_dir,
        validation_score=0.25,
        same_layout_score=0.2,
        training_score=0.45,
        write_training_plot=False,
    )

    assert training_metric_series(
        run_dir / "metrics.csv",
        metric="mean_path_efficiency",
    ) == [
        (
            1,
            0.4,
        ),
        (
            2,
            0.25,
        ),
    ]


def test_all_agent_training_progress_summarizes_full_training_curves(
    tmp_path,
    monkeypatch,
):
    output_dir = tmp_path / "tuning"
    write_fake_validation_artifacts(
        output_dir
        / "trials"
        / "trial_001"
        / "dqn",
        validation_score=0.25,
        same_layout_score=0.2,
        training_score=0.45,
    )
    write_fake_validation_artifacts(
        output_dir
        / "trials"
        / "trial_002"
        / "dqn",
        validation_score=0.75,
        same_layout_score=0.7,
        training_score=0.6,
        completed=False,
    )
    write_fake_validation_artifacts(
        output_dir
        / "trials"
        / "trial_001"
        / "ppo",
        validation_score=0.5,
        same_layout_score=0.45,
        training_score=0.55,
        completed=True,
    )
    write_fake_validation_artifacts(
        output_dir
        / "trials"
        / "trial_002"
        / "ppo",
        validation_score=0.25,
        same_layout_score=0.2,
        training_score=0.35,
        completed=True,
    )
    artifacts = discover_validation_metric_artifacts(
        output_dir
    )
    saved_figures = []

    import matplotlib.figure

    def fake_savefig(
        self,
        path,
        *args,
        **kwargs,
    ):
        saved_figures.append(
            {
                "path": path,
                "axis_count": len(self.axes),
                "line_counts": [
                    len(axis.lines)
                    for axis in self.axes
                ],
                "x_data": [
                    [
                        tuple(
                            int(value)
                            for value in line.get_xdata()
                        )
                        for line in axis.lines
                    ]
                    for axis in self.axes
                ],
                "line_widths": [
                    [
                        line.get_linewidth()
                        for line in axis.lines
                    ]
                    for axis in self.axes
                ],
                "texts": [
                    [
                        text.get_text()
                        for text in axis.texts
                    ]
                    for axis in self.axes
                ],
            }
        )

    monkeypatch.setattr(
        matplotlib.figure.Figure,
        "savefig",
        fake_savefig,
    )

    assert plot_all_agent_training_progress(
        artifacts,
        metric="mean_path_efficiency",
        output_path=(
            tmp_path
            / "summary"
            / "mean_path_efficiency_training_progress_by_agent.png"
        ),
    ) == 4

    assert saved_figures == [
        {
            "path": (
                tmp_path
                / "summary"
                / "mean_path_efficiency_training_progress_by_agent.png"
            ),
            "axis_count": 2,
            "line_counts": [
                2,
                2,
            ],
            "x_data": [
                [
                    (
                        1,
                        2,
                    ),
                    (
                        1,
                        2,
                    ),
                ],
                [
                    (
                        1,
                        2,
                    ),
                    (
                        1,
                        2,
                    )
                ],
            ],
            "line_widths": [
                [
                    1.2,
                    2.8,
                ],
                [
                    2.8,
                    1.2,
                ],
            ],
            "texts": [
                [
                    "INCOMPLETE"
                ],
                [],
            ],
        }
    ]


def test_best_trial_training_summary_plots_one_trial_per_agent(
    tmp_path,
    monkeypatch,
):
    output_dir = tmp_path / "tuning"
    write_fake_validation_artifacts(
        output_dir
        / "trials"
        / "trial_001"
        / "dqn",
        validation_score=0.25,
        same_layout_score=0.2,
        training_score=0.45,
    )
    write_fake_validation_artifacts(
        output_dir
        / "trials"
        / "trial_002"
        / "dqn",
        validation_score=0.75,
        same_layout_score=0.7,
        training_score=0.6,
        completed=False,
    )
    write_fake_validation_artifacts(
        output_dir
        / "trials"
        / "trial_001"
        / "ppo",
        validation_score=0.5,
        same_layout_score=0.45,
        training_score=0.55,
        completed=True,
    )
    write_fake_validation_artifacts(
        output_dir
        / "trials"
        / "trial_002"
        / "ppo",
        validation_score=0.25,
        same_layout_score=0.2,
        training_score=0.35,
        completed=True,
    )
    artifacts = discover_validation_metric_artifacts(
        output_dir
    )
    saved_figures = []

    import matplotlib.figure

    def fake_savefig(
        self,
        path,
        *args,
        **kwargs,
    ):
        saved_figures.append(
            {
                "path": path,
                "axis_count": len(self.axes),
                "line_counts": [
                    len(axis.lines)
                    for axis in self.axes
                ],
                "titles": [
                    axis.get_title()
                    for axis in self.axes
                ],
                "texts": [
                    [
                        text.get_text()
                        for text in axis.texts
                    ]
                    for axis in self.axes
                ],
                "text_colors": [
                    [
                        text.get_color()
                        for text in axis.texts
                    ]
                    for axis in self.axes
                ],
                "line_labels": [
                    [
                        line.get_label()
                        for line in axis.lines
                        if line.get_linestyle() != ":"
                    ]
                    for axis in self.axes
                ],
            }
        )

    monkeypatch.setattr(
        matplotlib.figure.Figure,
        "savefig",
        fake_savefig,
    )

    assert plot_best_trial_training_summary(
        artifacts,
        metric="mean_path_efficiency",
        output_path=(
            tmp_path
            / "summary"
            / "mean_path_efficiency_best_trial_training_history_summary.png"
        ),
    ) == 2

    assert saved_figures == [
        {
            "path": (
                tmp_path
                / "summary"
                / "mean_path_efficiency_best_trial_training_history_summary.png"
            ),
            "axis_count": 2,
            "line_counts": [
                4,
                4,
            ],
            "titles": [
                "dqn\nv 0.75  vs 0.70  t 0.55",
                "ppo\nv 0.50  vs 0.45  t 0.50",
            ],
            "texts": [
                [
                    "INCOMPLETE",
                    "best val epoch 1",
                ],
                [
                    "best val epoch 1",
                ],
            ],
            "text_colors": [
                [
                    "#c62828",
                    "#666666",
                ],
                [
                    "#666666",
                ],
            ],
            "line_labels": [
                [
                    "train",
                    "val",
                    "same-val",
                ],
                [
                    "train",
                    "val",
                    "same-val",
                ],
            ],
        }
    ]


def test_aligned_training_metric_series_filters_successful_efficiency(
    tmp_path,
):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (
        run_dir / "metrics.csv"
    ).write_text(
        (
            "episode,epoch,success,path_efficiency\n"
            "1,1,True,0.8\n"
            "2,1,False,0.2\n"
            "3,1,True,0.6\n"
            "4,2,False,0.9\n"
        )
    )

    assert aligned_training_metric_series(
        run_dir / "metrics.csv",
        metric="average_successful_path_efficiency",
        aligned_epochs={
            1,
            2,
        },
    ) == [
        (
            1,
            0.7,
        )
    ]


def test_best_performing_trials_marks_validation_metric_ties(tmp_path):
    output_dir = tmp_path / "tuning"
    write_fake_validation_artifacts(
        output_dir
        / "trials"
        / "trial_001"
        / "dqn",
        validation_score=0.5,
        same_layout_score=0.9,
    )
    write_fake_validation_artifacts(
        output_dir
        / "trials"
        / "trial_002"
        / "dqn",
        validation_score=0.7,
        same_layout_score=0.6,
    )
    write_fake_validation_artifacts(
        output_dir
        / "trials"
        / "trial_003"
        / "dqn",
        validation_score=0.7,
        same_layout_score=0.4,
    )

    artifacts = [
        artifact
        for artifact in discover_validation_metric_artifacts(
            output_dir
        )
        if artifact["algorithm"] == "dqn"
    ]

    assert best_performing_trials(
        artifacts,
        metric="mean_path_efficiency",
    ) == {
        2,
        3,
    }


def test_summarize_tuning_removes_stale_paginated_trial_grids(tmp_path):
    output_dir = tmp_path / "tuning"

    for trial_number in range(1, 6):
        write_fake_validation_artifacts(
            output_dir
            / "trials"
            / f"trial_{trial_number:03d}"
            / "ppo",
            validation_score=trial_number / 10,
            same_layout_score=trial_number / 20,
        )

    summary_output_dir = (
        tmp_path / "summary"
    )
    summary_output_dir.mkdir()
    (
        summary_output_dir
        / "ppo_validation_metrics_trials_page_02.png"
    ).write_bytes(b"stale")

    summarize_tuning_runs(
        SimpleNamespace(
            runs_dir=output_dir,
            output_dir=summary_output_dir,
            metric="mean_path_efficiency",
        )
    )

    assert (
        summary_output_dir
        / "ppo_validation_metrics_trials.png"
    ).exists()
    assert not (
        summary_output_dir
        / "ppo_validation_metrics_trials_page_02.png"
    ).exists()


def test_validation_metric_montage_uses_single_5_by_4_sheet_for_twenty_trials(
    tmp_path,
    monkeypatch,
):
    output_dir = tmp_path / "tuning"

    for trial_number in range(1, 21):
        write_fake_validation_artifacts(
            output_dir
            / "trials"
            / f"trial_{trial_number:03d}"
            / "ppo",
            validation_score=trial_number / 100,
            same_layout_score=trial_number / 200,
        )

    artifacts = discover_validation_metric_artifacts(
        output_dir
    )
    output_path = (
        tmp_path
        / "summary"
        / "ppo_validation_metrics_trials.png"
    )
    saved_figures = []

    import matplotlib.figure

    def fake_savefig(
        self,
        path,
        *args,
        **kwargs,
    ):
        saved_figures.append(
            {
                "path": path,
                "size_inches": tuple(
                    self.get_size_inches()
                ),
                "dpi": kwargs.get("dpi"),
                "pixel_size": tuple(
                    round(
                        size_in_inches
                        * kwargs.get("dpi")
                    )
                    for size_in_inches in self.get_size_inches()
                ),
                "axis_count": len(self.axes),
            }
        )

    monkeypatch.setattr(
        matplotlib.figure.Figure,
        "savefig",
        fake_savefig,
    )

    assert plot_validation_metric_montage(
        artifacts,
        algorithm="ppo",
        metric="mean_path_efficiency",
        output_path=output_path,
    ) == 20

    assert saved_figures == [
        {
            "path": output_path,
            "size_inches": pytest.approx(
                (
                    35.0,
                    28.8,
                )
            ),
            "dpi": 160,
            "pixel_size": (
                5600,
                4608,
            ),
            "axis_count": 20,
        }
    ]


def test_validation_metric_montage_reports_filled_trial_slots(
    tmp_path,
    monkeypatch,
):
    output_dir = tmp_path / "tuning"

    for trial_number in range(1, 4):
        write_fake_validation_artifacts(
            output_dir
            / "trials"
            / f"trial_{trial_number:03d}"
            / "ppo",
            validation_score=trial_number / 100,
            same_layout_score=trial_number / 200,
        )

    artifacts = discover_validation_metric_artifacts(
        output_dir
    )
    saved_figures = []

    import matplotlib.figure

    def fake_savefig(
        self,
        path,
        *args,
        **kwargs,
    ):
        saved_figures.append(
            {
                "axis_count": len(self.axes),
                "visible_axes": sum(
                    axis.axison
                    for axis in self.axes
                ),
            }
        )

    monkeypatch.setattr(
        matplotlib.figure.Figure,
        "savefig",
        fake_savefig,
    )

    assert plot_validation_metric_montage(
        artifacts,
        algorithm="ppo",
        metric="mean_path_efficiency",
        output_path=(
            tmp_path
            / "summary"
            / "ppo_validation_metrics_trials.png"
        ),
    ) == 3

    assert saved_figures == [
        {
            "axis_count": 20,
            "visible_axes": 20,
        }
    ]


def test_validation_metric_montage_uses_fixed_bounded_trial_scale(
    tmp_path,
    monkeypatch,
):
    output_dir = tmp_path / "tuning"
    write_fake_validation_artifacts(
        output_dir
        / "trials"
        / "trial_001"
        / "ppo",
        validation_score=0.02,
        same_layout_score=0.03,
        training_score=0.04,
    )

    artifacts = discover_validation_metric_artifacts(
        output_dir
    )
    saved_figures = []

    import matplotlib.figure

    def fake_savefig(
        self,
        path,
        *args,
        **kwargs,
    ):
        saved_figures.append(
            {
                "first_ylim": self.axes[0].get_ylim(),
            }
        )

    monkeypatch.setattr(
        matplotlib.figure.Figure,
        "savefig",
        fake_savefig,
    )

    assert plot_validation_metric_montage(
        artifacts,
        algorithm="ppo",
        metric="mean_path_efficiency",
        output_path=(
            tmp_path
            / "summary"
            / "ppo_validation_metrics_trials.png"
        ),
    ) == 1

    assert saved_figures[0]["first_ylim"] == (
        0.0,
        1.0,
    )


def test_validation_metric_montage_uses_requested_budget_x_axis_for_early_stop(
    tmp_path,
    monkeypatch,
):
    output_dir = tmp_path / "tuning"
    run_dir = (
        output_dir
        / "trials"
        / "trial_001"
        / "ppo"
    )
    write_fake_validation_artifacts(
        run_dir,
        validation_score=0.02,
        same_layout_score=0.03,
        training_score=0.04,
    )
    (
        run_dir / "training_summary.json"
    ).write_text(
        json.dumps(
            {
                "dataset_epochs_requested": 8,
                "task_selection_passes_requested": 8,
                "dataset_epochs_completed": 2,
                "stopped_early": True,
            }
        )
    )

    artifacts = discover_validation_metric_artifacts(
        output_dir
    )
    saved_figures = []

    import matplotlib.figure

    def fake_savefig(
        self,
        path,
        *args,
        **kwargs,
    ):
        first_axis = self.axes[0]
        saved_figures.append(
            {
                "first_xlim": first_axis.get_xlim(),
                "first_xticks": tuple(
                    int(tick)
                    for tick in first_axis.get_xticks()
                ),
                "first_xtick_labels_visible": [
                    label.get_visible()
                    for label in first_axis.get_xticklabels()
                ],
            }
        )

    monkeypatch.setattr(
        matplotlib.figure.Figure,
        "savefig",
        fake_savefig,
    )

    assert plot_validation_metric_montage(
        artifacts,
        algorithm="ppo",
        metric="mean_path_efficiency",
        output_path=(
            tmp_path
            / "summary"
            / "ppo_validation_metrics_trials.png"
        ),
    ) == 1

    assert saved_figures[0]["first_xlim"] == (
        0.0,
        8.0,
    )
    assert saved_figures[0]["first_xticks"] == (
        0,
        2,
        4,
        6,
        8,
    )
    assert saved_figures[0][
        "first_xtick_labels_visible"
    ]
    assert all(
        saved_figures[0][
            "first_xtick_labels_visible"
        ]
    )


def test_validation_metric_montage_labels_incomplete_trials(
    tmp_path,
    monkeypatch,
):
    output_dir = tmp_path / "tuning"
    write_fake_validation_artifacts(
        output_dir
        / "trials"
        / "trial_001"
        / "ppo",
        validation_score=0.5,
        same_layout_score=0.25,
        completed=False,
    )
    artifacts = discover_validation_metric_artifacts(
        output_dir
    )
    saved_figures = []

    import matplotlib.figure

    def fake_savefig(
        self,
        path,
        *args,
        **kwargs,
    ):
        saved_figures.append(
            {
                "title": self.axes[0].get_title(),
                "texts": [
                    text.get_text()
                    for text in self.axes[0].texts
                ],
                "text_colors": [
                    text.get_color()
                    for text in self.axes[0].texts
                ],
            }
        )

    monkeypatch.setattr(
        matplotlib.figure.Figure,
        "savefig",
        fake_savefig,
    )

    assert plot_validation_metric_montage(
        artifacts,
        algorithm="ppo",
        metric="mean_path_efficiency",
        output_path=(
            tmp_path
            / "summary"
            / "ppo_validation_metrics_trials.png"
        ),
    ) == 1

    assert saved_figures[0]["title"].startswith(
        "Trial 1 best\n"
    )
    assert saved_figures[0]["texts"] == [
        "INCOMPLETE",
        "best val epoch 1",
    ]
    assert saved_figures[0]["text_colors"] == [
        "#c62828",
        "#666666",
    ]


def test_tuning_artifact_early_stop_completion_honors_stale_summary(
    tmp_path,
):
    output_dir = tmp_path / "tuning"
    run_dir = (
        output_dir
        / "trials"
        / "trial_001"
        / "ppo"
    )
    write_fake_validation_artifacts(
        run_dir,
        validation_score=0.5,
        same_layout_score=0.25,
        completed=False,
        stopped_early=True,
    )
    artifacts = discover_validation_metric_artifacts(
        output_dir
    )
    assert len(artifacts) == 1
    assert artifact_is_incomplete(artifacts[0]) is False

    os.utime(
        run_dir / "training_summary.json",
        ns=(2_000, 2_000),
    )
    os.utime(
        run_dir / "validation_metrics.csv",
        ns=(3_000, 3_000),
    )

    assert artifact_is_incomplete(artifacts[0]) is True


def test_tuning_artifact_completion_uses_explicit_status(
    tmp_path,
):
    output_dir = tmp_path / "tuning"
    running_dir = (
        output_dir
        / "trials"
        / "trial_001"
        / "ppo"
    )
    early_stop_dir = (
        output_dir
        / "trials"
        / "trial_002"
        / "a2c"
    )

    write_fake_validation_artifacts(
        running_dir,
        validation_score=0.5,
        same_layout_score=0.25,
        completed=True,
        stopped_early=True,
        status="running",
    )
    write_fake_validation_artifacts(
        early_stop_dir,
        validation_score=0.4,
        same_layout_score=0.2,
        completed=False,
        status="early_stopped",
    )

    artifacts = discover_validation_metric_artifacts(
        output_dir
    )
    incomplete_by_trial = {
        artifact["trial"]: artifact_is_incomplete(
            artifact
        )
        for artifact in artifacts
    }

    assert incomplete_by_trial == {
        1: True,
        2: False,
    }


def test_validation_metric_montage_marks_empty_slots_incomplete(
    tmp_path,
    monkeypatch,
):
    output_dir = tmp_path / "tuning"
    write_fake_validation_artifacts(
        output_dir
        / "trials"
        / "trial_001"
        / "grpo",
        validation_score=0.5,
        same_layout_score=0.25,
        completed=True,
    )
    artifacts = discover_validation_metric_artifacts(
        output_dir
    )
    saved_figures = []

    import matplotlib.figure

    def fake_savefig(
        self,
        path,
        *args,
        **kwargs,
    ):
        saved_figures.append(
            {
                "empty_slot_texts": [
                    text.get_text()
                    for text in self.axes[1].texts
                ],
                "empty_slot_colors": [
                    text.get_color()
                    for text in self.axes[1].texts
                ],
            }
        )

    monkeypatch.setattr(
        matplotlib.figure.Figure,
        "savefig",
        fake_savefig,
    )

    assert plot_validation_metric_montage(
        artifacts,
        algorithm="grpo",
        metric="mean_path_efficiency",
        output_path=(
            tmp_path
            / "summary"
            / "grpo_validation_metrics_trials.png"
        ),
    ) == 1

    assert saved_figures == [
        {
            "empty_slot_texts": [
                "INCOMPLETE"
            ],
            "empty_slot_colors": [
                "#c62828"
            ],
        }
    ]


def test_validation_metric_montage_paginates_more_than_twenty_trials(
    tmp_path,
    monkeypatch,
):
    output_dir = tmp_path / "tuning"

    for trial_number in range(1, 22):
        write_fake_validation_artifacts(
            output_dir
            / "trials"
            / f"trial_{trial_number:03d}"
            / "ppo",
            validation_score=trial_number / 100,
            same_layout_score=trial_number / 200,
        )

    artifacts = discover_validation_metric_artifacts(
        output_dir
    )
    output_path = (
        tmp_path
        / "summary"
        / "ppo_validation_metrics_trials.png"
    )
    saved_figures = []

    import matplotlib.figure

    def fake_savefig(
        self,
        path,
        *args,
        **kwargs,
    ):
        saved_figures.append(
            {
                "path": path,
                "axis_count": len(self.axes),
                "visible_axes": sum(
                    axis.axison
                    for axis in self.axes
                ),
            }
        )

    monkeypatch.setattr(
        matplotlib.figure.Figure,
        "savefig",
        fake_savefig,
    )

    assert plot_validation_metric_montage(
        artifacts,
        algorithm="ppo",
        metric="mean_path_efficiency",
        output_path=output_path,
    ) == 21

    assert saved_figures == [
        {
            "path": output_path,
            "axis_count": 20,
            "visible_axes": 20,
        },
        {
            "path": (
                tmp_path
                / "summary"
                / "ppo_validation_metrics_trials_page_02.png"
            ),
            "axis_count": 20,
            "visible_axes": 20,
        },
    ]


def test_metric_series_by_split_reads_epoch_points(tmp_path):
    run_dir = tmp_path / "trial_001" / "ppo"
    write_fake_validation_artifacts(
        run_dir,
        validation_score=0.5,
        same_layout_score=0.25,
    )

    series = metric_series_by_split(
        run_dir / "validation_metrics.csv",
        "mean_path_efficiency",
    )

    assert series["validation"] == [
        (1, 0.5),
        (200, 0.5),
    ]
    assert series["same_layout"] == [
        (1, 0.25),
        (200, 0.25),
    ]


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
            '"task_selection_passes_requested": 3,'
            '"task_selection_passes_completed": 3,'
            '"dataset_epochs_completed": 3,'
            '"stopped_early": false,'
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
            '"task_selection_passes_requested": 3,'
            '"task_selection_passes_completed": 3,'
            '"dataset_epochs_completed": 3,'
            '"stopped_early": false,'
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


def test_read_completed_results_rejects_unfinished_artifacts(tmp_path):
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
            '"task_selection_passes_requested": 3,'
            '"task_selection_passes_completed": 2,'
            '"dataset_epochs_completed": 2,'
            '"stopped_early": false,'
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
            "task_selection_passes_completed": 2,
            "dataset_epochs_completed": 2,
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
            '"task_selection_passes_requested": 3,'
            '"task_selection_passes_completed": 3,'
            '"dataset_epochs_completed": 3,'
            '"stopped_early": false,'
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


def test_read_completed_results_ignores_newer_derived_artifacts(tmp_path):
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
    training_plot_path = (
        run_dir / "training_metrics.png"
    )
    validation_plot_path = (
        run_dir / "validation_metrics.png"
    )
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
            '"task_selection_passes_requested": 3,'
            '"task_selection_passes_completed": 3,'
            '"dataset_epochs_completed": 3,'
            '"stopped_early": false,'
            '"best_validation": {'
            '"mean_path_efficiency": 0.5'
            "}}"
        )
    )
    metrics_path.write_text("episode_return\n")
    training_plot_path.write_bytes(b"plot")
    validation_plot_path.write_bytes(b"plot")
    tensorboard_path.write_text("event")
    os.utime(
        checkpoint_path,
        ns=(1_000, 1_000),
    )
    os.utime(
        metrics_path,
        ns=(2_000, 2_000),
    )
    os.utime(
        training_summary_path,
        ns=(3_000, 3_000),
    )
    os.utime(
        evaluation_path,
        ns=(4_000, 4_000),
    )
    os.utime(
        training_plot_path,
        ns=(5_000, 5_000),
    )
    os.utime(
        validation_plot_path,
        ns=(5_000, 5_000),
    )
    os.utime(
        tensorboard_path,
        ns=(5_000, 5_000),
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

    assert list(completed) == [1]


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


@pytest.mark.parametrize(
    (
        "status",
        "completed",
        "stopped_early",
        "expected_result",
    ),
    [
        ("completed", False, False, True),
        ("early_stopped", False, False, True),
        ("running", True, True, False),
        ("interrupted", True, True, False),
    ],
)
def test_build_result_from_artifacts_uses_explicit_status(
    tmp_path,
    status,
    completed,
    stopped_early,
    expected_result,
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
        json.dumps(
            {
                "status": status,
                "algorithm": "ppo",
                "task_batch_size": 4,
                "rollout_group_size": 1,
                "dataset_epochs_completed": (
                    3
                    if completed
                    else 2
                ),
                "task_selection_passes_requested": 3,
                "task_selection_passes_completed": (
                    3
                    if completed
                    else 2
                ),
                "stopped_early": stopped_early,
                "hyperparameters": {
                    "learning_rate": 0.001,
                },
                "best_validation": {
                    "mean_path_efficiency": 0.75,
                    "success_rate": 1.0,
                    "average_episode_return": 2.0,
                    "average_successful_path_efficiency": 0.75,
                },
            }
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

    assert (result is not None) is expected_result


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


def test_load_search_space_rejects_grpo_singleton_rollout_group_size(tmp_path):
    path = tmp_path / "space.json"
    path.write_text(
        '{"learning_rate": {"type": "uniform", "low": 0.1, "high": 0.2}, '
        '"rollout_group_size": {"type": "choice", "values": [1, 2]}}'
    )

    with pytest.raises(
        ValueError,
        match="at least 2",
    ):
        load_search_space(
            "grpo",
            path,
        )


def test_load_search_space_rejects_grpo_group_size_alias_below_minimum(tmp_path):
    path = tmp_path / "space.json"
    path.write_text(
        '{"learning_rate": {"type": "uniform", "low": 0.1, "high": 0.2}, '
        '"group_size": {"type": "int", "low": 1, "high": 4}}'
    )

    with pytest.raises(
        ValueError,
        match="at least 2",
    ):
        load_search_space(
            "grpo",
            path,
        )


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
