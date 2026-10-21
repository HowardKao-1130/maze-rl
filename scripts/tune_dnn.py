from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = REPO_ROOT / "scripts" / "train.py"
EVALUATE_SCRIPT = REPO_ROOT / "scripts" / "evaluate.py"

for import_path in [
    REPO_ROOT / "src",
    REPO_ROOT,
]:
    import_path_text = str(import_path)

    if import_path_text not in sys.path:
        sys.path.insert(
            0,
            import_path_text,
        )

from scripts.train import (
    NEURAL_ALGORITHMS,
    NEURAL_HYPERPARAMETERS,
)
from scripts.experiment_naming import (
    DEFAULT_DATA_OUTPUT_ROOT,
    DEFAULT_DNN_TUNING_OUTPUT_ROOT,
    experiment_dataset_name_for_path,
)

DEFAULT_SEARCH_SPACES: dict[
    str,
    dict[str, dict[str, Any]],
] = {
    "dqn": {
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
        "epsilon_min": {
            "type": "choice",
            "values": [0.01, 0.03, 0.05, 0.1],
        },
        "epsilon_decay": {
            "type": "uniform",
            "low": 0.995,
            "high": 0.9999,
        },
        "min_replay_size": {
            "type": "choice",
            "values": [128, 256, 512, 1000],
        },
        "batch_size": {
            "type": "choice",
            "values": [32, 64, 128],
        },
        "target_update_interval": {
            "type": "choice",
            "values": [100, 250, 500, 1000],
        },
        "task_batch_size": {
            "type": "choice",
            "values": [1],
        },
        "rollout_group_size": {
            "type": "choice",
            "values": [1],
        },
    },
    "reinforce": {
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
        "entropy_coefficient": {
            "type": "loguniform",
            "low": 0.005,
            "high": 0.15,
        },
        "entropy_coefficient_min": {
            "type": "choice",
            "values": [0.0, 0.005, 0.01, 0.03],
        },
        "entropy_coefficient_decay": {
            "type": "uniform",
            "low": 0.995,
            "high": 0.9999,
        },
        "minibatch_size": {
            "type": "choice",
            "values": [32, 64, 128],
        },
        "task_batch_size": {
            "type": "choice",
            "values": [8, 16, 32],
        },
        "rollout_group_size": {
            "type": "choice",
            "values": [1],
        },
    },
    "a2c": {
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
        "gae_lambda": {
            "type": "uniform",
            "low": 0.85,
            "high": 0.99,
        },
        "value_coefficient": {
            "type": "uniform",
            "low": 0.25,
            "high": 1.0,
        },
        "entropy_coefficient": {
            "type": "loguniform",
            "low": 0.005,
            "high": 0.2,
        },
        "entropy_coefficient_min": {
            "type": "choice",
            "values": [0.0, 0.01, 0.03, 0.05],
        },
        "entropy_coefficient_decay": {
            "type": "uniform",
            "low": 0.995,
            "high": 0.9999,
        },
        "minibatch_size": {
            "type": "choice",
            "values": [32, 64, 128],
        },
        "task_batch_size": {
            "type": "choice",
            "values": [8, 16, 32],
        },
        "rollout_group_size": {
            "type": "choice",
            "values": [1],
        },
    },
    "ppo": {
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
        "gae_lambda": {
            "type": "uniform",
            "low": 0.85,
            "high": 0.99,
        },
        "clip_epsilon": {
            "type": "uniform",
            "low": 0.1,
            "high": 0.3,
        },
        "value_coefficient": {
            "type": "uniform",
            "low": 0.25,
            "high": 1.0,
        },
        "entropy_coefficient": {
            "type": "loguniform",
            "low": 0.001,
            "high": 0.08,
        },
        "update_epochs": {
            "type": "choice",
            "values": [2, 4, 6, 8],
        },
        "minibatch_size": {
            "type": "choice",
            "values": [32, 64, 128],
        },
        "task_batch_size": {
            "type": "choice",
            "values": [32, 64, 128],
        },
        "rollout_group_size": {
            "type": "choice",
            "values": [1],
        },
    },
    "grpo": {
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
        "clip_epsilon": {
            "type": "uniform",
            "low": 0.1,
            "high": 0.3,
        },
        "entropy_coefficient": {
            "type": "loguniform",
            "low": 0.001,
            "high": 0.08,
        },
        "entropy_coefficient_min": {
            "type": "choice",
            "values": [0.0, 0.001, 0.005, 0.01],
        },
        "entropy_coefficient_decay": {
            "type": "uniform",
            "low": 0.995,
            "high": 0.9999,
        },
        "minibatch_size": {
            "type": "choice",
            "values": [32, 64, 128],
        },
        "task_batch_size": {
            "type": "choice",
            "values": [8, 16, 32],
        },
        "rollout_group_size": {
            "type": "choice",
            "values": [4, 8, 16],
        },
    },
}

TRAINING_LOOP_PARAMETERS = {
    "task_batch_size",
    "rollout_group_size",
}

TRAINING_LOOP_PARAMETER_ALIASES = {
    "rollout_episodes": "task_batch_size",
    "group_size": "rollout_group_size",
}

MIN_GRPO_ROLLOUT_GROUP_SIZE = 2

RESULT_FIELDNAMES = [
    "trial",
    "algorithm",
    "seed",
    "objective",
    "mean_path_efficiency",
    "success_rate",
    "average_episode_return",
    "average_successful_path_efficiency",
    "weighted_path_success",
    "weighted_path_efficiency",
    "rollouts_per_task_requested",
    "task_batch_size",
    "rollout_group_size",
    "task_selection_passes_requested",
    "task_selection_passes_completed",
    "dataset_epochs_completed",
    "stopped_early",
    "checkpoint_path",
    "evaluation_path",
    "hyperparameters",
]

LEGACY_TUNING_CONFIG_FILENAME = "tuning_config.json"
TUNING_CONFIG_DIRNAME = "tuning_configs"
TUNING_ARGUMENT_FIELDS = [
    "algorithm",
    "sampler",
    "hyperparameter_combinations",
    "rollouts_per_task",
    "max_steps",
    "seed",
    "dataset",
    "train_dataset",
    "validation_dataset",
    "same_layout_dataset",
    "natural_validation_dataset",
    "test_dataset",
    "natural_test_dataset",
    "evaluate_best_on_test",
    "search_space",
    "validation_all_tasks",
    "q_snapshot_count",
    "early_stopping_patience",
    "early_stopping_min_delta",
    "tensorboard",
    "keep_plots",
]
TUNING_OPTIONAL_ARGUMENT_DEFAULTS = {
    "dataset": None,
    "natural_validation_dataset": None,
    "natural_test_dataset": None,
    "q_snapshot_count": 11,
    "sampler": "random",
    "validation_all_tasks": True,
}
TUNING_PATH_ARGUMENT_FIELDS = {
    "dataset",
    "train_dataset",
    "validation_dataset",
    "same_layout_dataset",
    "natural_validation_dataset",
    "test_dataset",
    "natural_test_dataset",
    "search_space",
}
COMMAND_TERMINATION_TIMEOUT_SECONDS = 10.0


class CommandInterrupted(Exception):
    def __init__(
        self,
        signum: int,
    ) -> None:
        super().__init__(
            f"Command interrupted by signal {signum}."
        )
        self.signum = signum


def parse_args(
    argv: list[str] | None = None,
) -> argparse.Namespace:
    if argv is None:
        argv = sys.argv[1:]


    parser = argparse.ArgumentParser(
        description=(
            "Random-search hyperparameter tuning for DNN maze "
            "RL agents."
        )
    )

    parser.add_argument(
        "--algorithm",
        required=True,
        choices=sorted(NEURAL_ALGORITHMS),
    )
    parser.add_argument(
        "--sampler",
        choices=[
            "random",
            "optuna",
        ],
        default="random",
        help=(
            "Hyperparameter sampler to use. Defaults to random "
            "search; choose optuna to use Optuna's TPE sampler."
        ),
    )
    parser.add_argument(
        "--optuna",
        dest="sampler",
        action="store_const",
        const="optuna",
        help=(
            "Use Optuna's TPE sampler for hyperparameter "
            "selection. Disabled by default."
        ),
    )
    parser.add_argument(
        "--hyperparameter-combinations",
        dest="hyperparameter_combinations",
        type=int,
        default=None,
        help=(
            "Number of hyperparameter combinations to sample."
        ),
    )
    parser.add_argument(
        "--trials",
        dest="hyperparameter_combinations",
        type=int,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--rollouts-per-task",
        "--dataset-epochs",
        dest="rollouts_per_task",
        type=int,
        default=200,
        help=(
            "Generous rollout budget per training task for every "
            "hyperparameter combination. --dataset-epochs is "
            "accepted as a "
            "backward-compatible alias."
        ),
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=200,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Optional tuning output directory override. When "
            "--dataset is a dataset directory, the default is "
            "runs/dnn/tuning/<dataset-directory>."
        ),
    )
    parser.add_argument(
        "--experiment-name",
        default=None,
        help=(
            "Optional prefix for automatically named experiment "
            "output under --experiment-root. The train dataset "
            "metadata provides the size and seed suffix."
        ),
    )
    parser.add_argument(
        "--experiment-root",
        type=Path,
        default=DEFAULT_DNN_TUNING_OUTPUT_ROOT,
        help=(
            "Root directory for automatically named experiment "
            "tuning outputs."
        ),
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help=(
            "Dataset bundle name under data/datasets, containing "
            "train.npz, validation.npz, validation_tune.npz, "
            "same_layout_new_goals.npz, "
            "validation_natural.npz, and optionally test.npz and "
            "test_natural.npz. Also determines the tuning "
            "output directory unless --output-dir or "
            "--experiment-name is provided."
        ),
    )
    parser.add_argument(
        "--train-dataset",
        type=Path,
        default=None,
        help=(
            "Explicit training split path. Prefer --dataset for "
            "standard dataset bundles."
        ),
    )
    parser.add_argument(
        "--validation-dataset",
        type=Path,
        default=None,
        help=(
            "Explicit validation split path used for selecting the "
            "best hyperparameter combination. Prefer --dataset for "
            "standard dataset bundles."
        ),
    )
    parser.add_argument(
        "--same-layout-dataset",
        type=Path,
        default=None,
        help=(
            "Explicit same-layout new-task split path to monitor "
            "beside validation performance. Prefer --dataset for "
            "standard dataset bundles."
        ),
    )
    parser.add_argument(
        "--natural-validation-dataset",
        type=Path,
        default=None,
        help=(
            "Explicit natural-layout validation split to monitor "
            "beside the primary val split. Prefer --dataset for "
            "standard dataset bundles."
        ),
    )
    parser.add_argument(
        "--test-dataset",
        type=Path,
        default=None,
        help=(
            "Explicit test split path. Used only with "
            "--evaluate-best-on-test. Prefer --dataset for standard "
            "dataset bundles."
        ),
    )
    parser.add_argument(
        "--natural-test-dataset",
        type=Path,
        default=None,
        help=(
            "Explicit natural-layout test split path. Used with "
            "--evaluate-best-on-test when provided."
        ),
    )
    parser.add_argument(
        "--evaluate-best-on-test",
        action="store_true",
        help=(
            "After tuning, evaluate the best validation "
            "hyperparameter combination on the held-out test "
            "dataset."
        ),
    )
    parser.add_argument(
        "--search-space",
        type=Path,
        default=None,
        help=(
            "Optional JSON search-space override for the selected "
            "algorithm."
        ),
    )
    parser.add_argument(
        "--validation-all-tasks",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Evaluate every validation task during tuning child "
            "runs. This is the default; use "
            "--no-validation-all-tasks to sample validation tasks."
        ),
    )
    parser.add_argument(
        "--q-snapshot-count",
        type=int,
        default=11,
        help=(
            "Number of evenly spaced model snapshots to save in "
            "each tuned checkpoint. Use 0 to disable snapshots."
        ),
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=35,
        help=(
            "Shared patience for stopping hyperparameter "
            "combinations after validation stalls."
        ),
    )
    parser.add_argument(
        "--early-stopping-min-delta",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--tensorboard",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Write TensorBoard event logs for tuning child runs. "
            "Defaults off to keep random-search sweeps lightweight."
        ),
    )
    parser.add_argument(
        "--keep-plots",
        action="store_true",
        help=(
            "Keep per-combination training and evaluation plots."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Skip completed hyperparameter combinations already "
            "recorded in tuning_results.csv and recover complete "
            "combination artifacts that were written before the "
            "tuner was interrupted. Incomplete combinations are "
            "continued from training_state.pt when available, "
            "otherwise rerun with their original sampled seed and "
            "hyperparameters. Existing tuning history uses the "
            "stored tuning arguments except explicit CLI "
            "overrides such as --hyperparameter-combinations "
            "and --[no-]tensorboard."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Print hyperparameter-combination commands without "
            "running them."
        ),
    )

    args = parser.parse_args(argv)
    args.hyperparameter_combinations_from_cli = (
        args.hyperparameter_combinations is not None
    )
    args.tensorboard_from_cli = args.tensorboard is not None

    if args.hyperparameter_combinations is None:
        args.hyperparameter_combinations = 20

    if args.tensorboard is None:
        args.tensorboard = False

    return args


def _json_scalar(value):
    if hasattr(value, "item"):
        return value.item()

    return value


def sample_from_spec(
    spec: dict[str, Any],
    rng: np.random.Generator,
):
    distribution = spec["type"]

    if distribution == "choice":
        values = spec["values"]
        index = int(
            rng.integers(len(values))
        )
        return _json_scalar(values[index])

    if distribution == "uniform":
        return float(
            rng.uniform(
                float(spec["low"]),
                float(spec["high"]),
            )
        )

    if distribution == "loguniform":
        low = float(spec["low"])
        high = float(spec["high"])

        if low <= 0 or high <= 0:
            raise ValueError(
                "loguniform bounds must be positive."
            )

        return float(
            math.exp(
                rng.uniform(
                    math.log(low),
                    math.log(high),
                )
            )
        )

    if distribution == "int":
        return int(
            rng.integers(
                int(spec["low"]),
                int(spec["high"]) + 1,
            )
        )

    raise ValueError(
        f"Unknown distribution type: {distribution}"
    )


def validate_search_space(
    algorithm: str,
    search_space: dict[str, dict[str, Any]],
) -> None:
    if algorithm != "grpo":
        return

    spec = search_space.get("rollout_group_size")
    if spec is None:
        return

    distribution = spec.get("type")
    minimum = MIN_GRPO_ROLLOUT_GROUP_SIZE

    if distribution == "choice":
        values = spec.get("values", [])
        if any(
            int(value) < minimum
            for value in values
        ):
            raise ValueError(
                "grpo rollout_group_size choices must all "
                f"be at least {minimum}."
            )
        return

    if distribution in {"int", "uniform", "loguniform"}:
        if float(spec["low"]) < minimum:
            raise ValueError(
                "grpo rollout_group_size lower bound must "
                f"be at least {minimum}."
            )
        return


def load_search_space(
    algorithm: str,
    path: Path | None,
) -> dict[str, dict[str, Any]]:
    if path is None:
        search_space = DEFAULT_SEARCH_SPACES[
            algorithm
        ]
        validate_search_space(
            algorithm,
            search_space,
        )
        return search_space

    with path.open() as file:
        data = json.load(file)

    if algorithm in data:
        data = data[algorithm]

    if not isinstance(data, dict):
        raise ValueError(
            "Search space JSON must contain a parameter mapping."
        )

    data = {
        TRAINING_LOOP_PARAMETER_ALIASES.get(
            name,
            name,
        ): spec
        for name, spec in data.items()
    }

    valid_parameters = set(
        NEURAL_HYPERPARAMETERS[algorithm]
    )
    valid_parameters.update(
        TRAINING_LOOP_PARAMETERS
    )

    unsupported = sorted(
        set(data) - valid_parameters
    )

    if unsupported:
        names = ", ".join(unsupported)
        raise ValueError(
            f"{algorithm} search space has unsupported "
            f"parameters: {names}"
        )

    validate_search_space(
        algorithm,
        data,
    )

    return data


def sample_hyperparameters(
    search_space: dict[str, dict[str, Any]],
    rng: np.random.Generator,
) -> dict[str, float | int]:
    return {
        name: sample_from_spec(
            spec,
            rng,
        )
        for name, spec in search_space.items()
    }


def sample_from_optuna_spec(
    name: str,
    spec: dict[str, Any],
    trial: Any,
):
    distribution = spec["type"]

    if distribution == "choice":
        return _json_scalar(
            trial.suggest_categorical(
                name,
                list(spec["values"]),
            )
        )

    if distribution == "uniform":
        return float(
            trial.suggest_float(
                name,
                float(spec["low"]),
                float(spec["high"]),
            )
        )

    if distribution == "loguniform":
        low = float(spec["low"])
        high = float(spec["high"])

        if low <= 0 or high <= 0:
            raise ValueError(
                "loguniform bounds must be positive."
            )

        return float(
            trial.suggest_float(
                name,
                low,
                high,
                log=True,
            )
        )

    if distribution == "int":
        return int(
            trial.suggest_int(
                name,
                int(spec["low"]),
                int(spec["high"]),
            )
        )

    raise ValueError(
        f"Unknown distribution type: {distribution}"
    )


def sample_hyperparameters_with_optuna(
    search_space: dict[str, dict[str, Any]],
    trial: Any,
) -> dict[str, float | int]:
    return {
        name: sample_from_optuna_spec(
            name,
            spec,
            trial,
        )
        for name, spec in search_space.items()
    }


def create_optuna_study(
    *,
    seed: int,
):
    try:
        import optuna
    except ImportError as exc:
        raise RuntimeError(
            "Optuna tuning requires the optional optuna "
            "dependency. Install it with "
            "`pip install 'maze-rl[tuning]'` or "
            "`pip install optuna`, then rerun with "
            "--sampler optuna."
        ) from exc

    return optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(
            seed=seed,
        ),
    )


def ordered_search_space(
    algorithm: str,
    search_space: dict[str, dict[str, Any]],
    parameter_order: list[str] | None = None,
) -> dict[str, dict[str, Any]]:
    if parameter_order is None:
        parameter_order = [
            name
            for name in DEFAULT_SEARCH_SPACES.get(
                algorithm,
                {},
            )
            if name in search_space
        ]

    ordered = {
        name: search_space[name]
        for name in parameter_order
        if name in search_space
    }

    ordered.update(
        {
            name: spec
            for name, spec in search_space.items()
            if name not in ordered
        }
    )

    return ordered


def serialize_tuning_argument(
    name: str,
    value: Any,
) -> Any:
    if name in TUNING_PATH_ARGUMENT_FIELDS:
        return None if value is None else str(value)

    return value


def deserialize_tuning_argument(
    name: str,
    value: Any,
) -> Any:
    if name in TUNING_PATH_ARGUMENT_FIELDS:
        return None if value is None else Path(value)

    return value


def build_tuning_config(
    args: argparse.Namespace,
    search_space: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    return {
        "version": 1,
        "arguments": {
            name: serialize_tuning_argument(
                name,
                getattr(
                    args,
                    name,
                    TUNING_OPTIONAL_ARGUMENT_DEFAULTS.get(
                        name
                    ),
                ),
            )
            for name in TUNING_ARGUMENT_FIELDS
            if hasattr(
                args,
                name,
            )
            or name in TUNING_OPTIONAL_ARGUMENT_DEFAULTS
        },
        "search_space": search_space,
        "search_space_parameter_order": list(
            search_space
        ),
    }


def write_tuning_config(
    path: Path,
    args: argparse.Namespace,
    search_space: dict[str, dict[str, Any]],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open("w") as file:
        json.dump(
            build_tuning_config(
                args,
                search_space,
            ),
            file,
            indent=2,
        )


def read_tuning_config(
    path: Path,
) -> dict[str, Any]:
    with path.open() as file:
        config = json.load(file)

    if "arguments" not in config:
        raise ValueError(
            f"{path} does not contain tuning arguments."
        )

    if "search_space" not in config:
        raise ValueError(
            f"{path} does not contain a search space."
        )

    return config


def tuning_config_path_for(
    output_dir: Path,
    algorithm: str,
) -> Path:
    return (
        output_dir
        / TUNING_CONFIG_DIRNAME
        / f"{algorithm}.json"
    )


def legacy_tuning_config_path(
    output_dir: Path,
) -> Path:
    return output_dir / LEGACY_TUNING_CONFIG_FILENAME


def tuning_config_algorithm(
    config: dict[str, Any],
) -> str | None:
    arguments = config.get(
        "arguments",
        {},
    )

    return arguments.get("algorithm")


def find_tuning_config_path(
    output_dir: Path,
    algorithm: str,
) -> Path | None:
    config_path = tuning_config_path_for(
        output_dir,
        algorithm,
    )

    if config_path.exists():
        return config_path

    legacy_config_path = legacy_tuning_config_path(
        output_dir
    )

    if not legacy_config_path.exists():
        return None

    legacy_config = read_tuning_config(
        legacy_config_path
    )

    if (
        tuning_config_algorithm(legacy_config)
        == algorithm
    ):
        return legacy_config_path

    return None


def apply_tuning_config(
    args: argparse.Namespace,
    config: dict[str, Any],
) -> None:
    arguments = config["arguments"]

    for name in TUNING_ARGUMENT_FIELDS:
        if (
            name == "hyperparameter_combinations"
            and getattr(
                args,
                "hyperparameter_combinations_from_cli",
                False,
            )
        ):
            continue

        if (
            name == "tensorboard"
            and getattr(
                args,
                "tensorboard_from_cli",
                False,
            )
        ):
            continue

        argument_name = name

        if (
            name == "hyperparameter_combinations"
            and name not in arguments
            and "trials" in arguments
        ):
            argument_name = "trials"

        if argument_name not in arguments:
            if name in TUNING_OPTIONAL_ARGUMENT_DEFAULTS:
                setattr(
                    args,
                    name,
                    TUNING_OPTIONAL_ARGUMENT_DEFAULTS[name],
                )
                continue

            raise ValueError(
                "Stored tuning config is missing argument "
                f"{name!r}."
            )

        setattr(
            args,
            name,
            deserialize_tuning_argument(
                name,
                arguments[argument_name],
            ),
        )


def tuning_results_has_algorithm(
    path: Path,
    algorithm: str,
) -> bool:
    if not path.exists():
        return False

    with path.open(
        newline="",
    ) as file:
        reader = csv.DictReader(file)
        return any(
            row.get("algorithm") == algorithm
            for row in reader
        )


def tuning_trial_artifacts_exist(
    output_dir: Path,
    algorithm: str,
) -> bool:
    trials_dir = output_dir / "trials"

    if not trials_dir.exists():
        return False

    return any(
        (trial_dir / algorithm).exists()
        for trial_dir in trials_dir.iterdir()
        if trial_dir.is_dir()
    )


def has_tuning_history(
    *,
    output_dir: Path,
    results_path: Path,
    algorithm: str,
) -> bool:
    if (
        find_tuning_config_path(
            output_dir,
            algorithm,
        )
        is not None
    ):
        return True

    if tuning_results_has_algorithm(
        results_path,
        algorithm,
    ):
        return True

    return tuning_trial_artifacts_exist(
        output_dir,
        algorithm,
    )


def cli_name(
    parameter_name: str,
) -> str:
    return "--" + parameter_name.replace(
        "_",
        "-",
    )


def cli_value(
    value: float | int,
) -> str:
    if isinstance(value, float):
        return f"{value:.12g}"

    return str(value)


def build_training_command(
    *,
    args: argparse.Namespace,
    trial_dir: Path,
    train_dataset: Path,
    trial_seed: int,
    hyperparameters: dict[str, float | int],
) -> list[str]:
    command = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--algorithm",
        args.algorithm,
        "--dataset",
        str(train_dataset),
        "--rollouts-per-task",
        str(args.rollouts_per_task),
        "--max-steps",
        str(args.max_steps),
        "--validation-dataset",
        str(args.validation_dataset),
        "--same-layout-dataset",
        str(args.same_layout_dataset),
        "--seed",
        str(trial_seed),
        "--output-dir",
        str(trial_dir),
        "--q-snapshot-count",
        str(args.q_snapshot_count),
    ]

    natural_validation_dataset = getattr(
        args,
        "natural_validation_dataset",
        None,
    )
    if natural_validation_dataset is not None:
        command.extend(
            [
                "--natural-validation-dataset",
                str(natural_validation_dataset),
            ]
        )

    if args.tensorboard:
        command.append("--tensorboard")
    else:
        command.append("--no-tensorboard")

    training_state_path = (
        trial_dir
        / args.algorithm
        / "training_state.pt"
    )
    if (
        getattr(
            args,
            "resume",
            False,
        )
        and training_state_path.exists()
    ):
        command.append(
            "--resume"
        )

    if args.keep_plots:
        command.append("--plot")
    else:
        command.append("--no-plot")

    if not getattr(
        args,
        "validation_all_tasks",
        True,
    ):
        command.append("--no-validation-all-tasks")

    if args.early_stopping_patience is not None:
        command.extend(
            [
                "--early-stopping-patience",
                str(args.early_stopping_patience),
            ]
        )

    if args.early_stopping_min_delta != 0.0:
        command.extend(
            [
                "--early-stopping-min-delta",
                cli_value(
                    args.early_stopping_min_delta
                ),
            ]
        )

    for name, value in sorted(
        hyperparameters.items()
    ):
        if name in TRAINING_LOOP_PARAMETERS:
            continue

        command.extend(
            [
                cli_name(name),
                cli_value(value),
            ]
        )

    if "task_batch_size" in hyperparameters:
        command.extend(
            [
                "--task-batch-size",
                cli_value(
                    hyperparameters[
                        "task_batch_size"
                    ]
                ),
            ]
        )

    if "rollout_group_size" in hyperparameters:
        command.extend(
            [
                "--rollout-group-size",
                cli_value(
                    hyperparameters[
                        "rollout_group_size"
                    ]
                ),
            ]
        )

    return command


def default_experiment_tuning_output_dir(
    *,
    experiment_name: str,
    train_dataset: Path,
    experiment_root: Path = DEFAULT_DNN_TUNING_OUTPUT_ROOT,
) -> Path:
    return (
        experiment_root
        / experiment_dataset_name_for_path(
            experiment_name,
            train_dataset,
        )
    )


def dataset_bundle_tuning_output_dir(
    *,
    dataset_dir: Path,
    experiment_root: Path = DEFAULT_DNN_TUNING_OUTPUT_ROOT,
) -> Path:
    return (
        experiment_root
        / dataset_dir.name
    )


def dataset_bundle_dir_for_reference(
    dataset: Path,
) -> Path:
    if dataset.is_absolute() or len(dataset.parts) > 1:
        return dataset

    return DEFAULT_DATA_OUTPUT_ROOT / dataset


def resolve_dataset_bundle_args(
    args: argparse.Namespace,
) -> None:
    if args.dataset is None:
        return

    dataset_dir = dataset_bundle_dir_for_reference(
        Path(args.dataset)
    )

    if not dataset_dir.is_dir():
        raise ValueError(
            "--dataset must name a dataset bundle under "
            f"{DEFAULT_DATA_OUTPUT_ROOT} containing train.npz: "
            f"{args.dataset}"
        )

    train_dataset = dataset_dir / "train.npz"
    validation_dataset = dataset_dir / "validation.npz"
    validation_tune_dataset = dataset_dir / "validation_tune.npz"
    same_layout_dataset = (
        dataset_dir / "same_layout_new_goals.npz"
    )
    natural_validation_dataset = (
        dataset_dir / "validation_natural.npz"
    )
    test_dataset = dataset_dir / "test.npz"
    natural_test_dataset = dataset_dir / "test_natural.npz"

    if not train_dataset.exists():
        raise ValueError(
            "Dataset directory must contain train.npz: "
            f"{dataset_dir}"
        )

    if args.train_dataset is None:
        args.train_dataset = train_dataset

    if args.validation_dataset is None:
        args.validation_dataset = (
            validation_tune_dataset
            if validation_tune_dataset.exists()
            else validation_dataset
        )

    if args.same_layout_dataset is None:
        args.same_layout_dataset = same_layout_dataset

    if getattr(args, "natural_validation_dataset", None) is None:
        if natural_validation_dataset.exists():
            args.natural_validation_dataset = (
                natural_validation_dataset
            )

    if args.test_dataset is None and test_dataset.exists():
        args.test_dataset = test_dataset

    if (
        getattr(args, "natural_test_dataset", None) is None
        and natural_test_dataset.exists()
    ):
        args.natural_test_dataset = natural_test_dataset

    if (
        args.output_dir is None
        and args.experiment_name is None
    ):
        args.output_dir = dataset_bundle_tuning_output_dir(
            dataset_dir=dataset_dir,
            experiment_root=args.experiment_root,
        )


def validate_tuning_paths(
    args: argparse.Namespace,
) -> None:
    missing_options = [
        option
        for option, value in [
            ("--train-dataset", args.train_dataset),
            (
                "--validation-dataset",
                args.validation_dataset,
            ),
            (
                "--same-layout-dataset",
                args.same_layout_dataset,
            ),
        ]
        if value is None
    ]

    if missing_options:
        raise ValueError(
            "Pass --dataset or explicit split paths for "
            + ", ".join(missing_options)
            + "."
        )

    if (
        args.evaluate_best_on_test
        and args.test_dataset is None
    ):
        raise ValueError(
            "--evaluate-best-on-test requires --dataset with "
            "test.npz or an explicit --test-dataset."
        )

    if args.output_dir is None:
        raise ValueError(
            "Pass --dataset, --output-dir, or --experiment-name "
            "so tuning has an explicit output directory."
        )


def build_evaluation_command(
    *,
    args: argparse.Namespace,
    checkpoint_path: Path,
    validation_dataset: Path,
    evaluation_path: Path,
    trial_seed: int,
) -> list[str]:
    command = [
        sys.executable,
        str(EVALUATE_SCRIPT),
        "--algorithm",
        args.algorithm,
        "--checkpoint",
        str(checkpoint_path),
        "--dataset",
        str(validation_dataset),
        "--max-steps",
        str(args.max_steps),
        "--seed",
        str(trial_seed),
        "--evaluation-output",
        str(evaluation_path),
        "--no-q-plots",
        "--no-q-videos",
        "--no-rollout-animations",
    ]

    if not args.keep_plots:
        command.append("--no-plot")

    command.append("--all-tasks")

    return command


def command_forward_signals() -> list[int]:
    signals = [
        signal.SIGINT,
        signal.SIGTERM,
    ]

    if hasattr(
        signal,
        "SIGHUP",
    ):
        signals.append(signal.SIGHUP)

    return signals


def _terminate_child_process(
    process: subprocess.Popen,
    *,
    initial_signal: int = signal.SIGTERM,
    timeout: float = COMMAND_TERMINATION_TIMEOUT_SECONDS,
) -> None:
    if process.poll() is not None:
        return

    try:
        if os.name == "posix":
            os.killpg(
                process.pid,
                initial_signal,
            )
        else:
            process.terminate()
    except ProcessLookupError:
        return

    try:
        process.wait(timeout=timeout)
        return
    except subprocess.TimeoutExpired:
        pass

    try:
        if os.name == "posix":
            os.killpg(
                process.pid,
                signal.SIGKILL,
            )
        else:
            process.kill()
    except ProcessLookupError:
        return

    process.wait()


def run_command(
    command: list[str],
    dry_run: bool,
) -> None:
    print(
        " ".join(command),
        flush=True,
    )

    if dry_run:
        return

    env = os.environ.copy()
    pythonpath_parts = [
        str(REPO_ROOT / "src"),
        str(REPO_ROOT),
    ]

    if env.get("PYTHONPATH"):
        pythonpath_parts.append(
            env["PYTHONPATH"]
        )

    env["PYTHONPATH"] = os.pathsep.join(
        pythonpath_parts
    )

    process = subprocess.Popen(
        command,
        cwd=REPO_ROOT,
        env=env,
        start_new_session=os.name == "posix",
    )
    previous_handlers = {}

    def handle_signal(
        signum,
        frame,
    ):
        raise CommandInterrupted(signum)

    for signum in command_forward_signals():
        previous_handlers[signum] = signal.getsignal(
            signum
        )
        signal.signal(
            signum,
            handle_signal,
        )

    try:
        returncode = process.wait()
    except CommandInterrupted as exc:
        _terminate_child_process(
            process,
            initial_signal=exc.signum,
        )
        raise SystemExit(128 + exc.signum) from None
    except KeyboardInterrupt:
        _terminate_child_process(
            process,
            initial_signal=signal.SIGINT,
        )
        raise SystemExit(128 + signal.SIGINT) from None
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(
                signum,
                handler,
            )

    if returncode != 0:
        raise subprocess.CalledProcessError(
            returncode,
            command,
        )


def summarize_evaluation(
    path: Path,
) -> dict[str, float]:
    with path.open(
        newline="",
    ) as file:
        rows = list(
            csv.DictReader(file)
        )

    if not rows:
        raise RuntimeError(
            f"No evaluation rows written to {path}"
        )

    returns = [
        float(row["episode_return"])
        for row in rows
    ]
    successes = [
        row["success"] == "True"
        for row in rows
    ]
    efficiencies = [
        float(row["path_efficiency"])
        for row in rows
    ]
    successful_efficiencies = [
        efficiency
        for success, efficiency in zip(
            successes,
            efficiencies,
        )
        if success
    ]
    weighted_rows = [
        (
            float(row["optimal_path_length"]),
            success,
            efficiency,
        )
        for row, success, efficiency in zip(
            rows,
            successes,
            efficiencies,
        )
        if row.get("optimal_path_length") not in {
            None,
            "",
        }
        and float(row["optimal_path_length"]) > 0
    ]
    weighted_path_length_total = sum(
        length
        for length, _, _ in weighted_rows
    )
    weighted_path_success = 0.0
    weighted_path_efficiency = 0.0
    if weighted_path_length_total > 0:
        weighted_path_success = sum(
            length
            for length, success, _ in weighted_rows
            if success
        ) / weighted_path_length_total
        weighted_path_efficiency = sum(
            length * efficiency
            for length, _, efficiency in weighted_rows
        ) / weighted_path_length_total

    return {
        "episodes": float(len(rows)),
        "success_rate": float(
            np.mean(successes)
        ),
        "average_episode_return": float(
            np.mean(returns)
        ),
        "mean_path_efficiency": float(
            np.mean(efficiencies)
        ),
        "average_successful_path_efficiency": (
            float(
                np.mean(
                    successful_efficiencies
                )
            )
            if successful_efficiencies
            else 0.0
        ),
        "weighted_path_success": float(
            weighted_path_success
        ),
        "weighted_path_efficiency": float(
            weighted_path_efficiency
        ),
    }


def read_training_summary(
    path: Path,
) -> dict[str, Any]:
    with path.open() as file:
        return json.load(file)


def _csv_int(
    value: str | None,
) -> int | None:
    if value in {
        None,
        "",
    }:
        return None

    return int(value)


def _csv_float(
    value: str | None,
) -> float | None:
    if value in {
        None,
        "",
    }:
        return None

    return float(value)


def _csv_bool(
    value: str | None,
) -> bool:
    return value == "True"


def parse_result_row(
    row: dict[str, str],
) -> dict[str, Any]:
    return {
        "trial": int(row["trial"]),
        "algorithm": row["algorithm"],
        "seed": int(row["seed"]),
        "objective": float(row["objective"]),
        "mean_path_efficiency": float(
            row["mean_path_efficiency"]
        ),
        "success_rate": float(row["success_rate"]),
        "average_episode_return": float(
            row["average_episode_return"]
        ),
        "average_successful_path_efficiency": float(
            row[
                "average_successful_path_efficiency"
            ]
        ),
        "weighted_path_success": _csv_float(
            row.get("weighted_path_success")
        ),
        "weighted_path_efficiency": _csv_float(
            row.get("weighted_path_efficiency")
        ),
        "rollouts_per_task_requested": _csv_int(
            row.get(
                "rollouts_per_task_requested"
            )
        ),
        "task_batch_size": _csv_int(
            row.get("task_batch_size")
        ),
        "rollout_group_size": _csv_int(
            row.get("rollout_group_size")
        ),
        "task_selection_passes_requested": _csv_int(
            row.get(
                "task_selection_passes_requested"
            )
        ),
        "task_selection_passes_completed": _csv_int(
            row.get(
                "task_selection_passes_completed"
            )
        ),
        "dataset_epochs_completed": _csv_int(
            row.get(
                "dataset_epochs_completed"
            )
        ),
        "stopped_early": _csv_bool(
            row.get("stopped_early")
        ),
        "checkpoint_path": row["checkpoint_path"],
        "evaluation_path": row["evaluation_path"],
        "hyperparameters": json.loads(
            row["hyperparameters"]
        ),
    }


def result_artifacts_exist(
    result: dict[str, Any],
) -> bool:
    return (
        Path(result["checkpoint_path"]).exists()
        and Path(result["evaluation_path"]).exists()
    )


def result_training_summary_path(
    result: dict[str, Any],
) -> Path:
    return (
        Path(result["checkpoint_path"]).parent
        / "training_summary.json"
    )


def result_activity_paths(
    result: dict[str, Any],
) -> list[Path]:
    run_dir = Path(result["checkpoint_path"]).parent
    paths = [
        run_dir / "checkpoint.pt",
        run_dir / "best_checkpoint.pt",
        run_dir / "checkpoint.pkl",
        run_dir / "best_checkpoint.pkl",
        run_dir / "metrics.csv",
        run_dir / "validation_metrics.csv",
    ]

    return [
        path
        for path in paths
        if path.exists()
    ]


def result_has_stale_completion_marker(
    result: dict[str, Any],
    *,
    training_summary_path: Path,
) -> bool:
    evaluation_path = Path(
        result["evaluation_path"]
    )
    completion_mtime_ns = max(
        training_summary_path.stat().st_mtime_ns,
        evaluation_path.stat().st_mtime_ns,
    )

    return any(
        path.stat().st_mtime_ns > completion_mtime_ns
        for path in result_activity_paths(result)
    )


def training_summary_completed(
    training_summary: dict[str, Any],
) -> bool:
    status = training_summary.get("status")

    if status in {
        "completed",
        "early_stopped",
    }:
        return True

    if status in {
        "running",
        "interrupted",
    }:
        return False

    if status is not None:
        return False

    if training_summary.get("stopped_early") is True:
        return True

    completed = training_summary.get(
        "task_selection_passes_completed",
        training_summary.get(
            "dataset_epochs_completed"
        ),
    )
    requested = training_summary.get(
        "task_selection_passes_requested",
        training_summary.get(
            "dataset_epochs_requested"
        ),
    )

    if completed is None or requested is None:
        return False

    try:
        return int(completed) >= int(requested)
    except (TypeError, ValueError):
        return False


def result_artifacts_match(
    result: dict[str, Any],
) -> bool:
    if not result_artifacts_exist(result):
        return False

    training_summary_path = result_training_summary_path(
        result
    )

    if not training_summary_path.exists():
        return False

    if result_has_stale_completion_marker(
        result,
        training_summary_path=training_summary_path,
    ):
        return False

    try:
        training_summary = read_training_summary(
            training_summary_path
        )
    except (OSError, json.JSONDecodeError):
        return False

    best_validation = training_summary.get(
        "best_validation"
    )

    if not isinstance(best_validation, dict):
        return False

    if not training_summary_completed(
        training_summary
    ):
        return False

    try:
        result_objective = float(
            result["objective"]
        )
        artifact_objective = float(
            best_validation["mean_path_efficiency"]
        )
    except (KeyError, TypeError, ValueError):
        return False

    if not math.isclose(
        result_objective,
        artifact_objective,
        rel_tol=1e-12,
        abs_tol=1e-15,
    ):
        return False

    artifact_summary_hyperparameters = (
        training_summary.get(
            "hyperparameters",
            {},
        )
    )

    if not isinstance(
        artifact_summary_hyperparameters,
        dict,
    ):
        return False

    artifact_hyperparameters = {
        **artifact_summary_hyperparameters,
        "task_batch_size": training_summary.get(
            "task_batch_size"
        ),
        "rollout_group_size": training_summary.get(
            "rollout_group_size"
        ),
    }

    return hyperparameters_match(
        result["hyperparameters"],
        artifact_hyperparameters,
    )


def read_completed_results(
    path: Path,
    algorithm: str,
) -> dict[int, dict[str, Any]]:
    if not path.exists():
        return {}

    completed = {}

    with path.open(
        newline="",
    ) as file:
        reader = csv.DictReader(file)

        for row in reader:
            if row.get("algorithm") != algorithm:
                continue

            result = parse_result_row(row)

            if result_artifacts_match(result):
                completed[
                    result["trial"]
                ] = result

    return completed


def comparable_hyperparameter_value(
    value: Any,
) -> int | float:
    if isinstance(value, bool):
        return int(value)

    if isinstance(value, int):
        return value

    return float(cli_value(float(value)))


def hyperparameters_match(
    expected: dict[str, float | int],
    actual: dict[str, Any],
) -> bool:
    for name, expected_value in expected.items():
        if name not in actual:
            return False

        if actual[name] is None:
            return False

        expected_comparable = (
            comparable_hyperparameter_value(
                expected_value
            )
        )
        actual_comparable = (
            comparable_hyperparameter_value(
                actual[name]
            )
        )

        if isinstance(
            expected_comparable,
            int,
        ) and isinstance(
            actual_comparable,
            int,
        ):
            if (
                expected_comparable
                != actual_comparable
            ):
                return False
            continue

        if not math.isclose(
            float(expected_comparable),
            float(actual_comparable),
            rel_tol=1e-12,
            abs_tol=1e-15,
        ):
            return False

    return True


def result_matches_trial(
    result: dict[str, Any],
    *,
    trial_seed: int,
    hyperparameters: dict[str, float | int],
) -> bool:
    return (
        result["seed"] == trial_seed
        and hyperparameters_match(
            hyperparameters,
            result["hyperparameters"],
        )
    )


def update_best_result(
    best_result: dict[str, Any] | None,
    result: dict[str, Any],
) -> dict[str, Any]:
    if (
        best_result is None
        or result["objective"]
        > best_result["objective"]
    ):
        return result

    return best_result


def build_result_from_artifacts(
    *,
    args: argparse.Namespace,
    trial: int,
    trial_seed: int,
    checkpoint_path: Path,
    best_checkpoint_path: Path,
    training_summary_path: Path,
    evaluation_path: Path,
) -> dict[str, Any] | None:
    if (
        not training_summary_path.exists()
        or not evaluation_path.exists()
    ):
        return None

    selected_checkpoint_path = (
        best_checkpoint_path
        if best_checkpoint_path.exists()
        else checkpoint_path
    )

    if not selected_checkpoint_path.exists():
        return None

    if result_has_stale_completion_marker(
        {
            "checkpoint_path": str(
                selected_checkpoint_path
            ),
            "evaluation_path": str(
                evaluation_path
            ),
        },
        training_summary_path=training_summary_path,
    ):
        return None

    try:
        training_summary = read_training_summary(
            training_summary_path
        )
    except (OSError, json.JSONDecodeError):
        return None

    if not isinstance(
        training_summary,
        dict,
    ) or "best_validation" not in training_summary:
        return None

    try:
        summarize_evaluation(
            evaluation_path
        )
    except (
        OSError,
        RuntimeError,
        KeyError,
        TypeError,
        ValueError,
    ):
        return None

    try:
        best_validation = training_summary[
            "best_validation"
        ]
        if not isinstance(
            best_validation,
            dict,
        ):
            return None
        if not training_summary_completed(
            training_summary
        ):
            return None
        objective = best_validation[
            "mean_path_efficiency"
        ]
        success_rate = best_validation[
            "success_rate"
        ]
        average_episode_return = best_validation[
            "average_episode_return"
        ]
        average_successful_path_efficiency = (
            best_validation[
                "average_successful_path_efficiency"
            ]
        )
        weighted_path_success = best_validation.get(
            "weighted_path_success"
        )
        weighted_path_efficiency = best_validation.get(
            "weighted_path_efficiency"
        )
        dataset_epochs_completed = training_summary[
            "dataset_epochs_completed"
        ]
        stopped_early = training_summary[
            "stopped_early"
        ]
        task_selection_passes_completed = (
            training_summary.get(
                "task_selection_passes_completed",
                dataset_epochs_completed,
            )
        )
        hyperparameters = training_summary.get(
            "hyperparameters",
            {},
        )
        result_hyperparameters = {
            **hyperparameters,
            "task_batch_size": training_summary.get(
                "task_batch_size"
            ),
            "rollout_group_size": training_summary.get(
                "rollout_group_size"
            ),
        }
    except (
        KeyError,
        TypeError,
    ):
        return None

    return {
        "trial": trial,
        "algorithm": args.algorithm,
        "seed": trial_seed,
        "objective": objective,
        "mean_path_efficiency": objective,
        "success_rate": success_rate,
        "average_episode_return": average_episode_return,
        "average_successful_path_efficiency": (
            average_successful_path_efficiency
        ),
        "weighted_path_success": weighted_path_success,
        "weighted_path_efficiency": weighted_path_efficiency,
        "rollouts_per_task_requested": (
            training_summary.get(
                "rollouts_per_task_requested"
            )
        ),
        "task_batch_size": training_summary.get(
            "task_batch_size"
        ),
        "rollout_group_size": training_summary.get(
            "rollout_group_size"
        ),
        "task_selection_passes_requested": (
            training_summary.get(
                "task_selection_passes_requested"
            )
        ),
        "task_selection_passes_completed": (
            task_selection_passes_completed
        ),
        "dataset_epochs_completed": (
            dataset_epochs_completed
        ),
        "stopped_early": stopped_early,
        "checkpoint_path": str(
            selected_checkpoint_path
        ),
        "evaluation_path": str(
            evaluation_path
        ),
        "hyperparameters": result_hyperparameters,
    }


def write_result_row(
    path: Path,
    row: dict[str, Any],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_header = not path.exists()

    with path.open(
        "a+",
        newline="",
    ) as file:
        if not write_header:
            file.seek(0)
            existing_fieldnames = file.readline().strip().split(",")

            if existing_fieldnames != RESULT_FIELDNAMES:
                if not all(
                    field in RESULT_FIELDNAMES
                    for field in existing_fieldnames
                ):
                    raise RuntimeError(
                        "Existing tuning results file has a different "
                        f"schema: {path}"
                    )

                file.seek(0)
                existing_rows = list(csv.DictReader(file))
                file.seek(0)
                file.truncate()

                writer = csv.DictWriter(
                    file,
                    fieldnames=RESULT_FIELDNAMES,
                )
                writer.writeheader()
                for existing_row in existing_rows:
                    writer.writerow(
                        {
                            field: existing_row.get(field, "")
                            for field in RESULT_FIELDNAMES
                        }
                    )

        file.seek(0, 2)
        writer = csv.DictWriter(
            file,
            fieldnames=RESULT_FIELDNAMES,
        )

        if write_header:
            writer.writeheader()

        writer.writerow(
            {
                **row,
                "hyperparameters": json.dumps(
                    row["hyperparameters"],
                    sort_keys=True,
                ),
            }
        )


def write_best_config(
    path: Path,
    best_result: dict[str, Any],
) -> None:
    with path.open("w") as file:
        json.dump(
            best_result,
            file,
            indent=2,
            sort_keys=True,
        )


def write_best_config_index(
    path: Path,
    algorithm: str,
    best_result: dict[str, Any],
) -> None:
    if path.exists():
        with path.open() as file:
            existing = json.load(file)
    else:
        existing = {}

    if (
        isinstance(existing, dict)
        and "algorithm" in existing
    ):
        existing = {
            existing["algorithm"]: existing
        }

    if not isinstance(existing, dict):
        existing = {}

    existing[algorithm] = best_result

    with path.open("w") as file:
        json.dump(
            existing,
            file,
            indent=2,
            sort_keys=True,
        )


def best_config_path_for_algorithm(
    output_dir: Path,
    algorithm: str,
) -> Path:
    return output_dir / f"best_config_{algorithm}.json"


def write_best_configs(
    *,
    best_config_path: Path,
    algorithm_best_config_path: Path,
    algorithm: str,
    best_result: dict[str, Any],
) -> None:
    write_best_config(
        algorithm_best_config_path,
        best_result,
    )
    write_best_config_index(
        best_config_path,
        algorithm,
        best_result,
    )


def main() -> None:
    args = parse_args()


    resolve_dataset_bundle_args(args)

    if args.output_dir is None:
        if args.experiment_name is None:
            raise ValueError(
                "Pass --dataset, --output-dir, or "
                "--experiment-name so tuning has an explicit "
                "output directory."
            )

        if args.train_dataset is None:
            raise ValueError(
                "--experiment-name requires --dataset or "
                "--train-dataset so the output directory can be "
                "derived from dataset metadata."
            )

        args.output_dir = default_experiment_tuning_output_dir(
            experiment_name=args.experiment_name,
            train_dataset=args.train_dataset,
            experiment_root=args.experiment_root,
        )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    results_path = (
        args.output_dir / "tuning_results.csv"
    )
    best_config_path = (
        args.output_dir / "best_config.json"
    )
    tuning_config_path = find_tuning_config_path(
        args.output_dir,
        args.algorithm,
    )
    scoped_tuning_config_path = (
        tuning_config_path_for(
            args.output_dir,
            args.algorithm,
        )
    )
    stored_search_space = None

    if args.resume:
        if tuning_config_path is not None:
            tuning_config = read_tuning_config(
                tuning_config_path
            )
            stored_algorithm = tuning_config_algorithm(
                tuning_config
            )

            if stored_algorithm != args.algorithm:
                raise RuntimeError(
                    "Stored tuning config algorithm "
                    f"{stored_algorithm!r} does not match "
                    f"requested algorithm {args.algorithm!r}."
                )

            apply_tuning_config(
                args,
                tuning_config,
            )
            stored_parameter_order = tuning_config.get(
                "search_space_parameter_order"
            )

            if (
                stored_parameter_order is None
                and tuning_config["arguments"].get(
                    "search_space"
                )
                is not None
            ):
                stored_parameter_order = list(
                    tuning_config["search_space"]
                )

            stored_search_space = ordered_search_space(
                args.algorithm,
                tuning_config["search_space"],
                stored_parameter_order,
            )
            print(
                "Loaded tuning arguments from "
                f"{tuning_config_path}.",
                flush=True,
            )

            resolve_dataset_bundle_args(args)
        elif has_tuning_history(
            output_dir=args.output_dir,
            results_path=results_path,
            algorithm=args.algorithm,
        ):
            raise RuntimeError(
                "Cannot resume existing "
                f"{args.algorithm} tuning history in "
                f"{args.output_dir} because "
                f"{scoped_tuning_config_path} is missing. "
                "The original tuning arguments are "
                "unavailable."
            )
    elif has_tuning_history(
        output_dir=args.output_dir,
        results_path=results_path,
        algorithm=args.algorithm,
    ):
        raise RuntimeError(
            f"{args.algorithm} tuning history already exists "
            f"in {args.output_dir}. Use --resume to continue "
            "with the stored tuning arguments, or choose a "
            "new --output-dir."
        )

    validate_tuning_paths(args)

    if not args.dry_run:
        for dataset_path in [
            args.train_dataset,
            args.validation_dataset,
            args.same_layout_dataset,
        ]:
            if not dataset_path.exists():
                raise FileNotFoundError(
                    dataset_path
                )

        if (
            args.evaluate_best_on_test
            and not args.test_dataset.exists()
        ):
            raise FileNotFoundError(
                args.test_dataset
                )

    if stored_search_space is not None:
        search_space = stored_search_space
    else:
        loaded_search_space = load_search_space(
            args.algorithm,
            args.search_space,
        )
        search_space = ordered_search_space(
            args.algorithm,
            loaded_search_space,
            (
                list(loaded_search_space)
                if args.search_space is not None
                else None
            ),
        )

    validate_search_space(
        args.algorithm,
        search_space,
    )

    if (
        not args.dry_run
        and not scoped_tuning_config_path.exists()
    ):
        write_tuning_config(
            scoped_tuning_config_path,
            args,
            search_space,
        )

    rng = np.random.default_rng(args.seed)
    optuna_study = (
        create_optuna_study(
            seed=args.seed,
        )
        if args.sampler == "optuna"
        else None
    )
    completed_results = (
        read_completed_results(
            results_path,
            args.algorithm,
        )
        if args.resume
        else {}
    )
    algorithm_best_config_path = (
        best_config_path_for_algorithm(
            args.output_dir,
            args.algorithm,
        )
    )
    best_result = None

    if args.resume:
        if completed_results:
            print(
                "Resuming tuning with "
                f"{len(completed_results)} completed "
                "hyperparameter combination(s) from "
                f"{results_path}.",
                flush=True,
            )
        else:
            print(
                "Resuming tuning; no completed hyperparameter "
                "combinations found "
                f"in {results_path}.",
                flush=True,
            )

    for trial in range(
        1,
        args.hyperparameter_combinations + 1,
    ):
        trial_seed = int(
            rng.integers(
                1_000_000_000
            )
        )
        optuna_trial = None

        if optuna_study is None:
            hyperparameters = (
                sample_hyperparameters(
                    search_space,
                    rng,
                )
            )
        else:
            optuna_trial = optuna_study.ask()
            hyperparameters = (
                sample_hyperparameters_with_optuna(
                    search_space,
                    optuna_trial,
                )
            )
        trial_dir = (
            args.output_dir
            / "trials"
            / f"trial_{trial:03d}"
        )
        run_dir = (
            trial_dir / args.algorithm
        )
        checkpoint_path = (
            run_dir / "checkpoint.pt"
        )
        best_checkpoint_path = (
            run_dir / "best_checkpoint.pt"
        )
        training_summary_path = (
            run_dir / "training_summary.json"
        )
        evaluation_path = (
            run_dir / "validation_evaluation.csv"
        )

        print()
        print(
            "Hyperparameter combination "
            f"{trial}/{args.hyperparameter_combinations} "
            f"({trial / args.hyperparameter_combinations:.0%}): "
            f"{json.dumps(hyperparameters, sort_keys=True)}",
            flush=True,
        )

        training_command = build_training_command(
            args=args,
            trial_dir=trial_dir,
            train_dataset=args.train_dataset,
            trial_seed=trial_seed,
            hyperparameters=hyperparameters,
        )
        evaluation_command = build_evaluation_command(
            args=args,
            checkpoint_path=best_checkpoint_path,
            validation_dataset=args.validation_dataset,
            evaluation_path=evaluation_path,
            trial_seed=trial_seed,
        )

        if args.resume and trial in completed_results:
            completed_result = completed_results[
                trial
            ]

            if not result_matches_trial(
                completed_result,
                trial_seed=trial_seed,
                hyperparameters=hyperparameters,
            ):
                print(
                    "Recorded hyperparameter combination "
                    f"{trial} does not match the current "
                    "sampled seed and hyperparameters; keeping "
                    "the completed recorded result.",
                    flush=True,
                )
            elif optuna_study is not None:
                optuna_study.tell(
                    optuna_trial,
                    float(
                        completed_result["objective"]
                    ),
                )

            previous_best_result = best_result
            best_result = update_best_result(
                best_result,
                completed_result,
            )

            if (
                not args.dry_run
                and best_result is not previous_best_result
            ):
                write_best_configs(
                    best_config_path=best_config_path,
                    algorithm_best_config_path=algorithm_best_config_path,
                    algorithm=args.algorithm,
                    best_result=best_result,
                )
            print(
                "Skipping completed hyperparameter combination "
                f"{trial}; checkpoint and evaluation "
                "artifacts are present.",
                flush=True,
            )
            continue

        if args.resume and not args.dry_run:
            recovered_result = build_result_from_artifacts(
                args=args,
                trial=trial,
                trial_seed=trial_seed,
                checkpoint_path=checkpoint_path,
                best_checkpoint_path=best_checkpoint_path,
                training_summary_path=training_summary_path,
                evaluation_path=evaluation_path,
            )

            if recovered_result is not None:
                if not result_matches_trial(
                    recovered_result,
                    trial_seed=trial_seed,
                    hyperparameters=hyperparameters,
                ):
                    print(
                        "Existing complete artifacts for "
                        "hyperparameter combination "
                        f"{trial} do not match the current "
                        "sampled seed and hyperparameters; "
                        "recording the recovered result instead "
                        "of rerunning.",
                        flush=True,
                    )
                elif optuna_study is not None:
                    optuna_study.tell(
                        optuna_trial,
                        float(
                            recovered_result[
                                "objective"
                            ]
                        ),
                    )

                write_result_row(
                    results_path,
                    recovered_result,
                )

                previous_best_result = best_result
                best_result = update_best_result(
                    best_result,
                    recovered_result,
                )
                if best_result is not previous_best_result:
                    write_best_configs(
                        best_config_path=best_config_path,
                        algorithm_best_config_path=algorithm_best_config_path,
                        algorithm=args.algorithm,
                        best_result=best_result,
                    )

                print(
                    "Recovered completed hyperparameter combination "
                    f"{trial} from existing artifacts.",
                    flush=True,
                )
                print(
                    "Validation mean path efficiency: "
                    f"{recovered_result['objective']:.3f}",
                    flush=True,
                )
                continue

        run_command(
            training_command,
            dry_run=args.dry_run,
        )
        run_command(
            evaluation_command,
            dry_run=args.dry_run,
        )

        if args.dry_run:
            if optuna_study is not None:
                optuna_study.tell(
                    optuna_trial,
                    0.0,
                )
            continue

        result = build_result_from_artifacts(
            args=args,
            trial=trial,
            trial_seed=trial_seed,
            checkpoint_path=checkpoint_path,
            best_checkpoint_path=best_checkpoint_path,
            training_summary_path=training_summary_path,
            evaluation_path=evaluation_path,
        )

        if result is None:
            raise RuntimeError(
                "Training finished but complete combination "
                f"artifacts were not found in {run_dir}."
            )

        write_result_row(
            results_path,
            result,
        )
        if optuna_study is not None:
            optuna_study.tell(
                optuna_trial,
                float(result["objective"]),
            )

        previous_best_result = best_result
        best_result = update_best_result(
            best_result,
            result,
        )
        if best_result is not previous_best_result:
            write_best_configs(
                best_config_path=best_config_path,
                algorithm_best_config_path=algorithm_best_config_path,
                algorithm=args.algorithm,
                best_result=best_result,
            )

        print(
            "Validation mean path efficiency: "
            f"{result['objective']:.3f}",
            flush=True,
        )

    if best_result is not None:
        print()
        print(
            "Best validation mean path efficiency: "
            f"{best_result['objective']:.3f} "
            "(hyperparameter combination "
            f"{best_result['trial']})"
        )
        print(
            f"Saved results to {results_path}"
        )
        print(
            "Saved algorithm best config to "
            f"{algorithm_best_config_path}"
        )
        print(
            "Saved indexed best configs to "
            f"{best_config_path}"
        )

        if args.evaluate_best_on_test:
            test_evaluation_path = (
                args.output_dir
                / "best_test_evaluation.csv"
            )
            run_command(
                build_evaluation_command(
                    args=args,
                    checkpoint_path=Path(
                        best_result[
                            "checkpoint_path"
                        ]
                    ),
                    validation_dataset=args.test_dataset,
                    evaluation_path=test_evaluation_path,
                    trial_seed=int(
                        best_result["seed"]
                    ),
                ),
                dry_run=False,
            )
            test_summary = summarize_evaluation(
                test_evaluation_path
            )
            print(
                "Best-combination test mean path efficiency: "
                f"{test_summary['mean_path_efficiency']:.3f}",
                flush=True,
            )

            natural_test_dataset = getattr(
                args,
                "natural_test_dataset",
                None,
            )
            if natural_test_dataset is not None:
                natural_test_evaluation_path = (
                    args.output_dir
                    / "best_test_natural_evaluation.csv"
                )
                run_command(
                    build_evaluation_command(
                        args=args,
                        checkpoint_path=Path(
                            best_result[
                                "checkpoint_path"
                            ]
                        ),
                        validation_dataset=natural_test_dataset,
                        evaluation_path=(
                            natural_test_evaluation_path
                        ),
                        trial_seed=int(
                            best_result["seed"]
                        ),
                    ),
                    dry_run=False,
                )
                natural_test_summary = summarize_evaluation(
                    natural_test_evaluation_path
                )
                print(
                    "Best-combination test-natural mean path "
                    "efficiency: "
                    f"{natural_test_summary['mean_path_efficiency']:.3f}",
                    flush=True,
                )


if __name__ == "__main__":
    main()
