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

SUMMARY_SPLITS = [
    "validation",
    "same_layout",
]
SUMMARY_SPLIT_LABELS = {
    "validation": "val",
    "same_layout": "val_with_same_layout",
}
SUMMARY_SPLIT_ALIASES = {
    "val": "validation",
    "validation": "validation",
    "same_layout": "same_layout",
    "val_with_same_layout": "same_layout",
}
SUMMARY_TRAIN_METRIC_ALIASES = {
    "average_episode_return": [
        "episode_return",
    ],
    "mean_path_efficiency": [
        "path_efficiency",
    ],
    "average_successful_path_efficiency": [
        "path_efficiency",
    ],
    "success_rate": [
        "success",
    ],
}
SUMMARY_SERIES_STYLES = {
    "validation": {
        "label": "val",
        "color": "#1f77b4",
        "linestyle": "-",
    },
    "same_layout": {
        "label": "same",
        "color": "#ff7f0e",
        "linestyle": "-",
    },
    "train": {
        "label": "train",
        "color": "#2ca02c",
        "linestyle": "-",
    },
}
SUMMARY_SCORE_LABELS = {
    "validation": "v",
    "same_layout": "s",
    "train": "t",
}
SUMMARY_METRIC_LABELS = {
    "average_episode_return": "Return",
    "mean_path_efficiency": "Path efficiency",
    "average_successful_path_efficiency": "Successful path efficiency",
    "success_rate": "Success rate",
}
SUMMARY_BOUNDED_METRICS = {
    "mean_path_efficiency",
    "success_rate",
    "average_successful_path_efficiency",
}
SUMMARY_MONTAGE_COLUMNS = 5
SUMMARY_MONTAGE_ROWS = 4
SUMMARY_MONTAGE_AXIS_WIDTH = 7.0
SUMMARY_MONTAGE_AXIS_HEIGHT = 7.2
SUMMARY_MONTAGE_DPI = 160


def rewrite_png_without_alpha(
    path: Path,
) -> None:
    if not path.exists():
        return

    import imageio.v2 as imageio

    image = imageio.imread(path)

    if (
        image.ndim != 3
        or image.shape[-1] != 4
        or image.dtype != np.uint8
    ):
        return

    rgb = image[..., :3].astype(np.uint16)
    alpha = image[..., 3:4].astype(np.uint16)
    white = np.uint16(255)
    composited = (
        rgb * alpha
        + white * (white - alpha)
        + np.uint16(127)
    ) // white
    imageio.imwrite(
        path,
        composited.astype(np.uint8),
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

RESULT_FIELDNAMES = [
    "trial",
    "algorithm",
    "seed",
    "objective",
    "mean_path_efficiency",
    "success_rate",
    "average_episode_return",
    "average_successful_path_efficiency",
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
    "hyperparameter_combinations",
    "rollouts_per_task",
    "max_steps",
    "seed",
    "train_dataset",
    "validation_dataset",
    "same_layout_dataset",
    "test_dataset",
    "evaluate_best_on_test",
    "search_space",
    "validation_interval",
    "q_snapshot_count",
    "early_stopping_patience",
    "early_stopping_min_delta",
    "tensorboard",
    "keep_plots",
]
TUNING_OPTIONAL_ARGUMENT_DEFAULTS = {
    "q_snapshot_count": 11,
}
TUNING_PATH_ARGUMENT_FIELDS = {
    "train_dataset",
    "validation_dataset",
    "same_layout_dataset",
    "test_dataset",
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


def parse_summarize_args(
    argv: list[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize DNN tuning artifacts with validation "
            "heatmaps, per-agent validation plot grids, and "
            "parallel-coordinate hyperparameter plots."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs") / "tuning",
        help=(
            "Tuning output directory containing trials."
        ),
    )
    parser.add_argument(
        "--summary-output-dir",
        type=Path,
        default=None,
        help=(
            "Directory for summary plots. Defaults to "
            "<output-dir>/summary_plots."
        ),
    )
    parser.add_argument(
        "--metric",
        default="mean_path_efficiency",
        help=(
            "Validation metrics CSV column to visualize in the "
            "summary plots."
        ),
    )

    args = parser.parse_args(argv)
    args.command = "summarize"
    return args


def parse_args(
    argv: list[str] | None = None,
) -> argparse.Namespace:
    if argv is None:
        argv = sys.argv[1:]

    if argv and argv[0] == "summarize":
        return parse_summarize_args(
            argv[1:]
        )

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
        default=Path("runs") / "tuning",
    )
    parser.add_argument(
        "--train-dataset",
        type=Path,
        default=Path("data") / "train.npz",
        help=(
            "Pre-generated training dataset path."
        ),
    )
    parser.add_argument(
        "--validation-dataset",
        type=Path,
        default=Path("data") / "validation.npz",
        help=(
            "Pre-generated validation dataset path used for "
            "selecting the best hyperparameter combination."
        ),
    )
    parser.add_argument(
        "--same-layout-dataset",
        type=Path,
        default=Path("data")
        / "same_layout_new_goals.npz",
        help=(
            "Pre-generated same-layout new-task dataset to monitor "
            "beside validation performance."
        ),
    )
    parser.add_argument(
        "--test-dataset",
        type=Path,
        default=Path("data") / "test.npz",
        help=(
            "Pre-generated test dataset path. Used only with "
            "--evaluate-best-on-test."
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
        "--validation-interval",
        type=int,
        default=1,
        help=(
            "Task-selection-pass interval for validation during "
            "each hyperparameter combination."
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
            "rerun with their original sampled seed and "
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
    args.command = "tune"
    args.hyperparameter_combinations_from_cli = (
        args.hyperparameter_combinations is not None
    )
    args.tensorboard_from_cli = args.tensorboard is not None

    if args.hyperparameter_combinations is None:
        args.hyperparameter_combinations = 100

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


def load_search_space(
    algorithm: str,
    path: Path | None,
) -> dict[str, dict[str, Any]]:
    if path is None:
        return DEFAULT_SEARCH_SPACES[
            algorithm
        ]

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
        "--validation-interval",
        str(args.validation_interval),
        "--seed",
        str(trial_seed),
        "--output-dir",
        str(trial_dir),
        "--q-snapshot-count",
        str(args.q_snapshot_count),
        "--no-progress",
    ]

    if args.tensorboard:
        command.append("--tensorboard")
    else:
        command.append("--no-tensorboard")

    if not args.keep_plots:
        command.append("--no-plot")

    command.append("--validation-all-tasks")

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
        "a",
        newline="",
    ) as file:
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


def trial_number_from_path(
    trial_dir: Path,
) -> int | None:
    prefix = "trial_"

    if not trial_dir.name.startswith(prefix):
        return None

    try:
        return int(
            trial_dir.name[len(prefix) :]
        )
    except ValueError:
        return None


def canonical_summary_split(
    split: str,
) -> str | None:
    return SUMMARY_SPLIT_ALIASES.get(
        split,
    )


def summary_split_label(
    split: str,
) -> str:
    return SUMMARY_SPLIT_LABELS.get(
        split,
        split,
    )


def discover_validation_metric_artifacts(
    output_dir: Path,
) -> list[dict[str, Any]]:
    trials_dir = output_dir / "trials"

    if not trials_dir.exists():
        return []

    artifacts = []

    for metrics_path in sorted(
        trials_dir.glob(
            "trial_*/*/validation_metrics.csv"
        )
    ):
        trial_number = trial_number_from_path(
            metrics_path.parents[1]
        )
        algorithm = metrics_path.parent.name

        if (
            trial_number is None
            or algorithm not in NEURAL_ALGORITHMS
        ):
            continue

        plot_path = (
            metrics_path.parent
            / "validation_metrics.png"
        )
        training_metrics_path = (
            metrics_path.parent
            / "metrics.csv"
        )
        training_plot_path = (
            metrics_path.parent
            / "training_metrics.png"
        )
        artifacts.append(
            {
                "algorithm": algorithm,
                "trial": trial_number,
                "metrics_path": metrics_path,
                "plot_path": plot_path,
                "training_metrics_path": training_metrics_path,
                "training_plot_path": training_plot_path,
            }
        )

    return artifacts


def best_metric_by_split(
    metrics_path: Path,
    metric: str,
) -> dict[str, float]:
    values: dict[str, list[float]] = {
        split: []
        for split in SUMMARY_SPLITS
    }

    with metrics_path.open(
        newline="",
    ) as file:
        reader = csv.DictReader(file)

        for row in reader:
            raw_value = row.get(metric)

            if raw_value in {
                None,
                "",
            }:
                continue

            split = canonical_summary_split(
                row.get(
                    "split",
                    "validation",
                )
            )

            if split not in values:
                continue

            values[split].append(
                float(raw_value)
            )

    return {
        split: max(split_values)
        for split, split_values in values.items()
        if split_values
    }


def collect_heatmap_cells(
    artifacts: list[dict[str, Any]],
    *,
    metric: str,
) -> list[dict[str, Any]]:
    cells = []

    for artifact in artifacts:
        split_values = best_metric_by_split(
            artifact["metrics_path"],
            metric,
        )

        for split, value in split_values.items():
            cells.append(
                {
                    "algorithm": artifact[
                        "algorithm"
                    ],
                    "trial": artifact["trial"],
                    "split": split,
                    "value": value,
                }
            )

    return cells


def artifact_summary_score(
    artifact: dict[str, Any],
    *,
    metric: str,
) -> float | None:
    split_values = best_metric_by_split(
        artifact["metrics_path"],
        metric,
    )

    if "validation" in split_values:
        return split_values["validation"]

    if split_values:
        return max(
            split_values.values()
        )

    return None


def parse_summary_float(
    raw_value: str | None,
) -> float | None:
    if raw_value in {
        None,
        "",
    }:
        return None

    if raw_value in {
        "True",
        "true",
        "1",
    }:
        return 1.0

    if raw_value in {
        "False",
        "false",
        "0",
    }:
        return 0.0

    return float(raw_value)


def train_metric_candidates(
    metric: str,
) -> list[str]:
    aliases = SUMMARY_TRAIN_METRIC_ALIASES.get(
        metric,
        [],
    )
    return [
        *aliases,
        metric,
    ]


def best_train_metric(
    metrics_path: Path,
    *,
    metric: str,
    aligned_epochs: set[int] | None = None,
) -> float | None:
    if not metrics_path.exists():
        return None

    with metrics_path.open(
        newline="",
    ) as file:
        rows = list(
            csv.DictReader(file)
        )

    for candidate in train_metric_candidates(
        metric
    ):
        values_by_epoch: dict[int, list[float]] = {}

        for row in rows:
            if candidate not in row:
                continue

            value = parse_summary_float(
                row.get(candidate)
            )

            if value is None:
                continue

            epoch = int(
                row.get("epoch")
                or row.get("episode")
                or 0
            )

            if (
                aligned_epochs is not None
                and epoch not in aligned_epochs
            ):
                continue

            values_by_epoch.setdefault(
                epoch,
                [],
            ).append(value)

        if values_by_epoch:
            return max(
                float(
                    np.mean(epoch_values)
                )
                for epoch_values in values_by_epoch.values()
            )

    return None


def validation_metric_series(
    metrics_path: Path,
    *,
    metric: str,
) -> dict[str, list[tuple[int, float]]]:
    points_by_split: dict[str, dict[int, list[float]]] = {
        split: {}
        for split in SUMMARY_SPLITS
    }

    with metrics_path.open(
        newline="",
    ) as file:
        reader = csv.DictReader(file)

        for row in reader:
            split = canonical_summary_split(
                row.get(
                    "split",
                    "validation",
                )
            )

            if split not in points_by_split:
                continue

            value = parse_summary_float(
                row.get(metric)
            )

            if value is None:
                continue

            epoch = int(
                row["dataset_epoch"]
            )
            points_by_split[split].setdefault(
                epoch,
                [],
            ).append(value)

    return {
        split: [
            (
                epoch,
                float(
                    np.mean(values)
                ),
            )
            for epoch, values in sorted(
                points_by_epoch.items()
            )
        ]
        for split, points_by_epoch in points_by_split.items()
        if points_by_epoch
    }


def validation_metric_epochs(
    series_by_split: dict[str, list[tuple[int, float]]],
) -> set[int]:
    return {
        epoch
        for series in series_by_split.values()
        for epoch, _ in series
    }


def aligned_training_metric_series(
    metrics_path: Path,
    *,
    metric: str,
    aligned_epochs: set[int],
) -> list[tuple[int, float]]:
    if not metrics_path.exists() or not aligned_epochs:
        return []

    with metrics_path.open(
        newline="",
    ) as file:
        rows = list(
            csv.DictReader(file)
        )

    for candidate in train_metric_candidates(
        metric
    ):
        values_by_epoch: dict[int, list[float]] = {}

        for row in rows:
            if candidate not in row:
                continue

            epoch = int(
                row.get("epoch")
                or row.get("episode")
                or 0
            )

            if epoch not in aligned_epochs:
                continue

            value = parse_summary_float(
                row.get(candidate)
            )

            if value is None:
                continue

            values_by_epoch.setdefault(
                epoch,
                [],
            ).append(value)

        if values_by_epoch:
            return [
                (
                    epoch,
                    float(
                        np.mean(values)
                    ),
                )
                for epoch, values in sorted(
                    values_by_epoch.items()
                )
            ]

    return []


def artifact_title_score_text(
    artifact: dict[str, Any],
    *,
    metric: str,
) -> str:
    split_values = best_metric_by_split(
        artifact["metrics_path"],
        metric,
    )
    labels = []

    for split in SUMMARY_SPLITS:
        if split not in split_values:
            continue

        labels.append(
            f"{SUMMARY_SCORE_LABELS[split]} "
            f"{split_values[split]:.2f}"
        )

    training_metrics_path = artifact.get(
        "training_metrics_path"
    )

    if (
        training_metrics_path is not None
        and training_metrics_path.exists()
    ):
        aligned_epochs = validation_metric_epochs(
            validation_metric_series(
                artifact["metrics_path"],
                metric=metric,
            )
        )
        train_score = best_train_metric(
            artifact["training_metrics_path"],
            metric=metric,
            aligned_epochs=aligned_epochs,
        )

        if train_score is not None:
            labels.append(
                f"{SUMMARY_SCORE_LABELS['train']} "
                f"{train_score:.2f}"
            )

    if not labels:
        return ""

    return "  ".join(labels)


def best_performing_trials(
    artifacts: list[dict[str, Any]],
    *,
    metric: str,
) -> set[int]:
    trial_scores = []

    for artifact in artifacts:
        score = artifact_summary_score(
            artifact,
            metric=metric,
        )

        if score is None:
            continue

        trial_scores.append(
            (
                artifact["trial"],
                score,
            )
        )

    if not trial_scores:
        return set()

    best_score = max(
        score
        for _, score in trial_scores
    )

    return {
        trial
        for trial, score in trial_scores
        if math.isclose(
            score,
            best_score,
            rel_tol=1e-12,
            abs_tol=1e-15,
        )
    }


def ordered_algorithms(
    algorithms: set[str],
) -> list[str]:
    ordered = [
        algorithm
        for algorithm in sorted(
            NEURAL_ALGORITHMS
        )
        if algorithm in algorithms
    ]
    ordered.extend(
        sorted(
            algorithms - set(ordered)
        )
    )
    return ordered


def artifact_training_summary_path(
    artifact: dict[str, Any],
) -> Path:
    return (
        artifact["metrics_path"].parent
        / "training_summary.json"
    )


def parse_parallel_coordinate_value(
    value: Any,
) -> float | None:
    if value is None or value == "":
        return None

    if isinstance(
        value,
        bool,
    ):
        return float(
            int(value)
        )

    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None

    if not math.isfinite(parsed):
        return None

    return parsed


def collect_parallel_coordinate_trials(
    artifacts: list[dict[str, Any]],
    *,
    metric: str,
) -> list[dict[str, Any]]:
    trials = []

    for artifact in artifacts:
        score = artifact_summary_score(
            artifact,
            metric=metric,
        )

        if score is None:
            continue

        training_summary_path = (
            artifact_training_summary_path(
                artifact
            )
        )

        if not training_summary_path.exists():
            continue

        try:
            training_summary = read_training_summary(
                training_summary_path
            )
        except (OSError, json.JSONDecodeError):
            continue

        hyperparameters = training_summary.get(
            "hyperparameters",
            {},
        )

        if not isinstance(
            hyperparameters,
            dict,
        ):
            continue

        raw_values = dict(
            hyperparameters
        )

        for name in TRAINING_LOOP_PARAMETERS:
            value = training_summary.get(name)

            if value is not None:
                raw_values[name] = value

        values = {}

        for name, value in raw_values.items():
            parsed = parse_parallel_coordinate_value(
                value
            )

            if parsed is not None:
                values[name] = parsed

        if not values:
            continue

        trials.append(
            {
                "algorithm": artifact["algorithm"],
                "trial": artifact["trial"],
                "score": score,
                "hyperparameters": values,
            }
        )

    return trials


def ordered_parallel_coordinate_parameters(
    trials: list[dict[str, Any]],
    *,
    algorithm: str,
) -> list[str]:
    names = {
        name
        for trial in trials
        for name in trial["hyperparameters"]
    }
    ordered_names = [
        name
        for name in DEFAULT_SEARCH_SPACES.get(
            algorithm,
            {},
        )
        if name in names
    ]
    ordered_names.extend(
        sorted(
            names - set(ordered_names)
        )
    )
    return ordered_names


def parallel_coordinate_uses_log_scale(
    *,
    algorithm: str,
    name: str,
    values: list[float],
) -> bool:
    if not values or any(
        value <= 0
        for value in values
    ):
        return False

    search_space = DEFAULT_SEARCH_SPACES.get(
        algorithm,
        {},
    ).get(name)

    if (
        search_space is not None
        and search_space.get("type") == "loguniform"
    ):
        return True

    return max(values) / min(values) >= 100.0


def parallel_coordinate_axis_label(
    name: str,
) -> str:
    return name.replace(
        "_",
        "\n",
    )


def parallel_coordinate_value_label(
    value: float,
) -> str:
    if math.isclose(
        value,
        round(value),
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        return str(int(round(value)))

    if (
        abs(value) < 0.01
        or abs(value) >= 1000
    ):
        return f"{value:.1e}"

    return f"{value:.3g}"


def normalized_parallel_coordinate_value(
    value: float,
    *,
    axis_min: float,
    axis_max: float,
) -> float:
    if math.isclose(
        axis_min,
        axis_max,
        rel_tol=1e-12,
        abs_tol=1e-15,
    ):
        return 0.5

    return (value - axis_min) / (
        axis_max - axis_min
    )


def plot_hyperparameter_parallel_coordinates(
    trials: list[dict[str, Any]],
    *,
    algorithm: str,
    metric: str,
    output_path: Path,
) -> int:
    plot_trials = [
        trial
        for trial in sorted(
            trials,
            key=lambda item: item["trial"],
        )
        if trial["algorithm"] == algorithm
    ]

    if not plot_trials:
        return 0

    parameter_names = (
        ordered_parallel_coordinate_parameters(
            plot_trials,
            algorithm=algorithm,
        )
    )

    if not parameter_names:
        return 0

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import cm, colors

    metric_label = SUMMARY_METRIC_LABELS.get(
        metric,
        metric,
    )
    axis_specs = []

    for name in parameter_names:
        raw_values = [
            trial["hyperparameters"][name]
            for trial in plot_trials
            if name in trial["hyperparameters"]
        ]
        use_log = parallel_coordinate_uses_log_scale(
            algorithm=algorithm,
            name=name,
            values=raw_values,
        )
        scaled_values = [
            math.log10(value)
            if use_log
            else value
            for value in raw_values
        ]
        axis_specs.append(
            {
                "name": name,
                "label": (
                    parallel_coordinate_axis_label(
                        name
                    )
                    + ("\nlog" if use_log else "")
                ),
                "use_log": use_log,
                "minimum": min(scaled_values),
                "maximum": max(scaled_values),
                "raw_minimum": min(raw_values),
                "raw_maximum": max(raw_values),
            }
        )

    scores = [
        trial["score"]
        for trial in plot_trials
    ]
    axis_specs.append(
        {
            "name": "__score__",
            "label": parallel_coordinate_axis_label(
                metric_label
            ),
            "use_log": False,
            "minimum": min(scores),
            "maximum": max(scores),
            "raw_minimum": min(scores),
            "raw_maximum": max(scores),
        }
    )

    x_positions = np.arange(
        len(axis_specs)
    )
    figure_width = max(
        9.0,
        1.15 * len(axis_specs),
    )
    figure, axis = plt.subplots(
        figsize=(
            figure_width,
            6.2,
        )
    )
    score_min = min(scores)
    score_max = max(scores)
    score_norm = colors.Normalize(
        vmin=score_min,
        vmax=(
            score_max
            if not math.isclose(
                score_min,
                score_max,
                rel_tol=1e-12,
                abs_tol=1e-15,
            )
            else score_min + 1.0
        ),
    )
    cmap = plt.get_cmap("viridis")
    best_score = score_max

    for trial in plot_trials:
        y_values = []

        for spec in axis_specs:
            if spec["name"] == "__score__":
                raw_value = trial["score"]
            else:
                raw_value = trial[
                    "hyperparameters"
                ].get(spec["name"])

            if raw_value is None:
                y_values.append(np.nan)
                continue

            scaled_value = (
                math.log10(raw_value)
                if spec["use_log"]
                else raw_value
            )
            y_values.append(
                normalized_parallel_coordinate_value(
                    scaled_value,
                    axis_min=spec["minimum"],
                    axis_max=spec["maximum"],
                )
            )

        is_best = math.isclose(
            trial["score"],
            best_score,
            rel_tol=1e-12,
            abs_tol=1e-15,
        )
        axis.plot(
            x_positions,
            y_values,
            color=cmap(
                score_norm(
                    trial["score"]
                )
            ),
            linewidth=2.8 if is_best else 1.35,
            alpha=0.95 if is_best else 0.55,
            zorder=3 if is_best else 2,
        )

    for x_position, spec in zip(
        x_positions,
        axis_specs,
    ):
        axis.axvline(
            x_position,
            color="#d9d9d9",
            linewidth=1.0,
            zorder=1,
        )
        axis.text(
            x_position,
            -0.08,
            parallel_coordinate_value_label(
                spec["raw_minimum"]
            ),
            ha="center",
            va="top",
            fontsize=8,
            color="#555555",
        )
        axis.text(
            x_position,
            1.05,
            parallel_coordinate_value_label(
                spec["raw_maximum"]
            ),
            ha="center",
            va="bottom",
            fontsize=8,
            color="#555555",
        )

    axis.set_xticks(x_positions)
    axis.set_xticklabels(
        [
            spec["label"]
            for spec in axis_specs
        ],
        fontsize=9,
    )
    axis.set_yticks([])
    axis.set_ylim(-0.13, 1.12)
    axis.set_xlim(
        -0.25,
        len(axis_specs) - 0.75,
    )
    axis.set_title(
        f"{algorithm} hyperparameters by {metric_label}"
    )
    axis.grid(
        False,
    )
    for spine in axis.spines.values():
        spine.set_visible(False)

    colorbar = figure.colorbar(
        cm.ScalarMappable(
            norm=score_norm,
            cmap=cmap,
        ),
        ax=axis,
        pad=0.02,
    )
    colorbar.set_label(metric_label)
    figure.tight_layout()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    figure.savefig(
        output_path,
        dpi=180,
    )
    plt.close(figure)

    return len(plot_trials)


def plot_tuning_summary_heatmap(
    cells: list[dict[str, Any]],
    *,
    metric: str,
    output_path: Path,
) -> None:
    if not cells:
        raise RuntimeError(
            "No validation metric rows found to summarize."
        )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    algorithms = ordered_algorithms(
        {
            cell["algorithm"]
            for cell in cells
        }
    )
    trials = sorted(
        {
            cell["trial"]
            for cell in cells
        }
    )
    rows = [
        (
            algorithm,
            split,
        )
        for algorithm in algorithms
        for split in SUMMARY_SPLITS
        if any(
            cell["algorithm"] == algorithm
            and cell["split"] == split
            for cell in cells
        )
    ]
    row_index = {
        row: index
        for index, row in enumerate(rows)
    }
    trial_index = {
        trial: index
        for index, trial in enumerate(trials)
    }
    matrix = np.full(
        (
            len(rows),
            len(trials),
        ),
        np.nan,
    )

    for cell in cells:
        matrix[
            row_index[
                (
                    cell["algorithm"],
                    cell["split"],
                )
            ],
            trial_index[cell["trial"]],
        ] = cell["value"]

    figure_width = max(
        7.0,
        0.45 * len(trials),
    )
    figure_height = max(
        3.0,
        0.45 * len(rows),
    )
    figure, axis = plt.subplots(
        figsize=(
            figure_width,
            figure_height,
        )
    )
    cmap = plt.get_cmap(
        "viridis"
    ).with_extremes(
        bad="#f2f2f2"
    )
    image = axis.imshow(
        matrix,
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        vmin=0.0,
        vmax=(
            1.0
            if metric in SUMMARY_BOUNDED_METRICS
            else None
        ),
    )
    axis.set_title(
        f"Best {metric} by tuning trial"
    )
    axis.set_xlabel("Trial")
    axis.set_ylabel("Agent / dataset")
    axis.set_yticks(
        np.arange(
            len(rows)
        )
    )
    axis.set_yticklabels(
        [
            f"{algorithm} / {summary_split_label(split)}"
            for algorithm, split in rows
        ]
    )

    max_tick_count = 25
    tick_stride = max(
        1,
        math.ceil(
            len(trials) / max_tick_count
        ),
    )
    tick_positions = np.arange(
        0,
        len(trials),
        tick_stride,
    )
    axis.set_xticks(tick_positions)
    axis.set_xticklabels(
        [
            str(trials[index])
            for index in tick_positions
        ],
        rotation=45,
        ha="right",
    )
    figure.colorbar(
        image,
        ax=axis,
        label=metric,
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


def plot_summary_metric_series(
    axis,
    artifact: dict[str, Any],
    *,
    metric: str,
) -> bool:
    validation_series = validation_metric_series(
        artifact["metrics_path"],
        metric=metric,
    )
    aligned_epochs = validation_metric_epochs(
        validation_series
    )
    plotted = False

    for split in SUMMARY_SPLITS:
        series = validation_series.get(split)

        if not series:
            continue

        style = SUMMARY_SERIES_STYLES[split]
        axis.plot(
            [epoch for epoch, _ in series],
            [value for _, value in series],
            marker="o",
            markersize=3.8,
            linewidth=2.4,
            color=style["color"],
            linestyle=style["linestyle"],
            label=style["label"],
        )
        plotted = True

    training_series = aligned_training_metric_series(
        artifact["training_metrics_path"],
        metric=metric,
        aligned_epochs=aligned_epochs,
    )

    if training_series:
        style = SUMMARY_SERIES_STYLES["train"]
        axis.plot(
            [epoch for epoch, _ in training_series],
            [value for _, value in training_series],
            marker="o",
            markersize=3.8,
            linewidth=2.4,
            color=style["color"],
            linestyle=style["linestyle"],
            label=style["label"],
        )
        plotted = True

    return plotted


def set_summary_axis_y_limits(
    axis,
    *,
    metric: str,
) -> None:
    if metric not in SUMMARY_BOUNDED_METRICS:
        return

    values = []

    for line in axis.lines:
        values.extend(
            float(value)
            for value in line.get_ydata()
            if np.isfinite(value)
        )

    if not values:
        axis.set_ylim(-0.05, 1.05)
        return

    minimum = min(values)
    maximum = max(values)
    upper = max(
        0.1,
        min(
            1.05,
            maximum * 1.12 + 0.02,
        ),
    )
    lower = max(
        -0.05,
        min(
            0.0,
            minimum - (upper - minimum) * 0.05,
        ),
    )
    if math.isclose(
        lower,
        upper,
        rel_tol=1e-12,
        abs_tol=1e-15,
    ):
        upper = lower + 0.1

    axis.set_ylim(
        lower,
        upper,
    )


def plot_validation_metric_montage(
    artifacts: list[dict[str, Any]],
    *,
    algorithm: str,
    metric: str,
    output_path: Path,
) -> int:
    plot_artifacts = [
        artifact
        for artifact in sorted(
            artifacts,
            key=lambda item: item["trial"],
        )
        if artifact["algorithm"] == algorithm
        and artifact["metrics_path"].exists()
    ]

    if not plot_artifacts:
        return 0

    best_trials = best_performing_trials(
        plot_artifacts,
        metric=metric,
    )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metric_label = SUMMARY_METRIC_LABELS.get(
        metric,
        metric,
    )
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    for stale_page_path in output_path.parent.glob(
        f"{output_path.stem}_page_*{output_path.suffix}"
    ):
        stale_page_path.unlink()

    plot_count = len(plot_artifacts)
    slot_count = (
        SUMMARY_MONTAGE_COLUMNS
        * SUMMARY_MONTAGE_ROWS
    )
    if plot_count > slot_count:
        raise RuntimeError(
            "Validation metric trial grids support at most "
            f"{slot_count} trials per agent; found "
            f"{plot_count} for {algorithm}."
        )

    figure, axes = plt.subplots(
        SUMMARY_MONTAGE_ROWS,
        SUMMARY_MONTAGE_COLUMNS,
        figsize=(
            (
                SUMMARY_MONTAGE_AXIS_WIDTH
                * SUMMARY_MONTAGE_COLUMNS
            ),
            (
                SUMMARY_MONTAGE_AXIS_HEIGHT
                * SUMMARY_MONTAGE_ROWS
            ),
        ),
        sharey=False,
        squeeze=False,
    )

    for axis in axes.reshape(-1):
        axis.axis("off")

    legend_handles = None
    legend_labels = None

    for index, (axis, artifact) in enumerate(zip(
        axes.reshape(-1),
        plot_artifacts,
    )):
        axis.axis("on")
        plotted = plot_summary_metric_series(
            axis,
            artifact,
            metric=metric,
        )
        is_best_trial = artifact["trial"] in best_trials
        title = f"Trial {artifact['trial']}"
        score_text = artifact_title_score_text(
            artifact,
            metric=metric,
        )

        if is_best_trial:
            title = f"{title} best"

        if score_text:
            title = f"{title}: {score_text}"

        axis.set_title(
            title,
            fontsize=18,
            pad=8,
        )
        axis.tick_params(
            labelsize=15,
        )
        axis.grid(
            True,
            alpha=0.22,
            linewidth=0.8,
        )

        set_summary_axis_y_limits(
            axis,
            metric=metric,
        )

        if plotted:
            legend_handles, legend_labels = (
                axis.get_legend_handles_labels()
            )

        if is_best_trial:
            axis.set_facecolor("#fffaf0")
            for spine in axis.spines.values():
                spine.set_linewidth(2.0)
                spine.set_edgecolor("#d27d00")

        row_index = index // SUMMARY_MONTAGE_COLUMNS
        column_index = index % SUMMARY_MONTAGE_COLUMNS

        if row_index == SUMMARY_MONTAGE_ROWS - 1:
            axis.set_xlabel(
                "Dataset epoch",
                fontsize=16,
            )
        else:
            axis.tick_params(
                labelbottom=False,
            )

        if column_index == 0:
            axis.set_ylabel(
                metric_label,
                fontsize=16,
            )

    for axis in axes.reshape(-1)[plot_count:]:
        axis.axis("on")
        axis.set_facecolor("#f7f7f7")
        axis.set_xticks([])
        axis.set_yticks([])
        axis.text(
            0.5,
            0.5,
            "No completed trial",
            ha="center",
            va="center",
            transform=axis.transAxes,
            color="#777777",
            fontsize=18,
        )
        for spine in axis.spines.values():
            spine.set_color("#dddddd")
            spine.set_linewidth(1.0)

    figure.suptitle(
        (
            f"{algorithm} validation and aligned training "
            "metrics across trials"
        ),
        fontsize=30,
        y=0.99,
    )
    if (
        legend_handles is not None
        and legend_labels is not None
    ):
        figure.legend(
            legend_handles,
            legend_labels,
            loc="upper center",
            ncol=len(legend_labels),
            bbox_to_anchor=(0.5, 0.96),
            frameon=False,
            fontsize=22,
        )
    figure.subplots_adjust(
        left=0.045,
        right=0.99,
        bottom=0.04,
        top=0.91,
        hspace=0.22,
        wspace=0.14,
    )
    figure.savefig(
        output_path,
        dpi=SUMMARY_MONTAGE_DPI,
        facecolor="white",
        edgecolor="white",
        transparent=False,
    )
    rewrite_png_without_alpha(output_path)
    plt.close(figure)

    return plot_count


def summarize_tuning(
    args: argparse.Namespace,
) -> None:
    summary_output_dir = (
        args.summary_output_dir
        if args.summary_output_dir is not None
        else args.output_dir / "summary_plots"
    )
    artifacts = discover_validation_metric_artifacts(
        args.output_dir
    )

    if not artifacts:
        raise RuntimeError(
            "No validation_metrics.csv artifacts found under "
            f"{args.output_dir / 'trials'}."
        )

    cells = collect_heatmap_cells(
        artifacts,
        metric=args.metric,
    )
    heatmap_path = (
        summary_output_dir
        / f"{args.metric}_heatmap.png"
    )
    plot_tuning_summary_heatmap(
        cells,
        metric=args.metric,
        output_path=heatmap_path,
    )
    print(
        f"Saved tuning heatmap to {heatmap_path}",
        flush=True,
    )

    parallel_trials = collect_parallel_coordinate_trials(
        artifacts,
        metric=args.metric,
    )

    for algorithm in ordered_algorithms(
        {
            artifact["algorithm"]
            for artifact in artifacts
        }
    ):
        montage_path = (
            summary_output_dir
            / f"{algorithm}_validation_metrics_trials.png"
        )

        filled_trial_slots = plot_validation_metric_montage(
            artifacts,
            algorithm=algorithm,
            metric=args.metric,
            output_path=montage_path,
        )

        if filled_trial_slots:
            print(
                "Saved validation metrics trial grid to "
                f"{montage_path} "
                f"({filled_trial_slots}/20 trial slots filled)",
                flush=True,
            )
        else:
            print(
                "Skipped validation metrics trial grid for "
                f"{algorithm}; no validation_metrics.png files "
                "were found.",
                flush=True,
            )

        parallel_path = (
            summary_output_dir
            / f"{algorithm}_{args.metric}_parallel_coordinates.png"
        )
        parallel_trial_count = (
            plot_hyperparameter_parallel_coordinates(
                parallel_trials,
                algorithm=algorithm,
                metric=args.metric,
                output_path=parallel_path,
            )
        )

        if parallel_trial_count:
            print(
                "Saved hyperparameter parallel-coordinate plot to "
                f"{parallel_path} "
                f"({parallel_trial_count} trial(s))",
                flush=True,
            )
        else:
            print(
                "Skipped hyperparameter parallel-coordinate plot "
                f"for {algorithm}; no numeric hyperparameters "
                "were found in training_summary.json.",
                flush=True,
            )


def main() -> None:
    args = parse_args()

    if args.command == "summarize":
        summarize_tuning(args)
        return

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
        hyperparameters = (
            sample_hyperparameters(
                search_space,
                rng,
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


if __name__ == "__main__":
    main()
