from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
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
    GROUPED_POLICY_ALGORITHMS,
    NEURAL_ALGORITHMS,
    NEURAL_HYPERPARAMETERS,
    POLICY_ROLLOUT_EPISODES,
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
        "rollout_episodes": {
            "type": "choice",
            "values": [8, 16, 32],
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
        "rollout_episodes": {
            "type": "choice",
            "values": [8, 16, 32],
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
        "rollout_episodes": {
            "type": "choice",
            "values": [32, 64, 128],
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
        "group_size": {
            "type": "choice",
            "values": [4, 8, 16],
        },
    },
}

TRAINING_LOOP_PARAMETERS = {
    "rollout_episodes",
    "group_size",
}


def parse_args() -> argparse.Namespace:
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
        "--trials",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--dataset-epochs",
        type=int,
        default=20,
        help=(
            "Generous maximum training budget for every trial."
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
            "trial selection."
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
            "After tuning, evaluate the best validation trial on "
            "the held-out test dataset."
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
        "--eval-all-tasks",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Evaluate every validation task. Use "
            "--no-eval-all-tasks with --eval-episodes for sampling."
        ),
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=200,
    )
    parser.add_argument(
        "--validation-interval",
        type=int,
        default=1,
        help=(
            "Dataset-epoch interval for validation during each "
            "trial."
        ),
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=None,
        help=(
            "Optional shared patience for stopping trials after "
            "validation stalls."
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
        default=False,
    )
    parser.add_argument(
        "--keep-plots",
        action="store_true",
        help="Keep per-trial training and evaluation plots.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print trial commands without running them.",
    )

    return parser.parse_args()


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

    valid_parameters = set(
        NEURAL_HYPERPARAMETERS[algorithm]
    )

    if (
        algorithm in POLICY_ROLLOUT_EPISODES
        or algorithm in GROUPED_POLICY_ALGORITHMS
    ):
        valid_parameters |= (
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
        "--dataset-epochs",
        str(args.dataset_epochs),
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
        "0",
    ]

    if args.tensorboard:
        command.append("--tensorboard")
    else:
        command.append("--no-tensorboard")

    if not args.keep_plots:
        command.append("--no-plot")

    if args.eval_all_tasks:
        command.append(
            "--validation-all-tasks"
        )
    else:
        command.extend(
            [
                "--no-validation-all-tasks",
                "--validation-episodes",
                str(args.eval_episodes),
            ]
        )

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

    if "rollout_episodes" in hyperparameters:
        command.extend(
            [
                "--rollout-episodes",
                cli_value(
                    hyperparameters[
                        "rollout_episodes"
                    ]
                ),
            ]
        )

    if "group_size" in hyperparameters:
        command.extend(
            [
                "--group-size",
                cli_value(
                    hyperparameters[
                        "group_size"
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
    ]

    if args.eval_all_tasks:
        command.append("--all-tasks")
    else:
        command.extend(
            [
                "--episodes",
                str(args.eval_episodes),
            ]
        )

    if not args.keep_plots:
        command.append("--no-plot")

    return command


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

    subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=env,
        check=True,
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


def write_result_row(
    path: Path,
    row: dict[str, Any],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "trial",
        "algorithm",
        "seed",
        "objective",
        "mean_path_efficiency",
        "success_rate",
        "average_episode_return",
        "average_successful_path_efficiency",
        "dataset_epochs_completed",
        "stopped_early",
        "checkpoint_path",
        "evaluation_path",
        "hyperparameters",
    ]
    write_header = not path.exists()

    with path.open(
        "a",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
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


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
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

    search_space = load_search_space(
        args.algorithm,
        args.search_space,
    )
    rng = np.random.default_rng(args.seed)
    results_path = (
        args.output_dir / "tuning_results.csv"
    )
    best_config_path = (
        args.output_dir / "best_config.json"
    )
    best_result = None

    for trial in range(
        1,
        args.trials + 1,
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
            f"Trial {trial}/{args.trials} "
            f"({trial / args.trials:.0%}): "
            f"{json.dumps(hyperparameters, sort_keys=True)}",
            flush=True,
        )

        run_command(
            build_training_command(
                args=args,
                trial_dir=trial_dir,
                train_dataset=args.train_dataset,
                trial_seed=trial_seed,
                hyperparameters=hyperparameters,
            ),
            dry_run=args.dry_run,
        )
        run_command(
            build_evaluation_command(
                args=args,
                checkpoint_path=best_checkpoint_path,
                validation_dataset=args.validation_dataset,
                evaluation_path=evaluation_path,
                trial_seed=trial_seed,
            ),
            dry_run=args.dry_run,
        )

        if args.dry_run:
            continue

        training_summary = read_training_summary(
            training_summary_path
        )
        best_validation = training_summary[
            "best_validation"
        ]
        objective = best_validation[
            "mean_path_efficiency"
        ]
        summarize_evaluation(
            evaluation_path
        )
        result = {
            "trial": trial,
            "algorithm": args.algorithm,
            "seed": trial_seed,
            "objective": objective,
            "mean_path_efficiency": best_validation[
                "mean_path_efficiency"
            ],
            "success_rate": best_validation[
                "success_rate"
            ],
            "average_episode_return": best_validation[
                "average_episode_return"
            ],
            "average_successful_path_efficiency": best_validation[
                "average_successful_path_efficiency"
            ],
            "dataset_epochs_completed": (
                training_summary[
                    "dataset_epochs_completed"
                ]
            ),
            "stopped_early": training_summary[
                "stopped_early"
            ],
            "checkpoint_path": str(
                best_checkpoint_path
                if best_checkpoint_path.exists()
                else checkpoint_path
            ),
            "evaluation_path": str(
                evaluation_path
            ),
            "hyperparameters": hyperparameters,
        }

        write_result_row(
            results_path,
            result,
        )

        if (
            best_result is None
            or objective
            > best_result["objective"]
        ):
            best_result = result
            write_best_config(
                best_config_path,
                best_result,
            )

        print(
            "Validation mean path efficiency: "
            f"{objective:.3f}",
            flush=True,
        )

    if best_result is not None:
        print()
        print(
            "Best validation mean path efficiency: "
            f"{best_result['objective']:.3f} "
            f"(trial {best_result['trial']})"
        )
        print(
            f"Saved results to {results_path}"
        )
        print(
            f"Saved best config to {best_config_path}"
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
                "Best-trial test mean path efficiency: "
                f"{test_summary['mean_path_efficiency']:.3f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
