from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass
import json
import os
from pathlib import Path
import pickle
import random
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(REPO_ROOT),
    )

import numpy as np

os.environ.setdefault(
    "CUBLAS_WORKSPACE_CONFIG",
    ":4096:8",
)

import torch

from maze_rl.agents.a2c import A2CAgent
from maze_rl.agents.dqn import DQNAgent
from maze_rl.agents.dyna_q import DynaQAgent
from maze_rl.agents.grpo import GRPOAgent
from maze_rl.agents.monte_carlo import MonteCarloAgent
from maze_rl.agents.ppo import PPOAgent
from maze_rl.agents.q_learning import QLearningAgent
from maze_rl.agents.reinforce import ReinforceAgent
from maze_rl.agents.sarsa import SarsaAgent
from maze_rl.envs.maze_env import MazeEnv
from maze_rl.evaluation.evaluator import evaluate
from maze_rl.training.metrics import append_metrics_csv
from maze_rl.training.plots import (
    plot_task_training_metrics,
    plot_training_metrics,
    plot_validation_metrics,
    write_task_training_metrics_html,
)
from maze_rl.training.trainers import (
    train_a2c_round,
    train_dqn_episode,
    train_grpo_round,
    train_monte_carlo_episode,
    train_ppo_round,
    train_q_learning_episode,
    train_reinforce_round,
    train_sarsa_episode,
)
from scripts.experiment_naming import (
    DEFAULT_EXPERIMENT_OUTPUT_ROOT,
    DEFAULT_TABULAR_OUTPUT_ROOT,
    experiment_dataset_name_for_path,
)


TABULAR_ALGORITHMS = {
    "mc",
    "sarsa",
    "q_learning",
    "dyna_q",
}

NEURAL_ALGORITHMS = {
    "dqn",
    "reinforce",
    "a2c",
    "ppo",
    "grpo",
}

NEURAL_SNAPSHOT_ALGORITHMS = {
    "dqn",
    "a2c",
    "ppo",
}

NEURAL_HYPERPARAMETERS = {
    "dqn": {
        "gamma",
        "learning_rate",
        "epsilon_start",
        "epsilon_min",
        "epsilon_decay",
        "replay_capacity",
        "min_replay_size",
        "batch_size",
        "train_frequency",
        "target_update_interval",
    },
    "reinforce": {
        "gamma",
        "learning_rate",
        "entropy_coefficient",
        "entropy_coefficient_min",
        "entropy_coefficient_decay",
        "minibatch_size",
    },
    "a2c": {
        "gamma",
        "gae_lambda",
        "learning_rate",
        "value_coefficient",
        "entropy_coefficient",
        "entropy_coefficient_min",
        "entropy_coefficient_decay",
        "minibatch_size",
    },
    "ppo": {
        "gamma",
        "gae_lambda",
        "learning_rate",
        "clip_epsilon",
        "value_coefficient",
        "entropy_coefficient",
        "update_epochs",
        "minibatch_size",
    },
    "grpo": {
        "gamma",
        "learning_rate",
        "clip_epsilon",
        "entropy_coefficient",
        "entropy_coefficient_min",
        "entropy_coefficient_decay",
        "minibatch_size",
    },
}

NEURAL_HYPERPARAMETER_DESTS = sorted(
    {
        name
        for names in NEURAL_HYPERPARAMETERS.values()
        for name in names
    }
)

DEFAULT_TASK_BATCH_SIZES = {
    "dqn": 1,
    "reinforce": 16,
    "a2c": 16,
    "ppo": 64,
    "grpo": 16,
}

DEFAULT_ROLLOUT_GROUP_SIZES = {
    "dqn": 1,
    "reinforce": 1,
    "a2c": 1,
    "ppo": 1,
    "grpo": 8,
}


@dataclass(frozen=True)
class BestConfigOverrides:
    path: Path
    hyperparameters: dict[str, float | int]
    task_batch_size: int | None = None
    rollout_group_size: int | None = None


def configure_reproducibility(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def tensorboard_default_enabled(
    algorithm: str,
) -> bool:
    del algorithm
    return False


BEST_CONFIG_DEFAULT_SENTINEL = Path(
    "__algorithm_default_best_config__"
)


def default_best_config_path(
    algorithm: str,
) -> Path:
    return (
        DEFAULT_EXPERIMENT_OUTPUT_ROOT
        / "tuning"
        / f"best_config_{algorithm}.json"
    )


def default_output_dir(
    algorithm: str,
) -> Path:
    if algorithm in TABULAR_ALGORITHMS:
        return DEFAULT_TABULAR_OUTPUT_ROOT

    return (
        DEFAULT_EXPERIMENT_OUTPUT_ROOT
        / f"final_{algorithm}_best"
    )


def default_experiment_output_dir(
    *,
    experiment_name: str,
    dataset_path: Path,
    algorithm: str,
    experiment_root: Path = DEFAULT_EXPERIMENT_OUTPUT_ROOT,
) -> Path:
    return (
        experiment_root
        / experiment_dataset_name_for_path(
            experiment_name,
            dataset_path,
        )
    )


def dataset_bundle_output_dir(
    *,
    dataset_dir: Path,
    algorithm: str,
    experiment_root: Path = DEFAULT_EXPERIMENT_OUTPUT_ROOT,
) -> Path:
    return (
        experiment_root
        / dataset_dir.name
    )


def resolve_dataset_bundle_args(args) -> None:
    dataset_path = Path(args.dataset)

    if not dataset_path.is_dir():
        args.dataset = dataset_path
        return

    dataset_dir = dataset_path
    train_dataset = dataset_dir / "train.npz"
    validation_dataset = dataset_dir / "validation.npz"
    same_layout_dataset = (
        dataset_dir / "same_layout_new_goals.npz"
    )

    if not train_dataset.exists():
        raise ValueError(
            "Dataset directory must contain train.npz: "
            f"{dataset_dir}"
        )

    args.dataset = train_dataset

    if args.validation_dataset is None:
        if validation_dataset.exists():
            args.validation_dataset = validation_dataset

    if args.same_layout_dataset is None:
        if same_layout_dataset.exists():
            args.same_layout_dataset = same_layout_dataset

    if (
        args.output_dir is None
        and args.experiment_name is None
    ):
        args.output_dir = dataset_bundle_output_dir(
            dataset_dir=dataset_dir,
            algorithm=args.algorithm,
            experiment_root=args.experiment_root,
        )


RESUME_METADATA_CLI_OPTIONS = {
    "dataset": ("--dataset",),
    "fixed_index": ("--fixed-index",),
    "max_steps": ("--max-steps",),
    "seed": ("--seed",),
    "rollouts_per_task": (
        "--rollouts-per-task",
        "--dataset-epochs",
    ),
    "task_batch_size": (
        "--task-batch-size",
        "--rollout-episodes",
    ),
    "rollout_group_size": (
        "--rollout-group-size",
        "--group-size",
    ),
    "validation_dataset": (
        "--validation-dataset",
    ),
    "same_layout_dataset": (
        "--same-layout-dataset",
    ),
    "validation_interval": (
        "--validation-interval",
    ),
    "validation_episodes": (
        "--validation-episodes",
    ),
    "validation_all_tasks": (
        "--validation-all-tasks",
        "--no-validation-all-tasks",
    ),
    "early_stopping_patience": (
        "--early-stopping-patience",
    ),
    "early_stopping_min_delta": (
        "--early-stopping-min-delta",
    ),
}


def cli_option_present(
    argv: list[str],
    option_names: tuple[str, ...],
) -> bool:
    return any(
        argument == option
        or argument.startswith(f"{option}=")
        for argument in argv
        for option in option_names
    )


def resume_metadata_fields_from_argv(
    argv: list[str],
) -> set[str]:
    return {
        field
        for field, option_names in RESUME_METADATA_CLI_OPTIONS.items()
        if cli_option_present(
            argv,
            option_names,
        )
    }


@dataclass(frozen=True)
class RolloutSpec:
    rollout: int
    dataset_epoch: int
    task_index: int


class HierarchicalTaskSampler:
    def __init__(
        self,
        task_indices,
        layout_indices,
        seed: int,
    ) -> None:
        self.rng = np.random.default_rng(
            seed
        )
        self.task_indices = [
            int(task_index)
            for task_index in task_indices
        ]

        if not self.task_indices:
            raise ValueError(
                "At least one task is required for training."
            )

        self.tasks_by_layout = defaultdict(list)

        for task_index in self.task_indices:
            layout_index = int(
                layout_indices[task_index]
            )
            self.tasks_by_layout[
                layout_index
            ].append(task_index)

        self.layout_indices = sorted(
            self.tasks_by_layout
        )
        self.total_task_count = len(
            self.task_indices
        )
        self.current_dataset_epoch = 1
        self.completed_dataset_epochs = 0
        self.seen_in_dataset_epoch = set()
        self.rollout = 0

    def _sample_task(self) -> int:
        available_layouts = [
            layout_index
            for layout_index in self.layout_indices
            if any(
                task_index
                not in self.seen_in_dataset_epoch
                for task_index in self.tasks_by_layout[
                    layout_index
                ]
            )
        ]

        if not available_layouts:
            raise ValueError(
                "No unseen tasks remain in the current dataset epoch."
            )

        layout_index = int(
            self.rng.choice(
                available_layouts
            )
        )
        available_tasks = [
            task_index
            for task_index in self.tasks_by_layout[
                layout_index
            ]
            if task_index
            not in self.seen_in_dataset_epoch
        ]

        return int(
            self.rng.choice(
                available_tasks
            )
        )

    def sample_collection(
        self,
        collection_size: int,
        target_dataset_epochs: int,
    ) -> list[RolloutSpec]:
        specs = []

        while (
            len(specs) < collection_size
            and self.completed_dataset_epochs
            < target_dataset_epochs
        ):
            task_index = self._sample_task()
            self.rollout += 1

            specs.append(
                RolloutSpec(
                    rollout=self.rollout,
                    dataset_epoch=self.current_dataset_epoch,
                    task_index=task_index,
                )
            )

            self.seen_in_dataset_epoch.add(
                task_index
            )

            if (
                len(self.seen_in_dataset_epoch)
                == self.total_task_count
            ):
                self.completed_dataset_epochs += 1
                self.current_dataset_epoch += 1
                self.seen_in_dataset_epoch.clear()

        return specs

    def state_dict(self) -> dict:
        return {
            "rng_state": self.rng.bit_generator.state,
            "current_dataset_epoch": self.current_dataset_epoch,
            "completed_dataset_epochs": self.completed_dataset_epochs,
            "seen_in_dataset_epoch": sorted(
                self.seen_in_dataset_epoch
            ),
            "rollout": self.rollout,
        }

    def load_state_dict(
        self,
        state: dict,
    ) -> None:
        self.rng.bit_generator.state = state[
            "rng_state"
        ]
        self.current_dataset_epoch = int(
            state["current_dataset_epoch"]
        )
        self.completed_dataset_epochs = int(
            state["completed_dataset_epochs"]
        )
        self.seen_in_dataset_epoch = {
            int(task_index)
            for task_index in state[
                "seen_in_dataset_epoch"
            ]
        }
        self.rollout = int(state["rollout"])


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--algorithm",
        required=True,
        choices=[
            "mc",
            "sarsa",
            "q_learning",
            "dyna_q",
            "dqn",
            "reinforce",
            "a2c",
            "ppo",
            "grpo",
        ],
    )

    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/train.npz"),
        help=(
            "Training dataset .npz file, or a dataset directory "
            "containing train.npz, validation.npz, and "
            "same_layout_new_goals.npz."
        ),
    )

    parser.add_argument(
        "--rollouts-per-task",
        "--dataset-epochs",
        dest="rollouts_per_task",
        type=int,
        default=100,
        help=(
            "Training budget expressed as rollout attempts allowed "
            "per training task. --dataset-epochs is accepted as a "
            "backward-compatible alias."
        ),
    )

    parser.add_argument(
        "--fixed-index",
        type=int,
        default=None,
        help="Train on one fixed task index.",
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
            "Parent directory for training artifacts. Defaults to "
            "runs/tabular for tabular algorithms and "
            "runs/dnn/final_<algorithm>_best for DNN algorithms."
        ),
    )
    parser.add_argument(
        "--experiment-name",
        default=None,
        help=(
            "Optional prefix for automatically named experiment "
            "output under --experiment-root. The dataset metadata "
            "provides the size and seed suffix."
        ),
    )
    parser.add_argument(
        "--experiment-root",
        type=Path,
        default=DEFAULT_EXPERIMENT_OUTPUT_ROOT,
        help=(
            "Root directory for automatically named experiment "
            "training outputs."
        ),
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Restore the full training_state.pt state for this run, "
            "including model, optimizer, sampler, RNG, counters, "
            "pending metrics, and validation state."
        ),
    )

    parser.add_argument(
        "--plot-output",
        type=Path,
        default=None,
        help=(
            "Training plot path. Defaults to "
            "<run directory>/training_metrics.png."
        ),
    )

    parser.add_argument(
        "--rolling-window",
        type=int,
        default=0,
        help=(
            "Number of task-selection passes used for rolling means. "
            "Use 0 to disable rolling means."
        ),
    )

    parser.set_defaults(no_plot=True)
    parser.add_argument(
        "--plot",
        dest="no_plot",
        action="store_false",
        help=(
            "Generate training-metric plots at the end of training. "
            "Disabled by default."
        ),
    )
    parser.add_argument(
        "--no-plot",
        dest="no_plot",
        action="store_true",
        help=(
            "Skip training-metric plots at the end of training. "
            "This is the default."
        ),
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Skip periodic per-rollout progress messages.",
    )

    parser.add_argument(
        "--task-plot-output",
        type=Path,
        default=None,
        help=(
            "Per-task training plot path. Defaults to "
            "<run directory>/task_training_metrics.png."
        ),
    )

    parser.add_argument(
        "--task-html-output",
        type=Path,
        default=None,
        help=(
            "Interactive per-task training plot path. Defaults to "
            "<run directory>/task_training_metrics.html."
        ),
    )

    parser.add_argument(
        "--q-snapshot-count",
        type=int,
        default=101,
        help=(
            "Number of evenly spaced tabular Q-table snapshots to "
            "save in checkpoints. The first snapshot is task-selection "
            "pass 1."
        ),
    )

    parser.add_argument(
        "--tensorboard",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Write TensorBoard event logs. Defaults off; pass "
            "--tensorboard to enable them."
        ),
    )

    parser.add_argument(
        "--tensorboard-dir",
        type=Path,
        default=None,
        help=(
            "TensorBoard log directory. Defaults to "
            "<run directory>/tensorboard."
        ),
    )

    parser.add_argument(
        "--best-config",
        nargs="?",
        const=BEST_CONFIG_DEFAULT_SENTINEL,
        type=Path,
        default=None,
        help=(
            "Load the tuned best hyperparameter combination for "
            "--algorithm. Omitting the path reads "
            "runs/dnn/tuning/best_config_<algorithm>.json. Explicit "
            "CLI hyperparameters override loaded values."
        ),
    )

    parser.add_argument(
        "--task-batch-size",
        "--rollout-episodes",
        dest="task_batch_size",
        type=int,
        default=None,
        help=(
            "Distinct tasks selected per training round. "
            "--rollout-episodes is accepted as a "
            "backward-compatible alias."
        ),
    )

    parser.add_argument(
        "--validation-dataset",
        type=Path,
        default=None,
        help=(
            "Optional validation dataset for periodic neural-agent "
            "selection."
        ),
    )
    parser.add_argument(
        "--same-layout-dataset",
        type=Path,
        default=None,
        help=(
            "Optional same-layout new-task dataset to evaluate "
            "alongside validation during periodic neural-agent "
            "selection."
        ),
    )
    parser.add_argument(
        "--validation-interval",
        type=int,
        default=1,
        help=(
            "Task-selection-pass interval for validation when "
            "--validation-dataset is provided."
        ),
    )
    parser.add_argument(
        "--validation-episodes",
        type=int,
        default=200,
    )
    parser.add_argument(
        "--validation-all-tasks",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Evaluate every validation task. Use "
            "--no-validation-all-tasks with --validation-episodes "
            "for sampling."
        ),
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=35,
        help=(
            "Stop after this many validation checks without "
            "sufficient improvement. Requires --validation-dataset."
        ),
    )
    parser.add_argument(
        "--early-stopping-min-delta",
        type=float,
        default=0.0,
        help=(
            "Minimum validation mean path-efficiency improvement "
            "required to reset early-stopping patience."
        ),
    )
    parser.add_argument(
        "--retroactive-early-stop",
        action="store_true",
        help=(
            "Only with --resume: allow early-stopping "
            "patience/min-delta to differ from the stored state, "
            "replay existing validation_metrics.csv with the requested "
            "rule, and stop immediately if that history already meets "
            "the early-stopping condition."
        ),
    )

    neural_group = parser.add_argument_group(
        "neural agent hyperparameters"
    )

    neural_group.add_argument(
        "--learning-rate",
        type=float,
        default=None,
    )
    neural_group.add_argument(
        "--gamma",
        type=float,
        default=None,
    )
    neural_group.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="DQN replay minibatch size.",
    )
    neural_group.add_argument(
        "--train-frequency",
        type=int,
        default=None,
        help=(
            "DQN environment transitions between replay updates. "
            "Defaults to the effective DQN batch size."
        ),
    )
    neural_group.add_argument(
        "--minibatch-size",
        type=int,
        default=None,
        help="Policy-gradient optimization minibatch size.",
    )
    neural_group.add_argument(
        "--epsilon-start",
        type=float,
        default=None,
    )
    neural_group.add_argument(
        "--epsilon-min",
        type=float,
        default=None,
    )
    neural_group.add_argument(
        "--epsilon-decay",
        type=float,
        default=None,
    )
    neural_group.add_argument(
        "--replay-capacity",
        type=int,
        default=None,
    )
    neural_group.add_argument(
        "--min-replay-size",
        type=int,
        default=None,
    )
    neural_group.add_argument(
        "--target-update-interval",
        type=int,
        default=None,
    )
    neural_group.add_argument(
        "--gae-lambda",
        type=float,
        default=None,
    )
    neural_group.add_argument(
        "--value-coefficient",
        type=float,
        default=None,
    )
    neural_group.add_argument(
        "--entropy-coefficient",
        type=float,
        default=None,
    )
    neural_group.add_argument(
        "--entropy-coefficient-min",
        type=float,
        default=None,
    )
    neural_group.add_argument(
        "--entropy-coefficient-decay",
        type=float,
        default=None,
    )
    neural_group.add_argument(
        "--clip-epsilon",
        type=float,
        default=None,
    )
    neural_group.add_argument(
        "--update-epochs",
        type=int,
        default=None,
    )
    neural_group.add_argument(
        "--rollout-group-size",
        "--group-size",
        dest="rollout_group_size",
        type=int,
        default=None,
        help=(
            "Rollouts generated per selected task in a task batch. "
            "Defaults to 1 for older DNN agents and 8 for GRPO. "
            "--group-size is accepted as a backward-compatible "
            "alias."
        ),
    )

    argv = sys.argv[1:]
    args = parser.parse_args()
    args.resume_metadata_cli_fields = (
        resume_metadata_fields_from_argv(
            argv,
        )
    )

    if args.best_config == BEST_CONFIG_DEFAULT_SENTINEL:
        args.best_config = default_best_config_path(
            args.algorithm
        )

    if (
        args.output_dir is None
        and args.experiment_name is None
        and not args.dataset.is_dir()
    ):
        args.output_dir = default_output_dir(
            args.algorithm
        )

    return args


def load_best_config_overrides(
    path: Path,
    algorithm: str,
) -> BestConfigOverrides:
    with path.open() as file:
        best_config = json.load(file)

    if not isinstance(best_config, dict):
        raise ValueError(
            f"{path} must contain a JSON object."
        )

    if "hyperparameters" in best_config:
        selected_config = best_config
    else:
        selected_config = best_config.get(
            algorithm
        )

        if selected_config is None:
            raise ValueError(
                f"{path} does not contain a best config for "
                f"algorithm {algorithm!r}."
            )

    if not isinstance(selected_config, dict):
        raise ValueError(
            f"{path} best config for {algorithm!r} must be a JSON object."
        )

    config_algorithm = selected_config.get(
        "algorithm"
    )
    if (
        config_algorithm is not None
        and config_algorithm != algorithm
    ):
        raise ValueError(
            f"{path} contains algorithm {config_algorithm!r}, "
            f"but --algorithm is {algorithm!r}."
        )

    raw_hyperparameters = selected_config.get(
        "hyperparameters",
        {},
    )
    if not isinstance(raw_hyperparameters, dict):
        raise ValueError(
            f"{path} hyperparameters must be a JSON object."
        )

    hyperparameters = dict(raw_hyperparameters)
    task_batch_size = hyperparameters.pop(
        "task_batch_size",
        selected_config.get("task_batch_size"),
    )
    rollout_group_size = hyperparameters.pop(
        "rollout_group_size",
        selected_config.get("rollout_group_size"),
    )

    return BestConfigOverrides(
        path=path,
        hyperparameters=hyperparameters,
        task_batch_size=task_batch_size,
        rollout_group_size=rollout_group_size,
    )


def neural_hyperparameters_from_args(
    args,
) -> dict[str, float | int]:
    return {
        name: getattr(args, name)
        for name in NEURAL_HYPERPARAMETER_DESTS
        if getattr(args, name) is not None
    }


def validate_neural_hyperparameters(
    algorithm: str,
    hyperparameters: dict[str, float | int],
) -> None:
    if not hyperparameters:
        return

    if algorithm not in NEURAL_ALGORITHMS:
        names = ", ".join(
            sorted(hyperparameters)
        )
        raise ValueError(
            "Neural hyperparameters are only supported for "
            f"DNN agents, but got {algorithm}: {names}"
        )

    unsupported = sorted(
        set(hyperparameters)
        - NEURAL_HYPERPARAMETERS[algorithm]
    )

    if unsupported:
        names = ", ".join(unsupported)
        raise ValueError(
            f"{algorithm} does not use these hyperparameters: "
            f"{names}"
        )


def task_batch_size_for_algorithm(
    algorithm: str,
    task_batch_size: int | None,
) -> int:
    if task_batch_size is None:
        return DEFAULT_TASK_BATCH_SIZES.get(
            algorithm,
            1,
        )

    if task_batch_size <= 0:
        raise ValueError(
            "--task-batch-size must be positive."
        )

    return task_batch_size


def rollout_group_size_for_algorithm(
    algorithm: str,
    rollout_group_size: int | None,
) -> int:
    if rollout_group_size is None:
        return DEFAULT_ROLLOUT_GROUP_SIZES.get(
            algorithm,
            1,
        )

    if rollout_group_size <= 0:
        raise ValueError(
            "--rollout-group-size must be positive."
        )

    if (
        algorithm == "grpo"
        and rollout_group_size < 2
    ):
        raise ValueError(
            "--rollout-group-size must be at least 2 "
            "for GRPO."
        )

    if (
        rollout_group_size > 1
        and algorithm not in NEURAL_ALGORITHMS
    ):
        raise ValueError(
            "--rollout-group-size greater than 1 is only "
            "supported for DNN agents."
        )

    return rollout_group_size


def task_selection_passes_for_budget(
    rollouts_per_task: int,
    rollout_group_size: int,
) -> int:
    if rollouts_per_task <= 0:
        raise ValueError(
            "--rollouts-per-task must be positive."
        )

    return max(
        1,
        (
            rollouts_per_task
            + rollout_group_size
            - 1
        )
        // rollout_group_size,
    )


def rollout_episodes_for_algorithm(
    algorithm: str,
    rollout_episodes: int | None,
) -> int:
    return task_batch_size_for_algorithm(
        algorithm,
        rollout_episodes,
    )


def group_size_for_algorithm(
    algorithm: str,
    group_size: int | None,
) -> int:
    return rollout_group_size_for_algorithm(
        algorithm,
        group_size,
    )


def training_epoch_budget_for_algorithm(
    algorithm: str,
    dataset_epochs: int,
    group_size: int,
) -> int:
    del algorithm

    return task_selection_passes_for_budget(
        dataset_epochs,
        max(1, group_size),
    )


def create_agent(
    algorithm,
    env,
    device,
    seed,
    hyperparameters=None,
):
    hyperparameters = dict(
        hyperparameters or {}
    )
    validate_neural_hyperparameters(
        algorithm,
        hyperparameters,
    )

    kwargs = {
        "action_count": env.action_space.n,
    }

    if algorithm == "mc":
        return MonteCarloAgent(
            **kwargs,
            seed=seed,
        )

    if algorithm == "sarsa":
        return SarsaAgent(
            **kwargs,
            seed=seed,
        )

    if algorithm == "q_learning":
        return QLearningAgent(
            **kwargs,
            seed=seed,
        )

    if algorithm == "dyna_q":
        return DynaQAgent(
            **kwargs,
            seed=seed,
        )

    neural_kwargs = {
        **kwargs,
        "height": env.height,
        "width": env.width,
        "device": device,
    }

    if algorithm == "dqn":
        dqn_kwargs = {
            "seed": seed,
            "replay_capacity": (
                64 * env.max_steps
            ),
        }
        dqn_kwargs.update(
            hyperparameters
        )

        return DQNAgent(
            **neural_kwargs,
            **dqn_kwargs,
        )

    if algorithm == "reinforce":
        return ReinforceAgent(
            **neural_kwargs,
            **hyperparameters,
        )

    if algorithm == "a2c":
        return A2CAgent(
            **neural_kwargs,
            **hyperparameters,
        )

    if algorithm == "ppo":
        return PPOAgent(
            **neural_kwargs,
            **hyperparameters,
        )

    if algorithm == "grpo":
        return GRPOAgent(
            **neural_kwargs,
            **hyperparameters,
        )

    raise ValueError(algorithm)


def save_checkpoint(
    path,
    algorithm,
    agent,
    q_snapshots=None,
    model_snapshots=None,
    hyperparameters=None,
):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if algorithm in TABULAR_ALGORITHMS:
        with path.open("wb") as file:
            pickle.dump(
                {
                    "algorithm": algorithm,
                    "q": agent.q_state_dict(),
                    "q_snapshots": q_snapshots or [],
                },
                file,
            )

        return

    if algorithm == "dqn":
        state_dict = (
            agent.online_network.state_dict()
        )

    elif algorithm in {
        "reinforce",
        "grpo",
    }:
        state_dict = (
            agent.policy.state_dict()
        )

    else:
        state_dict = (
            agent.network.state_dict()
        )

    torch.save(
        {
            "algorithm": algorithm,
            "model_state_dict": state_dict,
            "model_snapshots": model_snapshots or [],
            "hyperparameters": hyperparameters or {},
        },
        path,
    )


def current_neural_state_dict(
    algorithm,
    agent,
):
    if algorithm == "dqn":
        state_dict = (
            agent.online_network.state_dict()
        )

    elif algorithm in {
        "reinforce",
        "grpo",
    }:
        state_dict = (
            agent.policy.state_dict()
        )

    else:
        state_dict = (
            agent.network.state_dict()
        )

    return {
        key: value.detach().cpu().clone()
        for key, value in state_dict.items()
    }


def model_snapshot_state_dict(
    algorithm,
    agent,
):
    return current_neural_state_dict(
        algorithm,
        agent,
    )


def save_neural_checkpoint_state(
    path: Path,
    algorithm: str,
    model_state_dict,
    model_snapshots=None,
    hyperparameters=None,
    validation_metadata=None,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "algorithm": algorithm,
            "model_state_dict": model_state_dict,
            "model_snapshots": model_snapshots or [],
            "hyperparameters": hyperparameters or {},
            "validation": validation_metadata or {},
        },
        path,
    )


def network_module_for_algorithm(
    algorithm: str,
    agent,
):
    if algorithm == "dqn":
        return agent.online_network

    if algorithm in {
        "reinforce",
        "grpo",
    }:
        return agent.policy

    return agent.network


def agent_training_state_dict(
    algorithm: str,
    agent,
) -> dict:
    state = {
        "algorithm": algorithm,
    }

    if algorithm in TABULAR_ALGORITHMS:
        state.update(
            {
                "q": agent.q_state_dict(),
                "rng_state": agent.rng.bit_generator.state,
            }
        )

        if hasattr(agent, "model"):
            state["model"] = dict(agent.model)

        return state

    if algorithm == "dqn":
        state.update(
            {
                "online_network_state_dict": (
                    agent.online_network.state_dict()
                ),
                "target_network_state_dict": (
                    agent.target_network.state_dict()
                ),
                "optimizer_state_dict": (
                    agent.optimizer.state_dict()
                ),
                "epsilon": agent.epsilon,
                "environment_steps": (
                    agent.environment_steps
                ),
                "training_steps": agent.training_steps,
                "rng_state": agent.rng.bit_generator.state,
                "replay_buffer": list(
                    agent.replay_buffer.buffer
                ),
                "replay_buffer_rng_state": (
                    agent.replay_buffer
                    .rng
                    .bit_generator
                    .state
                ),
            }
        )
        return state

    state.update(
        {
            "model_state_dict": (
                network_module_for_algorithm(
                    algorithm,
                    agent,
                ).state_dict()
            ),
            "optimizer_state_dict": (
                agent.optimizer.state_dict()
            ),
        }
    )

    if hasattr(
        agent,
        "entropy_coefficient",
    ):
        state["entropy_coefficient"] = (
            agent.entropy_coefficient
        )

    return state


def load_agent_training_state(
    algorithm: str,
    agent,
    state: dict,
) -> None:
    if state.get("algorithm") != algorithm:
        raise RuntimeError(
            "Training state algorithm "
            f"{state.get('algorithm')!r} does not match "
            f"requested algorithm {algorithm!r}."
        )

    if algorithm in TABULAR_ALGORITHMS:
        agent.load_q_state_dict(state["q"])
        agent.rng.bit_generator.state = state[
            "rng_state"
        ]

        if hasattr(agent, "model"):
            agent.model = dict(
                state.get(
                    "model",
                    {},
                )
            )

        return

    if algorithm == "dqn":
        agent.online_network.load_state_dict(
            state[
                "online_network_state_dict"
            ]
        )
        agent.target_network.load_state_dict(
            state[
                "target_network_state_dict"
            ]
        )
        agent.optimizer.load_state_dict(
            state[
                "optimizer_state_dict"
            ]
        )
        agent.epsilon = float(state["epsilon"])
        agent.environment_steps = int(
            state["environment_steps"]
        )
        agent.training_steps = int(
            state["training_steps"]
        )
        agent.rng.bit_generator.state = state[
            "rng_state"
        ]
        agent.replay_buffer.buffer.clear()
        agent.replay_buffer.buffer.extend(
            state["replay_buffer"]
        )
        agent.replay_buffer.rng.bit_generator.state = (
            state[
                "replay_buffer_rng_state"
            ]
        )
        return

    network_module_for_algorithm(
        algorithm,
        agent,
    ).load_state_dict(
        state["model_state_dict"]
    )
    agent.optimizer.load_state_dict(
        state["optimizer_state_dict"]
    )

    if "entropy_coefficient" in state:
        agent.entropy_coefficient = float(
            state["entropy_coefficient"]
        )


def runtime_rng_state_dict() -> dict:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }

    if torch.cuda.is_available():
        state["torch_cuda"] = (
            torch.cuda.get_rng_state_all()
        )

    return state


def torch_rng_state_tensor(
    state,
) -> torch.Tensor:
    if isinstance(
        state,
        torch.Tensor,
    ):
        return (
            state.detach()
            .cpu()
            .to(dtype=torch.uint8)
            .contiguous()
        )

    return torch.as_tensor(
        state,
        dtype=torch.uint8,
        device="cpu",
    ).contiguous()


def load_runtime_rng_state(
    state: dict,
) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(
        torch_rng_state_tensor(
            state["torch"]
        )
    )

    if (
        torch.cuda.is_available()
        and "torch_cuda" in state
    ):
        torch.cuda.set_rng_state_all(
            [
                torch_rng_state_tensor(
                    cuda_state
                )
                for cuda_state in state[
                    "torch_cuda"
                ]
            ]
        )


def training_state_metadata(
    *,
    args,
    agent_hyperparameters: dict[str, float | int],
    task_selection_pass_budget: int,
    task_batch_size: int,
    rollout_group_size: int,
) -> dict:
    return {
        "algorithm": args.algorithm,
        "agent_hyperparameters": dict(
            agent_hyperparameters
        ),
        "dataset": str(args.dataset),
        "fixed_index": args.fixed_index,
        "max_steps": args.max_steps,
        "seed": args.seed,
        "rollouts_per_task": args.rollouts_per_task,
        "task_selection_pass_budget": (
            task_selection_pass_budget
        ),
        "task_batch_size": task_batch_size,
        "rollout_group_size": rollout_group_size,
        "validation_dataset": (
            None
            if args.validation_dataset is None
            else str(args.validation_dataset)
        ),
        "same_layout_dataset": (
            None
            if args.same_layout_dataset is None
            else str(args.same_layout_dataset)
        ),
        "validation_interval": args.validation_interval,
        "validation_episodes": args.validation_episodes,
        "validation_all_tasks": args.validation_all_tasks,
        "early_stopping_patience": (
            args.early_stopping_patience
        ),
        "early_stopping_min_delta": (
            args.early_stopping_min_delta
        ),
    }


def validate_training_state_metadata(
    state: dict,
    expected_metadata: dict,
    *,
    allowed_mismatches: set[str] | None = None,
) -> None:
    metadata = state.get("metadata")
    allowed_mismatches = allowed_mismatches or set()

    if metadata == expected_metadata:
        return

    comparable_metadata = dict(
        metadata or {}
    )
    comparable_expected = dict(
        expected_metadata
    )

    for key in allowed_mismatches:
        if key in comparable_metadata:
            comparable_metadata[key] = (
                comparable_expected.get(key)
            )

    if comparable_metadata != comparable_expected:
        mismatched_keys = sorted(
            {
                *comparable_metadata.keys(),
                *comparable_expected.keys(),
            }
        )
        mismatched_keys = [
            key
            for key in mismatched_keys
            if comparable_metadata.get(key)
            != comparable_expected.get(key)
        ]
        raise RuntimeError(
            "Stored training state does not match the "
            "requested training configuration. "
            f"Mismatched keys: {', '.join(mismatched_keys)}"
        )


def load_training_state_metadata(
    path: Path,
) -> dict:
    if not path.exists():
        raise RuntimeError(
            "--resume requested, but "
            f"{path} does not exist."
        )

    state = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )
    metadata = state.get("metadata")

    if not isinstance(
        metadata,
        dict,
    ):
        raise RuntimeError(
            f"{path} does not contain training metadata."
        )

    return metadata


def existing_training_output_paths(
    paths,
) -> list[Path]:
    return [
        path
        for path in paths
        if path.exists()
    ]


def confirm_overwrite_existing_training_output(
    existing_paths,
    *,
    interactive: bool | None = None,
    input_fn=input,
    output_stream=None,
) -> None:
    existing_paths = list(
        existing_paths
    )

    if not existing_paths:
        return

    if output_stream is None:
        output_stream = sys.stderr

    print(
        "WARNING: existing training outputs were found and this "
        "run was not started with --resume.",
        file=output_stream,
    )
    print(
        "Starting fresh will delete or overwrite:",
        file=output_stream,
    )

    for path in existing_paths:
        print(
            f"  - {path}",
            file=output_stream,
        )

    print(
        "Use --resume to continue from training_state.pt instead.",
        file=output_stream,
    )

    if interactive is None:
        interactive = sys.stdin.isatty()

    if not interactive:
        raise RuntimeError(
            "Refusing to overwrite existing training outputs without "
            "interactive confirmation."
        )

    response = input_fn(
        "Type OVERWRITE to delete/overwrite these files and start "
        "fresh: "
    )

    if response != "OVERWRITE":
        raise RuntimeError(
            "Aborted fresh training run; existing outputs were left "
            "unchanged."
        )


def remove_existing_training_output(
    existing_paths,
) -> None:
    for path in existing_paths:
        if path.exists():
            path.unlink()


def optional_metadata_path(
    metadata: dict,
    name: str,
) -> Path | None:
    value = metadata.get(name)

    if value is None:
        return None

    return Path(value)


def apply_retroactive_resume_metadata(
    args,
    metadata: dict,
) -> None:
    if metadata.get("algorithm") != args.algorithm:
        raise RuntimeError(
            "Stored training state algorithm does not match "
            f"--algorithm {args.algorithm!r}."
        )

    args.dataset = Path(
        metadata["dataset"]
    )
    args.fixed_index = metadata[
        "fixed_index"
    ]
    args.max_steps = int(
        metadata["max_steps"]
    )
    args.seed = int(
        metadata["seed"]
    )
    args.rollouts_per_task = int(
        metadata["rollouts_per_task"]
    )
    args.task_batch_size = metadata.get(
        "task_batch_size"
    )
    args.rollout_group_size = metadata.get(
        "rollout_group_size"
    )
    args.validation_dataset = optional_metadata_path(
        metadata,
        "validation_dataset",
    )
    args.same_layout_dataset = optional_metadata_path(
        metadata,
        "same_layout_dataset",
    )
    args.validation_interval = int(
        metadata["validation_interval"]
    )
    args.validation_episodes = int(
        metadata["validation_episodes"]
    )
    args.validation_all_tasks = bool(
        metadata["validation_all_tasks"]
    )


def apply_resume_training_state_defaults(
    args,
    metadata: dict,
) -> None:
    if metadata.get("algorithm") != args.algorithm:
        raise RuntimeError(
            "Stored training state algorithm does not match "
            f"--algorithm {args.algorithm!r}."
        )

    provided_fields = getattr(
        args,
        "resume_metadata_cli_fields",
        set(),
    )
    metadata_defaults = {
        "dataset": Path(metadata["dataset"]),
        "fixed_index": metadata["fixed_index"],
        "max_steps": int(metadata["max_steps"]),
        "seed": int(metadata["seed"]),
        "rollouts_per_task": int(
            metadata["rollouts_per_task"]
        ),
        "task_batch_size": metadata.get(
            "task_batch_size"
        ),
        "rollout_group_size": metadata.get(
            "rollout_group_size"
        ),
        "validation_dataset": optional_metadata_path(
            metadata,
            "validation_dataset",
        ),
        "same_layout_dataset": optional_metadata_path(
            metadata,
            "same_layout_dataset",
        ),
        "validation_interval": int(
            metadata["validation_interval"]
        ),
        "validation_episodes": int(
            metadata["validation_episodes"]
        ),
        "validation_all_tasks": bool(
            metadata["validation_all_tasks"]
        ),
        "early_stopping_patience": metadata[
            "early_stopping_patience"
        ],
        "early_stopping_min_delta": float(
            metadata["early_stopping_min_delta"]
        ),
    }

    for field, value in metadata_defaults.items():
        if field not in provided_fields:
            setattr(
                args,
                field,
                value,
            )


def resume_training_state_allowed_mismatches(
    args,
) -> set[str]:
    provided_fields = getattr(
        args,
        "resume_metadata_cli_fields",
        set(),
    )
    allowed_by_field = {
        "rollouts_per_task": {
            "rollouts_per_task",
            "task_selection_pass_budget",
        },
        "validation_interval": {
            "validation_interval",
        },
        "validation_episodes": {
            "validation_episodes",
        },
        "validation_all_tasks": {
            "validation_all_tasks",
        },
        "early_stopping_patience": {
            "early_stopping_patience",
        },
        "early_stopping_min_delta": {
            "early_stopping_min_delta",
        },
    }
    allowed = set()

    for field in provided_fields:
        allowed.update(
            allowed_by_field.get(
                field,
                set(),
            )
        )

    return allowed


def file_size(path: Path) -> int | None:
    if not path.exists():
        return None

    return path.stat().st_size


def truncate_file(
    path: Path,
    size: int | None,
) -> None:
    if size is None or not path.exists():
        return

    with path.open("r+b") as file:
        file.truncate(size)


def save_training_state(
    path: Path,
    state: dict,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    temporary_path = path.with_suffix(
        path.suffix + ".tmp"
    )
    torch.save(
        state,
        temporary_path,
    )
    temporary_path.replace(path)


def load_training_state(
    path: Path,
    device: torch.device,
) -> dict:
    try:
        return torch.load(
            path,
            map_location=device,
            weights_only=False,
        )
    except TypeError:
        return torch.load(
            path,
            map_location=device,
        )


def should_validate_epoch(
    epoch: int,
    final_epoch: int,
    validation_interval: int,
) -> bool:
    return (
        validation_interval > 0
        and (
            epoch % validation_interval == 0
            or epoch == final_epoch
        )
    )


def should_stop_early(
    patience: int | None,
    checks_without_improvement: int,
    has_positive_validation_score: bool,
) -> bool:
    return (
        patience is not None
        and has_positive_validation_score
        and checks_without_improvement >= patience
    )


def parse_validation_csv_row(
    row: dict[str, str],
) -> dict:
    return {
        "split": row.get(
            "split",
            "validation",
        ),
        "dataset_epoch": int(
            row["dataset_epoch"]
        ),
        "episodes": int(
            row["episodes"]
        ),
        "success_rate": float(
            row["success_rate"]
        ),
        "average_episode_return": float(
            row["average_episode_return"]
        ),
        "mean_path_efficiency": float(
            row["mean_path_efficiency"]
        ),
        "average_successful_path_efficiency": float(
            row[
                "average_successful_path_efficiency"
            ]
        ),
    }


TRAIN_GREEDY_VALIDATION_SPLITS = {
    "train",
    "train_greedy",
    "greedy_train",
    "train_eval",
}


def validation_metrics_have_train_greedy_split(
    path: Path,
) -> bool:
    if not path.exists():
        return False

    with path.open(
        newline="",
    ) as file:
        reader = csv.DictReader(file)

        for row in reader:
            if (
                row.get(
                    "split",
                    "validation",
                )
                in TRAIN_GREEDY_VALIDATION_SPLITS
            ):
                return True

    return False


def should_record_train_greedy_validation(
    *,
    resuming: bool,
    validation_metrics_path: Path,
) -> bool:
    if not resuming:
        return True

    return validation_metrics_have_train_greedy_split(
        validation_metrics_path
    )


def replay_early_stopping_from_validation_metrics(
    path: Path,
    *,
    patience: int | None,
    min_delta: float,
) -> dict:
    best_validation = None
    latest_validation_by_split = {}
    checks_without_improvement = 0
    has_positive_validation_score = False
    stopped_early = False
    stopped_epoch = None

    if not path.exists():
        return {
            "best_validation": best_validation,
            "latest_validation_by_split": (
                latest_validation_by_split
            ),
            "validation_checks_without_improvement": (
                checks_without_improvement
            ),
            "validation_has_positive_score": (
                has_positive_validation_score
            ),
            "stopped_early": stopped_early,
            "stopped_epoch": stopped_epoch,
        }

    with path.open(
        newline="",
    ) as file:
        reader = csv.DictReader(file)

        for raw_row in reader:
            row = parse_validation_csv_row(
                raw_row
            )
            split = row["split"]
            latest_validation_by_split[
                split
            ] = row

            if split != "validation":
                continue

            validation_score = row[
                "mean_path_efficiency"
            ]

            if validation_score > 0.0:
                has_positive_validation_score = True

            improved = (
                best_validation is None
                or validation_score
                > best_validation[
                    "mean_path_efficiency"
                ]
                + min_delta
            )

            if improved:
                checks_without_improvement = 0
                best_validation = row
            else:
                checks_without_improvement += 1

            if (
                not stopped_early
                and should_stop_early(
                    patience,
                    checks_without_improvement,
                    has_positive_validation_score,
                )
            ):
                stopped_early = True
                stopped_epoch = row[
                    "dataset_epoch"
                ]

    return {
        "best_validation": best_validation,
        "latest_validation_by_split": (
            latest_validation_by_split
        ),
        "validation_checks_without_improvement": (
            checks_without_improvement
        ),
        "validation_has_positive_score": (
            has_positive_validation_score
        ),
        "stopped_early": stopped_early,
        "stopped_epoch": stopped_epoch,
    }


def append_validation_csv(
    path: Path,
    row: dict,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "split",
        "dataset_epoch",
        "episodes",
        "success_rate",
        "average_episode_return",
        "mean_path_efficiency",
        "average_successful_path_efficiency",
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

        writer.writerow(row)


def write_training_summary(
    path: Path,
    summary: dict,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open("w") as file:
        json.dump(
            summary,
            file,
            indent=2,
            sort_keys=True,
        )


def format_progress_message(
    algorithm: str,
    metrics,
    agent,
) -> str:
    message = (
        f"{algorithm:12s} "
        f"task_selection_pass={metrics.epoch:5d} "
        f"rollout={metrics.episode:7d} "
        f"task={metrics.task_index:5d} "
        f"episode_return="
        f"{metrics.episode_return:7.3f} "
        f"steps={metrics.steps:3d} "
        f"success={metrics.success} "
        f"efficiency="
        f"{metrics.path_efficiency:.3f}"
    )

    if metrics.mean_abs_td_error is not None:
        message += (
            " mean_abs_td_error="
            f"{metrics.mean_abs_td_error:.6f}"
        )

    if algorithm == "dqn":
        message += (
            f" epsilon="
            f"{agent.epsilon:.3f}"
        )

    return message


def format_task_selection_pass_progress_prefix(
    dataset_epoch: int,
    task_selection_pass_budget: int,
) -> str:
    progress_percent = (
        dataset_epoch
        / task_selection_pass_budget
        * 100.0
    )
    return (
        "pass "
        f"{dataset_epoch:>{len(str(task_selection_pass_budget))}d}/"
        f"{task_selection_pass_budget} "
        f"[{progress_percent:.1f}%] | "
    )


def format_task_selection_pass_task_progress_message(
    dataset_epoch: int,
    task_selection_pass_budget: int,
    completed_tasks: int,
    total_tasks: int,
) -> str:
    task_progress_percent = (
        100.0
        if total_tasks <= 0
        else completed_tasks / total_tasks * 100.0
    )
    return (
        format_task_selection_pass_progress_prefix(
            dataset_epoch,
            task_selection_pass_budget,
        )
        + "task "
        f"{completed_tasks}/{total_tasks} "
        f"[{task_progress_percent:.1f}%]"
    )


def format_progress_continuation_prefix(
    dataset_epoch: int,
    task_selection_pass_budget: int,
) -> str:
    return " " * len(
        format_task_selection_pass_progress_prefix(
            dataset_epoch,
            task_selection_pass_budget,
        )
    )


def format_training_epoch_progress_message(
    dataset_epoch: int,
    task_selection_pass_budget: int,
    epoch_metrics,
    total_tasks: int | None = None,
    early_stopping_counter: int | None = None,
    early_stopping_patience: int | None = None,
) -> str:
    success_rate = float(
        np.mean(
            [
                metrics.success
                for metrics in epoch_metrics
            ]
        )
    )
    mean_path_efficiency = float(
        np.mean(
            [
                metrics.path_efficiency
                for metrics in epoch_metrics
            ]
        )
    )
    mean_steps = float(
        np.mean(
            [
                metrics.steps
                for metrics in epoch_metrics
            ]
        )
    )
    early_stopping_status = (
        "-"
        if (
            early_stopping_counter is None
            or early_stopping_patience is None
        )
        else f"{early_stopping_counter}/{early_stopping_patience}"
    )
    completed_tasks = (
        len(
            {
                int(metrics.task_index)
                for metrics in epoch_metrics
            }
        )
        if total_tasks is None
        else total_tasks
    )
    pass_task_count = (
        max(completed_tasks, 1)
        if total_tasks is None
        else total_tasks
    )

    return (
        format_task_selection_pass_task_progress_message(
            dataset_epoch,
            task_selection_pass_budget,
            completed_tasks,
            pass_task_count,
        )
        + "\n"
        + format_progress_continuation_prefix(
            dataset_epoch,
            task_selection_pass_budget,
        )
        + f"{'train':<12s} eff={mean_path_efficiency:.3f} "
        f"succ={success_rate:.3f} "
        f"steps={mean_steps:.1f} "
        f"es={early_stopping_status}"
    )


def format_validation_progress_message(
    split: str,
    dataset_epoch: int,
    task_selection_pass_budget: int,
    validation_score: float,
    success_rate: float | None = None,
    best_score: float | None = None,
) -> str:
    split_label = {
        "validation": "val",
        "same_layout": "same-val",
        "train_greedy": "train-greedy",
    }.get(split, split)
    message = (
        format_progress_continuation_prefix(
            dataset_epoch,
            task_selection_pass_budget,
        )
        + f"{split_label:<12s} eff={validation_score:.3f}"
    )

    if success_rate is not None:
        message += f" succ={success_rate:.3f}"

    if best_score is not None:
        message += f" (best={best_score:.3f})"

    return message


def create_tensorboard_writer(
    args,
    run_dir: Path,
):
    enabled = (
        tensorboard_default_enabled(
            args.algorithm
        )
        if args.tensorboard is None
        else args.tensorboard
    )

    if not enabled:
        return None

    from torch.utils.tensorboard import SummaryWriter

    log_dir = (
        args.tensorboard_dir
        if args.tensorboard_dir is not None
        else run_dir / "tensorboard"
    )

    writer = SummaryWriter(
        log_dir=str(log_dir)
    )

    print(
        f"Writing TensorBoard logs to {log_dir}"
    )

    return writer


def _mean_present(
    values,
) -> float | None:
    present_values = [
        value
        for value in values
        if value is not None
    ]

    if not present_values:
        return None

    return float(
        np.mean(present_values)
    )


def _write_tensorboard_scalars(
    writer,
    prefix: str,
    scalars,
    step: int,
) -> None:
    for name, value in scalars:
        if value is not None:
            writer.add_scalar(
                f"{prefix}/{name}",
                value,
                step,
            )


def episode_tensorboard_scalars(
    metrics,
    agent,
) -> list[tuple[str, float | int | None]]:
    scalars = [
        (
            "return",
            metrics.episode_return,
        ),
        (
            "success",
            float(metrics.success),
        ),
        (
            "steps",
            metrics.steps,
        ),
        (
            "path_efficiency",
            metrics.path_efficiency,
        ),
        (
            "wall_collisions",
            metrics.wall_collisions,
        ),
        (
            "internal_updates",
            metrics.internal_updates,
        ),
        (
            "mean_abs_td_error",
            metrics.mean_abs_td_error,
        ),
    ]

    if metrics.loss is not None:
        scalars.append(
            (
                "loss/total",
                metrics.loss,
            )
        )

    for field_name, name in [
        (
            "policy_loss",
            "loss/policy",
        ),
        (
            "value_loss",
            "loss/value",
        ),
        (
            "entropy",
            "policy/entropy",
        ),
        (
            "approximate_kl",
            "policy/approximate_kl",
        ),
        (
            "clip_fraction",
            "policy/clip_fraction",
        ),
        (
            "mean_value",
            "value/mean_prediction",
        ),
        (
            "mean_return",
            "value/mean_return",
        ),
        (
            "mean_advantage",
            "value/mean_advantage",
        ),
        (
            "mean_q_value",
            "dqn/mean_selected_q_value",
        ),
        (
            "max_q_value",
            "dqn/max_q_value",
        ),
        (
            "replay_size",
            "dqn/replay_size",
        ),
    ]:
        value = getattr(
            metrics,
            field_name,
        )

        if value is not None:
            scalars.append(
                (
                    name,
                    value,
                )
            )

    if hasattr(agent, "epsilon"):
        scalars.append(
            (
                "agent/epsilon",
                agent.epsilon,
            )
        )

    if hasattr(
        agent,
        "entropy_coefficient",
    ):
        scalars.append(
            (
                "agent/entropy_coefficient",
                agent.entropy_coefficient,
            )
        )

    return scalars


def aggregate_tensorboard_scalars(
    metrics_list,
    agent,
) -> list[tuple[str, float | None]]:
    scalars = [
        (
            "mean_return",
            float(
                np.mean(
                    [
                        metrics.episode_return
                        for metrics in metrics_list
                    ]
                )
            ),
        ),
        (
            "success_rate",
            float(
                np.mean(
                    [
                        metrics.success
                        for metrics in metrics_list
                    ]
                )
            ),
        ),
        (
            "mean_steps",
            float(
                np.mean(
                    [
                        metrics.steps
                        for metrics in metrics_list
                    ]
                )
            ),
        ),
        (
            "mean_path_efficiency",
            float(
                np.mean(
                    [
                        metrics.path_efficiency
                        for metrics in metrics_list
                    ]
                )
            ),
        ),
        (
            "mean_wall_collisions",
            float(
                np.mean(
                    [
                        metrics.wall_collisions
                        for metrics in metrics_list
                    ]
                )
            ),
        ),
        (
            "mean_internal_updates",
            _mean_present(
                [
                    metrics.internal_updates
                    for metrics in metrics_list
                ]
            ),
        ),
        (
            "mean_abs_td_error",
            _mean_present(
                [
                    metrics.mean_abs_td_error
                    for metrics in metrics_list
                ]
            ),
        ),
        (
            "loss/mean_total",
            _mean_present(
                [
                    metrics.loss
                    for metrics in metrics_list
                ]
            ),
        ),
        (
            "loss/mean_policy",
            _mean_present(
                [
                    metrics.policy_loss
                    for metrics in metrics_list
                ]
            ),
        ),
        (
            "loss/mean_value",
            _mean_present(
                [
                    metrics.value_loss
                    for metrics in metrics_list
                ]
            ),
        ),
        (
            "policy/mean_entropy",
            _mean_present(
                [
                    metrics.entropy
                    for metrics in metrics_list
                ]
            ),
        ),
        (
            "policy/mean_approximate_kl",
            _mean_present(
                [
                    metrics.approximate_kl
                    for metrics in metrics_list
                ]
            ),
        ),
        (
            "policy/mean_clip_fraction",
            _mean_present(
                [
                    metrics.clip_fraction
                    for metrics in metrics_list
                ]
            ),
        ),
        (
            "value/mean_prediction",
            _mean_present(
                [
                    metrics.mean_value
                    for metrics in metrics_list
                ]
            ),
        ),
        (
            "value/mean_return",
            _mean_present(
                [
                    metrics.mean_return
                    for metrics in metrics_list
                ]
            ),
        ),
        (
            "value/mean_advantage",
            _mean_present(
                [
                    metrics.mean_advantage
                    for metrics in metrics_list
                ]
            ),
        ),
        (
            "dqn/mean_selected_q_value",
            _mean_present(
                [
                    metrics.mean_q_value
                    for metrics in metrics_list
                ]
            ),
        ),
        (
            "dqn/max_q_value",
            _mean_present(
                [
                    metrics.max_q_value
                    for metrics in metrics_list
                ]
            ),
        ),
        (
            "dqn/mean_replay_size",
            _mean_present(
                [
                    metrics.replay_size
                    for metrics in metrics_list
                ]
            ),
        ),
    ]

    if hasattr(agent, "epsilon"):
        scalars.append(
            (
                "agent/epsilon",
                agent.epsilon,
            )
        )

    if hasattr(
        agent,
        "entropy_coefficient",
    ):
        scalars.append(
            (
                "agent/entropy_coefficient",
                agent.entropy_coefficient,
            )
        )

    return scalars


def log_tensorboard_episode(
    writer,
    metrics,
    agent,
    transition_step: int,
) -> None:
    if writer is None:
        return

    scalars = episode_tensorboard_scalars(
        metrics,
        agent,
    )

    _write_tensorboard_scalars(
        writer,
        "by_rollout",
        scalars,
        metrics.episode,
    )
    _write_tensorboard_scalars(
        writer,
        "by_transition",
        scalars,
        transition_step,
    )


def log_tensorboard_epoch(
    writer,
    epoch: int,
    epoch_metrics,
    agent,
) -> None:
    if writer is None or not epoch_metrics:
        return

    _write_tensorboard_scalars(
        writer,
        "by_dataset_epoch",
        aggregate_tensorboard_scalars(
            epoch_metrics,
            agent,
        ),
        epoch,
    )

    writer.flush()


def log_tensorboard_training_round(
    writer,
    training_round: int,
    round_metrics,
    agent,
) -> None:
    if writer is None or not round_metrics:
        return

    _write_tensorboard_scalars(
        writer,
        "by_training_round",
        aggregate_tensorboard_scalars(
            round_metrics,
            agent,
        ),
        training_round,
    )
    writer.flush()


def log_tensorboard_optimization_epoch(
    writer,
    optimization_epoch: int,
    round_metrics,
    agent,
) -> None:
    if writer is None or not round_metrics:
        return

    _write_tensorboard_scalars(
        writer,
        "by_optimization_epoch",
        aggregate_tensorboard_scalars(
            round_metrics,
            agent,
        ),
        optimization_epoch,
    )
    writer.flush()


def optimization_epochs_for_round(
    algorithm: str,
    agent,
    round_metrics,
) -> int:
    if not any(
        metrics.internal_updates
        for metrics in round_metrics
    ):
        return 0

    if algorithm == "ppo":
        return int(agent.update_epochs)

    if algorithm in {
        "reinforce",
        "a2c",
        "grpo",
    }:
        return 1

    return sum(
        int(metrics.internal_updates or 0)
        for metrics in round_metrics
    )


def main():
    args = parse_args()
    resolve_dataset_bundle_args(args)

    if args.output_dir is None:
        args.output_dir = default_experiment_output_dir(
            experiment_name=args.experiment_name,
            dataset_path=args.dataset,
            algorithm=args.algorithm,
            experiment_root=args.experiment_root,
        )

    if (
        args.retroactive_early_stop
        and not args.resume
    ):
        raise ValueError(
            "--retroactive-early-stop requires "
            "--resume."
        )

    resume_training_metadata = None
    retroactive_resume_metadata = None

    if args.resume:
        training_state_path = (
            args.output_dir
            / args.algorithm
            / "training_state.pt"
        )
        resume_training_metadata = (
            load_training_state_metadata(
                training_state_path
            )
        )

        if args.retroactive_early_stop:
            retroactive_resume_metadata = (
                resume_training_metadata
            )
            apply_retroactive_resume_metadata(
                args,
                retroactive_resume_metadata,
            )

            if args.validation_dataset is None:
                raise ValueError(
                    "--retroactive-early-stop requires a stored "
                    "validation dataset."
                )
        else:
            apply_resume_training_state_defaults(
                args,
                resume_training_metadata,
            )

    configure_reproducibility(args.seed)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    env = MazeEnv(
        dataset_path=args.dataset,
        max_steps=args.max_steps,
        fixed_index=args.fixed_index,
    )

    best_config_overrides = None
    if (
        args.best_config is not None
        and resume_training_metadata is None
    ):
        if args.algorithm not in NEURAL_ALGORITHMS:
            raise ValueError(
                "--best-config is only supported for DNN agents."
            )

        best_config_overrides = load_best_config_overrides(
            args.best_config,
            args.algorithm,
        )

    loaded_hyperparameters = (
        dict(
            resume_training_metadata[
                "agent_hyperparameters"
            ]
        )
        if resume_training_metadata is not None
        else (
            {}
            if best_config_overrides is None
            else best_config_overrides.hyperparameters
        )
    )
    explicit_hyperparameters = neural_hyperparameters_from_args(
        args
    )
    agent_hyperparameters = {
        **loaded_hyperparameters,
        **explicit_hyperparameters,
    }
    config_task_batch_size = (
        None
        if best_config_overrides is None
        else best_config_overrides.task_batch_size
    )
    task_batch_size = task_batch_size_for_algorithm(
        args.algorithm,
        (
            args.task_batch_size
            if args.task_batch_size is not None
            else config_task_batch_size
        ),
    )
    config_rollout_group_size = (
        None
        if best_config_overrides is None
        else best_config_overrides.rollout_group_size
    )
    rollout_group_size = (
        rollout_group_size_for_algorithm(
            args.algorithm,
            (
                args.rollout_group_size
                if args.rollout_group_size is not None
                else config_rollout_group_size
            ),
        )
    )
    task_selection_pass_budget = (
        task_selection_passes_for_budget(
            args.rollouts_per_task,
            rollout_group_size,
        )
    )

    agent = create_agent(
        algorithm=args.algorithm,
        env=env,
        device=device,
        seed=args.seed,
        hyperparameters=agent_hyperparameters,
    )

    if args.algorithm == "dqn":
        agent_hyperparameters.setdefault(
            "train_frequency",
            agent.train_frequency,
        )

    run_dir = (
        args.output_dir
        / args.algorithm
    )

    tensorboard_writer = create_tensorboard_writer(
        args,
        run_dir,
    )

    metrics_path = (
        run_dir / "metrics.csv"
    )
    validation_metrics_path = (
        run_dir / "validation_metrics.csv"
    )
    training_state_path = (
        run_dir / "training_state.pt"
    )
    best_checkpoint_path = (
        run_dir / "best_checkpoint.pt"
    )
    training_summary_path = (
        run_dir / "training_summary.json"
    )

    checkpoint_suffix = (
        ".pkl"
        if args.algorithm
        in TABULAR_ALGORITHMS
        else ".pt"
    )

    checkpoint_path = (
        run_dir
        / f"checkpoint{checkpoint_suffix}"
    )

    existing_output_paths = existing_training_output_paths(
        [
            metrics_path,
            validation_metrics_path,
            training_state_path,
            checkpoint_path,
            best_checkpoint_path,
            training_summary_path,
        ]
    )

    if existing_output_paths and not args.resume:
        confirm_overwrite_existing_training_output(
            existing_output_paths
        )
        remove_existing_training_output(
            existing_output_paths
        )

    validation_envs = {}

    if args.validation_dataset is not None:
        if args.algorithm not in NEURAL_ALGORITHMS:
            raise ValueError(
                "Periodic validation checkpoints are only "
                "supported for DNN agents."
            )

        if args.validation_interval <= 0:
            raise ValueError(
                "--validation-interval must be positive when "
                "--validation-dataset is provided."
            )

        if (
            args.early_stopping_patience is not None
            and args.early_stopping_patience < 0
        ):
            raise ValueError(
                "--early-stopping-patience cannot be negative."
            )

        if args.early_stopping_min_delta < 0.0:
            raise ValueError(
                "--early-stopping-min-delta cannot be negative."
            )

        validation_env = MazeEnv(
            dataset_path=args.validation_dataset,
            max_steps=args.max_steps,
        )
        validation_envs[
            "validation"
        ] = validation_env

        if args.same_layout_dataset is not None:
            validation_envs[
                "same_layout"
            ] = MazeEnv(
                dataset_path=args.same_layout_dataset,
                max_steps=args.max_steps,
            )

    elif args.same_layout_dataset is not None:
        raise ValueError(
            "--same-layout-dataset requires --validation-dataset."
        )

    task_indices = (
        [args.fixed_index]
        if args.fixed_index is not None
        else list(range(env.num_tasks))
    )

    reached_task_indices = set()
    cumulative_transitions = 0
    training_round = 0
    optimization_epoch = 0
    q_snapshots = []
    model_snapshots = []
    best_validation = None
    latest_validation_by_split = {}
    validation_checks_without_improvement = 0
    validation_has_positive_score = False
    stopped_early = False
    stopped_epoch = None
    q_snapshot_count = max(
        0,
        args.q_snapshot_count,
    )
    snapshot_epochs = [
        int(epoch)
        for epoch in np.unique(
            np.linspace(
                1,
                task_selection_pass_budget,
                num=min(
                    q_snapshot_count,
                    task_selection_pass_budget,
                ),
                dtype=np.int64,
            )
        )
    ]
    snapshot_indices = {
        epoch: index
        for index, epoch in enumerate(
            snapshot_epochs,
            start=1,
        )
    }

    def record_episode_metrics(metrics):
        nonlocal cumulative_transitions

        append_metrics_csv(
            metrics_path,
            metrics,
        )

        cumulative_transitions += int(
            metrics.steps
        )

        if metrics.success:
            reached_task_indices.add(
                int(metrics.task_index)
            )

        log_tensorboard_episode(
            tensorboard_writer,
            metrics,
            agent,
            cumulative_transitions,
        )

    task_progress_line_active = False

    def completed_task_count_for_epoch(
        epoch_metrics,
    ) -> int:
        return len(
            {
                int(metrics.task_index)
                for metrics in epoch_metrics
            }
        )

    def write_task_progress_line(
        dataset_epoch: int,
        epoch_metrics_by_epoch,
    ) -> None:
        nonlocal task_progress_line_active

        if args.no_progress or not sys.stdout.isatty():
            return

        completed_tasks = completed_task_count_for_epoch(
            epoch_metrics_by_epoch.get(
                dataset_epoch,
                [],
            )
        )
        message = format_task_selection_pass_task_progress_message(
            dataset_epoch,
            task_selection_pass_budget,
            completed_tasks,
            sampler.total_task_count,
        )
        print(
            f"\r{message}\033[K",
            end="",
            flush=True,
        )
        task_progress_line_active = True

    def print_progress_summary(
        message: str,
    ) -> None:
        nonlocal task_progress_line_active

        if (
            task_progress_line_active
            and sys.stdout.isatty()
        ):
            lines = message.splitlines()
            lines[0] = f"\r{lines[0]}\033[K"
            print(
                "\n".join(lines),
                flush=True,
            )
        else:
            print(message)

        task_progress_line_active = False

    sampler = HierarchicalTaskSampler(
        task_indices=task_indices,
        layout_indices=env.layout_indices,
        seed=args.seed,
    )
    rollout = 0
    restored_epoch_metrics_by_epoch = {}
    restored_next_epoch_to_log = 1
    expected_training_state_metadata = (
        training_state_metadata(
            args=args,
            agent_hyperparameters=agent_hyperparameters,
            task_selection_pass_budget=task_selection_pass_budget,
            task_batch_size=task_batch_size,
            rollout_group_size=rollout_group_size,
        )
    )

    if args.resume:
        if not training_state_path.exists():
            raise RuntimeError(
                "--resume requested, but "
                f"{training_state_path} does not exist."
            )

        training_state = load_training_state(
            training_state_path,
            device,
        )
        validate_training_state_metadata(
            training_state,
            expected_training_state_metadata,
            allowed_mismatches=(
                resume_training_state_allowed_mismatches(
                    args
                )
                | (
                    {
                        "early_stopping_patience",
                        "early_stopping_min_delta",
                    }
                    if args.retroactive_early_stop
                    else set()
                )
            ),
        )
        load_agent_training_state(
            args.algorithm,
            agent,
            training_state["agent"],
        )
        sampler.load_state_dict(
            training_state["sampler"]
        )
        load_runtime_rng_state(
            training_state["rng"]
        )

        progress_state = training_state[
            "progress"
        ]
        reached_task_indices = set(
            progress_state[
                "reached_task_indices"
            ]
        )
        cumulative_transitions = int(
            progress_state[
                "cumulative_transitions"
            ]
        )
        training_round = int(
            progress_state["training_round"]
        )
        optimization_epoch = int(
            progress_state[
                "optimization_epoch"
            ]
        )
        rollout = int(progress_state["rollout"])
        q_snapshots = list(
            progress_state.get(
                "q_snapshots",
                [],
            )
        )
        model_snapshots = list(
            progress_state.get(
                "model_snapshots",
                [],
            )
        )
        best_validation = progress_state.get(
            "best_validation"
        )
        latest_validation_by_split = dict(
            progress_state.get(
                "latest_validation_by_split",
                {},
            )
        )
        validation_checks_without_improvement = int(
            progress_state[
                "validation_checks_without_improvement"
            ]
        )
        validation_has_positive_score = bool(
            progress_state[
                "validation_has_positive_score"
            ]
        )
        stopped_early = bool(
            progress_state["stopped_early"]
        )
        stopped_epoch = progress_state[
            "stopped_epoch"
        ]
        restored_epoch_metrics_by_epoch = {
            int(epoch): list(epoch_metrics)
            for epoch, epoch_metrics in progress_state[
                "epoch_metrics_by_epoch"
            ].items()
        }
        restored_next_epoch_to_log = int(
            progress_state[
                "next_epoch_to_log"
            ]
        )

        file_offsets = training_state.get(
            "file_offsets",
            {},
        )
        truncate_file(
            metrics_path,
            file_offsets.get(
                "metrics_csv_size"
            ),
        )
        truncate_file(
            validation_metrics_path,
            file_offsets.get(
                "validation_metrics_csv_size"
            ),
        )

        if args.retroactive_early_stop:
            replayed_early_stop = (
                replay_early_stopping_from_validation_metrics(
                    validation_metrics_path,
                    patience=args.early_stopping_patience,
                    min_delta=args.early_stopping_min_delta,
                )
            )
            best_validation = replayed_early_stop[
                "best_validation"
            ]
            latest_validation_by_split = (
                replayed_early_stop[
                    "latest_validation_by_split"
                ]
            )
            validation_checks_without_improvement = int(
                replayed_early_stop[
                    "validation_checks_without_improvement"
                ]
            )
            validation_has_positive_score = bool(
                replayed_early_stop[
                    "validation_has_positive_score"
                ]
            )
            stopped_early = bool(
                replayed_early_stop[
                    "stopped_early"
                ]
            )
            stopped_epoch = replayed_early_stop[
                "stopped_epoch"
            ]

        print(
            "Loaded full training state from "
            f"{training_state_path}"
        )

        if args.retroactive_early_stop:
            if stopped_early:
                print(
                    "Retroactive early stopping condition was "
                    f"already met at dataset epoch {stopped_epoch}."
                )
            else:
                print(
                    "Replayed validation history for retroactive "
                    "early stopping; continuing training."
                )

    if validation_envs and should_record_train_greedy_validation(
        resuming=(
            args.resume
        ),
        validation_metrics_path=validation_metrics_path,
    ):
        validation_envs[
            "train_greedy"
        ] = MazeEnv(
            dataset_path=args.dataset,
            max_steps=args.max_steps,
        )

    def build_rollout_groups(
        rollout_specs,
    ):
        nonlocal rollout

        groups = []

        for spec in rollout_specs:
            group = []

            for _ in range(
                rollout_group_size
            ):
                rollout += 1
                group.append(
                    (
                        rollout,
                        spec.dataset_epoch,
                        spec.task_index,
                    )
                )

            groups.append(group)

        return groups

    def flatten_rollout_groups(
        rollout_groups,
    ):
        return [
            episode_spec
            for group in rollout_groups
            for episode_spec in group
        ]

    def save_training_progress_state(
        epoch_metrics_by_epoch,
        next_epoch_to_log: int,
    ) -> None:
        save_training_state(
            training_state_path,
            {
                "version": 1,
                "metadata": expected_training_state_metadata,
                "agent": agent_training_state_dict(
                    args.algorithm,
                    agent,
                ),
                "sampler": sampler.state_dict(),
                "rng": runtime_rng_state_dict(),
                "progress": {
                    "reached_task_indices": sorted(
                        reached_task_indices
                    ),
                    "cumulative_transitions": (
                        cumulative_transitions
                    ),
                    "training_round": training_round,
                    "optimization_epoch": (
                        optimization_epoch
                    ),
                    "rollout": rollout,
                    "q_snapshots": q_snapshots,
                    "model_snapshots": model_snapshots,
                    "best_validation": best_validation,
                    "latest_validation_by_split": (
                        latest_validation_by_split
                    ),
                    "validation_checks_without_improvement": (
                        validation_checks_without_improvement
                    ),
                    "validation_has_positive_score": (
                        validation_has_positive_score
                    ),
                    "stopped_early": stopped_early,
                    "stopped_epoch": stopped_epoch,
                    "epoch_metrics_by_epoch": (
                        epoch_metrics_by_epoch
                    ),
                    "next_epoch_to_log": (
                        next_epoch_to_log
                    ),
                },
                "file_offsets": {
                    "metrics_csv_size": file_size(
                        metrics_path
                    ),
                    "validation_metrics_csv_size": (
                        file_size(
                            validation_metrics_path
                        )
                    ),
                },
            },
        )

    def finish_dataset_epoch(
        dataset_epoch,
        epoch_metrics,
    ):
        nonlocal best_validation
        nonlocal latest_validation_by_split
        nonlocal stopped_early
        nonlocal stopped_epoch
        nonlocal validation_has_positive_score
        nonlocal validation_checks_without_improvement

        if not args.no_progress and epoch_metrics:
            print_progress_summary(
                format_training_epoch_progress_message(
                    dataset_epoch,
                    task_selection_pass_budget,
                    epoch_metrics,
                    total_tasks=sampler.total_task_count,
                    early_stopping_counter=(
                        validation_checks_without_improvement
                    ),
                    early_stopping_patience=(
                        args.early_stopping_patience
                    ),
                )
            )

        log_tensorboard_epoch(
            tensorboard_writer,
            dataset_epoch,
            epoch_metrics,
            agent,
        )

        if (
            args.algorithm in TABULAR_ALGORITHMS
            and dataset_epoch in snapshot_indices
        ):
            q_snapshots.append(
                {
                    "snapshot_index": snapshot_indices[
                        dataset_epoch
                    ],
                    "epoch": dataset_epoch,
                    "q": agent.q_state_dict(),
                    "reached_task_indices": sorted(
                        reached_task_indices
                    ),
                }
            )

        if (
            args.algorithm
            in NEURAL_SNAPSHOT_ALGORITHMS
            and dataset_epoch in snapshot_indices
        ):
            model_snapshots.append(
                {
                    "snapshot_index": snapshot_indices[
                        dataset_epoch
                    ],
                    "epoch": dataset_epoch,
                    "model_state_dict": model_snapshot_state_dict(
                        args.algorithm,
                        agent,
                    ),
                    "reached_task_indices": sorted(
                        reached_task_indices
                    ),
                }
            )

        if validation_envs and should_validate_epoch(
            dataset_epoch,
            task_selection_pass_budget,
            args.validation_interval,
        ):
            ordered_validation_splits = [
                split
                for split in (
                    "train_greedy",
                    "validation",
                    "same_layout",
                )
                if split in validation_envs
            ] + [
                split
                for split in validation_envs
                if split
                not in {
                    "train_greedy",
                    "validation",
                    "same_layout",
                }
            ]

            for split in ordered_validation_splits:
                validation_env = validation_envs[
                    split
                ]
                validation_task_indices = (
                    (
                        list(task_indices)
                        if split == "train_greedy"
                        else list(
                            range(
                                validation_env.num_tasks
                            )
                        )
                    )
                    if args.validation_all_tasks
                    else None
                )
                _, validation_summary = evaluate(
                    algorithm=args.algorithm,
                    agent=agent,
                    env=validation_env,
                    episodes=args.validation_episodes,
                    seed=args.seed,
                    task_indices=validation_task_indices,
                )
                validation_score = validation_summary[
                    "average_path_efficiency"
                ]
                validation_row = {
                    "split": split,
                    "dataset_epoch": dataset_epoch,
                    "episodes": validation_summary[
                        "episodes"
                    ],
                    "success_rate": validation_summary[
                        "success_rate"
                    ],
                    "average_episode_return": validation_summary[
                        "average_episode_return"
                    ],
                    "mean_path_efficiency": validation_score,
                    "average_successful_path_efficiency": (
                        validation_summary[
                            "average_successful_path_efficiency"
                        ]
                    ),
                }

                append_validation_csv(
                    validation_metrics_path,
                    validation_row,
                )
                latest_validation_by_split[
                    split
                ] = validation_row

                if split != "validation":
                    print(
                        format_validation_progress_message(
                            split,
                            dataset_epoch,
                            task_selection_pass_budget,
                            validation_score,
                            success_rate=validation_row[
                                "success_rate"
                            ],
                        )
                    )
                    continue

                if validation_score > 0.0:
                    validation_has_positive_score = True

                improved = (
                    best_validation is None
                    or validation_score
                    > best_validation[
                        "mean_path_efficiency"
                    ]
                    + args.early_stopping_min_delta
                )

                if improved:
                    validation_checks_without_improvement = 0
                    best_validation = validation_row
                    save_neural_checkpoint_state(
                        best_checkpoint_path,
                        args.algorithm,
                        current_neural_state_dict(
                            args.algorithm,
                            agent,
                        ),
                        model_snapshots=model_snapshots,
                        hyperparameters=agent_hyperparameters,
                        validation_metadata=validation_row,
                    )
                else:
                    validation_checks_without_improvement += 1

                print(
                    format_validation_progress_message(
                        "validation",
                        dataset_epoch,
                        task_selection_pass_budget,
                        validation_score,
                        success_rate=validation_row[
                            "success_rate"
                        ],
                        best_score=best_validation[
                            "mean_path_efficiency"
                        ],
                    )
                )

            if (
                should_stop_early(
                    args.early_stopping_patience,
                    validation_checks_without_improvement,
                    validation_has_positive_score,
                )
            ):
                stopped_early = True
                stopped_epoch = dataset_epoch

    def record_training_round(round_metrics):
        nonlocal training_round
        nonlocal optimization_epoch

        optimization_epochs = (
            optimization_epochs_for_round(
                args.algorithm,
                agent,
                round_metrics,
            )
        )

        if optimization_epochs <= 0:
            return

        training_round += 1

        log_tensorboard_training_round(
            tensorboard_writer,
            training_round,
            round_metrics,
            agent,
        )

        for _ in range(
            optimization_epochs
        ):
            optimization_epoch += 1
            log_tensorboard_optimization_epoch(
                tensorboard_writer,
                optimization_epoch,
                round_metrics,
                agent,
            )

    def maybe_finish_dataset_epochs(
        epoch_metrics_by_epoch,
        next_epoch_to_log,
    ) -> int:
        while (
            next_epoch_to_log
            < sampler.current_dataset_epoch
            and next_epoch_to_log
            in epoch_metrics_by_epoch
            and not stopped_early
        ):
            finish_dataset_epoch(
                next_epoch_to_log,
                epoch_metrics_by_epoch.pop(
                    next_epoch_to_log
                ),
            )
            next_epoch_to_log += 1

        return next_epoch_to_log

    if args.algorithm in {
        "reinforce",
        "a2c",
        "ppo",
    }:
        epoch_metrics_by_epoch = dict(
            restored_epoch_metrics_by_epoch
        )
        next_epoch_to_log = restored_next_epoch_to_log

        while (
            sampler.completed_dataset_epochs
            < task_selection_pass_budget
            and not stopped_early
        ):
            rollout_specs = sampler.sample_collection(
                collection_size=task_batch_size,
                target_dataset_epochs=task_selection_pass_budget,
            )
            episode_specs = flatten_rollout_groups(
                build_rollout_groups(
                    rollout_specs
                )
            )

            if not episode_specs:
                break

            if args.algorithm == "reinforce":
                round_metrics = (
                    train_reinforce_round(
                        env,
                        agent,
                        episode_specs,
                    )
                )
            elif args.algorithm == "a2c":
                round_metrics = train_a2c_round(
                    env,
                    agent,
                    episode_specs,
                )
            else:
                round_metrics = train_ppo_round(
                    env,
                    agent,
                    episode_specs,
                )

            for metrics in round_metrics:
                epoch_metrics_by_epoch.setdefault(
                    metrics.epoch,
                    [],
                ).append(metrics)

                record_episode_metrics(metrics)
                write_task_progress_line(
                    metrics.epoch,
                    epoch_metrics_by_epoch,
                )

            record_training_round(
                round_metrics
            )

            next_epoch_to_log = (
                maybe_finish_dataset_epochs(
                    epoch_metrics_by_epoch,
                    next_epoch_to_log,
                )
            )
            save_training_progress_state(
                epoch_metrics_by_epoch,
                next_epoch_to_log,
            )

    elif args.algorithm == "grpo":
        epoch_metrics_by_epoch = dict(
            restored_epoch_metrics_by_epoch
        )
        next_epoch_to_log = restored_next_epoch_to_log

        print(
            "grpo task_selection_passes="
            f"{task_selection_pass_budget} from rollouts_per_task="
            f"{args.rollouts_per_task}, task_batch_size="
            f"{task_batch_size}, rollout_group_size="
            f"{rollout_group_size}",
            flush=True,
        )

        while (
            sampler.completed_dataset_epochs
            < task_selection_pass_budget
            and not stopped_early
        ):
            rollout_specs = sampler.sample_collection(
                collection_size=task_batch_size,
                target_dataset_epochs=task_selection_pass_budget,
            )

            if not rollout_specs:
                break

            rollout_groups = build_rollout_groups(
                rollout_specs
            )

            round_metrics = train_grpo_round(
                env,
                agent,
                rollout_groups,
            )

            for metrics in round_metrics:
                epoch_metrics_by_epoch.setdefault(
                    metrics.epoch,
                    [],
                ).append(metrics)

                record_episode_metrics(metrics)
                write_task_progress_line(
                    metrics.epoch,
                    epoch_metrics_by_epoch,
                )

            record_training_round(
                round_metrics
            )

            next_epoch_to_log = (
                maybe_finish_dataset_epochs(
                    epoch_metrics_by_epoch,
                    next_epoch_to_log,
                )
            )
            save_training_progress_state(
                epoch_metrics_by_epoch,
                next_epoch_to_log,
            )

    else:
        epoch_metrics_by_epoch = dict(
            restored_epoch_metrics_by_epoch
        )
        next_epoch_to_log = restored_next_epoch_to_log

        while (
            sampler.completed_dataset_epochs
            < task_selection_pass_budget
            and not stopped_early
        ):
            rollout_specs = sampler.sample_collection(
                collection_size=task_batch_size,
                target_dataset_epochs=task_selection_pass_budget,
            )

            if not rollout_specs:
                break

            round_metrics = []

            for (
                rollout_id,
                dataset_epoch,
                task_index,
            ) in flatten_rollout_groups(
                build_rollout_groups(
                    rollout_specs
                )
            ):
                if args.algorithm == "mc":
                    metrics = (
                        train_monte_carlo_episode(
                            env,
                            agent,
                            rollout_id,
                            dataset_epoch,
                            task_index,
                        )
                    )

                elif args.algorithm == "sarsa":
                    metrics = train_sarsa_episode(
                        env,
                        agent,
                        rollout_id,
                        dataset_epoch,
                        task_index,
                    )

                elif args.algorithm in {
                    "q_learning",
                    "dyna_q",
                }:
                    metrics = (
                        train_q_learning_episode(
                            env,
                            agent,
                            rollout_id,
                            dataset_epoch,
                            task_index,
                        )
                    )

                else:
                    metrics = train_dqn_episode(
                        env,
                        agent,
                        rollout_id,
                        dataset_epoch,
                        task_index,
                    )

                round_metrics.append(metrics)
                record_episode_metrics(metrics)
                epoch_metrics_by_epoch.setdefault(
                    metrics.epoch,
                    [],
                ).append(metrics)
                write_task_progress_line(
                    metrics.epoch,
                    epoch_metrics_by_epoch,
                )

            record_training_round(
                round_metrics
            )

            next_epoch_to_log = (
                maybe_finish_dataset_epochs(
                    epoch_metrics_by_epoch,
                    next_epoch_to_log,
                )
            )
            save_training_progress_state(
                epoch_metrics_by_epoch,
                next_epoch_to_log,
            )

    save_training_progress_state(
        epoch_metrics_by_epoch,
        next_epoch_to_log,
    )

    save_checkpoint(
        checkpoint_path,
        args.algorithm,
        agent,
        q_snapshots=q_snapshots,
        model_snapshots=model_snapshots,
        hyperparameters=agent_hyperparameters,
    )

    summary = {
        "algorithm": args.algorithm,
        "rollouts_per_task_requested": args.rollouts_per_task,
        "task_batch_size": task_batch_size,
        "rollout_group_size": rollout_group_size,
        "task_selection_passes_requested": task_selection_pass_budget,
        "task_selection_passes_completed": (
            stopped_epoch
            if stopped_early
            else sampler.completed_dataset_epochs
        ),
        "dataset_epochs_requested": task_selection_pass_budget,
        "dataset_epochs_completed": (
            stopped_epoch
            if stopped_early
            else sampler.completed_dataset_epochs
        ),
        "optimization_epochs_requested": task_selection_pass_budget,
        "group_size": rollout_group_size,
        "stopped_early": stopped_early,
        "stopped_epoch": stopped_epoch,
        "checkpoint_path": str(
            checkpoint_path
        ),
        "training_state_path": str(
            training_state_path
        ),
        "hyperparameters": agent_hyperparameters,
    }

    if best_config_overrides is not None:
        summary["best_config_path"] = str(
            best_config_overrides.path
        )

    if best_validation is not None:
        summary["best_validation"] = (
            best_validation
        )
        summary["best_checkpoint_path"] = str(
            best_checkpoint_path
        )

    if latest_validation_by_split:
        summary["latest_validation_by_split"] = (
            latest_validation_by_split
        )

    write_training_summary(
        training_summary_path,
        summary,
    )

    print(
        f"Saved metrics to "
        f"{metrics_path}"
    )

    print(
        f"Saved checkpoint to "
        f"{checkpoint_path}"
    )

    print(
        "Saved full training state to "
        f"{training_state_path}"
    )

    if best_validation is not None:
        print(
            "Saved best validation checkpoint to "
            f"{best_checkpoint_path}"
        )
        print(
            "Saved validation metrics to "
            f"{validation_metrics_path}"
        )

    print(
        "Saved training summary to "
        f"{training_summary_path}"
    )

    if not args.no_plot:
        plot_path = (
            args.plot_output
            if args.plot_output is not None
            else run_dir / "training_metrics.png"
        )

        plot_training_metrics(
            metrics_path=metrics_path,
            output_path=plot_path,
            rolling_window=args.rolling_window,
        )

        print(
            f"Saved training plot to "
            f"{plot_path}"
        )

        task_plot_path = (
            args.task_plot_output
            if args.task_plot_output is not None
            else run_dir / "task_training_metrics.png"
        )

        plot_task_training_metrics(
            metrics_path=metrics_path,
            output_path=task_plot_path,
        )

        print(
            f"Saved task training plot to "
            f"{task_plot_path}"
        )

        task_html_path = (
            args.task_html_output
            if args.task_html_output is not None
            else run_dir / "task_training_metrics.html"
        )

        write_task_training_metrics_html(
            metrics_path=metrics_path,
            output_path=task_html_path,
        )

        print(
            f"Saved interactive task training plot to "
            f"{task_html_path}"
        )

    if validation_metrics_path.exists():
        validation_plot_path = (
            run_dir / "validation_metrics.png"
        )
        plot_validation_metrics(
            metrics_path=validation_metrics_path,
            output_path=validation_plot_path,
        )
        print(
            f"Saved validation plot to "
            f"{validation_plot_path}"
        )

    if tensorboard_writer is not None:
        tensorboard_writer.close()


if __name__ == "__main__":
    main()
