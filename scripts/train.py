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

POLICY_ROLLOUT_EPISODES = {
    "reinforce": 16,
    "a2c": 16,
    "ppo": 64,
}

GROUPED_POLICY_ALGORITHMS = {
    "grpo",
}


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
    return algorithm not in TABULAR_ALGORITHMS


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
        default="data/train.npz",
    )

    parser.add_argument(
        "--dataset-epochs",
        type=int,
        default=100,
        help=(
            "Number of dataset epochs to train. A dataset epoch "
            "completes when every selected task has appeared at "
            "least once in rollout collection."
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
        default=Path("runs"),
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Append to an existing metrics file instead of starting fresh.",
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
            "Number of dataset epochs used for rolling means. "
            "Use 0 to disable rolling means."
        ),
    )

    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip training plot generation.",
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
            "save in checkpoints. The first snapshot is dataset "
            "epoch 1."
        ),
    )

    parser.add_argument(
        "--tensorboard",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Write TensorBoard event logs. Defaults on for neural "
            "agents and off for tabular agents."
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
        "--rollout-episodes",
        type=int,
        default=None,
        help=(
            "Rollouts collected per policy-gradient training "
            "round. Defaults to the algorithm-specific value."
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
        default=0,
        help=(
            "Dataset-epoch interval for validation. Use 0 to "
            "disable periodic validation."
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
        default=None,
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
        "--group-size",
        type=int,
        default=None,
        help=(
            "GRPO rollouts per grouped update. GRPO runs "
            "ceil(dataset_epochs / group_size) optimization "
            "epochs."
        ),
    )

    return parser.parse_args()


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


def rollout_episodes_for_algorithm(
    algorithm: str,
    rollout_episodes: int | None,
) -> int:
    if rollout_episodes is None:
        return POLICY_ROLLOUT_EPISODES[
            algorithm
        ]

    if algorithm not in POLICY_ROLLOUT_EPISODES:
        raise ValueError(
            "--rollout-episodes is only supported for "
            "policy-gradient agents."
        )

    if rollout_episodes <= 0:
        raise ValueError(
            "--rollout-episodes must be positive."
        )

    return rollout_episodes


def group_size_for_algorithm(
    algorithm: str,
    group_size: int | None,
) -> int:
    if group_size is None:
        if algorithm in GROUPED_POLICY_ALGORITHMS:
            return 8

        return 0

    if algorithm not in GROUPED_POLICY_ALGORITHMS:
        raise ValueError(
            "--group-size is only supported for GRPO."
        )

    if group_size < 2:
        raise ValueError(
            "--group-size must be at least 2."
        )

    return group_size


def training_epoch_budget_for_algorithm(
    algorithm: str,
    dataset_epochs: int,
    group_size: int,
) -> int:
    if dataset_epochs <= 0:
        raise ValueError(
            "--dataset-epochs must be positive."
        )

    if algorithm in GROUPED_POLICY_ALGORITHMS:
        return max(
            1,
            (
                dataset_epochs
                + group_size
                - 1
            )
            // group_size,
        )

    return dataset_epochs


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
        f"dataset_epoch={metrics.epoch:5d} "
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

    agent_hyperparameters = (
        neural_hyperparameters_from_args(
            args
        )
    )
    group_size = group_size_for_algorithm(
        args.algorithm,
        args.group_size,
    )
    training_epoch_budget = (
        training_epoch_budget_for_algorithm(
            args.algorithm,
            args.dataset_epochs,
            group_size,
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
    best_checkpoint_path = (
        run_dir / "best_checkpoint.pt"
    )
    training_summary_path = (
        run_dir / "training_summary.json"
    )

    if metrics_path.exists() and not args.resume:
        metrics_path.unlink()

    if (
        validation_metrics_path.exists()
        and not args.resume
    ):
        validation_metrics_path.unlink()

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
                training_epoch_budget,
                num=min(
                    q_snapshot_count,
                    training_epoch_budget,
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

    sampler = HierarchicalTaskSampler(
        task_indices=task_indices,
        layout_indices=env.layout_indices,
        seed=args.seed,
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

        if dataset_epoch % 100 == 0:
            for metrics in sorted(
                epoch_metrics,
                key=lambda item: item.task_index,
            ):
                print(
                    format_progress_message(
                        args.algorithm,
                        metrics,
                        agent,
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
            training_epoch_budget,
            args.validation_interval,
        ):
            for split, validation_env in validation_envs.items():
                validation_task_indices = (
                    list(
                        range(
                            validation_env.num_tasks
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
                        f"{split} dataset_epoch="
                        f"{dataset_epoch:5d}/{training_epoch_budget} "
                        f"({dataset_epoch / training_epoch_budget:.0%}) "
                        f"mean_path_efficiency={validation_score:.3f}"
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
                    f"validation dataset_epoch="
                    f"{dataset_epoch:5d}/{training_epoch_budget} "
                    f"({dataset_epoch / training_epoch_budget:.0%}) "
                    f"mean_path_efficiency={validation_score:.3f} "
                    f"best="
                    f"{best_validation['mean_path_efficiency']:.3f}"
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

    if args.algorithm in POLICY_ROLLOUT_EPISODES:
        rollout_episodes = rollout_episodes_for_algorithm(
            args.algorithm,
            args.rollout_episodes,
        )
        epoch_metrics_by_epoch = {}
        next_epoch_to_log = 1

        while (
            sampler.completed_dataset_epochs
            < training_epoch_budget
            and not stopped_early
        ):
            rollout_specs = sampler.sample_collection(
                collection_size=rollout_episodes,
                target_dataset_epochs=training_epoch_budget,
            )
            episode_specs = [
                (
                    spec.rollout,
                    spec.dataset_epoch,
                    spec.task_index,
                )
                for spec in rollout_specs
            ]

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

            record_training_round(
                round_metrics
            )

            next_epoch_to_log = (
                maybe_finish_dataset_epochs(
                    epoch_metrics_by_epoch,
                    next_epoch_to_log,
                )
            )

    elif args.algorithm in GROUPED_POLICY_ALGORITHMS:
        if args.rollout_episodes is not None:
            rollout_episodes_for_algorithm(
                args.algorithm,
                args.rollout_episodes,
            )

        epoch_metrics_by_epoch = {}
        next_epoch_to_log = 1
        grpo_rollout = 0

        print(
            "grpo optimization_epochs="
            f"{training_epoch_budget} from dataset_epochs="
            f"{args.dataset_epochs} and group_size={group_size}",
            flush=True,
        )

        while (
            sampler.completed_dataset_epochs
            < training_epoch_budget
            and not stopped_early
        ):
            rollout_specs = sampler.sample_collection(
                collection_size=1,
                target_dataset_epochs=training_epoch_budget,
            )

            if not rollout_specs:
                break

            spec = rollout_specs[0]
            episode_specs = []

            for _ in range(group_size):
                grpo_rollout += 1
                episode_specs.append(
                    (
                        grpo_rollout,
                        spec.dataset_epoch,
                        spec.task_index,
                    )
                )

            round_metrics = train_grpo_round(
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

            record_training_round(
                round_metrics
            )

            next_epoch_to_log = (
                maybe_finish_dataset_epochs(
                    epoch_metrics_by_epoch,
                    next_epoch_to_log,
                )
            )

    else:
        if args.rollout_episodes is not None:
            rollout_episodes_for_algorithm(
                args.algorithm,
                args.rollout_episodes,
            )

        epoch_metrics_by_epoch = {}
        next_epoch_to_log = 1

        while (
            sampler.completed_dataset_epochs
            < training_epoch_budget
            and not stopped_early
        ):
            rollout_specs = sampler.sample_collection(
                collection_size=1,
                target_dataset_epochs=training_epoch_budget,
            )

            if not rollout_specs:
                break

            spec = rollout_specs[0]

            if args.algorithm == "mc":
                metrics = (
                    train_monte_carlo_episode(
                        env,
                        agent,
                        spec.rollout,
                        spec.dataset_epoch,
                        spec.task_index,
                    )
                )

            elif args.algorithm == "sarsa":
                metrics = train_sarsa_episode(
                    env,
                    agent,
                    spec.rollout,
                    spec.dataset_epoch,
                    spec.task_index,
                )

            elif args.algorithm in {
                "q_learning",
                "dyna_q",
            }:
                metrics = (
                    train_q_learning_episode(
                        env,
                        agent,
                        spec.rollout,
                        spec.dataset_epoch,
                        spec.task_index,
                    )
                )

            else:
                metrics = train_dqn_episode(
                    env,
                    agent,
                    spec.rollout,
                    spec.dataset_epoch,
                    spec.task_index,
                )

            record_episode_metrics(metrics)
            record_training_round(
                [metrics]
            )
            epoch_metrics_by_epoch.setdefault(
                metrics.epoch,
                [],
            ).append(metrics)

            next_epoch_to_log = (
                maybe_finish_dataset_epochs(
                    epoch_metrics_by_epoch,
                    next_epoch_to_log,
                )
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
        "dataset_epochs_requested": args.dataset_epochs,
        "dataset_epochs_completed": (
            stopped_epoch
            if stopped_early
            else sampler.completed_dataset_epochs
        ),
        "optimization_epochs_requested": training_epoch_budget,
        "group_size": (
            group_size
            if args.algorithm
            in GROUPED_POLICY_ALGORITHMS
            else None
        ),
        "stopped_early": stopped_early,
        "stopped_epoch": stopped_epoch,
        "checkpoint_path": str(
            checkpoint_path
        ),
        "hyperparameters": agent_hyperparameters,
    }

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
