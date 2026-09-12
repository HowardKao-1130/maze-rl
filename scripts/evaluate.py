from __future__ import annotations

import argparse
import csv
from pathlib import Path
import pickle
import random

import numpy as np
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
from maze_rl.evaluation.q_plots import plot_q_snapshots
from maze_rl.evaluation.q_plots import plot_neural_snapshots
from maze_rl.training.plots import (
    plot_task_training_metrics,
    plot_training_metrics,
    write_task_training_metrics_html,
)


TABULAR_ALGORITHMS = {
    "mc",
    "sarsa",
    "q_learning",
    "dyna_q",
}


def default_checkpoint_path(
    algorithm: str,
) -> Path:
    suffix = (
        ".pkl"
        if algorithm in TABULAR_ALGORITHMS
        else ".pt"
    )

    return (
        Path("runs")
        / algorithm
        / f"checkpoint{suffix}"
    )


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--algorithm",
        required=True,
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help=(
            "Checkpoint path. Defaults to "
            "runs/<algorithm>/checkpoint.pkl for tabular agents "
            "and runs/<algorithm>/checkpoint.pt for neural agents."
        ),
    )

    parser.add_argument(
        "--dataset",
        default="data/test.npz",
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=200,
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=200,
    )

    parser.add_argument(
        "--fixed-index",
        type=int,
        default=None,
        help="Evaluate one fixed task index.",
    )

    parser.add_argument(
        "--all-tasks",
        action="store_true",
        help=(
            "Evaluate every task in the dataset once instead of "
            "sampling --episodes random tasks."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help=(
            "Seed used for evaluation task sampling and stochastic "
            "tie-breaking."
        ),
    )

    parser.add_argument(
        "--training-metrics",
        type=Path,
        default=None,
        help=(
            "Training metrics CSV to plot. Defaults to "
            "<checkpoint directory>/metrics.csv."
        ),
    )

    parser.add_argument(
        "--plot-output",
        type=Path,
        default=None,
        help=(
            "Training plot path. Defaults to "
            "<checkpoint directory>/training_metrics.png."
        ),
    )

    parser.add_argument(
        "--rolling-window",
        type=int,
        default=0,
        help=(
            "Number of epochs used for rolling means. "
            "Use 0 to disable rolling means."
        ),
    )

    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip automatic training-metrics plot generation.",
    )

    parser.add_argument(
        "--q-plot-output-dir",
        type=Path,
        default=None,
        help=(
            "Directory for task Q-value plots. Defaults to "
            "<checkpoint directory>/q_value_plots."
        ),
    )

    parser.add_argument(
        "--q-plot-all-tasks",
        action="store_true",
        help=(
            "Plot Q-value snapshots for every task in the "
            "evaluation dataset instead of only evaluated tasks."
        ),
    )

    parser.add_argument(
        "--no-q-plots",
        action="store_true",
        help="Skip tabular Q-value snapshot plots.",
    )

    parser.add_argument(
        "--no-q-videos",
        action="store_true",
        help="Skip MP4 videos for tabular Q-value snapshots.",
    )

    parser.add_argument(
        "--q-video-fps",
        type=float,
        default=8.0,
        help="Frames per second for Q-value snapshot MP4 videos.",
    )

    parser.add_argument(
        "--task-plot-output",
        type=Path,
        default=None,
        help=(
            "Per-task training plot path. Defaults to "
            "<checkpoint directory>/task_training_metrics.png."
        ),
    )

    parser.add_argument(
        "--task-html-output",
        type=Path,
        default=None,
        help=(
            "Interactive per-task training plot path. Defaults to "
            "<checkpoint directory>/task_training_metrics.html."
        ),
    )

    parser.add_argument(
        "--evaluation-output",
        type=Path,
        default=None,
        help=(
            "Evaluation CSV path. Defaults to "
            "runs/<algorithm>/evaluation.csv."
        ),
    )

    args = parser.parse_args()

    if args.checkpoint is None:
        args.checkpoint = default_checkpoint_path(
            args.algorithm
        )

    return args


def create_agent(
    algorithm,
    env,
    device,
    seed,
):
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
        return DQNAgent(
            **neural_kwargs,
            seed=seed,
        )

    if algorithm == "reinforce":
        return ReinforceAgent(
            **neural_kwargs
        )

    if algorithm == "a2c":
        return A2CAgent(
            **neural_kwargs
        )

    if algorithm == "ppo":
        return PPOAgent(
            **neural_kwargs
        )

    if algorithm == "grpo":
        return GRPOAgent(
            **neural_kwargs
        )

    raise ValueError(
        algorithm
    )


def load_checkpoint(
    algorithm,
    checkpoint_path,
    agent,
    device,
):
    if algorithm in TABULAR_ALGORITHMS:
        with checkpoint_path.open(
            "rb"
        ) as file:
            checkpoint = (
                pickle.load(file)
            )

        agent.load_q_state_dict(
            checkpoint["q"]
        )

        return checkpoint

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    state_dict = checkpoint[
        "model_state_dict"
    ]

    if algorithm == "dqn":
        agent.online_network.load_state_dict(
            state_dict
        )

    elif algorithm in {
        "reinforce",
        "grpo",
    }:
        agent.policy.load_state_dict(
            state_dict
        )

    else:
        agent.network.load_state_dict(
            state_dict
        )

    return checkpoint


def main():
    args = parse_args()

    if (
        args.all_tasks
        and args.fixed_index is not None
    ):
        raise SystemExit(
            "--all-tasks cannot be combined with --fixed-index."
        )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

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

    agent = create_agent(
        args.algorithm,
        env,
        device,
        args.seed,
    )

    checkpoint = load_checkpoint(
        args.algorithm,
        args.checkpoint,
        agent,
        device,
    )

    task_indices = (
        list(range(env.num_tasks))
        if args.all_tasks
        else None
    )

    results, summary = evaluate(
        algorithm=args.algorithm,
        agent=agent,
        env=env,
        episodes=args.episodes,
        seed=args.seed,
        task_indices=task_indices,
    )

    evaluation_path = (
        args.evaluation_output
        if args.evaluation_output is not None
        else (
            Path("runs")
            / args.algorithm
            / "evaluation.csv"
        )
    )

    evaluation_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with evaluation_path.open(
        "w",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=results[0].keys(),
        )

        writer.writeheader()
        writer.writerows(results)

    print(
        f"Success rate: "
        f"{summary['success_rate']:.3f}"
    )

    print(
        "Average episode return: "
        f"{summary['average_episode_return']:.3f}"
    )

    print(
        "Average path efficiency: "
        f"{summary['average_path_efficiency']:.3f}"
    )

    print(
        "Average successful path "
        f"efficiency: "
        f"{summary['average_successful_path_efficiency']:.3f}"
    )

    print(
        f"Saved evaluation to "
        f"{evaluation_path}"
    )

    if (
        args.algorithm in TABULAR_ALGORITHMS
        and not args.no_q_plots
    ):
        q_snapshots = checkpoint.get(
            "q_snapshots",
            [],
        )

        if q_snapshots:
            q_plot_dir = (
                args.q_plot_output_dir
                if args.q_plot_output_dir is not None
                else (
                    args.checkpoint.parent
                    / "q_value_plots"
                )
            )

            task_indices = (
                list(range(env.num_tasks))
                if args.q_plot_all_tasks
                else [
                    int(result["task_index"])
                    for result in results
                ]
            )

            q_plot_result = plot_q_snapshots(
                env=env,
                q_snapshots=q_snapshots,
                task_indices=task_indices,
                output_dir=q_plot_dir,
                write_videos=not args.no_q_videos,
                video_fps=args.q_video_fps,
            )

            print(
                f"Saved {q_plot_result.image_count} Q-value plots to "
                f"{q_plot_dir}"
            )

            if q_plot_result.video_count:
                print(
                    f"Saved {q_plot_result.video_count} Q-value videos to "
                    f"{q_plot_dir}"
                )

            elif q_plot_result.skipped_video_reason:
                print(
                    "Skipped Q-value videos: "
                    f"{q_plot_result.skipped_video_reason}"
                )

        else:
            print(
                "Skipped Q-value plots because the checkpoint "
                "does not contain q_snapshots. Retrain with the "
                "current training script to create epoch snapshots."
            )

    if (
        args.algorithm in {
            "dqn",
            "a2c",
            "ppo",
        }
        and not args.no_q_plots
    ):
        model_snapshots = checkpoint.get(
            "model_snapshots",
            [],
        )

        value_name = (
            "Q-value"
            if args.algorithm == "dqn"
            else "state-value"
        )

        if model_snapshots:
            neural_plot_dir = (
                args.q_plot_output_dir
                if args.q_plot_output_dir is not None
                else (
                    args.checkpoint.parent
                    / (
                        "q_value_plots"
                        if args.algorithm == "dqn"
                        else "v_value_plots"
                    )
                )
            )

            neural_task_indices = (
                list(range(env.num_tasks))
                if args.q_plot_all_tasks
                else [
                    int(result["task_index"])
                    for result in results
                ]
            )

            neural_plot_result = plot_neural_snapshots(
                env=env,
                agent=agent,
                algorithm=args.algorithm,
                model_snapshots=model_snapshots,
                task_indices=neural_task_indices,
                output_dir=neural_plot_dir,
                write_videos=not args.no_q_videos,
                video_fps=args.q_video_fps,
            )

            print(
                f"Saved {neural_plot_result.image_count} {value_name} plots to "
                f"{neural_plot_dir}"
            )

            if neural_plot_result.video_count:
                print(
                    f"Saved {neural_plot_result.video_count} {value_name} videos to "
                    f"{neural_plot_dir}"
                )

            elif neural_plot_result.skipped_video_reason:
                print(
                    f"Skipped {value_name} videos: "
                    f"{neural_plot_result.skipped_video_reason}"
                )

        else:
            print(
                f"Skipped {value_name} plots because the checkpoint "
                "does not contain model_snapshots. Retrain with the "
                "current training script to create epoch snapshots."
            )

    if args.no_plot:
        return

    metrics_path = (
        args.training_metrics
        if args.training_metrics is not None
        else args.checkpoint.parent / "metrics.csv"
    )

    plot_path = (
        args.plot_output
        if args.plot_output is not None
        else args.checkpoint.parent / "training_metrics.png"
    )

    if not metrics_path.exists():
        print(
            "Skipped training plot because metrics CSV "
            f"was not found: {metrics_path}"
        )
        return

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
        else (
            args.checkpoint.parent
            / "task_training_metrics.png"
        )
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
        else (
            args.checkpoint.parent
            / "task_training_metrics.html"
        )
    )

    write_task_training_metrics_html(
        metrics_path=metrics_path,
        output_path=task_html_path,
    )

    print(
        f"Saved interactive task training plot to "
        f"{task_html_path}"
    )


if __name__ == "__main__":
    main()
