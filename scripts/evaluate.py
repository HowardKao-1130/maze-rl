from __future__ import annotations

import argparse
import csv
from pathlib import Path
import pickle
import random
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(SRC_ROOT),
    )

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
from maze_rl.evaluation.rollout_animations import (
    plot_rollout_animations,
)
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


def default_rollout_animation_output_dir(
    checkpoint_path: Path,
) -> Path:
    return checkpoint_path.parent / "rollout_animations"


def default_tuning_results_path() -> Path:
    return Path("runs") / "tuning" / "tuning_results.csv"


def infer_dataset_label(
    dataset_path: str | Path,
) -> str:
    filename = Path(dataset_path).name

    if filename == "validation.npz":
        return "val"

    if filename == "same_layout_new_goals.npz":
        return "val_with_same_layout"

    return Path(dataset_path).stem


def evaluation_csv_rows(
    results: list[dict],
) -> list[dict]:
    return [
        {
            key: value
            for key, value in result.items()
            if key != "rollout_trace"
        }
        for result in results
    ]


def same_layout_dataset_for(
    dataset_path: Path,
) -> Path | None:
    if dataset_path.name != "validation.npz":
        return None

    same_layout_path = (
        dataset_path.parent / "same_layout_new_goals.npz"
    )

    if not same_layout_path.exists():
        return None

    return same_layout_path


def progress_line(
    label: str,
    current: int,
    total: int,
) -> None:
    display_current = max(
        0,
        min(
            current,
            total,
        ),
    )
    percent = (
        100
        if total <= 0
        else int(
            round(
                100 * display_current / total
            )
        )
    )
    percent = max(
        0,
        min(
            100,
            percent,
        ),
    )
    sys.stdout.write(
        f"\r{label} [{percent:3d}%] ({display_current}/{total})"
    )
    sys.stdout.flush()


def finish_progress_line(
    label: str,
) -> None:
    sys.stdout.write(
        f"\r{label} step done{' ' * 20}\n"
    )
    sys.stdout.flush()


def best_trial_checkpoint_path(
    algorithm: str,
    tuning_results_path: Path,
) -> Path:
    with tuning_results_path.open(
        newline="",
    ) as file:
        rows = [
            row
            for row in csv.DictReader(file)
            if row.get("algorithm") == algorithm
            and row.get("checkpoint_path")
        ]

    if not rows:
        raise ValueError(
            "No tuning_results.csv rows found for "
            f"algorithm {algorithm!r} in {tuning_results_path}."
        )

    try:
        best_row = max(
            rows,
            key=lambda row: float(
                row["objective"]
            ),
        )
    except (
        KeyError,
        TypeError,
        ValueError,
    ) as error:
        raise ValueError(
            "Could not select a best trial because "
            f"{tuning_results_path} has invalid objective values."
        ) from error

    return Path(
        best_row["checkpoint_path"]
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
        "--best-trial",
        action="store_true",
        help=(
            "Use tuning_results.csv to evaluate the best recorded "
            "trial checkpoint for --algorithm."
        ),
    )

    parser.add_argument(
        "--tuning-results",
        type=Path,
        default=default_tuning_results_path(),
        help=(
            "CSV index used by --best-trial. Defaults to "
            "runs/tuning/tuning_results.csv."
        ),
    )

    parser.add_argument(
        "--dataset",
        type=Path,
        default="data/test.npz",
    )

    parser.add_argument(
        "--dataset-label",
        default=None,
        help=(
            "Label used in rollout animations. Defaults to val "
            "for validation.npz, val_with_same_layout for "
            "same_layout_new_goals.npz, otherwise the dataset stem."
        ),
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=200,
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help=(
            "Sample this many random tasks for evaluation. "
            "Omitting this flag evaluates every dataset task once."
        ),
    )

    parser.add_argument(
        "--fixed-index",
        type=int,
        default=None,
        help=(
            "Evaluate one fixed task index. Defaults to the "
            "legacy 200 episodes unless --episodes is provided."
        ),
    )

    parser.add_argument(
        "--all-tasks",
        action="store_true",
        help=(
            "Evaluate every task in the dataset once. This is the "
            "default and remains accepted for compatibility."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help=(
            "Seed used for reproducible task sampling and "
            "deterministic tie-breaking."
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
        "--rollout-animation-output-dir",
        type=Path,
        default=None,
        help=(
            "Directory for validation rollout MP4 animations. "
            "Defaults to <checkpoint directory>/rollout_animations."
        ),
    )

    parser.add_argument(
        "--no-rollout-animations",
        action="store_true",
        help="Skip validation rollout MP4 animations.",
    )

    parser.add_argument(
        "--rollout-animation-fps",
        type=float,
        default=8.0,
        help="Frames per second for rollout MP4 animations.",
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

    if (
        args.episodes is not None
        and args.episodes <= 0
    ):
        parser.error(
            "--episodes must be positive."
        )

    if (
        args.all_tasks
        and args.fixed_index is not None
    ):
        parser.error(
            "--all-tasks cannot be combined with --fixed-index."
        )

    if (
        args.best_trial
        and args.checkpoint is not None
    ):
        parser.error(
            "--best-trial cannot be combined with --checkpoint."
        )

    if args.checkpoint is None:
        if args.best_trial:
            try:
                args.checkpoint = best_trial_checkpoint_path(
                    args.algorithm,
                    args.tuning_results,
                )
            except (
                FileNotFoundError,
                ValueError,
            ) as error:
                parser.error(str(error))
        else:
            args.checkpoint = default_checkpoint_path(
                args.algorithm
            )

    if args.dataset_label is None:
        args.dataset_label = infer_dataset_label(
            args.dataset
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


def evaluation_episode_plan(
    env,
    *,
    episodes: int | None,
    fixed_index: int | None,
    all_tasks: bool,
) -> tuple[int, list[int] | None]:
    if fixed_index is not None:
        episode_count = (
            episodes
            if episodes is not None
            else 200
        )
        return episode_count, [
            fixed_index
        ] * episode_count

    if (
        all_tasks
        or episodes is None
    ):
        task_indices = list(
            range(env.num_tasks)
        )
        return len(task_indices), task_indices

    return episodes, None


def evaluate_dataset(
    *,
    algorithm,
    agent,
    dataset_path: Path,
    max_steps: int,
    seed: int,
    episodes: int | None,
    fixed_index: int | None,
    all_tasks: bool,
    capture_rollouts: bool,
    progress_label: str,
):
    env = MazeEnv(
        dataset_path=dataset_path,
        max_steps=max_steps,
    )
    episode_count, task_indices = evaluation_episode_plan(
        env,
        episodes=episodes,
        fixed_index=fixed_index,
        all_tasks=all_tasks,
    )
    progress_total = (
        len(task_indices)
        if task_indices is not None
        else episode_count
    )

    progress_line(
        progress_label,
        0,
        progress_total,
    )
    results, summary = evaluate(
        algorithm=algorithm,
        agent=agent,
        env=env,
        episodes=episode_count,
        seed=seed,
        task_indices=task_indices,
        capture_rollouts=capture_rollouts,
        progress_callback=lambda current, total: progress_line(
            progress_label,
            current,
            total,
        ),
    )
    finish_progress_line(progress_label)

    return env, results, summary


def main():
    args = parse_args()

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

    agent_env = MazeEnv(
        dataset_path=args.dataset,
        max_steps=args.max_steps,
    )

    agent = create_agent(
        args.algorithm,
        agent_env,
        device,
        args.seed,
    )

    checkpoint = load_checkpoint(
        args.algorithm,
        args.checkpoint,
        agent,
        device,
    )

    env, results, summary = evaluate_dataset(
        algorithm=args.algorithm,
        agent=agent,
        dataset_path=args.dataset,
        max_steps=args.max_steps,
        seed=args.seed,
        episodes=args.episodes,
        fixed_index=args.fixed_index,
        all_tasks=args.all_tasks,
        capture_rollouts=not args.no_rollout_animations,
        progress_label=f"Evaluate {args.dataset_label}",
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

    csv_rows = evaluation_csv_rows(results)

    with evaluation_path.open(
        "w",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=csv_rows[0].keys(),
        )

        writer.writeheader()
        writer.writerows(csv_rows)

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

    if not args.no_rollout_animations:
        rollout_animation_dir = (
            args.rollout_animation_output_dir
            if args.rollout_animation_output_dir is not None
            else default_rollout_animation_output_dir(
                args.checkpoint
            )
        )

        animation_inputs = [
            (
                env,
                results,
                args.dataset_label,
            )
        ]
        same_layout_dataset = same_layout_dataset_for(
            args.dataset
        )

        if same_layout_dataset is not None:
            (
                same_layout_env,
                same_layout_results,
                _,
            ) = evaluate_dataset(
                algorithm=args.algorithm,
                agent=agent,
                dataset_path=same_layout_dataset,
                max_steps=args.max_steps,
                seed=args.seed,
                episodes=args.episodes,
                fixed_index=args.fixed_index,
                all_tasks=args.all_tasks,
                capture_rollouts=True,
                progress_label="Evaluate val_with_same_layout",
            )
            animation_inputs.append(
                (
                    same_layout_env,
                    same_layout_results,
                    "val_with_same_layout",
                )
            )

        saved_animation_count = 0
        skipped_animation_reason = None

        for animation_env, animation_results, split_label in animation_inputs:
            progress_label = f"Animate {split_label}"

            rollout_animation_result = plot_rollout_animations(
                env=animation_env,
                results=animation_results,
                output_dir=rollout_animation_dir,
                split_label=split_label,
                fps=args.rollout_animation_fps,
                progress_callback=lambda current, total, label=progress_label: progress_line(
                    label,
                    current,
                    total,
                ),
                algorithm=args.algorithm,
                agent=agent,
                q_table=checkpoint.get("q"),
            )

            if rollout_animation_result.animation_count:
                saved_animation_count += (
                    rollout_animation_result.animation_count
                )
                finish_progress_line(progress_label)
            elif rollout_animation_result.skipped_animation_reason:
                skipped_animation_reason = (
                    rollout_animation_result.skipped_animation_reason
                )
                sys.stdout.write(
                    f"\r{progress_label} skipped{' ' * 20}\n"
                )
                sys.stdout.flush()
                break

        if saved_animation_count:
            print(
                f"Saved {saved_animation_count} "
                "rollout animation videos to "
                f"{rollout_animation_dir}"
            )

        elif skipped_animation_reason:
            print(
                "Skipped rollout animations: "
                f"{skipped_animation_reason}"
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
