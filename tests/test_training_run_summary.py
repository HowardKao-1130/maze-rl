from __future__ import annotations

from pathlib import Path

from scripts.summarize_training_runs import (
    collect_training_run_series,
    plot_training_run_summary,
    resolve_summary_paths,
)


def write_metrics_csv(
    path: Path,
    rows: list[tuple[int, float, bool]],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    path.write_text(
        "episode,epoch,task_index,layout_index,episode_return,"
        "steps,internal_updates,success,wall_collisions,"
        "optimal_path_length,path_efficiency,mean_abs_td_error,"
        "loss,policy_loss,value_loss,entropy,approximate_kl,"
        "clip_fraction,mean_value,mean_return,mean_advantage,"
        "mean_q_value,max_q_value,replay_size\n"
        + "".join(
            (
                f"{index},{epoch},0,0,1.0,10,,{success},0,10,"
                f"{path_efficiency},,,,,,,,,,,,,\n"
            )
            for index, (
                epoch,
                path_efficiency,
                success,
            ) in enumerate(
                rows,
                start=1,
            )
        )
    )


def write_validation_metrics_csv(
    path: Path,
    rows: list[tuple[str, int, float]],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    path.write_text(
        "split,dataset_epoch,episodes,success_rate,"
        "average_episode_return,mean_path_efficiency,"
        "average_successful_path_efficiency\n"
        + "".join(
            (
                f"{split},{epoch},2,{value},1.0,"
                f"{value},{value}\n"
            )
            for split, epoch, value in rows
        )
    )


def test_resolve_summary_paths_uses_dataset_name(
    tmp_path,
):
    args = type(
        "Args",
        (),
        {
            "dataset": tmp_path / "dataset_alpha",
            "runs_dir": None,
            "output_dir": None,
        },
    )()

    resolve_summary_paths(args)

    assert args.runs_dir == (
        Path("runs") / "dnn" / "training" / "dataset_alpha"
    )
    assert args.output_dir == args.runs_dir / "summary_plots"


def test_collect_training_run_series_averages_by_epoch(
    tmp_path,
):
    runs_dir = tmp_path / "runs"
    write_metrics_csv(
        runs_dir / "ppo_run" / "ppo" / "metrics.csv",
        [
            (1, 0.2, True),
            (1, 0.4, False),
            (2, 0.8, True),
        ],
    )
    write_validation_metrics_csv(
        runs_dir / "ppo_run" / "ppo" / "validation_metrics.csv",
        [
            ("validation", 1, 0.25),
            ("same_layout", 1, 0.35),
            ("train_greedy", 1, 0.45),
            ("train", 2, 0.90),
            ("validation", 2, 0.50),
        ],
    )

    series = collect_training_run_series(
        runs_dir,
        metric="path_efficiency",
    )

    assert len(series) == 1
    assert series[0].algorithm == "ppo"
    assert series[0].label == "ppo_run/ppo"
    assert series[0].train_points == [
        (1, 0.30000000000000004),
        (2, 0.8),
    ]
    assert series[0].validation_points_by_split[
        "validation"
    ] == [
        (1, 0.25),
        (2, 0.5),
    ]
    assert series[0].validation_points_by_split[
        "same_layout"
    ] == [
        (1, 0.35),
    ]
    assert series[0].validation_points_by_split[
        "train_greedy"
    ] == [
        (1, 0.45),
        (2, 0.9),
    ]


def test_plot_training_run_summary_writes_one_image(
    tmp_path,
):
    runs_dir = tmp_path / "runs"
    write_metrics_csv(
        runs_dir / "ppo" / "metrics.csv",
        [(1, 0.2, True), (2, 0.5, True)],
    )
    write_metrics_csv(
        runs_dir / "tabular" / "q_learning" / "metrics.csv",
        [(1, 0.1, False), (2, 0.3, True)],
    )
    series = collect_training_run_series(
        runs_dir,
        metric="mean_path_efficiency",
    )
    output_path = tmp_path / "summary.png"

    assert [
        item.algorithm
        for item in series
    ] == ["ppo"]

    plot_training_run_summary(
        series,
        metric="mean_path_efficiency",
        output_path=output_path,
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0
