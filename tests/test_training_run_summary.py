from __future__ import annotations

import os
from pathlib import Path

from scripts.summarize_training_runs import (
    collect_training_run_series,
    file_url_for_path,
    format_saved_file_message,
    plot_training_run_summary,
    resolve_summary_paths,
    subplot_title,
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


def write_training_summary(
    path: Path,
    *,
    completed: bool,
    stopped_early: bool = False,
    status: str | None = None,
) -> None:
    status_field = (
        f'"status": "{status}",'
        if status is not None
        else ""
    )
    path.write_text(
        (
            "{"
            f"{status_field}"
            '"task_selection_passes_requested": 3,'
            f'"task_selection_passes_completed": {3 if completed else 2},'
            f'"dataset_epochs_completed": {3 if completed else 2},'
            f'"stopped_early": {str(stopped_early).lower()}'
            "}"
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


def test_file_url_for_path_uses_local_file_uri_without_wsl(
    tmp_path,
    monkeypatch,
):
    monkeypatch.delenv(
        "WSL_DISTRO_NAME",
        raising=False,
    )
    output_path = tmp_path / "summary image.png"

    assert file_url_for_path(output_path) == (
        output_path.resolve().as_uri()
    )


def test_file_url_for_path_uses_wsl_unc_uri(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv(
        "WSL_DISTRO_NAME",
        "Ubuntu Test",
    )
    output_path = tmp_path / "summary image.png"

    assert file_url_for_path(output_path) == (
        "file://wsl.localhost/Ubuntu%20Test"
        f"{output_path.resolve().as_posix().replace(' ', '%20')}"
    )


def test_format_saved_file_message_includes_clickable_url(
    tmp_path,
    monkeypatch,
):
    monkeypatch.delenv(
        "WSL_DISTRO_NAME",
        raising=False,
    )
    output_path = tmp_path / "summary.png"

    assert format_saved_file_message(
        "Saved plot to ",
        output_path,
        " (1 item)",
    ) == (
        f"Saved plot to {output_path} (1 item)\n"
        f"Open: {output_path.resolve().as_uri()}"
    )


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
            ("same_layout", 2, 0.55),
            ("train_greedy", 1, 0.45),
            ("train", 2, 0.90),
            ("validation", 2, 0.50),
        ],
    )
    write_training_summary(
        runs_dir / "ppo_run" / "ppo" / "training_summary.json",
        completed=True,
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
        (2, 0.55),
    ]
    assert series[0].validation_points_by_split[
        "train_greedy"
    ] == [
        (1, 0.45),
        (2, 0.9),
    ]
    assert series[0].is_incomplete is False
    assert subplot_title(series[0]) == (
        "ppo\n"
        "t 0.80  "
        "v 0.50  vs 0.55  "
        "tg 0.90"
    )


def test_collect_training_run_series_discovers_residual_run(
    tmp_path,
):
    runs_dir = tmp_path / "runs"
    write_metrics_csv(
        runs_dir
        / "residual_run"
        / "ppo_residual"
        / "metrics.csv",
        [(1, 0.6, True)],
    )

    series = collect_training_run_series(
        runs_dir,
        metric="path_efficiency",
    )

    assert len(series) == 1
    assert series[0].algorithm == "ppo"
    assert series[0].label == (
        "residual_run/ppo_residual"
    )


def test_collect_training_run_series_discovers_mixed_baseline_and_residual_runs(
    tmp_path,
):
    runs_dir = tmp_path / "runs"
    write_metrics_csv(
        runs_dir / "baseline_run" / "ppo" / "metrics.csv",
        [(1, 0.4, True)],
    )
    write_metrics_csv(
        runs_dir
        / "residual_run"
        / "ppo_residual"
        / "metrics.csv",
        [(1, 0.6, True)],
    )

    series = collect_training_run_series(
        runs_dir,
        metric="path_efficiency",
    )

    assert [item.algorithm for item in series] == [
        "ppo",
        "ppo",
    ]
    assert {
        item.label
        for item in series
    } == {
        "baseline_run/ppo",
        "residual_run/ppo_residual",
    }


def test_collect_training_run_series_marks_incomplete_runs(
    tmp_path,
):
    runs_dir = tmp_path / "runs"
    run_dir = runs_dir / "ppo_run" / "ppo"
    write_metrics_csv(
        run_dir / "metrics.csv",
        [(1, 0.2, True)],
    )
    write_training_summary(
        run_dir / "training_summary.json",
        completed=False,
    )

    series = collect_training_run_series(
        runs_dir,
        metric="path_efficiency",
    )

    assert len(series) == 1
    assert series[0].is_incomplete is True


def test_collect_training_run_series_treats_early_stop_as_complete(
    tmp_path,
):
    runs_dir = tmp_path / "runs"
    run_dir = runs_dir / "ppo_run" / "ppo"
    write_metrics_csv(
        run_dir / "metrics.csv",
        [(1, 0.2, True)],
    )
    write_training_summary(
        run_dir / "training_summary.json",
        completed=False,
        stopped_early=True,
    )

    series = collect_training_run_series(
        runs_dir,
        metric="path_efficiency",
    )

    assert len(series) == 1
    assert series[0].is_incomplete is False


def test_collect_training_run_series_uses_explicit_status(
    tmp_path,
):
    runs_dir = tmp_path / "runs"
    running_dir = runs_dir / "running" / "ppo"
    early_stop_dir = runs_dir / "early_stop" / "a2c"
    completed_dir = runs_dir / "completed" / "dqn"

    for run_dir in [
        running_dir,
        early_stop_dir,
        completed_dir,
    ]:
        write_metrics_csv(
            run_dir / "metrics.csv",
            [(1, 0.2, True)],
        )

    write_training_summary(
        running_dir / "training_summary.json",
        completed=True,
        stopped_early=True,
        status="running",
    )
    write_training_summary(
        early_stop_dir / "training_summary.json",
        completed=False,
        status="early_stopped",
    )
    write_training_summary(
        completed_dir / "training_summary.json",
        completed=False,
        status="completed",
    )

    series = collect_training_run_series(
        runs_dir,
        metric="path_efficiency",
    )

    incomplete_by_label = {
        item.label: item.is_incomplete
        for item in series
    }

    assert incomplete_by_label == {
        "completed/dqn": False,
        "early_stop/a2c": False,
        "running/ppo": True,
    }


def test_collect_training_run_series_marks_stale_summary_incomplete(
    tmp_path,
):
    runs_dir = tmp_path / "runs"
    run_dir = runs_dir / "ppo_run" / "ppo"
    write_metrics_csv(
        run_dir / "metrics.csv",
        [(1, 0.2, True)],
    )
    write_training_summary(
        run_dir / "training_summary.json",
        completed=False,
        stopped_early=True,
    )
    training_state_path = run_dir / "training_state.pt"
    training_state_path.write_bytes(b"state")
    os.utime(
        run_dir / "training_summary.json",
        ns=(2_000, 2_000),
    )
    os.utime(
        training_state_path,
        ns=(3_000, 3_000),
    )

    series = collect_training_run_series(
        runs_dir,
        metric="path_efficiency",
    )

    assert len(series) == 1
    assert series[0].is_incomplete is True


def test_collect_training_run_series_marks_missing_summary_incomplete(
    tmp_path,
):
    runs_dir = tmp_path / "runs"
    run_dir = runs_dir / "ppo_run" / "ppo"
    write_metrics_csv(
        run_dir / "metrics.csv",
        [(1, 0.2, True)],
    )

    series = collect_training_run_series(
        runs_dir,
        metric="path_efficiency",
    )

    assert len(series) == 1
    assert series[0].is_incomplete is True


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


def test_plot_training_run_summary_labels_incomplete_runs(
    tmp_path,
    monkeypatch,
):
    runs_dir = tmp_path / "runs"
    run_dir = runs_dir / "ppo" / "ppo"
    write_metrics_csv(
        run_dir / "metrics.csv",
        [(1, 0.2, True), (2, 0.5, True)],
    )
    write_training_summary(
        run_dir / "training_summary.json",
        completed=False,
    )
    series = collect_training_run_series(
        runs_dir,
        metric="mean_path_efficiency",
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
                "vlines": [
                    tuple(line.get_xdata())
                    for line in self.axes[0].lines
                    if line.get_linestyle() == ":"
                ],
            }
        )

    monkeypatch.setattr(
        matplotlib.figure.Figure,
        "savefig",
        fake_savefig,
    )

    plot_training_run_summary(
        series,
        metric="mean_path_efficiency",
        output_path=tmp_path / "summary.png",
    )

    assert saved_figures == [
        {
            "title": (
                "ppo\n"
                "best t 0.50"
            ),
            "texts": ["INCOMPLETE"],
            "text_colors": ["#c62828"],
            "vlines": [],
        }
    ]
