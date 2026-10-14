from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any
from urllib.parse import quote

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(REPO_ROOT),
    )

from scripts.experiment_naming import (
    DEFAULT_DATA_OUTPUT_ROOT,
    DEFAULT_EXPERIMENT_OUTPUT_ROOT,
)


NEURAL_ALGORITHMS = [
    "dqn",
    "reinforce",
    "a2c",
    "ppo",
    "grpo",
]
NEURAL_ALGORITHM_ORDER = {
    algorithm: index
    for index, algorithm in enumerate(
        NEURAL_ALGORITHMS
    )
}


def file_url_for_path(
    path: Path,
) -> str:
    resolved_path = path.expanduser().resolve()
    wsl_distro = os.environ.get(
        "WSL_DISTRO_NAME"
    )

    if wsl_distro:
        return (
            "file://wsl.localhost/"
            f"{quote(wsl_distro, safe='')}"
            f"{quote(resolved_path.as_posix(), safe='/')}"
        )

    return resolved_path.as_uri()


def format_saved_file_message(
    prefix: str,
    path: Path,
    suffix: str = "",
) -> str:
    return (
        f"{prefix}{path}{suffix}\n"
        f"Open: {file_url_for_path(path)}"
    )

METRIC_LABELS = {
    "mean_path_efficiency": "Path efficiency",
    "success_rate": "Success rate",
    "average_episode_return": "Return",
    "steps": "Steps",
}
TRAIN_METRIC_ALIASES = {
    "mean_path_efficiency": "path_efficiency",
    "path_efficiency": "path_efficiency",
    "success_rate": "success",
    "success": "success",
    "average_episode_return": "episode_return",
    "episode_return": "episode_return",
    "steps": "steps",
}
BOUNDED_METRICS = {
    "mean_path_efficiency",
    "path_efficiency",
    "success_rate",
    "success",
}
SERIES_STYLES = {
    "validation": {
        "label": "val",
        "color": "#1f77b4",
        "linestyle": "-",
    },
    "same_layout": {
        "label": "same-val",
        "color": "#ff7f0e",
        "linestyle": "-",
    },
    "train": {
        "label": "train",
        "color": "#2ca02c",
        "linestyle": "-",
    },
    "train_greedy": {
        "label": "train (greedy)",
        "color": "#d62728",
        "linestyle": "--",
    },
}
SCORE_LABELS = {
    "train": "t",
    "validation": "v",
    "same_layout": "vs",
    "train_greedy": "tg",
}
SPLIT_ALIASES = {
    "validation": "validation",
    "val": "validation",
    "same_layout": "same_layout",
    "val_with_same_layout": "same_layout",
    "same-val": "same_layout",
    "same_val": "same_layout",
    "train": "train_greedy",
    "train_greedy": "train_greedy",
    "greedy_train": "train_greedy",
    "train_eval": "train_greedy",
}
PLOTTED_EVALUATION_SPLITS = (
    "validation",
    "same_layout",
    "train_greedy",
)


@dataclass(frozen=True)
class TrainingRunSeries:
    algorithm: str
    label: str
    metrics_path: Path
    train_points: list[tuple[int, float]]
    validation_points_by_split: dict[str, list[tuple[int, float]]]
    is_incomplete: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot DNN training histories under a runs directory."
        )
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help=(
            "Dataset bundle name under data/datasets or a dataset "
            "directory. Defaults --runs-dir to "
            "runs/dnn/training/<dataset>."
        ),
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=None,
        help=(
            "Directory to scan for training run metrics.csv files. "
            "Defaults to runs/dnn/training, or "
            "runs/dnn/training/<dataset> when --dataset is provided."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory for summary plots. Defaults to "
            "<runs-dir>/summary_plots."
        ),
    )
    parser.add_argument(
        "--metric",
        choices=sorted(TRAIN_METRIC_ALIASES),
        default="mean_path_efficiency",
        help="Training metric to average by dataset epoch.",
    )
    parser.add_argument(
        "--rolling-window",
        type=int,
        default=0,
        help=(
            "Number of dataset epochs used for rolling means. Use 0 to "
            "plot raw epoch means."
        ),
    )
    args = parser.parse_args()
    resolve_summary_paths(args)
    return args


def dataset_dir_for_reference(
    dataset: Path,
) -> Path:
    if dataset.is_absolute() or len(dataset.parts) > 1:
        return dataset

    return DEFAULT_DATA_OUTPUT_ROOT / dataset


def resolve_summary_paths(
    args: argparse.Namespace,
) -> None:
    if args.runs_dir is None:
        args.runs_dir = DEFAULT_EXPERIMENT_OUTPUT_ROOT

        if args.dataset is not None:
            args.runs_dir = (
                args.runs_dir
                / dataset_dir_for_reference(args.dataset).name
            )

    if args.output_dir is None:
        args.output_dir = args.runs_dir / "summary_plots"


def parse_float(value: str | None) -> float | None:
    if value in (None, ""):
        return None

    if value == "True":
        return 1.0
    if value == "False":
        return 0.0

    try:
        return float(value)
    except ValueError:
        return None


def infer_algorithm(metrics_path: Path) -> str:
    return metrics_path.parent.name


def canonical_metric(metric: str) -> str:
    if metric == "path_efficiency":
        return "mean_path_efficiency"
    if metric == "success":
        return "success_rate"
    if metric == "episode_return":
        return "average_episode_return"
    return metric


def metric_label(metric: str) -> str:
    return METRIC_LABELS.get(
        canonical_metric(metric),
        metric,
    )


def metric_column(metric: str) -> str:
    return TRAIN_METRIC_ALIASES[metric]


def validation_metric_column(
    metric: str,
) -> str | None:
    canonical = canonical_metric(metric)

    if canonical == "steps":
        return None

    return canonical


def canonical_split(
    split: str | None,
) -> str | None:
    if split is None:
        return None

    return SPLIT_ALIASES.get(
        split,
    )


def run_label(
    metrics_path: Path,
    runs_dir: Path,
) -> str:
    try:
        return metrics_path.parent.relative_to(
            runs_dir
        ).as_posix()
    except ValueError:
        return metrics_path.parent.as_posix()


def discover_metrics_paths(
    runs_dir: Path,
) -> list[Path]:
    if not runs_dir.exists():
        return []

    return [
        path
        for path in sorted(
            runs_dir.rglob("metrics.csv")
        )
    ]


def read_training_summary(
    path: Path,
) -> dict[str, Any]:
    with path.open() as file:
        return json.load(file)


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


def run_activity_paths(
    run_dir: Path,
) -> list[Path]:
    paths = [
        run_dir / "metrics.csv",
        run_dir / "validation_metrics.csv",
        run_dir / "training_state.pt",
        run_dir / "checkpoint.pt",
        run_dir / "best_checkpoint.pt",
        run_dir / "checkpoint.pkl",
        run_dir / "best_checkpoint.pkl",
    ]

    return [
        path
        for path in paths
        if path.exists()
    ]


def run_has_stale_completion_summary(
    summary_path: Path,
) -> bool:
    summary_mtime_ns = summary_path.stat().st_mtime_ns

    return any(
        path.stat().st_mtime_ns > summary_mtime_ns
        for path in run_activity_paths(
            summary_path.parent
        )
    )


def training_summary_completion_status(
    path: Path,
) -> bool | None:
    if not path.exists():
        return None

    if run_has_stale_completion_summary(path):
        return False

    try:
        training_summary = read_training_summary(
            path
        )
    except (OSError, json.JSONDecodeError):
        return None

    if not isinstance(
        training_summary,
        dict,
    ):
        return None

    return training_summary_completed(
        training_summary
    )


def run_completion_status(
    metrics_path: Path,
) -> bool | None:
    return training_summary_completion_status(
        metrics_path.parent / "training_summary.json"
    )


def run_is_incomplete(
    metrics_path: Path,
) -> bool:
    return run_completion_status(metrics_path) is not True


def is_neural_training_run(
    metrics_path: Path,
) -> bool:
    return infer_algorithm(metrics_path) in NEURAL_ALGORITHM_ORDER


def load_training_run_series(
    metrics_path: Path,
    *,
    runs_dir: Path,
    metric: str,
) -> TrainingRunSeries | None:
    if not is_neural_training_run(metrics_path):
        return None

    values_by_epoch: dict[int, list[float]] = {}
    column_name = metric_column(metric)

    with metrics_path.open(
        newline="",
    ) as file:
        reader = csv.DictReader(file)

        for row in reader:
            value = parse_float(
                row.get(column_name)
            )

            if value is None:
                continue

            epoch_value = parse_float(
                row.get("epoch")
            )

            if epoch_value is None:
                continue

            values_by_epoch.setdefault(
                int(epoch_value),
                [],
            ).append(value)

    if not values_by_epoch:
        return None

    train_points = [
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

    return TrainingRunSeries(
        algorithm=infer_algorithm(metrics_path),
        label=run_label(
            metrics_path,
            runs_dir,
        ),
        metrics_path=metrics_path,
        train_points=train_points,
        validation_points_by_split=(
            load_validation_run_series(
                metrics_path.parent / "validation_metrics.csv",
                metric=metric,
            )
        ),
        is_incomplete=run_is_incomplete(metrics_path),
    )


def load_validation_run_series(
    metrics_path: Path,
    *,
    metric: str,
) -> dict[str, list[tuple[int, float]]]:
    column_name = validation_metric_column(
        metric
    )

    if column_name is None or not metrics_path.exists():
        return {}

    values_by_split_epoch: dict[
        str,
        dict[int, list[float]],
    ] = {}

    with metrics_path.open(
        newline="",
    ) as file:
        reader = csv.DictReader(file)

        for row in reader:
            value = parse_float(
                row.get(column_name)
            )

            if value is None:
                continue

            epoch_value = parse_float(
                row.get("dataset_epoch")
            )

            if epoch_value is None:
                continue

            split = canonical_split(
                row.get(
                    "split",
                    "validation",
                )
            )
            if split is None:
                continue

            values_by_split_epoch.setdefault(
                split,
                {},
            ).setdefault(
                int(epoch_value),
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
                values_by_epoch.items()
            )
        ]
        for split, values_by_epoch in (
            values_by_split_epoch.items()
        )
    }


def rolling_points(
    points: list[tuple[int, float]],
    window: int,
) -> list[tuple[int, float]]:
    if window <= 1:
        return points

    smoothed = []

    for index, (epoch, _) in enumerate(points):
        start = max(
            0,
            index - window + 1,
        )
        values = [
            value
            for _, value in points[start : index + 1]
        ]
        smoothed.append(
            (
                epoch,
                float(
                    np.mean(values)
                ),
            )
        )

    return smoothed


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


def summary_axis_ticks(
    max_epoch: int,
) -> list[int]:
    if max_epoch <= 6:
        return list(range(max_epoch + 1))

    rough_step = max_epoch / 4
    magnitude = 10 ** np.floor(
        np.log10(rough_step)
    )

    step = int(magnitude)
    for multiplier in (
        1,
        2,
        5,
        10,
    ):
        candidate = int(
            multiplier * magnitude
        )

        if candidate >= rough_step:
            step = max(
                1,
                candidate,
            )
            break

    ticks = list(
        range(
            0,
            max_epoch + 1,
            step,
        )
    )

    if ticks[-1] != max_epoch:
        ticks.append(max_epoch)

    return ticks


def plotted_max_epoch(
    points: list[tuple[int, float]],
) -> int:
    return max(
        [
            epoch
            for epoch, _ in points
        ],
        default=0,
    )


def value_at_epoch(
    points: list[tuple[int, float]],
    epoch: int,
) -> float | None:
    for point_epoch, value in points:
        if point_epoch == epoch:
            return value

    return None


def best_validation_epoch(
    item: TrainingRunSeries,
) -> int | None:
    validation_points = item.validation_points_by_split.get(
        "validation"
    )

    if not validation_points:
        return None

    return min(
        validation_points,
        key=lambda point: (
            -point[1],
            point[0],
        ),
    )[0]


def selected_epoch_score_summary(
    item: TrainingRunSeries,
) -> str:
    epoch = best_validation_epoch(item)

    if epoch is None:
        best_train = max(
            value
            for _, value in item.train_points
        )
        return f"best {SCORE_LABELS['train']} {best_train:.2f}"

    scores = []
    train_score = value_at_epoch(
        item.train_points,
        epoch,
    )

    if train_score is not None:
        scores.append(
            f"{SCORE_LABELS['train']} {train_score:.2f}"
        )

    for split in PLOTTED_EVALUATION_SPLITS:
        score = value_at_epoch(
            item.validation_points_by_split.get(
                split,
                [],
            ),
            epoch,
        )

        if score is None:
            continue

        scores.append(
            f"{SCORE_LABELS[split]} {score:.2f}"
        )

    if not scores:
        return f"{SCORE_LABELS['validation']} epoch {epoch}"

    return "  ".join(scores)


def subplot_title(
    item: TrainingRunSeries,
) -> str:
    title = item.algorithm

    title = (
        f"{title}\n"
        f"{selected_epoch_score_summary(item)}"
    )

    return title


def draw_incomplete_label(axis) -> None:
    axis.text(
        0.02,
        0.96,
        "INCOMPLETE",
        transform=axis.transAxes,
        ha="left",
        va="top",
        color="#c62828",
        fontsize=11,
        fontweight="bold",
        bbox={
            "boxstyle": "round,pad=0.25",
            "facecolor": "white",
            "edgecolor": "#c62828",
            "linewidth": 1.0,
            "alpha": 0.92,
        },
    )


def draw_best_epoch_marker(
    axis,
    epoch: int | None,
) -> None:
    if epoch is None:
        return

    x_min, x_max = axis.get_xlim()
    x_mid = (x_min + x_max) / 2
    horizontal_alignment = (
        "right"
        if epoch > x_mid
        else "left"
    )
    point_offset = (
        -3
        if epoch > x_mid
        else 3
    )

    axis.axvline(
        epoch,
        color="#808080",
        linestyle=":",
        linewidth=1.6,
        alpha=1.0,
        zorder=1,
    )

    import matplotlib.transforms as transforms

    axis.text(
        epoch,
        0.03,
        f"best val epoch {epoch}",
        transform=(
            axis.get_xaxis_transform()
            + transforms.ScaledTranslation(
                point_offset / 72,
                0,
                axis.figure.dpi_scale_trans,
            )
        ),
        ha=horizontal_alignment,
        va="bottom",
        color="#666666",
        fontsize=9,
        bbox={
            "boxstyle": "round,pad=0.18",
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.9,
        },
    )


def collect_training_run_series(
    runs_dir: Path,
    *,
    metric: str,
) -> list[TrainingRunSeries]:
    series = []

    for metrics_path in discover_metrics_paths(
        runs_dir,
    ):
        run_series = load_training_run_series(
            metrics_path,
            runs_dir=runs_dir,
            metric=metric,
        )

        if run_series is not None:
            series.append(run_series)

    return series


def plot_training_run_summary(
    series: list[TrainingRunSeries],
    *,
    metric: str,
    output_path: Path,
    rolling_window: int = 0,
) -> None:
    if not series:
        raise RuntimeError(
            "No DNN training metrics found to plot."
        )

    os.environ.setdefault(
        "MPLCONFIGDIR",
        "/tmp/matplotlib",
    )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sorted_series = sorted(
        series,
        key=lambda item: (
            NEURAL_ALGORITHM_ORDER[item.algorithm],
            item.label,
        ),
    )
    column_count = min(
        3,
        len(sorted_series),
    )
    row_count = int(
        np.ceil(
            len(sorted_series) / column_count
        )
    )
    label = metric_label(metric)
    figure, axes = plt.subplots(
        row_count,
        column_count,
        figsize=(
            7.0 * column_count,
            5.2 * row_count,
        ),
        sharey=False,
        squeeze=False,
    )

    for axis in axes.reshape(-1):
        axis.axis("off")

    legend_handles = None
    legend_labels = None

    for index, (axis, item) in enumerate(zip(
        axes.reshape(-1),
        sorted_series,
    )):
        axis.axis("on")
        train_points = rolling_points(
            item.train_points,
            rolling_window,
        )
        plotted_point_sets = [
            train_points
        ]
        train_style = SERIES_STYLES["train"]
        axis.plot(
            [epoch for epoch, _ in train_points],
            [value for _, value in train_points],
            marker="o",
            markersize=3.8,
            linewidth=2.4,
            color=train_style["color"],
            linestyle=train_style["linestyle"],
            label=train_style["label"],
        )

        for split in PLOTTED_EVALUATION_SPLITS:
            points = item.validation_points_by_split.get(
                split
            )

            if not points:
                continue

            style = SERIES_STYLES[split]
            plotted_point_sets.append(points)
            axis.plot(
                [epoch for epoch, _ in points],
                [value for _, value in points],
                marker="o",
                markersize=3.8,
                linewidth=2.4,
                color=style["color"],
                linestyle=style["linestyle"],
                label=style["label"],
            )

        axis.set_title(
            subplot_title(item),
            fontsize=13,
            pad=8,
        )
        if item.is_incomplete:
            draw_incomplete_label(axis)

        axis.tick_params(
            labelsize=10,
        )
        axis.grid(
            True,
            alpha=0.22,
            linewidth=0.8,
        )

        max_epoch = max(
            plotted_max_epoch(points)
            for points in plotted_point_sets
        )
        axis.set_xlim(
            0,
            max_epoch,
        )
        axis.set_xticks(
            summary_axis_ticks(max_epoch)
        )
        draw_best_epoch_marker(
            axis,
            best_validation_epoch(item),
        )

        if metric in BOUNDED_METRICS:
            axis.set_ylim(
                0.0,
                1.0,
            )

        axis.set_xlabel(
            "Dataset epoch",
            fontsize=11,
        )
        axis.set_ylabel(
            label,
            fontsize=11,
        )

        legend_handles, legend_labels = (
            axis.get_legend_handles_labels()
        )

    title = (
        "DNN training metrics: "
        f"{label}"
    )

    if rolling_window > 1:
        title += f" ({rolling_window}-epoch rolling mean)"

    figure.suptitle(
        title,
        fontsize=16,
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
            bbox_to_anchor=(0.5, 0.95),
            frameon=False,
            fontsize=11,
        )

    figure.tight_layout(
        rect=(
            0,
            0,
            1,
            0.92,
        )
    )
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    figure.savefig(
        output_path,
        dpi=160,
        facecolor="white",
        edgecolor="white",
        transparent=False,
    )
    rewrite_png_without_alpha(output_path)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    series = collect_training_run_series(
        args.runs_dir,
        metric=args.metric,
    )
    output_path = (
        args.output_dir
        / f"{canonical_metric(args.metric)}_training_history_summary.png"
    )

    plot_training_run_summary(
        series,
        metric=args.metric,
        output_path=output_path,
        rolling_window=args.rolling_window,
    )

    print(
        format_saved_file_message(
            f"Saved {len(series)} training run curve(s) to ",
            output_path,
        )
    )


if __name__ == "__main__":
    main()
