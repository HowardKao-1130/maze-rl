from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot training history for all non-tuning runs under a runs "
            "directory."
        )
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=Path("runs"),
        help="Directory to scan for training run metrics.csv files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/training_history_summary.png"),
        help="Output PNG path.",
    )
    parser.add_argument(
        "--metric",
        choices=sorted(TRAIN_METRIC_ALIASES),
        default="mean_path_efficiency",
        help="Training metric to average by dataset epoch.",
    )
    parser.add_argument(
        "--exclude-dir",
        action="append",
        default=["tuning"],
        help=(
            "Directory name under --runs-dir to exclude. Can be "
            "passed more than once. Defaults to tuning."
        ),
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
    return parser.parse_args()


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


def is_excluded_metrics_path(
    metrics_path: Path,
    runs_dir: Path,
    excluded_dirs: set[str],
) -> bool:
    try:
        relative_path = metrics_path.relative_to(
            runs_dir
        )
    except ValueError:
        return False

    return bool(
        set(relative_path.parts).intersection(excluded_dirs)
    )


def discover_metrics_paths(
    runs_dir: Path,
    *,
    excluded_dirs: set[str],
) -> list[Path]:
    if not runs_dir.exists():
        return []

    return [
        path
        for path in sorted(
            runs_dir.rglob("metrics.csv")
        )
        if not is_excluded_metrics_path(
            path,
            runs_dir,
            excluded_dirs,
        )
    ]


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


def collect_training_run_series(
    runs_dir: Path,
    *,
    metric: str,
    excluded_dirs: set[str],
) -> list[TrainingRunSeries]:
    series = []

    for metrics_path in discover_metrics_paths(
        runs_dir,
        excluded_dirs=excluded_dirs,
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
            "No non-tuning neural training metrics found to plot."
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

        best_train = max(
            value
            for _, value in train_points
        )
        title = (
            f"{item.algorithm}: {item.label}\n"
            f"best train {best_train:.2f}"
        )
        axis.set_title(
            title,
            fontsize=13,
            pad=8,
        )
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
        "Neural training metrics across non-tuning runs: "
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
    runs_dir = args.runs_dir
    excluded_dirs = set(
        args.exclude_dir
    )
    series = collect_training_run_series(
        runs_dir,
        metric=args.metric,
        excluded_dirs=excluded_dirs,
    )

    plot_training_run_summary(
        series,
        metric=args.metric,
        output_path=args.output,
        rolling_window=args.rolling_window,
    )

    print(
        f"Saved {len(series)} training run curve(s) to {args.output}"
    )


if __name__ == "__main__":
    main()
