from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(REPO_ROOT),
    )

from scripts.experiment_naming import (
    DEFAULT_DATA_OUTPUT_ROOT,
    DEFAULT_DNN_TUNING_OUTPUT_ROOT,
)
from scripts.train import NEURAL_ALGORITHMS


TRAINING_LOOP_PARAMETERS = {
    "task_batch_size",
    "rollout_group_size",
}


def read_training_summary(
    path: Path,
) -> dict[str, Any]:
    with path.open() as file:
        return json.load(file)


def training_summary_completed(
    training_summary: dict[str, Any],
) -> bool:
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


def training_summary_completion_status(
    path: Path,
) -> bool | None:
    if not path.exists():
        return None

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

SUMMARY_SPLITS = [
    "validation",
    "same_layout",
]
SUMMARY_SPLIT_LABELS = {
    "train_greedy": "train-greedy",
    "validation": "val",
    "same_layout": "same-val",
}
SUMMARY_SPLIT_ALIASES = {
    "train": "train_greedy",
    "train_greedy": "train_greedy",
    "greedy_train": "train_greedy",
    "train_eval": "train_greedy",
    "val": "validation",
    "validation": "validation",
    "same_layout": "same_layout",
    "val_with_same_layout": "same_layout",
    "same-val": "same_layout",
    "same_val": "same_layout",
}
SUMMARY_MONTAGE_X_LIMIT = (
    0,
    205,
)
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
        "label": "same-val",
        "color": "#ff7f0e",
        "linestyle": "-",
    },
    "train_greedy": {
        "label": "train-greedy",
        "color": "#d62728",
        "linestyle": "--",
    },
    "train": {
        "label": "train",
        "color": "#2ca02c",
        "linestyle": "-",
    },
}
SUMMARY_SCORE_LABELS = {
    "train_greedy": "tg",
    "validation": "v",
    "same_layout": "vs",
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


def metric_series_by_split(
    metrics_path: Path,
    metric: str,
) -> dict[str, list[tuple[int, float]]]:
    series: dict[str, list[tuple[int, float]]] = {
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

            if split not in series:
                continue

            raw_epoch = row.get(
                "dataset_epoch",
                row.get("epoch"),
            )

            if raw_epoch in {
                None,
                "",
            }:
                continue

            series[split].append(
                (
                    int(raw_epoch),
                    float(raw_value),
                )
            )

    return {
        split: sorted(points)
        for split, points in series.items()
        if points
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


def training_metric_row_matches(
    row: dict[str, str],
    *,
    metric: str,
    candidate: str,
) -> bool:
    if (
        metric
        == "average_successful_path_efficiency"
        and candidate == "path_efficiency"
    ):
        return parse_summary_float(
            row.get("success")
        ) == 1.0

    return True


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

            if not training_metric_row_matches(
                row,
                metric=metric,
                candidate=candidate,
            ):
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


def value_at_epoch(
    series: list[tuple[int, float]],
    epoch: int,
) -> float | None:
    for point_epoch, value in series:
        if point_epoch == epoch:
            return value

    return None


def best_validation_epoch(
    series_by_split: dict[str, list[tuple[int, float]]],
) -> int | None:
    validation_series = series_by_split.get(
        "validation"
    )

    if not validation_series:
        return None

    return min(
        validation_series,
        key=lambda point: (
            -point[1],
            point[0],
        ),
    )[0]


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

            if not training_metric_row_matches(
                row,
                metric=metric,
                candidate=candidate,
            ):
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


def training_metric_series(
    metrics_path: Path,
    *,
    metric: str,
) -> list[tuple[int, float]]:
    if not metrics_path.exists():
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

            if not training_metric_row_matches(
                row,
                metric=metric,
                candidate=candidate,
            ):
                continue

            epoch = int(
                row.get("epoch")
                or row.get("episode")
                or 0
            )
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
    validation_series = validation_metric_series(
        artifact["metrics_path"],
        metric=metric,
    )
    selected_epoch = best_validation_epoch(
        validation_series
    )
    labels = []

    if selected_epoch is not None:
        for split in SUMMARY_SPLITS:
            value = value_at_epoch(
                validation_series.get(
                    split,
                    [],
                ),
                selected_epoch,
            )

            if value is None:
                continue

            labels.append(
                f"{SUMMARY_SCORE_LABELS[split]} "
                f"{value:.2f}"
            )
    else:
        split_values = best_metric_by_split(
            artifact["metrics_path"],
            metric,
        )

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
        train_score = None

        if selected_epoch is not None:
            train_score = value_at_epoch(
                aligned_training_metric_series(
                    artifact["training_metrics_path"],
                    metric=metric,
                    aligned_epochs={
                        selected_epoch,
                    },
                ),
                selected_epoch,
            )

        if train_score is None:
            aligned_epochs = validation_metric_epochs(
                validation_series
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


def best_artifacts_by_algorithm(
    artifacts: list[dict[str, Any]],
    *,
    metric: str,
) -> dict[str, dict[str, Any]]:
    best_artifacts = {}

    for algorithm in ordered_algorithms(
        {
            artifact["algorithm"]
            for artifact in artifacts
        }
    ):
        scored_artifacts = []

        for artifact in artifacts:
            if artifact["algorithm"] != algorithm:
                continue

            score = artifact_summary_score(
                artifact,
                metric=metric,
            )

            if score is None:
                continue

            scored_artifacts.append(
                (
                    artifact,
                    score,
                )
            )

        if not scored_artifacts:
            continue

        best_artifacts[algorithm] = min(
            scored_artifacts,
            key=lambda item: (
                -item[1],
                item[0]["trial"],
            ),
        )[0]

    return best_artifacts


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


def artifact_completion_status(
    artifact: dict[str, Any],
) -> bool | None:
    training_summary_path = artifact_training_summary_path(
        artifact
    )

    if artifact_has_stale_completion_summary(
        artifact,
        training_summary_path=training_summary_path,
    ):
        return False

    return training_summary_completion_status(
        training_summary_path
    )


def artifact_incomplete_label(
    artifact: dict[str, Any],
) -> str:
    return "INCOMPLETE" if artifact_is_incomplete(artifact) else ""


def artifact_is_incomplete(
    artifact: dict[str, Any],
) -> bool:
    return artifact_completion_status(artifact) is not True


def artifact_activity_paths(
    artifact: dict[str, Any],
) -> list[Path]:
    run_dir = artifact["metrics_path"].parent
    paths = [
        artifact["metrics_path"],
        artifact["training_metrics_path"],
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


def artifact_completion_marker_mtime_ns(
    artifact: dict[str, Any],
    *,
    training_summary_path: Path,
) -> int | None:
    run_dir = artifact["metrics_path"].parent
    marker_paths = [
        training_summary_path,
        run_dir / "validation_evaluation.csv",
    ]
    marker_mtimes = [
        path.stat().st_mtime_ns
        for path in marker_paths
        if path.exists()
    ]

    if not marker_mtimes:
        return None

    return max(marker_mtimes)


def artifact_has_stale_completion_summary(
    artifact: dict[str, Any],
    *,
    training_summary_path: Path,
) -> bool:
    marker_mtime_ns = artifact_completion_marker_mtime_ns(
        artifact,
        training_summary_path=training_summary_path,
    )

    if marker_mtime_ns is None:
        return False

    return any(
        path.stat().st_mtime_ns > marker_mtime_ns
        for path in artifact_activity_paths(artifact)
    )


def algorithm_is_incomplete(
    artifacts: list[dict[str, Any]],
    algorithm: str,
) -> bool:
    if any(
        artifact["algorithm"] == algorithm
        and artifact_is_incomplete(artifact)
        for artifact in artifacts
    ):
        return True

    trial_numbers = {
        artifact["trial"]
        for artifact in artifacts
    }
    algorithm_trials = {
        artifact["trial"]
        for artifact in artifacts
        if artifact["algorithm"] == algorithm
    }

    return bool(
        trial_numbers
        and algorithm_trials != trial_numbers
    )


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
    del algorithm
    names = {
        name
        for trial in trials
        for name in trial["hyperparameters"]
    }
    return sorted(names)


def parallel_coordinate_uses_log_scale(
    *,
    algorithm: str,
    name: str,
    values: list[float],
) -> bool:
    del algorithm, name

    if not values or any(
        value <= 0
        for value in values
    ):
        return False

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


def parse_summary_positive_int(
    value: Any,
) -> int | None:
    if isinstance(
        value,
        bool,
    ):
        return None

    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None

    if parsed <= 0:
        return None

    return parsed


def summary_axis_requested_budget(
    artifact: dict[str, Any],
) -> int | None:
    training_summary_path = artifact_training_summary_path(
        artifact
    )

    if not training_summary_path.exists():
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
    ):
        return None

    for name in (
        "dataset_epochs_requested",
        "task_selection_passes_requested",
    ):
        budget = parse_summary_positive_int(
            training_summary.get(name)
        )

        if budget is not None:
            return budget

    return None


def plotted_summary_max_epoch(
    axis,
) -> int | None:
    max_epoch = None

    for line in axis.lines:
        for raw_epoch in line.get_xdata():
            try:
                epoch = float(raw_epoch)
            except (TypeError, ValueError):
                continue

            if not math.isfinite(epoch):
                continue

            max_epoch = (
                epoch
                if max_epoch is None
                else max(max_epoch, epoch)
            )

    if max_epoch is None:
        return None

    return max(
        0,
        math.ceil(max_epoch),
    )


def summary_axis_ticks(
    max_epoch: int,
) -> list[int]:
    if max_epoch <= 6:
        return list(range(max_epoch + 1))

    rough_step = max_epoch / 4
    magnitude = 10 ** math.floor(
        math.log10(rough_step)
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


def set_summary_axis_x_scale(
    axis,
    artifact: dict[str, Any],
) -> None:
    requested_budget = summary_axis_requested_budget(
        artifact
    )
    observed_max_epoch = plotted_summary_max_epoch(
        axis
    )

    if requested_budget is None and observed_max_epoch is None:
        return

    max_epoch = (
        requested_budget
        if requested_budget is not None
        else observed_max_epoch
    )

    axis.set_xlim(
        0,
        max_epoch,
    )
    axis.set_xticks(
        summary_axis_ticks(max_epoch)
    )


def set_summary_axis_y_limits(
    axis,
    *,
    metric: str,
) -> None:
    if metric not in SUMMARY_BOUNDED_METRICS:
        return

    axis.set_ylim(
        0.0,
        1.0,
    )


def plot_all_agent_training_progress(
    artifacts: list[dict[str, Any]],
    *,
    metric: str,
    output_path: Path,
) -> int:
    series_by_artifact = []

    for artifact in sorted(
        artifacts,
        key=lambda item: (
            item["algorithm"],
            item["trial"],
        ),
    ):
        series = training_metric_series(
            artifact["training_metrics_path"],
            metric=metric,
        )

        if not series:
            continue

        series_by_artifact.append(
            (
                artifact,
                series,
            )
        )

    if not series_by_artifact:
        return 0

    algorithms = ordered_algorithms(
        {
            artifact["algorithm"]
            for artifact, _ in series_by_artifact
        }
    )
    artifacts_by_algorithm = {
        algorithm: [
            (
                artifact,
                series,
            )
            for artifact, series in series_by_artifact
            if artifact["algorithm"] == algorithm
        ]
        for algorithm in algorithms
    }
    best_trials_by_algorithm = {
        algorithm: best_performing_trials(
            [
                artifact
                for artifact, _ in algorithm_artifacts
            ],
            metric=metric,
        )
        for algorithm, algorithm_artifacts in artifacts_by_algorithm.items()
    }

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metric_label = SUMMARY_METRIC_LABELS.get(
        metric,
        metric,
    )
    column_count = min(
        2,
        len(algorithms),
    )
    row_count = math.ceil(
        len(algorithms) / column_count
    )
    figure, axes = plt.subplots(
        row_count,
        column_count,
        figsize=(
            8.5 * column_count,
            4.7 * row_count,
        ),
        squeeze=False,
    )

    for axis in axes.reshape(-1):
        axis.axis("off")

    for axis, algorithm in zip(
        axes.reshape(-1),
        algorithms,
    ):
        axis.axis("on")
        algorithm_artifacts = artifacts_by_algorithm[
            algorithm
        ]
        best_trials = best_trials_by_algorithm[
            algorithm
        ]
        best_label_used = False
        trial_label_used = False

        for artifact, series in algorithm_artifacts:
            is_best = artifact["trial"] in best_trials
            epochs = [
                epoch
                for epoch, _ in series
            ]
            values = [
                value
                for _, value in series
            ]
            label = None

            if is_best and not best_label_used:
                label = "best validation trial"
                best_label_used = True
            elif (
                not is_best
                and not trial_label_used
            ):
                label = "trial"
                trial_label_used = True

            axis.plot(
                epochs,
                values,
                color=(
                    "#d27d00"
                    if is_best
                    else "#9aa7b7"
                ),
                linewidth=2.8 if is_best else 1.2,
                alpha=0.95 if is_best else 0.42,
                label=label,
                zorder=3 if is_best else 2,
            )

        axis.set_title(
            (
                f"{algorithm} training progress "
                f"({len(algorithm_artifacts)} trial(s))"
            ),
            fontsize=13,
        )
        if algorithm_is_incomplete(
            artifacts,
            algorithm,
        ):
            draw_incomplete_label(axis)

        axis.set_xlabel("Dataset epoch")
        axis.set_ylabel(metric_label)
        axis.grid(
            True,
            alpha=0.22,
            linewidth=0.8,
        )
        set_summary_axis_y_limits(
            axis,
            metric=metric,
        )

        requested_budgets = [
            summary_axis_requested_budget(artifact)
            for artifact, _ in algorithm_artifacts
        ]
        requested_budgets = [
            budget
            for budget in requested_budgets
            if budget is not None
        ]
        observed_max_epoch = plotted_summary_max_epoch(
            axis
        )

        if (
            requested_budgets
            or observed_max_epoch is not None
        ):
            max_epoch = (
                max(requested_budgets)
                if requested_budgets
                else observed_max_epoch
            )
            axis.set_xlim(
                0,
                max_epoch,
            )
            axis.set_xticks(
                summary_axis_ticks(max_epoch)
            )

        if (
            best_label_used
            or trial_label_used
        ):
            axis.legend(
                loc="best",
                frameon=False,
            )

    figure.suptitle(
        f"DNN training progress by agent: {metric_label}",
        fontsize=16,
        y=0.995,
    )
    figure.tight_layout(
        rect=(
            0,
            0,
            1,
            0.97,
        )
    )
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    figure.savefig(
        output_path,
        dpi=170,
        facecolor="white",
        edgecolor="white",
        transparent=False,
    )
    rewrite_png_without_alpha(output_path)
    plt.close(figure)

    return len(series_by_artifact)


def plot_best_trial_training_summary(
    artifacts: list[dict[str, Any]],
    *,
    metric: str,
    output_path: Path,
) -> int:
    best_artifacts = best_artifacts_by_algorithm(
        artifacts,
        metric=metric,
    )
    records = []

    for algorithm in ordered_algorithms(
        set(best_artifacts)
    ):
        artifact = best_artifacts[algorithm]
        validation_series = validation_metric_series(
            artifact["metrics_path"],
            metric=metric,
        )
        training_series = training_metric_series(
            artifact["training_metrics_path"],
            metric=metric,
        )

        if not validation_series and not training_series:
            continue

        records.append(
            (
                algorithm,
                artifact,
                validation_series,
                training_series,
            )
        )

    if not records:
        return 0

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metric_label = SUMMARY_METRIC_LABELS.get(
        metric,
        metric,
    )
    column_count = min(
        3,
        len(records),
    )
    row_count = math.ceil(
        len(records) / column_count
    )
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

    for axis, (
        algorithm,
        artifact,
        validation_series,
        training_series,
    ) in zip(
        axes.reshape(-1),
        records,
    ):
        axis.axis("on")

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

        score = artifact_summary_score(
            artifact,
            metric=metric,
        )
        title = algorithm
        score_text = artifact_title_score_text(
            artifact,
            metric=metric,
        )

        if score_text:
            title = f"{title}\n{score_text}"
        elif score is not None:
            title = f"{title}\nbest val {score:.2f}"

        axis.set_title(
            title,
            fontsize=13,
            pad=8,
        )
        if (
            artifact_is_incomplete(artifact)
            or algorithm_is_incomplete(
                artifacts,
                algorithm,
            )
        ):
            draw_incomplete_label(axis)

        axis.tick_params(
            labelsize=10,
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
        set_summary_axis_x_scale(
            axis,
            artifact,
        )
        draw_best_epoch_marker(
            axis,
            best_validation_epoch(validation_series),
        )
        axis.set_xlabel(
            "Dataset epoch",
            fontsize=11,
        )
        axis.set_ylabel(
            metric_label,
            fontsize=11,
        )
        legend_handles, legend_labels = (
            axis.get_legend_handles_labels()
        )

    figure.suptitle(
        f"DNN tuning best trial metrics: {metric_label}",
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

    return len(records)


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

    slot_count = (
        SUMMARY_MONTAGE_COLUMNS
        * SUMMARY_MONTAGE_ROWS
    )
    plot_count = len(plot_artifacts)
    page_count = math.ceil(
        plot_count / slot_count
    )

    for page_index in range(page_count):
        page_artifacts = plot_artifacts[
            page_index * slot_count : (
                page_index + 1
            )
            * slot_count
        ]
        page_output_path = (
            output_path
            if page_index == 0
            else output_path.with_name(
                f"{output_path.stem}_page_"
                f"{page_index + 1:02d}"
                f"{output_path.suffix}"
            )
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
            page_artifacts,
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
                title = f"{title}\n{score_text}"

            axis.set_title(
                title,
                fontsize=18,
                pad=8,
            )
            if artifact_is_incomplete(artifact):
                draw_incomplete_label(axis)

            axis.tick_params(
                labelsize=15,
            )
            axis.set_xlim(
                *SUMMARY_MONTAGE_X_LIMIT
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
            set_summary_axis_x_scale(
                axis,
                artifact,
            )
            draw_best_epoch_marker(
                axis,
                best_validation_epoch(
                    validation_metric_series(
                        artifact["metrics_path"],
                        metric=metric,
                    )
                ),
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

            column_index = index % SUMMARY_MONTAGE_COLUMNS

            axis.set_xlabel(
                "Dataset epoch",
                fontsize=16,
            )

            if column_index == 0:
                axis.set_ylabel(
                    metric_label,
                    fontsize=16,
                )

        for axis in axes.reshape(-1)[len(page_artifacts):]:
            axis.axis("on")
            axis.set_facecolor("#f7f7f7")
            axis.set_xticks([])
            axis.set_yticks([])
            axis.text(
                0.5,
                0.5,
                "INCOMPLETE",
                ha="center",
                va="center",
                transform=axis.transAxes,
                color="#c62828",
                fontsize=18,
                fontweight="bold",
            )
            for spine in axis.spines.values():
                spine.set_color("#dddddd")
                spine.set_linewidth(1.0)

        page_label = (
            ""
            if page_count == 1
            else f" (page {page_index + 1}/{page_count})"
        )
        figure.suptitle(
            (
                f"{algorithm} validation and aligned training "
                f"metrics across trials{page_label}"
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
            page_output_path,
            dpi=SUMMARY_MONTAGE_DPI,
            facecolor="white",
            edgecolor="white",
            transparent=False,
        )
        rewrite_png_without_alpha(page_output_path)
        plt.close(figure)

    return plot_count



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
        args.runs_dir = DEFAULT_DNN_TUNING_OUTPUT_ROOT

        if args.dataset is not None:
            args.runs_dir = (
                args.runs_dir
                / dataset_dir_for_reference(args.dataset).name
            )

    if args.output_dir is None:
        args.output_dir = args.runs_dir / "summary_plots"


def parse_args(
    argv: list[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize DNN tuning runs with validation heatmaps, "
            "all-agent training progress plots, per-agent validation "
            "plot grids, and parallel-coordinate hyperparameter plots."
        )
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help=(
            "Dataset bundle name under data/datasets or a dataset "
            "directory. Defaults --runs-dir to "
            "runs/dnn/tuning/<dataset>."
        ),
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=None,
        help=(
            "Tuning run directory containing trials. Defaults to "
            "runs/dnn/tuning, or runs/dnn/tuning/<dataset> when "
            "--dataset is provided."
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
        default="mean_path_efficiency",
        help=(
            "Validation metrics CSV column to visualize in the "
            "summary plots."
        ),
    )

    args = parser.parse_args(argv)
    resolve_summary_paths(args)
    return args


def summarize_tuning_runs(
    args: argparse.Namespace,
) -> None:
    artifacts = discover_validation_metric_artifacts(
        args.runs_dir
    )

    if not artifacts:
        raise RuntimeError(
            "No validation_metrics.csv artifacts found under "
            f"{args.runs_dir / 'trials'}."
        )

    cells = collect_heatmap_cells(
        artifacts,
        metric=args.metric,
    )
    heatmap_path = (
        args.output_dir
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

    training_progress_path = (
        args.output_dir
        / f"{args.metric}_training_progress_by_agent.png"
    )
    training_progress_count = (
        plot_all_agent_training_progress(
            artifacts,
            metric=args.metric,
            output_path=training_progress_path,
        )
    )

    if training_progress_count:
        print(
            "Saved all-agent training progress plot to "
            f"{training_progress_path} "
            f"({training_progress_count} trial(s))",
            flush=True,
        )
    else:
        print(
            "Skipped all-agent training progress plot; no "
            "metrics.csv files with matching training metrics "
            "were found.",
            flush=True,
        )

    best_trial_summary_path = (
        args.output_dir
        / f"{args.metric}_best_trial_training_history_summary.png"
    )
    best_trial_summary_count = (
        plot_best_trial_training_summary(
            artifacts,
            metric=args.metric,
            output_path=best_trial_summary_path,
        )
    )

    if best_trial_summary_count:
        print(
            "Saved best-trial training summary plot to "
            f"{best_trial_summary_path} "
            f"({best_trial_summary_count} agent(s))",
            flush=True,
        )
    else:
        print(
            "Skipped best-trial training summary plot; no "
            "plottable best trials were found.",
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
            args.output_dir
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
                f"({filled_trial_slots} trial plot(s))",
                flush=True,
            )
        else:
            print(
                "Skipped validation metrics trial grid for "
                f"{algorithm}; no validation_metrics.csv files "
                "were found.",
                flush=True,
            )

        parallel_path = (
            args.output_dir
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
    summarize_tuning_runs(parse_args())


if __name__ == "__main__":
    main()
