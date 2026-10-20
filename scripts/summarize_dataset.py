"""Print aggregate task and stratification statistics for a dataset bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for import_root in (REPO_ROOT, REPO_ROOT / "src"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from maze_rl.maze.pathfinding import shortest_path_length


BUCKETS = ("short", "medium", "long")
SPLITS = (
    "train",
    "validation",
    "same_layout_new_goals",
    "validation_natural",
    "test",
    "test_natural",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize maze dataset task and bucket properties."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Dataset bundle directory or one split .npz path.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path for the machine-readable summary.",
    )
    return parser.parse_args()


def _bundle_dir(dataset: Path) -> Path:
    return dataset if dataset.is_dir() else dataset.parent


def _boundaries(metadata: dict) -> tuple[float, float] | None:
    stratification = metadata.get("stratification", {})
    if isinstance(stratification, dict):
        train_metadata = stratification.get("train", stratification)
        if isinstance(train_metadata, dict):
            values = train_metadata.get("difficulty_boundaries")
            if isinstance(values, dict):
                return float(values["short_max"]), float(values["medium_max"])
    return None


def _bucket(distance: int, boundaries: tuple[float, float] | None) -> str:
    if boundaries is None:
        return "unbucketed"
    if distance <= boundaries[0]:
        return "short"
    if distance <= boundaries[1]:
        return "medium"
    return "long"


def summarize_split(
    path: Path,
    boundaries: tuple[float, float] | None,
) -> dict[str, object]:
    with np.load(path) as data:
        layouts = data["layouts"] if "layouts" in data else data["mazes"]
        layout_indices = data["layout_indices"]
        starts = data["starts"]
        goals = data["goals"]

    counts = {bucket: 0 for bucket in BUCKETS}
    path_lengths = {bucket: [] for bucket in BUCKETS}
    all_lengths = []
    tasks_by_layout: dict[int, int] = {}

    for layout_index, start, goal in zip(layout_indices, starts, goals):
        layout_index = int(layout_index)
        start_position = tuple(int(value) for value in start)
        goal_position = tuple(int(value) for value in goal)
        distance = shortest_path_length(
            layouts[layout_index],
            start_position,
            goal_position,
        )
        if distance is None:
            raise ValueError(
                f"Unreachable task in {path}: layout {layout_index}, "
                f"{start_position}->{goal_position}"
            )
        distance = int(distance)
        all_lengths.append(distance)
        tasks_by_layout[layout_index] = tasks_by_layout.get(layout_index, 0) + 1
        bucket = _bucket(distance, boundaries)
        if bucket in counts:
            counts[bucket] += 1
            path_lengths[bucket].append(distance)

    summary: dict[str, object] = {
        "path": str(path),
        "layout_count": int(len(layouts)),
        "task_count": int(len(all_lengths)),
        "counts": counts,
        "mean_tasks_per_layout": (
            float(len(all_lengths) / len(layouts)) if len(layouts) else 0.0
        ),
        "mean_optimal_path_length": (
            float(np.mean(all_lengths)) if all_lengths else None
        ),
        "mean_optimal_path_length_by_bucket": {
            bucket: (
                float(np.mean(lengths)) if lengths else None
            )
            for bucket, lengths in path_lengths.items()
        },
        "contribution_min": min(tasks_by_layout.values(), default=0),
        "contribution_max": max(tasks_by_layout.values(), default=0),
    }
    return summary


def _print_table(
    headers: list[str],
    rows: list[list[str]],
    numeric_columns: set[int],
) -> None:
    widths = [
        max(
            len(header),
            *(len(row[index]) for row in rows),
        )
        for index, header in enumerate(headers)
    ]

    def format_row(row: list[str]) -> str:
        cells = []
        for index, value in enumerate(row):
            alignment = ">" if index in numeric_columns else "<"
            cells.append(f"{value:{alignment}{widths[index]}}")
        return "  ".join(cells)

    print(format_row(headers))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print(format_row(row))


def main() -> None:
    args = parse_args()
    bundle_dir = _bundle_dir(args.dataset)
    metadata_path = bundle_dir / "metadata.json"
    metadata = {}
    if metadata_path.exists():
        with metadata_path.open() as file:
            metadata = json.load(file)
    boundaries = _boundaries(metadata)

    summaries = {}
    for split in SPLITS:
        path = bundle_dir / f"{split}.npz"
        if path.exists():
            summaries[split] = summarize_split(path, boundaries)

    report = {
        "difficulty_boundaries": (
            None
            if boundaries is None
            else {
                "short_max": boundaries[0],
                "medium_max": boundaries[1],
            }
        ),
        "splits": summaries,
    }
    if isinstance(metadata.get("stratification"), dict):
        report["generation_diagnostics"] = metadata["stratification"]

    split_rows = []
    for split, summary in summaries.items():
        counts = summary["counts"]
        split_rows.append(
            [
                split,
                str(summary["task_count"]),
                str(counts["short"]),
                str(counts["medium"]),
                str(counts["long"]),
                f"{summary['mean_tasks_per_layout']:.2f}",
                f"{summary['mean_optimal_path_length'] or 0:.2f}",
            ]
        )
    _print_table(
        [
            "Split",
            "Tasks",
            "Short",
            "Medium",
            "Long",
            "Mean tasks/layout",
            "Mean path",
        ],
        split_rows,
        {1, 2, 3, 4, 5, 6},
    )

    bucket_rows = []
    for split, summary in summaries.items():
        for bucket in BUCKETS:
            mean_length = summary[
                "mean_optimal_path_length_by_bucket"
            ][bucket]
            bucket_rows.append(
                [
                    split,
                    bucket,
                    str(summary["counts"][bucket]),
                    f"{0 if mean_length is None else mean_length:.2f}",
                ]
            )
    print()
    _print_table(
        ["Split", "Bucket", "Tasks", "Mean optimal path length"],
        bucket_rows,
        {2, 3},
    )

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w") as file:
            json.dump(report, file, indent=2)


if __name__ == "__main__":
    main()
