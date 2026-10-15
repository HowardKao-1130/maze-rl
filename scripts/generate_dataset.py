import argparse
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for import_root in (REPO_ROOT, SRC_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from maze_rl.maze.dataset import (
    generate_mazes,
    generate_tasks_for_layouts,
    generate_split_seeds,
    save_dataset,
)
from maze_rl.maze.stratification import (
    DEFAULT_BUCKET_BUDGETS,
    DEFAULT_DIFFICULTY_BOUNDARIES,
    DifficultyBoundaries,
    generate_layouts,
    generate_stratified_tasks_for_layouts,
)
from scripts.experiment_naming import (
    DEFAULT_DATA_OUTPUT_ROOT,
    experiment_dataset_name,
)


def task_pairs_by_layout(
    layout_indices,
    starts,
    goals,
):
    pairs_by_layout = {}

    for layout_index, start, goal in zip(
        layout_indices,
        starts,
        goals,
    ):
        pairs_by_layout.setdefault(
            int(layout_index),
            set(),
        ).add(
            (
                tuple(int(x) for x in start),
                tuple(int(x) for x in goal),
            )
        )

    return pairs_by_layout


def _simple_progress_callback(label: str):
    def update(completed: int, total: int) -> None:
        _write_progress_line(f"{label}: {completed}/{total}")

    return update


def _write_progress_line(line: str) -> None:
    if sys.stdout.isatty():
        print(f"\033[2K\r{line}", end="", flush=True)
    else:
        print(line, flush=True)


def _stratified_progress_callback(label: str):
    if "validation" in label:
        short_label = "Stratified validation"
    else:
        short_label = "Stratified training"

    def update(event: dict[str, object]) -> None:
        counts = event["bucket_counts"]
        quotas = event["bucket_quotas"]
        status = " | ".join(
            f"{bucket[0].upper()} {counts[bucket]}/{quotas[bucket]}"
            for bucket in ("short", "medium", "long")
        )
        phase = event.get("phase")
        if phase == "indexing":
            line = (
                f"{short_label} | index "
                f"{event['layouts_completed']}/{event['layout_count']} | "
                f"{status}"
            )
        else:
            line = f"{short_label} | {status}"
        _write_progress_line(line)

    return update


def _finish_progress() -> None:
    if sys.stdout.isatty():
        print()


def _confirm_overwrite(output_paths: list[Path]) -> None:
    existing_paths = [path for path in output_paths if path.exists()]
    if not existing_paths:
        return

    print("Existing dataset files would be overwritten:")
    for path in existing_paths:
        print(f"  {path}")
    try:
        response = input("Overwrite these files? [y/N]: ")
    except EOFError:
        response = ""
    if response.strip().lower() not in {"y", "yes"}:
        raise SystemExit("Dataset generation cancelled.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate maze datasets."
    )

    parser.add_argument(
        "--train",
        "--train-mazes",
        dest="train_mazes",
        type=int,
        default=1000,
        help="Number of training layouts to generate.",
    )

    parser.add_argument(
        "--validation",
        "--validation-mazes",
        dest="validation_mazes",
        type=int,
        default=200,
        help="Number of validation layouts to generate.",
    )

    parser.add_argument(
        "--test",
        "--test-mazes",
        dest="test_mazes",
        type=int,
        default=200,
        help="Number of test layouts to generate.",
    )

    parser.add_argument(
        "--height",
        type=int,
        default=11,
    )

    parser.add_argument(
        "--width",
        type=int,
        default=11,
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
            "Directory for generated split files. Overrides "
            "--experiment-name and --output-root when provided."
        ),
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_DATA_OUTPUT_ROOT,
        help=(
            "Root directory for automatically named experiment "
            "datasets."
        ),
    )

    parser.add_argument(
        "--experiment-name",
        default=None,
        help=(
            "Optional prefix for an automatically named output "
            "directory, for example "
            "generalization_followup_2000x200x200_t5_seed42. "
            "Defaults to dataset."
        ),
    )

    parser.add_argument(
        "--wall-probability",
        type=float,
        default=0.25,
    )

    parser.add_argument(
        "--min-path-length",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--tasks-per-maze",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--min-tasks-per-maze",
        type=int,
        default=3,
        help="Minimum tasks a layout may contribute after repair.",
    )

    parser.add_argument(
        "--max-tasks-per-maze",
        type=int,
        default=7,
        help="Maximum tasks a layout may contribute after repair.",
    )

    parser.add_argument(
        "--stratify",
        action="store_true",
        help=(
            "Select training tasks using difficulty buckets and "
            "directed weighted Jaccard deduplication."
        ),
    )

    parser.add_argument(
        "--dedupe-threshold",
        type=float,
        default=0.75,
        help="Maximum directed weighted Jaccard similarity threshold.",
    )

    parser.add_argument(
        "--short-budget",
        type=float,
        default=DEFAULT_BUCKET_BUDGETS["short"],
        help="Fraction of stratified tasks allocated to short paths.",
    )

    parser.add_argument(
        "--medium-budget",
        type=float,
        default=DEFAULT_BUCKET_BUDGETS["medium"],
        help="Fraction of stratified tasks allocated to medium paths.",
    )

    parser.add_argument(
        "--long-budget",
        type=float,
        default=DEFAULT_BUCKET_BUDGETS["long"],
        help="Fraction of stratified tasks allocated to long paths.",
    )

    parser.add_argument(
        "--short-max-path-length",
        type=int,
        default=int(DEFAULT_DIFFICULTY_BOUNDARIES.short_max),
        help="Maximum shortest-path length in the short bucket.",
    )

    parser.add_argument(
        "--medium-max-path-length",
        type=int,
        default=int(DEFAULT_DIFFICULTY_BOUNDARIES.medium_max),
        help="Maximum shortest-path length in the medium bucket.",
    )

    parser.add_argument(
        "--allocation-attempts",
        type=int,
        default=8,
        help="Number of seeded global allocation attempts.",
    )

    parser.add_argument(
        "--local-search-budget",
        type=int,
        default=2_000,
        help="Candidate-extension attempts allowed per layout.",
    )

    parser.add_argument(
        "--global-repair-budget",
        type=int,
        default=None,
        help="Global repair attempts; defaults to five per layout.",
    )

    args = parser.parse_args()

    if not 0 < args.dedupe_threshold <= 1:
        parser.error("--dedupe-threshold must be in (0, 1].")
    if args.stratify and not 0 < args.min_tasks_per_maze <= args.tasks_per_maze <= args.max_tasks_per_maze:
        parser.error(
            "Require --min-tasks-per-maze <= --tasks-per-maze <= "
            "--max-tasks-per-maze."
        )
    if args.allocation_attempts <= 0 or args.local_search_budget <= 0:
        parser.error("Allocation and search budgets must be positive.")
    if args.global_repair_budget is not None and args.global_repair_budget <= 0:
        parser.error("--global-repair-budget must be positive.")
    if (
        args.short_max_path_length < 0
        or args.medium_max_path_length <= args.short_max_path_length
    ):
        parser.error(
            "Require 0 <= --short-max-path-length < "
            "--medium-max-path-length."
        )
    args.bucket_budgets = {
        "short": args.short_budget,
        "medium": args.medium_budget,
        "long": args.long_budget,
    }
    if any(value < 0 for value in args.bucket_budgets.values()):
        parser.error("Bucket budgets must be non-negative.")
    if not np.isclose(sum(args.bucket_budgets.values()), 1.0):
        parser.error("Bucket budgets must sum to 1.0.")

    if args.output_dir is None:
        args.output_dir = args.output_root / experiment_dataset_name(
            args.experiment_name,
            train_mazes=args.train_mazes,
            validation_mazes=args.validation_mazes,
            test_mazes=args.test_mazes,
            tasks_per_maze=args.tasks_per_maze,
            seed=args.seed,
            stratify=args.stratify,
            dedupe_threshold=args.dedupe_threshold,
            bucket_budgets=args.bucket_budgets,
        )

    return args


def main() -> None:
    args = parse_args()
    train_path = args.output_dir / "train.npz"
    validation_path = args.output_dir / "validation.npz"
    validation_stratified_path = (
        args.output_dir / "validation_stratified.npz"
    )
    same_layout_path = (
        args.output_dir / "same_layout_new_goals.npz"
    )
    test_path = args.output_dir / "test.npz"
    metadata_path = args.output_dir / "metadata.json"
    output_paths = [
        train_path,
        validation_path,
        same_layout_path,
        test_path,
        metadata_path,
    ]
    if args.stratify:
        output_paths.append(validation_stratified_path)
    _confirm_overwrite(output_paths)

    if args.validation_mazes > args.train_mazes:
        raise SystemExit(
            "--validation-mazes cannot exceed --train-mazes because "
            "same_layout_new_goals.npz uses validation-sized held-out "
            "tasks on unique training layouts."
        )

    print(f"Writing datasets to {args.output_dir}")
    print(f"  train:        {train_path}")
    print(f"  validation:   {validation_path}")
    if args.stratify:
        print(
            "  val-stratified: "
            f"{validation_stratified_path}"
        )
    print(f"  same-layout:  {same_layout_path}")
    print(f"  test:         {test_path}")
    print(f"  metadata:     {metadata_path}")

    (
        train_seeds,
        validation_seeds,
        test_seeds,
    ) = generate_split_seeds(
        train_size=args.train_mazes,
        validation_size=args.validation_mazes,
        test_size=args.test_mazes,
        master_seed=args.seed,
    )

    if args.stratify:
        train_layouts = generate_layouts(
            seeds=train_seeds,
            height=args.height,
            width=args.width,
            wall_probability=args.wall_probability,
            min_path_length=args.min_path_length,
            minimum_eligible_tasks=args.min_tasks_per_maze,
            progress_callback=_simple_progress_callback(
                "Generating training mazes"
            ),
        )
        train_layout_indices = None
        train_starts = None
        train_goals = None
    else:
        (
            train_layouts,
            train_layout_indices,
            train_starts,
            train_goals,
        ) = generate_mazes(
            seeds=train_seeds,
            height=args.height,
            width=args.width,
            wall_probability=args.wall_probability,
            min_path_length=args.min_path_length,
            tasks_per_maze=args.tasks_per_maze,
            progress_callback=_simple_progress_callback(
                "Generating training mazes"
            ),
        )
    _finish_progress()

    stratification_metadata = None
    if args.stratify:
        (
            train_layout_indices,
            train_starts,
            train_goals,
            stratification_metadata,
        ) = generate_stratified_tasks_for_layouts(
            train_layouts,
            train_seeds,
            tasks_per_maze=args.tasks_per_maze,
            min_path_length=args.min_path_length,
            dedupe_threshold=args.dedupe_threshold,
            bucket_budgets=args.bucket_budgets,
            random_seed=args.seed,
            difficulty_boundaries=DifficultyBoundaries(
                short_max=args.short_max_path_length,
                medium_max=args.medium_max_path_length,
            ),
            allocation_attempts=args.allocation_attempts,
            min_tasks_per_maze=args.min_tasks_per_maze,
            max_tasks_per_maze=args.max_tasks_per_maze,
            local_search_budget=args.local_search_budget,
            global_repair_budget=args.global_repair_budget,
            progress_callback=_stratified_progress_callback(
                "Selecting stratified training tasks"
            ),
        )
        _finish_progress()

    (
        validation_layouts,
        validation_layout_indices,
        validation_starts,
        validation_goals,
    ) = generate_mazes(
        seeds=validation_seeds,
        height=args.height,
        width=args.width,
        wall_probability=args.wall_probability,
        min_path_length=args.min_path_length,
        tasks_per_maze=args.tasks_per_maze,
        progress_callback=_simple_progress_callback(
            "Generating validation mazes"
        ),
    )
    _finish_progress()

    validation_stratified_metadata = None
    validation_stratified_indices = None
    validation_stratified_starts = None
    validation_stratified_goals = None
    if args.stratify:
        train_boundaries = DifficultyBoundaries(
            **stratification_metadata["difficulty_boundaries"]
        )
        (
            validation_stratified_indices,
            validation_stratified_starts,
            validation_stratified_goals,
            validation_stratified_metadata,
        ) = generate_stratified_tasks_for_layouts(
            validation_layouts,
            validation_seeds,
            tasks_per_maze=args.tasks_per_maze,
            min_path_length=args.min_path_length,
            dedupe_threshold=args.dedupe_threshold,
            bucket_budgets=args.bucket_budgets,
            random_seed=args.seed + 1,
            allocation_attempts=args.allocation_attempts,
            min_tasks_per_maze=args.min_tasks_per_maze,
            max_tasks_per_maze=args.max_tasks_per_maze,
            local_search_budget=args.local_search_budget,
            global_repair_budget=args.global_repair_budget,
            difficulty_boundaries=train_boundaries,
            progress_callback=_stratified_progress_callback(
                "Selecting stratified validation tasks"
            ),
        )
        _finish_progress()

    same_layout_maze_count = args.validation_mazes
    same_layout_layouts = train_layouts[
        :same_layout_maze_count
    ]
    same_layout_seeds = train_seeds[
        :same_layout_maze_count
    ]
    (
        same_layout_indices,
        same_layout_starts,
        same_layout_goals,
    ) = generate_tasks_for_layouts(
        layouts=same_layout_layouts,
        seeds=same_layout_seeds,
        tasks_per_maze=args.tasks_per_maze,
        min_path_length=args.min_path_length,
        excluded_tasks_by_layout=task_pairs_by_layout(
            train_layout_indices,
            train_starts,
            train_goals,
        ),
        progress_callback=_simple_progress_callback(
            "Generating same-layout tasks"
        ),
    )
    _finish_progress()

    (
        test_layouts,
        test_layout_indices,
        test_starts,
        test_goals,
    ) = generate_mazes(
        seeds=test_seeds,
        height=args.height,
        width=args.width,
        wall_probability=args.wall_probability,
        min_path_length=args.min_path_length,
        tasks_per_maze=args.tasks_per_maze,
        progress_callback=_simple_progress_callback(
            "Generating test mazes"
        ),
    )
    _finish_progress()

    save_dataset(
        train_path,
        train_layouts,
        train_seeds,
        train_layout_indices,
        train_starts,
        train_goals,
    )

    save_dataset(
        validation_path,
        validation_layouts,
        validation_seeds,
        validation_layout_indices,
        validation_starts,
        validation_goals,
    )

    if args.stratify:
        save_dataset(
            validation_stratified_path,
            validation_layouts,
            validation_seeds,
            validation_stratified_indices,
            validation_stratified_starts,
            validation_stratified_goals,
        )

    save_dataset(
        same_layout_path,
        same_layout_layouts,
        same_layout_seeds,
        same_layout_indices,
        same_layout_starts,
        same_layout_goals,
    )

    save_dataset(
        test_path,
        test_layouts,
        test_seeds,
        test_layout_indices,
        test_starts,
        test_goals,
    )

    metadata = {
        "generator": "random_obstacle_grid",
        "height": args.height,
        "width": args.width,
        "wall_probability": args.wall_probability,
        "min_path_length": args.min_path_length,
        "master_seed": args.seed,
        "train_mazes": args.train_mazes,
        "validation_mazes": args.validation_mazes,
        "test_mazes": args.test_mazes,
        "tasks_per_maze": args.tasks_per_maze,
        "min_tasks_per_maze": args.min_tasks_per_maze,
        "max_tasks_per_maze": args.max_tasks_per_maze,
        "local_search_budget": args.local_search_budget,
        "global_repair_budget": args.global_repair_budget,
        "stratify": args.stratify,
        "dedupe_threshold": args.dedupe_threshold,
        "bucket_budgets": args.bucket_budgets,
        "difficulty_boundaries": (
            stratification_metadata["difficulty_boundaries"]
            if stratification_metadata is not None
            else None
        ),
        "same_layout_new_goals": {
            "source": "train_layouts",
            "layout_count": same_layout_maze_count,
            "tasks_per_maze": args.tasks_per_maze,
        },
    }
    if stratification_metadata is not None:
        metadata["stratification"] = {
            "train": stratification_metadata,
            "validation_stratified": validation_stratified_metadata,
        }

    with metadata_path.open("w") as file:
        json.dump(
            metadata,
            file,
            indent=2,
        )

    print()
    print("Dataset generated successfully.")
    print(f"Output directory:   {args.output_dir}")
    print("Stored files:")
    print(f"  train:        {train_path}")
    print(f"  validation:   {validation_path}")
    if args.stratify:
        print(
            "  val-stratified: "
            f"{validation_stratified_path}"
        )
    print(f"  same-layout:  {same_layout_path}")
    print(f"  test:         {test_path}")
    print(f"  metadata:     {metadata_path}")
    print(
        f"Train layouts:      {train_layouts.shape}"
    )
    print(
        f"Validation layouts: {validation_layouts.shape}"
    )
    print(
        f"Test layouts:       {test_layouts.shape}"
    )
    print(f"Train tasks:        {len(train_starts)}")
    print(
        f"Validation tasks:   {len(validation_starts)}"
    )
    print(
        "Same-layout tasks:  "
        f"{len(same_layout_starts)}"
    )
    print(f"Test tasks:         {len(test_starts)}")


if __name__ == "__main__":
    main()
