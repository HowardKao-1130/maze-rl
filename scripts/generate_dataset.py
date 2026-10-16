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
    format_failure_diagnostics,
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


def _write_progress_line(line: str, *, persist: bool = False) -> None:
    # Keep live progress on one physical line even when stdout is wrapped by
    # a terminal runner that does not report itself as a TTY.
    end = "\n" if persist else ""
    print(f"\033[2K\r{line}", end=end, flush=True)


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
            attempt = event.get("allocation_attempt")
            attempts = event.get("allocation_attempts")
            attempt_status = (
                f"attempt {attempt}/{attempts} | "
                if attempt is not None and attempts is not None
                else ""
            )
            if phase == "allocation":
                bucket = event.get("allocation_bucket")
                bucket_completed = event.get("allocation_bucket_completed")
                bucket_total = event.get("allocation_bucket_total")
                slots_completed = event.get("allocation_slots_completed")
                slots_total = event.get("allocation_slots_total")
                if all(
                    value is not None
                    for value in (
                        bucket,
                        bucket_completed,
                        bucket_total,
                        slots_completed,
                        slots_total,
                    )
                ):
                    phase_status = (
                        "allocating slots | "
                        f"{str(bucket)[0].upper()} "
                        f"{bucket_completed}/{bucket_total} | "
                        f"total {slots_completed}/{slots_total} | "
                    )
                else:
                    phase_status = "allocating slots | "
            elif phase == "selection":
                phase_status = (
                    "selecting tasks | "
                    f"layouts {event['layouts_completed']}/"
                    f"{event['layout_count']} | "
                )
            elif phase == "repair":
                phase_status = (
                    "repairing allocations | "
                    f"attempts {event['repair_attempts']}/"
                    f"{event['repair_budget']} | "
                )
            else:
                phase_status = f"{phase} | " if phase else ""
            reason = event.get("reason")
            reason_status = f" ({reason})" if reason else ""
            diagnostics = event.get("diagnostics")
            diagnostics_status = ""
            if isinstance(diagnostics, dict):
                diagnostics_status = (
                    "; " + format_failure_diagnostics(diagnostics)
                )
            line = (
                f"{short_label} | {attempt_status}{phase_status}"
                f"{status}{reason_status}{diagnostics_status}"
            )
        _write_progress_line(
            line,
            persist=bool(event.get("persist")),
        )

    return update


def _finish_progress() -> None:
    print()


def _confirm_overwrite(output_paths: list[Path]) -> list[Path]:
    existing_paths = [path for path in output_paths if path.exists()]
    if not existing_paths:
        return []

    print("Existing dataset files would be overwritten:")
    for path in existing_paths:
        print(f"  {path}")
    try:
        response = input("Overwrite these files? [y/N]: ")
    except EOFError:
        response = ""
    if response.strip().lower() not in {"y", "yes"}:
        raise SystemExit("Dataset generation cancelled.")
    return existing_paths


def _remove_dataset_files(output_paths: list[Path]) -> None:
    for path in output_paths:
        if not path.exists():
            continue
        if not path.is_file():
            raise SystemExit(
                f"Cannot remove existing dataset path that is not a file: "
                f"{path}"
            )
        print(f"Removing old dataset file: {path}")
        path.unlink()


def _load_dataset_arrays(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path) as data:
        layouts = data["layouts"] if "layouts" in data else data["mazes"]
        return (
            layouts,
            data["seeds"],
            data["layout_indices"],
            data["starts"],
            data["goals"],
        )


def _load_metadata(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    with path.open() as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise SystemExit(f"Dataset metadata must be an object: {path}")
    return value


def _write_metadata(path: Path, metadata: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as file:
        json.dump(metadata, file, indent=2)


def _build_metadata(
    args: argparse.Namespace,
    *,
    same_layout_maze_count: int,
    stratification_metadata: dict[str, object] | None,
    validation_stratified_metadata: dict[str, object] | None,
    completed_splits: set[str],
    complete: bool,
) -> dict[str, object]:
    metadata: dict[str, object] = {
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
            else (
                {
                    "short_max": float(args.short_max_path_length),
                    "medium_max": float(args.medium_max_path_length),
                }
                if args.stratify
                else None
            )
        ),
        "same_layout_new_goals": {
            "source": "train_layouts",
            "layout_count": same_layout_maze_count,
            "tasks_per_maze": args.tasks_per_maze,
        },
        "generation_status": "complete" if complete else "in_progress",
        "completed_splits": sorted(completed_splits),
    }
    if args.stratify:
        metadata["stratification"] = {
            "train": stratification_metadata,
            "validation_stratified": validation_stratified_metadata,
        }
    return metadata


def _validate_resume_metadata(
    args: argparse.Namespace,
    metadata: dict[str, object],
) -> None:
    expected: dict[str, object] = {
        "height": args.height,
        "width": args.width,
        "wall_probability": args.wall_probability,
        "min_path_length": args.min_path_length,
        "master_seed": args.seed,
        "train_mazes": args.train_mazes,
        "validation_mazes": args.validation_mazes,
        "test_mazes": args.test_mazes,
        "tasks_per_maze": args.tasks_per_maze,
        "stratify": args.stratify,
        "dedupe_threshold": args.dedupe_threshold,
        "bucket_budgets": args.bucket_budgets,
        "min_tasks_per_maze": args.min_tasks_per_maze,
        "max_tasks_per_maze": args.max_tasks_per_maze,
        "local_search_budget": args.local_search_budget,
        "global_repair_budget": args.global_repair_budget,
    }
    if args.stratify:
        expected["difficulty_boundaries"] = {
            "short_max": float(args.short_max_path_length),
            "medium_max": float(args.medium_max_path_length),
        }
    comparable_metadata = dict(metadata)
    if "difficulty_boundaries" not in comparable_metadata:
        stratification = comparable_metadata.get("stratification")
        if isinstance(stratification, dict):
            train_metadata = stratification.get("train")
            if isinstance(train_metadata, dict):
                comparable_metadata["difficulty_boundaries"] = (
                    train_metadata.get("difficulty_boundaries")
                )
    mismatches = [
        field
        for field, value in expected.items()
        if field in comparable_metadata
        and comparable_metadata[field] != value
    ]
    if mismatches:
        raise SystemExit(
            "Resume configuration does not match existing metadata for: "
            + ", ".join(mismatches)
        )


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
        "--resume",
        action="store_true",
        help=(
            "Resume generation by loading existing split files and "
            "generating only missing splits."
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
        validation_stratified_path,
        same_layout_path,
        test_path,
        metadata_path,
    ]
    if args.validation_mazes > args.train_mazes:
        raise SystemExit(
            "--validation-mazes cannot exceed --train-mazes because "
            "same_layout_new_goals.npz uses validation-sized held-out "
            "tasks on unique training layouts."
        )
    checkpoint_metadata = {}
    if args.resume:
        checkpoint_metadata = _load_metadata(metadata_path)
        if checkpoint_metadata:
            _validate_resume_metadata(args, checkpoint_metadata)
    else:
        existing_paths = _confirm_overwrite(output_paths)
        _remove_dataset_files(existing_paths)

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

    split_paths = [
        ("train", train_path),
        ("validation", validation_path),
        ("same_layout_new_goals", same_layout_path),
        ("test", test_path),
    ]
    if args.stratify:
        split_paths.append(
            ("validation_stratified", validation_stratified_path)
        )
    completed_splits = {
        split for split, path in split_paths if path.exists()
    }
    stratification_checkpoint = checkpoint_metadata.get(
        "stratification",
        {},
    )
    if not isinstance(stratification_checkpoint, dict):
        stratification_checkpoint = {}
    stratification_metadata = stratification_checkpoint.get("train")
    if not isinstance(stratification_metadata, dict):
        stratification_metadata = None
    validation_stratified_metadata = stratification_checkpoint.get(
        "validation_stratified"
    )
    if not isinstance(validation_stratified_metadata, dict):
        validation_stratified_metadata = None

    same_layout_maze_count = args.validation_mazes

    def checkpoint() -> None:
        _write_metadata(
            metadata_path,
            _build_metadata(
                args,
                same_layout_maze_count=same_layout_maze_count,
                stratification_metadata=stratification_metadata,
                validation_stratified_metadata=validation_stratified_metadata,
                completed_splits=completed_splits,
                complete=False,
            ),
        )

    if train_path.exists():
        print("Loading existing train split.")
        (
            train_layouts,
            _saved_train_seeds,
            train_layout_indices,
            train_starts,
            train_goals,
        ) = _load_dataset_arrays(train_path)
    else:
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

    if not args.stratify:
        stratification_metadata = None
    elif not train_path.exists():
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
    elif stratification_metadata is None:
        raise SystemExit(
            "Cannot resume stratified generation: train metadata is "
            "missing from metadata.json."
        )

    if not train_path.exists():
        save_dataset(
            train_path,
            train_layouts,
            train_seeds,
            train_layout_indices,
            train_starts,
            train_goals,
        )
        completed_splits.add("train")
        checkpoint()

    if validation_path.exists():
        print("Loading existing validation split.")
        (
            validation_layouts,
            _saved_validation_seeds,
            validation_layout_indices,
            validation_starts,
            validation_goals,
        ) = _load_dataset_arrays(validation_path)
    else:
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
        save_dataset(
            validation_path,
            validation_layouts,
            validation_seeds,
            validation_layout_indices,
            validation_starts,
            validation_goals,
        )
        completed_splits.add("validation")
        checkpoint()

    if args.stratify:
        if validation_stratified_path.exists():
            print("Loading existing stratified validation split.")
            (
                _validation_stratified_layouts,
                _saved_validation_stratified_seeds,
                validation_stratified_indices,
                validation_stratified_starts,
                validation_stratified_goals,
            ) = _load_dataset_arrays(validation_stratified_path)
        else:
            train_boundaries = DifficultyBoundaries(
                short_max=args.short_max_path_length,
                medium_max=args.medium_max_path_length,
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
            save_dataset(
                validation_stratified_path,
                validation_layouts,
                validation_seeds,
                validation_stratified_indices,
                validation_stratified_starts,
                validation_stratified_goals,
            )
            completed_splits.add("validation_stratified")
            checkpoint()

    if same_layout_path.exists():
        print("Loading existing same-layout split.")
        (
            same_layout_layouts,
            _saved_same_layout_seeds,
            same_layout_indices,
            same_layout_starts,
            same_layout_goals,
        ) = _load_dataset_arrays(same_layout_path)
    else:
        same_layout_layouts = train_layouts[:same_layout_maze_count]
        same_layout_seeds = train_seeds[:same_layout_maze_count]
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
        save_dataset(
            same_layout_path,
            same_layout_layouts,
            same_layout_seeds,
            same_layout_indices,
            same_layout_starts,
            same_layout_goals,
        )
        completed_splits.add("same_layout_new_goals")
        checkpoint()

    if test_path.exists():
        print("Loading existing test split.")
        (
            test_layouts,
            _saved_test_seeds,
            test_layout_indices,
            test_starts,
            test_goals,
        ) = _load_dataset_arrays(test_path)
    else:
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
            test_path,
            test_layouts,
            test_seeds,
            test_layout_indices,
            test_starts,
            test_goals,
        )
        completed_splits.add("test")
        checkpoint()

    metadata = _build_metadata(
        args,
        same_layout_maze_count=same_layout_maze_count,
        stratification_metadata=stratification_metadata,
        validation_stratified_metadata=validation_stratified_metadata,
        completed_splits=completed_splits,
        complete=True,
    )
    _write_metadata(metadata_path, metadata)

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
