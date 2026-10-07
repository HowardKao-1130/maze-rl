import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(REPO_ROOT),
    )

from maze_rl.maze.dataset import (
    generate_mazes,
    generate_tasks_for_layouts,
    generate_split_seeds,
    save_dataset,
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

    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = args.output_root / experiment_dataset_name(
            args.experiment_name,
            train_mazes=args.train_mazes,
            validation_mazes=args.validation_mazes,
            test_mazes=args.test_mazes,
            tasks_per_maze=args.tasks_per_maze,
            seed=args.seed,
        )

    return args


def main() -> None:
    args = parse_args()
    train_path = args.output_dir / "train.npz"
    validation_path = args.output_dir / "validation.npz"
    same_layout_path = (
        args.output_dir / "same_layout_new_goals.npz"
    )
    test_path = args.output_dir / "test.npz"
    metadata_path = args.output_dir / "metadata.json"

    if args.validation_mazes > args.train_mazes:
        raise SystemExit(
            "--validation-mazes cannot exceed --train-mazes because "
            "same_layout_new_goals.npz uses validation-sized held-out "
            "tasks on unique training layouts."
        )

    print(f"Writing datasets to {args.output_dir}")
    print(f"  train:        {train_path}")
    print(f"  validation:   {validation_path}")
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

    print("Generating training mazes...")
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
    )

    print("Generating validation mazes...")
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
    )

    print("Generating same-layout held-out tasks...")
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
    )

    print("Generating test mazes...")
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
    )

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
        "same_layout_new_goals": {
            "source": "train_layouts",
            "layout_count": same_layout_maze_count,
            "tasks_per_maze": args.tasks_per_maze,
        },
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
