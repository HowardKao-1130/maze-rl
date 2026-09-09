import argparse
import json
from pathlib import Path

from maze_rl.maze.dataset import (
    generate_mazes,
    generate_tasks_for_layouts,
    generate_split_seeds,
    save_dataset,
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
        default=Path("data"),
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

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.validation_mazes > args.train_mazes:
        raise SystemExit(
            "--validation-mazes cannot exceed --train-mazes because "
            "same_layout_new_goals.npz uses validation-sized held-out "
            "tasks on unique training layouts."
        )

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
        args.output_dir / "train.npz",
        train_layouts,
        train_seeds,
        train_layout_indices,
        train_starts,
        train_goals,
    )

    save_dataset(
        args.output_dir / "validation.npz",
        validation_layouts,
        validation_seeds,
        validation_layout_indices,
        validation_starts,
        validation_goals,
    )

    save_dataset(
        args.output_dir / "same_layout_new_goals.npz",
        same_layout_layouts,
        same_layout_seeds,
        same_layout_indices,
        same_layout_starts,
        same_layout_goals,
    )

    save_dataset(
        args.output_dir / "test.npz",
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

    metadata_path = args.output_dir / "metadata.json"

    with metadata_path.open("w") as file:
        json.dump(
            metadata,
            file,
            indent=2,
        )

    print()
    print("Dataset generated successfully.")
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
