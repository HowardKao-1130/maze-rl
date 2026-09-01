import argparse
import json
from pathlib import Path

from maze_rl.maze.dataset import (
    generate_mazes,
    generate_split_seeds,
    save_dataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate maze datasets."
    )

    parser.add_argument(
        "--train",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--validation",
        type=int,
        default=200,
    )

    parser.add_argument(
        "--test",
        type=int,
        default=200,
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

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    (
        train_seeds,
        validation_seeds,
        test_seeds,
    ) = generate_split_seeds(
        train_size=args.train,
        validation_size=args.validation,
        test_size=args.test,
        master_seed=args.seed,
    )

    print("Generating training mazes...")
    train_mazes, train_starts, train_goals = generate_mazes(
        seeds=train_seeds,
        height=args.height,
        width=args.width,
        wall_probability=args.wall_probability,
        min_path_length=args.min_path_length,
    )

    print("Generating validation mazes...")
    validation_mazes, validation_starts, validation_goals = generate_mazes(
        seeds=validation_seeds,
        height=args.height,
        width=args.width,
        wall_probability=args.wall_probability,
        min_path_length=args.min_path_length,
    )

    print("Generating test mazes...")
    test_mazes, test_starts, test_goals = generate_mazes(
        seeds=test_seeds,
        height=args.height,
        width=args.width,
        wall_probability=args.wall_probability,
        min_path_length=args.min_path_length,
    )

    save_dataset(
        args.output_dir / "train.npz",
        train_mazes,
        train_seeds,
        train_starts,
        train_goals,
    )

    save_dataset(
        args.output_dir / "validation.npz",
        validation_mazes,
        validation_seeds,
        validation_starts,
        validation_goals,
    )

    save_dataset(
        args.output_dir / "test.npz",
        test_mazes,
        test_seeds,
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
        "train_size": args.train,
        "validation_size": args.validation,
        "test_size": args.test,
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
    print(f"Train:      {train_mazes.shape}")
    print(f"Validation: {validation_mazes.shape}")
    print(f"Test:       {test_mazes.shape}")


if __name__ == "__main__":
    main()