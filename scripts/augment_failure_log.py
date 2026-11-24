"""Build a stratified augmented dataset from an interactive failure log."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from maze_rl.maze.augmentation_dataset import (
    build_mixed_family_es_arrays,
    build_split_arrays,
    group_source_tasks,
    load_failure_log,
    split_source_families,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--failure-log", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--difficulty-buckets", type=int, default=5)
    parser.add_argument("--perturbations-per-task", type=int, default=0)
    parser.add_argument("--max-variants-per-family", type=int, default=64)
    parser.add_argument("--max-edits", type=int, default=4)
    parser.add_argument("--max-attempts", type=int, default=100)
    parser.add_argument("--add-probability", type=float, default=0.5)
    parser.add_argument("--max-path-increase", type=int)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument(
        "--split-mode",
        choices=("family", "mixed-family-es"),
        default="family",
        help=(
            "family assigns whole families to every split. "
            "mixed-family-es keeps test family-held out and splits "
            "development families into train and same-family ES groups."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    tasks = load_failure_log(args.failure_log)
    families = group_source_tasks(tasks)
    splits = split_source_families(
        families,
        train_fraction=args.train_fraction,
        validation_fraction=args.validation_fraction,
        test_fraction=args.test_fraction,
        difficulty_buckets=args.difficulty_buckets,
        seed=args.seed,
    )

    if args.split_mode == "mixed-family-es":
        development_families = (
            splits["train"] + splits["validation"]
        )
        development_fraction = (
            args.validation_fraction
            / (args.train_fraction + args.validation_fraction)
        )
        test_arrays = build_split_arrays(
            splits["test"],
            seed=args.seed + 2,
            perturbations_per_task=args.perturbations_per_task,
            max_variants_per_family=args.max_variants_per_family,
            max_edits=args.max_edits,
            max_attempts=args.max_attempts,
            add_probability=args.add_probability,
            max_path_increase=args.max_path_increase,
        )
        test_task_identities = {
            (
                test_arrays["layouts"][int(layout_index)].shape,
                test_arrays["layouts"][int(layout_index)].tobytes(),
                tuple(int(value) for value in start),
                tuple(int(value) for value in goal),
            )
            for layout_index, start, goal in zip(
                test_arrays["layout_indices"],
                test_arrays["starts"],
                test_arrays["goals"],
            )
        }
        train_arrays, validation_arrays = build_mixed_family_es_arrays(
            development_families,
            seed=args.seed,
            validation_fraction=development_fraction,
            perturbations_per_task=args.perturbations_per_task,
            max_variants_per_family=args.max_variants_per_family,
            max_edits=args.max_edits,
            max_attempts=args.max_attempts,
            add_probability=args.add_probability,
            max_path_increase=args.max_path_increase,
            excluded_task_identities=test_task_identities,
        )
        arrays_by_split = {
            "train": train_arrays,
            "validation": validation_arrays,
            "test": test_arrays,
        }
    else:
        arrays_by_split = {
            split_name: build_split_arrays(
                splits[split_name],
                seed=args.seed + split_index,
                perturbations_per_task=args.perturbations_per_task,
                max_variants_per_family=args.max_variants_per_family,
                max_edits=args.max_edits,
                max_attempts=args.max_attempts,
                add_probability=args.add_probability,
                max_path_increase=args.max_path_increase,
            )
            for split_index, split_name in enumerate(
                ("train", "validation", "test")
            )
        }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "source_failure_log": str(args.failure_log),
        "split_mode": args.split_mode,
        "family_overlap_policy": (
            "train_validation_shared_test_family_held_out"
            if args.split_mode == "mixed-family-es"
            else "all_splits_family_disjoint"
        ),
        "seed": args.seed,
        "difficulty_buckets": args.difficulty_buckets,
        "perturbations_per_task": args.perturbations_per_task,
        "max_variants_per_family": args.max_variants_per_family,
        "max_edits": args.max_edits,
        "max_attempts": args.max_attempts,
        "add_probability": args.add_probability,
        "max_path_increase": args.max_path_increase,
        "split_fractions": {
            "train": args.train_fraction,
            "validation": args.validation_fraction,
            "test": args.test_fraction,
        },
        "source_records": len(tasks),
        "source_families": len(families),
        "splits": {},
    }
    for split_name in ("train", "validation", "test"):
        arrays = arrays_by_split[split_name]
        np.savez_compressed(args.output_dir / f"{split_name}.npz", **arrays)
        metadata["splits"][split_name] = {
            "families": int(np.unique(arrays["family_indices"]).size),
            "tasks": len(arrays["starts"]),
            "layouts": len(arrays["layouts"]),
        }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
