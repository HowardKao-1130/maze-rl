import numpy as np
import pytest

from maze_rl.maze.stratification import (
    TaskCandidate,
    build_candidate_index,
    bucket_quotas,
    directed_weighted_jaccard,
    generate_stratified_tasks_for_layouts,
    threshold_schedule,
)
from maze_rl.maze.pathfinding import shortest_distances
from scripts.experiment_naming import experiment_dataset_name
from scripts.generate_dataset import parse_args


def test_directed_profiles_have_expected_similarity_properties():
    maze = np.zeros((3, 7), dtype=np.uint8)
    index = build_candidate_index(maze, min_path_length=1)
    first = TaskCandidate((1, 0), (1, 3), 3)
    reverse = TaskCandidate((1, 3), (1, 0), 3)
    disjoint = TaskCandidate((1, 4), (1, 6), 2)

    first_profile = index.profile(first)
    reverse_profile = index.profile(reverse)
    disjoint_profile = index.profile(disjoint)

    assert directed_weighted_jaccard(first_profile, first_profile) == 1
    assert directed_weighted_jaccard(first_profile, reverse_profile) == 0
    assert directed_weighted_jaccard(first_profile, disjoint_profile) == 0
    assert directed_weighted_jaccard(first_profile, disjoint_profile) == (
        directed_weighted_jaccard(disjoint_profile, first_profile)
    )
    assert 0 <= directed_weighted_jaccard(
        first_profile,
        disjoint_profile,
    ) <= 1
    assert sum(first_profile.values()) == pytest.approx(first.distance)
    assert sum(reverse_profile.values()) == pytest.approx(reverse.distance)


def test_candidate_index_records_exact_shortest_path_lengths():
    maze = np.zeros((4, 4), dtype=np.uint8)
    index = build_candidate_index(maze, min_path_length=2)

    for candidate in index.candidates:
        assert candidate.distance == shortest_distances(
            maze,
            candidate.start,
        )[candidate.goal]


def test_bucket_quotas_use_configured_budget_and_sum_to_total():
    assert bucket_quotas(
        10,
        {"short": 0.2, "medium": 0.4, "long": 0.4},
    ) == {"short": 2, "medium": 4, "long": 4}
    assert sum(
        bucket_quotas(
            7,
            {"short": 0.2, "medium": 0.4, "long": 0.4},
        ).values()
    ) == 7


def test_threshold_fallback_does_not_relax_the_configured_threshold():
    assert threshold_schedule(0.75) == (0.75,)


def test_stratified_dataset_name_includes_budgets_and_threshold():
    name = experiment_dataset_name(
        "experiment",
        train_mazes=10,
        validation_mazes=2,
        test_mazes=2,
        tasks_per_maze=5,
        seed=42,
        stratify=True,
        dedupe_threshold=0.75,
        bucket_budgets={
            "short": 0.2,
            "medium": 0.4,
            "long": 0.4,
        },
    )

    assert name.endswith("_stratified_s20_m40_l40_dedupe075")


def test_cli_defaults_configure_stratification(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        ["generate_dataset.py", "--stratify"],
    )

    args = parse_args()

    assert args.dedupe_threshold == 0.75
    assert args.min_tasks_per_maze == 3
    assert args.max_tasks_per_maze == 7
    assert args.local_search_budget == 2_000
    assert args.bucket_budgets == {
        "short": 0.2,
        "medium": 0.4,
        "long": 0.4,
    }
    assert args.output_dir.name.endswith(
        "_stratified_s20_m40_l40_dedupe075"
    )


def test_stratified_generation_is_reproducible_and_records_configuration():
    layouts = np.zeros((2, 7, 7), dtype=np.uint8)
    seeds = np.array([11, 22])
    kwargs = {
        "tasks_per_maze": 5,
        "min_path_length": 3,
        "dedupe_threshold": 0.75,
        "bucket_budgets": {
            "short": 0.2,
            "medium": 0.4,
            "long": 0.4,
        },
        "random_seed": 42,
    }

    first = generate_stratified_tasks_for_layouts(layouts, seeds, **kwargs)
    second = generate_stratified_tasks_for_layouts(layouts, seeds, **kwargs)

    assert np.array_equal(first[0], second[0])
    assert np.array_equal(first[1], second[1])
    assert np.array_equal(first[2], second[2])
    assert first[3]["bucket_counts"] == {
        "short": 2,
        "medium": 4,
        "long": 4,
    }
    assert first[3]["dedupe_threshold"] == 0.75
