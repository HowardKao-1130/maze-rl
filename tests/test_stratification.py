import numpy as np
import pytest
from types import SimpleNamespace

from maze_rl.maze.stratification import (
    DifficultyBoundaries,
    TaskCandidate,
    apply_difficulty_buckets,
    build_candidate_index,
    bucket_quotas,
    directed_weighted_jaccard,
    _failure_diagnostics,
    format_failure_diagnostics,
    generate_stratified_tasks_for_layouts,
    allocate_bucket_slots,
    threshold_schedule,
)
from maze_rl.maze.pathfinding import shortest_distances
from scripts.experiment_naming import experiment_dataset_name
from scripts.generate_dataset import (
    _confirm_overwrite,
    _write_progress_line,
    parse_args,
)


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


def test_explicit_difficulty_boundaries_assign_inclusive_ranges():
    maze = np.zeros((1, 30), dtype=np.uint8)
    index = build_candidate_index(maze, min_path_length=1)
    apply_difficulty_buckets(
        [index],
        DifficultyBoundaries(short_max=15, medium_max=25),
    )

    buckets_by_distance = {
        candidate.distance: candidate.bucket
        for candidate in index.candidates
    }
    assert buckets_by_distance[15] == "short"
    assert buckets_by_distance[16] == "medium"
    assert buckets_by_distance[25] == "medium"
    assert buckets_by_distance[26] == "long"


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


def test_failure_diagnostics_distinguish_search_stopping_reasons():
    diagnostics = _failure_diagnostics(
        {
            "budget_exhausted_layouts": [2],
            "proven_infeasible_layouts": [4, 7],
        },
        unresolved_before_repair=[
            (2, "long"),
            (4, "medium"),
            (7, "long"),
        ],
        unresolved_after_repair=[(7, "long")],
        contributions=[5, 5, 4, 5, 2, 5, 5, 2],
        min_tasks_per_maze=3,
        max_tasks_per_maze=7,
        repair_metadata={
            "repair_attempts": 3,
            "successful_reallocations": 2,
        },
    )

    assert diagnostics["budget_exhausted_count"] == 1
    assert diagnostics["infeasible_assignment_count"] == 2
    assert diagnostics["contribution_below_min_layouts"] == [4, 7]
    assert diagnostics["unresolved_after_repair_by_bucket"] == {
        "short": 0,
        "medium": 0,
        "long": 1,
    }
    message = format_failure_diagnostics(diagnostics)
    assert "budget-exhausted layouts=1 [2]" in message
    assert "infeasible current assignments=2 [4,7]" in message
    assert "repairs=2/3" in message


def test_slot_allocator_preserves_contribution_bounds_and_bucket_quotas():
    layouts = np.zeros((10, 11, 11), dtype=np.uint8)
    indexes = [
        build_candidate_index(maze, min_path_length=3)
        for maze in layouts
    ]
    apply_difficulty_buckets(
        indexes,
        DifficultyBoundaries(short_max=5, medium_max=8),
    )
    quotas = {"short": 10, "medium": 20, "long": 20}

    assignments = allocate_bucket_slots(
        indexes,
        tasks_per_maze=5,
        quotas=quotas,
        rng=np.random.default_rng(42),
        min_tasks_per_maze=3,
        max_tasks_per_maze=7,
    )

    assert all(sum(assignment.values()) == 5 for assignment in assignments)
    assert {
        bucket: sum(assignment[bucket] for assignment in assignments)
        for bucket in ("short", "medium", "long")
    } == quotas


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

    assert args.resume is False
    assert args.dedupe_threshold == 0.75
    assert args.min_tasks_per_maze == 3
    assert args.max_tasks_per_maze == 7
    assert args.local_search_budget == 2_000
    assert args.short_max_path_length == 10
    assert args.medium_max_path_length == 20
    assert args.bucket_budgets == {
        "short": 0.2,
        "medium": 0.4,
        "long": 0.4,
    }
    assert args.output_dir.name.endswith(
        "_stratified_s20_m40_l40_dedupe075"
    )


def test_overwrite_confirmation_requires_explicit_yes(tmp_path, monkeypatch):
    existing_file = tmp_path / "train.npz"
    existing_file.touch()

    monkeypatch.setattr("builtins.input", lambda _: "n")
    with pytest.raises(SystemExit, match="cancelled"):
        _confirm_overwrite([existing_file])

    monkeypatch.setattr("builtins.input", lambda _: "yes")
    _confirm_overwrite([existing_file])


def test_live_progress_updates_stay_on_one_line(capsys):
    _write_progress_line("selecting tasks | layouts 0/5000")
    _write_progress_line("selecting tasks | layouts 1/5000")
    _write_progress_line("selection complete", persist=True)

    assert capsys.readouterr().out == (
        "selecting tasks | layouts 0/5000"
        "\r\033[Jselecting tasks | layouts 1/5000"
        "\r\033[Jselection complete\n"
    )


def test_live_progress_clears_wrapped_rows(monkeypatch, capsys):
    monkeypatch.setattr(
        "scripts.generate_dataset.shutil.get_terminal_size",
        lambda fallback: SimpleNamespace(columns=10),
    )

    _write_progress_line("123456789012345")
    _write_progress_line("updated", persist=True)

    assert capsys.readouterr().out == (
        "123456789012345"
        "\033[1A\r\033[Jupdated\n"
    )


def test_stratified_generation_is_reproducible_and_records_configuration():
    layouts = np.zeros((2, 7, 7), dtype=np.uint8)
    seeds = np.array([11, 22])
    progress_events = []
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
        "difficulty_boundaries": DifficultyBoundaries(
            short_max=5,
            medium_max=8,
        ),
        "progress_callback": progress_events.append,
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
    assert first[3]["difficulty_boundaries"] == {
        "short_max": 5.0,
        "medium_max": 8.0,
    }
    assert progress_events[0]["phase"] == "indexing"
    assert progress_events[-1]["allocation_attempt"] == 1
    assert progress_events[-1]["allocation_attempts"] == 8
    assert progress_events[-1]["bucket_counts"] == {
        "short": 2,
        "medium": 4,
        "long": 4,
    }
    allocation_events = [
        event
        for event in progress_events
        if event["phase"] == "allocation"
        and "allocation_bucket" in event
    ]
    assert allocation_events[-1]["allocation_slots_completed"] == 10
    assert allocation_events[-1]["allocation_slots_total"] == 10
    selection_events = [
        event
        for event in progress_events
        if event["phase"] == "selection"
    ]
    assert selection_events[0]["layouts_completed"] == 0
    assert selection_events[0]["layout_count"] == 2
    assert selection_events[-1]["layouts_completed"] == 2
