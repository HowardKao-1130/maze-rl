import numpy as np
import pytest
from types import SimpleNamespace

from maze_rl.maze import stratification as stratification_module
from maze_rl.maze.stratification import (
    DifficultyBoundaries,
    TaskCandidate,
    apply_difficulty_buckets,
    build_candidate_index,
    bucket_quotas,
    directed_path_jaccard,
    expected_pairwise_directed_path_jaccard,
    _failure_diagnostics,
    format_failure_diagnostics,
    generate_layouts,
    generate_stratified_tasks_for_layouts,
    allocate_bucket_slots,
    _select_tasks_for_maze,
    threshold_schedule,
)
from maze_rl.maze.generator import count_valid_tasks_by_min_path_length
from maze_rl.maze.pathfinding import shortest_distances
from scripts.experiment_naming import experiment_dataset_name
from scripts.generate_dataset import (
    _confirm_overwrite,
    _finish_progress,
    _prompt_existing_dataset_action,
    _stratified_progress_callback,
    _write_progress_line,
    parse_args,
)


def test_expected_pairwise_directed_path_jaccard_has_expected_properties():
    shared = ((1, 1), (1, 2))
    first_only = ((1, 2), (1, 3))
    first_alternative = ((1, 1), (2, 1))
    second_only = ((1, 2), (2, 2))

    first_paths = (
        (shared, first_only),
        (shared, first_alternative),
    )
    second_paths = (
        (shared, first_only),
        (shared, second_only),
    )

    assert expected_pairwise_directed_path_jaccard(
        first_paths,
        first_paths,
    ) == pytest.approx(2 / 3)
    assert expected_pairwise_directed_path_jaccard(
        first_paths,
        second_paths,
    ) == pytest.approx(0.5)
    assert expected_pairwise_directed_path_jaccard(
        second_paths,
        first_paths,
    ) == pytest.approx(0.5)
    assert directed_path_jaccard(first_paths[0], first_paths[0]) == 1
    assert directed_path_jaccard(
        first_paths[0],
        tuple((edge[1], edge[0]) for edge in first_paths[0]),
    ) == 0


def test_candidate_similarity_uses_expected_directed_path_similarity():
    maze = np.zeros((3, 7), dtype=np.uint8)
    index = build_candidate_index(maze, min_path_length=1)
    first = TaskCandidate((1, 0), (1, 3), 3)
    reverse = TaskCandidate((1, 3), (1, 0), 3)

    assert index.similarity(first, first) == 1
    assert index.similarity(first, reverse) == 0


def test_deduplication_only_compares_tasks_within_the_same_bucket():
    candidates = {
        "short": (
            TaskCandidate(
                (0, 0),
                (0, 1),
                1,
                bucket="short",
            ),
            TaskCandidate(
                (0, 1),
                (0, 2),
                1,
                bucket="short",
            ),
        ),
        "medium": (),
        "long": (
            TaskCandidate(
                (0, 0),
                (1, 0),
                1,
                bucket="long",
            ),
        ),
    }

    class AlwaysSimilarIndex:
        def candidates_by_bucket(self):
            return candidates

        def similarity(self, first, second):
            return 1.0

    cross_bucket_result = _select_tasks_for_maze(
        AlwaysSimilarIndex(),
        {"short": 1, "medium": 0, "long": 1},
        threshold=0.7,
        rng=np.random.default_rng(42),
    )
    assert cross_bucket_result.completed is True
    assert len(cross_bucket_result.selected) == 2

    same_bucket_result = _select_tasks_for_maze(
        AlwaysSimilarIndex(),
        {"short": 2, "medium": 0, "long": 0},
        threshold=0.7,
        rng=np.random.default_rng(42),
    )
    assert same_bucket_result.completed is False
    assert len(same_bucket_result.selected) == 1


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


def test_task_capacity_counts_multiple_thresholds_in_one_result():
    maze = np.zeros((1, 4), dtype=np.uint8)

    counts = count_valid_tasks_by_min_path_length(
        maze,
        (1, 3),
    )

    assert counts == {1: 12, 3: 2}


def test_layout_generation_rejects_selection_infeasible_candidates(monkeypatch):
    attempts = []

    def fake_generate_valid_maze(**kwargs):
        maze = np.zeros((1, 20), dtype=np.uint8)
        return maze, (0, 0), (0, 19)

    def fake_select_tasks(index, assignment, **kwargs):
        attempts.append(dict(assignment))
        return SimpleNamespace(
            selected=[],
            completed=len(attempts) == 2,
            candidate_attempts=1,
            backtracking_events=1,
            budget_exhausted=False,
        )

    monkeypatch.setattr(
        stratification_module,
        "generate_valid_maze",
        fake_generate_valid_maze,
    )
    monkeypatch.setattr(
        stratification_module,
        "_select_tasks_for_maze",
        fake_select_tasks,
    )

    layouts = generate_layouts(
        seeds=np.array([7]),
        height=1,
        width=20,
        wall_probability=0.0,
        min_path_length=3,
        minimum_eligible_tasks=2,
        minimum_long_eligible_tasks=1,
        long_path_min_length=16,
        required_bucket_counts={
            "short": 0,
            "medium": 1,
            "long": 1,
        },
        difficulty_boundaries=DifficultyBoundaries(
            short_max=2,
            medium_max=15,
        ),
    )

    assert layouts.shape == (1, 1, 20)
    assert len(attempts) == 2
    assert attempts[0] == {"short": 0, "medium": 0, "long": 1}


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
    assert len(message.splitlines()) == 6


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

    assert name.endswith(
        "_stratified_s20_m40_l40_dedupe075_withinbucket"
    )


def test_cli_defaults_configure_stratification(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        ["generate_dataset.py", "--stratify"],
    )

    args = parse_args()

    assert args.resume is False
    assert args.dedupe_threshold == 0.7
    assert args.min_path_length == 3
    assert args.min_tasks_per_maze == 3
    assert args.max_tasks_per_maze == 7
    assert args.local_search_budget == 2_000
    assert args.short_max_path_length == 2
    assert args.medium_max_path_length == 15
    assert args.bucket_budgets == {
        "short": 0.0,
        "medium": 0.6,
        "long": 0.4,
    }
    assert args.minimum_long_eligible_tasks == 2
    assert args.output_dir.name.endswith(
        "_stratified_s00_m60_l40_dedupe070_withinbucket"
    )


def test_overwrite_confirmation_requires_explicit_yes(tmp_path, monkeypatch):
    existing_file = tmp_path / "train.npz"
    existing_file.touch()

    monkeypatch.setattr("builtins.input", lambda _: "n")
    with pytest.raises(SystemExit, match="cancelled"):
        _confirm_overwrite([existing_file])

    monkeypatch.setattr("builtins.input", lambda _: "yes")
    _confirm_overwrite([existing_file])


@pytest.mark.parametrize(
    ("response", "expected"),
    [("override", "override"), ("resume", "resume")],
)
def test_existing_dataset_prompt_selects_action(
    tmp_path,
    monkeypatch,
    response,
    expected,
):
    existing_file = tmp_path / "train.npz"
    existing_file.touch()

    monkeypatch.setattr("builtins.input", lambda _: response)

    assert _prompt_existing_dataset_action([existing_file]) == expected


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


def test_different_progress_streams_do_not_overwrite_each_other(capsys):
    _write_progress_line("training", progress_key="training")
    _write_progress_line("validation", progress_key="validation")

    assert capsys.readouterr().out == "training\nvalidation"


def test_stratified_progress_separates_phase_messages(capsys):
    update = _stratified_progress_callback("training")
    update(
        {
            "phase": "indexing",
            "layouts_completed": 2,
            "layout_count": 2,
            "bucket_counts": {"short": 0, "medium": 0, "long": 0},
            "bucket_quotas": {"short": 0, "medium": 6, "long": 4},
        }
    )
    update(
        {
            "phase": "allocation",
            "selection_attempt": 1,
            "selection_attempts": 1,
            "allocation_bucket": "long",
            "allocation_bucket_completed": 1,
            "allocation_bucket_total": 4,
            "allocation_slots_completed": 1,
            "allocation_slots_total": 10,
            "bucket_counts": {"short": 0, "medium": 0, "long": 1},
            "bucket_quotas": {"short": 0, "medium": 6, "long": 4},
        }
    )
    _finish_progress()

    output = capsys.readouterr().out
    assert "Stratified training | index 2/2" in output
    assert "\nStratified training | attempt 1/1 | allocating slots" in output


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
    assert first[3]["dedupe_scope"] == "within_layout_bucket"
    assert first[3]["difficulty_boundaries"] == {
        "short_max": 5.0,
        "medium_max": 8.0,
    }
    assert progress_events[0]["phase"] == "indexing"
    assert progress_events[-1]["selection_attempt"] == 1
    assert progress_events[-1]["selection_attempts"] == 8
    assert progress_events[-1]["bucket_counts"] == {
        "short": 2,
        "medium": 4,
        "long": 4,
    }
    assert first[3]["bucket_quotas_per_layout"] == {
        "short": 1,
        "medium": 2,
        "long": 2,
    }
    assert first[3]["contributions_by_layout"] == [5, 5]
    assert not any(
        event["phase"] == "allocation"
        for event in progress_events
    )
    selection_events = [
        event
        for event in progress_events
        if event["phase"] == "selection"
    ]
    assert selection_events[0]["layouts_completed"] == 0
    assert selection_events[0]["layout_count"] == 2
    assert selection_events[-1]["layouts_completed"] == 2
