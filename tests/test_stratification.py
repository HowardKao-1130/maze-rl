from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from maze_rl.maze import stratification as stratification_module
from maze_rl.maze.stratification import (
    DifficultyBoundaries,
    TaskCandidate,
    apply_difficulty_buckets,
    build_candidate_index,
    bucket_quotas,
    directed_path_jaccard,
    expected_pairwise_directed_path_jaccard,
    generate_stratified_dataset,
    _select_tasks_for_maze,
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


def test_stratified_generation_checks_and_preserves_full_quota(monkeypatch):
    attempts = []

    def fake_generate_valid_maze(**kwargs):
        maze = np.zeros((1, 20), dtype=np.uint8)
        return maze, (0, 0), (0, 19)

    monkeypatch.setattr(
        stratification_module,
        "generate_valid_maze",
        fake_generate_valid_maze,
    )
    def fake_select_tasks(index, assignment, **kwargs):
        attempts.append(dict(assignment))
        selected = [
            TaskCandidate((0, 0), (0, 1), 1, bucket="short"),
            TaskCandidate((0, 0), (0, 5), 5, bucket="medium"),
            TaskCandidate((0, 0), (0, 19), 19, bucket="long"),
        ]
        return SimpleNamespace(
            selected=selected,
            completed=len(attempts) == 2,
            candidate_attempts=1,
            backtracking_events=1,
            budget_exhausted=False,
        )

    monkeypatch.setattr(
        stratification_module,
        "_select_tasks_for_maze",
        fake_select_tasks,
    )

    layouts, layout_indices, starts, goals, metadata = (
        generate_stratified_dataset(
            seeds=np.array([7]),
            height=1,
            width=20,
            wall_probability=0.0,
            tasks_per_maze=3,
            min_path_length=1,
            bucket_budgets={"short": 1 / 3, "medium": 1 / 3, "long": 1 / 3},
            selection_attempts=2,
            difficulty_boundaries=DifficultyBoundaries(
                short_max=2,
                medium_max=15,
            ),
        )
    )

    assert layouts.shape == (1, 1, 20)
    assert len(attempts) == 2
    assert attempts[0] == {"short": 1, "medium": 1, "long": 1}
    assert np.array_equal(layout_indices, np.zeros(3, dtype=np.int64))
    assert starts.tolist() == [[0, 0], [0, 0], [0, 0]]
    assert goals.tolist() == [[0, 1], [0, 5], [0, 19]]
    assert metadata["selection_attempts"] == [2]


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


def test_stratified_dataset_name_uses_canonical_prefix_and_suffix():
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

    assert name == (
        "dataset_10x2x2_t5_seed42"
        "_stratified_s20_m40_l40_dedupe075"
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
    assert args.local_search_budget == 2_000
    assert args.short_max_path_length == 2
    assert args.medium_max_path_length == 15
    assert args.bucket_budgets == {
        "short": 0.0,
        "medium": 0.6,
        "long": 0.4,
    }
    assert args.output_dir == Path(
        "data/datasets/"
        "dataset_1000x200x200_t5_seed42"
        "_stratified_s00_m60_l40_dedupe070"
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


def test_stratified_progress_updates_phases_on_one_line(capsys):
    update = _stratified_progress_callback("training")
    update(
        {
            "phase": "generation",
            "layouts_completed": 2,
            "layout_count": 2,
            "bucket_counts": {"short": 0, "medium": 0, "long": 0},
            "bucket_quotas": {"short": 0, "medium": 6, "long": 4},
        }
    )
    update(
        {
            "phase": "selection",
            "layouts_completed": 1,
            "layout_count": 2,
            "selection_attempt": 1,
            "selection_attempts": 2,
            "reason": "retrying layout",
            "bucket_counts": {"short": 0, "medium": 0, "long": 1},
            "bucket_quotas": {"short": 0, "medium": 6, "long": 4},
        }
    )
    _finish_progress()

    output = capsys.readouterr().out
    assert "Stratified training | generate 2/2" in output
    assert "Stratified training | select 1/2" in output
    assert "retry 1/2" in output
    assert "\033[JStratified training | select 1/2" in output
    assert output.endswith("\n")


def test_stratified_generation_is_reproducible_and_records_configuration():
    seeds = np.array([11, 22])
    progress_events = []
    kwargs = {
        "height": 7,
        "width": 7,
        "wall_probability": 0.0,
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

    first = generate_stratified_dataset(seeds=seeds, **kwargs)
    second = generate_stratified_dataset(seeds=seeds, **kwargs)

    assert np.array_equal(first[0], second[0])
    assert np.array_equal(first[1], second[1])
    assert np.array_equal(first[2], second[2])
    assert np.array_equal(first[3], second[3])
    assert first[4]["bucket_counts"] == {
        "short": 2,
        "medium": 4,
        "long": 4,
    }
    assert first[4]["dedupe_threshold"] == 0.75
    assert first[4]["dedupe_scope"] == "within_layout_bucket"
    assert first[4]["difficulty_boundaries"] == {
        "short_max": 5.0,
        "medium_max": 8.0,
    }
    assert progress_events[0]["phase"] == "selection"
    assert progress_events[-1]["phase"] == "generation"
    assert progress_events[-1]["bucket_counts"] == {
        "short": 2,
        "medium": 4,
        "long": 4,
    }
    assert first[4]["bucket_quotas_per_layout"] == {
        "short": 1,
        "medium": 2,
        "long": 2,
    }
    assert first[4]["contributions_by_layout"] == [5, 5]
