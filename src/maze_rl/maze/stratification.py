"""Difficulty-stratified task generation for maze datasets."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import time
from typing import Callable, Iterable, Mapping

import numpy as np

from maze_rl.maze.pathfinding import (
    Position,
    traversable_positions,
)
from maze_rl.maze.generator import (
    generate_valid_maze,
)


BUCKETS = ("short", "medium", "long")
ALLOCATION_ORDER = ("long", "medium", "short")
DEFAULT_BUCKET_BUDGETS = {
    "short": 0.00,
    "medium": 0.60,
    "long": 0.40,
}
Edge = tuple[Position, Position]
TaskKey = tuple[Position, Position]
Path = tuple[Edge, ...]

# Exact path-pair enumeration is useful for small shortest-path DAGs, while
# open layouts can contain too many shortest paths to enumerate safely.  The
# fallback remains an unbiased, reproducible estimate of the same expectation.
PATH_SAMPLE_COUNT = 128
MAX_EXACT_PATHS = 256
MAX_EXACT_PAIR_COUNT = 4_096
DEDUPE_SCOPE = "within_layout_bucket"


def _shortest_distances_and_counts(
    maze: np.ndarray,
    start: Position,
) -> tuple[dict[Position, int], dict[Position, int]]:
    height, width = maze.shape
    distances = {start: 0}
    path_counts = {start: 1}
    queue = deque([start])
    directions = [
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1),
    ]

    while queue:
        row, col = queue.popleft()
        next_distance = distances[(row, col)] + 1
        for d_row, d_col in directions:
            next_row = row + d_row
            next_col = col + d_col
            next_position = (next_row, next_col)
            if not (
                0 <= next_row < height
                and 0 <= next_col < width
            ):
                continue
            if maze[next_row, next_col] == 1:
                continue
            if next_position not in distances:
                distances[next_position] = next_distance
                path_counts[next_position] = path_counts[(row, col)]
                queue.append(next_position)
            elif distances[next_position] == next_distance:
                path_counts[next_position] += path_counts[(row, col)]

    return distances, path_counts


@dataclass(frozen=True)
class TaskCandidate:
    start: Position
    goal: Position
    distance: int
    bucket: str = ""

    @property
    def key(self) -> TaskKey:
        return self.start, self.goal


@dataclass(frozen=True)
class PathCollection:
    paths: tuple[frozenset[Edge], ...]
    exact: bool
    total_paths: int


@dataclass(frozen=True)
class DifficultyBoundaries:
    short_max: float
    medium_max: float

    def bucket_for(self, distance: int) -> str:
        if distance <= self.short_max:
            return "short"
        if distance <= self.medium_max:
            return "medium"
        return "long"

    def as_dict(self) -> dict[str, float]:
        return {
            "short_max": self.short_max,
            "medium_max": self.medium_max,
        }


DEFAULT_DIFFICULTY_BOUNDARIES = DifficultyBoundaries(
    short_max=2.0,
    medium_max=15.0,
)


@dataclass
class MazeCandidateIndex:
    maze: np.ndarray
    candidates: tuple[TaskCandidate, ...]
    source_data: dict[
        Position,
        tuple[dict[Position, int], dict[Position, int]],
    ] = field(default_factory=dict)
    path_cache: dict[TaskKey, PathCollection] = field(
        default_factory=dict,
    )
    similarity_cache: dict[tuple[TaskKey, TaskKey], float] = field(
        default_factory=dict,
    )

    def candidates_by_bucket(self) -> dict[str, tuple[TaskCandidate, ...]]:
        return {
            bucket: tuple(
                candidate
                for candidate in self.candidates
                if candidate.bucket == bucket
            )
            for bucket in BUCKETS
        }

    def source_analysis(
        self,
        source: Position,
    ) -> tuple[dict[Position, int], dict[Position, int]]:
        if source not in self.source_data:
            self.source_data[source] = _shortest_distances_and_counts(
                self.maze,
                source,
            )
        return self.source_data[source]

    def _shortest_path_successors(
        self,
        candidate: TaskCandidate,
    ) -> tuple[dict[Position, tuple[Position, ...]], dict[Position, int]]:
        start_distances, _ = self.source_analysis(candidate.start)
        goal_distances, goal_counts = self.source_analysis(candidate.goal)
        shortest_length = start_distances[candidate.goal]
        successors: dict[Position, tuple[Position, ...]] = {}

        for position, distance_from_start in start_distances.items():
            row, col = position
            next_positions = []
            for next_position in (
                (row - 1, col),
                (row + 1, col),
                (row, col - 1),
                (row, col + 1),
            ):
                if next_position not in goal_distances:
                    continue
                if (
                    distance_from_start
                    + 1
                    + goal_distances[next_position]
                    != shortest_length
                ):
                    continue
                next_positions.append(next_position)
            if next_positions:
                successors[position] = tuple(next_positions)

        return successors, goal_counts

    @staticmethod
    def _path_to_edges(
        positions: list[Position],
    ) -> frozenset[Edge]:
        return frozenset(
            (positions[index], positions[index + 1])
            for index in range(len(positions) - 1)
        )

    def _enumerate_shortest_paths(
        self,
        candidate: TaskCandidate,
        successors: dict[Position, tuple[Position, ...]],
    ) -> tuple[frozenset[Edge], ...]:
        paths: list[frozenset[Edge]] = []

        def visit(position: Position, positions: list[Position]) -> None:
            if position == candidate.goal:
                paths.append(self._path_to_edges(positions))
                return
            for next_position in successors[position]:
                positions.append(next_position)
                visit(next_position, positions)
                positions.pop()

        visit(candidate.start, [candidate.start])
        return tuple(paths)

    def _sample_shortest_paths(
        self,
        candidate: TaskCandidate,
        successors: dict[Position, tuple[Position, ...]],
        goal_counts: dict[Position, int],
        rng: np.random.Generator,
        sample_count: int,
    ) -> tuple[frozenset[Edge], ...]:
        paths: list[frozenset[Edge]] = []
        for _ in range(sample_count):
            position = candidate.start
            positions = [position]
            while position != candidate.goal:
                next_positions = successors[position]
                weights = [goal_counts[next_position] for next_position in next_positions]
                draw = int(rng.integers(sum(weights)))
                for next_position, weight in zip(next_positions, weights):
                    if draw < weight:
                        position = next_position
                        positions.append(position)
                        break
                    draw -= weight
            paths.append(self._path_to_edges(positions))
        return tuple(paths)

    def paths(self, candidate: TaskCandidate) -> PathCollection:
        if candidate.key in self.path_cache:
            return self.path_cache[candidate.key]

        successors, goal_counts = self._shortest_path_successors(candidate)
        total_paths = int(goal_counts[candidate.start])
        if total_paths <= MAX_EXACT_PATHS:
            paths = self._enumerate_shortest_paths(candidate, successors)
            collection = PathCollection(
                paths=paths,
                exact=True,
                total_paths=total_paths,
            )
        else:
            seed = np.random.SeedSequence(
                [
                    candidate.start[0],
                    candidate.start[1],
                    candidate.goal[0],
                    candidate.goal[1],
                ]
            )
            paths = self._sample_shortest_paths(
                candidate,
                successors,
                goal_counts,
                np.random.default_rng(seed),
                PATH_SAMPLE_COUNT,
            )
            collection = PathCollection(
                paths=paths,
                exact=False,
                total_paths=total_paths,
            )

        self.path_cache[candidate.key] = collection
        return collection

    def similarity(
        self,
        first: TaskCandidate,
        second: TaskCandidate,
    ) -> float:
        cache_key = tuple(sorted((first.key, second.key)))
        if cache_key not in self.similarity_cache:
            first_collection = self.paths(first)
            second_collection = self.paths(second)
            if (
                first_collection.exact
                and second_collection.exact
                and len(first_collection.paths) * len(second_collection.paths)
                <= MAX_EXACT_PAIR_COUNT
            ):
                similarity = expected_pairwise_directed_path_jaccard(
                    first_collection.paths,
                    second_collection.paths,
                )
            else:
                similarity = _sampled_expected_pairwise_similarity(
                    first,
                    second,
                    first_collection,
                    second_collection,
                )
            self.similarity_cache[cache_key] = similarity
        return self.similarity_cache[cache_key]


@dataclass
class SearchResult:
    selected: list[TaskCandidate]
    completed: bool
    candidate_attempts: int
    backtracking_events: int
    budget_exhausted: bool


def _empty_assignment() -> dict[str, int]:
    return {bucket: 0 for bucket in BUCKETS}


def generate_stratified_dataset(
    seeds: np.ndarray,
    height: int,
    width: int,
    wall_probability: float,
    tasks_per_maze: int,
    min_path_length: int,
    dedupe_threshold: float = 0.7,
    bucket_budgets: Mapping[str, float] | None = None,
    random_seed: int = 0,
    selection_attempts: int = 8,
    local_search_budget: int = 2_000,
    difficulty_boundaries: DifficultyBoundaries | None = None,
    progress_callback: Callable[[dict[str, object]], None] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    """Generate layouts and their complete stratified task assignments.

    Each candidate maze is indexed once. The complete per-layout bucket
    assignment is selected before the maze is accepted, and the selected task
    pairs are retained for output. Selection retries reuse the same index;
    failed candidates are regenerated rather than retried in a later pass.
    """
    if tasks_per_maze <= 0:
        raise ValueError("tasks_per_maze must be positive.")
    if not 0 < dedupe_threshold <= 1:
        raise ValueError("dedupe_threshold must be in (0, 1].")
    if selection_attempts <= 0 or local_search_budget <= 0:
        raise ValueError(
            "Selection attempts and search budget must be positive."
        )
    budgets = validate_bucket_budgets(
        bucket_budgets or DEFAULT_BUCKET_BUDGETS,
    )
    per_layout_quotas = bucket_quotas(tasks_per_maze, budgets)
    total_quotas = {
        bucket: per_layout_quotas[bucket] * len(seeds)
        for bucket in BUCKETS
    }
    boundaries = difficulty_boundaries or DEFAULT_DIFFICULTY_BOUNDARIES

    layouts = np.empty(
        (len(seeds), height, width),
        dtype=np.uint8,
    )
    task_count = len(seeds) * tasks_per_maze
    layout_indices = np.empty(task_count, dtype=np.int64)
    starts = np.empty((task_count, 2), dtype=np.int64)
    goals = np.empty((task_count, 2), dtype=np.int64)
    candidate_counts_by_layout: list[dict[str, int]] = []
    selection_attempts_by_layout: list[int] = []
    generation_attempts_by_layout: list[int] = []
    bucket_counts = _empty_assignment()
    started_at = time.perf_counter()

    def report(
        phase: str,
        layouts_completed: int,
        *,
        selection_attempt: int | None = None,
        reason: str | None = None,
    ) -> None:
        if progress_callback is None:
            return
        event: dict[str, object] = {
            "phase": phase,
            "layouts_completed": layouts_completed,
            "layout_count": len(seeds),
            "bucket_counts": dict(bucket_counts),
            "bucket_quotas": dict(total_quotas),
            "selection_attempts": selection_attempts,
        }
        if selection_attempt is not None:
            event["selection_attempt"] = selection_attempt
        if reason is not None:
            event["reason"] = reason
        progress_callback(event)

    for layout_index, seed in enumerate(seeds):
        rng = np.random.default_rng(int(seed))
        selected: list[TaskCandidate] | None = None
        generation_attempt = 0
        for generation_attempt in range(1, 1001):
            maze, _, _ = generate_valid_maze(
                height=height,
                width=width,
                wall_probability=wall_probability,
                rng=rng,
                min_path_length=min_path_length,
            )
            index = build_candidate_index(
                maze,
                min_path_length=min_path_length,
            )
            apply_difficulty_buckets([index], boundaries)
            capacities = _candidate_bucket_counts(index)
            if any(
                capacities[bucket] < per_layout_quotas[bucket]
                for bucket in BUCKETS
            ):
                continue

            for selection_attempt in range(1, selection_attempts + 1):
                selection_rng = np.random.default_rng(
                    np.random.SeedSequence(
                        [
                            random_seed,
                            layout_index,
                            int(seed),
                            selection_attempt,
                        ]
                    )
                )
                result = _select_tasks_for_maze(
                    index,
                    per_layout_quotas,
                    threshold=dedupe_threshold,
                    rng=selection_rng,
                    search_budget=local_search_budget,
                )
                report(
                    "selection",
                    layout_index,
                    selection_attempt=selection_attempt,
                    reason=None if result.completed else "retrying layout",
                )
                if result.completed:
                    selected = result.selected
                    selection_attempts_by_layout.append(selection_attempt)
                    break
            if selected is not None:
                break

        if selected is None:
            raise RuntimeError(
                "Could not generate a layout with the requested stratified "
                "task quotas after generation and selection retries."
            )

        layouts[layout_index] = maze
        candidate_counts_by_layout.append(capacities)
        generation_attempts_by_layout.append(generation_attempt)
        task_start = layout_index * tasks_per_maze
        for offset, candidate in enumerate(selected):
            output_index = task_start + offset
            layout_indices[output_index] = layout_index
            starts[output_index] = candidate.start
            goals[output_index] = candidate.goal
            bucket_counts[candidate.bucket] += 1
        report("generation", layout_index + 1)

    metadata: dict[str, object] = {
        "dedupe_metric": "expected_pairwise_directed_path_jaccard",
        "dedupe_scope": DEDUPE_SCOPE,
        "dedupe_threshold": dedupe_threshold,
        "bucket_budgets": budgets,
        "bucket_quotas": total_quotas,
        "bucket_counts": bucket_counts,
        "candidate_counts": {
            bucket: sum(
                counts[bucket] for counts in candidate_counts_by_layout
            )
            for bucket in BUCKETS
        },
        "candidate_counts_by_layout": candidate_counts_by_layout,
        "difficulty_boundaries": boundaries.as_dict(),
        "difficulty_boundaries_source": "configured absolute lengths",
        "tasks_per_maze_target": tasks_per_maze,
        "contributions_by_layout": [tasks_per_maze] * len(seeds),
        "bucket_quotas_per_layout": per_layout_quotas,
        "selection_attempts": selection_attempts_by_layout,
        "generation_attempts": generation_attempts_by_layout,
        "local_search_budget": local_search_budget,
        "path_sample_count": PATH_SAMPLE_COUNT,
        "max_exact_paths": MAX_EXACT_PATHS,
        "max_exact_pair_count": MAX_EXACT_PAIR_COUNT,
        "generation_runtime_seconds": time.perf_counter() - started_at,
    }
    return layouts, layout_indices, starts, goals, metadata


def build_candidate_index(
    maze: np.ndarray,
    *,
    min_path_length: int,
    excluded_tasks: set[TaskKey] | None = None,
) -> MazeCandidateIndex:
    """Build a lightweight task index using one BFS per traversable source."""
    excluded_tasks = excluded_tasks or set()
    candidates: list[TaskCandidate] = []
    source_data: dict[
        Position,
        tuple[dict[Position, int], dict[Position, int]],
    ] = {}

    for start in traversable_positions(maze):
        distances, counts = _shortest_distances_and_counts(maze, start)
        source_data[start] = distances, counts

        for goal, distance in distances.items():
            if goal == start or distance < min_path_length:
                continue
            if (start, goal) in excluded_tasks:
                continue
            candidates.append(
                TaskCandidate(
                    start=start,
                    goal=goal,
                    distance=distance,
                )
            )

    return MazeCandidateIndex(
        maze=maze,
        candidates=tuple(candidates),
        source_data=source_data,
    )


def derive_difficulty_boundaries(
    indexes: Iterable[MazeCandidateIndex],
    bucket_budgets: Mapping[str, float],
) -> DifficultyBoundaries:
    distances = [
        candidate.distance
        for index in indexes
        for candidate in index.candidates
    ]
    if not distances:
        raise ValueError("No eligible task candidates were found.")

    return DifficultyBoundaries(
        short_max=float(
            np.quantile(distances, bucket_budgets["short"])
        ),
        medium_max=float(
            np.quantile(
                distances,
                bucket_budgets["short"]
                + bucket_budgets["medium"],
            )
        ),
    )


def apply_difficulty_buckets(
    indexes: Iterable[MazeCandidateIndex],
    boundaries: DifficultyBoundaries,
) -> None:
    for index in indexes:
        index.candidates = tuple(
            TaskCandidate(
                start=candidate.start,
                goal=candidate.goal,
                distance=candidate.distance,
                bucket=boundaries.bucket_for(candidate.distance),
            )
            for candidate in index.candidates
        )


def _candidate_bucket_counts(
    index: MazeCandidateIndex,
) -> dict[str, int]:
    """Return bucket capacities without creating per-bucket candidate tuples."""
    counts = _empty_assignment()
    for candidate in index.candidates:
        counts[candidate.bucket] += 1
    return counts


def validate_bucket_budgets(
    bucket_budgets: Mapping[str, float],
) -> dict[str, float]:
    budgets = {
        bucket: float(bucket_budgets[bucket])
        for bucket in BUCKETS
    }
    if any(value < 0 for value in budgets.values()):
        raise ValueError("Bucket budgets must be non-negative.")
    if not np.isclose(sum(budgets.values()), 1.0):
        raise ValueError("Bucket budgets must sum to 1.0.")
    return budgets


def bucket_quotas(
    total_slots: int,
    bucket_budgets: Mapping[str, float],
) -> dict[str, int]:
    budgets = validate_bucket_budgets(bucket_budgets)
    raw = {
        bucket: total_slots * budgets[bucket]
        for bucket in BUCKETS
    }
    quotas = {bucket: int(np.floor(value)) for bucket, value in raw.items()}
    remaining = total_slots - sum(quotas.values())

    order = sorted(
        BUCKETS,
        key=lambda bucket: (
            -(raw[bucket] - quotas[bucket]),
            BUCKETS.index(bucket),
        ),
    )
    for bucket in order[:remaining]:
        quotas[bucket] += 1
    return quotas


def directed_path_jaccard(
    first: Iterable[Edge],
    second: Iterable[Edge],
) -> float:
    """Return Jaccard similarity while preserving edge direction."""
    first_edges = frozenset(first)
    second_edges = frozenset(second)
    union_size = len(first_edges | second_edges)
    if union_size == 0:
        return 1.0
    return len(first_edges & second_edges) / union_size


def expected_pairwise_directed_path_jaccard(
    first_paths: Iterable[Iterable[Edge]],
    second_paths: Iterable[Iterable[Edge]],
) -> float:
    """Return the expected directed path-Jaccard over path pairs.

    Each input path collection is treated as a uniform distribution.  The
    caller may therefore pass all shortest paths for an exact result or
    independent uniform samples for an unbiased estimate of that result.
    """
    first_paths = tuple(frozenset(path) for path in first_paths)
    second_paths = tuple(frozenset(path) for path in second_paths)
    if not first_paths or not second_paths:
        raise ValueError("Both path collections must be non-empty.")
    return sum(
        directed_path_jaccard(first, second)
        for first in first_paths
        for second in second_paths
    ) / (len(first_paths) * len(second_paths))


def _sampled_expected_pairwise_similarity(
    first: TaskCandidate,
    second: TaskCandidate,
    first_collection: PathCollection,
    second_collection: PathCollection,
) -> float:
    if first.key == second.key:
        return 1.0

    if first.key > second.key:
        first, second = second, first
        first_collection, second_collection = (
            second_collection,
            first_collection,
        )
    ordered = (first, second)
    rng = np.random.default_rng(
        np.random.SeedSequence(
            [
                coordinate
                for candidate in ordered
                for position in candidate.key
                for coordinate in position
            ]
        )
    )

    def draw_paths(
        collection: PathCollection,
    ) -> tuple[frozenset[Edge], ...]:
        if not collection.exact:
            return collection.paths
        indices = rng.integers(
            len(collection.paths),
            size=PATH_SAMPLE_COUNT,
        )
        return tuple(collection.paths[int(index)] for index in indices)

    return expected_pairwise_directed_path_jaccard(
        draw_paths(first_collection),
        draw_paths(second_collection),
    )


def _select_tasks_for_maze(
    index: MazeCandidateIndex,
    assignment: Mapping[str, int],
    *,
    threshold: float,
    rng: np.random.Generator,
    search_budget: int = 2_000,
) -> SearchResult:
    candidates_by_bucket = index.candidates_by_bucket()
    choices: list[list[TaskCandidate]] = []
    for bucket in ALLOCATION_ORDER:
        bucket_choices = list(candidates_by_bucket[bucket])
        rng.shuffle(bucket_choices)
        choices.extend(
            [bucket_choices] * assignment[bucket]
        )

    selected: list[TaskCandidate] = []
    selected_by_bucket: dict[str, list[TaskCandidate]] = {
        bucket: []
        for bucket in BUCKETS
    }
    best_selected: list[TaskCandidate] = []
    selected_keys: set[TaskKey] = set()
    candidate_attempts = 0
    backtracking_events = 0
    budget_exhausted = False

    def visit(slot_index: int) -> bool:
        nonlocal candidate_attempts
        nonlocal backtracking_events
        nonlocal best_selected
        nonlocal budget_exhausted
        if slot_index == len(choices):
            return True

        for candidate in choices[slot_index]:
            candidate_attempts += 1
            if candidate_attempts > search_budget:
                budget_exhausted = True
                return False

            if len(selected) > len(best_selected):
                best_selected = list(selected)

            if candidate.key in selected_keys:
                continue
            if any(
                index.similarity(candidate, previous) >= threshold
                for previous in selected_by_bucket[candidate.bucket]
            ):
                continue

            selected.append(candidate)
            selected_by_bucket[candidate.bucket].append(candidate)
            selected_keys.add(candidate.key)
            if visit(slot_index + 1):
                return True
            selected.pop()
            selected_by_bucket[candidate.bucket].pop()
            selected_keys.remove(candidate.key)
            backtracking_events += 1

            if budget_exhausted:
                return False

        return False

    completed = visit(0)
    if len(selected) > len(best_selected):
        best_selected = list(selected)

    return SearchResult(
        selected=list(selected if completed else best_selected),
        completed=completed,
        candidate_attempts=candidate_attempts,
        backtracking_events=backtracking_events,
        budget_exhausted=budget_exhausted,
    )
