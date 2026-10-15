"""Difficulty-stratified task generation for maze datasets."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Iterable, Mapping

import numpy as np

from maze_rl.maze.pathfinding import (
    Position,
    traversable_positions,
)


BUCKETS = ("short", "medium", "long")
ALLOCATION_ORDER = ("long", "medium", "short")
DEFAULT_BUCKET_BUDGETS = {
    "short": 0.20,
    "medium": 0.40,
    "long": 0.40,
}
Edge = tuple[Position, Position]
TaskKey = tuple[Position, Position]


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


@dataclass
class MazeCandidateIndex:
    maze: np.ndarray
    candidates: tuple[TaskCandidate, ...]
    source_data: dict[
        Position,
        tuple[dict[Position, int], dict[Position, int]],
    ] = field(default_factory=dict)
    profile_cache: dict[TaskKey, dict[Edge, float]] = field(
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

    def profile(self, candidate: TaskCandidate) -> dict[Edge, float]:
        if candidate.key in self.profile_cache:
            return self.profile_cache[candidate.key]

        start_distances, start_counts = self.source_analysis(
            candidate.start,
        )
        goal_distances, goal_counts = self.source_analysis(
            candidate.goal,
        )
        shortest_length = start_distances[candidate.goal]
        total_paths = start_counts[candidate.goal]
        profile: dict[Edge, float] = {}

        for position, distance_from_start in start_distances.items():
            row, col = position
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

                profile[(position, next_position)] = (
                    start_counts[position]
                    * goal_counts[next_position]
                    / total_paths
                )

        self.profile_cache[candidate.key] = profile
        return profile

    def similarity(
        self,
        first: TaskCandidate,
        second: TaskCandidate,
    ) -> float:
        cache_key = tuple(sorted((first.key, second.key)))
        if cache_key not in self.similarity_cache:
            self.similarity_cache[cache_key] = directed_weighted_jaccard(
                self.profile(first),
                self.profile(second),
            )
        return self.similarity_cache[cache_key]


class QuotaAllocationError(RuntimeError):
    """Raised when requested bucket quotas cannot fit candidate capacity."""

    def __init__(
        self,
        shortfalls: Mapping[str, int],
    ) -> None:
        self.shortfalls = dict(shortfalls)
        details = ", ".join(
            f"{bucket}={count}"
            for bucket, count in self.shortfalls.items()
            if count
        )
        super().__init__(
            "Could not allocate requested stratified bucket quotas"
            + (f" ({details})" if details else ".")
        )


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


def allocate_bucket_slots(
    indexes: list[MazeCandidateIndex],
    *,
    tasks_per_maze: int,
    quotas: Mapping[str, int],
    rng: np.random.Generator,
) -> list[dict[str, int]]:
    assignments = [
        {bucket: 0 for bucket in BUCKETS}
        for _ in indexes
    ]
    candidate_counts = [
        {
            bucket: len(index.candidates_by_bucket()[bucket])
            for bucket in BUCKETS
        }
        for index in indexes
    ]
    tie_breakers = rng.random((len(indexes), len(BUCKETS)))
    shortfalls = {bucket: 0 for bucket in BUCKETS}

    for bucket in ALLOCATION_ORDER:
        for _ in range(int(quotas[bucket])):
            eligible = [
                layout_index
                for layout_index, assignment in enumerate(assignments)
                if sum(assignment.values()) < tasks_per_maze
                and assignment[bucket] < candidate_counts[layout_index][bucket]
            ]
            if not eligible:
                shortfalls[bucket] += 1
                continue

            layout_index = max(
                eligible,
                key=lambda candidate_index: (
                    candidate_counts[candidate_index][bucket]
                    - assignments[candidate_index][bucket],
                    tie_breakers[candidate_index][BUCKETS.index(bucket)],
                ),
            )
            assignments[layout_index][bucket] += 1

    if any(shortfalls.values()):
        raise QuotaAllocationError(shortfalls)
    return assignments


def directed_weighted_jaccard(
    first: Mapping[Edge, float],
    second: Mapping[Edge, float],
) -> float:
    """Return weighted Jaccard similarity for directed sparse profiles."""
    keys = first.keys() | second.keys()
    numerator = sum(min(first.get(edge, 0.0), second.get(edge, 0.0)) for edge in keys)
    denominator = sum(max(first.get(edge, 0.0), second.get(edge, 0.0)) for edge in keys)
    return numerator / denominator if denominator else 0.0


def _select_tasks_for_maze(
    index: MazeCandidateIndex,
    assignment: Mapping[str, int],
    *,
    threshold: float,
    rng: np.random.Generator,
    max_search_nodes: int = 10_000,
) -> list[TaskCandidate] | None:
    candidates_by_bucket = index.candidates_by_bucket()
    choices: list[list[TaskCandidate]] = []
    for bucket in ALLOCATION_ORDER[::-1]:
        bucket_choices = list(candidates_by_bucket[bucket])
        rng.shuffle(bucket_choices)
        choices.extend(
            [bucket_choices] * assignment[bucket]
        )

    selected: list[TaskCandidate] = []
    selected_keys: set[TaskKey] = set()
    search_nodes = 0

    def visit(slot_index: int) -> bool:
        nonlocal search_nodes
        search_nodes += 1
        if search_nodes > max_search_nodes:
            return False
        if slot_index == len(choices):
            return True

        for candidate in choices[slot_index]:
            if candidate.key in selected_keys:
                continue
            if any(
                index.similarity(candidate, previous) >= threshold
                for previous in selected
            ):
                continue

            selected.append(candidate)
            selected_keys.add(candidate.key)
            if visit(slot_index + 1):
                return True
            selected.pop()
            selected_keys.remove(candidate.key)
        return False

    return selected if visit(0) else None


def threshold_schedule(threshold: float) -> tuple[float, ...]:
    if not 0 < threshold <= 1:
        raise ValueError("The deduplication threshold must be in (0, 1].")
    return tuple(
        dict.fromkeys(
            value
            for value in (threshold, 0.90, 0.95, 1.00)
            if value >= threshold
        )
    )


def generate_stratified_tasks_for_layouts(
    layouts: np.ndarray,
    seeds: np.ndarray,
    *,
    tasks_per_maze: int,
    min_path_length: int,
    dedupe_threshold: float = 0.75,
    bucket_budgets: Mapping[str, float] | None = None,
    random_seed: int = 0,
    excluded_tasks_by_layout: dict[int, set[TaskKey]] | None = None,
    allocation_attempts: int = 8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    """Select reproducible, difficulty-balanced tasks for each layout."""
    if tasks_per_maze <= 0:
        raise ValueError("tasks_per_maze must be positive.")
    budgets = validate_bucket_budgets(
        bucket_budgets or DEFAULT_BUCKET_BUDGETS,
    )
    excluded_tasks_by_layout = excluded_tasks_by_layout or {}
    indexes = [
        build_candidate_index(
            maze,
            min_path_length=min_path_length,
            excluded_tasks=excluded_tasks_by_layout.get(layout_index),
        )
        for layout_index, maze in enumerate(layouts)
    ]
    insufficient = [
        layout_index
        for layout_index, index in enumerate(indexes)
        if len(index.candidates) < tasks_per_maze
    ]
    if insufficient:
        raise RuntimeError(
            "Layouts without enough eligible tasks: "
            + ", ".join(str(index) for index in insufficient[:10])
        )

    boundaries = derive_difficulty_boundaries(indexes, budgets)
    apply_difficulty_buckets(indexes, boundaries)
    candidate_counts_by_layout = [
        {
            bucket: len(index.candidates_by_bucket()[bucket])
            for bucket in BUCKETS
        }
        for index in indexes
    ]
    candidate_counts = {
        bucket: sum(
            counts[bucket]
            for counts in candidate_counts_by_layout
        )
        for bucket in BUCKETS
    }
    quotas = bucket_quotas(len(layouts) * tasks_per_maze, budgets)
    last_error: Exception | None = None

    for threshold in threshold_schedule(dedupe_threshold):
        for attempt in range(allocation_attempts):
            allocation_rng = np.random.default_rng(
                np.random.SeedSequence(
                    [random_seed, int(round(threshold * 1000)), attempt]
                )
            )
            try:
                assignments = allocate_bucket_slots(
                    indexes,
                    tasks_per_maze=tasks_per_maze,
                    quotas=quotas,
                    rng=allocation_rng,
                )
            except QuotaAllocationError as error:
                last_error = error
                continue

            selected_by_layout: list[list[TaskCandidate]] = []
            failed = False
            for layout_index, (index, assignment, seed) in enumerate(
                zip(indexes, assignments, seeds)
            ):
                selection_rng = np.random.default_rng(
                    np.random.SeedSequence(
                        [random_seed, attempt, layout_index, int(seed)]
                    )
                )
                selected = _select_tasks_for_maze(
                    index,
                    assignment,
                    threshold=threshold,
                    rng=selection_rng,
                )
                if selected is None:
                    failed = True
                    break
                selected_by_layout.append(selected)

            if failed:
                continue

            task_count = len(layouts) * tasks_per_maze
            layout_indices = np.empty(task_count, dtype=np.int64)
            starts = np.empty((task_count, 2), dtype=np.int64)
            goals = np.empty((task_count, 2), dtype=np.int64)
            bucket_counts = {bucket: 0 for bucket in BUCKETS}
            output_index = 0

            for layout_index, selected in enumerate(selected_by_layout):
                for candidate in selected:
                    layout_indices[output_index] = layout_index
                    starts[output_index] = candidate.start
                    goals[output_index] = candidate.goal
                    bucket_counts[candidate.bucket] += 1
                    output_index += 1

            metadata: dict[str, object] = {
                "dedupe_threshold": threshold,
                "bucket_budgets": budgets,
                "bucket_quotas": quotas,
                "bucket_counts": bucket_counts,
                "candidate_counts": candidate_counts,
                "candidate_counts_by_layout": candidate_counts_by_layout,
                "difficulty_boundaries": boundaries.as_dict(),
                "candidate_distribution": (
                    "all eligible ordered pairs across training layouts"
                ),
            }
            return layout_indices, starts, goals, metadata

    raise RuntimeError(
        "Could not select stratified tasks after retries. "
        f"Requested quotas: {quotas}. Last error: {last_error}"
    )
