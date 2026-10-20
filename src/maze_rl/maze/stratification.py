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
    count_valid_tasks_by_min_path_length,
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


@dataclass
class SearchResult:
    selected: list[TaskCandidate]
    completed: bool
    candidate_attempts: int
    backtracking_events: int
    budget_exhausted: bool


def _unresolved_counts(
    unresolved: Iterable[tuple[int, str]],
) -> dict[str, int]:
    counts = _empty_assignment()
    for _, bucket in unresolved:
        if bucket in counts:
            counts[bucket] += 1
    return counts


def _failure_diagnostics(
    search_metadata: Mapping[str, object],
    unresolved_before_repair: Iterable[tuple[int, str]],
    unresolved_after_repair: Iterable[tuple[int, str]],
    contributions: list[int],
    *,
    min_tasks_per_maze: int,
    max_tasks_per_maze: int,
    repair_metadata: Mapping[str, object],
) -> dict[str, object]:
    budget_exhausted_layouts = [
        int(layout_index)
        for layout_index in search_metadata.get(
            "budget_exhausted_layouts",
            search_metadata.get("budget_exhausted", []),
        )
    ]
    infeasible_assignment_layouts = [
        int(layout_index)
        for layout_index in search_metadata.get(
            "proven_infeasible_layouts",
            search_metadata.get("proven_infeasible", []),
        )
    ]
    unresolved_before_repair = list(unresolved_before_repair)
    unresolved_after_repair = list(unresolved_after_repair)
    below_min_layouts = [
        layout_index
        for layout_index, contribution in enumerate(contributions)
        if contribution < min_tasks_per_maze
    ]
    above_max_layouts = [
        layout_index
        for layout_index, contribution in enumerate(contributions)
        if contribution > max_tasks_per_maze
    ]
    return {
        "budget_exhausted_layouts": budget_exhausted_layouts,
        "budget_exhausted_count": len(budget_exhausted_layouts),
        "infeasible_assignment_layouts": infeasible_assignment_layouts,
        "infeasible_assignment_count": len(infeasible_assignment_layouts),
        "unresolved_before_repair_by_bucket": _unresolved_counts(
            unresolved_before_repair,
        ),
        "unresolved_after_repair_by_bucket": _unresolved_counts(
            unresolved_after_repair,
        ),
        "contribution_below_min_layouts": below_min_layouts,
        "contribution_below_min_count": len(below_min_layouts),
        "contribution_above_max_layouts": above_max_layouts,
        "contribution_above_max_count": len(above_max_layouts),
        "repair_attempts": int(repair_metadata.get("repair_attempts", 0)),
        "successful_reallocations": int(
            repair_metadata.get("successful_reallocations", 0)
        ),
    }


def format_failure_diagnostics(
    diagnostics: Mapping[str, object],
    *,
    layout_preview_limit: int = 5,
) -> str:
    def preview(key: str) -> str:
        values = diagnostics.get(key, [])
        if not isinstance(values, list) or not values:
            return "none"
        shown = ",".join(str(value) for value in values[:layout_preview_limit])
        if len(values) > layout_preview_limit:
            shown += ",..."
        return shown

    before = diagnostics.get("unresolved_before_repair_by_bucket", {})
    after = diagnostics.get("unresolved_after_repair_by_bucket", {})
    return "\n".join((
        (
            "budget-exhausted layouts="
            f"{diagnostics.get('budget_exhausted_count', 0)}"
            f" [{preview('budget_exhausted_layouts')}]"
        ),
        (
            "infeasible current assignments="
            f"{diagnostics.get('infeasible_assignment_count', 0)}"
            f" [{preview('infeasible_assignment_layouts')}]"
        ),
        (
            "below-min layouts="
            f"{diagnostics.get('contribution_below_min_count', 0)}"
            f" [{preview('contribution_below_min_layouts')}]"
        ),
        (
            "unresolved before repair="
            f"S {before.get('short', 0)}/M {before.get('medium', 0)}"
            f"/L {before.get('long', 0)}"
        ),
        (
            "unresolved after repair="
            f"S {after.get('short', 0)}/M {after.get('medium', 0)}"
            f"/L {after.get('long', 0)}"
        ),
        (
            "repairs="
            f"{diagnostics.get('successful_reallocations', 0)}"
            f"/{diagnostics.get('repair_attempts', 0)}"
        ),
    ))


def _empty_assignment() -> dict[str, int]:
    return {bucket: 0 for bucket in BUCKETS}


def _assignment_for_tasks(
    tasks: Iterable[TaskCandidate],
) -> dict[str, int]:
    assignment = _empty_assignment()
    for task in tasks:
        assignment[task.bucket] += 1
    return assignment


def generate_layouts(
    seeds: np.ndarray,
    height: int,
    width: int,
    wall_probability: float,
    min_path_length: int,
    minimum_eligible_tasks: int = 1,
    minimum_long_eligible_tasks: int = 0,
    long_path_min_length: int | None = None,
    required_bucket_counts: Mapping[str, int] | None = None,
    dedupe_threshold: float = 0.7,
    selection_random_seed: int = 0,
    selection_search_budget: int = 2_000,
    difficulty_boundaries: DifficultyBoundaries | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> np.ndarray:
    """Generate layouts with capacity for later stratified selection.

    The optional long-task requirement is checked before task selection, so
    layouts are not rejected later merely because their geometry cannot
    support the requested long-task quota. When fixed bucket counts are
    supplied, the layout must also pass the configured non-duplicate long-task
    selection search before it is accepted.
    """
    if minimum_eligible_tasks < 0:
        raise ValueError("minimum_eligible_tasks must be non-negative.")
    if minimum_long_eligible_tasks < 0:
        raise ValueError(
            "minimum_long_eligible_tasks must be non-negative."
        )
    if minimum_long_eligible_tasks and long_path_min_length is None:
        raise ValueError(
            "long_path_min_length is required when long-task capacity "
            "is requested."
        )
    if not 0 < dedupe_threshold <= 1:
        raise ValueError("dedupe_threshold must be in (0, 1].")
    if selection_search_budget <= 0:
        raise ValueError("selection_search_budget must be positive.")

    fixed_bucket_counts: dict[str, int] | None = None
    if required_bucket_counts is not None:
        fixed_bucket_counts = {
            bucket: int(required_bucket_counts[bucket])
            for bucket in BUCKETS
        }
        if any(value < 0 for value in fixed_bucket_counts.values()):
            raise ValueError("required bucket counts must be non-negative.")
        if not sum(fixed_bucket_counts.values()):
            raise ValueError("required bucket counts must not be empty.")
        if minimum_eligible_tasks > sum(fixed_bucket_counts.values()):
            raise ValueError(
                "minimum_eligible_tasks cannot exceed the fixed bucket "
                "task count."
            )
        boundaries = difficulty_boundaries or DEFAULT_DIFFICULTY_BOUNDARIES

    layouts = np.empty(
        (len(seeds), height, width),
        dtype=np.uint8,
    )

    if progress_callback is not None:
        progress_callback(0, len(seeds))
    for layout_index, seed in enumerate(seeds):
        rng = np.random.default_rng(int(seed))
        for _ in range(1000):
            maze, _, _ = generate_valid_maze(
                height=height,
                width=width,
                wall_probability=wall_probability,
                rng=rng,
                min_path_length=min_path_length,
            )
            if fixed_bucket_counts is None:
                thresholds = [min_path_length]
                if minimum_long_eligible_tasks:
                    thresholds.append(int(long_path_min_length))
                capacities = count_valid_tasks_by_min_path_length(
                    maze,
                    tuple(thresholds),
                )

                if capacities[min_path_length] < minimum_eligible_tasks:
                    continue

                if (
                    minimum_long_eligible_tasks
                    and capacities[int(long_path_min_length)]
                    < minimum_long_eligible_tasks
                ):
                    continue
            else:
                index = build_candidate_index(
                    maze,
                    min_path_length=min_path_length,
                )
                apply_difficulty_buckets([index], boundaries)
                capacities = _candidate_bucket_counts(index)
                if any(
                    capacities[bucket] < fixed_bucket_counts[bucket]
                    for bucket in BUCKETS
                ):
                    continue

                selection_rng = np.random.default_rng(
                    np.random.SeedSequence(
                        [selection_random_seed, layout_index, int(seed)]
                    )
                )
                long_selection = _select_tasks_for_maze(
                    index,
                    {
                        "short": 0,
                        "medium": 0,
                        "long": fixed_bucket_counts["long"],
                    },
                    threshold=dedupe_threshold,
                    rng=selection_rng,
                    search_budget=selection_search_budget,
                )
                if not long_selection.completed:
                    continue

            layouts[layout_index] = maze
            if progress_callback is not None:
                progress_callback(layout_index + 1, len(seeds))
            break
        else:
            raise RuntimeError(
                "Could not generate a layout with enough eligible tasks."
            )

    return layouts


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


def allocate_bucket_slots(
    indexes: list[MazeCandidateIndex] | None = None,
    *,
    tasks_per_maze: int,
    quotas: Mapping[str, int],
    rng: np.random.Generator,
    min_tasks_per_maze: int = 1,
    max_tasks_per_maze: int | None = None,
    candidate_counts_by_layout: list[Mapping[str, int]] | None = None,
    progress_callback: Callable[[dict[str, object]], None] | None = None,
) -> list[dict[str, int]]:
    """Allocate bucket slots while preserving feasibility invariants.

    The allocation is deterministic for a seeded RNG.  Randomness is used
    only to break otherwise equal choices; capacity and lower/upper-bound
    feasibility determine which layouts are eligible.
    """
    if max_tasks_per_maze is None:
        max_tasks_per_maze = tasks_per_maze
    if not 0 < min_tasks_per_maze <= max_tasks_per_maze:
        raise ValueError(
            "Expected min_tasks_per_maze <= max_tasks_per_maze."
        )
    allocation_max_tasks = tasks_per_maze

    if candidate_counts_by_layout is None:
        if indexes is None:
            raise ValueError(
                "Provide indexes or candidate_counts_by_layout."
            )
        candidate_counts_by_layout = [
            _candidate_bucket_counts(index)
            for index in indexes
        ]

    assignments = [
        {bucket: 0 for bucket in BUCKETS}
        for _ in candidate_counts_by_layout
    ]
    candidate_counts = candidate_counts_by_layout
    tie_breakers = rng.random((len(candidate_counts), len(BUCKETS)))
    shortfalls = {bucket: 0 for bucket in BUCKETS}

    for layout_index in range(len(assignments)):
        remaining_capacity = sum(
            candidate_counts[layout_index][bucket]
            for bucket in BUCKETS
        )
        if remaining_capacity < min_tasks_per_maze:
            raise QuotaAllocationError(
                {"contribution_bounds": 1}
            )

    for bucket in BUCKETS:
        if int(quotas[bucket]) > sum(
            candidate_counts[layout_index][bucket]
            for layout_index in range(len(candidate_counts))
        ):
            raise QuotaAllocationError(
                {bucket: int(quotas[bucket])}
            )

    remaining_total = sum(
        int(quotas[bucket]) for bucket in BUCKETS
    )
    minimum_needed = len(candidate_counts) * min_tasks_per_maze
    maximum_available = sum(
        min(
            allocation_max_tasks,
            sum(candidate_counts[layout_index].values()),
        )
        for layout_index in range(len(candidate_counts))
    )
    if not minimum_needed <= remaining_total <= maximum_available:
        raise QuotaAllocationError(
            {"contribution_bounds": len(candidate_counts)}
        )

    for bucket in ALLOCATION_ORDER:
        bucket_total = int(quotas[bucket])
        report_interval = max(bucket_total // 100, 1)
        for bucket_index in range(bucket_total):
            eligible = []
            for layout_index, assignment in enumerate(assignments):
                if sum(assignment.values()) >= allocation_max_tasks:
                    continue
                if assignment[bucket] >= candidate_counts[layout_index][bucket]:
                    continue
                current = sum(assignment.values())
                minimum_needed_after = minimum_needed - int(
                    current < min_tasks_per_maze
                )
                if remaining_total - 1 < minimum_needed_after:
                    continue
                eligible.append(layout_index)
            if not eligible:
                shortfalls[bucket] += 1
                continue

            later_buckets = ALLOCATION_ORDER[
                ALLOCATION_ORDER.index(bucket) + 1 :
            ]

            def allocation_score(
                candidate_index: int,
            ) -> tuple[float, float, int, float]:
                bucket_capacity = (
                    candidate_counts[candidate_index][bucket]
                    - assignments[candidate_index][bucket]
                )
                future_capacity = sum(
                    candidate_counts[candidate_index][future_bucket]
                    - assignments[candidate_index][future_bucket]
                    for future_bucket in later_buckets
                )
                future_demand = sum(
                    int(quotas[future_bucket])
                    - sum(
                        assignment[future_bucket]
                        for assignment in assignments
                    )
                    for future_bucket in later_buckets
                )
                future_slack = future_capacity - future_demand
                return (
                    -future_slack,
                    -bucket_capacity,
                    max(
                        min_tasks_per_maze
                        - sum(assignments[candidate_index].values()),
                        0,
                    ),
                    tie_breakers[
                        candidate_index
                    ][BUCKETS.index(bucket)],
                )

            layout_index = max(
                eligible,
                key=allocation_score,
            )
            assignments[layout_index][bucket] += 1
            remaining_total -= 1
            minimum_needed -= int(
                sum(assignments[layout_index].values())
                <= min_tasks_per_maze
            )
            bucket_completed = bucket_index + 1
            if progress_callback is not None and (
                bucket_completed == 1
                or bucket_completed == bucket_total
                or bucket_completed % report_interval == 0
            ):
                progress_callback(
                    {
                        "phase": "allocation",
                        "allocation_bucket": bucket,
                        "allocation_bucket_completed": bucket_completed,
                        "allocation_bucket_total": bucket_total,
                        "allocation_slots_completed": sum(
                            sum(assignment.values())
                            for assignment in assignments
                        ),
                        "allocation_slots_total": sum(
                            int(quotas[current_bucket])
                            for current_bucket in BUCKETS
                        ),
                        "bucket_counts": {
                            current_bucket: sum(
                                assignment[current_bucket]
                                for assignment in assignments
                            )
                            for current_bucket in BUCKETS
                        },
                        "bucket_quotas": dict(quotas),
                    }
                )

    if any(shortfalls.values()):
        raise QuotaAllocationError(shortfalls)
    if any(
        sum(assignment.values()) != tasks_per_maze
        for assignment in assignments
    ):
        raise QuotaAllocationError(
            {"contribution_bounds": len(assignments)}
        )
    return assignments


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


def threshold_schedule(threshold: float) -> tuple[float, ...]:
    if not 0 < threshold <= 1:
        raise ValueError("The deduplication threshold must be in (0, 1].")
    return (threshold,)


def _constrained_layout_order(
    candidate_counts_by_layout: list[Mapping[str, int]],
    assignments: list[Mapping[str, int]],
    rng: np.random.Generator,
) -> list[int]:
    tie_breakers = rng.random(len(candidate_counts_by_layout))
    order_keys = []
    for layout_index, (candidate_counts, assignment) in enumerate(
        zip(candidate_counts_by_layout, assignments)
    ):
        required = sum(assignment.values())
        available = sum(
            candidate_counts[bucket]
            for bucket in BUCKETS
            if assignment[bucket]
        )
        pressure = available / max(required, 1)
        order_keys.append(
            (
                pressure,
                available,
                tie_breakers[layout_index],
                layout_index,
            )
        )
    return [
        key[-1]
        for key in sorted(order_keys)
    ]


def _select_all_layouts(
    layout_index_factory: Callable[[int], MazeCandidateIndex],
    candidate_counts_by_layout: list[Mapping[str, int]],
    assignments: list[dict[str, int]],
    *,
    threshold: float,
    seeds: np.ndarray,
    random_seed: int,
    local_search_budget: int,
    bucket_quotas: Mapping[str, int],
    selection_attempt: int,
    selection_attempts: int,
    progress_callback: Callable[[dict[str, object]], None] | None = None,
) -> tuple[list[SearchResult], dict[str, object]]:
    results: list[SearchResult | None] = [None] * len(assignments)
    order_rng = np.random.default_rng(random_seed)
    order = _constrained_layout_order(
        candidate_counts_by_layout,
        assignments,
        order_rng,
    )
    if progress_callback is not None:
        progress_callback(
            {
                "phase": "selection",
                "layouts_completed": 0,
                "layout_count": len(assignments),
                "bucket_counts": _empty_assignment(),
                "bucket_quotas": dict(bucket_quotas),
                "selection_attempt": selection_attempt,
                "selection_attempts": selection_attempts,
            }
        )
    for layout_index in order:
        selection_rng = np.random.default_rng(
            np.random.SeedSequence(
                [random_seed, layout_index, int(seeds[layout_index])]
            )
        )
        index = layout_index_factory(layout_index)
        results[layout_index] = _select_tasks_for_maze(
            index,
            assignments[layout_index],
            threshold=threshold,
            rng=selection_rng,
            search_budget=local_search_budget,
        )
        del index
        if progress_callback is not None:
            bucket_counts = _assignment_for_tasks(
                task
                for result in results
                if result is not None
                for task in result.selected
            )
            progress_callback(
                {
                    "phase": "selection",
                    "layouts_completed": sum(
                        result is not None for result in results
                    ),
                    "layout_count": len(assignments),
                    "bucket_counts": bucket_counts,
                    "bucket_quotas": dict(bucket_quotas),
                    "selection_attempt": selection_attempt,
                    "selection_attempts": selection_attempts,
                }
            )

    finalized_results = [result for result in results if result is not None]
    budget_exhausted_layouts = [
        layout_index
        for layout_index, result in enumerate(results)
        if result is not None and result.budget_exhausted
    ]
    proven_infeasible_layouts = [
        layout_index
        for layout_index, result in enumerate(results)
        if result is not None
        and not result.completed
        and not result.budget_exhausted
    ]
    return finalized_results, {
        "selection_order": order,
        "candidate_attempts": [
            result.candidate_attempts
            for result in finalized_results
        ],
        "backtracking_events": [
            result.backtracking_events
            for result in finalized_results
        ],
        "budget_exhausted": [
            index
            for index, result in enumerate(finalized_results)
            if result.budget_exhausted
        ],
        "proven_infeasible": [
            index
            for index, result in enumerate(finalized_results)
            if not result.completed and not result.budget_exhausted
        ],
        "budget_exhausted_layouts": budget_exhausted_layouts,
        "proven_infeasible_layouts": proven_infeasible_layouts,
        "budget_exhausted_count": len(budget_exhausted_layouts),
        "proven_infeasible_count": len(proven_infeasible_layouts),
        "completed": [
            result.completed
            for result in finalized_results
        ],
        "successful_completions": sum(
            result.completed
            for result in finalized_results
        ),
    }


def _repair_unresolved_allocations(
    layout_index_factory: Callable[[int], MazeCandidateIndex],
    candidate_counts_by_layout: list[Mapping[str, int]],
    assignments: list[dict[str, int]],
    selected_by_layout: list[list[TaskCandidate]],
    unresolved: list[tuple[int, str]],
    *,
    threshold: float,
    seeds: np.ndarray,
    random_seed: int,
    min_tasks_per_maze: int,
    max_tasks_per_maze: int,
    local_search_budget: int,
    repair_budget: int,
    bucket_quotas: Mapping[str, int],
    allocation_attempt: int,
    allocation_attempts: int,
    progress_callback: Callable[[dict[str, object]], None] | None = None,
) -> tuple[list[tuple[int, str]], dict[str, object]]:
    repair_attempts = 0
    successful_reallocations = 0
    recipient_checks = 0
    repair_candidate_attempts = 0
    repair_backtracking_events = 0

    for source_index, bucket in list(unresolved):
        if repair_attempts >= repair_budget:
            break

        # A transfer is allowed only if the source remains within its
        # contribution lower bound.  Incomplete source selections therefore
        # remain unresolved and cause the attempt to retry or fail explicitly.
        if len(selected_by_layout[source_index]) < min_tasks_per_maze:
            continue

        current_bucket_total = sum(
            assignment[bucket] for assignment in assignments
        )
        if current_bucket_total >= int(bucket_quotas[bucket]):
            continue

        recipients = []
        for layout_index, assignment in enumerate(assignments):
            if layout_index == source_index:
                continue
            current_total = sum(assignment.values())
            capacity = (
                candidate_counts_by_layout[layout_index][bucket]
                - assignment[bucket]
            )
            if current_total >= max_tasks_per_maze or capacity <= 0:
                continue
            recipients.append((capacity, layout_index))

        recipients.sort(reverse=True)
        moved = False
        for _, recipient_index in recipients:
            if repair_attempts >= repair_budget:
                break
            repair_attempts += 1
            recipient_checks += 1
            tentative = dict(assignments[recipient_index])
            tentative[bucket] += 1
            selection_rng = np.random.default_rng(
                np.random.SeedSequence(
                    [random_seed, 10_000 + repair_attempts, recipient_index, int(seeds[recipient_index])]
                )
            )
            index = layout_index_factory(recipient_index)
            result = _select_tasks_for_maze(
                index,
                tentative,
                threshold=threshold,
                rng=selection_rng,
                search_budget=local_search_budget,
            )
            del index
            if not result.completed:
                repair_candidate_attempts += result.candidate_attempts
                repair_backtracking_events += result.backtracking_events
                if progress_callback is not None:
                    progress_callback(
                        {
                            "phase": "repair",
                            "repair_attempts": repair_attempts,
                            "repair_budget": repair_budget,
                            "bucket_counts": _assignment_for_tasks(
                                task
                                for selected in selected_by_layout
                                for task in selected
                            ),
                            "bucket_quotas": dict(bucket_quotas),
                            "allocation_attempt": allocation_attempt,
                            "allocation_attempts": allocation_attempts,
                        }
                    )
                continue
            assignments[recipient_index] = tentative
            selected_by_layout[recipient_index] = result.selected
            successful_reallocations += 1
            repair_candidate_attempts += result.candidate_attempts
            repair_backtracking_events += result.backtracking_events
            if progress_callback is not None:
                progress_callback(
                    {
                        "phase": "repair",
                        "repair_attempts": repair_attempts,
                        "repair_budget": repair_budget,
                        "bucket_counts": _assignment_for_tasks(
                            task
                            for selected in selected_by_layout
                            for task in selected
                        ),
                        "bucket_quotas": dict(bucket_quotas),
                        "allocation_attempt": allocation_attempt,
                        "allocation_attempts": allocation_attempts,
                    }
                )
            moved = True
            break

        if moved:
            unresolved.remove((source_index, bucket))

    return unresolved, {
        "repair_attempts": repair_attempts,
        "recipient_checks": recipient_checks,
        "successful_reallocations": successful_reallocations,
        "candidate_attempts": repair_candidate_attempts,
        "backtracking_events": repair_backtracking_events,
        "repair_budget": repair_budget,
    }


def generate_stratified_tasks_for_layouts(
    layouts: np.ndarray,
    seeds: np.ndarray,
    *,
    tasks_per_maze: int,
    min_path_length: int,
    dedupe_threshold: float = 0.7,
    bucket_budgets: Mapping[str, float] | None = None,
    random_seed: int = 0,
    excluded_tasks_by_layout: dict[int, set[TaskKey]] | None = None,
    allocation_attempts: int = 8,
    min_tasks_per_maze: int = 3,
    max_tasks_per_maze: int = 7,
    local_search_budget: int = 2_000,
    global_repair_budget: int | None = None,
    difficulty_boundaries: DifficultyBoundaries | None = None,
    progress_callback: Callable[[dict[str, object]], None] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    """Select reproducible fixed-quota tasks for each layout.

    Every layout receives the same bucket assignment derived from the
    per-layout budget. Failed assignments are retried with a new selection
    seed; they are never transferred to another layout.
    """
    if tasks_per_maze <= 0:
        raise ValueError("tasks_per_maze must be positive.")
    if not 0 < min_tasks_per_maze <= tasks_per_maze <= max_tasks_per_maze:
        raise ValueError(
            "Expected min_tasks_per_maze <= tasks_per_maze <= "
            "max_tasks_per_maze."
        )
    if len(layouts) != len(seeds):
        raise ValueError("layouts and seeds must have the same length.")
    if local_search_budget <= 0:
        raise ValueError("local_search_budget must be positive.")
    budgets = validate_bucket_budgets(
        bucket_budgets or DEFAULT_BUCKET_BUDGETS,
    )
    excluded_tasks_by_layout = excluded_tasks_by_layout or {}
    per_layout_quotas = bucket_quotas(tasks_per_maze, budgets)
    quotas = {
        bucket: per_layout_quotas[bucket] * len(layouts)
        for bucket in BUCKETS
    }
    if progress_callback is not None:
        progress_callback(
            {
                "phase": "indexing",
                "layouts_completed": 0,
                "layout_count": len(layouts),
                "bucket_counts": _empty_assignment(),
                "bucket_quotas": dict(quotas),
            }
        )
    boundaries = difficulty_boundaries or DEFAULT_DIFFICULTY_BOUNDARIES

    def build_layout_index(layout_index: int) -> MazeCandidateIndex:
        index = build_candidate_index(
            layouts[layout_index],
            min_path_length=min_path_length,
            excluded_tasks=excluded_tasks_by_layout.get(layout_index),
        )
        apply_difficulty_buckets([index], boundaries)
        return index

    # Keep only compact bucket capacities for the fixed-assignment selection
    # pass. Detailed candidate indexes contain Python dictionaries for every
    # BFS source and must be released after each layout; retaining all of
    # them makes large stratified datasets consume several gigabytes.
    candidate_counts_by_layout = []
    for layout_index in range(len(layouts)):
        index = build_layout_index(layout_index)
        candidate_counts_by_layout.append(_candidate_bucket_counts(index))
        del index
        if progress_callback is not None:
            progress_callback(
                {
                    "phase": "indexing",
                    "layouts_completed": layout_index + 1,
                    "layout_count": len(layouts),
                    "bucket_counts": _empty_assignment(),
                    "bucket_quotas": dict(quotas),
                }
            )
    insufficient = [
        layout_index
        for layout_index, counts in enumerate(candidate_counts_by_layout)
        if any(
            counts[bucket] < per_layout_quotas[bucket]
            for bucket in BUCKETS
        )
    ]
    if insufficient:
        raise RuntimeError(
            "Layouts without enough candidates for fixed bucket quotas: "
            + ", ".join(str(index) for index in insufficient[:10])
        )

    candidate_counts = {
        bucket: sum(
            counts[bucket]
            for counts in candidate_counts_by_layout
        )
        for bucket in BUCKETS
    }
    started_at = time.perf_counter()
    last_error: Exception | None = None
    last_failure_diagnostics: dict[str, object] | None = None
    repair_metadata = {
        "repair_attempts": 0,
        "recipient_checks": 0,
        "successful_reallocations": 0,
        "candidate_attempts": 0,
        "backtracking_events": 0,
        "repair_budget": 0,
    }

    def report_attempt(
        attempt: int,
        phase: str,
        bucket_counts: Mapping[str, int],
        reason: str | None = None,
        persist: bool = False,
        diagnostics: Mapping[str, object] | None = None,
    ) -> None:
        if progress_callback is None:
            return
        event: dict[str, object] = {
            "phase": phase,
            "selection_attempt": attempt,
            "selection_attempts": allocation_attempts,
            "bucket_counts": dict(bucket_counts),
            "bucket_quotas": dict(quotas),
        }
        if reason is not None:
            event["reason"] = reason
        if persist:
            event["persist"] = True
        if diagnostics is not None:
            event["diagnostics"] = dict(diagnostics)
        progress_callback(event)

    for selection_attempt in range(allocation_attempts):
        attempt_number = selection_attempt + 1
        assignments = [
            dict(per_layout_quotas)
            for _ in layouts
        ]
        results, search_metadata = _select_all_layouts(
            build_layout_index,
            candidate_counts_by_layout,
            assignments,
            threshold=dedupe_threshold,
            seeds=seeds,
            random_seed=random_seed + selection_attempt,
            local_search_budget=local_search_budget,
            bucket_quotas=quotas,
            selection_attempt=attempt_number,
            selection_attempts=allocation_attempts,
            progress_callback=progress_callback,
        )
        selected_by_layout = [result.selected for result in results]
        unresolved: list[tuple[int, str]] = []
        for layout_index, (assignment, result) in enumerate(
            zip(assignments, results)
        ):
            selected_assignment = _assignment_for_tasks(result.selected)
            for bucket in BUCKETS:
                unresolved.extend(
                    (layout_index, bucket)
                    for _ in range(
                        max(
                            assignment[bucket]
                            - selected_assignment[bucket],
                            0,
                        )
                    )
                )

        contributions = [len(selected) for selected in selected_by_layout]
        failure_diagnostics = _failure_diagnostics(
            search_metadata,
            unresolved_before_repair=unresolved,
            unresolved_after_repair=unresolved,
            contributions=contributions,
            min_tasks_per_maze=tasks_per_maze,
            max_tasks_per_maze=tasks_per_maze,
            repair_metadata=repair_metadata,
        )
        if unresolved or any(
            contribution != tasks_per_maze
            for contribution in contributions
        ):
            last_error = RuntimeError(
                "A layout could not satisfy its fixed bucket assignment."
            )
            last_failure_diagnostics = failure_diagnostics
            report_attempt(
                attempt_number,
                "retrying",
                _assignment_for_tasks(
                    task
                    for selected in selected_by_layout
                    for task in selected
                ),
                reason="fixed per-layout assignment was infeasible",
                persist=True,
                diagnostics=failure_diagnostics,
            )
            continue

        bucket_counts = {
            bucket: sum(
                task.bucket == bucket
                for selected in selected_by_layout
                for task in selected
            )
            for bucket in BUCKETS
        }
        if bucket_counts != quotas:
            last_error = RuntimeError(
                f"Fixed bucket assignments produced unexpected totals: "
                f"{bucket_counts}"
            )
            last_failure_diagnostics = failure_diagnostics
            report_attempt(
                attempt_number,
                "retrying",
                bucket_counts,
                reason="fixed bucket totals did not match",
                persist=True,
                diagnostics=failure_diagnostics,
            )
            continue

        task_count = sum(contributions)
        layout_indices = np.empty(task_count, dtype=np.int64)
        starts = np.empty((task_count, 2), dtype=np.int64)
        goals = np.empty((task_count, 2), dtype=np.int64)
        output_index = 0
        for layout_index, selected in enumerate(selected_by_layout):
            for candidate in selected:
                layout_indices[output_index] = layout_index
                starts[output_index] = candidate.start
                goals[output_index] = candidate.goal
                output_index += 1

        metadata: dict[str, object] = {
            "dedupe_metric": "expected_pairwise_directed_path_jaccard",
            "dedupe_scope": DEDUPE_SCOPE,
            "dedupe_threshold": dedupe_threshold,
            "bucket_budgets": budgets,
            "bucket_quotas": quotas,
            "bucket_counts": bucket_counts,
            "candidate_counts": candidate_counts,
            "candidate_counts_by_layout": candidate_counts_by_layout,
            "difficulty_boundaries": boundaries.as_dict(),
            "difficulty_boundaries_source": "configured absolute lengths",
            "tasks_per_maze_target": tasks_per_maze,
            "min_tasks_per_maze": min_tasks_per_maze,
            "max_tasks_per_maze": max_tasks_per_maze,
            "contributions_by_layout": contributions,
            "bucket_quotas_per_layout": per_layout_quotas,
            "selection_attempts": selection_attempt + 1,
            "local_search_budget": local_search_budget,
            "global_repair_budget": None,
            "path_sample_count": PATH_SAMPLE_COUNT,
            "max_exact_paths": MAX_EXACT_PATHS,
            "max_exact_pair_count": MAX_EXACT_PAIR_COUNT,
            "search": search_metadata,
            "repair": repair_metadata,
            "unresolved_before_repair_by_bucket": {
                bucket: sum(
                    unresolved_bucket == bucket
                    for _, unresolved_bucket in unresolved
                )
                for bucket in BUCKETS
            },
            "unresolved_allocations": [],
            "generation_runtime_seconds": time.perf_counter() - started_at,
        }
        return layout_indices, starts, goals, metadata

    failure_details = ""
    if last_failure_diagnostics is not None:
        failure_details = (
            "\nDiagnostics:\n  "
            + format_failure_diagnostics(last_failure_diagnostics)
            .replace("\n", "\n  ")
        )
    raise RuntimeError(
        "Could not select fixed per-layout stratified tasks after "
        "retries. "
        f"Requested quotas: {quotas}. Last error: {last_error}"
        f"{failure_details}"
    )
