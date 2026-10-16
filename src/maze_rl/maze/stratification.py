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
    generate_valid_tasks_for_maze,
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


DEFAULT_DIFFICULTY_BOUNDARIES = DifficultyBoundaries(
    short_max=10.0,
    medium_max=20.0,
)


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
    return (
        "budget-exhausted layouts="
        f"{diagnostics.get('budget_exhausted_count', 0)}"
        f" [{preview('budget_exhausted_layouts')}]; "
        "infeasible current assignments="
        f"{diagnostics.get('infeasible_assignment_count', 0)}"
        f" [{preview('infeasible_assignment_layouts')}]; "
        "below-min layouts="
        f"{diagnostics.get('contribution_below_min_count', 0)}"
        f" [{preview('contribution_below_min_layouts')}]; "
        "unresolved before repair="
        f"S {before.get('short', 0)}/M {before.get('medium', 0)}"
        f"/L {before.get('long', 0)}; "
        "after repair="
        f"S {after.get('short', 0)}/M {after.get('medium', 0)}"
        f"/L {after.get('long', 0)}; "
        "repairs="
        f"{diagnostics.get('successful_reallocations', 0)}"
        f"/{diagnostics.get('repair_attempts', 0)}"
    )


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
    progress_callback: Callable[[int, int], None] | None = None,
) -> np.ndarray:
    """Generate layouts independently of stratified task selection."""
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
            try:
                generate_valid_tasks_for_maze(
                    maze=maze,
                    task_count=minimum_eligible_tasks,
                    rng=rng,
                    min_path_length=min_path_length,
                )
            except RuntimeError:
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
                for previous in selected
            ):
                continue

            selected.append(candidate)
            selected_keys.add(candidate.key)
            if visit(slot_index + 1):
                return True
            selected.pop()
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
    allocation_attempt: int,
    allocation_attempts: int,
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
                "allocation_attempt": allocation_attempt,
                "allocation_attempts": allocation_attempts,
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
                    "allocation_attempt": allocation_attempt,
                    "allocation_attempts": allocation_attempts,
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
    dedupe_threshold: float = 0.75,
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
    """Select reproducible, difficulty-balanced tasks for each layout."""
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
    quotas = bucket_quotas(len(layouts) * tasks_per_maze, budgets)
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

    # Keep only compact bucket capacities for the global allocation pass.
    # Detailed candidate indexes contain Python dictionaries for every BFS
    # source and must be released after each layout; retaining all of them
    # makes large stratified datasets consume several gigabytes.
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
        if sum(counts.values()) < min_tasks_per_maze
    ]
    if insufficient:
        raise RuntimeError(
            "Layouts without enough eligible tasks: "
            + ", ".join(str(index) for index in insufficient[:10])
        )

    candidate_counts = {
        bucket: sum(
            counts[bucket]
            for counts in candidate_counts_by_layout
        )
        for bucket in BUCKETS
    }
    repair_budget = (
        5 * len(layouts)
        if global_repair_budget is None
        else global_repair_budget
    )
    started_at = time.perf_counter()
    last_error: Exception | None = None
    last_failure_diagnostics: dict[str, object] | None = None
    repair_attempts_used = 0
    repair_totals = {
        "repair_attempts": 0,
        "recipient_checks": 0,
        "successful_reallocations": 0,
        "candidate_attempts": 0,
        "backtracking_events": 0,
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
            "allocation_attempt": attempt,
            "allocation_attempts": allocation_attempts,
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

    for allocation_attempt in range(allocation_attempts):
        attempt_number = allocation_attempt + 1
        report_attempt(
            attempt_number,
            "allocation",
            _empty_assignment(),
        )
        allocation_rng = np.random.default_rng(
            np.random.SeedSequence([random_seed, allocation_attempt])
        )
        try:
            assignments = allocate_bucket_slots(
                tasks_per_maze=tasks_per_maze,
                quotas=quotas,
                rng=allocation_rng,
                min_tasks_per_maze=min_tasks_per_maze,
                max_tasks_per_maze=max_tasks_per_maze,
                candidate_counts_by_layout=candidate_counts_by_layout,
                progress_callback=(
                    lambda event: progress_callback(
                        {
                            **event,
                            "allocation_attempt": attempt_number,
                            "allocation_attempts": allocation_attempts,
                        }
                    )
                    if progress_callback is not None
                    else None
                ),
            )
        except QuotaAllocationError as error:
            last_error = error
            last_failure_diagnostics = None
            report_attempt(
                attempt_number,
                "retrying",
                _empty_assignment(),
                reason=str(error),
                persist=True,
            )
            continue

        results, search_metadata = _select_all_layouts(
            build_layout_index,
            candidate_counts_by_layout,
            assignments,
            threshold=dedupe_threshold,
            seeds=seeds,
            random_seed=random_seed + allocation_attempt,
            local_search_budget=local_search_budget,
            bucket_quotas=quotas,
            allocation_attempt=allocation_attempt + 1,
            allocation_attempts=allocation_attempts,
            progress_callback=progress_callback,
        )
        selected_by_layout = [result.selected for result in results]
        unresolved: list[tuple[int, str]] = []
        for layout_index, (assignment, result) in enumerate(
            zip(assignments, results)
        ):
            if result.completed:
                continue
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
            assignments[layout_index] = selected_assignment

        if unresolved and repair_attempts_used < repair_budget:
            unresolved_before_repair = list(unresolved)
            remaining_repair_budget = repair_budget - repair_attempts_used
            unresolved, repair_metadata = _repair_unresolved_allocations(
                build_layout_index,
                candidate_counts_by_layout,
                assignments,
                selected_by_layout,
                unresolved,
                threshold=dedupe_threshold,
                seeds=seeds,
                random_seed=random_seed + allocation_attempt,
                min_tasks_per_maze=min_tasks_per_maze,
                max_tasks_per_maze=max_tasks_per_maze,
                local_search_budget=local_search_budget,
                repair_budget=remaining_repair_budget,
                bucket_quotas=quotas,
                allocation_attempt=allocation_attempt + 1,
                allocation_attempts=allocation_attempts,
                progress_callback=progress_callback,
            )
            repair_attempts_used += repair_metadata["repair_attempts"]
            for key in repair_totals:
                repair_totals[key] += repair_metadata[key]
        else:
            unresolved_before_repair = list(unresolved)
            repair_metadata = {
                "repair_attempts": 0,
                "recipient_checks": 0,
                "successful_reallocations": 0,
                "repair_budget": repair_budget,
            }

        contributions = [
            len(selected)
            for selected in selected_by_layout
        ]
        failure_diagnostics = _failure_diagnostics(
            search_metadata,
            unresolved_before_repair=unresolved_before_repair,
            unresolved_after_repair=unresolved,
            contributions=contributions,
            min_tasks_per_maze=min_tasks_per_maze,
            max_tasks_per_maze=max_tasks_per_maze,
            repair_metadata=repair_metadata,
        )
        if unresolved or any(
            contribution < min_tasks_per_maze
            or contribution > max_tasks_per_maze
            for contribution in contributions
        ):
            last_error = RuntimeError(
                "Unresolved stratified task allocations."
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
                reason="unresolved allocation or contribution bounds",
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
                f"Bucket quotas changed during repair: {bucket_counts}"
            )
            last_failure_diagnostics = failure_diagnostics
            report_attempt(
                attempt_number,
                "retrying",
                bucket_counts,
                reason="bucket quotas changed during repair",
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
            "allocation_attempts": allocation_attempt + 1,
            "local_search_budget": local_search_budget,
            "global_repair_budget": repair_budget,
            "search": search_metadata,
            "repair": {
                **repair_totals,
                "repair_budget": repair_budget,
            },
            "unresolved_before_repair_by_bucket": {
                bucket: sum(
                    unresolved_bucket == bucket
                    for _, unresolved_bucket in unresolved_before_repair
                )
                for bucket in BUCKETS
            },
            "unresolved_allocations": unresolved,
            "generation_runtime_seconds": time.perf_counter() - started_at,
        }
        return layout_indices, starts, goals, metadata

    failure_details = ""
    if last_failure_diagnostics is not None:
        failure_details = (
            " Diagnostics: "
            + format_failure_diagnostics(last_failure_diagnostics)
        )
    raise RuntimeError(
        "Could not select stratified tasks after retries without "
        "relaxing constraints. "
        f"Requested quotas: {quotas}. Last error: {last_error}"
        f"{failure_details}"
    )
