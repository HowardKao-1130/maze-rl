"""Task-level maze augmentation primitives."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from maze_rl.maze.pathfinding import shortest_path_length


Position = tuple[int, int]


@dataclass(frozen=True)
class MazeTask:
    """A maze layout and its ordered start/goal task."""

    layout: np.ndarray
    start: Position
    goal: Position


@dataclass(frozen=True)
class TaskVariant:
    """A generated task plus enough lineage to audit its augmentation."""

    layout: np.ndarray
    start: Position
    goal: Position
    transform: str
    reversed_endpoints: bool
    source_path_length: int
    optimal_path_length: int
    perturbation_index: int | None = None
    wall_additions: int = 0
    wall_removals: int = 0


def _validate_task(task: MazeTask) -> int:
    layout = np.asarray(task.layout)
    if layout.ndim != 2:
        raise ValueError("Maze layouts must be two-dimensional.")
    if not np.isin(layout, (0, 1)).all():
        raise ValueError("Maze layouts may contain only 0 and 1 values.")

    height, width = layout.shape
    for name, position in (("start", task.start), ("goal", task.goal)):
        row, column = position
        if not (0 <= row < height and 0 <= column < width):
            raise ValueError(f"{name} must be inside the maze layout.")
        if layout[position] == 1:
            raise ValueError(f"{name} cannot be a wall.")
    if task.start == task.goal:
        raise ValueError("Start and goal must be different cells.")

    path_length = shortest_path_length(
        layout,
        task.start,
        task.goal,
    )
    if path_length is None:
        raise ValueError("Maze task must have a reachable goal.")
    return int(path_length)


def _rotate_position(
    position: Position,
    shape: tuple[int, int],
    quarter_turns: int,
) -> Position:
    height, width = shape
    row, column = position
    if quarter_turns == 0:
        return row, column
    if quarter_turns == 1:
        return width - 1 - column, row
    if quarter_turns == 2:
        return height - 1 - row, width - 1 - column
    if quarter_turns == 3:
        return column, height - 1 - row
    raise ValueError("quarter_turns must be between 0 and 3.")


def _spatial_transforms(
    layout: np.ndarray,
    start: Position,
    goal: Position,
):
    """Yield the eight D4 transforms of a task."""
    for quarter_turns in range(4):
        rotated = np.rot90(layout, quarter_turns)
        rotated_start = _rotate_position(
            start,
            layout.shape,
            quarter_turns,
        )
        rotated_goal = _rotate_position(
            goal,
            layout.shape,
            quarter_turns,
        )
        rotation_name = f"rotate_{quarter_turns * 90}"
        yield (
            rotation_name,
            np.array(rotated, copy=True),
            rotated_start,
            rotated_goal,
        )

        transformed_width = rotated.shape[1]
        reflected = np.fliplr(rotated)
        reflected_start = (
            rotated_start[0],
            transformed_width - 1 - rotated_start[1],
        )
        reflected_goal = (
            rotated_goal[0],
            transformed_width - 1 - rotated_goal[1],
        )
        yield (
            f"rotate_{quarter_turns * 90}_reflect",
            np.array(reflected, copy=True),
            reflected_start,
            reflected_goal,
        )


def _task_identity(
    variant: TaskVariant,
) -> tuple[tuple[int, int], bytes, Position, Position]:
    return (
        variant.layout.shape,
        variant.layout.tobytes(),
        variant.start,
        variant.goal,
    )


def _spatial_variants_for_base(base: TaskVariant) -> list[TaskVariant]:
    variants: list[TaskVariant] = []
    seen: set[tuple[tuple[int, int], bytes, Position, Position]] = set()
    for name, layout, start, goal in _spatial_transforms(
        base.layout,
        base.start,
        base.goal,
    ):
        for reversed_endpoints in (False, True):
            variant_start, variant_goal = (
                (goal, start)
                if reversed_endpoints
                else (start, goal)
            )
            variant = TaskVariant(
                layout=layout,
                start=variant_start,
                goal=variant_goal,
                transform=name,
                reversed_endpoints=reversed_endpoints,
                source_path_length=base.source_path_length,
                optimal_path_length=base.optimal_path_length,
                perturbation_index=base.perturbation_index,
                wall_additions=base.wall_additions,
                wall_removals=base.wall_removals,
            )
            identity = _task_identity(variant)
            if identity in seen:
                continue
            seen.add(identity)
            variants.append(variant)
    return variants


def generate_spatial_variants(task: MazeTask) -> list[TaskVariant]:
    """Generate unique D4 and endpoint-reversal variants for one task."""
    source_path_length = _validate_task(task)
    base = TaskVariant(
        layout=np.asarray(task.layout, dtype=np.uint8),
        start=task.start,
        goal=task.goal,
        transform="identity",
        reversed_endpoints=False,
        source_path_length=source_path_length,
        optimal_path_length=source_path_length,
    )
    return _spatial_variants_for_base(base)


def generate_perturbed_variants(
    task: MazeTask,
    count: int,
    *,
    rng: np.random.Generator,
    max_edits: int = 4,
    max_attempts: int = 100,
    add_probability: float = 0.5,
    max_path_increase: int | None = None,
) -> list[TaskVariant]:
    """Generate wall perturbations whose path length never decreases.

    Each perturbation starts from the original task. Invalid candidates are
    rejected, so the returned count can be lower than ``count``.
    """
    if count < 0:
        raise ValueError("count must be non-negative.")
    if max_edits <= 0 or max_attempts <= 0:
        raise ValueError("max_edits and max_attempts must be positive.")
    if not 0 <= add_probability <= 1:
        raise ValueError("add_probability must be between 0 and 1.")
    if max_path_increase is not None and max_path_increase < 0:
        raise ValueError("max_path_increase must be non-negative.")

    source_path_length = _validate_task(task)
    source_layout = np.asarray(task.layout, dtype=np.uint8)
    variants: list[TaskVariant] = []
    height, width = source_layout.shape

    for perturbation_index in range(count):
        layout = np.array(source_layout, copy=True)
        edited_positions: set[Position] = set()
        rejected_positions: set[Position] = set()
        additions = 0
        removals = 0
        path_length = source_path_length

        for _ in range(max_attempts):
            if additions + removals >= max_edits:
                break

            add_candidates = [
                (row, column)
                for row in range(height)
                for column in range(width)
                if layout[row, column] == 0
                and (row, column) not in {task.start, task.goal}
                and (row, column) not in edited_positions
                and (row, column) not in rejected_positions
            ]
            remove_candidates = [
                (row, column)
                for row in range(height)
                for column in range(width)
                if layout[row, column] == 1
                and (row, column) not in edited_positions
                and (row, column) not in rejected_positions
            ]
            if not add_candidates and not remove_candidates:
                break

            should_add = bool(rng.random() < add_probability)
            if should_add and add_candidates:
                candidates = add_candidates
                new_value = 1
            elif remove_candidates:
                candidates = remove_candidates
                new_value = 0
            elif add_candidates:
                candidates = add_candidates
                new_value = 1
            else:
                break

            position = candidates[int(rng.integers(len(candidates)))]
            old_value = int(layout[position])
            layout[position] = new_value
            candidate_path_length = shortest_path_length(
                layout,
                task.start,
                task.goal,
            )
            upper_bound = (
                None
                if max_path_increase is None
                else source_path_length + max_path_increase
            )
            invalid = (
                candidate_path_length is None
                or candidate_path_length < source_path_length
                or (
                    upper_bound is not None
                    and candidate_path_length > upper_bound
                )
            )
            if invalid:
                layout[position] = old_value
                rejected_positions.add(position)
                continue

            edited_positions.add(position)
            path_length = int(candidate_path_length)
            if new_value == 1:
                additions += 1
            else:
                removals += 1

        if not edited_positions:
            continue
        variants.append(
            TaskVariant(
                layout=layout,
                start=task.start,
                goal=task.goal,
                transform="identity",
                reversed_endpoints=False,
                source_path_length=source_path_length,
                optimal_path_length=path_length,
                perturbation_index=perturbation_index,
                wall_additions=additions,
                wall_removals=removals,
            )
        )
    return variants


def augment_task(
    task: MazeTask,
    *,
    perturbation_count: int = 0,
    rng: np.random.Generator | None = None,
    max_edits: int = 4,
    max_attempts: int = 100,
    add_probability: float = 0.5,
    max_path_increase: int | None = None,
) -> list[TaskVariant]:
    """Generate perturbations first, then the full spatial/reversal orbit."""
    if perturbation_count and rng is None:
        raise ValueError("rng is required when generating perturbations.")

    source_path_length = _validate_task(task)
    base = TaskVariant(
        layout=np.asarray(task.layout, dtype=np.uint8),
        start=task.start,
        goal=task.goal,
        transform="identity",
        reversed_endpoints=False,
        source_path_length=source_path_length,
        optimal_path_length=source_path_length,
    )
    bases = [base]
    if perturbation_count:
        bases.extend(
            generate_perturbed_variants(
                task,
                perturbation_count,
                rng=rng,
                max_edits=max_edits,
                max_attempts=max_attempts,
                add_probability=add_probability,
                max_path_increase=max_path_increase,
            )
        )

    variants: list[TaskVariant] = []
    seen: set[tuple[tuple[int, int], bytes, Position, Position]] = set()
    for base_variant in bases:
        for variant in _spatial_variants_for_base(base_variant):
            identity = _task_identity(variant)
            if identity in seen:
                continue
            seen.add(identity)
            variants.append(variant)
    return variants
