from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from maze_rl.envs.maze_env import MazeEnv
from maze_rl.maze.pathfinding import shortest_path_length


class VectorizedMazeEnv:
    """Array-backed maze environments for neural-network rollouts.

    The vector environment deliberately has a small API tailored to the
    training collectors.  Each slot represents one independent task episode;
    completed slots become no-ops until the next reset.
    """

    ACTION_DELTAS = np.asarray(
        [
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),
        ],
        dtype=np.int64,
    )
    ACTION_COUNT = len(ACTION_DELTAS)

    def __init__(
        self,
        dataset_path,
        max_steps: int = 200,
        goal_reward: float = 1.0,
        step_penalty: float = -0.01,
        wall_penalty: float = -0.05,
    ) -> None:
        data = np.load(dataset_path)

        if "layouts" in data:
            layouts = data["layouts"]
            layout_indices = data["layout_indices"]
        else:
            layouts = data["mazes"]
            layout_indices = np.arange(
                len(data["starts"]),
                dtype=np.int64,
            )

        self._initialize(
            layouts=layouts,
            layout_indices=layout_indices,
            starts=data["starts"],
            goals=data["goals"],
            max_steps=max_steps,
            goal_reward=goal_reward,
            step_penalty=step_penalty,
            wall_penalty=wall_penalty,
        )

    @classmethod
    def from_maze_env(cls, env: MazeEnv):
        """Create a vector view using the serial environment's loaded data."""
        vector_env = cls.__new__(cls)
        vector_env._initialize(
            layouts=env.layouts,
            layout_indices=env.layout_indices,
            starts=env.starts,
            goals=env.goals,
            max_steps=env.max_steps,
            goal_reward=env.goal_reward,
            step_penalty=env.step_penalty,
            wall_penalty=env.wall_penalty,
        )
        return vector_env

    def _initialize(
        self,
        *,
        layouts: np.ndarray,
        layout_indices: np.ndarray,
        starts: np.ndarray,
        goals: np.ndarray,
        max_steps: int,
        goal_reward: float,
        step_penalty: float,
        wall_penalty: float,
    ) -> None:
        self.layouts = np.asarray(layouts)
        self.layout_indices = np.asarray(
            layout_indices,
            dtype=np.int64,
        )
        self.starts = np.asarray(starts, dtype=np.int64)
        self.goals = np.asarray(goals, dtype=np.int64)

        if self.layouts.ndim != 3:
            raise ValueError(
                "layouts must have shape [num_layouts, height, width]."
            )
        if self.starts.ndim != 2 or self.starts.shape[1] != 2:
            raise ValueError(
                "starts must have shape [num_tasks, 2]."
            )
        if self.goals.shape != self.starts.shape:
            raise ValueError(
                "goals must have the same shape as starts."
            )
        if len(self.layout_indices) != len(self.starts):
            raise ValueError(
                "layout_indices must contain one entry per task."
            )

        self.num_layouts = len(self.layouts)
        self.num_tasks = len(self.starts)
        self.height = self.layouts.shape[1]
        self.width = self.layouts.shape[2]
        self.max_steps = int(max_steps)
        self.goal_reward = float(goal_reward)
        self.step_penalty = float(step_penalty)
        self.wall_penalty = float(wall_penalty)

        self.num_envs = 0
        self.task_indices = np.empty(0, dtype=np.int64)
        self.current_layout_indices = np.empty(
            0,
            dtype=np.int64,
        )
        self.agent_positions = np.empty((0, 2), dtype=np.int64)
        self.goal_positions = np.empty((0, 2), dtype=np.int64)
        self.step_counts = np.empty(0, dtype=np.int64)
        self.wall_collisions = np.empty(0, dtype=np.int64)
        self.optimal_path_lengths = np.empty(
            0,
            dtype=object,
        )
        self.terminated_flags = np.empty(0, dtype=bool)
        self.truncated_flags = np.empty(0, dtype=bool)
        self.observations = np.empty(
            (0, 3, self.height, self.width),
            dtype=np.float32,
        )

    def _validate_task_indices(self, task_indices) -> np.ndarray:
        indices = np.asarray(task_indices)
        if indices.ndim != 1:
            raise ValueError(
                "task_indices must be a one-dimensional sequence."
            )
        if indices.size == 0:
            return indices.astype(np.int64)
        if not np.issubdtype(indices.dtype, np.integer):
            raise ValueError(
                "task_indices must contain integers."
            )
        indices = indices.astype(np.int64, copy=False)
        if np.any(indices < 0) or np.any(indices >= self.num_tasks):
            raise IndexError(
                "task_indices contains an index outside the dataset."
            )
        return indices

    def reset(
        self,
        task_indices: Sequence[int] | np.ndarray,
    ):
        task_indices = self._validate_task_indices(task_indices)
        self.num_envs = len(task_indices)
        self.task_indices = task_indices.copy()
        self.current_layout_indices = self.layout_indices[
            task_indices
        ]
        self.agent_positions = self.starts[
            task_indices
        ].copy()
        self.goal_positions = self.goals[
            task_indices
        ].copy()
        self.step_counts = np.zeros(
            self.num_envs,
            dtype=np.int64,
        )
        self.wall_collisions = np.zeros(
            self.num_envs,
            dtype=np.int64,
        )
        self.optimal_path_lengths = np.asarray(
            [
                shortest_path_length(
                    self.layouts[layout_index],
                    tuple(position),
                    tuple(goal),
                )
                for layout_index, position, goal in zip(
                    self.current_layout_indices,
                    self.agent_positions,
                    self.goal_positions,
                )
            ],
            dtype=object,
        )
        self.terminated_flags = np.zeros(
            self.num_envs,
            dtype=bool,
        )
        self.truncated_flags = np.zeros(
            self.num_envs,
            dtype=bool,
        )
        self.observations = self._build_observations()

        return self.observations.copy(), [{} for _ in task_indices]

    def _build_observations(self) -> np.ndarray:
        observations = np.zeros(
            (self.num_envs, 3, self.height, self.width),
            dtype=np.float32,
        )
        if self.num_envs == 0:
            return observations

        observations[:, 0] = self.layouts[
            self.current_layout_indices
        ]
        slots = np.arange(self.num_envs)
        observations[
            slots,
            1,
            self.agent_positions[:, 0],
            self.agent_positions[:, 1],
        ] = 1.0
        observations[
            slots,
            2,
            self.goal_positions[:, 0],
            self.goal_positions[:, 1],
        ] = 1.0
        return observations

    def _validate_actions(self, actions) -> np.ndarray:
        actions = np.asarray(actions)
        if actions.shape != (self.num_envs,):
            raise ValueError(
                "actions must have shape "
                f"({self.num_envs},), got {actions.shape}."
            )
        if not np.issubdtype(actions.dtype, np.integer):
            raise ValueError(
                "actions must contain integer action indices."
            )
        actions = actions.astype(np.int64, copy=False)
        if np.any(actions < 0) or np.any(actions >= self.ACTION_COUNT):
            raise ValueError(
                f"actions must be in [0, {self.ACTION_COUNT})."
            )
        return actions

    def step(self, actions):
        actions = self._validate_actions(actions)
        active = ~(
            self.terminated_flags
            | self.truncated_flags
        )
        slots = np.flatnonzero(active)

        rewards = np.zeros(
            self.num_envs,
            dtype=np.float32,
        )
        transition_terminated = np.zeros(
            self.num_envs,
            dtype=bool,
        )
        transition_truncated = np.zeros(
            self.num_envs,
            dtype=bool,
        )
        hit_wall = np.zeros(
            self.num_envs,
            dtype=bool,
        )

        if len(slots):
            self.step_counts[slots] += 1
            rewards[slots] = self.step_penalty

            deltas = self.ACTION_DELTAS[actions[slots]]
            positions = self.agent_positions[slots]
            next_positions = positions + deltas

            rows = next_positions[:, 0]
            cols = next_positions[:, 1]
            in_bounds = (
                (rows >= 0)
                & (rows < self.height)
                & (cols >= 0)
                & (cols < self.width)
            )
            safe_rows = np.clip(rows, 0, self.height - 1)
            safe_cols = np.clip(cols, 0, self.width - 1)
            blocked = np.ones(len(slots), dtype=bool)
            blocked[in_bounds] = (
                self.layouts[
                    self.current_layout_indices[slots[in_bounds]],
                    safe_rows[in_bounds],
                    safe_cols[in_bounds],
                ]
                == 1
            )
            hit_wall[slots] = ~in_bounds | blocked

            movable = ~hit_wall[slots]
            movable_slots = slots[movable]
            self.agent_positions[movable_slots] = next_positions[movable]
            wall_slots = slots[hit_wall[slots]]
            self.wall_collisions[wall_slots] += 1
            rewards[wall_slots] = self.wall_penalty

            transition_terminated[slots] = np.all(
                self.agent_positions[slots]
                == self.goal_positions[slots],
                axis=1,
            )
            rewards[transition_terminated] = self.goal_reward
            transition_truncated[slots] = (
                self.step_counts[slots] >= self.max_steps
            ) & ~transition_terminated[slots]

            self.terminated_flags |= transition_terminated
            self.truncated_flags |= transition_truncated

        self.observations = self._build_observations()
        infos = []
        for slot in range(self.num_envs):
            path_efficiency = 0.0
            if (
                self.terminated_flags[slot]
                and self.optimal_path_lengths[slot] is not None
                and self.step_counts[slot] > 0
            ):
                path_efficiency = (
                    self.optimal_path_lengths[slot]
                    / self.step_counts[slot]
                )
            infos.append(
                {
                    "task_index": int(self.task_indices[slot]),
                    "layout_index": int(
                        self.current_layout_indices[slot]
                    ),
                    "maze_index": int(
                        self.current_layout_indices[slot]
                    ),
                    "hit_wall": bool(hit_wall[slot]),
                    "success": bool(self.terminated_flags[slot]),
                    "steps": int(self.step_counts[slot]),
                    "wall_collisions": int(
                        self.wall_collisions[slot]
                    ),
                    "optimal_path_length": (
                        None
                        if self.optimal_path_lengths[slot] is None
                        else int(self.optimal_path_lengths[slot])
                    ),
                    "path_efficiency": float(path_efficiency),
                }
            )

        return (
            self.observations.copy(),
            rewards,
            transition_terminated,
            transition_truncated,
            infos,
        )
