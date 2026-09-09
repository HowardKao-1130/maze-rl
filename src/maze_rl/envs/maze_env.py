from __future__ import annotations

from pathlib import Path

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from maze_rl.maze.pathfinding import shortest_path_length

class MazeEnv(gym.Env):
    """
    Maze navigation environment.

    Maze representation:
        0 = free
        1 = wall

    Observation:
        3 x H x W float32 array

        channel 0 = walls
        channel 1 = agent position
        channel 2 = goal position

    Actions:
        0 = up
        1 = down
        2 = left
        3 = right
    """

    metadata = {"render_modes": []}

    ACTION_DELTAS = {
        0: (-1, 0),
        1: (1, 0),
        2: (0, -1),
        3: (0, 1),
    }

    def __init__(
        self,
        dataset_path: str | Path,
        max_steps: int = 200,
        goal_reward: float = 1.0,
        step_penalty: float = -0.01,
        wall_penalty: float = -0.05,
        fixed_index: int | None = None,
    ):
        super().__init__()

        data = np.load(dataset_path)

        if "layouts" in data:
            self.layouts = data["layouts"]
            self.layout_indices = data[
                "layout_indices"
            ]
        else:
            self.layouts = data["mazes"]
            self.layout_indices = np.arange(
                len(data["starts"])
            )

        self.mazes = self.layouts
        self.starts = data["starts"]
        self.goals = data["goals"]

        self.num_layouts = len(self.layouts)
        self.num_tasks = len(self.starts)
        self.num_mazes = self.num_layouts

        self.height = self.layouts.shape[1]
        self.width = self.layouts.shape[2]

        self.max_steps = max_steps
        self.goal_reward = goal_reward
        self.step_penalty = step_penalty
        self.wall_penalty = wall_penalty
        self.fixed_index = fixed_index

        self.action_space = spaces.Discrete(4)

        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(3, self.height, self.width),
            dtype=np.float32,
        )

        self.task_index = 0
        self.layout_index = 0
        self.maze_index = 0
        self.maze = None
        self.agent_position = None
        self.goal_position = None
        self.steps = 0
        self.wall_collisions = 0
        self.optimal_path_length = None

    def _get_observation(self) -> np.ndarray:
        observation = np.zeros(
            (3, self.height, self.width),
            dtype=np.float32,
        )

        observation[0] = self.maze

        agent_row, agent_col = self.agent_position
        goal_row, goal_col = self.goal_position

        observation[1, agent_row, agent_col] = 1.0
        observation[2, goal_row, goal_col] = 1.0

        return observation

    def tabular_state(
        self,
    ) -> tuple[int, int, int, int, int]:
        """
        State used by tabular algorithms.

        layout_index and goal position are included so
        tabular agents do not merge tasks with different
        layouts or terminal goals.
        """
        row, col = self.agent_position
        goal_row, goal_col = self.goal_position
        return (
            self.layout_index,
            row,
            col,
            goal_row,
            goal_col,
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ):
        super().reset(seed=seed)

        if (
            options is not None
            and "task_index" in options
        ):
            self.task_index = int(
                options["task_index"]
            )
        elif self.fixed_index is None:
            self.task_index = int(
                self.np_random.integers(self.num_tasks)
            )
        else:
            self.task_index = self.fixed_index

        self.layout_index = int(
            self.layout_indices[self.task_index]
        )
        self.maze_index = self.layout_index

        self.maze = self.layouts[self.layout_index]

        self.agent_position = tuple(
            int(x) for x in self.starts[self.task_index]
        )

        self.goal_position = tuple(
            int(x) for x in self.goals[self.task_index]
        )

        self.steps = 0
        self.wall_collisions = 0
        self.optimal_path_length = shortest_path_length(
            self.maze,
            self.agent_position,
            self.goal_position,
        )
        return self._get_observation(), {}

    def step(self, action: int):
        self.steps += 1

        row, col = self.agent_position
        d_row, d_col = self.ACTION_DELTAS[action]

        next_row = row + d_row
        next_col = col + d_col

        hit_wall = (
            next_row < 0
            or next_row >= self.height
            or next_col < 0
            or next_col >= self.width
            or self.maze[next_row, next_col] == 1
        )

        if hit_wall:
            self.wall_collisions += 1
            reward = self.wall_penalty
        else:
            self.agent_position = (
                next_row,
                next_col,
            )
            reward = self.step_penalty

        terminated = (
            self.agent_position
            == self.goal_position
        )

        if terminated:
            reward = self.goal_reward

        truncated = (
            self.steps >= self.max_steps
            and not terminated
        )

        path_efficiency = 0.0
        if (
            terminated
            and self.optimal_path_length is not None
            and self.steps > 0
        ):
            path_efficiency = (
                self.optimal_path_length
                / self.steps
            )

        info = {
            "task_index": self.task_index,
            "layout_index": self.layout_index,
            "maze_index": self.maze_index,
            "hit_wall": hit_wall,
            "success": terminated,
            "steps": self.steps,
            "wall_collisions": self.wall_collisions,
            "optimal_path_length": self.optimal_path_length,
            "path_efficiency": path_efficiency,
        }

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            info,
        )
