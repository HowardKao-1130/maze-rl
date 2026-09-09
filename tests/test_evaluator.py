from __future__ import annotations

import numpy as np

from maze_rl.evaluation.evaluator import evaluate


class OneStepEnv:
    def __init__(self) -> None:
        self.num_tasks = 5
        self.np_random = np.random.default_rng()
        self.task_index = 0
        self.layout_index = 0

    def reset(
        self,
        *,
        seed=None,
        options=None,
    ):
        if seed is not None:
            self.np_random = np.random.default_rng(seed)

        if options is not None:
            self.task_index = int(options["task_index"])
        else:
            self.task_index = int(
                self.np_random.integers(self.num_tasks)
            )

        self.layout_index = self.task_index
        return np.zeros((1,), dtype=np.float32), {}

    def step(self, action):
        return (
            np.zeros((1,), dtype=np.float32),
            float(self.task_index),
            True,
            False,
            {
                "task_index": self.task_index,
                "layout_index": self.layout_index,
                "success": True,
                "steps": 1,
                "wall_collisions": 0,
                "optimal_path_length": 1,
                "path_efficiency": 1.0,
            },
        )

    def tabular_state(self):
        return self.task_index


class GreedyAgent:
    device = "cpu"

    def choose_action(
        self,
        state,
        explore=True,
    ):
        return 0


def task_indices(results):
    return [
        result["task_index"]
        for result in results
    ]


def test_evaluate_seed_repeats_random_task_sample():
    first_results, _ = evaluate(
        algorithm="q_learning",
        agent=GreedyAgent(),
        env=OneStepEnv(),
        episodes=8,
        seed=123,
    )
    second_results, _ = evaluate(
        algorithm="q_learning",
        agent=GreedyAgent(),
        env=OneStepEnv(),
        episodes=8,
        seed=123,
    )

    assert task_indices(first_results) == task_indices(
        second_results
    )


def test_evaluate_task_indices_override_random_sampling():
    results, summary = evaluate(
        algorithm="q_learning",
        agent=GreedyAgent(),
        env=OneStepEnv(),
        episodes=99,
        seed=123,
        task_indices=[4, 2, 0],
    )

    assert task_indices(results) == [4, 2, 0]
    assert summary["episodes"] == 3
