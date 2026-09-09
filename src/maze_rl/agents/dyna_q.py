from __future__ import annotations

import numpy as np

from maze_rl.agents.q_learning import QLearningAgent


class DynaQAgent(QLearningAgent):
    def __init__(
        self,
        action_count: int,
        alpha: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 0.1,
        planning_steps: int = 10,
        seed: int = 42,
    ) -> None:
        super().__init__(
            action_count=action_count,
            alpha=alpha,
            gamma=gamma,
            epsilon=epsilon,
            seed=seed,
        )

        self.planning_steps = planning_steps

        self.model = {}

    def update(
        self,
        state,
        action: int,
        reward: float,
        next_state,
        terminated: bool,
    ) -> float:
        td_errors = []

        # Real experience update.
        td_errors.append(
            super().update(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                terminated=terminated,
            )
        )

        # Store learned transition model.
        self.model[(state, action)] = (
            reward,
            next_state,
            terminated,
        )

        model_keys = list(
            self.model.keys()
        )

        # Planning updates.
        for _ in range(self.planning_steps):
            index = int(
                self.rng.integers(
                    len(model_keys)
                )
            )

            (
                simulated_state,
                simulated_action,
            ) = model_keys[index]

            (
                simulated_reward,
                simulated_next_state,
                simulated_terminated,
            ) = self.model[
                (
                    simulated_state,
                    simulated_action,
                )
            ]

            td_errors.append(
                super().update(
                    state=simulated_state,
                    action=simulated_action,
                    reward=simulated_reward,
                    next_state=simulated_next_state,
                    terminated=simulated_terminated,
                )
            )

        return float(
            np.mean(np.abs(td_errors))
        )
