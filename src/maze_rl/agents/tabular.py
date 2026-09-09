from __future__ import annotations

from collections import defaultdict

import numpy as np


class TabularAgent:
    def __init__(
        self,
        action_count: int,
        epsilon: float = 0.1,
        seed: int = 42,
    ) -> None:
        self.action_count = action_count
        self.epsilon = epsilon

        self.rng = np.random.default_rng(seed)

        self.q = defaultdict(
            lambda: np.zeros(
                action_count,
                dtype=np.float64,
            )
        )

    # epsilon-greedy action selection
    def choose_action(
        self,
        state,
        explore: bool = True,
    ) -> int:
        if (
            explore
            and self.rng.random() < self.epsilon
        ):
            return int(
                self.rng.integers(self.action_count)
            )

        values = self.q[state]

        best_actions = np.flatnonzero(
            values == values.max()
        )

        return int(
            self.rng.choice(best_actions)
        )

    def q_state_dict(self) -> dict:
        """
        Convert defaultdict into a normal serializable dict.
        """
        return {
            state: values.copy()
            for state, values in self.q.items()
        }

    def load_q_state_dict(self, state_dict: dict) -> None:
        self.q.clear()

        for state, values in state_dict.items():
            self.q[state] = np.asarray(
                values,
                dtype=np.float64,
            )