from __future__ import annotations

from collections import defaultdict

from maze_rl.agents.tabular import TabularAgent


class MonteCarloAgent(TabularAgent):
    def __init__(
        self,
        action_count: int,
        gamma: float = 0.99,
        epsilon: float = 0.1,
        seed: int = 42,
    ) -> None:
        super().__init__(
            action_count=action_count,
            epsilon=epsilon,
            seed=seed,
        )

        self.gamma = gamma
        self.visit_counts = defaultdict(int)

    def update_episode(
        self,
        trajectory,
    ) -> int:
        """
        First-visit Monte Carlo control.

        trajectory:
            [(state, action, reward), ...]
        """

        returns = []

        G = 0.0

        for state, action, reward in reversed(
            trajectory
        ):
            G = reward + self.gamma * G

            returns.append(
                (state, action, G)
            )

        returns.reverse()

        visited = set()
        update_count = 0

        for state, action, G in returns:
            key = (state, action)

            if key in visited:
                continue

            visited.add(key)

            self.visit_counts[key] += 1

            n = self.visit_counts[key]

            self.q[state][action] += (
                G - self.q[state][action]
            ) / n

            update_count += 1

        return update_count
