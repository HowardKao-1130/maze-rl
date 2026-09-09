from __future__ import annotations

from maze_rl.agents.tabular import TabularAgent


class QLearningAgent(TabularAgent):
    def __init__(
        self,
        action_count: int,
        alpha: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 0.1,
        seed: int = 42,
    ) -> None:
        super().__init__(
            action_count=action_count,
            epsilon=epsilon,
            seed=seed,
        )

        self.alpha = alpha
        self.gamma = gamma

    def update(
        self,
        state,
        action: int,
        reward: float,
        next_state,
        terminated: bool,
    ) -> float:
        target = reward

        if not terminated:
            target += (
                self.gamma
                * self.q[next_state].max()
            )

        td_error = (
            target
            - self.q[state][action]
        )

        self.q[state][action] += (
            self.alpha * td_error
        )

        return float(td_error)
