from __future__ import annotations

from maze_rl.agents.tabular import TabularAgent


class SarsaAgent(TabularAgent):
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
        next_action: int | None,
        terminated: bool,
    ) -> float:
        target = reward

        if not terminated:
            if next_action is None:
                raise ValueError(
                    "next_action is required for non-terminal Sarsa updates"
                )

            target += (
                self.gamma
                * self.q[next_state][next_action]
            )

        td_error = (
            target
            - self.q[state][action]
        )

        self.q[state][action] += (
            self.alpha * td_error
        )

        return float(td_error)
