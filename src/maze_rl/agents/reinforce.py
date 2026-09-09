from __future__ import annotations

import numpy as np
import torch
from torch.distributions import Categorical

from maze_rl.agents.networks import PolicyNetwork


class ReinforceAgent:
    def __init__(
        self,
        height: int,
        width: int,
        action_count: int,
        device: torch.device,
        gamma: float = 0.99,
        learning_rate: float = 3e-4,
        entropy_coefficient: float = 0.05,
        entropy_coefficient_min: float = 0.01,
        entropy_coefficient_decay: float = 0.9995,
        minibatch_size: int = 64,
    ) -> None:
        self.device = device
        self.gamma = gamma
        self.entropy_coefficient = (
            entropy_coefficient
        )
        self.entropy_coefficient_min = (
            entropy_coefficient_min
        )
        self.entropy_coefficient_decay = (
            entropy_coefficient_decay
        )
        self.minibatch_size = minibatch_size

        self.policy = PolicyNetwork(
            height,
            width,
            action_count,
        ).to(device)

        self.optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=learning_rate,
        )

    def decay_entropy_coefficient(
        self,
    ) -> None:
        self.entropy_coefficient = max(
            self.entropy_coefficient_min,
            self.entropy_coefficient
            * self.entropy_coefficient_decay,
        )

    @torch.no_grad()
    def choose_action(
        self,
        observation: np.ndarray,
    ) -> int:
        observation_tensor = torch.as_tensor(
            observation,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        logits = self.policy(
            observation_tensor
        )

        distribution = Categorical(
            logits=logits
        )

        action = distribution.sample()

        return int(action.item())

    def compute_returns(
        self,
        rewards,
    ):
        returns = []

        G = 0.0

        for reward in reversed(rewards):
            G = reward + self.gamma * G
            returns.append(G)

        returns.reverse()

        return returns

    def update_rollout(
        self,
        observations,
        actions,
        returns,
    ) -> float:
        observations_tensor = (
            torch.as_tensor(
                np.stack(observations),
                dtype=torch.float32,
                device=self.device,
            )
        )

        actions_tensor = torch.as_tensor(
            actions,
            dtype=torch.long,
            device=self.device,
        )

        returns_tensor = torch.as_tensor(
            returns,
            dtype=torch.float32,
            device=self.device,
        )
        raw_returns_tensor = returns_tensor

        if len(returns_tensor) > 1:
            returns_tensor = (
                returns_tensor
                - returns_tensor.mean()
            ) / (
                returns_tensor.std()
                + 1e-8
            )

        indices = torch.randperm(
            len(actions_tensor),
            device=self.device,
        )

        total_loss = 0.0
        total_policy_loss = 0.0
        total_entropy = 0.0
        minibatches = 0

        for start in range(
            0,
            len(indices),
            self.minibatch_size,
        ):
            minibatch_indices = indices[
                start:start + self.minibatch_size
            ]

            logits = self.policy(
                observations_tensor[
                    minibatch_indices
                ]
            )

            distribution = Categorical(
                logits=logits
            )

            log_probabilities = (
                distribution.log_prob(
                    actions_tensor[
                        minibatch_indices
                    ]
                )
            )

            entropy = (
                distribution
                .entropy()
                .mean()
            )

            policy_loss = -(
                log_probabilities
                * returns_tensor[
                    minibatch_indices
                ]
            ).mean()

            loss = (
                policy_loss
                - self.entropy_coefficient
                * entropy
            )

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += float(loss.item())
            total_policy_loss += float(
                policy_loss.item()
            )
            total_entropy += float(
                entropy.item()
            )
            minibatches += 1

        self.decay_entropy_coefficient()

        return {
            "loss": total_loss / minibatches,
            "policy_loss": (
                total_policy_loss
                / minibatches
            ),
            "entropy": (
                total_entropy
                / minibatches
            ),
            "mean_return": float(
                raw_returns_tensor.mean().item()
            ),
        }
