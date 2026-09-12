from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from maze_rl.agents.networks import PolicyNetwork


class GRPOAgent:
    def __init__(
        self,
        height: int,
        width: int,
        action_count: int,
        device: torch.device,
        gamma: float = 0.99,
        learning_rate: float = 3e-4,
        clip_epsilon: float = 0.2,
        entropy_coefficient: float = 0.02,
        entropy_coefficient_min: float = 0.0,
        entropy_coefficient_decay: float = 0.9995,
        minibatch_size: int = 64,
    ) -> None:
        self.device = device
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.entropy_coefficient = entropy_coefficient
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
    ):
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

        return (
            int(action.item()),
            float(
                distribution.log_prob(
                    action
                ).item()
            ),
        )

    def compute_episode_return(
        self,
        rewards,
    ) -> float:
        episode_return = 0.0

        for reward in reversed(rewards):
            episode_return = (
                reward
                + self.gamma
                * episode_return
            )

        return float(episode_return)

    def update(
        self,
        observations,
        actions,
        old_log_probabilities,
        advantages,
    ) -> dict[str, float]:
        observations_tensor = torch.as_tensor(
            np.stack(observations),
            dtype=torch.float32,
            device=self.device,
        )
        actions_tensor = torch.as_tensor(
            actions,
            dtype=torch.long,
            device=self.device,
        )
        old_log_probabilities_tensor = torch.as_tensor(
            old_log_probabilities,
            dtype=torch.float32,
            device=self.device,
        )
        advantages_tensor = torch.as_tensor(
            advantages,
            dtype=torch.float32,
            device=self.device,
        )

        indices = torch.randperm(
            len(actions_tensor),
            device=self.device,
        )

        total_loss = 0.0
        total_policy_loss = 0.0
        total_entropy = 0.0
        total_approximate_kl = 0.0
        total_clip_fraction = 0.0
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
            new_log_probabilities = (
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

            log_ratio = (
                new_log_probabilities
                - old_log_probabilities_tensor[
                    minibatch_indices
                ]
            )
            ratio = torch.exp(log_ratio)
            minibatch_advantages = (
                advantages_tensor[
                    minibatch_indices
                ]
            )
            unclipped = (
                ratio
                * minibatch_advantages
            )
            clipped = (
                torch.clamp(
                    ratio,
                    1.0 - self.clip_epsilon,
                    1.0 + self.clip_epsilon,
                )
                * minibatch_advantages
            )
            policy_loss = -torch.min(
                unclipped,
                clipped,
            ).mean()
            loss = (
                policy_loss
                - self.entropy_coefficient
                * entropy
            )

            with torch.no_grad():
                approximate_kl = (
                    (ratio - 1.0)
                    - log_ratio
                ).mean()
                clip_fraction = (
                    (
                        torch.abs(
                            ratio - 1.0
                        )
                        > self.clip_epsilon
                    )
                    .float()
                    .mean()
                )

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                self.policy.parameters(),
                max_norm=0.5,
            )
            self.optimizer.step()

            total_loss += float(loss.item())
            total_policy_loss += float(
                policy_loss.item()
            )
            total_entropy += float(
                entropy.item()
            )
            total_approximate_kl += float(
                approximate_kl.item()
            )
            total_clip_fraction += float(
                clip_fraction.item()
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
            "approximate_kl": (
                total_approximate_kl
                / minibatches
            ),
            "clip_fraction": (
                total_clip_fraction
                / minibatches
            ),
            "mean_advantage": float(
                advantages_tensor.mean().item()
            ),
        }
