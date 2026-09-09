from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from maze_rl.agents.networks import ActorCriticNetwork


class A2CAgent:
    def __init__(
        self,
        height: int,
        width: int,
        action_count: int,
        device: torch.device,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        learning_rate: float = 1e-4,
        value_coefficient: float = 0.5,
        entropy_coefficient: float = 0.10,
        entropy_coefficient_min: float = 0.03,
        entropy_coefficient_decay: float = 0.9998,
        minibatch_size: int = 64,
    ) -> None:
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda

        self.value_coefficient = (
            value_coefficient
        )

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

        self.network = ActorCriticNetwork(
            height,
            width,
            action_count,
        ).to(device)

        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
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

        logits, value = self.network(
            observation_tensor
        )

        distribution = Categorical(
            logits=logits
        )

        action = distribution.sample()

        return (
            int(action.item()),
            float(value.item()),
        )

    @torch.no_grad()
    def estimate_value(
        self,
        observation: np.ndarray,
    ) -> float:
        observation_tensor = torch.as_tensor(
            observation,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        _, value = self.network(
            observation_tensor
        )

        return float(value.item())

    def compute_gae(
        self,
        rewards,
        values,
        terminated_flags,
        last_value: float,
    ):
        advantages = np.zeros(
            len(rewards),
            dtype=np.float32,
        )

        gae = 0.0

        for t in reversed(
            range(len(rewards))
        ):
            if t == len(rewards) - 1:
                next_value = last_value
            else:
                next_value = values[t + 1]

            nonterminal = (
                0.0
                if terminated_flags[t]
                else 1.0
            )

            delta = (
                rewards[t]
                + self.gamma
                * next_value
                * nonterminal
                - values[t]
            )

            gae = (
                delta
                + self.gamma
                * self.gae_lambda
                * nonterminal
                * gae
            )

            advantages[t] = gae

        returns = (
            advantages
            + np.asarray(
                values,
                dtype=np.float32,
            )
        )

        return advantages, returns

    def update_rollout(
        self,
        observations,
        actions,
        advantages,
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

        advantages_tensor = torch.as_tensor(
            advantages,
            dtype=torch.float32,
            device=self.device,
        )
        raw_advantages_tensor = (
            advantages_tensor
        )

        returns_tensor = torch.as_tensor(
            returns,
            dtype=torch.float32,
            device=self.device,
        )

        if len(advantages_tensor) > 1:
            advantages_tensor = (
                advantages_tensor
                - advantages_tensor.mean()
            ) / (
                advantages_tensor.std()
                + 1e-8
            )

        indices = torch.randperm(
            len(actions_tensor),
            device=self.device,
        )

        total_loss = 0.0
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_value = 0.0
        minibatches = 0

        for start in range(
            0,
            len(indices),
            self.minibatch_size,
        ):
            minibatch_indices = indices[
                start:start + self.minibatch_size
            ]

            logits, values = self.network(
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

            actor_loss = -(
                log_probabilities
                * advantages_tensor[
                    minibatch_indices
                ]
            ).mean()

            critic_loss = nn.functional.mse_loss(
                values,
                returns_tensor[
                    minibatch_indices
                ],
            )

            loss = (
                actor_loss
                + self.value_coefficient
                * critic_loss
                - self.entropy_coefficient
                * entropy
            )

            self.optimizer.zero_grad()
            loss.backward()

            nn.utils.clip_grad_norm_(
                self.network.parameters(),
                max_norm=0.5,
            )

            self.optimizer.step()

            total_loss += float(loss.item())
            total_policy_loss += float(
                actor_loss.item()
            )
            total_value_loss += float(
                critic_loss.item()
            )
            total_entropy += float(
                entropy.item()
            )
            total_value += float(
                values.mean().item()
            )
            minibatches += 1

        self.decay_entropy_coefficient()

        return {
            "loss": total_loss / minibatches,
            "policy_loss": (
                total_policy_loss
                / minibatches
            ),
            "value_loss": (
                total_value_loss
                / minibatches
            ),
            "entropy": (
                total_entropy
                / minibatches
            ),
            "mean_value": (
                total_value
                / minibatches
            ),
            "mean_return": float(
                returns_tensor.mean().item()
            ),
            "mean_advantage": float(
                raw_advantages_tensor.mean().item()
            ),
        }
