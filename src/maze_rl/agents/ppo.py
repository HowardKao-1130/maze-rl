from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from maze_rl.agents.networks import ActorCriticNetwork


class PPOAgent:
    def __init__(
        self,
        height: int,
        width: int,
        action_count: int,
        device: torch.device,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        learning_rate: float = 3e-4,
        clip_epsilon: float = 0.2,
        value_coefficient: float = 0.5,
        entropy_coefficient: float = 0.01,
        update_epochs: int = 4,
        minibatch_size: int = 64,
    ) -> None:
        self.device = device

        self.gamma = gamma
        self.gae_lambda = gae_lambda

        self.clip_epsilon = clip_epsilon

        self.value_coefficient = (
            value_coefficient
        )

        self.entropy_coefficient = (
            entropy_coefficient
        )

        self.update_epochs = update_epochs
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
            float(
                distribution
                .log_prob(action)
                .item()
            ),
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

    def update(
        self,
        observations,
        actions,
        old_log_probabilities,
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

        old_log_probabilities_tensor = (
            torch.as_tensor(
                old_log_probabilities,
                dtype=torch.float32,
                device=self.device,
            )
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

        total_loss = 0.0
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_approximate_kl = 0.0
        total_clip_fraction = 0.0
        total_value = 0.0
        minibatches = 0

        for _ in range(
            self.update_epochs
        ):
            indices = torch.randperm(
                len(actions_tensor),
                device=self.device,
            )

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

                ratio = torch.exp(
                    new_log_probabilities
                    - old_log_probabilities_tensor[
                        minibatch_indices
                    ]
                )

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

                value_loss = (
                    nn.functional.mse_loss(
                        values,
                        returns_tensor[
                            minibatch_indices
                        ],
                    )
                )

                loss = (
                    policy_loss
                    + self.value_coefficient
                    * value_loss
                    - self.entropy_coefficient
                    * entropy
                )

                with torch.no_grad():
                    log_ratio = (
                        new_log_probabilities
                        - old_log_probabilities_tensor[
                            minibatch_indices
                        ]
                    )
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
                    self.network.parameters(),
                    max_norm=0.5,
                )

                self.optimizer.step()

                total_loss += float(
                    loss.item()
                )
                total_policy_loss += float(
                    policy_loss.item()
                )
                total_value_loss += float(
                    value_loss.item()
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
                total_value += float(
                    values.mean().item()
                )
                minibatches += 1

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
            "approximate_kl": (
                total_approximate_kl
                / minibatches
            ),
            "clip_fraction": (
                total_clip_fraction
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
