from __future__ import annotations

import torch
from torch import nn


class MazeEncoder(nn.Module):
    def __init__(
        self,
        height: int,
        width: int,
        feature_dim: int = 128,
    ) -> None:
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(
                3,
                32,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            dummy = torch.zeros(
                1,
                3,
                height,
                width,
            )

            flattened_size = (
                self.conv(dummy).shape[1]
            )

        self.fc = nn.Sequential(
            nn.Linear(
                flattened_size,
                feature_dim,
            ),
            nn.ReLU(),
        )

    def forward(
        self,
        observation: torch.Tensor,
    ) -> torch.Tensor:
        return self.fc(
            self.conv(observation)
        )


class QNetwork(nn.Module):
    def __init__(
        self,
        height: int,
        width: int,
        action_count: int,
    ) -> None:
        super().__init__()

        self.encoder = MazeEncoder(
            height,
            width,
        )

        self.head = nn.Linear(
            128,
            action_count,
        )

    def forward(
        self,
        observation: torch.Tensor,
    ) -> torch.Tensor:
        features = self.encoder(
            observation
        )

        return self.head(features)


class PolicyNetwork(nn.Module):
    def __init__(
        self,
        height: int,
        width: int,
        action_count: int,
    ) -> None:
        super().__init__()

        self.encoder = MazeEncoder(
            height,
            width,
        )

        self.policy = nn.Linear(
            128,
            action_count,
        )

    def forward(
        self,
        observation: torch.Tensor,
    ) -> torch.Tensor:
        features = self.encoder(
            observation
        )

        return self.policy(features)


class ActorCriticNetwork(nn.Module):
    def __init__(
        self,
        height: int,
        width: int,
        action_count: int,
    ) -> None:
        super().__init__()

        self.encoder = MazeEncoder(
            height,
            width,
        )

        self.actor = nn.Linear(
            128,
            action_count,
        )

        self.critic = nn.Linear(
            128,
            1,
        )

    def forward(
        self,
        observation: torch.Tensor,
    ):
        features = self.encoder(
            observation
        )

        logits = self.actor(features)

        value = self.critic(
            features
        ).squeeze(-1)

        return logits, value