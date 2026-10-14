from __future__ import annotations

import torch
from torch import nn


NETWORK_ARCHITECTURES = (
    "baseline",
    "residual",
)
DEFAULT_RESIDUAL_LAYERS = 2


def validate_network_architecture(
    network_architecture: str,
    residual_layers: int,
) -> None:
    if network_architecture not in NETWORK_ARCHITECTURES:
        choices = ", ".join(NETWORK_ARCHITECTURES)
        raise ValueError(
            "network_architecture must be one of "
            f"{choices}, got {network_architecture!r}."
        )

    if residual_layers <= 0:
        raise ValueError(
            "residual_layers must be positive."
        )


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
            ),
        )
        self.activation = nn.ReLU()

    def forward(
        self,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        return self.activation(
            self.layers(inputs) + inputs
        )


class MazeEncoder(nn.Module):
    def __init__(
        self,
        height: int,
        width: int,
        feature_dim: int = 128,
        network_architecture: str = "baseline",
        residual_layers: int = DEFAULT_RESIDUAL_LAYERS,
    ) -> None:
        super().__init__()

        validate_network_architecture(
            network_architecture,
            residual_layers,
        )

        if network_architecture == "baseline":
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
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(
                    3,
                    64,
                    kernel_size=3,
                    padding=1,
                ),
                nn.ReLU(),
                *(
                    ResidualBlock(64)
                    for _ in range(residual_layers)
                ),
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
        network_architecture: str = "baseline",
        residual_layers: int = DEFAULT_RESIDUAL_LAYERS,
    ) -> None:
        super().__init__()

        self.encoder = MazeEncoder(
            height,
            width,
            network_architecture=network_architecture,
            residual_layers=residual_layers,
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
        network_architecture: str = "baseline",
        residual_layers: int = DEFAULT_RESIDUAL_LAYERS,
    ) -> None:
        super().__init__()

        self.encoder = MazeEncoder(
            height,
            width,
            network_architecture=network_architecture,
            residual_layers=residual_layers,
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
        network_architecture: str = "baseline",
        residual_layers: int = DEFAULT_RESIDUAL_LAYERS,
    ) -> None:
        super().__init__()

        self.encoder = MazeEncoder(
            height,
            width,
            network_architecture=network_architecture,
            residual_layers=residual_layers,
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
