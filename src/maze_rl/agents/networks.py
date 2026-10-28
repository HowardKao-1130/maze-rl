from __future__ import annotations

import torch
from torch import nn


NETWORK_ARCHITECTURES = (
    "baseline",
    "residual",
)
DEFAULT_RESIDUAL_LAYERS = 2
DEFAULT_KERNEL_SIZE = 3
DEFAULT_BASE_CHANNELS = 32
DEFAULT_CHANNEL_MULTIPLIER = 2
DEFAULT_CONV_DEPTH = 2
DEFAULT_FEATURE_DIM = 128


def validate_encoder_configuration(
    *,
    kernel_size: int,
    base_channels: int,
    channel_multiplier: int,
    conv_depth: int,
    feature_dim: int,
) -> None:
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError(
            "kernel_size must be a positive odd integer."
        )

    if base_channels <= 0:
        raise ValueError(
            "base_channels must be positive."
        )

    if channel_multiplier <= 0:
        raise ValueError(
            "channel_multiplier must be positive."
        )

    if conv_depth <= 0:
        raise ValueError(
            "conv_depth must be positive."
        )

    if feature_dim <= 0:
        raise ValueError(
            "feature_dim must be positive."
        )


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
    def __init__(
        self,
        channels: int,
        kernel_size: int = DEFAULT_KERNEL_SIZE,
    ) -> None:
        super().__init__()

        padding = kernel_size // 2

        self.layers = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=kernel_size,
                padding=padding,
            ),
            nn.ReLU(),
            nn.Conv2d(
                channels,
                channels,
                kernel_size=kernel_size,
                padding=padding,
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
        kernel_size: int = DEFAULT_KERNEL_SIZE,
        base_channels: int = DEFAULT_BASE_CHANNELS,
        channel_multiplier: int = DEFAULT_CHANNEL_MULTIPLIER,
        conv_depth: int = DEFAULT_CONV_DEPTH,
    ) -> None:
        super().__init__()

        validate_network_architecture(
            network_architecture,
            residual_layers,
        )
        validate_encoder_configuration(
            kernel_size=kernel_size,
            base_channels=base_channels,
            channel_multiplier=channel_multiplier,
            conv_depth=conv_depth,
            feature_dim=feature_dim,
        )

        padding = kernel_size // 2

        if network_architecture == "baseline":
            layers = []
            in_channels = 3
            for layer_index in range(conv_depth):
                out_channels = base_channels * (
                    channel_multiplier ** layer_index
                )
                layers.extend(
                    [
                        nn.Conv2d(
                            in_channels,
                            out_channels,
                            kernel_size=kernel_size,
                            padding=padding,
                        ),
                        nn.ReLU(),
                    ]
                )
                in_channels = out_channels

            self.conv = nn.Sequential(
                *layers,
                nn.Flatten(),
            )
        else:
            channels = base_channels * channel_multiplier
            self.conv = nn.Sequential(
                nn.Conv2d(
                    3,
                    channels,
                    kernel_size=kernel_size,
                    padding=padding,
                ),
                nn.ReLU(),
                *(
                    ResidualBlock(
                        channels,
                        kernel_size=kernel_size,
                    )
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
        feature_dim: int = DEFAULT_FEATURE_DIM,
        kernel_size: int = DEFAULT_KERNEL_SIZE,
        base_channels: int = DEFAULT_BASE_CHANNELS,
        channel_multiplier: int = DEFAULT_CHANNEL_MULTIPLIER,
        conv_depth: int = DEFAULT_CONV_DEPTH,
    ) -> None:
        super().__init__()

        self.encoder = MazeEncoder(
            height,
            width,
            network_architecture=network_architecture,
            residual_layers=residual_layers,
            feature_dim=feature_dim,
            kernel_size=kernel_size,
            base_channels=base_channels,
            channel_multiplier=channel_multiplier,
            conv_depth=conv_depth,
        )

        self.head = nn.Linear(
            feature_dim,
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
        feature_dim: int = DEFAULT_FEATURE_DIM,
        kernel_size: int = DEFAULT_KERNEL_SIZE,
        base_channels: int = DEFAULT_BASE_CHANNELS,
        channel_multiplier: int = DEFAULT_CHANNEL_MULTIPLIER,
        conv_depth: int = DEFAULT_CONV_DEPTH,
    ) -> None:
        super().__init__()

        self.encoder = MazeEncoder(
            height,
            width,
            network_architecture=network_architecture,
            residual_layers=residual_layers,
            feature_dim=feature_dim,
            kernel_size=kernel_size,
            base_channels=base_channels,
            channel_multiplier=channel_multiplier,
            conv_depth=conv_depth,
        )

        self.policy = nn.Linear(
            feature_dim,
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
        feature_dim: int = DEFAULT_FEATURE_DIM,
        kernel_size: int = DEFAULT_KERNEL_SIZE,
        base_channels: int = DEFAULT_BASE_CHANNELS,
        channel_multiplier: int = DEFAULT_CHANNEL_MULTIPLIER,
        conv_depth: int = DEFAULT_CONV_DEPTH,
    ) -> None:
        super().__init__()

        self.encoder = MazeEncoder(
            height,
            width,
            network_architecture=network_architecture,
            residual_layers=residual_layers,
            feature_dim=feature_dim,
            kernel_size=kernel_size,
            base_channels=base_channels,
            channel_multiplier=channel_multiplier,
            conv_depth=conv_depth,
        )

        self.actor = nn.Linear(
            feature_dim,
            action_count,
        )

        self.critic = nn.Linear(
            feature_dim,
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
