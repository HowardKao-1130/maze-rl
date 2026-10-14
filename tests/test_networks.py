import pytest
import torch
from torch import nn

from maze_rl.agents.networks import (
    ActorCriticNetwork,
    MazeEncoder,
    ResidualBlock,
)


def test_baseline_encoder_preserves_current_shape_and_layers():
    encoder = MazeEncoder(11, 11)

    assert tuple(
        encoder(torch.zeros(2, 3, 11, 11)).shape
    ) == (2, 128)
    assert not any(
        isinstance(module, ResidualBlock)
        for module in encoder.conv
    )
    assert sum(
        isinstance(module, nn.Conv2d)
        for module in encoder.conv
    ) == 2


def test_residual_encoder_defaults_to_two_blocks():
    encoder = MazeEncoder(
        11,
        11,
        network_architecture="residual",
    )

    assert tuple(
        encoder(torch.zeros(2, 3, 11, 11)).shape
    ) == (2, 128)
    assert sum(
        isinstance(module, ResidualBlock)
        for module in encoder.conv
    ) == 2


def test_residual_encoder_accepts_configurable_block_count():
    encoder = MazeEncoder(
        11,
        11,
        network_architecture="residual",
        residual_layers=4,
    )

    assert sum(
        isinstance(module, ResidualBlock)
        for module in encoder.conv
    ) == 4


def test_residual_actor_critic_has_expected_output_shapes():
    network = ActorCriticNetwork(
        11,
        11,
        4,
        network_architecture="residual",
        residual_layers=3,
    )

    logits, values = network(
        torch.zeros(5, 3, 11, 11)
    )

    assert tuple(logits.shape) == (5, 4)
    assert tuple(values.shape) == (5,)


@pytest.mark.parametrize(
    "residual_layers",
    [0, -1],
)
def test_residual_encoder_rejects_non_positive_layer_count(
    residual_layers,
):
    with pytest.raises(
        ValueError,
        match="residual_layers must be positive",
    ):
        MazeEncoder(
            11,
            11,
            network_architecture="residual",
            residual_layers=residual_layers,
        )
