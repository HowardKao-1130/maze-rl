from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from maze_rl.agents.networks import QNetwork


@dataclass
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    terminated: bool


class ReplayBuffer:
    def __init__(
        self,
        capacity: int,
        seed: int = 42,
    ) -> None:
        self.buffer = deque(
            maxlen=capacity
        )

        self.rng = np.random.default_rng(
            seed
        )

    def __len__(self) -> int:
        return len(self.buffer)

    def add(
        self,
        transition: Transition,
    ) -> None:
        self.buffer.append(transition)

    def sample(
        self,
        batch_size: int,
    ):
        indices = self.rng.choice(
            len(self.buffer),
            size=batch_size,
            replace=False,
        )

        return [
            self.buffer[int(index)]
            for index in indices
        ]


class DQNAgent:
    def __init__(
        self,
        height: int,
        width: int,
        action_count: int,
        device: torch.device,
        gamma: float = 0.99,
        learning_rate: float = 3e-4,
        epsilon_start: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.9995,
        replay_capacity: int = 12_800,
        min_replay_size: int = 1_000,
        batch_size: int = 64,
        target_update_interval: int = 500,
        seed: int = 42,
    ) -> None:
        self.action_count = action_count
        self.device = device
        self.gamma = gamma

        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.batch_size = batch_size
        self.min_replay_size = min_replay_size

        self.target_update_interval = (
            target_update_interval
        )

        self.training_steps = 0

        self.rng = np.random.default_rng(seed)

        self.online_network = QNetwork(
            height,
            width,
            action_count,
        ).to(device)

        self.target_network = QNetwork(
            height,
            width,
            action_count,
        ).to(device)

        self.target_network.load_state_dict(
            self.online_network.state_dict()
        )

        self.target_network.eval()

        self.optimizer = torch.optim.Adam(
            self.online_network.parameters(),
            lr=learning_rate,
        )

        self.loss_function = (
            nn.SmoothL1Loss()
        )

        self.replay_buffer = ReplayBuffer(
            capacity=replay_capacity,
            seed=seed,
        )

    def choose_action(
        self,
        observation: np.ndarray,
        explore: bool = True,
    ) -> int:
        if (
            explore
            and self.rng.random()
            < self.epsilon
        ):
            return int(
                self.rng.integers(
                    self.action_count
                )
            )

        observation_tensor = torch.as_tensor(
            observation,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        with torch.no_grad():
            q_values = self.online_network(
                observation_tensor
            )

        return int(
            q_values.argmax(
                dim=1
            ).item()
        )

    def store_transition(
        self,
        transition: Transition,
    ) -> None:
        self.replay_buffer.add(
            transition
        )

    def train_step(
        self,
    ) -> dict | None:
        if (
            len(self.replay_buffer)
            < max(
                self.batch_size,
                self.min_replay_size,
            )
        ):
            return None

        batch = self.replay_buffer.sample(
            self.batch_size
        )

        states = torch.as_tensor(
            np.stack(
                [
                    transition.state
                    for transition in batch
                ]
            ),
            dtype=torch.float32,
            device=self.device,
        )

        actions = torch.as_tensor(
            [
                transition.action
                for transition in batch
            ],
            dtype=torch.long,
            device=self.device,
        )

        rewards = torch.as_tensor(
            [
                transition.reward
                for transition in batch
            ],
            dtype=torch.float32,
            device=self.device,
        )

        next_states = torch.as_tensor(
            np.stack(
                [
                    transition.next_state
                    for transition in batch
                ]
            ),
            dtype=torch.float32,
            device=self.device,
        )

        terminated = torch.as_tensor(
            [
                transition.terminated
                for transition in batch
            ],
            dtype=torch.float32,
            device=self.device,
        )

        q_values = self.online_network(
            states
        )

        selected_q_values = q_values.gather(
            1,
            actions.unsqueeze(1),
        ).squeeze(1)

        with torch.no_grad():
            next_q_values = (
                self.target_network(
                    next_states
                )
                .max(dim=1)
                .values
            )

            targets = (
                rewards
                + self.gamma
                * (1.0 - terminated)
                * next_q_values
            )

        loss = self.loss_function(
            selected_q_values,
            targets,
        )

        self.optimizer.zero_grad()
        loss.backward()

        nn.utils.clip_grad_norm_(
            self.online_network.parameters(),
            max_norm=10.0,
        )

        self.optimizer.step()

        self.training_steps += 1

        self.epsilon = max(
            self.epsilon_min,
            self.epsilon
            * self.epsilon_decay,
        )

        if (
            self.training_steps
            % self.target_update_interval
            == 0
        ):
            self.target_network.load_state_dict(
                self.online_network.state_dict()
            )

        return {
            "loss": float(loss.item()),
            "mean_q_value": float(
                selected_q_values
                .mean()
                .item()
            ),
            "max_q_value": float(
                q_values.max().item()
            ),
            "replay_size": len(
                self.replay_buffer
            ),
        }
