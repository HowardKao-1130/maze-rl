from __future__ import annotations

import numpy as np
import pytest
import torch

from maze_rl.agents.a2c import A2CAgent
from maze_rl.agents.dqn import DQNAgent
from maze_rl.agents.dqn import Transition
from maze_rl.agents.reinforce import ReinforceAgent
from maze_rl.agents.sarsa import SarsaAgent
from maze_rl.training.trainers import train_grpo_round
from maze_rl.training.trainers import train_sarsa_episode
from maze_rl.training.trainers import train_reinforce_round
from scripts.train import HierarchicalTaskSampler


class OneStepGoalEnv:
    def __init__(self) -> None:
        self.state = "start"

    def reset(
        self,
        *,
        options=None,
    ):
        self.state = "start"
        return None, {}

    def tabular_state(self):
        return self.state

    def step(self, action):
        self.state = "goal"
        return (
            None,
            1.0,
            True,
            False,
            {
                "task_index": 0,
                "layout_index": 0,
                "success": True,
                "steps": 1,
                "wall_collisions": 0,
                "optimal_path_length": 1,
                "path_efficiency": 1.0,
            },
        )


def test_sarsa_episode_does_not_materialize_terminal_goal_state():
    agent = SarsaAgent(
        action_count=4,
        seed=123,
    )

    train_sarsa_episode(
        env=OneStepGoalEnv(),
        agent=agent,
        episode=1,
        epoch=1,
        task_index=0,
    )

    assert "start" in agent.q
    assert "goal" not in agent.q


class OneStepArrayEnv:
    def reset(
        self,
        *,
        options=None,
    ):
        return np.zeros(
            (3, 1, 1),
            dtype=np.float32,
        ), {}

    def step(self, action):
        return (
            np.zeros(
                (3, 1, 1),
                dtype=np.float32,
            ),
            1.0,
            True,
            False,
            {
                "task_index": 0,
                "layout_index": 0,
                "success": True,
                "steps": 1,
                "wall_collisions": 0,
                "optimal_path_length": 1,
                "path_efficiency": 1.0,
            },
        )


class OneStepRewardEnv:
    def reset(
        self,
        *,
        options=None,
    ):
        self.task_index = (
            options or {}
        ).get(
            "task_index",
            0,
        )

        return np.zeros(
            (3, 1, 1),
            dtype=np.float32,
        ), {}

    def step(self, action):
        reward = float(action)

        return (
            np.zeros(
                (3, 1, 1),
                dtype=np.float32,
            ),
            reward,
            True,
            False,
            {
                "task_index": self.task_index,
                "layout_index": 0,
                "success": True,
                "steps": 1,
                "wall_collisions": 0,
                "optimal_path_length": 1,
                "path_efficiency": 1.0,
            },
        )


class FakeReinforceAgent:
    minibatch_size = 2

    def choose_action(self, observation):
        return 0

    def compute_returns(self, rewards):
        return rewards

    def update_rollout(
        self,
        observations,
        actions,
        returns,
    ):
        assert len(observations) == 3
        assert len(actions) == 3
        assert len(returns) == 3
        return 7.0


def test_reinforce_round_batches_metric_accounting():
    metrics = train_reinforce_round(
        env=OneStepArrayEnv(),
        agent=FakeReinforceAgent(),
        episode_specs=[
            (1, 1, 0),
            (2, 1, 0),
            (3, 1, 0),
        ],
    )

    assert [
        metric.internal_updates
        for metric in metrics
    ] == [
        None,
        None,
        2,
    ]
    assert [
        metric.loss
        for metric in metrics
    ] == [
        None,
        None,
        7.0,
    ]


def test_dqn_train_frequency_throttles_optimizer_updates():
    agent = DQNAgent(
        height=1,
        width=1,
        action_count=2,
        device=torch.device("cpu"),
        batch_size=2,
        min_replay_size=2,
        train_frequency=3,
        epsilon_decay=0.9,
        seed=123,
    )
    observation = np.zeros(
        (3, 1, 1),
        dtype=np.float32,
    )

    update_results = []

    for _ in range(6):
        agent.store_transition(
            Transition(
                state=observation,
                action=0,
                reward=0.0,
                next_state=observation,
                terminated=False,
            )
        )
        update_results.append(
            agent.train_step() is not None
        )

    assert update_results == [
        False,
        False,
        True,
        False,
        False,
        True,
    ]
    assert agent.training_steps == 2
    assert agent.environment_steps == 6
    assert agent.epsilon == pytest.approx(
        0.9**5
    )


class FakeGRPOAgent:
    minibatch_size = 2
    gamma = 1.0

    def __init__(self) -> None:
        self.received_advantages = None

    def choose_action(self, observation):
        return 0, -0.5

    def compute_episode_return(
        self,
        rewards,
    ):
        return rewards[0]

    def update(
        self,
        observations,
        actions,
        old_log_probabilities,
        advantages,
    ):
        self.received_advantages = advantages

        return {
            "loss": 3.0,
            "policy_loss": 2.0,
            "entropy": 0.1,
            "mean_advantage": float(
                np.mean(advantages)
            ),
        }


class SequenceGRPOAgent(FakeGRPOAgent):
    def __init__(
        self,
        actions,
    ) -> None:
        super().__init__()
        self.actions = list(actions)

    def choose_action(self, observation):
        return self.actions.pop(0), -0.5


def test_grpo_round_uses_group_relative_advantages():
    agent = FakeGRPOAgent()

    metrics = train_grpo_round(
        env=OneStepArrayEnv(),
        agent=agent,
        rollout_groups=[
            (1, 1, 0),
            (2, 1, 0),
            (3, 1, 0),
        ],
    )

    assert agent.received_advantages == [
        0.0,
        0.0,
        0.0,
    ]
    assert [
        metric.internal_updates
        for metric in metrics
    ] == [
        None,
        None,
        2,
    ]
    assert metrics[-1].loss == 3.0


def test_grpo_round_normalizes_advantages_within_each_task_group():
    agent = SequenceGRPOAgent(
        actions=[
            1,
            3,
            10,
            14,
        ]
    )

    metrics = train_grpo_round(
        env=OneStepRewardEnv(),
        agent=agent,
        rollout_groups=[
            [
                (1, 1, 0),
                (2, 1, 0),
            ],
            [
                (3, 1, 1),
                (4, 1, 1),
            ],
        ],
    )

    assert agent.received_advantages == [
        -1.0,
        1.0,
        -1.0,
        1.0,
    ]
    assert [
        metric.task_index
        for metric in metrics
    ] == [
        0,
        0,
        1,
        1,
    ]


def test_hierarchical_sampler_allows_collection_to_span_dataset_epochs():
    sampler = HierarchicalTaskSampler(
        task_indices=[0],
        layout_indices=[0],
        seed=123,
    )

    specs = sampler.sample_collection(
        collection_size=16,
        target_dataset_epochs=2,
    )

    assert len(specs) == 2
    assert [
        spec.task_index
        for spec in specs
    ] == [
        0,
        0,
    ]
    assert [
        spec.dataset_epoch
        for spec in specs
    ] == [
        1,
        2,
    ]


def test_a2c_gae_bootstraps_truncation_but_not_termination():
    agent = A2CAgent(
        height=1,
        width=1,
        action_count=2,
        device=torch.device("cpu"),
        gamma=1.0,
        gae_lambda=1.0,
    )

    truncated_advantages, truncated_returns = (
        agent.compute_gae(
            rewards=[1.0],
            values=[2.0],
            terminated_flags=[False],
            last_value=5.0,
        )
    )

    terminated_advantages, terminated_returns = (
        agent.compute_gae(
            rewards=[1.0],
            values=[2.0],
            terminated_flags=[True],
            last_value=5.0,
        )
    )

    assert truncated_advantages.tolist() == [
        4.0
    ]
    assert truncated_returns.tolist() == [
        6.0
    ]
    assert terminated_advantages.tolist() == [
        -1.0
    ]
    assert terminated_returns.tolist() == [
        1.0
    ]


def test_reinforce_entropy_coefficient_decays_to_floor():
    agent = ReinforceAgent(
        height=1,
        width=1,
        action_count=2,
        device=torch.device("cpu"),
        entropy_coefficient=0.05,
        entropy_coefficient_min=0.02,
        entropy_coefficient_decay=0.5,
    )

    agent.decay_entropy_coefficient()
    assert agent.entropy_coefficient == 0.025

    agent.decay_entropy_coefficient()
    assert agent.entropy_coefficient == 0.02


def test_a2c_entropy_coefficient_decays_to_floor():
    agent = A2CAgent(
        height=1,
        width=1,
        action_count=2,
        device=torch.device("cpu"),
        entropy_coefficient=0.05,
        entropy_coefficient_min=0.02,
        entropy_coefficient_decay=0.5,
    )

    agent.decay_entropy_coefficient()
    assert agent.entropy_coefficient == 0.025

    agent.decay_entropy_coefficient()
    assert agent.entropy_coefficient == 0.02
