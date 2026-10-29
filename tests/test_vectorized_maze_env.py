from __future__ import annotations

import numpy as np
import pytest
import torch

from maze_rl.envs.maze_env import MazeEnv
from maze_rl.envs.vectorized_maze_env import VectorizedMazeEnv
from maze_rl.agents.dqn import DQNAgent
from maze_rl.training.trainers import train_grpo_round
from maze_rl.training.trainers import train_dqn_round
from maze_rl.training.trainers import train_reinforce_round


def write_dataset(path):
    layouts = np.asarray(
        [
            [
                [0, 0, 1, 0],
                [1, 0, 1, 0],
                [0, 0, 0, 0],
                [0, 1, 1, 0],
            ],
            [
                [0, 0, 0, 0],
                [0, 1, 1, 0],
                [0, 0, 0, 0],
                [0, 0, 1, 0],
            ],
        ],
        dtype=np.int8,
    )
    np.savez(
        path,
        layouts=layouts,
        layout_indices=np.asarray([0, 1, 0]),
        starts=np.asarray([[0, 0], [0, 0], [3, 0]]),
        goals=np.asarray([[2, 3], [3, 3], [0, 1]]),
    )


def test_vectorized_step_matches_serial_envs(tmp_path):
    dataset_path = tmp_path / "tasks.npz"
    write_dataset(dataset_path)

    task_indices = np.asarray([0, 1, 2])
    serial_envs = [
        MazeEnv(dataset_path, max_steps=5)
        for _ in task_indices
    ]
    vector_env = VectorizedMazeEnv(
        dataset_path,
        max_steps=5,
    )

    vector_observations, _ = vector_env.reset(task_indices)
    serial_observations = []
    for serial_env, task_index in zip(
        serial_envs,
        task_indices,
    ):
        observation, _ = serial_env.reset(
            options={"task_index": int(task_index)}
        )
        serial_observations.append(observation)

    assert np.array_equal(
        vector_observations,
        np.asarray(serial_observations),
    )

    active = np.ones(len(task_indices), dtype=bool)
    action_batches = [
        np.asarray([1, 1, 3]),
        np.asarray([1, 3, 0]),
        np.asarray([3, 3, 1]),
        np.asarray([3, 0, 0]),
        np.asarray([0, 0, 2]),
    ]

    for actions in action_batches:
        (
            vector_observations,
            vector_rewards,
            vector_terminated,
            vector_truncated,
            vector_infos,
        ) = vector_env.step(actions)

        for slot in np.flatnonzero(active):
            (
                serial_observation,
                serial_reward,
                serial_terminated,
                serial_truncated,
                serial_info,
            ) = serial_envs[slot].step(int(actions[slot]))

            assert np.array_equal(
                vector_observations[slot],
                serial_observation,
            )
            assert vector_rewards[slot] == serial_reward
            assert vector_terminated[slot] == serial_terminated
            assert vector_truncated[slot] == serial_truncated
            assert vector_infos[slot] == serial_info

            if serial_terminated or serial_truncated:
                active[slot] = False

        if not np.any(active):
            break


def test_vectorized_reset_and_step_validate_batch_shapes(tmp_path):
    dataset_path = tmp_path / "tasks.npz"
    write_dataset(dataset_path)
    vector_env = VectorizedMazeEnv(dataset_path)
    vector_env.reset([0, 1])

    with pytest.raises(ValueError, match="actions must have shape"):
        vector_env.step([0])

    with pytest.raises(ValueError, match="integer action"):
        vector_env.step([0.0, 1.0])

    with pytest.raises(IndexError, match="outside the dataset"):
        vector_env.reset([3])


class OneStepBatchGRPOAgent:
    def __init__(self):
        self.batch_sizes = []
        self.received_advantages = None
        self.minibatch_size = 2

    def choose_actions(self, observations):
        self.batch_sizes.append(len(observations))
        return (
            np.full(len(observations), 3, dtype=np.int64),
            np.zeros(len(observations), dtype=np.float32),
        )

    def compute_episode_return(self, rewards):
        return float(sum(rewards))

    def update(
        self,
        observations,
        actions,
        old_log_probabilities,
        advantages,
    ):
        del observations, actions, old_log_probabilities
        self.received_advantages = list(advantages)
        return {"loss": 0.0}


def test_grpo_uses_vectorized_collection_for_maze_env(tmp_path):
    dataset_path = tmp_path / "one_step_tasks.npz"
    np.savez(
        dataset_path,
        layouts=np.zeros((1, 2, 2), dtype=np.int8),
        layout_indices=np.asarray([0, 0]),
        starts=np.asarray([[0, 0], [1, 0]]),
        goals=np.asarray([[0, 1], [1, 1]]),
    )
    env = MazeEnv(dataset_path, max_steps=3)
    agent = OneStepBatchGRPOAgent()

    metrics = train_grpo_round(
        env,
        agent,
        [
            [
                (1, 1, 0),
                (2, 1, 0),
            ]
        ],
    )

    assert agent.batch_sizes == [2]
    assert agent.received_advantages == [0.0, 0.0]
    assert [metric.steps for metric in metrics] == [1, 1]
    assert all(metric.success for metric in metrics)
    assert isinstance(
        env._grpo_vectorized_env,
        VectorizedMazeEnv,
    )


def test_dqn_round_uses_vectorized_environment_slots(tmp_path):
    dataset_path = tmp_path / "dqn_tasks.npz"
    np.savez(
        dataset_path,
        layouts=np.zeros((1, 2, 2), dtype=np.int8),
        layout_indices=np.asarray([0, 0]),
        starts=np.asarray([[0, 0], [1, 0]]),
        goals=np.asarray([[0, 1], [1, 1]]),
    )
    env = MazeEnv(dataset_path, max_steps=1)
    agent = DQNAgent(
        2,
        2,
        4,
        torch.device("cpu"),
        epsilon_start=0.0,
        min_replay_size=2,
        batch_size=2,
        train_frequency=1,
    )

    metrics = train_dqn_round(
        env,
        agent,
        [
            (1, 1, 0),
            (2, 1, 1),
        ],
    )

    assert len(metrics) == 2
    assert [metric.steps for metric in metrics] == [1, 1]
    assert len(agent.replay_buffer) == 2
    assert isinstance(
        env._dnn_vectorized_env,
        VectorizedMazeEnv,
    )


class TruncatedReinforceAgent:
    minibatch_size = 2

    def choose_actions(self, observations):
        return np.zeros(len(observations), dtype=np.int64)

    def compute_returns(self, rewards):
        return list(rewards)

    def update_rollout(self, observations, actions, returns):
        assert len(observations) == len(actions) == len(returns)
        return 0.0


def test_reinforce_vectorized_collection_does_not_bootstrap_values(tmp_path):
    dataset_path = tmp_path / "reinforce_tasks.npz"
    np.savez(
        dataset_path,
        layouts=np.zeros((1, 2, 2), dtype=np.int8),
        layout_indices=np.asarray([0]),
        starts=np.asarray([[0, 0]]),
        goals=np.asarray([[1, 1]]),
    )
    env = MazeEnv(dataset_path, max_steps=1)

    metrics = train_reinforce_round(
        env,
        TruncatedReinforceAgent(),
        [(1, 1, 0)],
    )

    assert len(metrics) == 1
    assert metrics[0].steps == 1
    assert metrics[0].success is False
