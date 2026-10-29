from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from torch.distributions import Categorical

from maze_rl.envs.maze_env import MazeEnv
from maze_rl.envs.vectorized_maze_env import (
    VectorizedMazeEnv,
)


def greedy_action_from_scores(
    scores: torch.Tensor,
    action_count: int,
) -> int:
    expected_shape = (1, action_count)

    if tuple(scores.shape) != expected_shape:
        raise ValueError(
            "Expected neural policy scores with shape "
            f"{expected_shape}, got {tuple(scores.shape)}."
        )

    if not torch.isfinite(scores).all():
        raise ValueError(
            "Neural policy produced non-finite action scores."
        )

    action_scores = (
        scores.squeeze(0)
        .detach()
        .cpu()
        .tolist()
    )

    return max(
        range(action_count),
        key=action_scores.__getitem__,
    )


def greedy_actions_from_scores(
    scores: torch.Tensor,
    action_count: int,
) -> np.ndarray:
    if scores.ndim != 2 or scores.shape[1] != action_count:
        raise ValueError(
            "Expected batched neural policy scores with shape "
            f"[batch, {action_count}], got {tuple(scores.shape)}."
        )
    if not torch.isfinite(scores).all():
        raise ValueError(
            "Neural policy produced non-finite action scores."
        )
    return (
        scores.argmax(dim=1)
        .detach()
        .cpu()
        .numpy()
    )


def choose_greedy_action(
    algorithm,
    agent,
    env,
    observation,
):
    if algorithm in {
        "mc",
        "sarsa",
        "q_learning",
        "dyna_q",
    }:
        return agent.choose_action(
            env.tabular_state(),
            explore=False,
        )

    observation_tensor = torch.as_tensor(
        observation,
        dtype=torch.float32,
        device=agent.device,
    ).unsqueeze(0)

    with torch.no_grad():
        if algorithm == "dqn":
            values = agent.online_network(
                observation_tensor
            )

            return greedy_action_from_scores(
                values,
                env.action_space.n,
            )

        if algorithm in {
            "reinforce",
            "grpo",
        }:
            logits = agent.policy(
                observation_tensor
            )

        else:
            logits, _ = agent.network(
                observation_tensor
            )

        return greedy_action_from_scores(
            logits,
            env.action_space.n,
        )


def current_agent_position(
    env,
) -> list[int]:
    return [
        int(coord)
        for coord in env.agent_position
    ]


@torch.inference_mode()
def choose_greedy_actions(
    algorithm,
    agent,
    observations: np.ndarray,
    action_count: int,
) -> np.ndarray:
    observations_tensor = torch.as_tensor(
        observations,
        dtype=torch.float32,
        device=agent.device,
    )
    if algorithm == "dqn":
        scores = agent.online_network(
            observations_tensor
        )
    elif algorithm in {"reinforce", "grpo"}:
        scores = agent.policy(
            observations_tensor
        )
    else:
        scores, _ = agent.network(
            observations_tensor
        )
    return greedy_actions_from_scores(
        scores,
        action_count,
    )


def _evaluation_task_indices(
    env: MazeEnv,
    episodes: int,
    seed: int | None,
    task_indices: Sequence[int] | None,
) -> list[int]:
    if task_indices is not None:
        return [int(index) for index in task_indices]

    sampled_indices = []
    for episode in range(episodes):
        reset_kwargs = {}
        if episode == 0 and seed is not None:
            reset_kwargs["seed"] = seed
        env.reset(**reset_kwargs)
        sampled_indices.append(int(env.task_index))
    return sampled_indices


def _evaluate_vectorized(
    algorithm,
    agent,
    env: MazeEnv,
    episodes: int,
    seed: int | None,
    task_indices: Sequence[int] | None,
    capture_rollouts: bool,
    progress_callback,
):
    episode_task_indices = _evaluation_task_indices(
        env,
        episodes,
        seed,
        task_indices,
    )
    vector_env = getattr(
        env,
        "_evaluation_vectorized_env",
        None,
    )
    if vector_env is None:
        vector_env = VectorizedMazeEnv.from_maze_env(env)
        env._evaluation_vectorized_env = vector_env

    observations, _ = vector_env.reset(episode_task_indices)
    results = [None] * len(episode_task_indices)
    episode_returns = np.zeros(
        len(episode_task_indices),
        dtype=np.float64,
    )
    traces = [None] * len(episode_task_indices)
    if capture_rollouts:
        traces = [
            {
                "positions": [
                    [int(coord) for coord in position]
                ],
                "actions": [],
                "rewards": [],
            }
            for position in vector_env.agent_positions
        ]

    active_slots = np.ones(
        len(episode_task_indices),
        dtype=bool,
    )
    while np.any(active_slots):
        slots = np.flatnonzero(active_slots)
        actions = choose_greedy_actions(
            algorithm,
            agent,
            observations[slots],
            vector_env.ACTION_COUNT,
        )
        all_actions = np.zeros(
            len(episode_task_indices),
            dtype=np.int64,
        )
        all_actions[slots] = actions
        (
            next_observations,
            rewards,
            terminated,
            truncated,
            infos,
        ) = vector_env.step(all_actions)

        for index, slot in enumerate(slots):
            episode_returns[slot] += float(rewards[slot])
            if traces[slot] is not None:
                traces[slot]["actions"].append(
                    int(actions[index])
                )
                traces[slot]["rewards"].append(
                    float(rewards[slot])
                )
                traces[slot]["positions"].append(
                    [
                        int(coord)
                        for coord in vector_env.agent_positions[slot]
                    ]
                )

            if terminated[slot] or truncated[slot]:
                result = {
                    "episode": int(slot + 1),
                    "task_index": infos[slot]["task_index"],
                    "layout_index": infos[slot]["layout_index"],
                    "episode_return": float(
                        episode_returns[slot]
                    ),
                    "success": infos[slot]["success"],
                    "steps": infos[slot]["steps"],
                    "wall_collisions": infos[slot][
                        "wall_collisions"
                    ],
                    "optimal_path_length": infos[slot][
                        "optimal_path_length"
                    ],
                    "path_efficiency": infos[slot][
                        "path_efficiency"
                    ],
                }
                if traces[slot] is not None:
                    traces[slot]["terminated"] = bool(
                        terminated[slot]
                    )
                    traces[slot]["truncated"] = bool(
                        truncated[slot]
                    )
                    result["rollout_trace"] = traces[slot]
                results[slot] = result
                active_slots[slot] = False

        observations = next_observations

    for episode, result in enumerate(results, start=1):
        result["episode"] = episode
        if progress_callback is not None:
            progress_callback(
                episode,
                len(results),
            )

    return results, _summarize_results(results)


def _summarize_results(results):
    success_rate = np.mean(
        [
            result["success"]
            for result in results
        ]
    )
    average_episode_return = np.mean(
        [
            result["episode_return"]
            for result in results
        ]
    )
    average_path_efficiency = np.mean(
        [
            result["path_efficiency"]
            for result in results
        ]
    )

    successful_efficiencies = [
        result["path_efficiency"]
        for result in results
        if result["success"]
    ]

    weighted_path_lengths = [
        result["optimal_path_length"]
        for result in results
        if result["optimal_path_length"] is not None
        and result["optimal_path_length"] > 0
    ]
    weighted_path_length_total = sum(
        weighted_path_lengths
    )
    weighted_path_success = 0.0
    weighted_path_efficiency = 0.0

    if weighted_path_length_total > 0:
        weighted_path_success = sum(
            result["optimal_path_length"]
            for result in results
            if result["optimal_path_length"] is not None
            and result["optimal_path_length"] > 0
            and result["success"]
        ) / weighted_path_length_total
        weighted_path_efficiency = sum(
            result["optimal_path_length"]
            * result["path_efficiency"]
            for result in results
            if result["optimal_path_length"] is not None
            and result["optimal_path_length"] > 0
        ) / weighted_path_length_total

    average_efficiency = (
        np.mean(successful_efficiencies)
        if successful_efficiencies
        else 0.0
    )

    return {
        "episodes": len(results),
        "success_rate": float(success_rate),
        "average_episode_return": float(
            average_episode_return
        ),
        "average_path_efficiency": float(
            average_path_efficiency
        ),
        "average_successful_path_efficiency": float(
            average_efficiency
        ),
        "weighted_path_success": float(
            weighted_path_success
        ),
        "weighted_path_efficiency": float(
            weighted_path_efficiency
        ),
    }


def evaluate(
    algorithm,
    agent,
    env,
    episodes: int,
    seed: int | None = None,
    task_indices: Sequence[int] | None = None,
    capture_rollouts: bool = False,
    progress_callback=None,
):
    if isinstance(env, MazeEnv) and algorithm not in {
        "mc",
        "sarsa",
        "q_learning",
        "dyna_q",
    }:
        return _evaluate_vectorized(
            algorithm,
            agent,
            env,
            episodes,
            seed,
            task_indices,
            capture_rollouts,
            progress_callback,
        )

    results = []
    episode_task_indices = (
        task_indices
        if task_indices is not None
        else [None] * episodes
    )

    for episode, task_index in enumerate(
        episode_task_indices,
        start=1,
    ):
        reset_kwargs = {}

        if episode == 1 and seed is not None:
            reset_kwargs["seed"] = seed

        if task_index is not None:
            reset_kwargs["options"] = {
                "task_index": int(task_index),
            }

        observation, _ = env.reset(
            **reset_kwargs
        )

        episode_return = 0.0
        rollout_trace = (
            {
                "positions": [
                    current_agent_position(env)
                ],
                "actions": [],
                "rewards": [],
            }
            if capture_rollouts
            else None
        )

        while True:
            action = choose_greedy_action(
                algorithm,
                agent,
                env,
                observation,
            )

            (
                observation,
                reward,
                terminated,
                truncated,
                info,
            ) = env.step(action)

            episode_return += reward

            if rollout_trace is not None:
                rollout_trace["actions"].append(
                    int(action)
                )
                rollout_trace["rewards"].append(
                    float(reward)
                )
                rollout_trace["positions"].append(
                    current_agent_position(env)
                )

            if terminated or truncated:
                break

        result = {
            "episode": episode,
            "task_index": info["task_index"],
            "layout_index": info["layout_index"],
            "episode_return": episode_return,
            "success": info["success"],
            "steps": info["steps"],
            "wall_collisions": info[
                "wall_collisions"
            ],
            "optimal_path_length": info[
                "optimal_path_length"
            ],
            "path_efficiency": info[
                "path_efficiency"
            ],
        }

        if rollout_trace is not None:
            rollout_trace["terminated"] = bool(
                terminated
            )
            rollout_trace["truncated"] = bool(
                truncated
            )
            result["rollout_trace"] = rollout_trace

        results.append(result)

        if progress_callback is not None:
            progress_callback(
                episode,
                len(episode_task_indices),
            )

    return results, _summarize_results(results)
