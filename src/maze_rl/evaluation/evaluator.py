from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from torch.distributions import Categorical


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

    average_efficiency = (
        np.mean(
            successful_efficiencies
        )
        if successful_efficiencies
        else 0.0
    )

    summary = {
        "episodes": len(results),
        "success_rate": float(
            success_rate
        ),
        "average_episode_return": float(
            average_episode_return
        ),
        "average_path_efficiency": float(
            average_path_efficiency
        ),
        "average_successful_path_efficiency":
            float(average_efficiency),
    }

    return results, summary
