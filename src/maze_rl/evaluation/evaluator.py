from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from torch.distributions import Categorical


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

            return int(
                values.argmax(
                    dim=1
                ).item()
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

        return int(
            logits.argmax(
                dim=1
            ).item()
        )


def evaluate(
    algorithm,
    agent,
    env,
    episodes: int,
    seed: int | None = None,
    task_indices: Sequence[int] | None = None,
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

            if terminated or truncated:
                break

        results.append(
            {
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
