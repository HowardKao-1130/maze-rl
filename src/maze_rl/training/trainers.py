from __future__ import annotations

import copy
from dataclasses import replace

import numpy as np

from maze_rl.agents.a2c import A2CAgent
from maze_rl.agents.dqn import (
    DQNAgent,
    Transition,
)
from maze_rl.agents.dyna_q import DynaQAgent
from maze_rl.agents.grpo import GRPOAgent
from maze_rl.agents.monte_carlo import MonteCarloAgent
from maze_rl.agents.ppo import PPOAgent
from maze_rl.agents.q_learning import QLearningAgent
from maze_rl.agents.reinforce import ReinforceAgent
from maze_rl.agents.sarsa import SarsaAgent
from maze_rl.envs.maze_env import MazeEnv
from maze_rl.envs.vectorized_maze_env import (
    VectorizedMazeEnv,
)
from maze_rl.training.metrics import EpisodeMetrics


def _metrics(
    episode: int,
    epoch: int,
    episode_return: float,
    info: dict,
    internal_updates: int | None,
    mean_abs_td_error: float | None,
    loss: float | None,
    diagnostics: dict | None = None,
) -> EpisodeMetrics:
    diagnostics = diagnostics or {}

    return EpisodeMetrics(
        episode=episode,
        epoch=epoch,
        task_index=info["task_index"],
        layout_index=info["layout_index"],
        episode_return=episode_return,
        steps=info["steps"],
        internal_updates=internal_updates,
        success=info["success"],
        wall_collisions=info["wall_collisions"],
        optimal_path_length=info[
            "optimal_path_length"
        ],
        path_efficiency=info[
            "path_efficiency"
        ],
        mean_abs_td_error=mean_abs_td_error,
        loss=diagnostics.get(
            "loss",
            loss,
        ),
        policy_loss=diagnostics.get(
            "policy_loss"
        ),
        value_loss=diagnostics.get(
            "value_loss"
        ),
        entropy=diagnostics.get(
            "entropy"
        ),
        approximate_kl=diagnostics.get(
            "approximate_kl"
        ),
        clip_fraction=diagnostics.get(
            "clip_fraction"
        ),
        mean_value=diagnostics.get(
            "mean_value"
        ),
        mean_return=diagnostics.get(
            "mean_return"
        ),
        mean_advantage=diagnostics.get(
            "mean_advantage"
        ),
        mean_q_value=diagnostics.get(
            "mean_q_value"
        ),
        max_q_value=diagnostics.get(
            "max_q_value"
        ),
        replay_size=diagnostics.get(
            "replay_size"
        ),
    )


def _reset_env(env, task_index: int | None):
    if task_index is None:
        return env.reset()

    return env.reset(
        options={
            "task_index": task_index,
        }
    )


def _minibatch_count(
    transition_count: int,
    minibatch_size: int,
) -> int:
    return (
        transition_count
        + minibatch_size
        - 1
    ) // minibatch_size


def _mean_diagnostics(
    diagnostics_list,
):
    if not diagnostics_list:
        return {}

    keys = sorted(
        {
            key
            for diagnostics in diagnostics_list
            for key in diagnostics
        }
    )

    averaged = {}

    for key in keys:
        values = [
            diagnostics[key]
            for diagnostics in diagnostics_list
            if diagnostics.get(key) is not None
        ]

        if values:
            averaged[key] = float(
                np.mean(values)
            )

    return averaged


def _apply_round_diagnostics(
    metrics,
    internal_updates: int,
    diagnostics: dict,
):
    if not isinstance(
        diagnostics,
        dict,
    ):
        diagnostics = {
            "loss": diagnostics
        }

    last_index = len(metrics) - 1

    return [
        replace(
            metric,
            internal_updates=(
                internal_updates
                if index == last_index
                else None
            ),
            **(
                diagnostics
                if index == last_index
                else {}
            ),
        )
        for index, metric in enumerate(metrics)
    ]


def train_monte_carlo_episode(
    env,
    agent: MonteCarloAgent,
    episode: int,
    epoch: int,
    task_index: int | None = None,
):
    _, _ = _reset_env(env, task_index)

    trajectory = []
    episode_return = 0.0

    while True:
        state = env.tabular_state()

        action = agent.choose_action(
            state,
            explore=True,
        )

        (
            _,
            reward,
            terminated,
            truncated,
            info,
        ) = env.step(action)

        trajectory.append(
            (state, action, reward)
        )

        episode_return += reward

        if terminated or truncated:
            break

    internal_updates = agent.update_episode(
        trajectory
    )

    return _metrics(
        episode,
        epoch,
        episode_return,
        info,
        internal_updates,
        None,
        None,
    )


def train_sarsa_episode(
    env,
    agent: SarsaAgent,
    episode: int,
    epoch: int,
    task_index: int | None = None,
):
    _, _ = _reset_env(env, task_index)

    state = env.tabular_state()

    action = agent.choose_action(
        state,
        explore=True,
    )

    episode_return = 0.0
    td_errors = []

    while True:
        (
            _,
            reward,
            terminated,
            truncated,
            info,
        ) = env.step(action)

        next_state = env.tabular_state()

        next_action = (
            None
            if terminated
            else agent.choose_action(
                next_state,
                explore=True,
            )
        )

        td_errors.append(
            agent.update(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                next_action=next_action,
                terminated=terminated,
            )
        )

        episode_return += reward

        if terminated or truncated:
            break

        state = next_state
        action = next_action

    return _metrics(
        episode,
        epoch,
        episode_return,
        info,
        len(td_errors),
        float(np.mean(np.abs(td_errors))),
        None,
    )


def train_q_learning_episode(
    env,
    agent: QLearningAgent | DynaQAgent,
    episode: int,
    epoch: int,
    task_index: int | None = None,
):
    _, _ = _reset_env(env, task_index)

    episode_return = 0.0
    td_errors = []

    while True:
        state = env.tabular_state()

        action = agent.choose_action(
            state,
            explore=True,
        )

        (
            _,
            reward,
            terminated,
            truncated,
            info,
        ) = env.step(action)

        next_state = env.tabular_state()

        td_errors.append(
            agent.update(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                terminated=terminated,
            )
        )

        episode_return += reward

        if terminated or truncated:
            break

    planning_steps = (
        agent.planning_steps
        if isinstance(agent, DynaQAgent)
        else 0
    )

    return _metrics(
        episode,
        epoch,
        episode_return,
        info,
        info["steps"] * (1 + planning_steps),
        float(np.mean(np.abs(td_errors))),
        None,
    )


def train_dqn_episode(
    env,
    agent: DQNAgent,
    episode: int,
    epoch: int,
    task_index: int | None = None,
):
    observation, _ = _reset_env(env, task_index)

    episode_return = 0.0
    losses = []

    while True:
        action = agent.choose_action(
            observation,
            explore=True,
        )

        (
            next_observation,
            reward,
            terminated,
            truncated,
            info,
        ) = env.step(action)

        agent.store_transition(
            Transition(
                state=observation,
                action=action,
                reward=reward,
                next_state=next_observation,
                terminated=terminated,
            )
        )

        loss = agent.train_step()

        if loss is not None:
            losses.append(loss)

        episode_return += reward

        observation = next_observation

        if terminated or truncated:
            break

    diagnostics = (
        _mean_diagnostics(losses)
        if losses
        else {}
    )

    return _metrics(
        episode,
        epoch,
        episode_return,
        info,
        len(losses),
        None,
        None,
        diagnostics,
    )

def train_reinforce_episode(
    env,
    agent: ReinforceAgent,
    episode: int,
    epoch: int,
    task_index: int | None = None,
):
    return train_reinforce_round(
        env,
        agent,
        [
            (
                episode,
                epoch,
                task_index,
            )
        ],
    )[0]


def _collect_reinforce_episode(
    env,
    agent: ReinforceAgent,
    episode: int,
    epoch: int,
    task_index: int | None = None,
):
    observation, _ = _reset_env(env, task_index)

    episode_return = 0.0

    observations = []
    actions = []
    rewards = []

    while True:
        action = agent.choose_action(
            observation
        )

        observations.append(
            observation.copy()
        )
        actions.append(action)

        (
            observation,
            reward,
            terminated,
            truncated,
            info,
        ) = env.step(action)

        rewards.append(reward)

        episode_return += reward

        if terminated or truncated:
            break

    returns = agent.compute_returns(
        rewards
    )

    return (
        _metrics(
            episode,
            epoch,
            episode_return,
            info,
            None,
            None,
            None,
        ),
        observations,
        actions,
        returns,
    )


def train_reinforce_round(
    env,
    agent: ReinforceAgent,
    episode_specs,
):
    if isinstance(env, MazeEnv):
        rollouts = _collect_vectorized_on_policy_rollouts(
            env,
            agent,
            episode_specs,
            algorithm="reinforce",
        )
        metrics = [rollout["metrics"] for rollout in rollouts]
        observations = [
            observation
            for rollout in rollouts
            for observation in rollout["observations"]
        ]
        actions = [
            action
            for rollout in rollouts
            for action in rollout["actions"]
        ]
        returns = [
            value
            for rollout in rollouts
            for value in rollout["returns"]
        ]
        loss = agent.update_rollout(
            observations=observations,
            actions=actions,
            returns=returns,
        )
        return _apply_round_diagnostics(
            metrics,
            _minibatch_count(
                len(actions),
                agent.minibatch_size,
            ),
            loss,
        )

    metrics = []
    observations = []
    actions = []
    returns = []

    for (
        episode,
        epoch,
        task_index,
    ) in episode_specs:
        (
            episode_metrics,
            episode_observations,
            episode_actions,
            episode_returns,
        ) = _collect_reinforce_episode(
            env,
            agent,
            episode,
            epoch,
            task_index,
        )

        metrics.append(episode_metrics)
        observations.extend(
            episode_observations
        )
        actions.extend(episode_actions)
        returns.extend(episode_returns)

    loss = agent.update_rollout(
        observations=observations,
        actions=actions,
        returns=returns,
    )

    internal_updates = _minibatch_count(
        len(actions),
        agent.minibatch_size,
    )

    return _apply_round_diagnostics(
        metrics,
        internal_updates,
        loss,
    )


def train_a2c_episode(
    env,
    agent: A2CAgent,
    episode: int,
    epoch: int,
    task_index: int | None = None,
):
    return train_a2c_round(
        env,
        agent,
        [
            (
                episode,
                epoch,
                task_index,
            )
        ],
    )[0]


def _collect_a2c_episode(
    env,
    agent: A2CAgent,
    episode: int,
    epoch: int,
    task_index: int | None = None,
):
    observation, _ = _reset_env(env, task_index)

    observations = []
    actions = []
    values = []
    rewards = []
    terminated_flags = []

    episode_return = 0.0

    while True:
        (
            action,
            value,
        ) = agent.choose_action(
            observation
        )

        observations.append(
            observation.copy()
        )
        actions.append(action)
        values.append(value)

        (
            next_observation,
            reward,
            terminated,
            truncated,
            info,
        ) = env.step(action)

        rewards.append(reward)
        terminated_flags.append(
            terminated
        )

        episode_return += reward

        observation = next_observation

        if terminated or truncated:
            break

    if terminated:
        last_value = 0.0
    else:
        # Time-limit truncation: bootstrap from
        # the final non-terminal state.
        last_value = agent.estimate_value(
            observation
        )

    advantages, returns = (
        agent.compute_gae(
            rewards=rewards,
            values=values,
            terminated_flags=terminated_flags,
            last_value=last_value,
        )
    )

    return (
        _metrics(
            episode,
            epoch,
            episode_return,
            info,
            None,
            None,
            None,
        ),
        observations,
        actions,
        advantages,
        returns,
    )


def train_a2c_round(
    env,
    agent: A2CAgent,
    episode_specs,
):
    if isinstance(env, MazeEnv):
        rollouts = _collect_vectorized_on_policy_rollouts(
            env,
            agent,
            episode_specs,
            algorithm="a2c",
        )
        metrics = [rollout["metrics"] for rollout in rollouts]
        observations = [
            observation
            for rollout in rollouts
            for observation in rollout["observations"]
        ]
        actions = [
            action
            for rollout in rollouts
            for action in rollout["actions"]
        ]
        advantages = [
            value
            for rollout in rollouts
            for value in rollout["advantages"]
        ]
        returns = [
            value
            for rollout in rollouts
            for value in rollout["returns"]
        ]
        loss = agent.update_rollout(
            observations=observations,
            actions=actions,
            advantages=advantages,
            returns=returns,
        )
        return _apply_round_diagnostics(
            metrics,
            _minibatch_count(
                len(actions),
                agent.minibatch_size,
            ),
            loss,
        )

    metrics = []
    observations = []
    actions = []
    advantages = []
    returns = []

    for (
        episode,
        epoch,
        task_index,
    ) in episode_specs:
        (
            episode_metrics,
            episode_observations,
            episode_actions,
            episode_advantages,
            episode_returns,
        ) = _collect_a2c_episode(
            env,
            agent,
            episode,
            epoch,
            task_index,
        )

        metrics.append(episode_metrics)
        observations.extend(
            episode_observations
        )
        actions.extend(episode_actions)
        advantages.extend(
            episode_advantages
        )
        returns.extend(episode_returns)

    loss = agent.update_rollout(
        observations=observations,
        actions=actions,
        advantages=advantages,
        returns=returns,
    )

    internal_updates = _minibatch_count(
        len(actions),
        agent.minibatch_size,
    )

    return _apply_round_diagnostics(
        metrics,
        internal_updates,
        loss,
    )


def train_ppo_episode(
    env,
    agent: PPOAgent,
    episode: int,
    epoch: int,
    task_index: int | None = None,
):
    return train_ppo_round(
        env,
        agent,
        [
            (
                episode,
                epoch,
                task_index,
            )
        ],
    )[0]


def _collect_ppo_episode(
    env,
    agent: PPOAgent,
    episode: int,
    epoch: int,
    task_index: int | None = None,
):
    observation, _ = _reset_env(env, task_index)

    observations = []
    actions = []
    log_probabilities = []
    values = []
    rewards = []
    terminated_flags = []

    episode_return = 0.0

    while True:
        (
            action,
            log_probability,
            value,
        ) = agent.choose_action(
            observation
        )

        observations.append(
            observation.copy()
        )

        actions.append(action)
        log_probabilities.append(
            log_probability
        )
        values.append(value)

        (
            next_observation,
            reward,
            terminated,
            truncated,
            info,
        ) = env.step(action)

        rewards.append(reward)
        terminated_flags.append(
            terminated
        )

        episode_return += reward

        observation = next_observation

        if terminated or truncated:
            break

    if terminated:
        last_value = 0.0
    else:
        # Time-limit truncation: bootstrap from
        # the final non-terminal state.
        last_value = agent.estimate_value(
            observation
        )

    advantages, returns = (
        agent.compute_gae(
            rewards=rewards,
            values=values,
            terminated_flags=terminated_flags,
            last_value=last_value,
        )
    )

    return (
        _metrics(
            episode,
            epoch,
            episode_return,
            info,
            None,
            None,
            None,
        ),
        observations,
        actions,
        log_probabilities,
        advantages,
        returns,
    )


def train_ppo_round(
    env,
    agent: PPOAgent,
    episode_specs,
):
    if isinstance(env, MazeEnv):
        rollouts = _collect_vectorized_on_policy_rollouts(
            env,
            agent,
            episode_specs,
            algorithm="ppo",
        )
        metrics = [rollout["metrics"] for rollout in rollouts]
        observations = [
            observation
            for rollout in rollouts
            for observation in rollout["observations"]
        ]
        actions = [
            action
            for rollout in rollouts
            for action in rollout["actions"]
        ]
        log_probabilities = [
            value
            for rollout in rollouts
            for value in rollout["log_probabilities"]
        ]
        advantages = [
            value
            for rollout in rollouts
            for value in rollout["advantages"]
        ]
        returns = [
            value
            for rollout in rollouts
            for value in rollout["returns"]
        ]
        loss = agent.update(
            observations=observations,
            actions=actions,
            old_log_probabilities=log_probabilities,
            advantages=advantages,
            returns=returns,
        )
        return _apply_round_diagnostics(
            metrics,
            agent.update_epochs
            * _minibatch_count(
                len(actions),
                agent.minibatch_size,
            ),
            loss,
        )

    metrics = []
    observations = []
    actions = []
    log_probabilities = []
    advantages = []
    returns = []

    for (
        episode,
        epoch,
        task_index,
    ) in episode_specs:
        (
            episode_metrics,
            episode_observations,
            episode_actions,
            episode_log_probabilities,
            episode_advantages,
            episode_returns,
        ) = _collect_ppo_episode(
            env,
            agent,
            episode,
            epoch,
            task_index,
        )

        metrics.append(episode_metrics)
        observations.extend(
            episode_observations
        )
        actions.extend(episode_actions)
        log_probabilities.extend(
            episode_log_probabilities
        )
        advantages.extend(
            episode_advantages
        )
        returns.extend(episode_returns)

    loss = agent.update(
        observations=observations,
        actions=actions,
        old_log_probabilities=log_probabilities,
        advantages=advantages,
        returns=returns,
    )

    internal_updates = (
        agent.update_epochs
        * _minibatch_count(
            len(actions),
            agent.minibatch_size,
        )
    )

    return _apply_round_diagnostics(
        metrics,
        internal_updates,
        loss,
    )


def _collect_vectorized_on_policy_rollouts(
    env: MazeEnv,
    agent,
    episode_specs,
    algorithm: str,
):
    vector_env = getattr(
        env,
        "_dnn_vectorized_env",
        None,
    )
    if vector_env is None:
        vector_env = VectorizedMazeEnv.from_maze_env(env)
        env._dnn_vectorized_env = vector_env

    episode_specs = list(episode_specs)
    task_indices = [
        _sample_grpo_task_index(env, task_index)
        for _, _, task_index in episode_specs
    ]
    observations, _ = vector_env.reset(task_indices)
    rollouts = []
    for slot, (episode, epoch, _) in enumerate(episode_specs):
        rollouts.append(
            {
                "episode": episode,
                "epoch": epoch,
                "observations": [],
                "actions": [],
                "log_probabilities": [],
                "values": [],
                "rewards": [],
                "terminated_flags": [],
                "slot": slot,
            }
        )

    active_slots = np.ones(
        len(rollouts),
        dtype=bool,
    )
    while np.any(active_slots):
        slots = np.flatnonzero(active_slots)
        active_observations = observations[slots]

        if algorithm == "reinforce":
            actions = agent.choose_actions(
                active_observations
            )
            values = None
            log_probabilities = None
        elif algorithm == "a2c":
            actions, values = agent.choose_actions(
                active_observations
            )
            log_probabilities = None
        elif algorithm == "ppo":
            (
                actions,
                log_probabilities,
                values,
            ) = agent.choose_actions(
                active_observations
            )
        else:
            raise ValueError(
                f"Unsupported on-policy algorithm: {algorithm}."
            )

        all_actions = np.zeros(
            len(rollouts),
            dtype=np.int64,
        )
        for index, slot in enumerate(slots):
            rollout = rollouts[slot]
            rollout["observations"].append(
                observations[slot].copy()
            )
            rollout["actions"].append(int(actions[index]))
            if values is not None:
                rollout["values"].append(float(values[index]))
            if log_probabilities is not None:
                rollout["log_probabilities"].append(
                    float(log_probabilities[index])
                )
            all_actions[slot] = actions[index]

        (
            next_observations,
            rewards,
            terminated,
            truncated,
            infos,
        ) = vector_env.step(all_actions)
        for slot in slots:
            rollout = rollouts[slot]
            rollout["rewards"].append(float(rewards[slot]))
            rollout["terminated_flags"].append(
                bool(terminated[slot])
            )
            if terminated[slot] or truncated[slot]:
                rollout["metrics"] = _metrics(
                    rollout["episode"],
                    rollout["epoch"],
                    float(sum(rollout["rewards"])),
                    infos[slot],
                    None,
                    None,
                    None,
                )
                rollout["last_observation"] = (
                    next_observations[slot].copy()
                )
                rollout["terminated"] = bool(terminated[slot])
                active_slots[slot] = False

        observations = next_observations

    truncated_slots = [
        slot
        for slot, rollout in enumerate(rollouts)
        if not rollout["terminated"]
    ]
    bootstrap_values = {}
    if truncated_slots:
        last_observations = np.asarray(
            [
                rollouts[slot]["last_observation"]
                for slot in truncated_slots
            ]
        )
        estimate_values = getattr(
            agent,
            "estimate_values",
            None,
        )
        if estimate_values is not None:
            values = estimate_values(last_observations)
        else:
            values = np.asarray(
                [
                    agent.estimate_value(
                        observation
                    )
                    for observation in last_observations
                ]
            )
        bootstrap_values = {
            slot: float(value)
            for slot, value in zip(
                truncated_slots,
                values,
            )
        }

    for slot, rollout in enumerate(rollouts):
        if algorithm == "reinforce":
            rollout["returns"] = agent.compute_returns(
                rollout["rewards"]
            )
        else:
            last_value = (
                0.0
                if rollout["terminated"]
                else bootstrap_values[slot]
            )
            advantages, returns = agent.compute_gae(
                rewards=rollout["rewards"],
                values=rollout["values"],
                terminated_flags=rollout["terminated_flags"],
                last_value=last_value,
            )
            rollout["advantages"] = advantages
            rollout["returns"] = returns

    return rollouts


def train_grpo_episode(
    env,
    agent: GRPOAgent,
    episode: int,
    epoch: int,
    task_index: int | None = None,
):
    return train_grpo_round(
        env,
        agent,
        [
            [
                (
                    episode,
                    epoch,
                    task_index,
                )
            ]
        ]
    )[0]


def _copy_rollout_env(env):
    return copy.copy(env)


def _choose_grpo_actions(
    agent: GRPOAgent,
    observations,
):
    actions, log_probabilities = (
        _choose_grpo_action_arrays(
            agent,
            observations,
        )
    )

    return [
        (
            int(action),
            float(log_probability),
        )
        for action, log_probability in zip(
            actions,
            log_probabilities,
        )
    ]


def _choose_grpo_action_arrays(
    agent: GRPOAgent,
    observations,
):
    choose_actions = getattr(
        agent,
        "choose_actions",
        None,
    )

    if choose_actions is None:
        action_pairs = [
            agent.choose_action(observation)
            for observation in observations
        ]
        if not action_pairs:
            return (
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.float32),
            )
        actions, log_probabilities = zip(*action_pairs)
        return (
            np.asarray(actions, dtype=np.int64),
            np.asarray(log_probabilities, dtype=np.float32),
        )

    actions, log_probabilities = choose_actions(
        observations
    )

    actions = np.asarray(actions, dtype=np.int64)
    log_probabilities = np.asarray(
        log_probabilities,
        dtype=np.float32,
    )
    if actions.shape != (len(observations),):
        raise ValueError(
            "GRPO choose_actions must return one action per observation."
        )
    if log_probabilities.shape != actions.shape:
        raise ValueError(
            "GRPO choose_actions must return one log probability per action."
        )
    return actions, log_probabilities


def _sample_grpo_task_index(env, task_index):
    if task_index is not None:
        return int(task_index)

    fixed_index = getattr(env, "fixed_index", None)
    if fixed_index is not None:
        return int(fixed_index)

    return int(env.np_random.integers(env.num_tasks))


def _collect_grpo_rollout_groups_vectorized(
    env: MazeEnv,
    agent: GRPOAgent,
    rollout_groups,
):
    vector_env = getattr(
        env,
        "_grpo_vectorized_env",
        None,
    )
    if vector_env is None:
        vector_env = VectorizedMazeEnv.from_maze_env(env)
        env._grpo_vectorized_env = vector_env

    grouped_rollouts = []
    all_rollouts = []
    task_indices = []

    for episode_specs in rollout_groups:
        group = []
        for (
            episode,
            epoch,
            task_index,
        ) in episode_specs:
            group.append(
                {
                    "episode": episode,
                    "epoch": epoch,
                    "observation": None,
                    "observations": [],
                    "actions": [],
                    "log_probabilities": [],
                    "rewards": [],
                }
            )
            task_indices.append(
                _sample_grpo_task_index(env, task_index)
            )
        grouped_rollouts.append(group)
        all_rollouts.extend(group)

    observations, _ = vector_env.reset(task_indices)
    for slot, rollout in enumerate(all_rollouts):
        rollout["slot"] = slot
        rollout["observation"] = observations[slot]

    active_slots = np.ones(
        len(all_rollouts),
        dtype=bool,
    )

    while np.any(active_slots):
        slots = np.flatnonzero(active_slots)
        actions, log_probabilities = _choose_grpo_action_arrays(
            agent,
            observations[slots],
        )
        all_actions = np.zeros(
            len(all_rollouts),
            dtype=np.int64,
        )

        for index, slot in enumerate(slots):
            rollout = all_rollouts[slot]
            rollout["observations"].append(
                observations[slot].copy()
            )
            rollout["actions"].append(int(actions[index]))
            rollout["log_probabilities"].append(
                float(log_probabilities[index])
            )
            all_actions[slot] = actions[index]

        (
            next_observations,
            rewards,
            terminated,
            truncated,
            infos,
        ) = vector_env.step(all_actions)

        for slot in slots:
            rollout = all_rollouts[slot]
            rollout["rewards"].append(float(rewards[slot]))
            done = bool(terminated[slot] or truncated[slot])
            if done:
                rollout["metrics"] = _metrics(
                    rollout["episode"],
                    rollout["epoch"],
                    float(sum(rollout["rewards"])),
                    infos[slot],
                    None,
                    None,
                    None,
                )
                rollout["return"] = agent.compute_episode_return(
                    rollout["rewards"]
                )
                active_slots[slot] = False

        observations = next_observations

    return grouped_rollouts


def _collect_grpo_rollout_groups(
    env,
    agent: GRPOAgent,
    rollout_groups,
):
    if isinstance(env, MazeEnv):
        return _collect_grpo_rollout_groups_vectorized(
            env,
            agent,
            rollout_groups,
        )

    active_rollouts = []
    grouped_rollouts = []

    for episode_specs in rollout_groups:
        group = []

        for (
            episode,
            epoch,
            task_index,
        ) in episode_specs:
            rollout_env = _copy_rollout_env(env)
            observation, _ = _reset_env(
                rollout_env,
                task_index,
            )
            rollout = {
                "env": rollout_env,
                "episode": episode,
                "epoch": epoch,
                "observation": observation,
                "observations": [],
                "actions": [],
                "log_probabilities": [],
                "rewards": [],
            }
            group.append(rollout)
            active_rollouts.append(rollout)

        grouped_rollouts.append(group)

    while active_rollouts:
        action_pairs = _choose_grpo_actions(
            agent,
            [
                rollout["observation"]
                for rollout in active_rollouts
            ],
        )
        next_active_rollouts = []

        for rollout, (
            action,
            log_probability,
        ) in zip(active_rollouts, action_pairs):
            rollout["observations"].append(
                rollout["observation"].copy()
            )
            rollout["actions"].append(action)
            rollout["log_probabilities"].append(
                log_probability
            )

            (
                observation,
                reward,
                terminated,
                truncated,
                info,
            ) = rollout["env"].step(action)

            rollout["rewards"].append(reward)

            if terminated or truncated:
                rollout["metrics"] = _metrics(
                    rollout["episode"],
                    rollout["epoch"],
                    float(sum(rollout["rewards"])),
                    info,
                    None,
                    None,
                    None,
                )
                rollout["return"] = (
                    agent.compute_episode_return(
                        rollout["rewards"]
                    )
                )
            else:
                rollout["observation"] = observation
                next_active_rollouts.append(rollout)

        active_rollouts = next_active_rollouts

    return grouped_rollouts


def train_grpo_round(
    env,
    agent: GRPOAgent,
    rollout_groups,
):
    metrics = []
    observations = []
    actions = []
    log_probabilities = []
    advantages = []
    all_returns = []

    if rollout_groups and isinstance(
        rollout_groups[0],
        tuple,
    ):
        rollout_groups = [rollout_groups]

    grouped_rollouts = _collect_grpo_rollout_groups(
        env,
        agent,
        rollout_groups,
    )

    for group in grouped_rollouts:
        group_observations = [
            rollout["observations"]
            for rollout in group
        ]
        group_actions = [
            rollout["actions"]
            for rollout in group
        ]
        group_log_probabilities = [
            rollout["log_probabilities"]
            for rollout in group
        ]
        group_returns = [
            rollout["return"]
            for rollout in group
        ]

        for rollout in group:
            metrics.append(rollout["metrics"])
            all_returns.append(
                rollout["return"]
            )

        returns_array = np.asarray(
            group_returns,
            dtype=np.float32,
        )
        returns_std = float(
            returns_array.std()
        )

        if returns_std <= 1e-8:
            episode_advantages = np.zeros_like(
                returns_array
            )
        else:
            episode_advantages = (
                returns_array
                - returns_array.mean()
            ) / returns_std

        for index, advantage in enumerate(
            episode_advantages
        ):
            observations.extend(
                group_observations[index]
            )
            actions.extend(
                group_actions[index]
            )
            log_probabilities.extend(
                group_log_probabilities[index]
            )
            advantages.extend(
                [float(advantage)]
                * len(group_actions[index])
            )

    if not metrics:
        return metrics

    returns_array = np.asarray(
        all_returns,
        dtype=np.float32,
    )

    diagnostics = agent.update(
        observations=observations,
        actions=actions,
        old_log_probabilities=log_probabilities,
        advantages=advantages,
    )
    diagnostics["mean_return"] = float(
        returns_array.mean()
    )

    internal_updates = _minibatch_count(
        len(actions),
        agent.minibatch_size,
    )

    return _apply_round_diagnostics(
        metrics,
        internal_updates,
        diagnostics,
    )
