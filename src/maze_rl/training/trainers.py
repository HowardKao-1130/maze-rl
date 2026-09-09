from __future__ import annotations

from dataclasses import replace

import numpy as np

from maze_rl.agents.a2c import A2CAgent
from maze_rl.agents.dqn import (
    DQNAgent,
    Transition,
)
from maze_rl.agents.dyna_q import DynaQAgent
from maze_rl.agents.monte_carlo import MonteCarloAgent
from maze_rl.agents.ppo import PPOAgent
from maze_rl.agents.q_learning import QLearningAgent
from maze_rl.agents.reinforce import ReinforceAgent
from maze_rl.agents.sarsa import SarsaAgent
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
