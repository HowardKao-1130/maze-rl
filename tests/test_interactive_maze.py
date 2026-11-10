from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from maze_rl.envs.maze_env import MazeEnv
from scripts.interactive_maze import (
    AGENT_PANEL_POSITIONS,
    AvailableAgent,
    InteractiveMazeApp,
    TARGET_ALGORITHMS,
    discover_available_agents,
    load_random_v3_task,
    parse_args,
)


def test_interactive_cli_defaults_to_11_by_11():
    args = parse_args([])

    assert args.height == 11
    assert args.width == 11


def test_interactive_cli_accepts_custom_dimensions():
    args = parse_args(
        [
            "--height",
            "7",
            "--width",
            "13",
        ]
    )

    assert (args.height, args.width) == (7, 13)


def test_interactive_cli_rejects_non_positive_dimensions():
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--height",
                "0",
            ]
        )


def test_discover_available_agents_uses_completed_tier_checkpoints(
    tmp_path,
):
    training_root = (
        tmp_path
        / "runs"
        / "dnn"
        / "dataset_2000x200x200_t5_seed42"
        / "training"
    )
    complete_dir = training_root / "dqn"
    complete_dir.mkdir(parents=True)
    (complete_dir / "best_checkpoint.pt").touch()
    (complete_dir / "training_summary.json").write_text(
        json.dumps({"status": "early_stopped"})
    )

    running_dir = training_root / "ppo"
    running_dir.mkdir(parents=True)
    (running_dir / "best_checkpoint.pt").touch()
    (running_dir / "training_summary.json").write_text(
        json.dumps({"status": "running"})
    )

    agents = discover_available_agents(tmp_path)

    assert [agent.name for agent in agents] == ["dqn_v1"]


def test_discover_available_agents_prefers_latest_completed_generation(
    tmp_path,
):
    for dataset_name, status in (
        ("dataset_2000x200x200_t5_seed42", "early_stopped"),
        ("dataset_5000x500x500_t5_seed42", "early_stopped"),
        (
            "dataset_5000x500x500_t5_seed42_stratified_s00_m60_l40_dedupe050",
            "early_stopped",
        ),
    ):
        training_root = (
            tmp_path / "runs" / "dnn" / dataset_name / "training" / "dqn"
        )
        training_root.mkdir(parents=True)
        (training_root / "best_checkpoint.pt").touch()
        (training_root / "training_summary.json").write_text(
            json.dumps({"status": status})
        )

    agents = discover_available_agents(tmp_path)

    assert [agent.name for agent in agents] == ["dqn_v3"]


def test_load_random_v3_task_uses_requested_split(tmp_path):
    dataset_dir = (
        tmp_path
        / "data"
        / "datasets"
        / "dataset_5000x500x500_t5_seed42_stratified_s00_m60_l40_dedupe050"
    )
    dataset_dir.mkdir(parents=True)
    (dataset_dir / "metadata.json").write_text(
        json.dumps({"stratify": True})
    )
    layout = np.zeros((1, 3, 3), dtype=np.uint8)
    starts = np.asarray([[0, 0]])
    goals = np.asarray([[2, 2]])
    common = {
        "layouts": layout,
        "layout_indices": np.asarray([0]),
        "seeds": np.asarray([42]),
        "starts": starts,
        "goals": goals,
    }
    np.savez(dataset_dir / "train_tune.npz", **common)
    np.savez(dataset_dir / "validation_natural.npz", **common)

    train_task = load_random_v3_task("train", tmp_path)
    val_task = load_random_v3_task("val", tmp_path)

    assert train_task.source.name == "train_tune.npz"
    assert val_task.source.name == "validation_natural.npz"
    assert train_task.start == (0, 0)
    assert train_task.goal == (2, 2)
    assert np.array_equal(train_task.layout, layout[0])


def test_load_random_v3_task_rejects_unknown_split(tmp_path):
    with pytest.raises(ValueError, match="train.*val"):
        load_random_v3_task("test", tmp_path)


def test_maze_env_can_be_created_from_arrays():
    maze = np.zeros((3, 4), dtype=np.uint8)
    env = MazeEnv.from_arrays(
        maze,
        start=(0, 0),
        goal=(2, 3),
    )

    observation, _ = env.reset()

    assert observation.shape == (3, 3, 4)
    assert env.agent_position == (0, 0)
    assert env.goal_position == (2, 3)

    observation, _, terminated, truncated, _ = env.step(3)

    assert observation.shape == (3, 3, 4)
    assert not terminated
    assert not truncated


def test_reset_run_handles_completed_animation():
    class FakeTimer:
        def __init__(self):
            self.stop_count = 0

        def stop(self):
            self.stop_count += 1

    app = object.__new__(InteractiveMazeApp)
    app.inference_timer = FakeTimer()
    app.panel_states = []
    app.current_run_record = None
    app.status_text = SimpleNamespace(set_text=lambda _text: None)
    app.figure = SimpleNamespace(
        canvas=SimpleNamespace(draw_idle=lambda: None)
    )

    app._reset_run()

    assert app.inference_timer is None
    assert app.status_text is not None


def test_wall_drag_paints_the_interpolated_cursor_segment():
    app = object.__new__(InteractiveMazeApp)
    app.maze = np.zeros((5, 5), dtype=np.uint8)
    app.start = (0, 0)
    app.goal = (4, 4)

    changed = app._paint_wall_segment((1, 0), (3, 4))

    assert changed
    assert app.maze[1, 0] == 1
    assert app.maze[2, 2] == 1
    assert app.maze[3, 4] == 1


def test_wall_drag_never_overwrites_start_or_goal():
    app = object.__new__(InteractiveMazeApp)
    app.maze = np.zeros((5, 5), dtype=np.uint8)
    app.start = (0, 0)
    app.goal = (4, 4)

    app._paint_wall_segment(app.start, app.goal)

    assert app.maze[app.start] == 0
    assert app.maze[app.goal] == 0


def test_source_title_reports_optimal_steps():
    app = object.__new__(InteractiveMazeApp)
    app.maze = np.zeros((3, 4), dtype=np.uint8)
    app.start = (0, 0)
    app.goal = (2, 3)

    assert app._source_title() == "Build maze (0, 0) (optimal: 5 steps)"

    app.maze[1, :] = 1
    assert app._source_title() == (
        "Build maze (0, 0) (optimal: unavailable)"
    )


def test_inference_record_persists_failed_agent_and_maze_data(tmp_path):
    app = object.__new__(InteractiveMazeApp)
    app.failure_log_path = tmp_path / "nested" / "failure_runs.jsonl"
    app.recorded_failure_keys = set()
    app.current_run_record = {
        "schema_version": 2,
        "run_id": "test-run",
        "started_at": "2026-01-01T00:00:00+00:00",
        "task_key": InteractiveMazeApp._task_key(
            np.asarray([[0, 1, 0], [0, 0, 0]], dtype=np.uint8),
            (0, 0),
            (1, 2),
        ),
        "maze": {
            "height": 2,
            "width": 3,
            "layout": [[0, 1, 0], [0, 0, 0]],
            "start": [0, 0],
            "goal": [1, 2],
            "optimal_steps": 3,
        },
        "agents": {
            "failed": [],
            "reserved": ["grpo"],
        },
    }
    app.status_text = SimpleNamespace(set_text=lambda _text: None)
    app.figure = SimpleNamespace(
        canvas=SimpleNamespace(draw_idle=lambda: None)
    )
    state = SimpleNamespace(
        option=AvailableAgent(
            "dqn_v1",
            "dqn",
            "low",
            tmp_path / "dqn.pt",
        ),
        rollout_actions=[1, 3],
        rollout_positions=[(0, 0), (0, 1), (1, 1)],
    )

    app._record_panel_outcome(state, "loop")
    first_record = json.loads(
        json.dumps(app.current_run_record)
    )
    app._persist_inference_record("completed")
    app.current_run_record = first_record
    app._persist_inference_record("completed")

    records = [
        json.loads(line)
        for line in app.failure_log_path.read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    assert len(records) == 1
    record = records[0]
    assert record["maze"]["layout"] == [[0, 1, 0], [0, 0, 0]]
    assert record["agents"]["reserved"] == ["grpo"]
    assert record["agents"]["failed"][0]["algorithm"] == "dqn"
    assert record["agents"]["failed"][0]["version"] == "v1"
    assert record["agents"]["failed"][0]["result"] == "loop"
    assert "tier" not in record["agents"]["failed"][0]
    assert "actions" not in record["agents"]["failed"][0]
    assert "positions" not in record["agents"]["failed"][0]


def test_agent_panels_have_requested_order_and_positions():
    assert TARGET_ALGORITHMS == ("a2c", "dqn", "grpo", "ppo", "reinforce")
    assert AGENT_PANEL_POSITIONS == (
        (0, 1),
        (0, 2),
        (1, 0),
        (1, 1),
        (1, 2),
    )
