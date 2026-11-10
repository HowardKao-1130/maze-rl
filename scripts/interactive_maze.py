"""Build one maze and compare the latest trained agents in parallel."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import secrets
import sys
from uuid import uuid4

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import matplotlib.pyplot as plt
from matplotlib.widgets import Button, RadioButtons
import numpy as np
import torch

from maze_rl.envs.maze_env import MazeEnv
from maze_rl.evaluation.evaluator import choose_greedy_action
from maze_rl.maze.pathfinding import shortest_path_length
from scripts.evaluate import (
    checkpoint_encoder_configuration,
    checkpoint_network_configuration,
    create_agent,
    load_checkpoint,
)


TARGET_ALGORITHMS = ("a2c", "dqn", "grpo", "ppo", "reinforce")
GENERATION_BY_TIER = {"low": "v1", "med": "v2", "high": "v3"}
TIER_PRIORITY = {"low": 1, "med": 2, "high": 3}
PANEL_ROWS = 2
PANEL_COLUMNS = 3
SOURCE_PANEL = (0, 0)
AGENT_PANEL_POSITIONS = (
    (0, 1),
    (0, 2),
    (1, 0),
    (1, 1),
    (1, 2),
)
DEFAULT_HEIGHT = 11
DEFAULT_WIDTH = 11
DEFAULT_MAX_STEPS = 200
DEFAULT_FRAME_INTERVAL_MS = 250
DEFAULT_FAILURE_LOG = (
    REPO_ROOT / "runs" / "interactive_maze" / "failure_runs.jsonl"
)
RANDOM_TASK_SPLITS = {
    "train": ("train.npz", "train_tune.npz"),
    "val": (
        "validation.npz",
        "validation_tune.npz",
        "validation_natural.npz",
    ),
}
LOOP_REPEAT_LIMIT = 3


@dataclass(frozen=True)
class AvailableAgent:
    name: str
    algorithm: str
    tier: str
    checkpoint: Path


@dataclass(frozen=True)
class SampledTask:
    layout: np.ndarray
    start: tuple[int, int]
    goal: tuple[int, int]
    source: Path


@dataclass
class MazeRunState:
    row: int
    column: int
    algorithm: str
    option: AvailableAgent | None
    axis: object
    image: object
    layout: np.ndarray | None = None
    environment: MazeEnv | None = None
    agent: object | None = None
    start: tuple[int, int] | None = None
    goal: tuple[int, int] | None = None
    agent_position: tuple[int, int] | None = None
    current_observation: np.ndarray | None = None
    rollout_positions: list[tuple[int, int]] = field(default_factory=list)
    rollout_actions: list[int] = field(default_factory=list)
    rollout_info: list[dict] = field(default_factory=list)
    visited_positions: dict[tuple[int, int], int] = field(
        default_factory=dict
    )
    face_annotation: object | None = None
    message_annotation: object | None = None
    outcome_face: str | None = None
    load_error: str | None = None


def discover_available_agents(
    repo_root: Path = REPO_ROOT,
) -> list[AvailableAgent]:
    """Find the latest completed checkpoint for each target algorithm.

    Dataset directories are discovered rather than hard-coded. A high-tier
    checkpoint wins over medium and low, and incomplete training runs are
    ignored so their panels remain reserved.
    """
    available_by_algorithm: dict[str, AvailableAgent] = {}
    runs_root = repo_root / "runs" / "dnn"
    for training_root in sorted(runs_root.glob("*/training")):
        dataset_name = training_root.parent.name
        tier = _dataset_tier(repo_root, dataset_name)
        if tier is None:
            continue

        for algorithm_root in sorted(training_root.iterdir()):
            if not algorithm_root.is_dir():
                continue
            algorithm = algorithm_root.name
            if algorithm not in TARGET_ALGORITHMS:
                continue
            checkpoint = algorithm_root / "best_checkpoint.pt"
            summary_path = (
                algorithm_root / "training_summary.json"
            )
            if not checkpoint.exists() or not summary_path.exists():
                continue

            try:
                summary = json.loads(
                    summary_path.read_text()
                )
            except (OSError, json.JSONDecodeError):
                continue

            if summary.get("status") == "running":
                continue

            candidate = AvailableAgent(
                name=f"{algorithm}_{GENERATION_BY_TIER[tier]}",
                algorithm=algorithm,
                tier=tier,
                checkpoint=checkpoint,
            )
            current = available_by_algorithm.get(algorithm)
            if current is None or TIER_PRIORITY[tier] > TIER_PRIORITY[
                current.tier
            ]:
                available_by_algorithm[algorithm] = candidate

    return sorted(
        available_by_algorithm.values(),
        key=lambda option: TARGET_ALGORITHMS.index(option.algorithm),
    )


def _dataset_tier(
    repo_root: Path,
    dataset_name: str,
) -> str | None:
    """Infer the low/med/high label from dataset metadata and naming."""
    metadata_path = (
        repo_root
        / "data"
        / "datasets"
        / dataset_name
        / "metadata.json"
    )
    metadata = {}
    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text())
        except (OSError, json.JSONDecodeError):
            metadata = {}

    if metadata.get("stratify") is True or "dedupe" in dataset_name:
        return "high"

    train_mazes = metadata.get("train_mazes")
    if train_mazes is None:
        if "5000x" in dataset_name:
            train_mazes = 5000
        elif "2000x" in dataset_name:
            train_mazes = 2000

    if train_mazes is None:
        return None
    return "med" if int(train_mazes) >= 5000 else "low"


def _v3_dataset_directory(
    repo_root: Path = REPO_ROOT,
) -> Path:
    dataset_root = repo_root / "data" / "datasets"
    candidates = sorted(
        path
        for path in dataset_root.glob("*")
        if path.is_dir()
        and _dataset_tier(repo_root, path.name) == "high"
        and any(
            (path / filename).exists()
            for filenames in RANDOM_TASK_SPLITS.values()
            for filename in filenames
        )
    )
    if not candidates:
        raise FileNotFoundError(
            "No completed v3/high-tier dataset was found under "
            f"{dataset_root}."
        )
    return candidates[-1]


def load_random_v3_task(
    split: str,
    repo_root: Path = REPO_ROOT,
) -> SampledTask:
    """Load an unseeded random task from the v3 train or val splits."""
    if split not in RANDOM_TASK_SPLITS:
        raise ValueError(
            f"Unknown random task split {split!r}; choose 'train' or 'val'."
        )

    dataset_directory = _v3_dataset_directory(repo_root)
    split_paths = [
        dataset_directory / filename
        for filename in RANDOM_TASK_SPLITS[split]
        if (dataset_directory / filename).exists()
    ]
    if not split_paths:
        raise FileNotFoundError(
            f"No {split} task split was found in {dataset_directory}."
        )

    counts = []
    total_tasks = 0
    for path in split_paths:
        with np.load(path) as dataset:
            count = len(dataset["starts"])
        counts.append(count)
        total_tasks += count

    selected_index = secrets.randbelow(total_tasks)
    selected_path = split_paths[0]
    for path, count in zip(split_paths, counts):
        if selected_index < count:
            selected_path = path
            break
        selected_index -= count

    with np.load(selected_path) as dataset:
        layout_index = int(dataset["layout_indices"][selected_index])
        layout = np.array(
            dataset["layouts"][layout_index],
            dtype=np.uint8,
            copy=True,
        )
        start = tuple(
            int(value) for value in dataset["starts"][selected_index]
        )
        goal = tuple(
            int(value) for value in dataset["goals"][selected_index]
        )
    return SampledTask(
        layout=layout,
        start=start,
        goal=goal,
        source=selected_path,
    )


def parse_args(
    argv: list[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build one maze and compare the latest trained agents "
            "in parallel."
        )
    )
    parser.add_argument(
        "--height",
        type=int,
        default=DEFAULT_HEIGHT,
        help=f"Maze height (default: {DEFAULT_HEIGHT}).",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=DEFAULT_WIDTH,
        help=f"Maze width (default: {DEFAULT_WIDTH}).",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=DEFAULT_MAX_STEPS,
        help=f"Maximum inference steps (default: {DEFAULT_MAX_STEPS}).",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=DEFAULT_FRAME_INTERVAL_MS,
        help=(
            "Milliseconds between animation frames "
            f"(default: {DEFAULT_FRAME_INTERVAL_MS})."
        ),
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device for inference (default: cpu).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Agent seed used for deterministic tie-breaking.",
    )
    parser.add_argument(
        "--failure-log",
        type=Path,
        default=DEFAULT_FAILURE_LOG,
        help=(
            "JSONL path for interactive inference run records "
            f"(default: {DEFAULT_FAILURE_LOG})."
        ),
    )
    args = parser.parse_args(argv)

    if args.height <= 0 or args.width <= 0:
        parser.error("--height and --width must be positive.")
    if args.height == 1 and args.width == 1:
        parser.error("The maze needs at least two cells.")
    if args.max_steps <= 0:
        parser.error("--max-steps must be positive.")
    if args.interval <= 0:
        parser.error("--interval must be positive.")
    return args


def _default_positions(
    height: int,
    width: int,
) -> tuple[tuple[int, int], tuple[int, int]]:
    return (0, 0), (height - 1, width - 1)


class InteractiveMazeApp:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.available_agents = {
            option.algorithm: option
            for option in discover_available_agents()
        }
        self.maze = np.zeros(
            (args.height, args.width),
            dtype=np.uint8,
        )
        self.start, self.goal = _default_positions(
            args.height,
            args.width,
        )
        self.mode = "Wall"
        self._wall_drag_active = False
        self._last_wall_position: tuple[int, int] | None = None
        self._edit_invalidated_runs = False
        self.inference_timer = None
        self.device = torch.device(args.device)
        self.agent_cache = {}
        self.preloaded_agents = {}
        self.failure_log_path = args.failure_log.expanduser().resolve()
        self.recorded_failure_keys = self._load_recorded_failure_keys()
        self.current_run_record: dict | None = None

        self.figure, axes = plt.subplots(
            PANEL_ROWS,
            PANEL_COLUMNS,
            figsize=(14, 9),
            squeeze=False,
        )
        self.figure.subplots_adjust(
            left=0.04,
            right=0.98,
            bottom=0.06,
            top=0.74,
            wspace=0.28,
            hspace=0.38,
        )
        self.source_axis = axes[SOURCE_PANEL]
        self.source_image = self._install_grid_axis(
            self.source_axis,
            self._render_grid(
                self.maze,
                self.start,
                self.goal,
                None,
            ),
            self._source_title(),
        )
        self.panel_states: list[MazeRunState] = []
        for (row, column), algorithm in zip(
            AGENT_PANEL_POSITIONS,
            TARGET_ALGORITHMS,
        ):
            axis = axes[row, column]
            option = self.available_agents.get(algorithm)
            image = self._install_grid_axis(
                axis,
                self._blank_grid(),
                self._initial_panel_title(algorithm, option),
            )
            self.panel_states.append(
                MazeRunState(
                    row=row,
                    column=column,
                    algorithm=algorithm,
                    option=option,
                    axis=axis,
                    image=image,
                )
            )

        self._install_qt_timer()

        self.status_text = self.figure.text(
            0.04,
            0.965,
            "",
            fontsize=10,
            ha="left",
            va="top",
        )
        source_bbox = self.source_axis.get_position()
        agent_bbox = axes[0, 1].get_position()
        edit_group_width = 0.28
        edit_group_height = 0.15
        edit_group_center_x = (source_bbox.x0 + source_bbox.x1) / 2
        edit_group_y = 0.79
        edit_group_x = edit_group_center_x - edit_group_width / 2
        edit_group_axis = self.figure.add_axes(
            [
                edit_group_x,
                edit_group_y,
                edit_group_width,
                edit_group_height,
            ]
        )
        edit_group_axis.set_facecolor("#f5f5f5")
        edit_group_axis.set_xticks([])
        edit_group_axis.set_yticks([])
        edit_group_axis.set_title("Edit mode", fontsize=10, pad=4)
        self.edit_group_axis = edit_group_axis

        radio_axis = self.figure.add_axes(
            [
                edit_group_x + 0.01,
                edit_group_y + 0.02,
                0.13,
                0.10,
            ],
            facecolor="none",
        )
        self.mode_radio = RadioButtons(
            radio_axis,
            ("Wall", "Start", "Goal"),
            active=0,
        )
        for spine in radio_axis.spines.values():
            spine.set_visible(False)
        self.mode_radio.on_clicked(self._set_mode)

        random_button_x = edit_group_x + 0.145
        random_button_width = 0.11
        random_button_height = 0.035
        random_button_gap = 0.02
        random_val_y = edit_group_y + 0.03
        random_train_y = (
            random_val_y + random_button_height + random_button_gap
        )
        random_train_axis = self.figure.add_axes(
            [
                random_button_x,
                random_train_y,
                random_button_width,
                random_button_height,
            ]
        )
        self.random_train_button = Button(
            random_train_axis,
            "Load random train",
            color="#6a1b9a",
            hovercolor="#8e24aa",
        )
        self.random_train_button.label.set_color("white")
        self.random_train_button.label.set_fontsize(7)
        self.random_train_button.on_clicked(
            lambda _event: self._load_random_task("train")
        )
        random_val_axis = self.figure.add_axes(
            [
                random_button_x,
                random_val_y,
                random_button_width,
                random_button_height,
            ]
        )
        self.random_val_button = Button(
            random_val_axis,
            "Load random val",
            color="#6a1b9a",
            hovercolor="#8e24aa",
        )
        self.random_val_button.label.set_color("white")
        self.random_val_button.label.set_fontsize(7)
        self.random_val_button.on_clicked(
            lambda _event: self._load_random_task("val")
        )

        button_height = 0.055
        button_y = edit_group_y + (edit_group_height - button_height) / 2
        button_gap = 0.015
        run_width = 0.10
        reset_width = 0.08
        clear_width = 0.09
        run_x = agent_bbox.x0
        run_axis = self.figure.add_axes(
            [run_x, button_y, run_width, button_height]
        )
        self.run_button = Button(
            run_axis,
            "Run inference",
            color="#2e7d32",
            hovercolor="#43a047",
        )
        self.run_button.on_clicked(self._run_inference)

        reset_axis = self.figure.add_axes(
            [
                run_x + run_width + button_gap,
                button_y,
                reset_width,
                button_height,
            ]
        )
        self.reset_button = Button(
            reset_axis,
            "Reset run",
            color="#1565c0",
            hovercolor="#1e88e5",
        )
        self.reset_button.on_clicked(self._reset_run)

        clear_axis = self.figure.add_axes(
            [
                run_x + run_width + reset_width + 2 * button_gap,
                button_y,
                clear_width,
                button_height,
            ]
        )
        self.clear_button = Button(
            clear_axis,
            "Clear walls",
            color="#c62828",
            hovercolor="#e53935",
        )
        self.clear_button.on_clicked(self._clear_walls)
        for button in (
            self.run_button,
            self.reset_button,
            self.clear_button,
        ):
            button.label.set_color("white")

        self.figure.canvas.mpl_connect(
            "button_press_event",
            self._on_grid_click,
        )
        self.figure.canvas.mpl_connect(
            "motion_notify_event",
            self._on_grid_motion,
        )
        self.figure.canvas.mpl_connect(
            "button_release_event",
            self._on_grid_release,
        )
        self._preload_agents()

    def _install_qt_timer(self) -> None:
        try:
            from matplotlib.backends.qt_compat import (
                QtCore,
                QtWidgets,
            )
        except ImportError as error:
            raise RuntimeError(
                "The interactive maze requires a Qt Matplotlib backend "
                "for its inference timer. Start it with QtAgg."
            ) from error

        window = getattr(
            self.figure.canvas.manager,
            "window",
            None,
        )
        app = QtWidgets.QApplication.instance()
        if window is None or app is None:
            raise RuntimeError(
                "The interactive maze requires a Qt Matplotlib backend "
                "for its inference timer. Start it with QtAgg."
            )
        self.qt_core = QtCore
        self.qt_app = app
        self.qt_timer_parent = window

    def _load_agent(
        self,
        option: AvailableAgent,
        environment: MazeEnv,
    ):
        if not option.checkpoint.exists():
            raise FileNotFoundError(
                f"Checkpoint does not exist: {option.checkpoint}"
            )

        cache_key = (
            option.name,
            self.args.height,
            self.args.width,
            str(self.device),
        )
        if cache_key in self.agent_cache:
            return self.agent_cache[cache_key]

        network_architecture, residual_layers = (
            checkpoint_network_configuration(
                option.algorithm,
                option.checkpoint,
                self.device,
            )
        )
        encoder_configuration = checkpoint_encoder_configuration(
            option.algorithm,
            option.checkpoint,
            self.device,
        )
        agent = create_agent(
            algorithm=option.algorithm,
            env=environment,
            device=self.device,
            seed=self.args.seed,
            network_architecture=network_architecture,
            residual_layers=residual_layers,
            **encoder_configuration,
        )
        try:
            load_checkpoint(
                option.algorithm,
                option.checkpoint,
                agent,
                self.device,
            )
        except RuntimeError as error:
            raise ValueError(
                "The checkpoint shape does not match the selected maze "
                f"size ({self.args.height}×{self.args.width}). "
                "Use a checkpoint trained for this grid size."
            ) from error

        for network_name in ("online_network", "policy", "network"):
            network = getattr(agent, network_name, None)
            if network is not None:
                network.eval()
        self.agent_cache[cache_key] = agent
        return agent

    def _source_title(self) -> str:
        optimal_steps = shortest_path_length(
            self.maze,
            self.start,
            self.goal,
        )
        if optimal_steps is None:
            return "Build maze (0, 0) (optimal: unavailable)"
        return (
            f"Build maze (0, 0) (optimal: {optimal_steps} steps)"
        )

    def _preload_agents(self) -> None:
        self.status_text.set_text(
            "Loading latest completed agents..."
        )
        self.figure.canvas.draw_idle()
        self.qt_app.processEvents()
        for state in self.panel_states:
            if state.option is None:
                continue
            self._set_panel_title(state, "Loading")
            self.qt_app.processEvents()
            environment = MazeEnv.from_arrays(
                self.maze.copy(),
                self.start,
                self.goal,
                max_steps=self.args.max_steps,
            )
            try:
                self.preloaded_agents[state.algorithm] = self._load_agent(
                    state.option,
                    environment,
                )
            except Exception as error:
                state.load_error = str(error)
                self._show_load_error(state)
                continue
            state.load_error = None
            self._set_panel_title(state, "Ready")
        self.status_text.set_text(
            ""
        )
        self.figure.canvas.draw_idle()

    def _start_inference_record(self) -> None:
        optimal_steps = shortest_path_length(
            self.maze,
            self.start,
            self.goal,
        )
        task_key = self._task_key(
            self.maze,
            self.start,
            self.goal,
        )
        self.current_run_record = {
            "schema_version": 2,
            "run_id": uuid4().hex,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "task_key": task_key,
            "maze": {
                "height": int(self.args.height),
                "width": int(self.args.width),
                "layout": self.maze.astype(int).tolist(),
                "start": list(self.start),
                "goal": list(self.goal),
                "optimal_steps": (
                    None if optimal_steps is None else int(optimal_steps)
                ),
            },
            "agents": {
                "failed": [],
                "reserved": [
                    state.algorithm
                    for state in self.panel_states
                    if state.option is None
                ],
            },
        }

    @staticmethod
    def _task_key(
        layout: np.ndarray,
        start: tuple[int, int],
        goal: tuple[int, int],
    ) -> str:
        task = {
            "layout": layout.astype(int).tolist(),
            "start": list(start),
            "goal": list(goal),
        }
        encoded = json.dumps(
            task,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _load_recorded_failure_keys(self) -> set[str]:
        if not self.failure_log_path.exists():
            return set()
        keys = set()
        try:
            lines = self.failure_log_path.read_text(
                encoding="utf-8"
            ).splitlines()
        except OSError:
            return keys
        for line in lines:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            task_key = record.get("task_key")
            if task_key is None:
                maze = record.get("maze", {})
                layout = maze.get("layout")
                start = maze.get("start")
                goal = maze.get("goal")
                if layout is not None and start is not None and goal is not None:
                    task_key = self._task_key(
                        np.asarray(layout, dtype=np.uint8),
                        tuple(start),
                        tuple(goal),
                    )
            if task_key is None:
                continue
            for agent in record.get("agents", {}).get("failed", []):
                algorithm = agent.get("algorithm")
                version = agent.get("version")
                if algorithm and version:
                    keys.add(f"{task_key}:{algorithm}:{version}")
        return keys

    def _agent_record(self, state: MazeRunState) -> dict:
        option = state.option
        if option is None:
            raise ValueError("Cannot record an unavailable agent.")
        return {
            "algorithm": option.algorithm,
            "version": option.name.rsplit("_", 1)[-1],
            "checkpoint": str(option.checkpoint),
        }

    def _record_panel_outcome(
        self,
        state: MazeRunState,
        result: str,
        *,
        stage: str = "rollout",
    ) -> None:
        if self.current_run_record is None or state.option is None:
            return
        agent_record = self._agent_record(state)
        agent_record.update(
            {
                "stage": stage,
                "result": result,
                "steps": len(state.rollout_actions),
            }
        )
        if result != "success":
            self.current_run_record["agents"]["failed"].append(
                agent_record
            )

    def _persist_inference_record(self, status: str) -> None:
        if self.current_run_record is None:
            return
        record = {
            **self.current_run_record,
            "status": status,
            "finished_at": datetime.now(timezone.utc).isoformat(),
        }
        task_key = record["task_key"]
        new_failures = []
        for failure in record["agents"]["failed"]:
            failure_key = (
                f"{task_key}:{failure['algorithm']}:{failure['version']}"
            )
            if failure_key in self.recorded_failure_keys:
                continue
            new_failures.append(failure)
            self.recorded_failure_keys.add(failure_key)
        record["agents"]["failed"] = new_failures
        if not new_failures:
            self.current_run_record = None
            return
        try:
            self.failure_log_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )
            with self.failure_log_path.open(
                "a",
                encoding="utf-8",
            ) as output:
                json.dump(record, output, sort_keys=True)
                output.write("\n")
        except OSError as error:
            self.status_text.set_text(
                f"Could not write inference record: {error}"
            )
            self.figure.canvas.draw_idle()
            return
        self.current_run_record = None

    def _blank_grid(self) -> np.ndarray:
        return np.ones((*self.maze.shape, 3), dtype=float)

    def _render_grid(
        self,
        layout: np.ndarray,
        start: tuple[int, int],
        goal: tuple[int, int],
        agent_position: tuple[int, int] | None,
    ) -> np.ndarray:
        grid = np.ones((*layout.shape, 3), dtype=float)
        grid[layout == 1] = (0.05, 0.05, 0.05)
        grid[start] = (0.20, 0.75, 0.30)
        grid[goal] = (0.95, 0.25, 0.20)
        if agent_position is not None:
            grid[agent_position] = (0.20, 0.40, 0.95)
        if agent_position == goal:
            grid[goal] = (0.55, 0.20, 0.85)
        return grid

    def _install_grid_axis(
        self,
        axis,
        image_data: np.ndarray,
        title: str,
    ):
        image = axis.imshow(image_data, interpolation="none")
        axis.set_xticks(np.arange(self.args.width))
        axis.set_yticks(np.arange(self.args.height))
        axis.set_xticklabels([])
        axis.set_yticklabels([])
        axis.set_xticks(
            np.arange(-0.5, self.args.width, 1),
            minor=True,
        )
        axis.set_yticks(
            np.arange(-0.5, self.args.height, 1),
            minor=True,
        )
        axis.grid(
            which="minor",
            color="#888888",
            linewidth=0.8,
        )
        axis.tick_params(
            which="both",
            bottom=False,
            left=False,
        )
        axis.set_title(title, fontsize=10)
        return image

    def _initial_panel_title(
        self,
        algorithm: str,
        option: AvailableAgent | None,
    ) -> str:
        if option is None:
            return f"{algorithm.upper()} · reserved\nTraining not complete"
        generation = option.name.rsplit("_", 1)[-1]
        return f"{algorithm.upper()} · {generation}\nReady"

    def _set_panel_title(
        self,
        state: MazeRunState,
        status: str | None = None,
    ) -> None:
        if state.option is None:
            state.axis.set_title(
                f"{state.algorithm.upper()} · reserved\n"
                "Training not complete",
                fontsize=10,
            )
            return
        generation = state.option.name.rsplit("_", 1)[-1]
        title = f"{state.algorithm.upper()} · {generation}"
        if status is not None:
            title += f"\n{status}"
        state.axis.set_title(title, fontsize=10)

    def _show_load_error(self, state: MazeRunState) -> None:
        self._set_panel_title(state, "Load failed")
        state.message_annotation = state.axis.text(
            0.5,
            0.5,
            state.load_error or "Agent could not be loaded.",
            transform=state.axis.transAxes,
            ha="center",
            va="center",
            fontsize=8,
            wrap=True,
        )

    def _set_mode(self, label: str) -> None:
        self.mode = label
        descriptions = {
            "Wall": "click cells to toggle obstacles",
            "Start": "click one free cell to place the start",
            "Goal": "click one free cell to place the goal",
        }
        self.status_text.set_text(
            f"{label} mode: {descriptions[label]}."
        )
        self._wall_drag_active = False
        self._last_wall_position = None
        self.figure.canvas.draw_idle()

    def _load_random_task(self, split: str) -> None:
        try:
            task = load_random_v3_task(split)
        except (FileNotFoundError, ValueError) as error:
            self.status_text.set_text(
                f"Cannot load random task: {error}"
            )
            self.figure.canvas.draw_idle()
            return

        if task.layout.shape != (self.args.height, self.args.width):
            self.status_text.set_text(
                "Cannot load random task: v3 task dimensions are "
                f"{task.layout.shape[0]}×{task.layout.shape[1]}, but the "
                f"editor is {self.args.height}×{self.args.width}."
            )
            self.figure.canvas.draw_idle()
            return

        self.maze = task.layout
        self.start = task.start
        self.goal = task.goal
        self._invalidate_runs()
        self._render_source()
        self.status_text.set_text(
            f"Loaded random {split} task from "
            f"{task.source.name}."
        )
        self.figure.canvas.draw_idle()

    def _grid_position(self, event) -> tuple[int, int] | None:
        if event.inaxes is not self.source_axis:
            return None
        if event.xdata is None or event.ydata is None:
            return None

        row = int(round(event.ydata))
        col = int(round(event.xdata))
        if not (
            0 <= row < self.args.height
            and 0 <= col < self.args.width
        ):
            return None
        return row, col

    def _on_grid_click(self, event) -> None:
        if event.button != 1:
            return
        position = self._grid_position(event)
        if position is None:
            return

        self._wall_drag_active = self.mode == "Wall"
        self._last_wall_position = position
        self._edit_invalidated_runs = False
        changed = False

        if self.mode == "Wall":
            if position not in {self.start, self.goal}:
                self.maze[position] = 1 - self.maze[position]
                changed = True
        elif self.mode == "Start":
            if position != self.goal and self.maze[position] == 0:
                changed = self.start != position
                self.start = position
        elif self.mode == "Goal":
            if position != self.start and self.maze[position] == 0:
                changed = self.goal != position
                self.goal = position

        if changed:
            self._invalidate_runs_for_edit()
            self._render_source()
            self.figure.canvas.draw_idle()

    def _on_grid_motion(self, event) -> None:
        if not self._wall_drag_active or self.mode != "Wall":
            return
        position = self._grid_position(event)
        if position is None:
            return
        previous = self._last_wall_position or position
        changed = self._paint_wall_segment(previous, position)
        self._last_wall_position = position
        if changed:
            self._invalidate_runs_for_edit()
            self._render_source()
            self.figure.canvas.draw_idle()

    def _on_grid_release(self, _event) -> None:
        self._wall_drag_active = False
        self._last_wall_position = None
        self._edit_invalidated_runs = False

    def _paint_wall_segment(
        self,
        start: tuple[int, int],
        end: tuple[int, int],
    ) -> bool:
        changed = False
        steps = max(
            abs(end[0] - start[0]),
            abs(end[1] - start[1]),
        )
        for index in range(steps + 1):
            fraction = 0 if steps == 0 else index / steps
            row = int(round(start[0] + fraction * (end[0] - start[0])))
            column = int(
                round(start[1] + fraction * (end[1] - start[1]))
            )
            position = (row, column)
            if position in {self.start, self.goal}:
                continue
            if self.maze[position] == 0:
                self.maze[position] = 1
                changed = True
        return changed

    def _render_source(self) -> None:
        self.source_image.set_data(
            self._render_grid(
                self.maze,
                self.start,
                self.goal,
                None,
            )
        )
        self.source_axis.set_title(
            self._source_title(),
            fontsize=10,
        )

    def _clear_face_annotation(self, state: MazeRunState) -> None:
        if state.face_annotation is not None:
            state.face_annotation.remove()
            state.face_annotation = None
        if state.message_annotation is not None:
            state.message_annotation.remove()
            state.message_annotation = None

    def _render_panel(self, state: MazeRunState) -> None:
        self._clear_face_annotation(state)
        if state.layout is None or state.start is None or state.goal is None:
            state.image.set_data(self._blank_grid())
            return
        state.image.set_data(
            self._render_grid(
                state.layout,
                state.start,
                state.goal,
                state.agent_position,
            )
        )
        if state.outcome_face is not None and state.agent_position is not None:
            row, column = state.agent_position
            state.face_annotation = state.axis.text(
                column,
                row,
                state.outcome_face,
                fontsize=24,
                ha="center",
                va="center",
                zorder=5,
                bbox={
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.75,
                },
            )

    def _reset_panel(self, state: MazeRunState) -> None:
        self._clear_face_annotation(state)
        state.layout = None
        state.environment = None
        state.agent = None
        state.start = None
        state.goal = None
        state.agent_position = None
        state.current_observation = None
        state.rollout_positions.clear()
        state.rollout_actions.clear()
        state.rollout_info.clear()
        state.visited_positions.clear()
        state.outcome_face = None
        self._render_panel(state)
        if state.load_error is None:
            self._set_panel_title(state)
        else:
            self._show_load_error(state)

    def _stop_inference(self) -> None:
        if self.inference_timer is not None:
            self.inference_timer.stop()
            self.inference_timer = None

    def _invalidate_runs(self) -> None:
        if self.current_run_record is not None:
            self._persist_inference_record("interrupted")
        self._stop_inference()
        for state in self.panel_states:
            self._reset_panel(state)

    def _invalidate_runs_for_edit(self) -> None:
        if self._edit_invalidated_runs:
            return
        self._invalidate_runs()
        self._edit_invalidated_runs = True

    def _clear_walls(self, _event) -> None:
        self.maze.fill(0)
        self._invalidate_runs()
        self._render_source()
        self.status_text.set_text(
            "Wall mode: click or drag across cells to create walls."
        )
        self.figure.canvas.draw_idle()

    def _reset_run(self, _event=None) -> None:
        self._invalidate_runs()
        self.status_text.set_text("")
        self.figure.canvas.draw_idle()

    def _run_inference(self, _event) -> None:
        path_length = shortest_path_length(
            self.maze,
            self.start,
            self.goal,
        )
        if path_length is None:
            self.status_text.set_text(
                "Cannot run: start and goal are not connected."
            )
            self.figure.canvas.draw_idle()
            return

        self._invalidate_runs()
        self._start_inference_record()
        self.status_text.set_text(
            "Starting inference in the available agent panels..."
        )
        self.figure.canvas.draw_idle()
        self.qt_app.processEvents()

        active_runs = 0
        for state in self.panel_states:
            if state.option is None:
                continue
            state.layout = self.maze.copy()
            state.start = self.start
            state.goal = self.goal
            state.environment = MazeEnv.from_arrays(
                state.layout,
                state.start,
                state.goal,
                max_steps=self.args.max_steps,
            )
            self._set_panel_title(state, "Starting")
            self._render_panel(state)
            if state.load_error is not None:
                self._show_load_error(state)
                self._record_panel_outcome(
                    state,
                    state.load_error,
                    stage="load",
                )
                continue
            state.agent = self.preloaded_agents.get(state.algorithm)
            if state.agent is None:
                state.load_error = "Agent was not preloaded."
                self._show_load_error(state)
                self._record_panel_outcome(
                    state,
                    state.load_error,
                    stage="load",
                )
                continue

            observation, _ = state.environment.reset()
            state.current_observation = observation
            state.agent_position = state.environment.agent_position
            state.rollout_positions = [state.agent_position]
            state.visited_positions = {state.agent_position: 1}
            self._set_panel_title(state, "Running")
            self._render_panel(state)
            active_runs += 1

        self.status_text.set_text(
            ""
        )
        self.qt_app.processEvents()
        if active_runs == 0:
            self.status_text.set_text(
                "No completed target agents are available yet."
            )
            self._persist_inference_record("completed")
            self.figure.canvas.draw_idle()
            return

        self.inference_timer = self.qt_core.QTimer(
            self.qt_timer_parent
        )
        self.inference_timer.setInterval(self.args.interval)
        self.inference_timer.timeout.connect(
            self._advance_inference
        )
        self.inference_timer.start()
        self.figure.canvas.draw_idle()

    def _advance_inference(self) -> None:
        active_runs = 0
        for state in self.panel_states:
            if state.current_observation is None:
                continue
            active_runs += 1
            self._advance_panel(state)
        if active_runs == 0 or not any(
            state.current_observation is not None
            for state in self.panel_states
        ):
            self._stop_inference()
            self._persist_inference_record("completed")
            self.status_text.set_text("")
        self.figure.canvas.draw_idle()

    def _advance_panel(self, state: MazeRunState) -> None:
        if state.environment is None or state.agent is None:
            return
        try:
            action = choose_greedy_action(
                state.option.algorithm,
                state.agent,
                state.environment,
                state.current_observation,
            )
            (
                state.current_observation,
                reward,
                terminated,
                truncated,
                info,
            ) = state.environment.step(action)
        except Exception as error:
            self._finish_panel(state, f"error: {error}")
            return

        state.agent_position = state.environment.agent_position
        state.rollout_actions.append(int(action))
        state.rollout_positions.append(state.environment.agent_position)
        state.rollout_info.append(
            {**info, "reward": float(reward)}
        )

        if terminated:
            self._finish_panel(state, "success")
            return
        if info["hit_wall"]:
            self._finish_panel(state, "wall collision")
            return
        if truncated:
            self._finish_panel(state, "step limit")
            return

        position = state.agent_position
        state.visited_positions[position] = (
            state.visited_positions.get(position, 0) + 1
        )
        if state.visited_positions[position] >= LOOP_REPEAT_LIMIT:
            state.rollout_info[-1]["looped"] = True
            self._finish_panel(state, "loop")
            return

        self._render_panel(state)

    def _finish_panel(self, state: MazeRunState, result: str) -> None:
        state.current_observation = None
        state.outcome_face = "☺" if result == "success" else "☹"
        if state.rollout_info:
            state.rollout_info[-1]["failure_reason"] = (
                None if result == "success" else result
            )
        self._record_panel_outcome(state, result)
        self._render_panel(state)
        self._set_panel_title(
            state,
            f"{result} ({len(state.rollout_actions)} steps)",
        )

    def show(self) -> None:
        plt.show()


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    app = InteractiveMazeApp(args)
    app.show()


if __name__ == "__main__":
    main()
