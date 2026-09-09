from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys


def q_video_path(
    algorithm: str,
    task_index: int,
    runs_dir: Path,
) -> Path:
    task_name = f"task_{task_index:05d}"

    return (
        runs_dir
        / algorithm
        / "q_value_plots"
        / task_name
        / f"{task_name}_q_values.mp4"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Open a Q-value snapshot MP4."
    )

    parser.add_argument(
        "--algorithm",
        required=True,
    )
    parser.add_argument(
        "--task-index",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=Path("runs"),
    )
    parser.add_argument(
        "--video",
        type=Path,
        default=None,
        help=(
            "Explicit MP4 path. Overrides --algorithm, "
            "--task-index, and --runs-dir."
        ),
    )

    return parser.parse_args()


def open_path(path: Path) -> None:
    if sys.platform == "darwin":
        subprocess.run(
            ["open", str(path)],
            check=True,
        )
        return

    if os.name == "nt":
        os.startfile(path)  # type: ignore[attr-defined]
        return

    subprocess.run(
        ["xdg-open", str(path)],
        check=True,
    )


def main() -> None:
    args = parse_args()
    path = (
        args.video
        if args.video is not None
        else q_video_path(
            algorithm=args.algorithm,
            task_index=args.task_index,
            runs_dir=args.runs_dir,
        )
    )

    if not path.exists():
        raise SystemExit(
            f"Q-value video not found: {path}"
        )

    open_path(path)


if __name__ == "__main__":
    main()
