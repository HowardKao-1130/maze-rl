from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import subprocess


EVENT_FILE_PREFIX = "events.out.tfevents."


def default_tensorboard_dir(
    algorithm: str,
    runs_dir: Path = Path("runs"),
) -> Path:
    return runs_dir / algorithm / "tensorboard"


def tensorboard_event_files(
    logdir: Path,
) -> list[Path]:
    return sorted(
        path
        for path in logdir.glob(
            f"{EVENT_FILE_PREFIX}*"
        )
        if path.is_file()
    )


def latest_event_file(
    logdir: Path,
) -> Path:
    event_files = tensorboard_event_files(
        logdir
    )

    if not event_files:
        raise FileNotFoundError(
            f"No TensorBoard event files found in {logdir}"
        )

    return max(
        event_files,
        key=lambda path: (
            path.stat().st_mtime,
            path.name,
        ),
    )


def latest_logdir(
    event_file: Path,
    link_root: Path,
) -> Path:
    digest = hashlib.sha256(
        event_file.resolve().as_posix().encode(
            "utf-8"
        )
    ).hexdigest()[:16]

    link_dir = link_root / digest
    link_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    link_path = link_dir / event_file.name

    if link_path.exists() or link_path.is_symlink():
        link_path.unlink()

    link_path.symlink_to(
        event_file.resolve()
    )

    return link_dir


def build_tensorboard_command(
    logdir: Path,
    tensorboard_args: list[str],
) -> list[str]:
    return [
        "tensorboard",
        "--logdir",
        str(logdir),
        *tensorboard_args,
    ]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Open TensorBoard for one maze-rl run. By default, "
            "only the latest event file is shown."
        )
    )

    parser.add_argument(
        "--algorithm",
        default="dqn",
        choices=[
            "dqn",
            "reinforce",
            "a2c",
            "ppo",
            "grpo",
        ],
        help=(
            "Algorithm run to open when --logdir is not provided."
        ),
    )

    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=Path("runs"),
    )

    parser.add_argument(
        "--logdir",
        type=Path,
        default=None,
        help=(
            "TensorBoard directory to inspect. Defaults to "
            "runs/<algorithm>/tensorboard."
        ),
    )

    parser.add_argument(
        "--event-file",
        type=Path,
        default=None,
        help=(
            "Specific TensorBoard event file to show. If omitted, "
            "the latest event file in --logdir is used."
        ),
    )

    parser.add_argument(
        "--all-events",
        action="store_true",
        help=(
            "Show every event file in --logdir instead of only the "
            "latest one."
        ),
    )

    parser.add_argument(
        "--link-root",
        type=Path,
        default=Path("/tmp/maze-rl-tensorboard"),
        help=(
            "Directory used for temporary symlink logdirs when "
            "showing a single event file."
        ),
    )

    return parser.parse_known_args()


def main() -> None:
    args, tensorboard_args = parse_args()

    logdir = (
        args.logdir
        if args.logdir is not None
        else default_tensorboard_dir(
            args.algorithm,
            args.runs_dir,
        )
    )

    if args.all_events:
        selected_logdir = logdir
        print(
            f"Opening all TensorBoard events in {selected_logdir}"
        )
    else:
        event_file = (
            args.event_file
            if args.event_file is not None
            else latest_event_file(logdir)
        )

        if not event_file.is_file():
            raise FileNotFoundError(
                f"TensorBoard event file not found: {event_file}"
            )

        selected_logdir = latest_logdir(
            event_file=event_file,
            link_root=args.link_root,
        )

        print(
            f"Opening TensorBoard event file {event_file}"
        )

    subprocess.run(
        build_tensorboard_command(
            selected_logdir,
            tensorboard_args,
        ),
        check=True,
    )


if __name__ == "__main__":
    main()
