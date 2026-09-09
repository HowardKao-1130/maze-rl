from __future__ import annotations

import argparse
from pathlib import Path

from maze_rl.training.plots import (
    plot_task_training_metrics,
    plot_training_metrics,
    write_task_training_metrics_html,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot aggregate training metrics."
    )

    parser.add_argument(
        "--metrics",
        type=Path,
        default=Path("runs/mc/metrics.csv"),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/mc/training_metrics.png"),
    )

    parser.add_argument(
        "--rolling-window",
        type=int,
        default=0,
        help=(
            "Number of epochs used for rolling means. "
            "Use 0 to disable rolling means."
        ),
    )

    parser.add_argument(
        "--task-output",
        type=Path,
        default=Path("runs/mc/task_training_metrics.png"),
        help="Per-task training plot path.",
    )

    parser.add_argument(
        "--task-html-output",
        type=Path,
        default=Path("runs/mc/task_training_metrics.html"),
        help="Interactive per-task training plot path.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    plot_training_metrics(
        metrics_path=args.metrics,
        output_path=args.output,
        rolling_window=args.rolling_window,
    )

    print(f"Saved plot to {args.output}")

    plot_task_training_metrics(
        metrics_path=args.metrics,
        output_path=args.task_output,
    )

    print(f"Saved task plot to {args.task_output}")

    write_task_training_metrics_html(
        metrics_path=args.metrics,
        output_path=args.task_html_output,
    )

    print(
        f"Saved interactive task plot to {args.task_html_output}"
    )


if __name__ == "__main__":
    main()
