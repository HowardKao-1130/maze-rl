from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import csv


@dataclass
class EpisodeMetrics:
    episode: int
    epoch: int
    task_index: int
    layout_index: int
    episode_return: float
    steps: int
    internal_updates: int | None
    success: bool
    wall_collisions: int
    optimal_path_length: int | None
    path_efficiency: float
    mean_abs_td_error: float | None = None
    loss: float | None = None
    policy_loss: float | None = None
    value_loss: float | None = None
    entropy: float | None = None
    approximate_kl: float | None = None
    clip_fraction: float | None = None
    mean_value: float | None = None
    mean_return: float | None = None
    mean_advantage: float | None = None
    mean_q_value: float | None = None
    max_q_value: float | None = None
    replay_size: int | None = None


def append_metrics_csv(
    path: str | Path,
    metrics: EpisodeMetrics,
) -> None:
    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    row = asdict(metrics)

    with path.open(
        "a+",
        newline="",
    ) as file:
        file.seek(0)
        existing_header = file.readline().strip()
        fieldnames = list(row.keys())
        write_header = not existing_header

        if existing_header:
            existing_fieldnames = existing_header.split(",")

            if existing_fieldnames != fieldnames:
                if not all(
                    field in fieldnames
                    for field in existing_fieldnames
                ):
                    raise RuntimeError(
                        "Existing metrics file has a different schema: "
                        f"{path}"
                    )

                file.seek(0)
                existing_rows = list(
                    csv.DictReader(file)
                )

                file.seek(0)
                file.truncate()

                writer = csv.DictWriter(
                    file,
                    fieldnames=fieldnames,
                )
                writer.writeheader()

                for existing_row in existing_rows:
                    writer.writerow(
                        {
                            field: existing_row.get(
                                field,
                                "",
                            )
                            for field in fieldnames
                        }
                    )

        file.seek(0, 2)

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        if write_header:
            writer.writeheader()

        writer.writerow(row)
