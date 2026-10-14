from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Mapping


WEIGHTED_PATH_SUCCESS = "weighted_path_success"
WEIGHTED_PATH_EFFICIENCY = "weighted_path_efficiency"


def parse_summary_float(value: Any) -> float | None:
    if value in (None, ""):
        return None

    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None

    return parsed


def parse_summary_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value

    if value in (True, 1, "True", "true", "1"):
        return True
    if value in (False, 0, "False", "false", "0"):
        return False

    return None


def _path_length(row: Mapping[str, Any]) -> float | None:
    value = parse_summary_float(
        row.get("optimal_path_length")
    )

    if value is None or value <= 0:
        return None

    return value


def _success(row: Mapping[str, Any]) -> bool | None:
    return parse_summary_bool(row.get("success"))


def _efficiency(row: Mapping[str, Any]) -> float | None:
    return parse_summary_float(row.get("path_efficiency"))


def aggregate_episode_rows(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, float] | None:
    rows = list(rows)
    if not rows:
        return None

    successes = [
        success
        for row in rows
        if (success := _success(row)) is not None
    ]
    efficiencies = [
        efficiency
        for row in rows
        if (efficiency := _efficiency(row)) is not None
    ]

    path_length_rows = [
        (length, row)
        for row in rows
        if (length := _path_length(row)) is not None
    ]
    path_length_total = sum(
        length
        for length, _ in path_length_rows
    )

    summary: dict[str, float] = {
        "episodes": float(len(rows)),
    }

    if successes:
        summary["success_rate"] = sum(successes) / len(successes)

    if efficiencies:
        summary["mean_path_efficiency"] = (
            sum(efficiencies) / len(efficiencies)
        )

    if path_length_total > 0:
        summary[WEIGHTED_PATH_SUCCESS] = sum(
            length * float(bool(_success(row)))
            for length, row in path_length_rows
            if _success(row) is not None
        ) / path_length_total

        weighted_efficiency_rows = [
            (length, _efficiency(row))
            for length, row in path_length_rows
            if _efficiency(row) is not None
        ]
        weighted_efficiency_total = sum(
            length
            for length, _ in weighted_efficiency_rows
        )
        if weighted_efficiency_total > 0:
            summary[WEIGHTED_PATH_EFFICIENCY] = sum(
                length * efficiency
                for length, efficiency in weighted_efficiency_rows
            ) / weighted_efficiency_total

    return summary


def group_episode_rows(
    rows: Iterable[Mapping[str, Any]],
    key: str,
) -> dict[int, list[Mapping[str, Any]]]:
    grouped: dict[int, list[Mapping[str, Any]]] = defaultdict(list)

    for row in rows:
        value = parse_summary_float(row.get(key))
        if value is None:
            continue
        grouped[int(value)].append(row)

    return dict(grouped)


def aggregate_episode_rows_by(
    rows: Iterable[Mapping[str, Any]],
    key: str,
) -> dict[int, dict[str, float]]:
    summaries = {}

    for group, group_rows in sorted(
        group_episode_rows(rows, key).items()
    ):
        summary = aggregate_episode_rows(group_rows)
        if summary is not None:
            summaries[group] = summary

    return summaries


def path_length_metrics(
    rows: Iterable[Mapping[str, Any]],
) -> dict[int, dict[str, float]]:
    grouped: dict[int, list[Mapping[str, Any]]] = defaultdict(list)

    for row in rows:
        length = _path_length(row)
        if length is None:
            continue
        grouped[int(length)].append(row)

    summaries = {}
    for length, length_rows in sorted(grouped.items()):
        summary = aggregate_episode_rows(length_rows)
        if summary is not None:
            summaries[length] = summary

    return summaries
