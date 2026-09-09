from __future__ import annotations

from collections import defaultdict
import csv
import json
import math
import os
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    "/tmp/maze-rl-matplotlib",
)

import matplotlib
import numpy as np

matplotlib.use("Agg")

import matplotlib.pyplot as plt


X_AXIS_LABELS = {
    "epoch": "Epoch",
    "env_steps": "Cumulative env steps",
    "internal_updates": "Cumulative internal updates",
}

HTML_MAX_POINTS_PER_SERIES = 2000


def parse_optional_float(value) -> float | None:
    if value is None or value == "":
        return None

    return float(value)


def parse_optional_bool(value) -> bool | None:
    if value is None or value == "":
        return None

    if value in {"True", "true", "1"}:
        return True

    if value in {"False", "false", "0"}:
        return False

    raise ValueError(
        f"Invalid boolean value: {value}"
    )


def load_rows(path: Path) -> list[dict]:
    with path.open(newline="") as file:
        reader = csv.DictReader(file)
        return list(reader)


def add_cumulative_counters(
    rows: list[dict],
) -> list[dict]:
    cumulative_env_steps = 0.0
    cumulative_internal_updates = 0.0
    has_internal_updates = any(
        "internal_updates" in row
        and parse_optional_float(
            row["internal_updates"]
        )
        is not None
        for row in rows
    )

    enriched_rows = []

    for row in rows:
        enriched_row = dict(row)

        steps = metric_value(
            row,
            "steps",
        )

        if steps is not None:
            cumulative_env_steps += steps
            enriched_row[
                "_cumulative_env_steps"
            ] = cumulative_env_steps

        internal_updates = metric_value(
            row,
            "internal_updates",
        )

        if has_internal_updates:
            if internal_updates is None:
                enriched_row[
                    "_cumulative_internal_updates"
                ] = None
            else:
                cumulative_internal_updates += (
                    internal_updates
                )

                enriched_row[
                    "_cumulative_internal_updates"
                ] = cumulative_internal_updates

        enriched_rows.append(enriched_row)

    return enriched_rows


def metric_value(
    row: dict,
    preferred_name: str,
    legacy_name: str | None = None,
) -> float | None:
    if preferred_name in row:
        return parse_optional_float(
            row[preferred_name]
        )

    if legacy_name is not None and legacy_name in row:
        return parse_optional_float(
            row[legacy_name]
        )

    return None


def success_value(row: dict) -> float | None:
    if "success" not in row:
        return None

    value = parse_optional_bool(
        row["success"]
    )

    if value is None:
        return None

    return float(value)


def epoch_value(row: dict) -> int:
    return int(
        row.get("epoch") or row["episode"]
    )


def axis_value(
    row: dict,
    x_axis: str,
) -> float | None:
    if x_axis == "epoch":
        return float(epoch_value(row))

    if x_axis == "env_steps":
        return metric_value(
            row,
            "_cumulative_env_steps",
        )

    if x_axis == "internal_updates":
        return metric_value(
            row,
            "_cumulative_internal_updates",
        )

    raise ValueError(x_axis)


def available_x_axes(
    rows: list[dict],
) -> list[str]:
    axes = [
        "epoch",
        "env_steps",
    ]

    if any(
        axis_value(row, "internal_updates")
        is not None
        for row in rows
    ):
        axes.append("internal_updates")

    return axes


def mean_by_epoch(
    rows: list[dict],
    metric_name: str,
    legacy_name: str | None = None,
    x_axis: str = "epoch",
) -> tuple[np.ndarray, np.ndarray]:
    values_by_epoch: dict[int, list[float]] = {}
    x_values_by_epoch: dict[int, list[float]] = {}

    for row in rows:
        epoch = epoch_value(row)
        x_value = axis_value(row, x_axis)
        value = metric_value(
            row,
            metric_name,
            legacy_name,
        )

        if value is None or x_value is None:
            continue

        values_by_epoch.setdefault(
            epoch,
            [],
        ).append(value)
        x_values_by_epoch.setdefault(
            epoch,
            [],
        ).append(x_value)

    sorted_epochs = sorted(values_by_epoch)

    x_values = np.array(
        [
            max(x_values_by_epoch[epoch])
            for epoch in sorted_epochs
        ],
        dtype=np.float64,
    )

    values = np.array(
        [
            np.mean(values_by_epoch[epoch])
            for epoch in sorted_epochs
        ],
        dtype=np.float64,
    )

    return x_values, values


def success_rate_by_epoch(
    rows: list[dict],
    x_axis: str = "epoch",
) -> tuple[np.ndarray, np.ndarray]:
    values_by_epoch: dict[int, list[float]] = {}
    x_values_by_epoch: dict[int, list[float]] = {}

    for row in rows:
        epoch = epoch_value(row)
        x_value = axis_value(row, x_axis)
        value = success_value(row)

        if value is None or x_value is None:
            continue

        values_by_epoch.setdefault(
            epoch,
            [],
        ).append(value)
        x_values_by_epoch.setdefault(
            epoch,
            [],
        ).append(x_value)

    sorted_epochs = sorted(values_by_epoch)

    x_values = np.array(
        [
            max(x_values_by_epoch[epoch])
            for epoch in sorted_epochs
        ],
        dtype=np.float64,
    )

    values = np.array(
        [
            np.mean(values_by_epoch[epoch])
            for epoch in sorted_epochs
        ],
        dtype=np.float64,
    )

    return x_values, values


def rolling_mean(
    values: np.ndarray,
    window: int,
) -> np.ndarray:
    if window <= 1 or len(values) == 0:
        return values

    window = min(window, len(values))
    weights = np.ones(window) / window

    return np.convolve(
        values,
        weights,
        mode="valid",
    )


def rolling_epochs(
    epochs: np.ndarray,
    window: int,
) -> np.ndarray:
    if window <= 1 or len(epochs) == 0:
        return epochs

    window = min(window, len(epochs))
    return epochs[window - 1:]


def plot_series(
    axis,
    x_values: np.ndarray,
    values: np.ndarray,
    rolling_window: int,
    ylabel: str,
    label: str,
) -> None:
    axis.plot(
        x_values,
        values,
        color="#b8c0cc",
        linewidth=1,
        alpha=0.7,
        label=label,
    )

    if rolling_window <= 1:
        axis.set_ylabel(ylabel)
        axis.grid(
            True,
            alpha=0.25,
        )
        axis.legend()
        return

    smooth_values = rolling_mean(
        values,
        rolling_window,
    )
    smooth_epochs = rolling_epochs(
        x_values,
        rolling_window,
    )

    axis.plot(
        smooth_epochs,
        smooth_values,
        color="#136f63",
        linewidth=2,
        label=(
            f"{rolling_window}-epoch rolling mean "
            f"of {label}"
        ),
    )

    axis.set_ylabel(ylabel)
    axis.grid(
        True,
        alpha=0.25,
    )
    axis.legend()


def add_task_legend(axis, task_count: int) -> None:
    columns = max(
        1,
        min(4, math.ceil(task_count / 20)),
    )

    axis.legend(
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        fontsize=7,
        ncol=columns,
        borderaxespad=0,
    )


def plot_task_series(
    axis,
    rows: list[dict],
    metric_name: str,
    ylabel: str,
    legacy_name: str | None = None,
    is_success: bool = False,
    x_axis: str = "epoch",
    show_legend: bool = True,
) -> bool:
    values_by_task: dict[int, list[tuple[float, float]]] = (
        defaultdict(list)
    )

    for row in rows:
        if "task_index" not in row:
            continue

        x_value = axis_value(row, x_axis)
        task_index = int(row["task_index"])

        if is_success:
            value = success_value(row)
        else:
            value = metric_value(
                row,
                metric_name,
                legacy_name,
            )

        if value is None or x_value is None:
            continue

        values_by_task[task_index].append(
            (x_value, value)
        )

    if not values_by_task:
        return False

    for task_index, points in sorted(
        values_by_task.items()
    ):
        points.sort()
        task_x_values = np.array(
            [x_value for x_value, _ in points],
            dtype=np.float64,
        )
        task_values = np.array(
            [value for _, value in points],
            dtype=np.float64,
        )

        axis.plot(
            task_x_values,
            task_values,
            linewidth=0.8,
            alpha=0.35,
            label=f"task {task_index}",
        )

    axis.set_ylabel(ylabel)
    axis.grid(
        True,
        alpha=0.25,
    )

    if show_legend:
        add_task_legend(
            axis,
            len(values_by_task),
        )

    return True


def metric_specs(
    rows: list[dict],
) -> list[dict]:
    specs = [
        {
            "name": "episode_return",
            "legacy_name": "reward",
            "ylabel": "Episode return",
            "label": "mean episode return",
            "is_success": False,
        },
        {
            "name": "success",
            "legacy_name": None,
            "ylabel": "Success rate",
            "label": "mean success rate",
            "is_success": True,
        },
        {
            "name": "steps",
            "legacy_name": None,
            "ylabel": "Env steps per episode",
            "label": "mean env steps per episode",
            "is_success": False,
        },
    ]

    has_td_errors = any(
        metric_value(
            row,
            "mean_abs_td_error",
        )
        is not None
        for row in rows
    )

    if has_td_errors:
        specs.append(
            {
                "name": "mean_abs_td_error",
                "legacy_name": None,
                "ylabel": "Mean absolute TD error",
                "label": "mean absolute TD error",
                "is_success": False,
            }
        )

    return specs


def axes_grid(
    axes,
    row_count: int,
    column_count: int,
) -> np.ndarray:
    axes = np.asarray(axes)

    if axes.ndim == 0:
        return axes.reshape(1, 1)

    if row_count == 1:
        return axes.reshape(1, column_count)

    if column_count == 1:
        return axes.reshape(row_count, 1)

    return axes


def values_for_spec(
    rows: list[dict],
    spec: dict,
    x_axis: str,
) -> tuple[np.ndarray, np.ndarray]:
    if spec["is_success"]:
        return success_rate_by_epoch(
            rows,
            x_axis=x_axis,
        )

    return mean_by_epoch(
        rows,
        spec["name"],
        legacy_name=spec["legacy_name"],
        x_axis=x_axis,
    )


def format_x_axis_title(x_axis: str) -> str:
    return X_AXIS_LABELS[x_axis]


def task_series_payload(
    rows: list[dict],
    spec: dict,
    x_axis: str,
) -> list[dict]:
    values_by_task: dict[int, list[dict]] = defaultdict(
        list
    )

    for row in rows:
        if "task_index" not in row:
            continue

        x_value = axis_value(row, x_axis)
        task_index = int(row["task_index"])

        if spec["is_success"]:
            value = success_value(row)
        else:
            value = metric_value(
                row,
                spec["name"],
                spec["legacy_name"],
            )

        if value is None or x_value is None:
            continue

        values_by_task[task_index].append(
            {
                "x": float(x_value),
                "y": float(value),
            }
        )

    series = []

    for task_index, points in sorted(
        values_by_task.items()
    ):
        points.sort(
            key=lambda point: point["x"]
        )
        series.append(
            {
                "task": task_index,
                "points": downsample_points(
                    points,
                    HTML_MAX_POINTS_PER_SERIES,
                ),
            }
        )

    return series


def downsample_points(
    points: list[dict],
    max_points: int,
) -> list[dict]:
    if len(points) <= max_points:
        return points

    indices = np.linspace(
        0,
        len(points) - 1,
        num=max_points,
        dtype=np.int64,
    )

    return [
        points[int(index)]
        for index in np.unique(indices)
    ]


def task_metrics_html_payload(
    rows: list[dict],
) -> dict:
    specs = metric_specs(rows)
    x_axes = available_x_axes(rows)
    tasks = sorted(
        {
            int(row["task_index"])
            for row in rows
            if "task_index" in row
            and row["task_index"] != ""
        }
    )

    charts = []

    for spec in specs:
        for x_axis in x_axes:
            charts.append(
                {
                    "title": (
                        f"{spec['ylabel']} vs "
                        f"{format_x_axis_title(x_axis)}"
                    ),
                    "xLabel": format_x_axis_title(
                        x_axis
                    ),
                    "yLabel": spec["ylabel"],
                    "series": task_series_payload(
                        rows,
                        spec,
                        x_axis,
                    ),
                }
            )

    return {
        "tasks": tasks,
        "charts": charts,
    }


def write_task_training_metrics_html(
    metrics_path: Path,
    output_path: Path,
) -> None:
    rows = add_cumulative_counters(
        load_rows(metrics_path)
    )

    if not rows:
        raise RuntimeError(
            f"No metrics found in {metrics_path}"
        )

    payload = json.dumps(
        task_metrics_html_payload(rows)
    ).replace("</", "<\\/")

    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Task training metrics</title>
<style>
:root {{
  color-scheme: light;
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  color: #17202a;
  background: #f6f8fb;
}}
body {{
  margin: 0;
}}
main {{
  max-width: 1280px;
  margin: 0 auto;
  padding: 24px;
}}
h1 {{
  margin: 0 0 4px;
  font-size: 24px;
  font-weight: 700;
}}
.source {{
  margin: 0 0 18px;
  color: #5f6b7a;
  font-size: 13px;
}}
.controls {{
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  align-items: center;
  margin-bottom: 18px;
}}
button {{
  border: 1px solid #bdc7d3;
  border-radius: 6px;
  background: #ffffff;
  color: #17202a;
  padding: 7px 10px;
  font: inherit;
  cursor: pointer;
}}
.task-list {{
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
}}
.task-option {{
  display: inline-flex;
  align-items: center;
  gap: 5px;
  padding: 4px 7px;
  border: 1px solid #d3dae4;
  border-radius: 6px;
  background: #ffffff;
  font-size: 12px;
}}
.swatch {{
  width: 10px;
  height: 10px;
  border-radius: 50%;
  display: inline-block;
}}
.chart-grid {{
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(420px, 1fr));
  gap: 16px;
}}
.chart {{
  background: #ffffff;
  border: 1px solid #dce3eb;
  border-radius: 8px;
  padding: 12px;
}}
.chart-title {{
  margin: 0 0 8px;
  font-size: 14px;
  font-weight: 650;
}}
svg {{
  display: block;
  width: 100%;
  height: auto;
}}
.axis text {{
  fill: #667587;
  font-size: 11px;
}}
.axis path,
.axis line {{
  stroke: #9aa8b8;
}}
.grid-line {{
  stroke: #e5ebf2;
}}
@media (max-width: 560px) {{
  main {{
    padding: 14px;
  }}
  .chart-grid {{
    grid-template-columns: 1fr;
  }}
}}
</style>
</head>
<body>
<main>
<h1>Task training metrics</h1>
<p class="source">{metrics_path.as_posix()}</p>
<div class="controls">
  <button id="select-all" type="button">All</button>
  <button id="select-none" type="button">None</button>
  <div id="task-list" class="task-list"></div>
</div>
<div id="charts" class="chart-grid"></div>
</main>
<script>
const payload = {payload};
const colors = [
  "#136f63", "#c44536", "#3f7cac", "#d58936",
  "#6f4e7c", "#4d908e", "#9d4edd", "#577590",
  "#bc6c25", "#2a9d8f", "#e76f51", "#386641"
];
let selected = new Set(payload.tasks.map(String));

function colorForTask(task) {{
  const index = Math.abs(Number(task)) % colors.length;
  return colors[index];
}}

function makeSvg(tag, attrs = {{}}) {{
  const node = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [key, value] of Object.entries(attrs)) {{
    node.setAttribute(key, value);
  }}
  return node;
}}

function makeControls() {{
  const list = document.getElementById("task-list");
  list.innerHTML = "";
  for (const task of payload.tasks) {{
    const label = document.createElement("label");
    label.className = "task-option";
    const input = document.createElement("input");
    input.type = "checkbox";
    input.value = String(task);
    input.checked = selected.has(String(task));
    input.addEventListener("change", () => {{
      if (input.checked) {{
        selected.add(input.value);
      }} else {{
        selected.delete(input.value);
      }}
      drawCharts();
    }});
    const swatch = document.createElement("span");
    swatch.className = "swatch";
    swatch.style.backgroundColor = colorForTask(task);
    const text = document.createElement("span");
    text.textContent = `task ${{task}}`;
    label.append(input, swatch, text);
    list.append(label);
  }}
}}

function scale(domainMin, domainMax, rangeMin, rangeMax) {{
  if (domainMin === domainMax) {{
    return () => (rangeMin + rangeMax) / 2;
  }}
  return value => rangeMin + (
    (value - domainMin) / (domainMax - domainMin)
  ) * (rangeMax - rangeMin);
}}

function formatTick(value) {{
  if (Math.abs(value) >= 1000) {{
    return value.toExponential(1);
  }}
  return Number(value.toPrecision(4)).toString();
}}

function drawChart(container, chart) {{
  container.innerHTML = "";
  const width = 760;
  const height = 270;
  const margin = {{ left: 62, right: 18, top: 16, bottom: 46 }};
  const innerWidth = width - margin.left - margin.right;
  const innerHeight = height - margin.top - margin.bottom;
  const svg = makeSvg("svg", {{
    viewBox: `0 0 ${{width}} ${{height}}`,
    role: "img"
  }});

  const visibleSeries = chart.series.filter(
    series => selected.has(String(series.task))
  );
  const points = visibleSeries.flatMap(series => series.points);

  if (points.length === 0) {{
    svg.append(makeSvg("text", {{
      x: width / 2,
      y: height / 2,
      "text-anchor": "middle",
      fill: "#667587"
    }}));
    svg.lastChild.textContent = "No selected data";
    container.append(svg);
    return;
  }}

  let minX = Math.min(...points.map(point => point.x));
  let maxX = Math.max(...points.map(point => point.x));
  let minY = Math.min(...points.map(point => point.y));
  let maxY = Math.max(...points.map(point => point.y));
  if (minY === maxY) {{
    minY -= 1;
    maxY += 1;
  }}
  const yPad = (maxY - minY) * 0.05;
  minY -= yPad;
  maxY += yPad;

  const x = scale(minX, maxX, margin.left, margin.left + innerWidth);
  const y = scale(maxY, minY, margin.top, margin.top + innerHeight);

  const axisGroup = makeSvg("g", {{ class: "axis" }});
  for (let index = 0; index <= 4; index += 1) {{
    const t = index / 4;
    const tickX = minX + (maxX - minX) * t;
    const tickY = minY + (maxY - minY) * t;
    const px = x(tickX);
    const py = y(tickY);
    svg.append(makeSvg("line", {{
      class: "grid-line",
      x1: margin.left,
      x2: margin.left + innerWidth,
      y1: py,
      y2: py
    }}));
    const xText = makeSvg("text", {{
      x: px,
      y: height - 14,
      "text-anchor": "middle"
    }});
    xText.textContent = formatTick(tickX);
    axisGroup.append(xText);
    const yText = makeSvg("text", {{
      x: margin.left - 8,
      y: py + 4,
      "text-anchor": "end"
    }});
    yText.textContent = formatTick(tickY);
    axisGroup.append(yText);
  }}

  axisGroup.append(makeSvg("line", {{
    x1: margin.left,
    x2: margin.left,
    y1: margin.top,
    y2: margin.top + innerHeight
  }}));
  axisGroup.append(makeSvg("line", {{
    x1: margin.left,
    x2: margin.left + innerWidth,
    y1: margin.top + innerHeight,
    y2: margin.top + innerHeight
  }}));
  svg.append(axisGroup);

  for (const series of visibleSeries) {{
    const pointString = series.points
      .map(point => `${{x(point.x)}},${{y(point.y)}}`)
      .join(" ");
    svg.append(makeSvg("polyline", {{
      points: pointString,
      fill: "none",
      stroke: colorForTask(series.task),
      "stroke-width": 1.8,
      "stroke-opacity": 0.9
    }}));
  }}

  const xLabel = makeSvg("text", {{
    x: margin.left + innerWidth / 2,
    y: height - 2,
    "text-anchor": "middle",
    fill: "#3b4856",
    "font-size": 12
  }});
  xLabel.textContent = chart.xLabel;
  svg.append(xLabel);
  const yLabel = makeSvg("text", {{
    x: 14,
    y: margin.top + innerHeight / 2,
    transform: `rotate(-90 14 ${{margin.top + innerHeight / 2}})`,
    "text-anchor": "middle",
    fill: "#3b4856",
    "font-size": 12
  }});
  yLabel.textContent = chart.yLabel;
  svg.append(yLabel);

  container.append(svg);
}}

function drawCharts() {{
  const charts = document.getElementById("charts");
  charts.innerHTML = "";
  for (const chart of payload.charts) {{
    const section = document.createElement("section");
    section.className = "chart";
    const title = document.createElement("h2");
    title.className = "chart-title";
    title.textContent = chart.title;
    const figure = document.createElement("div");
    section.append(title, figure);
    charts.append(section);
    drawChart(figure, chart);
  }}
}}

document.getElementById("select-all").addEventListener("click", () => {{
  selected = new Set(payload.tasks.map(String));
  makeControls();
  drawCharts();
}});
document.getElementById("select-none").addEventListener("click", () => {{
  selected = new Set();
  makeControls();
  drawCharts();
}});

makeControls();
drawCharts();
</script>
</body>
</html>
"""

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    output_path.write_text(
        html,
        encoding="utf-8",
    )


def plot_training_metrics(
    metrics_path: Path,
    output_path: Path,
    rolling_window: int = 0,
) -> None:
    rows = add_cumulative_counters(
        load_rows(metrics_path)
    )

    if not rows:
        raise RuntimeError(
            f"No metrics found in {metrics_path}"
        )

    specs = metric_specs(rows)
    x_axes = available_x_axes(rows)
    figure, axes = plt.subplots(
        len(specs),
        len(x_axes),
        figsize=(5.5 * len(x_axes), 3.2 * len(specs)),
        sharex="col",
    )
    axes = axes_grid(
        axes,
        len(specs),
        len(x_axes),
    )

    for row_index, spec in enumerate(specs):
        for column_index, x_axis in enumerate(x_axes):
            x_values, values = values_for_spec(
                rows,
                spec,
                x_axis,
            )

            axis = axes[row_index, column_index]
            plot_series(
                axis=axis,
                x_values=x_values,
                values=values,
                rolling_window=rolling_window,
                ylabel=(
                    spec["ylabel"]
                    if column_index == 0
                    else ""
                ),
                label=spec["label"],
            )

            if spec["is_success"]:
                axis.set_ylim(-0.05, 1.05)

            if row_index == 0:
                axis.set_title(
                    f"vs {format_x_axis_title(x_axis)}"
                )

            if row_index == len(specs) - 1:
                axis.set_xlabel(
                    format_x_axis_title(x_axis)
                )

    figure.suptitle(
        metrics_path.as_posix()
    )
    figure.tight_layout()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    figure.savefig(
        output_path,
        dpi=150,
    )
    plt.close(figure)


def plot_task_training_metrics(
    metrics_path: Path,
    output_path: Path,
) -> None:
    rows = add_cumulative_counters(
        load_rows(metrics_path)
    )

    if not rows:
        raise RuntimeError(
            f"No metrics found in {metrics_path}"
        )

    specs = metric_specs(rows)
    x_axes = available_x_axes(rows)
    figure, axes = plt.subplots(
        len(specs),
        len(x_axes),
        figsize=(5.5 * len(x_axes), 3.2 * len(specs)),
        sharex="col",
    )
    axes = axes_grid(
        axes,
        len(specs),
        len(x_axes),
    )

    for row_index, spec in enumerate(specs):
        for column_index, x_axis in enumerate(x_axes):
            axis = axes[row_index, column_index]
            plot_task_series(
                axis=axis,
                rows=rows,
                metric_name=spec["name"],
                legacy_name=spec["legacy_name"],
                ylabel=(
                    spec["ylabel"]
                    if column_index == 0
                    else ""
                ),
                is_success=spec["is_success"],
                x_axis=x_axis,
                show_legend=(
                    row_index == 0
                    and column_index == len(x_axes) - 1
                ),
            )

            if spec["is_success"]:
                axis.set_ylim(-0.05, 1.05)

            if row_index == 0:
                axis.set_title(
                    f"vs {format_x_axis_title(x_axis)}"
                )

            if row_index == len(specs) - 1:
                axis.set_xlabel(
                    format_x_axis_title(x_axis)
                )

    figure.suptitle(
        f"Per-task metrics: {metrics_path.as_posix()}"
    )
    figure.tight_layout(
        rect=(0, 0, 0.82, 1),
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    figure.savefig(
        output_path,
        dpi=150,
    )
    plt.close(figure)
