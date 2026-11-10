from __future__ import annotations

import numpy as np

from maze_rl.training.plots import (
    available_x_axes,
    axis_value,
    load_rows,
    mean_by_epoch,
)


def test_training_plots_use_dataset_epoch():
    rows = [
        {
            "episode": "10",
            "epoch": "1",
            "steps": "5",
            "path_efficiency": "0.2",
        },
        {
            "episode": "11",
            "epoch": "1",
            "steps": "6",
            "path_efficiency": "0.4",
        },
        {
            "episode": "12",
            "epoch": "2",
            "steps": "7",
            "path_efficiency": "0.8",
        },
    ]

    assert available_x_axes(rows) == [
        "epoch",
        "env_steps",
    ]
    assert axis_value(rows[0], "epoch") == 1.0

    x_values, values = mean_by_epoch(
        rows,
        "path_efficiency",
        x_axis="epoch",
    )

    np.testing.assert_array_equal(
        x_values,
        np.array([1.0, 2.0]),
    )
    np.testing.assert_allclose(
        values,
        np.array([0.3, 0.8]),
    )


def test_training_plot_rows_stop_at_early_stop_or_budget(tmp_path):
    metrics_path = tmp_path / "metrics.csv"
    metrics_path.write_text(
        "episode,epoch,path_efficiency\n"
        "1,1,0.1\n"
        "2,2,0.2\n"
        "3,3,0.3\n"
        "4,4,0.4\n"
        "5,5,0.5\n"
    )
    (tmp_path / "training_summary.json").write_text(
        '{"dataset_epochs_requested": 4, '
        '"stopped_early": true, "stopped_epoch": 3}'
    )

    assert [
        int(row["epoch"])
        for row in load_rows(metrics_path)
    ] == [1, 2, 3]
