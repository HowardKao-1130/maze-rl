from scripts.summary_metrics import (
    WEIGHTED_PATH_EFFICIENCY,
    WEIGHTED_PATH_SUCCESS,
    aggregate_episode_rows,
    path_length_metrics,
)


def test_aggregate_episode_rows_computes_length_weighted_metrics():
    rows = [
        {
            "success": "True",
            "optimal_path_length": "10",
            "path_efficiency": "0.8",
        },
        {
            "success": "False",
            "optimal_path_length": "20",
            "path_efficiency": "0.0",
        },
        {
            "success": "True",
            "optimal_path_length": "20",
            "path_efficiency": "0.5",
        },
    ]

    summary = aggregate_episode_rows(rows)

    assert summary["success_rate"] == 2 / 3
    assert summary["mean_path_efficiency"] == (0.8 + 0.0 + 0.5) / 3
    assert summary[WEIGHTED_PATH_SUCCESS] == 30 / 50
    assert summary[WEIGHTED_PATH_EFFICIENCY] == (8 + 10) / 50


def test_path_length_metrics_groups_success_and_efficiency_by_length():
    metrics = path_length_metrics(
        [
            {
                "success": "True",
                "optimal_path_length": "10",
                "path_efficiency": "1.0",
            },
            {
                "success": "False",
                "optimal_path_length": "10",
                "path_efficiency": "0.0",
            },
            {
                "success": "True",
                "optimal_path_length": "20",
                "path_efficiency": "0.5",
            },
        ]
    )

    assert metrics[10]["episodes"] == 2.0
    assert metrics[10]["success_rate"] == 0.5
    assert metrics[10]["mean_path_efficiency"] == 0.5
    assert metrics[20]["success_rate"] == 1.0
    assert metrics[20]["mean_path_efficiency"] == 0.5
