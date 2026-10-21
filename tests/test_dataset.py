import numpy as np

from maze_rl.maze.dataset import select_layout_subset


def test_select_layout_subset_preserves_tasks_and_reindexes_layouts():
    layouts = np.arange(4 * 2).reshape(4, 2)
    seeds = np.arange(10, 14)
    layout_indices = np.repeat(np.arange(4), 2)
    starts = np.arange(8 * 2).reshape(8, 2)
    goals = starts + 100

    selected = select_layout_subset(
        layouts,
        seeds,
        layout_indices,
        starts,
        goals,
        layout_count=2,
        seed=7,
    )

    selected_layouts, selected_seeds, selected_indices, selected_starts, selected_goals = selected

    assert len(selected_layouts) == 2
    assert len(selected_seeds) == 2
    assert np.array_equal(
        selected_layouts,
        layouts[selected_seeds - 10],
    )
    assert np.array_equal(
        selected_indices,
        np.repeat(np.arange(2), 2),
    )
    assert np.array_equal(selected_goals, selected_starts + 100)
