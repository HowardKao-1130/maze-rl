from pathlib import Path

import pytest

from scripts.dataset_paths import (
    DatasetBundlePaths,
    dataset_bundle_dir_for_reference,
    dataset_bundle_for_reference,
)


DATASET_NAME = (
    "dataset_5000x500x500_t5_seed42_stratified_s00_m60_l40_dedupe050"
)


def test_combined_dataset_bundle_exposes_train_and_tuning_splits(
    tmp_path,
):
    dataset_dir = tmp_path / DATASET_NAME
    dataset_dir.mkdir()
    for filename in (
        "train.npz",
        "train_tune.npz",
        "validation.npz",
        "validation_tune.npz",
        "same_layout_new_goals.npz",
        "validation_natural.npz",
        "test.npz",
        "test_natural.npz",
    ):
        (dataset_dir / filename).touch()

    bundle = dataset_bundle_for_reference(
        Path(DATASET_NAME),
        data_root=tmp_path,
    )

    assert bundle == DatasetBundlePaths(dataset_dir)
    assert bundle.train.parent == bundle.train_tune.parent
    assert bundle.validation.parent == bundle.validation_tune.parent
    assert bundle.train_tune.name == "train_tune.npz"
    assert bundle.validation_tune.name == "validation_tune.npz"


def test_dataset_bundle_reference_preserves_explicit_directory():
    explicit = Path("data") / "datasets" / DATASET_NAME

    assert dataset_bundle_dir_for_reference(explicit) == explicit


def test_dataset_bundle_requires_full_training_split(tmp_path):
    dataset_dir = tmp_path / DATASET_NAME
    dataset_dir.mkdir()
    (dataset_dir / "validation.npz").touch()

    with pytest.raises(ValueError, match="train.npz"):
        dataset_bundle_for_reference(
            dataset_dir,
            data_root=tmp_path,
        )
