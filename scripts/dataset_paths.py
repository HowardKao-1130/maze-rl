"""Shared paths and roles for generated dataset bundles.

Generated datasets are bundles rather than isolated split files.  In
particular, the training layout and its smaller tuning subset live beside one
another in the same canonical ``dataset_*`` directory.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from scripts.experiment_naming import DEFAULT_DATA_OUTPUT_ROOT


@dataclass(frozen=True)
class DatasetBundlePaths:
    """Paths for every split that may be present in a dataset bundle."""

    directory: Path

    @property
    def metadata(self) -> Path:
        return self.directory / "metadata.json"

    @property
    def train(self) -> Path:
        return self.directory / "train.npz"

    @property
    def train_tune(self) -> Path:
        return self.directory / "train_tune.npz"

    @property
    def validation(self) -> Path:
        return self.directory / "validation.npz"

    @property
    def validation_tune(self) -> Path:
        return self.directory / "validation_tune.npz"

    @property
    def same_layout(self) -> Path:
        return self.directory / "same_layout_new_goals.npz"

    @property
    def validation_natural(self) -> Path:
        return self.directory / "validation_natural.npz"

    @property
    def test(self) -> Path:
        return self.directory / "test.npz"

    @property
    def test_natural(self) -> Path:
        return self.directory / "test_natural.npz"

    def require_train(self) -> "DatasetBundlePaths":
        if not self.train.exists():
            raise ValueError(
                "Dataset directory must contain train.npz: "
                f"{self.directory}"
            )

        return self

    def require_training_splits(self) -> "DatasetBundlePaths":
        self.require_train()
        missing = [
            path.name
            for path in (self.validation, self.same_layout)
            if not path.exists()
        ]
        if missing:
            raise ValueError(
                "Dataset directory is missing required training split(s): "
                + ", ".join(missing)
            )

        return self

    def require_tuning_splits(self) -> "DatasetBundlePaths":
        self.require_train()
        missing = [
            path.name
            for path in (
                self.train_tune,
                self.validation_tune,
                self.same_layout,
            )
            if not path.exists()
        ]
        if missing:
            raise ValueError(
                "Dataset directory is missing required tuning split(s): "
                + ", ".join(missing)
            )

        return self


def dataset_bundle_dir_for_reference(
    dataset: Path,
    *,
    data_root: Path = DEFAULT_DATA_OUTPUT_ROOT,
) -> Path:
    """Expand a bare dataset name while preserving explicit paths."""

    if dataset.is_absolute() or len(dataset.parts) > 1:
        return dataset

    return data_root / dataset


def dataset_bundle_for_reference(
    dataset: Path,
    *,
    data_root: Path = DEFAULT_DATA_OUTPUT_ROOT,
) -> DatasetBundlePaths:
    """Resolve a dataset name/path to a validated bundle."""

    directory = dataset_bundle_dir_for_reference(
        dataset,
        data_root=data_root,
    )
    if not directory.is_dir():
        raise ValueError(
            "Dataset reference must name a dataset bundle directory: "
            f"{dataset}"
        )

    return DatasetBundlePaths(directory).require_train()
