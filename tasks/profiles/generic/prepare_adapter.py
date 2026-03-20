from __future__ import annotations

from tasks.profiles.generic.common import GenericDatasetSplits
from tasks.profiles.generic.common import load_generic_config
from tasks.profiles.generic.common import load_generic_splits


class GenericPrepareAdapter:
    def load_splits(self, csv_path: str | None) -> GenericDatasetSplits:
        config = load_generic_config(csv_path)
        return load_generic_splits(csv_path, config)
