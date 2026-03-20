from __future__ import annotations

from tasks.profiles.tagging.common import TaggingDatasetSplits
from tasks.profiles.tagging.common import load_tagging_splits


class TaggingPrepareAdapter:
    def load_splits(self, csv_path: str | None) -> TaggingDatasetSplits:
        return load_tagging_splits(csv_path)
