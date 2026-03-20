from __future__ import annotations

from prepare import load_dataset_splits


class NLPPrepareAdapter:
    def load_splits(self, csv_path: str | None) -> object:
        return load_dataset_splits(csv_path)
