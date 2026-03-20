from __future__ import annotations


class GenericPrepareAdapter:
    def load_splits(self, csv_path: str | None) -> object:
        raise NotImplementedError("generic prepare adapter is not implemented yet")
