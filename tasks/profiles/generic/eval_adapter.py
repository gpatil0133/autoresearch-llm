from __future__ import annotations

from typing import Any

from tasks.base import TaskContext


class GenericEvalAdapter:
    def evaluate(self, context: TaskContext, predictions_path: str) -> tuple[dict[str, float], dict[str, Any]]:
        raise NotImplementedError("generic eval adapter is not implemented yet")
