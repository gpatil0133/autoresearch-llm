from __future__ import annotations

from typing import Any, Mapping

from tasks.base import TaskContext


class GenericInferenceAdapter:
    def infer(self, context: TaskContext, train_summary: Mapping[str, Any]) -> dict[str, Any]:
        raise NotImplementedError("generic inference adapter is not implemented yet")
