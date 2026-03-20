from __future__ import annotations

from typing import Any, Mapping

from tasks.base import TaskContext


class TaggingInferenceAdapter:
    def infer(self, context: TaskContext, train_summary: Mapping[str, Any]) -> dict[str, Any]:
        raise NotImplementedError("tagging inference adapter is not implemented yet")
