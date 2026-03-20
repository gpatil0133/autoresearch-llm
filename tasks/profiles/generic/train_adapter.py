from __future__ import annotations

from typing import Any, Mapping

from tasks.base import TaskContext


class GenericTrainAdapter:
    def train(self, context: TaskContext, selected_model: Mapping[str, Any]) -> dict[str, Any]:
        raise NotImplementedError("generic train adapter is not implemented yet")
