from __future__ import annotations

from pathlib import Path
from typing import Any

import eval as eval_module
from tasks.base import TaskContext


class NLPEvalAdapter:
    def evaluate(self, context: TaskContext, predictions_path: str) -> tuple[dict[str, float], dict[str, Any]]:
        metrics, alignment_stats = eval_module.evaluate_predictions(
            predictions_path=Path(predictions_path),
            split=context.split,
            csv_path=context.csv_path,
        )
        return metrics, alignment_stats
