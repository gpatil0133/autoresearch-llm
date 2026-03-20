from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol


@dataclass(frozen=True)
class TaskContext:
    profile_id: str
    split: str
    csv_path: str | None
    run_tag: str
    experiment_id: str
    artifact_dir: Path


class PrepareAdapter(Protocol):
    def load_splits(self, csv_path: str | None) -> Any:
        ...


class TrainAdapter(Protocol):
    def train(self, context: TaskContext, selected_model: Mapping[str, Any]) -> dict[str, Any]:
        ...


class InferenceAdapter(Protocol):
    def infer(self, context: TaskContext, train_summary: Mapping[str, Any]) -> dict[str, Any]:
        ...


class EvalAdapter(Protocol):
    def evaluate(self, context: TaskContext, predictions_path: str) -> tuple[dict[str, float], dict[str, Any]]:
        ...


@dataclass(frozen=True)
class TaskProfile:
    profile_id: str
    description: str
    prepare: PrepareAdapter
    train: TrainAdapter
    inference: InferenceAdapter
    evaluator: EvalAdapter
    default_metric_key: str = "val_metric"
