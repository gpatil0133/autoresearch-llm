from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from tasks.base import TaskContext
from tasks.profiles.generic.common import examples_for_split
from tasks.profiles.generic.common import load_generic_config
from tasks.profiles.generic.common import load_generic_splits
from tasks.profiles.generic.common import train_bow_multiclass


class GenericTrainAdapter:
    def train(self, context: TaskContext, selected_model: Mapping[str, Any]) -> dict[str, Any]:
        config = load_generic_config(context.csv_path)
        splits = load_generic_splits(context.csv_path, config)
        train_examples = examples_for_split(splits, "train")
        if not train_examples:
            raise ValueError("generic profile requires at least one train example")

        model_payload = train_bow_multiclass(
            train_examples,
            input_fields=config.input_fields,
            label_space=config.label_space,
        )
        model_payload["task_kind"] = config.task_kind

        artifact_dir = Path(context.artifact_dir)
        checkpoint_dir = artifact_dir / "checkpoint"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        model_path = checkpoint_dir / "generic_model.json"
        model_path.write_text(json.dumps(model_payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")

        return {
            "model_name": str(selected_model.get("model_name", "generic-bow-baseline")),
            "model_family": str(selected_model.get("model_family", "encoder")),
            "model_registry_id": str(selected_model.get("registry_id", "generic-baseline")),
            "selection_strategy": str(selected_model.get("selection_strategy", "round_robin")),
            "task_kind": config.task_kind,
            "train_examples": len(train_examples),
            "label_space_size": len(config.label_space),
            "checkpoint_path": str(checkpoint_dir),
            "config_json": json.dumps(
                {
                    "task_profile": context.profile_id,
                    "task_kind": config.task_kind,
                    "input_fields": list(config.input_fields),
                    "target_field": config.target_field,
                    "metrics": list(config.metrics),
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            "val_metric": 0.0,
            "json_schema_compliance": 1.0,
        }
