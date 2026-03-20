from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from tasks.base import TaskContext
from tasks.profiles.generic.common import build_text
from tasks.profiles.generic.common import examples_for_split
from tasks.profiles.generic.common import load_generic_config
from tasks.profiles.generic.common import load_generic_splits
from tasks.profiles.generic.common import predict_label


ROOT_DIR = Path(__file__).resolve().parents[3]


class GenericInferenceAdapter:
    def infer(self, context: TaskContext, train_summary: Mapping[str, Any]) -> dict[str, Any]:
        config = load_generic_config(context.csv_path)
        splits = load_generic_splits(context.csv_path, config)
        examples = examples_for_split(splits, context.split)
        if not examples:
            raise ValueError(f"No examples found for split={context.split}")

        checkpoint_path = str(train_summary.get("checkpoint_path", "")).strip()
        if not checkpoint_path:
            raise ValueError("generic inference requires checkpoint_path in train_summary")

        model_path = Path(checkpoint_path) / "generic_model.json"
        if not model_path.exists():
            raise FileNotFoundError(f"generic model artifact not found: {model_path}")

        model_payload = json.loads(model_path.read_text(encoding="utf-8"))
        if not isinstance(model_payload, Mapping):
            raise ValueError("generic model artifact is invalid (expected JSON object)")

        rows: list[dict[str, Any]] = []
        for example in examples:
            text = build_text(example, config.input_fields)
            predicted_label = predict_label(model_payload, text)
            is_valid = predicted_label in set(config.label_space)

            rows.append(
                {
                    "example_id": example.example_id,
                    "prediction": {
                        "label": predicted_label,
                    },
                    "is_valid_json": bool(is_valid),
                    "validation_error": "" if is_valid else "predicted_label_not_in_label_space",
                }
            )

        predictions_dir = ROOT_DIR / "predictions"
        predictions_dir.mkdir(parents=True, exist_ok=True)
        predictions_path = predictions_dir / f"{context.experiment_id}_generic_{context.split}_predictions.jsonl"

        with predictions_path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
                handle.write("\n")

        valid = sum(1 for row in rows if bool(row.get("is_valid_json")))
        total = len(rows)
        return {
            "predictions_path": str(predictions_path),
            "num_examples": total,
            "valid_predictions": valid,
            "invalid_predictions": total - valid,
            "json_schema_compliance": (valid / total) if total else 0.0,
        }
