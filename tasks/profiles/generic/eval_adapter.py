from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from typing import Mapping

from tasks.base import TaskContext
from tasks.profiles.generic.common import accuracy_score
from tasks.profiles.generic.common import examples_for_split
from tasks.profiles.generic.common import load_generic_config
from tasks.profiles.generic.common import load_generic_splits
from tasks.profiles.generic.common import macro_f1_score
from tasks.profiles.generic.common import weighted_val_metric


def _load_predictions_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                payload = json.loads(text)
            except json.JSONDecodeError as exc:
                rows.append(
                    {
                        "example_id": "",
                        "prediction": {"label": ""},
                        "is_valid_json": False,
                        "validation_error": f"jsonl_parse_error@line_{line_number}:{exc}",
                    }
                )
                continue
            if isinstance(payload, dict):
                rows.append(payload)
            else:
                rows.append(
                    {
                        "example_id": "",
                        "prediction": {"label": ""},
                        "is_valid_json": False,
                        "validation_error": f"jsonl_row_not_object@line_{line_number}",
                    }
                )
    return rows


class GenericEvalAdapter:
    def evaluate(self, context: TaskContext, predictions_path: str) -> tuple[dict[str, float], dict[str, Any]]:
        config = load_generic_config(context.csv_path)
        splits = load_generic_splits(context.csv_path, config)
        references = examples_for_split(splits, context.split)
        if not references:
            raise ValueError(f"No reference examples found for split={context.split}")

        rows = _load_predictions_jsonl(Path(predictions_path))
        by_example_id: dict[str, Mapping[str, Any]] = {}
        duplicate_prediction_ids = 0
        unkeyed_prediction_rows = 0

        for row in rows:
            example_id = str(row.get("example_id", "")).strip()
            if not example_id:
                unkeyed_prediction_rows += 1
                continue
            if example_id in by_example_id:
                duplicate_prediction_ids += 1
            by_example_id[example_id] = row

        matched_examples = 0
        missing_predictions = 0
        invalid_predictions = 0

        predicted_labels: list[str] = []
        reference_labels: list[str] = []
        allowed_labels = set(config.label_space)

        for example in references:
            row = by_example_id.get(example.example_id)
            if row is None:
                missing_predictions += 1
                invalid_predictions += 1
                predicted_labels.append("")
                reference_labels.append(example.label)
                continue

            matched_examples += 1
            prediction_payload = row.get("prediction", row)
            if not isinstance(prediction_payload, Mapping):
                invalid_predictions += 1
                predicted_labels.append("")
                reference_labels.append(example.label)
                continue

            label = str(prediction_payload.get("label", "")).strip()
            if (not bool(row.get("is_valid_json", True))) or (label not in allowed_labels):
                invalid_predictions += 1
                predicted_labels.append("")
                reference_labels.append(example.label)
                continue

            predicted_labels.append(label)
            reference_labels.append(example.label)

        accuracy = accuracy_score(predicted_labels, reference_labels)
        macro_f1 = macro_f1_score(predicted_labels, reference_labels, config.label_space)

        total_references = len(references)
        json_schema_compliance = (total_references - invalid_predictions) / float(total_references)
        json_schema_compliance = float(max(0.0, min(1.0, json_schema_compliance)))

        metrics = {
            "accuracy": float(accuracy),
            "macro_f1": float(macro_f1),
            "json_schema_compliance": float(json_schema_compliance),
        }
        val_metric = weighted_val_metric(metrics, config.val_metric_weights)
        metrics["val_metric"] = float(val_metric)

        alignment = {
            "prediction_rows": len(rows),
            "matched_examples": matched_examples,
            "missing_predictions": missing_predictions,
            "invalid_predictions": invalid_predictions,
            "duplicate_prediction_ids": duplicate_prediction_ids,
            "unkeyed_prediction_rows": unkeyed_prediction_rows,
        }

        return metrics, alignment
