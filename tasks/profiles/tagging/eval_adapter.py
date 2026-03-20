from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from typing import Mapping

from tasks.base import TaskContext
from tasks.profiles.tagging.common import TagEntity
from tasks.profiles.tagging.common import examples_for_split
from tasks.profiles.tagging.common import f1_from_sets
from tasks.profiles.tagging.common import load_tagging_splits
from tasks.profiles.tagging.common import token_f1


TAGGING_VAL_WEIGHTS = {
    "entity_typed_f1": 0.60,
    "entity_span_f1": 0.30,
    "json_schema_compliance": 0.10,
}


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
                        "prediction": {"labels": []},
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
                        "prediction": {"labels": []},
                        "is_valid_json": False,
                        "validation_error": f"jsonl_row_not_object@line_{line_number}",
                    }
                )
    return rows


def _coerce_predicted_labels(payload: Mapping[str, Any]) -> tuple[list[TagEntity], bool]:
    labels_payload = payload.get("labels", [])
    if not isinstance(labels_payload, list):
        return [], False

    labels: list[TagEntity] = []
    for item in labels_payload:
        if not isinstance(item, Mapping):
            return [], False
        start = item.get("start")
        end = item.get("end")
        entity_type = str(item.get("type", item.get("entity_type", ""))).strip()
        try:
            start_value = int(start)
            end_value = int(end)
        except (TypeError, ValueError):
            return [], False

        if start_value < 0 or end_value <= start_value or not entity_type:
            return [], False
        labels.append(TagEntity(start=start_value, end=end_value, entity_type=entity_type))

    labels.sort(key=lambda item: (item.start, item.end, item.entity_type))
    return labels, True


class TaggingEvalAdapter:
    def evaluate(self, context: TaskContext, predictions_path: str) -> tuple[dict[str, float], dict[str, Any]]:
        splits = load_tagging_splits(context.csv_path)
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

        span_predicted: set[tuple[int, int, str]] = set()
        span_reference: set[tuple[int, int, str]] = set()
        typed_predicted: set[tuple[int, int, str, str]] = set()
        typed_reference: set[tuple[int, int, str, str]] = set()
        token_inputs: list[tuple[Any, list[TagEntity]]] = []

        for example in references:
            row = by_example_id.get(example.example_id)
            if row is None:
                missing_predictions += 1
                invalid_predictions += 1
                token_inputs.append((example, []))
                span_reference.update((example.example_id, item.start, item.end) for item in example.labels)
                typed_reference.update(
                    (example.example_id, item.start, item.end, item.entity_type) for item in example.labels
                )
                continue

            matched_examples += 1
            prediction_payload = row.get("prediction", row)
            if not isinstance(prediction_payload, Mapping):
                invalid_predictions += 1
                token_inputs.append((example, []))
                span_reference.update((example.example_id, item.start, item.end) for item in example.labels)
                typed_reference.update(
                    (example.example_id, item.start, item.end, item.entity_type) for item in example.labels
                )
                continue

            predicted_labels, schema_ok = _coerce_predicted_labels(prediction_payload)
            if (not bool(row.get("is_valid_json", True))) or (not schema_ok):
                invalid_predictions += 1
                predicted_labels = []

            token_inputs.append((example, predicted_labels))

            span_predicted.update((example.example_id, item.start, item.end) for item in predicted_labels)
            span_reference.update((example.example_id, item.start, item.end) for item in example.labels)

            typed_predicted.update(
                (example.example_id, item.start, item.end, item.entity_type) for item in predicted_labels
            )
            typed_reference.update(
                (example.example_id, item.start, item.end, item.entity_type) for item in example.labels
            )

        entity_span_f1 = f1_from_sets(span_predicted, span_reference)
        entity_typed_f1 = f1_from_sets(typed_predicted, typed_reference)
        token_level_f1 = token_f1(token_inputs)

        total_references = len(references)
        json_schema_compliance = (total_references - invalid_predictions) / float(total_references)
        json_schema_compliance = float(max(0.0, min(1.0, json_schema_compliance)))

        val_metric = (
            TAGGING_VAL_WEIGHTS["entity_typed_f1"] * entity_typed_f1
            + TAGGING_VAL_WEIGHTS["entity_span_f1"] * entity_span_f1
            + TAGGING_VAL_WEIGHTS["json_schema_compliance"] * json_schema_compliance
        )

        metrics = {
            "entity_span_f1": float(entity_span_f1),
            "entity_typed_f1": float(entity_typed_f1),
            "token_f1": float(token_level_f1),
            "json_schema_compliance": float(json_schema_compliance),
            "val_metric": float(val_metric),
        }
        alignment = {
            "prediction_rows": len(rows),
            "matched_examples": matched_examples,
            "missing_predictions": missing_predictions,
            "invalid_predictions": invalid_predictions,
            "duplicate_prediction_ids": duplicate_prediction_ids,
            "unkeyed_prediction_rows": unkeyed_prediction_rows,
        }
        return metrics, alignment
