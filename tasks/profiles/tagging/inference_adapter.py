from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from tasks.base import TaskContext
from tasks.profiles.tagging.common import TagEntity
from tasks.profiles.tagging.common import examples_for_split
from tasks.profiles.tagging.common import load_tagging_splits


ROOT_DIR = Path(__file__).resolve().parents[3]


def _find_spans(raw_text: str, phrase: str) -> list[tuple[int, int]]:
    lower_text = raw_text.lower()
    lower_phrase = phrase.lower()
    if not lower_phrase:
        return []

    spans: list[tuple[int, int]] = []
    cursor = 0
    while True:
        start = lower_text.find(lower_phrase, cursor)
        if start < 0:
            break
        end = start + len(lower_phrase)
        spans.append((start, end))
        cursor = end
    return spans


def _load_model_payload(train_summary: Mapping[str, Any]) -> Mapping[str, Any]:
    checkpoint_path = str(train_summary.get("checkpoint_path", "")).strip()
    if not checkpoint_path:
        return {"phrase_lexicon": {}}

    model_path = Path(checkpoint_path) / "tagging_model.json"
    if not model_path.exists():
        return {"phrase_lexicon": {}}

    payload = json.loads(model_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        return {"phrase_lexicon": {}}
    return payload


def _prediction_record(example_id: str, labels: list[TagEntity]) -> dict[str, Any]:
    return {
        "example_id": example_id,
        "prediction": {
            "labels": [
                {
                    "start": label.start,
                    "end": label.end,
                    "type": label.entity_type,
                }
                for label in labels
            ]
        },
        "is_valid_json": True,
        "validation_error": "",
    }


class TaggingInferenceAdapter:
    def infer(self, context: TaskContext, train_summary: Mapping[str, Any]) -> dict[str, Any]:
        splits = load_tagging_splits(context.csv_path)
        examples = examples_for_split(splits, context.split)
        if not examples:
            raise ValueError(f"No examples found for split={context.split}")

        model_payload = _load_model_payload(train_summary)
        phrase_lexicon_raw = model_payload.get("phrase_lexicon", {})
        phrase_lexicon = (
            dict(phrase_lexicon_raw)
            if isinstance(phrase_lexicon_raw, Mapping)
            else {}
        )

        sorted_phrases = sorted(phrase_lexicon.items(), key=lambda item: (-len(item[0]), item[0]))
        prediction_rows: list[dict[str, Any]] = []

        for example in examples:
            labels: list[TagEntity] = []
            occupied: list[tuple[int, int]] = []

            for phrase, entity_type_raw in sorted_phrases:
                entity_type = str(entity_type_raw).strip()
                if not phrase or not entity_type:
                    continue

                for start, end in _find_spans(example.raw_text, phrase):
                    if any(start < used_end and end > used_start for used_start, used_end in occupied):
                        continue
                    occupied.append((start, end))
                    labels.append(TagEntity(start=start, end=end, entity_type=entity_type))

            labels.sort(key=lambda item: (item.start, item.end, item.entity_type))
            prediction_rows.append(_prediction_record(example.example_id, labels))

        predictions_dir = ROOT_DIR / "predictions"
        predictions_dir.mkdir(parents=True, exist_ok=True)
        predictions_path = predictions_dir / f"{context.experiment_id}_tagging_{context.split}_predictions.jsonl"

        with predictions_path.open("w", encoding="utf-8") as handle:
            for row in prediction_rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
                handle.write("\n")

        valid = sum(1 for row in prediction_rows if bool(row.get("is_valid_json")))
        total = len(prediction_rows)

        return {
            "predictions_path": str(predictions_path),
            "num_examples": total,
            "valid_predictions": valid,
            "invalid_predictions": total - valid,
            "json_schema_compliance": (valid / total) if total else 0.0,
        }
