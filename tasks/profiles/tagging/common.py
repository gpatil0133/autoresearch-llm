from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ALLOWED_SPLITS = {"train", "val", "test"}


@dataclass(frozen=True)
class TagEntity:
    start: int
    end: int
    entity_type: str

    @property
    def key_span(self) -> tuple[int, int]:
        return (self.start, self.end)

    @property
    def key_typed(self) -> tuple[int, int, str]:
        return (self.start, self.end, self.entity_type)


@dataclass(frozen=True)
class TaggingExample:
    example_id: str
    split: str
    raw_text: str
    labels: tuple[TagEntity, ...]


@dataclass(frozen=True)
class TaggingDatasetSplits:
    train: tuple[TaggingExample, ...]
    val: tuple[TaggingExample, ...]
    test: tuple[TaggingExample, ...]


def _stable_example_id(raw_text: str, split: str, index: int) -> str:
    digest = hashlib.md5(f"{split}:{index}:{raw_text}".encode("utf-8")).hexdigest()[:12]
    return f"tag-{split}-{digest}"


def _coerce_labels(value: Any, *, row_index: int) -> tuple[TagEntity, ...]:
    if value is None:
        return ()

    if isinstance(value, str):
        text = value.strip()
        if not text:
            return ()
        payload = json.loads(text)
    else:
        payload = value

    if not isinstance(payload, list):
        raise ValueError(f"labels must be a JSON array at row {row_index}")

    entities: list[TagEntity] = []
    for item in payload:
        if not isinstance(item, Mapping):
            raise ValueError(f"label entries must be objects at row {row_index}")
        start = int(item.get("start", item.get("start_char", -1)))
        end = int(item.get("end", item.get("end_char", -1)))
        entity_type = str(item.get("type", item.get("entity_type", ""))).strip()
        if start < 0 or end <= start or not entity_type:
            continue
        entities.append(TagEntity(start=start, end=end, entity_type=entity_type))

    entities.sort(key=lambda item: (item.start, item.end, item.entity_type))
    return tuple(entities)


def load_tagging_splits(csv_path: str | None) -> TaggingDatasetSplits:
    if not csv_path:
        raise ValueError("tagging profile requires --csv-path")

    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"tagging csv not found: {path}")

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError("tagging csv is missing header")

        rows = list(reader)

    train: list[TaggingExample] = []
    val: list[TaggingExample] = []
    test: list[TaggingExample] = []

    for index, row in enumerate(rows, start=1):
        split = str(row.get("split", "")).strip().lower()
        if split not in ALLOWED_SPLITS:
            raise ValueError(f"Unsupported split {split!r} at row {index}; expected train|val|test")

        raw_text = str(row.get("raw_text", "")).strip()
        if not raw_text:
            raise ValueError(f"Missing raw_text at row {index}")

        example_id = str(row.get("example_id", "")).strip() or _stable_example_id(raw_text, split, index)
        labels = _coerce_labels(row.get("labels"), row_index=index)
        example = TaggingExample(example_id=example_id, split=split, raw_text=raw_text, labels=labels)

        if split == "train":
            train.append(example)
        elif split == "val":
            val.append(example)
        else:
            test.append(example)

    return TaggingDatasetSplits(train=tuple(train), val=tuple(val), test=tuple(test))


def examples_for_split(splits: TaggingDatasetSplits, split: str) -> tuple[TaggingExample, ...]:
    normalized = split.strip().lower()
    if normalized == "train":
        return splits.train
    if normalized == "val":
        return splits.val
    if normalized == "test":
        return splits.test
    raise ValueError(f"Unsupported split: {split}")


def spans_to_token_labels(raw_text: str, labels: Sequence[TagEntity]) -> list[str]:
    tokens_with_spans: list[tuple[str, int, int]] = []
    cursor = 0
    for token in raw_text.split():
        start = raw_text.find(token, cursor)
        if start < 0:
            continue
        end = start + len(token)
        tokens_with_spans.append((token, start, end))
        cursor = end

    token_labels: list[str] = []
    for _, token_start, token_end in tokens_with_spans:
        label_value = "O"
        for entity in labels:
            overlaps = token_start < entity.end and token_end > entity.start
            if overlaps:
                label_value = entity.entity_type
                break
        token_labels.append(label_value)
    return token_labels


def f1_from_sets(predicted: set[Any], reference: set[Any]) -> float:
    tp = len(predicted & reference)
    fp = len(predicted - reference)
    fn = len(reference - predicted)
    denominator = (2 * tp) + fp + fn
    if denominator <= 0:
        return 1.0
    return (2 * tp) / float(denominator)


def token_f1(examples: Iterable[tuple[TaggingExample, Sequence[TagEntity]]]) -> float:
    true_positive = 0
    false_positive = 0
    false_negative = 0

    for example, predicted_labels in examples:
        gold_tokens = spans_to_token_labels(example.raw_text, example.labels)
        pred_tokens = spans_to_token_labels(example.raw_text, predicted_labels)

        max_len = max(len(gold_tokens), len(pred_tokens))
        for index in range(max_len):
            gold_label = gold_tokens[index] if index < len(gold_tokens) else "O"
            pred_label = pred_tokens[index] if index < len(pred_tokens) else "O"

            if pred_label == gold_label and gold_label != "O":
                true_positive += 1
            elif pred_label != gold_label:
                if pred_label != "O":
                    false_positive += 1
                if gold_label != "O":
                    false_negative += 1

    denominator = (2 * true_positive) + false_positive + false_negative
    if denominator <= 0:
        return 1.0
    return (2 * true_positive) / float(denominator)
