from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ALLOWED_SPLITS = {"train", "val", "test"}


@dataclass(frozen=True)
class GenericConfig:
    task_kind: str
    input_fields: tuple[str, ...]
    target_field: str
    label_space: tuple[str, ...]
    prediction_format: str
    metrics: tuple[str, ...]
    val_metric_weights: dict[str, float]


@dataclass(frozen=True)
class GenericExample:
    example_id: str
    split: str
    features: dict[str, str]
    label: str


@dataclass(frozen=True)
class GenericDatasetSplits:
    train: tuple[GenericExample, ...]
    val: tuple[GenericExample, ...]
    test: tuple[GenericExample, ...]


def _stable_example_id(split: str, row_index: int, signature: str) -> str:
    digest = hashlib.md5(f"{split}:{row_index}:{signature}".encode("utf-8")).hexdigest()[:12]
    return f"gen-{split}-{digest}"


def _tokenize_text(value: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9_']+", value.lower())


def resolve_config_path(csv_path: str | None, config_path: str | None = None) -> Path:
    if config_path:
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"generic config path not found: {path}")
        return path

    if csv_path:
        csv = Path(csv_path)
        sibling = csv.with_suffix(".config.json")
        if sibling.exists():
            return sibling

    default_path = Path(__file__).resolve().parent / "config.example.json"
    if default_path.exists():
        return default_path

    raise FileNotFoundError(
        "generic config file not found. Provide a sibling '<dataset>.config.json' or pass config_path explicitly."
    )


def load_generic_config(csv_path: str | None, config_path: str | None = None) -> GenericConfig:
    path = resolve_config_path(csv_path, config_path=config_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"generic config must be a JSON object: {path}")

    task_kind = str(payload.get("task_kind", "")).strip().lower()
    if task_kind != "classification":
        raise ValueError(
            f"Unsupported generic task_kind={task_kind!r}. Phase-6 supports only 'classification'."
        )

    input_fields_payload = payload.get("input_fields")
    if not isinstance(input_fields_payload, list) or not input_fields_payload:
        raise ValueError("generic config requires non-empty 'input_fields' list")
    input_fields = tuple(str(item).strip() for item in input_fields_payload if str(item).strip())
    if not input_fields:
        raise ValueError("generic config 'input_fields' list cannot be empty after normalization")

    target_field = str(payload.get("target_field", "")).strip()
    if not target_field:
        raise ValueError("generic config requires 'target_field'")

    label_space_payload = payload.get("label_space")
    if not isinstance(label_space_payload, list) or len(label_space_payload) < 2:
        raise ValueError("generic config requires 'label_space' with at least two labels")
    label_space = tuple(str(item).strip() for item in label_space_payload if str(item).strip())
    if len(label_space) < 2:
        raise ValueError("generic config 'label_space' must contain at least two non-empty labels")

    prediction_format = str(payload.get("prediction_format", "json")).strip().lower() or "json"
    if prediction_format != "json":
        raise ValueError("generic config currently supports only prediction_format='json'")

    metrics_payload = payload.get("metrics", ["macro_f1", "accuracy"])
    if not isinstance(metrics_payload, list) or not metrics_payload:
        raise ValueError("generic config requires non-empty 'metrics' list")
    metrics = tuple(str(item).strip() for item in metrics_payload if str(item).strip())
    unsupported = sorted(set(metrics) - {"macro_f1", "accuracy", "json_schema_compliance"})
    if unsupported:
        raise ValueError(f"Unsupported generic metrics: {unsupported}. Allowed: macro_f1, accuracy, json_schema_compliance")

    val_metric_payload = payload.get("val_metric")
    if not isinstance(val_metric_payload, Mapping) or not val_metric_payload:
        raise ValueError("generic config requires non-empty 'val_metric' weight mapping")

    val_metric_weights: dict[str, float] = {}
    for metric_name, weight_value in val_metric_payload.items():
        metric_key = str(metric_name).strip()
        if metric_key not in metrics:
            raise ValueError(f"val_metric includes {metric_key!r} which is not declared in 'metrics'")
        try:
            weight = float(weight_value)
        except (TypeError, ValueError):
            raise ValueError(f"val_metric weight for {metric_key!r} must be numeric") from None
        if weight < 0.0:
            raise ValueError(f"val_metric weight for {metric_key!r} must be non-negative")
        val_metric_weights[metric_key] = weight

    if not val_metric_weights:
        raise ValueError("val_metric mapping cannot be empty")

    total_weight = sum(val_metric_weights.values())
    if total_weight <= 0.0:
        raise ValueError("sum(val_metric weights) must be > 0")

    normalized_weights = {
        key: (value / total_weight)
        for key, value in val_metric_weights.items()
    }

    return GenericConfig(
        task_kind=task_kind,
        input_fields=input_fields,
        target_field=target_field,
        label_space=label_space,
        prediction_format=prediction_format,
        metrics=metrics,
        val_metric_weights=normalized_weights,
    )


def load_generic_splits(csv_path: str | None, config: GenericConfig) -> GenericDatasetSplits:
    if not csv_path:
        raise ValueError("generic profile requires --csv-path")

    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"generic csv not found: {path}")

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError("generic csv is missing header")

        field_names = set(reader.fieldnames)
        if "split" not in field_names:
            raise ValueError("generic csv requires 'split' column")
        if config.target_field not in field_names:
            raise ValueError(f"generic csv missing target field {config.target_field!r}")
        missing_inputs = [field for field in config.input_fields if field not in field_names]
        if missing_inputs:
            raise ValueError(f"generic csv missing configured input_fields: {missing_inputs}")

        rows = list(reader)

    train: list[GenericExample] = []
    val: list[GenericExample] = []
    test: list[GenericExample] = []
    label_space = set(config.label_space)

    for row_index, row in enumerate(rows, start=1):
        split = str(row.get("split", "")).strip().lower()
        if split not in ALLOWED_SPLITS:
            raise ValueError(f"Unsupported split {split!r} at row {row_index}; expected train|val|test")

        label = str(row.get(config.target_field, "")).strip()
        if label not in label_space:
            raise ValueError(
                f"Row {row_index} has label {label!r} not in configured label_space {sorted(label_space)}"
            )

        features = {field: str(row.get(field, "")).strip() for field in config.input_fields}
        if not any(features.values()):
            raise ValueError(f"Row {row_index} has empty values for all configured input_fields")

        signature = "|".join(features.values()) + f"|{label}"
        example_id = str(row.get("example_id", "")).strip() or _stable_example_id(split, row_index, signature)
        example = GenericExample(example_id=example_id, split=split, features=features, label=label)

        if split == "train":
            train.append(example)
        elif split == "val":
            val.append(example)
        else:
            test.append(example)

    return GenericDatasetSplits(train=tuple(train), val=tuple(val), test=tuple(test))


def examples_for_split(splits: GenericDatasetSplits, split: str) -> tuple[GenericExample, ...]:
    normalized = split.strip().lower()
    if normalized == "train":
        return splits.train
    if normalized == "val":
        return splits.val
    if normalized == "test":
        return splits.test
    raise ValueError(f"Unsupported split: {split}")


def build_text(example: GenericExample, input_fields: Sequence[str]) -> str:
    return " ".join(example.features.get(field, "") for field in input_fields).strip()


def train_bow_multiclass(
    examples: Sequence[GenericExample],
    *,
    input_fields: Sequence[str],
    label_space: Sequence[str],
) -> dict[str, Any]:
    label_doc_counts = {label: 0 for label in label_space}
    token_counts: dict[str, dict[str, int]] = {label: {} for label in label_space}
    total_tokens = {label: 0 for label in label_space}
    vocabulary: set[str] = set()

    for example in examples:
        label_doc_counts[example.label] = label_doc_counts.get(example.label, 0) + 1
        tokens = _tokenize_text(build_text(example, input_fields))
        for token in tokens:
            vocabulary.add(token)
            per_label = token_counts.setdefault(example.label, {})
            per_label[token] = per_label.get(token, 0) + 1
            total_tokens[example.label] = total_tokens.get(example.label, 0) + 1

    total_docs = max(1, len(examples))
    priors = {
        label: label_doc_counts.get(label, 0) / float(total_docs)
        for label in label_space
    }

    return {
        "label_space": list(label_space),
        "input_fields": list(input_fields),
        "priors": priors,
        "token_counts": token_counts,
        "total_tokens": total_tokens,
        "vocabulary": sorted(vocabulary),
    }


def predict_label(model_payload: Mapping[str, Any], text: str) -> str:
    label_space = [str(item) for item in model_payload.get("label_space", [])]
    if not label_space:
        return ""

    priors = model_payload.get("priors", {})
    token_counts = model_payload.get("token_counts", {})
    total_tokens = model_payload.get("total_tokens", {})
    vocabulary_size = max(1, len(model_payload.get("vocabulary", [])))

    tokens = _tokenize_text(text)
    if not tokens:
        return max(label_space, key=lambda label: float(priors.get(label, 0.0)))

    best_label = label_space[0]
    best_score = -1e18

    for label in label_space:
        prior = float(priors.get(label, 0.0))
        score = math.log(max(prior, 1e-12))
        per_label_counts = token_counts.get(label, {})
        denom = float(total_tokens.get(label, 0)) + vocabulary_size
        for token in tokens:
            count = float(per_label_counts.get(token, 0))
            score += math.log((count + 1.0) / denom)

        if score > best_score:
            best_score = score
            best_label = label

    return best_label


def accuracy_score(predictions: Sequence[str], references: Sequence[str]) -> float:
    if len(predictions) != len(references):
        raise ValueError("accuracy_score expects equal-length prediction and reference arrays")
    if not references:
        return 0.0
    correct = sum(1 for predicted, reference in zip(predictions, references) if predicted == reference)
    return correct / float(len(references))


def macro_f1_score(predictions: Sequence[str], references: Sequence[str], label_space: Sequence[str]) -> float:
    if len(predictions) != len(references):
        raise ValueError("macro_f1_score expects equal-length prediction and reference arrays")
    if not label_space:
        return 0.0

    per_label_f1: list[float] = []
    for label in label_space:
        tp = 0
        fp = 0
        fn = 0
        for predicted, reference in zip(predictions, references):
            if predicted == label and reference == label:
                tp += 1
            elif predicted == label and reference != label:
                fp += 1
            elif predicted != label and reference == label:
                fn += 1

        denom = (2 * tp) + fp + fn
        if denom <= 0:
            per_label_f1.append(0.0)
        else:
            per_label_f1.append((2 * tp) / float(denom))

    return sum(per_label_f1) / float(len(per_label_f1))


def weighted_val_metric(metrics: Mapping[str, float], weights: Mapping[str, float]) -> float:
    value = 0.0
    for key, weight in weights.items():
        value += float(weight) * float(metrics.get(key, 0.0))
    return float(value)
