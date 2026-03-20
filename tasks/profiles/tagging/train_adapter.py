from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from tasks.base import TaskContext
from tasks.profiles.tagging.common import examples_for_split
from tasks.profiles.tagging.common import load_tagging_splits


class TaggingTrainAdapter:
    def train(self, context: TaskContext, selected_model: Mapping[str, Any]) -> dict[str, Any]:
        splits = load_tagging_splits(context.csv_path)
        train_examples = examples_for_split(splits, "train")
        if not train_examples:
            raise ValueError("tagging profile requires at least one train example")

        phrase_type_counts: dict[str, dict[str, int]] = {}
        label_counts: dict[str, int] = {}

        for example in train_examples:
            for label in example.labels:
                phrase = example.raw_text[label.start:label.end].strip().lower()
                if not phrase:
                    continue
                if phrase not in phrase_type_counts:
                    phrase_type_counts[phrase] = {}
                phrase_type_counts[phrase][label.entity_type] = phrase_type_counts[phrase].get(label.entity_type, 0) + 1
                label_counts[label.entity_type] = label_counts.get(label.entity_type, 0) + 1

        phrase_lexicon: dict[str, str] = {}
        for phrase, counts in phrase_type_counts.items():
            best_type, _ = max(counts.items(), key=lambda item: (item[1], item[0]))
            phrase_lexicon[phrase] = best_type

        artifact_dir = Path(context.artifact_dir)
        checkpoint_dir = artifact_dir / "checkpoint"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        model_path = checkpoint_dir / "tagging_model.json"

        model_payload = {
            "profile_id": context.profile_id,
            "experiment_id": context.experiment_id,
            "label_counts": dict(sorted(label_counts.items())),
            "phrase_lexicon": dict(sorted(phrase_lexicon.items(), key=lambda item: (-len(item[0]), item[0]))),
        }
        model_path.write_text(json.dumps(model_payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")

        total_labels = sum(label_counts.values())
        top_label = max(label_counts.items(), key=lambda item: item[1])[0] if label_counts else ""

        return {
            "model_name": str(selected_model.get("model_name", "tagging-lexical-baseline")),
            "model_family": str(selected_model.get("model_family", "encoder")),
            "model_registry_id": str(selected_model.get("registry_id", "tagging-baseline")),
            "selection_strategy": str(selected_model.get("selection_strategy", "round_robin")),
            "train_examples": len(train_examples),
            "train_label_count": total_labels,
            "label_space_size": len(label_counts),
            "most_common_label": top_label,
            "checkpoint_path": str(checkpoint_dir),
            "config_json": json.dumps(
                {
                    "task_profile": context.profile_id,
                    "train_examples": len(train_examples),
                    "label_space": sorted(label_counts),
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            "val_metric": 0.0,
            "json_schema_compliance": 1.0,
        }
