from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Mapping

from tasks.base import TaskContext


ROOT_DIR = Path(__file__).resolve().parents[3]


def _run_command(command: list[str], timeout_seconds: int = 0) -> subprocess.CompletedProcess[str]:
    timeout = None if timeout_seconds <= 0 else timeout_seconds
    return subprocess.run(
        command,
        cwd=str(ROOT_DIR),
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def _parse_key_value_output(stdout_text: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for line in stdout_text.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        parsed[key.strip()] = value.strip()
    return parsed


class NLPTrainAdapter:
    def train(self, context: TaskContext, selected_model: Mapping[str, Any]) -> dict[str, Any]:
        artifact_dir = Path(context.artifact_dir)
        summary_path = artifact_dir / "train_summary.json"
        checkpoint_dir = artifact_dir / "checkpoint"

        selected_config = dict(selected_model.get("config", {}))
        model_registry_id = str(selected_model.get("registry_id", "")).strip()
        selection_strategy = str(selected_model.get("selection_strategy", "round_robin")).strip() or "round_robin"

        command = [
            "uv",
            "run",
            "train.py",
            "--experiment-index",
            "0",
            "--selection-strategy",
            selection_strategy,
            "--run-tag",
            context.run_tag,
            "--summary-path",
            str(summary_path),
            "--checkpoint-dir",
            str(checkpoint_dir),
        ]

        if model_registry_id:
            command.extend(["--model-registry-id", model_registry_id])
        if context.csv_path:
            command.extend(["--csv-path", context.csv_path])
        if "learning_rate" in selected_config:
            command.extend(["--learning-rate", str(selected_config["learning_rate"])])
        if "batch_size" in selected_config:
            command.extend(["--train-batch-size", str(selected_config["batch_size"])])
        if "max_sequence_length" in selected_config:
            command.extend(["--max-seq-len", str(selected_config["max_sequence_length"])])
        if "use_lora" in selected_config:
            command.extend(["--use-lora", "true" if bool(selected_config["use_lora"]) else "false"])

        result = _run_command(command)
        if result.returncode != 0:
            raise RuntimeError(
                f"nlp_analysis train failed code={result.returncode}\n"
                f"stdout_tail={result.stdout[-1200:]}\n"
                f"stderr_tail={result.stderr[-1200:]}"
            )

        if summary_path.exists():
            return json.loads(summary_path.read_text(encoding="utf-8"))

        parsed = _parse_key_value_output(result.stdout)
        return {
            "model_name": parsed.get("model_name", selected_model.get("model_name", "")),
            "model_family": parsed.get("model_family", selected_model.get("model_family", "")),
            "model_registry_id": parsed.get("model_registry_id", model_registry_id),
            "val_metric": float(parsed.get("val_metric", 0.0)),
            "json_schema_compliance": float(parsed.get("json_schema_compliance", 0.0)),
            "config_json": parsed.get("config_json", json.dumps(selected_config, sort_keys=True)),
            "checkpoint_path": str(checkpoint_dir),
        }
