from __future__ import annotations

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


class NLPInferenceAdapter:
    def infer(self, context: TaskContext, train_summary: Mapping[str, Any]) -> dict[str, Any]:
        model_family = str(train_summary.get("model_family", "")).strip()
        model_name = str(train_summary.get("model_name", "")).strip()
        checkpoint_path = str(train_summary.get("checkpoint_path", "")).strip()

        if not model_family:
            raise ValueError("nlp_analysis inference requires model_family in train_summary")
        if not model_name:
            raise ValueError("nlp_analysis inference requires model_name in train_summary")

        predictions_dir = ROOT_DIR / "predictions"
        predictions_path = predictions_dir / f"{context.experiment_id}_{model_family}_{context.split}_predictions.jsonl"

        command = [
            "uv",
            "run",
            "inference.py",
            "--model-family",
            model_family,
            "--model-name",
            model_name,
            "--split",
            context.split,
            "--run-tag",
            context.experiment_id,
            "--output-path",
            str(predictions_path),
        ]
        if context.csv_path:
            command.extend(["--csv-path", context.csv_path])
        if checkpoint_path:
            command.extend(["--checkpoint-path", checkpoint_path])

        result = _run_command(command)
        if result.returncode != 0:
            raise RuntimeError(
                f"nlp_analysis inference failed code={result.returncode}\n"
                f"stdout_tail={result.stdout[-1200:]}\n"
                f"stderr_tail={result.stderr[-1200:]}"
            )

        parsed = _parse_key_value_output(result.stdout)
        parsed.setdefault("predictions_path", str(predictions_path))
        return parsed
