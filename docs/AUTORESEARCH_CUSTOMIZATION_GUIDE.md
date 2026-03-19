# AutoResearch Customization Blueprint

Date: 2026-03-19

## Why this is needed

Today, the end-to-end loop is tightly coupled to one NLP survey schema (`text_analysis`) and one family of task assumptions:

- `prepare.py`: fixed data contract, normalization, and scoring for survey analysis
- `train.py`: fixed decoder/encoder heads and objective flow for that schema
- `inference.py`: fixed output assembly and schema validation for that schema
- `eval.py`: fixed alignment/scoring keys and `val_metric` composition
- `run_experiments.py`: orchestrates the above with task-specific CLI assumptions

This works well for current NLP analysis, but adding:

- sequence tagging with different labels,
- classification-only tasks,
- extraction tasks with different output shape,
- or non-NLP/tabular style objectives,

requires invasive edits across multiple files.

---

## Goal

Make AutoResearch task-customizable so the **same orchestration loop** can run many task types by swapping a task profile.

Target support:

1. NLP analysis (current behavior)
2. Tagging tasks
3. Generic tasks (classification/regression/extraction/custom)

Core principle:

- Keep `run_experiments.py` stable as orchestration backbone
- Move task-specific logic behind interfaces (adapters/plugins)
- Select behavior via `--task-profile` instead of editing core files

---

## Design summary (recommended architecture)

Create a `tasks/` plugin layer with one default profile that mirrors existing behavior.

## Proposed structure

```text
tasks/
  __init__.py
  registry.py
  base.py
  profiles/
    nlp_analysis/
      __init__.py
      profile.py
      prepare_adapter.py
      train_adapter.py
      inference_adapter.py
      eval_adapter.py
    tagging/
      __init__.py
      profile.py
      prepare_adapter.py
      train_adapter.py
      inference_adapter.py
      eval_adapter.py
    generic/
      __init__.py
      profile.py
      prepare_adapter.py
      train_adapter.py
      inference_adapter.py
      eval_adapter.py
```

### Contract split

- **Prepare Adapter**: parse/normalize raw data and provide split views
- **Train Adapter**: run task-specific training loop and return summary/checkpoint metadata
- **Inference Adapter**: produce predictions in profile-specific format
- **Eval Adapter**: align and score predictions, return metrics including `val_metric`
- **Task Profile**: bundles all adapters + metadata + CLI defaults

---

## Step-by-step migration plan

## Phase 1: Introduce interfaces (no behavior change)

Add interfaces and a profile registry. Keep current scripts using existing logic, but add optional task-profile dispatch.

Outcome:

- Default `nlp_analysis` profile returns exact current behavior
- Existing CLI commands still work unchanged

## Phase 2: Isolate current survey logic into `nlp_analysis` profile

Move/duplicate current contract pieces from:

- `prepare.py`
- `train.py`
- `inference.py`
- `eval.py`

into adapters for `nlp_analysis`.

Outcome:

- Current behavior preserved
- Core scripts become profile runners

## Phase 3: Add `tagging` profile

Implement token/sequence tagging contract and metrics (span-F1 / token-F1 / entity-F1 depending on task).

## Phase 4: Add `generic` profile

Support config-driven objective declaration:

- `task_kind`: `classification | regression | extraction | custom`
- per-task metric composition and prediction schema

## Phase 5: Harden operational UX

- add profile validation CLI
- add smoke tests per profile
- add reproducible profile config templates

---

## Patch/snippet set (implementation-ready)

These are targeted snippets you can apply incrementally.

> Note: snippets are shown as patches/templates for implementation guidance. They are intentionally minimal and do not modify current repo automatically.

## 1) New base interfaces (`tasks/base.py`)

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence


@dataclass(frozen=True)
class TaskContext:
    profile_id: str
    split: str
    csv_path: str | None
    run_tag: str
    experiment_id: str
    artifact_dir: Path


class PrepareAdapter(Protocol):
    def load_splits(self, csv_path: str | None) -> Any:
        ...


class TrainAdapter(Protocol):
    def train(self, context: TaskContext, selected_model: Mapping[str, Any]) -> dict[str, Any]:
        ...


class InferenceAdapter(Protocol):
    def infer(self, context: TaskContext, train_summary: Mapping[str, Any]) -> dict[str, Any]:
        ...


class EvalAdapter(Protocol):
    def evaluate(self, context: TaskContext, predictions_path: str) -> tuple[dict[str, float], dict[str, Any]]:
        ...


@dataclass(frozen=True)
class TaskProfile:
    profile_id: str
    description: str
    prepare: PrepareAdapter
    train: TrainAdapter
    inference: InferenceAdapter
    evaluator: EvalAdapter
    default_metric_key: str = "val_metric"
```

## 2) Profile registry (`tasks/registry.py`)

```python
from __future__ import annotations

from typing import Dict

from tasks.base import TaskProfile
from tasks.profiles.nlp_analysis.profile import build_profile as build_nlp_analysis_profile
from tasks.profiles.tagging.profile import build_profile as build_tagging_profile
from tasks.profiles.generic.profile import build_profile as build_generic_profile


def list_profiles() -> dict[str, TaskProfile]:
    profiles = [
        build_nlp_analysis_profile(),
        build_tagging_profile(),
        build_generic_profile(),
    ]
    return {item.profile_id: item for item in profiles}


def get_profile(profile_id: str) -> TaskProfile:
    registry = list_profiles()
    if profile_id not in registry:
        available = ", ".join(sorted(registry))
        raise ValueError(f"Unknown task profile: {profile_id}. Available: {available}")
    return registry[profile_id]
```

## 3) `run_experiments.py` minimal dispatch patch

```diff
*** Begin Patch
*** Update File: run_experiments.py
@@
-from model_selector import SelectionStrategy, select_model
+from model_selector import SelectionStrategy, select_model
+from tasks.base import TaskContext
+from tasks.registry import get_profile, list_profiles
@@
 class OrchestratorConfig:
@@
     leaderboard_path: str
+    task_profile: str
@@
 def _build_arg_parser() -> argparse.ArgumentParser:
@@
+    parser.add_argument("--task-profile", type=str, default="nlp_analysis")
@@
 def _single_experiment(
@@
-    if args.skip_train:
+    profile = get_profile(args.task_profile)
+    context = TaskContext(
+        profile_id=args.task_profile,
+        split=args.split,
+        csv_path=args.csv_path,
+        run_tag=args.run_tag,
+        experiment_id=run.experiment_id,
+        artifact_dir=artifact_dir,
+    )
+
+    if args.skip_train:
         train_summary = {
@@
-    else:
-        train_summary = _train_experiment(run, selected.config, args)
+    else:
+        train_summary = profile.train.train(context=context, selected_model={
+            "registry_id": run.registry_id,
+            "model_name": run.model_name,
+            "model_family": run.model_family,
+            "config": selected.config,
+        })
         run.model_name = str(train_summary.get("model_name", run.model_name))
         run.model_family = str(train_summary.get("model_family", run.model_family))
 
-    infer_summary = _infer_experiment(run, train_summary, args)
+    infer_summary = profile.inference.infer(context=context, train_summary=train_summary)
     run.predictions_path = str(infer_summary.get("predictions_path", run.predictions_path))
 
-    val_metric, json_compliance, status, metrics, leaderboard = _evaluate_and_log(run, train_summary, args)
+    metrics, alignment_stats = profile.evaluator.evaluate(
+        context=context,
+        predictions_path=run.predictions_path,
+    )
+    val_metric = float(metrics.get("val_metric", 0.0))
+    json_compliance = float(metrics.get("json_schema_compliance", 0.0))
+    status = "keep" if val_metric >= _read_best_metric(Path(args.leaderboard_path)) else "discard"
+    leaderboard = eval_module.append_result(
+        ExperimentRecord(
+            experiment_id=run.experiment_id,
+            model_registry_id=run.registry_id,
+            hf_model_name=run.model_name,
+            model_family=run.model_family,
+            val_metric=val_metric,
+            json_schema_compliance=json_compliance,
+            status=status,
+            run_tag=run.run_tag,
+            selection_strategy=run.strategy,
+            checkpoint_path=str(train_summary.get("checkpoint_path", "")),
+            predictions_path=run.predictions_path,
+            metrics=dict(metrics),
+            config={"task_profile": args.task_profile, "alignment_stats": alignment_stats},
+            notes=f"profile={args.task_profile}",
+        ),
+        results_path=Path(args.results_path),
+        leaderboard_path=Path(args.leaderboard_path),
+    )
*** End Patch
```

## 4) `train.py` profile-aware invocation patch

```diff
*** Begin Patch
*** Update File: train.py
@@
 import argparse
@@
+from tasks.registry import get_profile
+from tasks.base import TaskContext
@@
 def _build_arg_parser() -> argparse.ArgumentParser:
@@
+    parser.add_argument("--task-profile", type=str, default="nlp_analysis")
@@
 def main(argv: Sequence[str] | None = None) -> None:
     args = _build_arg_parser().parse_args(argv)
+
+    profile = get_profile(args.task_profile)
+    context = TaskContext(
+        profile_id=args.task_profile,
+        split="train",
+        csv_path=args.csv_path,
+        run_tag=args.run_tag,
+        experiment_id=f"{args.run_tag}_{args.experiment_index:04d}",
+        artifact_dir=Path(args.checkpoint_dir).parent if args.checkpoint_dir else Path("artifacts"),
+    )
+    summary = profile.train.train(
+        context=context,
+        selected_model={
+            "registry_id": runtime.model_registry_id,
+            "model_name": runtime.model_name,
+            "model_family": runtime.model_family,
+            "config": asdict(runtime),
+        },
+    )
+    # Existing print contract can serialize `summary`
*** End Patch
```

## 5) NLP analysis adapter (wrap current behavior)

`tasks/profiles/nlp_analysis/profile.py`

```python
from __future__ import annotations

from tasks.base import TaskProfile
from .prepare_adapter import NLPPrepareAdapter
from .train_adapter import NLPTrainAdapter
from .inference_adapter import NLPInferenceAdapter
from .eval_adapter import NLPEvalAdapter


def build_profile() -> TaskProfile:
    return TaskProfile(
        profile_id="nlp_analysis",
        description="Current survey text_analysis workflow",
        prepare=NLPPrepareAdapter(),
        train=NLPTrainAdapter(),
        inference=NLPInferenceAdapter(),
        evaluator=NLPEvalAdapter(),
    )
```

Adapter methods can internally call existing module functions initially (composition over rewrite), then be gradually decoupled.

## 6) Tagging profile skeleton

`tasks/profiles/tagging/profile.py`

```python
from __future__ import annotations

from tasks.base import TaskProfile
from .prepare_adapter import TaggingPrepareAdapter
from .train_adapter import TaggingTrainAdapter
from .inference_adapter import TaggingInferenceAdapter
from .eval_adapter import TaggingEvalAdapter


def build_profile() -> TaskProfile:
    return TaskProfile(
        profile_id="tagging",
        description="Token/sequence tagging tasks (BIO/BILOU or span labels)",
        prepare=TaggingPrepareAdapter(),
        train=TaggingTrainAdapter(),
        inference=TaggingInferenceAdapter(),
        evaluator=TaggingEvalAdapter(),
    )
```

Tagging dataset expectation example:

```json
{
  "example_id": "ex-001",
  "split": "train",
  "raw_text": "Battery life is amazing but camera is bad",
  "labels": [
    {"start": 0, "end": 12, "type": "ASPECT"},
    {"start": 16, "end": 23, "type": "SENTIMENT_POS"},
    {"start": 32, "end": 38, "type": "ASPECT"},
    {"start": 42, "end": 45, "type": "SENTIMENT_NEG"}
  ]
}
```

Metrics recommendation for tagging profile:

- `entity_span_f1`
- `entity_typed_f1`
- `token_f1` (optional)
- `json_schema_compliance` (if JSON prediction artifact is used)

Sample val metric composition:

```python
TAGGING_VAL_WEIGHTS = {
    "entity_typed_f1": 0.60,
    "entity_span_f1": 0.30,
    "json_schema_compliance": 0.10,
}
```

## 7) Generic profile schema (config-driven)

`tasks/profiles/generic/profile.py`

```python
from __future__ import annotations

from tasks.base import TaskProfile
from .prepare_adapter import GenericPrepareAdapter
from .train_adapter import GenericTrainAdapter
from .inference_adapter import GenericInferenceAdapter
from .eval_adapter import GenericEvalAdapter


def build_profile() -> TaskProfile:
    return TaskProfile(
        profile_id="generic",
        description="Config-driven generic objective",
        prepare=GenericPrepareAdapter(),
        train=GenericTrainAdapter(),
        inference=GenericInferenceAdapter(),
        evaluator=GenericEvalAdapter(),
    )
```

Add a profile config file template:

`tasks/profiles/generic/config.example.json`

```json
{
  "task_kind": "classification",
  "input_fields": ["text"],
  "target_field": "label",
  "label_space": ["A", "B", "C"],
  "prediction_format": "json",
  "metrics": ["macro_f1", "accuracy"],
  "val_metric": {
    "macro_f1": 0.8,
    "accuracy": 0.2
  }
}
```

---

## CLI standardization

Make all major entrypoints profile-aware:

- `train.py --task-profile ...`
- `inference.py --task-profile ...`
- `eval.py --task-profile ...`
- `run_experiments.py --task-profile ...`

Add a helper command (new or integrated):

```bash
uv run run_experiments.py --list-task-profiles
```

Expected output:

- `nlp_analysis`
- `tagging`
- `generic`

---

## Backward compatibility strategy

To avoid breaking current runs:

1. Default profile is `nlp_analysis`
2. Existing arguments remain valid
3. Existing `results.tsv` columns remain unchanged
4. Add profile metadata in `config_json` only

This guarantees current automation scripts still run.

---

## Validation checklist for each new profile

## Data contract

- Can load all splits reliably
- Handles missing/optional fields explicitly
- Produces deterministic `example_id`

## Training

- Emits checkpoint path
- Emits model + family metadata
- Handles time-budget logic or equivalent budget contract

## Inference

- Produces one prediction row per example (or explicit invalid marker)
- Includes `example_id`
- Includes schema validation flags/errors

## Evaluation

- Aligns predictions to references by `example_id`
- Computes profile-specific metrics
- Exposes a scalar `val_metric`

## Orchestration

- Keep/discard decision uses `val_metric`
- State file logs `task_profile`
- Results and leaderboard remain append-only

---

## Suggested rollout sequence in this repo

1. Add `tasks/base.py`, `tasks/registry.py`, `tasks/profiles/nlp_analysis/*`
2. Wire only `run_experiments.py` to profile dispatch (default `nlp_analysis`)
3. Verify parity with one controlled run (`--task-profile nlp_analysis`)
4. Add `tagging` profile (minimal viable metrics)
5. Add `generic` profile (classification first, then expand)
6. Add docs for profile authoring template

---

## Example operational commands after migration

NLP analysis (current behavior via profile):

```bash
uv run run_experiments.py --num-experiments 2 --task-profile nlp_analysis --run-tag phase7-nlp
```

Tagging profile:

```bash
uv run run_experiments.py --num-experiments 2 --task-profile tagging --csv-path data/tagging_dataset.csv --run-tag phase7-tag
```

Generic profile:

```bash
uv run run_experiments.py --num-experiments 2 --task-profile generic --csv-path data/generic_task.csv --run-tag phase7-generic
```

---

## Risk points and mitigations

1. **Metric incomparability across tasks**
   - Mitigation: track `task_profile` in `config_json`; compare leaderboard within-profile by default.

2. **Schema drift between profiles**
   - Mitigation: each profile owns validator + prediction contract; enforce with adapter tests.

3. **Overly coupled model-family assumptions**
   - Mitigation: profile train/inference adapters choose families and heads; orchestrator remains agnostic.

4. **Future code duplication across profiles**
   - Mitigation: extract shared helpers (`tasks/common/metrics.py`, `tasks/common/io.py`, `tasks/common/modeling.py`).

---

## Minimum acceptance criteria

You can consider customization complete when all of these are true:

- `run_experiments.py` runs unchanged core loop with `--task-profile`
- `nlp_analysis` profile reproduces current output behavior
- `tagging` profile can run at least one train-infer-eval cycle end-to-end
- `generic` profile supports at least one classification dataset end-to-end
- `results.tsv` / `leaderboard.json` remain append-only and backward-compatible

---

## Implementation note for this request

Per your instruction, no existing code was changed in this step. This document provides the exact architecture and patch/snippet guidance required to implement the customization in a controlled follow-up.
