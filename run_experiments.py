"""
Phase-6 autonomous experiment orchestration.

Runs non-destructive experiment cycles:
select model/config -> train -> inference -> eval/log -> keep/discard decision.

Decisions are recorded in an append-only state file. No git reset/rewrite behavior.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import eval as eval_module
from eval import ExperimentRecord
from model_selector import SelectionStrategy, select_model
from tasks.common.runtime import build_task_context
from tasks.common.runtime import list_task_profile_ids
from tasks.common.runtime import resolve_task_profile
from tasks.common.validation import format_validation_report
from tasks.common.validation import resolve_validation_csv_path
from tasks.common.validation import validate_task_profile


ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_STATE_PATH = ROOT_DIR / "artifacts" / "experiment_states.jsonl"
DEFAULT_RESULTS_PATH = ROOT_DIR / "results.tsv"
DEFAULT_LEADERBOARD_PATH = ROOT_DIR / "leaderboard.json"


@dataclass(frozen=True)
class OrchestratorConfig:
    num_experiments: int
    start_index: int
    selection_strategy: str
    random_seed: int
    split: str
    csv_path: str | None
    run_tag: str
    sleep_seconds: float
    decoder_json_decoding: str
    max_new_tokens: int
    batch_size: int
    metadata_token_confidence: float
    metadata_min_chars: int
    max_metadata_entities: int
    max_metadata_total_chars: int
    dry_run: bool
    skip_train: bool
    train_timeout_seconds: int
    inference_timeout_seconds: int
    state_path: str
    results_path: str
    leaderboard_path: str
    task_profile: str


@dataclass
class ExperimentRun:
    experiment_id: str
    experiment_index: int
    registry_id: str
    model_name: str
    model_family: str
    strategy: str
    run_tag: str
    train_summary_path: str
    checkpoint_dir: str
    predictions_path: str
    status: str = "completed"
    val_metric: float = 0.0
    json_schema_compliance: float = 0.0
    notes: str = ""


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _stable_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _parse_key_value_output(stdout_text: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for line in stdout_text.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        parsed[key.strip()] = value.strip()
    return parsed


def _run_command(command: Sequence[str], timeout_seconds: int) -> subprocess.CompletedProcess[str]:
    timeout = None if timeout_seconds <= 0 else timeout_seconds
    return subprocess.run(
        list(command),
        cwd=str(ROOT_DIR),
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def _read_best_metric(leaderboard_path: Path) -> float:
    if not leaderboard_path.exists():
        return 0.0
    payload = json.loads(leaderboard_path.read_text(encoding="utf-8"))
    leaders = payload.get("leaders") if isinstance(payload, dict) else None
    if not isinstance(leaders, list) or not leaders:
        return 0.0
    best = leaders[0]
    try:
        return float(best.get("val_metric", 0.0))
    except (TypeError, ValueError):
        return 0.0


def _record_state(state_path: Path, payload: Mapping[str, Any]) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    with state_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(payload), ensure_ascii=False, sort_keys=True))
        handle.write("\n")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run autonomous train->inference->eval experiment cycles")
    parser.add_argument("--list-task-profiles", action="store_true")
    parser.add_argument("--task-profile", type=str, default="nlp_analysis")
    parser.add_argument("--validate-task-profile", action="store_true")
    parser.add_argument("--validate-all-task-profiles", action="store_true")
    parser.add_argument("--skip-preflight-checks", action="store_true")
    parser.add_argument("--num-experiments", type=int, default=1)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument(
        "--selection-strategy",
        choices=[item.value for item in SelectionStrategy],
        default=SelectionStrategy.ROUND_ROBIN.value,
    )
    parser.add_argument("--random-seed", type=int, default=17)
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--csv-path", type=str, default=None)
    parser.add_argument("--run-tag", type=str, default="phase6")
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--decoder-json-decoding", choices=["constrained", "legacy"], default="constrained")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--metadata-token-confidence", type=float, default=0.80)
    parser.add_argument("--metadata-min-chars", type=int, default=3)
    parser.add_argument("--max-metadata-entities", type=int, default=10)
    parser.add_argument("--max-metadata-total-chars", type=int, default=256)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--train-timeout-seconds", type=int, default=0)
    parser.add_argument("--inference-timeout-seconds", type=int, default=0)
    parser.add_argument("--state-path", type=Path, default=DEFAULT_STATE_PATH)
    parser.add_argument("--results-path", type=Path, default=DEFAULT_RESULTS_PATH)
    parser.add_argument("--leaderboard-path", type=Path, default=DEFAULT_LEADERBOARD_PATH)
    return parser


def _make_experiment_id(run_tag: str, experiment_index: int, registry_id: str) -> str:
    return f"{run_tag}_{experiment_index:04d}_{registry_id}_{_utc_timestamp()}"


def _run_profile_validation(profile_id: str, split: str, csv_path: str | None) -> bool:
    profile = resolve_task_profile(profile_id)
    resolved_csv_path = resolve_validation_csv_path(
        profile_id=profile_id,
        csv_path=csv_path,
        root_dir=ROOT_DIR,
    )
    report = validate_task_profile(
        profile=profile,
        split=split,
        csv_path=resolved_csv_path,
    )
    print("---")
    print(format_validation_report(report))
    return bool(report.ok)


def _train_experiment(
    run: ExperimentRun,
    selected_config: Mapping[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    summary_path = Path(run.train_summary_path)
    checkpoint_dir = Path(run.checkpoint_dir)

    command = [
        "uv",
        "run",
        "train.py",
        "--experiment-index",
        str(run.experiment_index),
        "--selection-strategy",
        run.strategy,
        "--model-registry-id",
        run.registry_id,
        "--run-tag",
        run.run_tag,
        "--random-seed",
        str(args.random_seed),
        "--summary-path",
        str(summary_path),
        "--checkpoint-dir",
        str(checkpoint_dir),
    ]
    if args.csv_path:
        command.extend(["--csv-path", args.csv_path])

    if "learning_rate" in selected_config:
        command.extend(["--learning-rate", str(selected_config["learning_rate"])])
    if "batch_size" in selected_config:
        command.extend(["--train-batch-size", str(selected_config["batch_size"])])
    if "max_sequence_length" in selected_config:
        command.extend(["--max-seq-len", str(selected_config["max_sequence_length"])])
    if "use_lora" in selected_config:
        command.extend(["--use-lora", "true" if bool(selected_config["use_lora"]) else "false"])

    result = _run_command(command, timeout_seconds=args.train_timeout_seconds)
    if result.returncode != 0:
        raise RuntimeError(
            f"train_failed code={result.returncode}\n"
            f"stdout_tail={result.stdout[-1200:]}\n"
            f"stderr_tail={result.stderr[-1200:]}"
        )

    if summary_path.exists():
        return json.loads(summary_path.read_text(encoding="utf-8"))

    parsed = _parse_key_value_output(result.stdout)
    return {
        "model_name": parsed.get("model_name", run.model_name),
        "model_family": parsed.get("model_family", run.model_family),
        "model_registry_id": parsed.get("model_registry_id", run.registry_id),
        "val_metric": float(parsed.get("val_metric", 0.0)),
        "json_schema_compliance": float(parsed.get("json_schema_compliance", 0.0)),
        "config_json": parsed.get("config_json", "{}"),
        "checkpoint_path": str(checkpoint_dir),
    }


def _infer_experiment(run: ExperimentRun, train_summary: Mapping[str, Any], args: argparse.Namespace) -> dict[str, str]:
    checkpoint_path = str(train_summary.get("checkpoint_path", "")).strip()

    command = [
        "uv",
        "run",
        "inference.py",
        "--model-family",
        run.model_family,
        "--model-name",
        run.model_name,
        "--split",
        args.split,
        "--run-tag",
        run.experiment_id,
        "--output-path",
        run.predictions_path,
        "--batch-size",
        str(max(1, args.batch_size)),
        "--max-new-tokens",
        str(max(1, args.max_new_tokens)),
        "--decoder-json-decoding",
        args.decoder_json_decoding,
        "--metadata-token-confidence",
        str(max(0.0, min(1.0, float(args.metadata_token_confidence)))),
        "--metadata-min-chars",
        str(max(1, int(args.metadata_min_chars))),
        "--max-metadata-entities",
        str(max(1, int(args.max_metadata_entities))),
        "--max-metadata-total-chars",
        str(max(16, int(args.max_metadata_total_chars))),
    ]
    if args.csv_path:
        command.extend(["--csv-path", args.csv_path])
    if checkpoint_path:
        command.extend(["--checkpoint-path", checkpoint_path])

    result = _run_command(command, timeout_seconds=args.inference_timeout_seconds)
    if result.returncode != 0:
        raise RuntimeError(
            f"inference_failed code={result.returncode}\n"
            f"stdout_tail={result.stdout[-1200:]}\n"
            f"stderr_tail={result.stderr[-1200:]}"
        )

    parsed = _parse_key_value_output(result.stdout)
    return parsed


def _evaluate_and_log(
    run: ExperimentRun,
    train_summary: Mapping[str, Any],
    args: argparse.Namespace,
) -> tuple[float, float, str, dict[str, Any], dict[str, Any]]:
    metrics, alignment_stats = eval_module.evaluate_predictions(
        predictions_path=Path(run.predictions_path),
        split=args.split,
        csv_path=args.csv_path,
    )

    val_metric = float(metrics.get("val_metric", 0.0))
    json_compliance = float(metrics.get("json_schema_compliance", 0.0))
    best_before = _read_best_metric(Path(args.leaderboard_path))
    status = "keep" if val_metric >= best_before else "discard"

    raw_config = train_summary.get("config_json", "{}")
    try:
        parsed_train_config = json.loads(raw_config) if isinstance(raw_config, str) else dict(raw_config)
    except (TypeError, ValueError):
        parsed_train_config = {}

    record = ExperimentRecord(
        experiment_id=run.experiment_id,
        model_registry_id=run.registry_id,
        hf_model_name=run.model_name,
        model_family=run.model_family,
        val_metric=val_metric,
        json_schema_compliance=json_compliance,
        status=status,
        run_tag=run.run_tag,
        selection_strategy=run.strategy,
        checkpoint_path=str(train_summary.get("checkpoint_path", "")),
        predictions_path=run.predictions_path,
        metrics=dict(metrics),
        config={
            "split": args.split,
            "csv_path": args.csv_path or "",
            "selected_registry_id": run.registry_id,
            "selected_model_name": run.model_name,
            "selected_model_family": run.model_family,
            "train_config": parsed_train_config,
            "alignment_stats": alignment_stats,
        },
        notes=f"phase6_orchestrated:{status}",
    )

    leaderboard = eval_module.append_result(
        record,
        results_path=Path(args.results_path),
        leaderboard_path=Path(args.leaderboard_path),
    )

    return val_metric, json_compliance, status, metrics, leaderboard


def _single_experiment(
    experiment_index: int,
    args: argparse.Namespace,
) -> ExperimentRun:
    selected = select_model(
        experiment_index=experiment_index,
        strategy=SelectionStrategy(args.selection_strategy),
        random_seed=args.random_seed,
    )

    experiment_id = _make_experiment_id(args.run_tag, experiment_index, selected.spec.registry_id)
    artifact_dir = ROOT_DIR / "artifacts" / "experiments" / experiment_id
    train_summary_path = artifact_dir / "train_summary.json"
    checkpoint_dir = artifact_dir / "checkpoint"
    predictions_path = ROOT_DIR / "predictions" / f"{experiment_id}_{selected.spec.family.value}_{args.split}_predictions.jsonl"

    run = ExperimentRun(
        experiment_id=experiment_id,
        experiment_index=experiment_index,
        registry_id=selected.spec.registry_id,
        model_name=selected.spec.hf_model_name,
        model_family=selected.spec.family.value,
        strategy=selected.strategy.value,
        run_tag=args.run_tag,
        train_summary_path=str(train_summary_path),
        checkpoint_dir=str(checkpoint_dir),
        predictions_path=str(predictions_path),
    )

    profile = resolve_task_profile(args.task_profile)
    context = build_task_context(
        profile_id=args.task_profile,
        split=args.split,
        csv_path=args.csv_path,
        run_tag=args.run_tag,
        experiment_id=run.experiment_id,
        root_dir=ROOT_DIR,
        artifact_dir=artifact_dir,
    )

    if args.dry_run:
        return run

    if args.skip_train:
        train_summary = {
            "model_name": run.model_name,
            "model_family": run.model_family,
            "model_registry_id": run.registry_id,
            "selection_strategy": run.strategy,
            "config_json": _stable_json(selected.config),
            "checkpoint_path": "",
        }
    else:
        train_summary = profile.train.train(
            context=context,
            selected_model={
                "registry_id": run.registry_id,
                "model_name": run.model_name,
                "model_family": run.model_family,
                "selection_strategy": run.strategy,
                "config": selected.config,
            },
        )
        run.model_name = str(train_summary.get("model_name", run.model_name))
        run.model_family = str(train_summary.get("model_family", run.model_family))

    infer_summary = profile.inference.infer(context=context, train_summary=train_summary)
    run.predictions_path = str(infer_summary.get("predictions_path", run.predictions_path))

    metrics, alignment_stats = profile.evaluator.evaluate(
        context=context,
        predictions_path=run.predictions_path,
    )

    val_metric = float(metrics.get("val_metric", 0.0))
    json_compliance = float(metrics.get("json_schema_compliance", 0.0))
    best_before = _read_best_metric(Path(args.leaderboard_path))
    status = "keep" if val_metric >= best_before else "discard"

    raw_config = train_summary.get("config_json", "{}")
    try:
        parsed_train_config = json.loads(raw_config) if isinstance(raw_config, str) else dict(raw_config)
    except (TypeError, ValueError):
        parsed_train_config = {}

    leaderboard = eval_module.append_result(
        ExperimentRecord(
            experiment_id=run.experiment_id,
            model_registry_id=run.registry_id,
            hf_model_name=run.model_name,
            model_family=run.model_family,
            val_metric=val_metric,
            json_schema_compliance=json_compliance,
            status=status,
            run_tag=run.run_tag,
            selection_strategy=run.strategy,
            checkpoint_path=str(train_summary.get("checkpoint_path", "")),
            predictions_path=run.predictions_path,
            metrics=dict(metrics),
            config={
                "split": args.split,
                "csv_path": args.csv_path or "",
                "selected_registry_id": run.registry_id,
                "selected_model_name": run.model_name,
                "selected_model_family": run.model_family,
                "train_config": parsed_train_config,
                "alignment_stats": alignment_stats,
                "task_profile": args.task_profile,
            },
            notes=f"phase6_orchestrated:{status};profile={args.task_profile}",
        ),
        results_path=Path(args.results_path),
        leaderboard_path=Path(args.leaderboard_path),
    )

    run.val_metric = val_metric
    run.json_schema_compliance = json_compliance
    run.status = status
    run.notes = _stable_json(
        {
            "metrics": metrics,
            "leaderboard_total": leaderboard.get("total_experiments", 0),
        }
    )

    return run


def main(argv: Sequence[str] | None = None) -> None:
    args = _build_arg_parser().parse_args(argv)

    if bool(args.list_task_profiles):
        for profile_id in list_task_profile_ids():
            print(profile_id)
        return

    if bool(args.validate_all_task_profiles):
        all_ok = True
        for profile_id in list_task_profile_ids():
            ok = _run_profile_validation(profile_id=profile_id, split=args.split, csv_path=args.csv_path)
            all_ok = all_ok and ok
        if not all_ok:
            raise SystemExit(2)
        return

    if bool(args.validate_task_profile):
        ok = _run_profile_validation(profile_id=args.task_profile, split=args.split, csv_path=args.csv_path)
        if not ok:
            raise SystemExit(2)
        return

    if not bool(args.skip_preflight_checks):
        ok = _run_profile_validation(profile_id=args.task_profile, split=args.split, csv_path=args.csv_path)
        if not ok:
            raise SystemExit(2)

    args.state_path = Path(args.state_path)
    args.results_path = Path(args.results_path)
    args.leaderboard_path = Path(args.leaderboard_path)

    config = OrchestratorConfig(
        num_experiments=args.num_experiments,
        start_index=args.start_index,
        selection_strategy=args.selection_strategy,
        random_seed=args.random_seed,
        split=args.split,
        csv_path=args.csv_path,
        run_tag=args.run_tag,
        sleep_seconds=float(max(0.0, args.sleep_seconds)),
        decoder_json_decoding=args.decoder_json_decoding,
        max_new_tokens=max(1, args.max_new_tokens),
        batch_size=max(1, args.batch_size),
        metadata_token_confidence=float(max(0.0, min(1.0, args.metadata_token_confidence))),
        metadata_min_chars=max(1, int(args.metadata_min_chars)),
        max_metadata_entities=max(1, int(args.max_metadata_entities)),
        max_metadata_total_chars=max(16, int(args.max_metadata_total_chars)),
        dry_run=bool(args.dry_run),
        skip_train=bool(args.skip_train),
        train_timeout_seconds=int(max(0, args.train_timeout_seconds)),
        inference_timeout_seconds=int(max(0, args.inference_timeout_seconds)),
        state_path=str(args.state_path),
        results_path=str(args.results_path),
        leaderboard_path=str(args.leaderboard_path),
        task_profile=args.task_profile,
    )

    print("---")
    print(f"run_tag: {config.run_tag}")
    print(f"num_experiments: {config.num_experiments}")
    print(f"start_index: {config.start_index}")
    print(f"selection_strategy: {config.selection_strategy}")
    print(f"task_profile: {config.task_profile}")
    print(f"split: {config.split}")
    print(f"dry_run: {config.dry_run}")
    print(f"skip_train: {config.skip_train}")

    outcomes: list[ExperimentRun] = []
    for offset in range(config.num_experiments):
        experiment_index = config.start_index + offset
        t0 = time.time()

        try:
            run = _single_experiment(experiment_index, args)
            elapsed = time.time() - t0
            outcomes.append(run)

            state_payload = {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "elapsed_seconds": elapsed,
                **asdict(run),
                "dry_run": config.dry_run,
                "skip_train": config.skip_train,
            }
            _record_state(args.state_path, state_payload)

            print("---")
            print(f"experiment_id: {run.experiment_id}")
            print(f"experiment_index: {run.experiment_index}")
            print(f"model_registry_id: {run.registry_id}")
            print(f"model_name: {run.model_name}")
            print(f"model_family: {run.model_family}")
            print(f"status: {run.status}")
            print(f"val_metric: {run.val_metric:.6f}")
            print(f"json_schema_compliance: {run.json_schema_compliance:.6f}")
            print(f"predictions_path: {run.predictions_path}")
            print(f"elapsed_seconds: {elapsed:.2f}")

        except Exception as exc:
            elapsed = time.time() - t0
            error_payload = {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "experiment_index": experiment_index,
                "status": "crash",
                "error": str(exc),
                "elapsed_seconds": elapsed,
                "run_tag": config.run_tag,
            }
            _record_state(args.state_path, error_payload)
            print("---")
            print(f"experiment_index: {experiment_index}")
            print("status: crash")
            print(f"error: {exc}")

        if config.sleep_seconds > 0 and offset < config.num_experiments - 1:
            time.sleep(config.sleep_seconds)

    print("---")
    print(f"state_path: {args.state_path}")
    print(f"completed_runs: {len(outcomes)}")


if __name__ == "__main__":
    main()
