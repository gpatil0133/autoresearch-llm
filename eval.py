"""
Evaluation and benchmark tracking utilities for survey autoresearch.

Phase-2 scope in this file:
- Own append-only writes to results.tsv.
- Derive leaderboard.json from results.tsv (no hand editing).
- Provide lightweight experiment summaries and a small CLI.

Task-level scoring is introduced in later phases; this module is prepared for that
by accepting arbitrary metrics/config payloads per experiment.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from prepare import ModelFamily, SurveyExample, load_dataset_splits, resolve_dataset_path, score_predictions
from tasks.common.runtime import build_task_context
from tasks.common.runtime import resolve_task_profile
from tasks.common.validation import format_validation_report
from tasks.common.validation import resolve_validation_csv_path
from tasks.common.validation import validate_task_profile


ROOT_DIR = Path(__file__).resolve().parent
RESULTS_PATH = ROOT_DIR / "results.tsv"
LEADERBOARD_PATH = ROOT_DIR / "leaderboard.json"

RESULTS_COLUMNS = (
	"timestamp_utc",
	"experiment_id",
	"run_tag",
	"model_registry_id",
	"hf_model_name",
	"model_family",
	"selection_strategy",
	"val_metric",
	"json_schema_compliance",
	"status",
	"checkpoint_path",
	"predictions_path",
	"metrics_json",
	"config_json",
	"notes",
)

TOP_K_LEADERBOARD = 20
DEFAULT_METRIC_KEYS = (
	"overall_sentiment_macro_f1",
	"sentence_sentiment_macro_f1",
	"sentence_sentiment_accuracy",
	"metadata_span_f1",
	"metadata_typed_f1",
	"theme_f1",
	"emotion_f1",
	"json_schema_compliance",
	"val_metric",
)


@dataclass(frozen=True)
class ExperimentRecord:
	experiment_id: str
	model_registry_id: str
	hf_model_name: str
	model_family: str
	val_metric: float
	json_schema_compliance: float
	status: str = "completed"
	run_tag: str = "default"
	selection_strategy: str = "round_robin"
	checkpoint_path: str = ""
	predictions_path: str = ""
	metrics: Mapping[str, Any] | None = None
	config: Mapping[str, Any] | None = None
	notes: str = ""
	timestamp_utc: str | None = None

	def to_row(self) -> dict[str, str]:
		timestamp = self.timestamp_utc or _utc_now_iso()
		return {
			"timestamp_utc": timestamp,
			"experiment_id": self.experiment_id,
			"run_tag": self.run_tag,
			"model_registry_id": self.model_registry_id,
			"hf_model_name": self.hf_model_name,
			"model_family": self.model_family,
			"selection_strategy": self.selection_strategy,
			"val_metric": f"{float(self.val_metric):.8f}",
			"json_schema_compliance": f"{float(self.json_schema_compliance):.8f}",
			"status": self.status,
			"checkpoint_path": self.checkpoint_path,
			"predictions_path": self.predictions_path,
			"metrics_json": _stable_json_dumps(self.metrics or {}),
			"config_json": _stable_json_dumps(self.config or {}),
			"notes": self.notes.replace("\t", " "),
		}


def _utc_now_iso() -> str:
	return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _stable_json_dumps(payload: Mapping[str, Any]) -> str:
	return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _safe_float(value: Any, fallback: float = 0.0) -> float:
	try:
		return float(value)
	except (TypeError, ValueError):
		return fallback


def _coerce_record(payload: Mapping[str, Any]) -> ExperimentRecord:
	required = [
		"experiment_id",
		"model_registry_id",
		"hf_model_name",
		"model_family",
		"val_metric",
		"json_schema_compliance",
	]
	missing = [key for key in required if key not in payload]
	if missing:
		raise ValueError(f"Missing required fields in record payload: {missing}")

	return ExperimentRecord(
		experiment_id=str(payload["experiment_id"]),
		model_registry_id=str(payload["model_registry_id"]),
		hf_model_name=str(payload["hf_model_name"]),
		model_family=str(payload["model_family"]),
		val_metric=float(payload["val_metric"]),
		json_schema_compliance=float(payload["json_schema_compliance"]),
		status=str(payload.get("status", "completed")),
		run_tag=str(payload.get("run_tag", "default")),
		selection_strategy=str(payload.get("selection_strategy", "round_robin")),
		checkpoint_path=str(payload.get("checkpoint_path", "")),
		predictions_path=str(payload.get("predictions_path", "")),
		metrics=payload.get("metrics", {}),
		config=payload.get("config", {}),
		notes=str(payload.get("notes", "")),
		timestamp_utc=str(payload.get("timestamp_utc")) if payload.get("timestamp_utc") else None,
	)


def _parse_metrics_json(text: str) -> dict[str, Any]:
	if not text:
		return {}
	payload = json.loads(text)
	if not isinstance(payload, dict):
		return {}
	return payload


def _read_results_rows(results_path: Path = RESULTS_PATH) -> list[dict[str, str]]:
	if not results_path.exists():
		return []
	with results_path.open("r", encoding="utf-8", newline="") as handle:
		reader = csv.DictReader(handle, delimiter="\t")
		if reader.fieldnames != list(RESULTS_COLUMNS):
			raise ValueError(
				"results.tsv header mismatch. "
				"Expected current tracker columns before appending new rows."
			)
		return list(reader)


def ensure_results_tsv(results_path: Path = RESULTS_PATH) -> None:
	results_path.parent.mkdir(parents=True, exist_ok=True)
	if results_path.exists():
		_read_results_rows(results_path)
		return
	with results_path.open("w", encoding="utf-8", newline="") as handle:
		writer = csv.DictWriter(handle, fieldnames=list(RESULTS_COLUMNS), delimiter="\t")
		writer.writeheader()


def append_result(
	record: ExperimentRecord | Mapping[str, Any],
	*,
	results_path: Path = RESULTS_PATH,
	leaderboard_path: Path = LEADERBOARD_PATH,
) -> dict[str, Any]:
	ensure_results_tsv(results_path)
	experiment = record if isinstance(record, ExperimentRecord) else _coerce_record(record)

	with results_path.open("a", encoding="utf-8", newline="") as handle:
		writer = csv.DictWriter(handle, fieldnames=list(RESULTS_COLUMNS), delimiter="\t")
		writer.writerow(experiment.to_row())

	leaderboard = build_leaderboard(results_path=results_path, leaderboard_path=leaderboard_path)
	return leaderboard


def build_leaderboard(
	*,
	results_path: Path = RESULTS_PATH,
	leaderboard_path: Path = LEADERBOARD_PATH,
	top_k: int = TOP_K_LEADERBOARD,
) -> dict[str, Any]:
	rows = _read_results_rows(results_path)
	parsed_rows: list[dict[str, Any]] = []
	for row in rows:
		parsed_rows.append(
			{
				**row,
				"val_metric": _safe_float(row.get("val_metric"), fallback=0.0),
				"json_schema_compliance": _safe_float(row.get("json_schema_compliance"), fallback=0.0),
				"metrics": _parse_metrics_json(row.get("metrics_json", "")),
				"config": _parse_metrics_json(row.get("config_json", "")),
			}
		)

	ranked = sorted(
		parsed_rows,
		key=lambda item: (
			item["val_metric"],
			item["json_schema_compliance"],
			item.get("timestamp_utc", ""),
		),
		reverse=True,
	)
	leaders = ranked[: max(0, top_k)]

	by_family: dict[str, dict[str, Any]] = {}
	for item in ranked:
		family = item.get("model_family") or "unknown"
		if family not in by_family:
			by_family[family] = {
				"best_val_metric": item["val_metric"],
				"best_experiment_id": item.get("experiment_id", ""),
				"num_experiments": 1,
			}
		else:
			by_family[family]["num_experiments"] += 1

	payload = {
		"generated_at_utc": _utc_now_iso(),
		"source_results_path": str(results_path),
		"top_k": top_k,
		"total_experiments": len(parsed_rows),
		"leaders": [
			{
				"rank": index + 1,
				"experiment_id": item.get("experiment_id", ""),
				"run_tag": item.get("run_tag", ""),
				"model_registry_id": item.get("model_registry_id", ""),
				"hf_model_name": item.get("hf_model_name", ""),
				"model_family": item.get("model_family", ""),
				"selection_strategy": item.get("selection_strategy", ""),
				"val_metric": item["val_metric"],
				"json_schema_compliance": item["json_schema_compliance"],
				"status": item.get("status", ""),
				"timestamp_utc": item.get("timestamp_utc", ""),
			}
			for index, item in enumerate(leaders)
		],
		"by_family": by_family,
	}

	leaderboard_path.parent.mkdir(parents=True, exist_ok=True)
	with leaderboard_path.open("w", encoding="utf-8") as handle:
		json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
	return payload


def experiment_summary(
	*,
	results_path: Path = RESULTS_PATH,
	leaderboard_path: Path = LEADERBOARD_PATH,
) -> str:
	leaderboard = build_leaderboard(results_path=results_path, leaderboard_path=leaderboard_path)
	if not leaderboard["leaders"]:
		return "No experiments logged yet."

	best = leaderboard["leaders"][0]
	lines = [
		f"total_experiments: {leaderboard['total_experiments']}",
		f"best_experiment_id: {best['experiment_id']}",
		f"best_model: {best['model_registry_id']} ({best['hf_model_name']})",
		f"best_family: {best['model_family']}",
		f"best_val_metric: {best['val_metric']:.6f}",
		f"best_json_schema_compliance: {best['json_schema_compliance']:.6f}",
		f"best_status: {best['status']}",
	]
	return "\n".join(lines)


def _build_arg_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Benchmark tracker and leaderboard utilities")
	parser.add_argument("--results-path", type=Path, default=RESULTS_PATH)
	parser.add_argument("--leaderboard-path", type=Path, default=LEADERBOARD_PATH)

	subparsers = parser.add_subparsers(dest="command", required=True)

	subparsers.add_parser("init", help="Create results.tsv with header if needed")

	add_parser = subparsers.add_parser("add", help="Append one experiment record from JSON")
	add_parser.add_argument(
		"--record-json",
		type=Path,
		required=True,
		help="Path to a JSON file with experiment fields",
	)

	board_parser = subparsers.add_parser("leaderboard", help="Regenerate leaderboard.json")
	board_parser.add_argument("--top-k", type=int, default=TOP_K_LEADERBOARD)

	score_parser = subparsers.add_parser(
		"score",
		help="Evaluate predictions JSONL on a split, log metrics, and refresh leaderboard",
	)
	score_parser.add_argument("--predictions-path", type=Path, required=True)
	score_parser.add_argument("--split", choices=["train", "val", "test"], default="val")
	score_parser.add_argument("--csv-path", type=str, default=None)
	score_parser.add_argument("--experiment-id", type=str, required=True)
	score_parser.add_argument("--model-registry-id", type=str, required=True)
	score_parser.add_argument("--hf-model-name", type=str, required=True)
	score_parser.add_argument(
		"--model-family",
		choices=[item.value for item in ModelFamily],
		required=True,
	)
	score_parser.add_argument("--selection-strategy", type=str, default="round_robin")
	score_parser.add_argument("--run-tag", type=str, default="default")
	score_parser.add_argument("--status", type=str, default="completed")
	score_parser.add_argument("--checkpoint-path", type=str, default="")
	score_parser.add_argument("--notes", type=str, default="")
	score_parser.add_argument("--task-profile", type=str, default="nlp_analysis")

	subparsers.add_parser("summary", help="Print concise summary")
	return parser


def _load_json_file(path: Path) -> Any:
	with path.open("r", encoding="utf-8") as handle:
		return json.load(handle)


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
						"example_id": None,
						"prediction": {},
						"is_valid_json": False,
						"validation_error": f"jsonl_parse_error@line_{line_number}: {exc}",
					}
				)
				continue

			if isinstance(payload, dict):
				rows.append(payload)
			else:
				rows.append(
					{
						"example_id": None,
						"prediction": {},
						"is_valid_json": False,
						"validation_error": f"jsonl_row_not_object@line_{line_number}",
					}
				)
	return rows


def _examples_for_split(split_name: str, csv_path: str | None) -> list[SurveyExample]:
	splits = load_dataset_splits(csv_path)
	if split_name == "train":
		return list(splits.train)
	if split_name == "val":
		return list(splits.val)
	if split_name == "test":
		return list(splits.test)
	raise ValueError(f"Unsupported split: {split_name}")


def _align_predictions_to_references(
	references: Sequence[SurveyExample],
	prediction_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[Mapping[str, Any] | str], dict[str, Any]]:
	by_example_id: dict[str, Mapping[str, Any]] = {}
	duplicate_ids = 0
	invalid_records = 0
	unkeyed_records = 0

	for row in prediction_rows:
		example_id_raw = row.get("example_id")
		example_id = str(example_id_raw).strip() if example_id_raw is not None else ""
		if not example_id:
			unkeyed_records += 1
			continue
		if example_id in by_example_id:
			duplicate_ids += 1
		by_example_id[example_id] = row

	aligned: list[Mapping[str, Any] | str] = []
	missing_predictions = 0
	for example in references:
		row = by_example_id.get(example.example_id)
		if row is None:
			missing_predictions += 1
			invalid_records += 1
			aligned.append("<missing-prediction>")
			continue

		prediction = row.get("prediction", row)
		if prediction is None:
			invalid_records += 1
			aligned.append("<invalid-prediction>")
			continue

		is_valid_json = row.get("is_valid_json")
		if is_valid_json is False:
			invalid_records += 1
			aligned.append("<invalid-prediction>")
			continue

		aligned.append(prediction)

	stats = {
		"prediction_rows": len(prediction_rows),
		"matched_examples": len(references) - missing_predictions,
		"missing_predictions": missing_predictions,
		"invalid_predictions": invalid_records,
		"duplicate_prediction_ids": duplicate_ids,
		"unkeyed_prediction_rows": unkeyed_records,
	}
	return aligned, stats


def evaluate_predictions(
	*,
	predictions_path: Path,
	split: str,
	csv_path: str | None = None,
) -> tuple[dict[str, float], dict[str, Any]]:
	prediction_rows = _load_predictions_jsonl(predictions_path)
	resolved_csv_path = str(resolve_dataset_path(csv_path))
	references = _examples_for_split(split, resolved_csv_path)
	if not references:
		raise ValueError(f"No examples available for split={split!r}")

	aligned_predictions, alignment_stats = _align_predictions_to_references(
		references,
		prediction_rows,
	)
	alignment_stats["resolved_csv_path"] = resolved_csv_path
	metrics = score_predictions(references, aligned_predictions)
	return metrics, alignment_stats


def main(argv: Sequence[str] | None = None) -> None:
	parser = _build_arg_parser()
	args = parser.parse_args(argv)

	if args.command == "init":
		ensure_results_tsv(results_path=args.results_path)
		print(f"initialized_results_tsv: {args.results_path}")
		return

	if args.command == "add":
		payload = _load_json_file(args.record_json)
		if not isinstance(payload, dict):
			raise ValueError("record JSON must be an object")
		leaderboard = append_result(
			payload,
			results_path=args.results_path,
			leaderboard_path=args.leaderboard_path,
		)
		print(f"appended_experiment: {payload.get('experiment_id', '')}")
		print(f"leaderboard_total: {leaderboard['total_experiments']}")
		return

	if args.command == "leaderboard":
		leaderboard = build_leaderboard(
			results_path=args.results_path,
			leaderboard_path=args.leaderboard_path,
			top_k=args.top_k,
		)
		print(f"leaderboard_total: {leaderboard['total_experiments']}")
		print(f"leaderboard_path: {args.leaderboard_path}")
		return

	if args.command == "score":
		resolved_profile = resolve_task_profile(args.task_profile)
		preflight = validate_task_profile(
			profile=resolved_profile,
			split=args.split,
			csv_path=resolve_validation_csv_path(
				profile_id=args.task_profile,
				csv_path=args.csv_path,
				root_dir=ROOT_DIR,
			),
		)
		if not preflight.ok:
			raise ValueError(f"task-profile preflight failed\n{format_validation_report(preflight)}")
		if args.task_profile == "nlp_analysis":
			metrics, alignment_stats = evaluate_predictions(
				predictions_path=args.predictions_path,
				split=args.split,
				csv_path=args.csv_path,
			)
		else:
			profile = resolved_profile
			context = build_task_context(
				profile_id=args.task_profile,
				split=args.split,
				csv_path=args.csv_path,
				run_tag=args.run_tag,
				experiment_id=args.experiment_id,
				root_dir=ROOT_DIR,
			)
			metrics, alignment_stats = profile.evaluator.evaluate(
				context=context,
				predictions_path=str(args.predictions_path),
			)

		record = ExperimentRecord(
			experiment_id=args.experiment_id,
			model_registry_id=args.model_registry_id,
			hf_model_name=args.hf_model_name,
			model_family=args.model_family,
			val_metric=float(metrics.get("val_metric", 0.0)),
			json_schema_compliance=float(metrics.get("json_schema_compliance", 0.0)),
			status=args.status,
			run_tag=args.run_tag,
			selection_strategy=args.selection_strategy,
			checkpoint_path=args.checkpoint_path,
			predictions_path=str(args.predictions_path),
			metrics=dict(metrics),
			config={
				"split": args.split,
				"csv_path": args.csv_path or "",
				"predictions_path": str(args.predictions_path),
				"alignment_stats": alignment_stats,
				"task_profile": args.task_profile,
			},
			notes=args.notes,
		)

		leaderboard = append_result(
			record,
			results_path=args.results_path,
			leaderboard_path=args.leaderboard_path,
		)

		for key in DEFAULT_METRIC_KEYS:
			if key in metrics:
				print(f"{key}: {float(metrics[key]):.6f}")
		print(f"prediction_rows: {alignment_stats['prediction_rows']}")
		print(f"matched_examples: {alignment_stats['matched_examples']}")
		print(f"missing_predictions: {alignment_stats['missing_predictions']}")
		print(f"invalid_predictions: {alignment_stats['invalid_predictions']}")
		print(f"duplicate_prediction_ids: {alignment_stats['duplicate_prediction_ids']}")
		print(f"unkeyed_prediction_rows: {alignment_stats['unkeyed_prediction_rows']}")
		print(f"appended_experiment: {args.experiment_id}")
		print(f"leaderboard_total: {leaderboard['total_experiments']}")
		print(f"results_path: {args.results_path}")
		print(f"leaderboard_path: {args.leaderboard_path}")
		return

	if args.command == "summary":
		print(
			experiment_summary(
				results_path=args.results_path,
				leaderboard_path=args.leaderboard_path,
			)
		)
		return

	raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
	main()

