from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from tasks.common.runtime import list_task_profile_ids
from tasks.common.runtime import resolve_task_profile
from tasks.common.validation import format_validation_report
from tasks.common.validation import resolve_validation_csv_path
from tasks.common.validation import validate_task_profile


ROOT_DIR = Path(__file__).resolve().parent


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Phase-7 smoke checks for task profiles")
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--task-profile", type=str, default=None)
    parser.add_argument("--csv-path", type=str, default=None)
    parser.add_argument("--fail-fast", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _build_arg_parser().parse_args(argv)

    profile_ids = [args.task_profile] if args.task_profile else list_task_profile_ids()
    has_failures = False

    for profile_id in profile_ids:
        profile = resolve_task_profile(profile_id)
        report = validate_task_profile(
            profile=profile,
            split=args.split,
            csv_path=resolve_validation_csv_path(
                profile_id=profile_id,
                csv_path=args.csv_path,
                root_dir=ROOT_DIR,
            ),
        )
        print("---")
        print(format_validation_report(report))
        if not report.ok:
            has_failures = True
            if args.fail_fast:
                break

    if has_failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main(sys.argv[1:])
