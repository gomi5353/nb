#!/usr/bin/env python3
"""Check how well a LeRobot dataset covers object initial poses, against your own spec.

Reads only meta/info.json and data/**/*.parquet (no simulator, no videos).
Every episode counts, successful or failed. Guide: docs/object_pose_coverage.md

  # 1. Create a spec skeleton (one block per object_pose.* column in the dataset)
  uv run python scripts/verify_object_pose_coverage.py init \\
      --dataset datasets/<repo_id> --out specs/my_coverage.ttl

  # 2. Replace every "TODO" in the spec with your own numbers, then check
  uv run python scripts/verify_object_pose_coverage.py check \\
      --dataset datasets/<repo_id> --spec specs/my_coverage.ttl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from data_verification import format_report, verify_object_pose_coverage, write_spec
from data_verification.dataset import DatasetError, read_initial_placements


def cmd_init(args: argparse.Namespace) -> int:
    observed = None
    if args.dataset is not None:
        try:
            observed = read_initial_placements(args.dataset)
        except DatasetError as exc:
            print(f"ERROR  {exc}", file=sys.stderr)
            return 1
    if args.objects:
        names = [n.strip() for n in args.objects.split(",") if n.strip()]
    elif observed is not None:
        names = observed.object_names
    else:
        print("ERROR  pass --dataset (objects taken from its object_pose.* columns) or --objects", file=sys.stderr)
        return 1
    try:
        path = write_spec(args.out, names, observed, dataset_label=args.dataset.name if args.dataset else "")
    except FileExistsError as exc:
        print(f"ERROR  {exc}", file=sys.stderr)
        return 1
    print(f"Wrote {path} with {len(names)} object block(s): {', '.join(names)}")
    print('Next: replace every "TODO" with your own values, then run the check subcommand.')
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    report = verify_object_pose_coverage(args.dataset, args.spec, extra_shapes=args.extra_shapes)
    print(format_report(report))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report.to_dict(), indent=2) + "\n")
    return 0 if report.passed and not (args.strict and report.warnings) else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    init = sub.add_parser("init", help="Write a spec skeleton to fill in.")
    init.add_argument("--out", required=True, type=Path, help="Spec file to create (.ttl). Never overwritten.")
    init.add_argument("--dataset", type=Path, help="Dataset to take object names and observed ranges from.")
    init.add_argument("--objects", help="Comma-separated object names instead of the dataset's columns.")
    init.set_defaults(func=cmd_init)

    check = sub.add_parser("check", help="Check a dataset against a filled-in spec.")
    check.add_argument("--dataset", required=True, type=Path, help="Local LeRobot v3 dataset directory.")
    check.add_argument("--spec", required=True, type=Path, help="Your coverage spec (.ttl).")
    check.add_argument("--json", type=Path, help="Also write the full report as JSON.")
    check.add_argument("--strict", action="store_true", help="Also fail on warnings.")
    check.add_argument(
        "--extra-shapes", type=Path, action="append", default=[], help="Additional SHACL rules (.ttl); repeatable."
    )
    check.set_defaults(func=cmd_check)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
