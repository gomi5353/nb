#!/usr/bin/env python3
"""Check how well a LeRobot dataset covers object initial positions, against your own spec.

Reads only meta/info.json and data/**/*.parquet (no simulator, no videos).
Every episode in the dataset counts (the repo recorder saves successful episodes only).
Guide: docs/object_position_coverage.md

  # 1. Create a spec skeleton (one block per object_position.* column in the dataset)
  uv run python scripts/verify_object_position_coverage.py init \\
      --dataset datasets/<repo_id> --out specs/my_coverage.ttl

  # 2. Replace every "TODO" in the spec with your own numbers, then check
  uv run python scripts/verify_object_position_coverage.py check \\
      --dataset datasets/<repo_id> --spec specs/my_coverage.ttl

  # 3. After a rollout (scripts/rollout.py writes outputs/rollout/<timestamp>.jsonl):
  #    read the eval file's randomization ranges once, and again whenever it changes,
  uv run python scripts/verify_object_position_coverage.py rollout-range --eval eval/<task>_eval.py
  #    then compare where the rollout's runs started with where your dataset covers
  uv run python scripts/verify_object_position_coverage.py compare \\
      --dataset datasets/<repo_id> --spec specs/my_coverage.ttl \\
      --rollout-log outputs/rollout/<timestamp>.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from data_verification import (
    compare_rollout,
    format_comparison,
    format_report,
    verify_object_position_coverage,
    write_spec,
)
from data_verification.dataset import DatasetError, read_initial_placements
from data_verification.rollout_range import RolloutRangeError, write_range_file

GRAPH_DIR = Path("outputs/object_position_coverage")


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
        print("ERROR  pass --dataset (objects taken from its object_position.* columns) or --objects", file=sys.stderr)
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
    graph_out = None if args.no_graph_out else GRAPH_DIR / f"{args.dataset.resolve().name}.ttl"
    report = verify_object_position_coverage(args.dataset, args.spec, extra_shapes=args.extra_shapes, graph_out=graph_out)
    print(format_report(report))
    if graph_out is not None:
        print(f"Graph written to {graph_out}")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report.to_dict(), indent=2) + "\n")
    return 0 if report.passed and not (args.strict and report.warnings) else 1


def cmd_rollout_range(args: argparse.Namespace) -> int:
    try:
        path, task = write_range_file(args.eval, args.out)
    except RolloutRangeError as exc:
        print(f"ERROR  {exc}", file=sys.stderr)
        return 1
    print(f"Wrote {path} from {task.source_file} (SHA-256 {task.sha256[:12]}):")
    for r in task.ranges:
        print(f"  {r.name}: x {r.x_range[0]}..{r.x_range[1]}  y {r.y_range[0]}..{r.y_range[1]}")
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    graph_out = None
    if not args.no_graph_out:
        graph_out = GRAPH_DIR / f"{args.dataset.resolve().name}__rollout_{args.rollout_log.stem}.ttl"
    result = compare_rollout(
        args.dataset, args.spec, args.rollout_log, args.rollout_range, record_cfg=args.record_cfg, graph_out=graph_out
    )
    print(format_comparison(result))
    if graph_out is not None and graph_out.exists():
        print(f"Graph written to {graph_out}")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result.to_dict(), indent=2) + "\n")
    return 0 if result.passed and not (args.strict and result.warnings) else 1


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
    check.add_argument(
        "--no-graph-out",
        action="store_true",
        help=f"Do not write the RDF graph that SHACL validates (default: {GRAPH_DIR}/<dataset>.ttl).",
    )
    check.add_argument("--strict", action="store_true", help="Also fail on warnings.")
    check.add_argument(
        "--extra-shapes", type=Path, action="append", default=[], help="Additional SHACL rules (.ttl); repeatable."
    )
    check.set_defaults(func=cmd_check)

    rollout_range = sub.add_parser(
        "rollout-range", help="Read an eval file's object position randomization into a rollout range file."
    )
    rollout_range.add_argument("--eval", required=True, type=Path, help="Evaluation task file, e.g. eval/cup_stacking_eval.py.")
    rollout_range.add_argument(
        "--out", type=Path, help="Range file to write (default: outputs/rollout_ranges/<eval file stem>.ttl). Overwritten."
    )
    rollout_range.set_defaults(func=cmd_rollout_range)

    compare = sub.add_parser("compare", help="Compare a finished rollout's run positions with the dataset's coverage.")
    compare.add_argument("--dataset", required=True, type=Path, help="Local LeRobot v3 dataset directory.")
    compare.add_argument("--spec", required=True, type=Path, help="Your coverage spec (.ttl); gives the cell size.")
    compare.add_argument("--rollout-log", required=True, type=Path, help="outputs/rollout/<timestamp>.jsonl written by scripts/rollout.py.")
    compare.add_argument(
        "--rollout-range",
        type=Path,
        help="Rollout range file (default: outputs/rollout_ranges/<stem of the eval file the rollout ran>.ttl).",
    )
    compare.add_argument(
        "--record-cfg", type=Path, help="Your recording task cfg; also print pose ranges that cover the cells to record in."
    )
    compare.add_argument("--json", type=Path, help="Also write the full comparison as JSON.")
    compare.add_argument(
        "--no-graph-out",
        action="store_true",
        help=f"Do not write the RDF graph (default: {GRAPH_DIR}/<dataset>__rollout_<run>.ttl).",
    )
    compare.add_argument("--strict", action="store_true", help="Also fail on warnings (runs the dataset does not cover).")
    compare.set_defaults(func=cmd_compare)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
