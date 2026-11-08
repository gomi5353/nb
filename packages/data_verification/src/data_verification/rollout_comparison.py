"""Compare a finished rollout's runs with the coverage of the dataset the policy was trained on, per axis.

Run after ``scripts/rollout.py`` has written its per-run log. The comparison
starts from the rollout: its training dataset gives the requirements, its
finished runs (succeeded or failed; aborted runs are not counted) give the
initial positions. For the object position axis, each run is placed into the
lattice of the requirement's grid: column i = floor((x - areaXMin) / cell
width), row j likewise, with the area's upper edges in the last column and row.
A run inside the area is covered when its cell has at least minEpisodesPerCell
dataset episodes; a run outside the area is not covered.

A run is in distribution when it is covered on every axis; for the object
position axis, that holds for every requirement of the training dataset.
Further axes add an ``AxisComparison`` to ``ROLLOUT_AXES`` (see
docs/adding_coverage_factor.md).
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from decimal import Decimal
from pathlib import Path

from pyshacl import validate
from rdflib import RDF, Graph, URIRef

from .coverage import (
    SH,
    Finding,
    _packaged_graph,
    build_verification_graph,
    cell_counts,
    cell_edge,
    lattice_index,
    requirement_rows,
)
from .graph import CP, add_rollout, dataset_iri
from .rollout_log import RolloutLog, RolloutLogError, read_rollout_log

# Initial positions of the finished runs of ?rollout (bound by the caller).
_RUN_POSITIONS_QUERY = """
SELECT ?index ?status ?name ?x ?y WHERE {
    ?rollout cp:hasRun ?run .
    ?run cp:index ?index ; cp:status ?status ; cp:hasTaskObject ?obj .
    FILTER (?status IN ("succeeded", "failed"))
    ?obj cp:name ?name ; cp:hasPosition ?p .
    ?p cp:positionX ?x ; cp:positionY ?y .
}
"""

OBJECT_POSITION_AXIS = "object_position"


@dataclass
class RunCell:
    """One cell of the requirement grid's lattice, inside the area or (for runs that started there) outside it."""

    i: int
    j: int
    x_range: tuple[float, float]
    y_range: tuple[float, float]
    inside: bool
    episodes: int = 0
    min_episodes: int = 0
    runs: int = 0
    successes: int = 0

    @property
    def sampled(self) -> bool:
        return self.runs > 0

    @property
    def covered(self) -> bool:
        return self.inside and self.episodes >= self.min_episodes

    @property
    def needed(self) -> int:
        return max(0, self.min_episodes - self.episodes) if self.inside else 0


@dataclass
class ObjectComparison:
    object_name: str
    cell_size: tuple[float, float]
    requirement_area: tuple[float, float, float, float]
    min_episodes: int
    cells: list[list[RunCell]] = field(default_factory=list)
    """cells[j][i]: the requirement grid, column i (x), row j (y), row-major index k = j * numCellW + i."""
    outside: list[RunCell] = field(default_factory=list)
    """Lattice cells outside the area that runs started in, sorted by (j, i)."""
    runs: int = 0
    runs_covered: int = 0
    successes_covered: int = 0
    successes_not_covered: int = 0

    @property
    def runs_outside(self) -> int:
        return sum(c.runs for c in self.outside)

    @property
    def sampled_cells(self) -> int:
        return sum(c.sampled for row in self.cells for c in row) + len(self.outside)

    @property
    def covered_cells(self) -> int:
        return sum(c.sampled and c.covered for row in self.cells for c in row)


@dataclass
class AxisComparison:
    axis: str
    objects: list[ObjectComparison]
    run_covered: dict[int, bool]
    """Run index -> whether it is covered on this axis."""


@dataclass
class FillTarget:
    """A requirement cell to record more episodes in."""

    object_name: str
    i: int
    j: int
    x_range: tuple[float, float]
    y_range: tuple[float, float]
    episodes: int
    needed: int
    runs: int
    failed: int


@dataclass
class AreaSuggestion:
    """Grow the requirement area by whole cells so it contains the runs that started outside it."""

    object_name: str
    current: tuple[float, float, float, float]
    suggested: tuple[float, float, float, float]
    num_cell_w: int
    num_cell_h: int


@dataclass
class RolloutComparison:
    dataset: str
    spec: str
    rollout_log: str
    eval_file: str = ""
    seed: int | None = None
    runs: int = 0
    successes: int = 0
    aborted: int = 0
    errors: list[str] = field(default_factory=list)
    warnings: list[Finding] = field(default_factory=list)
    axes: list[AxisComparison] = field(default_factory=list)
    runs_in_distribution: int = 0
    successes_in_distribution: int = 0
    to_fill: list[FillTarget] = field(default_factory=list)
    widen: list[AreaSuggestion] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.errors

    @property
    def fully_covered(self) -> bool:
        return self.passed and self.runs > 0 and self.runs_in_distribution == self.runs

    def to_dict(self) -> dict:
        result = asdict(self)
        result["passed"] = self.passed
        result["fully_covered"] = self.fully_covered
        return result


@dataclass
class _Requirement:
    name: str
    x0: Decimal
    x1: Decimal
    y0: Decimal
    y1: Decimal
    nw: int
    nh: int
    min_episodes: int
    counts: Counter

    @property
    def size(self) -> tuple[Decimal, Decimal]:
        return (self.x1 - self.x0) / self.nw, (self.y1 - self.y0) / self.nh

    def lattice_index(self, x: Decimal, y: Decimal) -> tuple[int, int]:
        """Column and row of the lattice cell (x, y) is in, with the membership rule of the coverage check."""
        return lattice_index(x, self.x0, self.x1, self.nw), lattice_index(y, self.y0, self.y1, self.nh)

    def cell(self, i: int, j: int) -> RunCell:
        inside = 0 <= i < self.nw and 0 <= j < self.nh
        x_range = cell_edge(self.x0, self.x1, self.nw, i), cell_edge(self.x0, self.x1, self.nw, i + 1)
        y_range = cell_edge(self.y0, self.y1, self.nh, j), cell_edge(self.y0, self.y1, self.nh, j + 1)
        cell = RunCell(i, j, _floats(x_range), _floats(y_range), inside)
        if inside:
            cell.episodes, cell.min_episodes = self.counts[j * self.nw + i], self.min_episodes
        return cell


def compare_rollout(
    dataset_root: str | Path,
    spec: str | Path,
    rollout_log: str | Path,
    graph_out: str | Path | None = None,
) -> RolloutComparison:
    """Compare the rollout in ``rollout_log`` with the coverage of its training dataset ``dataset_root`` under ``spec``."""
    result = RolloutComparison(dataset=str(dataset_root), spec=str(spec), rollout_log=str(rollout_log))
    try:
        log = read_rollout_log(rollout_log)
    except RolloutLogError as exc:
        result.errors.append(str(exc))
        return result
    result.eval_file = str(log.header.get("eval_file", ""))
    result.seed = log.header.get("seed")
    result.aborted = log.aborted
    result.runs = log.finished
    result.successes = log.successes
    if not log.finished:
        result.errors.append(f"{rollout_log}: no finished run (all aborted or none run)")
        return result

    graph, dataset_report = build_verification_graph(dataset_root, spec)
    result.errors.extend(dataset_report.errors)
    if result.errors:
        return result
    dataset = dataset_iri(Path(dataset_root).resolve().name)
    requirements = _requirements(graph, dataset, result)
    if result.errors:
        return result
    rollout = add_rollout(graph, log, dataset)
    _check_training_dataset(log, Path(dataset_root), result)

    _validate_rollout(graph, result)
    if graph_out is not None:
        Path(graph_out).parent.mkdir(parents=True, exist_ok=True)
        graph.serialize(destination=Path(graph_out), format="turtle")
    if result.errors:
        return result

    for comparer in ROLLOUT_AXES:
        result.axes.append(comparer(graph, rollout, result, requirements))
    in_distribution = [r for r in log.finished_runs if all(axis.run_covered.get(r.index, False) for axis in result.axes)]
    result.runs_in_distribution = len(in_distribution)
    result.successes_in_distribution = sum(r.succeeded for r in in_distribution)
    _guidance(result, requirements)
    return result


def _requirements(graph: Graph, dataset: URIRef, result: RolloutComparison) -> dict[str, _Requirement]:
    counts = cell_counts(graph, dataset)
    requirements: dict[str, _Requirement] = {}
    for row in requirement_rows(graph, dataset):
        name = str(row.name)
        if name in requirements:
            result.errors.append(f"Spec has more than one requirement for '{name}'; compare needs one per object")
            continue
        requirements[name] = _Requirement(
            name,
            *(Decimal(str(v.toPython())) for v in (row.x0, row.x1, row.y0, row.y1)),
            int(row.nx),
            int(row.ny),
            int(row.min),
            counts.get(str(row.req), Counter()),
        )
    if not requirements:
        result.errors.append("The spec has no object position requirement; nothing to compare")
    return requirements


def _check_training_dataset(log: RolloutLog, dataset_root: Path, result: RolloutComparison) -> None:
    """Warn when the checkpoint was trained on a different dataset than the one given to compare."""
    checkpoint = log.header.get("checkpoint")
    if not checkpoint:
        return
    config = next(
        (p / "train_config.json" for p in (Path(checkpoint), *Path(checkpoint).parents) if (p / "train_config.json").is_file()),
        None,
    )
    if config is None:
        return
    try:
        repo_id = json.loads(config.read_text())["dataset"]["repo_id"]
    except (ValueError, KeyError, TypeError):
        return
    if not isinstance(repo_id, str) or not repo_id:
        return
    if dataset_root.resolve().parts[-len(Path(repo_id).parts) :] != Path(repo_id).parts:
        result.warnings.append(
            Finding(
                "Warning",
                f"The rollout's checkpoint was trained on '{repo_id}' ({config}), not on {dataset_root}; "
                "the comparison describes a dataset the policy did not learn from",
                [str(checkpoint)],
            )
        )


def _validate_rollout(graph: Graph, result: RolloutComparison) -> None:
    _, results_graph, _ = validate(
        data_graph=graph,
        shacl_graph=_packaged_graph("shapes", "rollout_comparison.ttl"),
        ont_graph=_packaged_graph("ontology", "object_position.ttl"),
        inference="none",
        abort_on_first=False,
    )
    grouped: dict[tuple[str, str], list[str]] = {}
    for res in results_graph.subjects(RDF.type, SH.ValidationResult):
        severity = str(results_graph.value(res, SH.resultSeverity)).rsplit("#", 1)[-1]
        message = str(results_graph.value(res, SH.resultMessage) or "")
        grouped.setdefault((severity, message), []).append(str(results_graph.value(res, SH.focusNode)))
    for (severity, message), nodes in sorted(grouped.items()):
        if severity == "Violation":
            result.errors.append(message)
        else:
            result.warnings.append(Finding(severity, message, sorted(nodes)))


def compare_object_position(
    graph: Graph, rollout: URIRef, result: RolloutComparison, requirements: dict[str, _Requirement]
) -> AxisComparison:
    positions: dict[str, dict[int, tuple[Decimal, Decimal]]] = {}
    outcome: dict[int, bool] = {}
    for row in graph.query(_RUN_POSITIONS_QUERY, initNs={"cp": CP}, initBindings={"rollout": rollout}):
        index = int(row["index"])
        outcome[index] = str(row.status) == "succeeded"
        positions.setdefault(str(row.name), {})[index] = (Decimal(str(row.x)), Decimal(str(row.y)))

    run_covered = {index: True for index in outcome}
    objects = []
    for name, req in sorted(requirements.items()):
        dx, dy = req.size
        obj = ObjectComparison(name, _floats((dx, dy)), _floats((req.x0, req.x1, req.y0, req.y1)), req.min_episodes)
        obj.cells = [[req.cell(i, j) for i in range(req.nw)] for j in range(req.nh)]
        outside: dict[tuple[int, int], RunCell] = {}
        for index, (x, y) in sorted(positions.get(name, {}).items()):
            i, j = req.lattice_index(x, y)
            cell = obj.cells[j][i] if 0 <= i < req.nw and 0 <= j < req.nh else outside.setdefault((i, j), req.cell(i, j))
            ok = outcome[index]
            cell.runs += 1
            cell.successes += ok
            obj.runs += 1
            if cell.covered:
                obj.runs_covered += 1
                obj.successes_covered += ok
            else:
                obj.successes_not_covered += ok
                run_covered[index] = False
        obj.outside = [outside[key] for key in sorted(outside, key=lambda k: (k[1], k[0]))]
        objects.append(obj)
        not_covered = [c for row in obj.cells for c in row if c.sampled and not c.covered] + obj.outside
        if not_covered:
            runs = sum(c.runs for c in not_covered)
            failed = runs - sum(c.successes for c in not_covered)
            result.warnings.append(
                Finding(
                    "Warning",
                    f"'{name}': {len(not_covered)} of {obj.sampled_cells} sampled cell(s) are not covered by the "
                    f"dataset ({runs} run(s), {failed} failed)",
                    [f"{name}/cell/({c.i}, {c.j})" for c in not_covered],
                )
            )

    for name, runs in sorted(positions.items()):
        if name in requirements:
            continue
        if len(set(runs.values())) > 1:
            result.warnings.append(
                Finding(
                    "Warning",
                    f"'{name}' starts at different positions across the rollout's runs, but the spec has no "
                    "requirement for it; it is not compared",
                    [name],
                )
            )
    return AxisComparison(OBJECT_POSITION_AXIS, objects, run_covered)


# One comparer per generalization axis: (graph, rollout, result, requirements) -> AxisComparison.
# The graph holds the spec, the training dataset and the rollout with its runs. Groups that add
# a factor append their comparer here; a run must be covered on every axis.
ROLLOUT_AXES: list[Callable[..., AxisComparison]] = [compare_object_position]


def _guidance(result: RolloutComparison, requirements: dict[str, _Requirement]) -> None:
    targets: list[FillTarget] = []
    widen_cells: dict[str, list[RunCell]] = {}
    for axis in result.axes:
        if axis.axis != OBJECT_POSITION_AXIS:
            continue
        for obj in axis.objects:
            for cell in (c for row in obj.cells for c in row if c.sampled and not c.covered):
                targets.append(
                    FillTarget(
                        obj.object_name, cell.i, cell.j, cell.x_range, cell.y_range, cell.episodes, cell.needed,
                        cell.runs, cell.runs - cell.successes,
                    )
                )
            if obj.outside:
                widen_cells[obj.object_name] = obj.outside
    result.to_fill = sorted(targets, key=lambda t: (-t.failed, -t.needed, t.object_name, t.j, t.i))

    for name, cells in sorted(widen_cells.items()):
        req = requirements[name]
        dx, dy = req.size
        # Grow by whole cells, so the cell size and the cells already recorded stay the same.
        left = max(0, -min(c.i for c in cells))
        right = max(0, max(c.i for c in cells) - (req.nw - 1))
        down = max(0, -min(c.j for c in cells))
        up = max(0, max(c.j for c in cells) - (req.nh - 1))
        result.widen.append(
            AreaSuggestion(
                name,
                _floats((req.x0, req.x1, req.y0, req.y1)),
                _floats((req.x0 - left * dx, req.x1 + right * dx, req.y0 - down * dy, req.y1 + up * dy)),
                req.nw + left + right,
                req.nh + down + up,
            )
        )


def _floats(values) -> tuple[float, ...]:
    return tuple(float(v) for v in values)


def format_comparison(result: RolloutComparison, max_nodes: int = 3) -> str:
    lines = [
        f"Dataset : {result.dataset}",
        f"Spec    : {result.spec}",
        f"Rollout : {result.rollout_log}" + (f"  ({result.eval_file}, seed {result.seed})" if result.eval_file else ""),
    ]
    aborted = f", {result.aborted} aborted (not counted)" if result.aborted else ""
    lines += [f"Runs    : {result.runs} finished, {result.successes} succeeded{aborted}", ""]
    for axis in result.axes:
        for obj in axis.objects:
            lines += _object_lines(obj)
    if result.axes and result.runs:
        out = result.runs - result.runs_in_distribution
        lines.append(
            f"In distribution (covered on every axis: {', '.join(a.axis for a in result.axes)}): "
            f"{result.runs_in_distribution}/{result.runs} runs ({_pct(result.runs_in_distribution, result.runs)})"
        )
        lines.append(
            f"  success {_rate(result.successes_in_distribution, result.runs_in_distribution)} in distribution vs "
            f"{_rate(result.successes - result.successes_in_distribution, out)} outside"
        )
        lines.append("")
    if result.to_fill:
        total = sum(t.needed for t in result.to_fill)
        lines.append(f"Record more episodes here ({total} more at least; cells with failed rollout runs first):")
        for t in result.to_fill:
            cx, cy = (t.x_range[0] + t.x_range[1]) / 2, (t.y_range[0] + t.y_range[1]) / 2
            lines.append(
                f"  {t.object_name} cell (i={t.i}, j={t.j}) x=[{t.x_range[0]:.3f}, {t.x_range[1]:.3f}] "
                f"y=[{t.y_range[0]:.3f}, {t.y_range[1]:.3f}] centre ({cx:.3f}, {cy:.3f}): "
                f"has {t.episodes}, needs {t.needed} more; {t.runs} rollout run(s) started here, {t.failed} failed"
            )
        lines.append("")
    if result.widen:
        lines.append("Rollout runs started outside your area. Grow it by whole cells (same cell size), then record there:")
        for w in result.widen:
            lines.append(
                f"  {w.object_name}: area x=[{w.current[0]:.3f}, {w.current[1]:.3f}] y=[{w.current[2]:.3f}, {w.current[3]:.3f}]"
                f" -> areaXMin {w.suggested[0]:.6g} ; areaXMax {w.suggested[1]:.6g} ; areaYMin {w.suggested[2]:.6g} ;"
                f" areaYMax {w.suggested[3]:.6g} ; numCellW {w.num_cell_w} ; numCellH {w.num_cell_h}"
            )
        lines.append("")
    for error in result.errors:
        lines.append(f"ERROR     {error}")
    for f in result.warnings:
        nodes = ", ".join(f.focus_nodes[:max_nodes])
        more = f" (+{f.count - max_nodes} more)" if f.count > max_nodes else ""
        lines.append(f"WARNING   {f.message}  [{nodes}{more}]")
    lines.append("")
    if not result.passed:
        lines.append("RESULT: comparison could not be made")
    elif result.fully_covered:
        lines.append("RESULT: the dataset covers every rollout run")
    else:
        lines.append("RESULT: the dataset does NOT cover every rollout run; see above")
    return "\n".join(lines)


def _object_lines(obj: ObjectComparison) -> list[str]:
    x0, x1, y0, y1 = obj.requirement_area
    nh, nw = len(obj.cells), len(obj.cells[0])
    lines = [
        f"[{obj.object_name}] your area x={x0:.3f}..{x1:.3f} y={y0:.3f}..{y1:.3f}, "
        f"cell {obj.cell_size[0]:.4g} x {obj.cell_size[1]:.4g}, {nw}x{nh} cells, >= {obj.min_episodes} episodes each",
        "  successes/runs per cell; ! = runs started in a cell with too few dataset episodes, - = no runs",
    ]
    texts = [[("-" if not c.sampled else f"{c.successes}/{c.runs}") + ("" if c.covered or not c.sampled else "!") for c in row] for row in obj.cells]
    width = max(4, *(len(t) for row in texts for t in row))
    for j in reversed(range(nh)):  # +y at the top
        lines.append(f"  y{j:<2} " + " ".join(f"{t:>{width}}" for t in texts[j]))
    lines.append("       " + " ".join(f"{'x' + str(i):>{width}}" for i in range(nw)))
    if obj.outside:
        lines.append(f"  {obj.runs_outside} run(s) started outside your area, in {len(obj.outside)} cell(s) of the same size")
    not_covered = obj.runs - obj.runs_covered
    lines.append(
        f"  covered: {obj.covered_cells}/{obj.sampled_cells} sampled cells, {obj.runs_covered}/{obj.runs} runs "
        f"({_pct(obj.runs_covered, obj.runs)})\n  success: {_rate(obj.successes_covered, obj.runs_covered)} covered vs "
        f"{_rate(obj.successes_not_covered, not_covered)} not covered"
    )
    lines.append("")
    return lines


def _pct(n: int, d: int) -> str:
    return f"{100 * n / d:.0f}%" if d else "-"


def _rate(n: int, d: int) -> str:
    return f"{n}/{d}" if d else "0/0"
