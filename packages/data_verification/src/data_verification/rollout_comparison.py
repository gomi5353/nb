"""Compare a finished rollout's coverage with a dataset's coverage, per generalization axis.

Run after ``scripts/rollout.py`` has written its per-episode log. For the object
position axis, each object's rollout range (read from the evaluation task file)
is split into a rollout grid with the student's cell size for that object,
starting at the range minimum; the last column and row are clipped to the range.
Each rollout cell counts the runs that started in it and how many succeeded; it
has no minimum. A sampled rollout cell (one with at least one run) is covered
when it lies inside the student's area and every student cell it overlaps (at
most 2 x 2, since the cell sizes are equal) has at least minEpisodesPerCell
dataset episodes.

A run is in distribution when, on every axis, the cell it started in is covered;
for the object position axis, that holds for every randomized object. Further
axes add an ``AxisComparison`` to ``ROLLOUT_AXES`` (see docs/adding_coverage_factor.md).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from decimal import ROUND_CEILING, Decimal
from pathlib import Path

from pyshacl import validate
from rdflib import RDF, XSD, Graph

from .coverage import (
    _CELL_COUNTS_QUERY,
    _REQUIREMENTS_QUERY,
    SH,
    Finding,
    _packaged_graph,
    build_verification_graph,
)
from .graph import CP, add_rollout, add_rollout_grid
from .rollout_log import RolloutLog, RolloutLogError, read_rollout_log
from .rollout_range import (
    RolloutRangeError,
    default_range_path,
    file_sha256,
    range_file_sha256,
    read_task_ranges,
    regenerate_hint,
)

_RANGES_QUERY = """
SELECT ?range ?name ?area ?x0 ?x1 ?y0 ?y1 WHERE {
    ?range a cp:ObjectPositionRolloutRange ; cp:hasTaskObject ?obj ; cp:hasAreaConfiguration ?area .
    ?obj cp:hasName ?n . ?n cp:nameValue ?name .
    ?area cp:areaXMin ?x0 ; cp:areaXMax ?x1 ; cp:areaYMin ?y0 ; cp:areaYMax ?y1 .
}
"""

# Rollout cell bounds: cellSize steps from the area minimum, the last column
# and row end at the area maximum. Membership as in cp:CellShape: lower edges
# inclusive, upper edges exclusive except on the last column and row.
_RUN_CELLS_QUERY = """
SELECT ?range ?k ?epi ?ok WHERE {
    ?rollout a cp:Rollout ; cp:hasRolloutRange ?range ; cp:hasEpisode ?episode .
    ?range a cp:ObjectPositionRolloutRange ; cp:hasTaskObject ?rangeObj ; cp:hasAreaConfiguration ?area .
    ?rangeObj cp:hasName ?rangeName . ?rangeName cp:nameValue ?name .
    ?area cp:areaXMin ?ax0 ; cp:areaXMax ?ax1 ; cp:areaYMin ?ay0 ; cp:areaYMax ?ay1 ; cp:hasCoverageGrid ?grid .
    ?grid cp:cellSizeX ?dx ; cp:cellSizeY ?dy ; cp:numCellW ?nw ; cp:numCellH ?nh ; cp:hasCell ?cell .
    ?cell cp:hasIndex ?index . ?index cp:indexValue ?k .
    BIND (xsd:integer(FLOOR(?k / ?nw)) AS ?j)
    BIND (?k - ?nw * ?j AS ?i)
    BIND (?ax0 + ?i * ?dx AS ?x0)
    BIND (IF(?i = ?nw - 1, ?ax1, ?ax0 + (?i + 1) * ?dx) AS ?x1)
    BIND (?ay0 + ?j * ?dy AS ?y0)
    BIND (IF(?j = ?nh - 1, ?ay1, ?ay0 + (?j + 1) * ?dy) AS ?y1)
    ?episode cp:hasIndex ?ei ; cp:succeeded ?ok ; cp:hasTaskObject ?obj .
    ?ei cp:indexValue ?epi .
    ?obj cp:hasName ?objName ; cp:hasPosition ?p .
    ?objName cp:nameValue ?name .
    ?p cp:positionX ?x ; cp:positionY ?y .
    FILTER (?x >= ?x0 && (?x < ?x1 || (?i = ?nw - 1 && ?x = ?x1)) &&
            ?y >= ?y0 && (?y < ?y1 || (?j = ?nh - 1 && ?y = ?y1)))
}
"""

OBJECT_POSITION_AXIS = "object_position"


@dataclass
class StudentCell:
    """One cell of the student's requirement grid that a rollout cell overlaps."""

    object_name: str
    i: int
    j: int
    x_range: tuple[float, float]
    y_range: tuple[float, float]
    episodes: int
    min_episodes: int

    @property
    def needed(self) -> int:
        return max(0, self.min_episodes - self.episodes)


@dataclass
class RolloutCell:
    i: int
    j: int
    x_range: tuple[float, float]
    y_range: tuple[float, float]
    runs: int = 0
    successes: int = 0
    covered: bool = False
    outside_area: bool = False
    overlapped: list[StudentCell] = field(default_factory=list)

    @property
    def sampled(self) -> bool:
        return self.runs > 0

    @property
    def short_cells(self) -> list[StudentCell]:
        return [c for c in self.overlapped if c.needed > 0]


@dataclass
class ObjectComparison:
    object_name: str
    x_range: tuple[float, float]
    y_range: tuple[float, float]
    has_requirement: bool
    cell_size: tuple[float, float] | None = None
    requirement_area: tuple[float, float, float, float] | None = None
    cells: list[list[RolloutCell]] = field(default_factory=list)
    """cells[j][i]: column i (x), row j (y), row-major index k = j * numCellW + i."""
    runs: int = 0
    runs_covered: int = 0
    successes_covered: int = 0
    successes_not_covered: int = 0
    runs_outside_range: int = 0

    @property
    def sampled_cells(self) -> int:
        return sum(c.sampled for row in self.cells for c in row)

    @property
    def covered_cells(self) -> int:
        return sum(c.sampled and c.covered for row in self.cells for c in row)


@dataclass
class AxisComparison:
    axis: str
    objects: list[ObjectComparison]
    run_covered: dict[int, bool]
    """Rollout episode index -> whether its cell(s) on this axis are covered."""


@dataclass
class FillTarget:
    """A student cell to record more episodes in."""

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
    """Grow the requirement area by whole cells so it contains the sampled rollout cells."""

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
    rollout_range: str = ""
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
    record_cfg: str = ""
    pose_ranges: dict[str, dict[str, tuple[float, float]]] = field(default_factory=dict)

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

    def column(self, i: int) -> tuple[Decimal, Decimal]:
        return _cell_bounds(self.x0, self.x1, self.nw, i)

    def row(self, j: int) -> tuple[Decimal, Decimal]:
        return _cell_bounds(self.y0, self.y1, self.nh, j)

    def student_cell(self, i: int, j: int) -> StudentCell:
        return StudentCell(
            self.name, i, j, _floats(self.column(i)), _floats(self.row(j)), self.counts[j * self.nw + i], self.min_episodes
        )


def compare_rollout(
    dataset_root: str | Path,
    spec: str | Path,
    rollout_log: str | Path,
    rollout_range: str | Path | None = None,
    record_cfg: str | Path | None = None,
    graph_out: str | Path | None = None,
) -> RolloutComparison:
    """Compare the finished rollout in ``rollout_log`` with ``dataset_root``'s coverage under ``spec``.

    ``rollout_range`` defaults to the generated range file of the evaluation task
    file the rollout ran (``outputs/rollout_ranges/<eval file stem>.ttl``).
    ``record_cfg`` (a recording task cfg) adds suggested pose ranges, as offsets
    from that cfg's init_state.pos, that cover the cells to record in.
    """
    result = RolloutComparison(dataset=str(dataset_root), spec=str(spec), rollout_log=str(rollout_log))
    try:
        log = read_rollout_log(rollout_log)
    except RolloutLogError as exc:
        result.errors.append(str(exc))
        return result
    result.eval_file = str(log.header.get("eval_file", ""))
    result.seed = log.header.get("seed")
    result.aborted = log.aborted
    result.runs = len(log.episodes)
    result.successes = sum(e.succeeded for e in log.episodes)
    if not log.episodes:
        result.errors.append(f"{rollout_log}: no finished episode (all aborted or none run)")
        return result

    range_path = Path(rollout_range) if rollout_range is not None else default_range_path(result.eval_file)
    result.rollout_range = str(range_path)
    range_graph = _load_range_graph(range_path, log, result)
    if range_graph is None:
        return result

    graph, dataset_report = build_verification_graph(dataset_root, spec)
    result.errors.extend(dataset_report.errors)
    if result.errors:
        return result
    graph += range_graph
    requirements = _requirements(graph, result)
    if result.errors:
        return result
    add_rollout(graph, log)
    _add_rollout_grids(graph, requirements)

    _validate_rollout(graph, result)
    if graph_out is not None:
        Path(graph_out).parent.mkdir(parents=True, exist_ok=True)
        graph.serialize(destination=Path(graph_out), format="turtle")
    if result.errors:
        return result

    for comparer in ROLLOUT_AXES:
        result.axes.append(comparer(graph, result, requirements))
    in_distribution = [
        e for e in log.episodes if all(axis.run_covered.get(e.index, False) for axis in result.axes)
    ]
    result.runs_in_distribution = len(in_distribution)
    result.successes_in_distribution = sum(e.succeeded for e in in_distribution)
    _guidance(result, requirements, record_cfg)
    return result


def _load_range_graph(range_path: Path, log: RolloutLog, result: RolloutComparison) -> Graph | None:
    eval_file = result.eval_file
    hint = regenerate_hint(eval_file or "<eval file>")
    if not range_path.is_file():
        result.errors.append(f"No rollout range file {range_path} for {eval_file or 'this rollout'}. Generate it: {hint}")
        return None
    graph = Graph()
    try:
        graph.parse(range_path, format="turtle")
    except Exception as exc:  # rdflib raises several parser error types
        result.errors.append(f"Rollout range file {range_path} is not valid Turtle: {exc}. Regenerate it: {hint}")
        return None
    shas = range_file_sha256(graph)
    if len(shas) != 1:
        result.errors.append(f"{range_path} must come from exactly one eval file (found {len(shas)} hashes). Regenerate it: {hint}")
        return None
    range_sha = shas.pop()
    if eval_file and Path(eval_file).is_file() and file_sha256(eval_file) != range_sha:
        result.errors.append(f"{eval_file} changed since {range_path} was generated. Regenerate it: {hint}")
        return None
    log_sha = log.header.get("eval_sha256")
    if log_sha != range_sha:
        result.errors.append(
            f"The rollout ran {eval_file or 'its eval file'} with SHA-256 {str(log_sha)[:12]}, but {range_path} was "
            f"generated from SHA-256 {range_sha[:12]}: the randomization ranges may differ. Rerun the rollout with the "
            f"current eval file, or regenerate the range file from the file the rollout used."
        )
        return None
    return graph


def _requirements(graph: Graph, result: RolloutComparison) -> dict[str, _Requirement]:
    counts: dict[str, Counter] = {}
    for row in graph.query(_CELL_COUNTS_QUERY, initNs={"cp": CP, "xsd": XSD}):
        counts.setdefault(str(row.req), Counter())[int(row.k)] = int(row.n)
    requirements: dict[str, _Requirement] = {}
    for row in graph.query(_REQUIREMENTS_QUERY, initNs={"cp": CP}):
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
    return requirements


def _rollout_ranges(graph: Graph) -> list[tuple]:
    rows = graph.query(_RANGES_QUERY, initNs={"cp": CP})
    return sorted(
        ((row.range, str(row.name), row.area, *(Decimal(str(v.toPython())) for v in (row.x0, row.x1, row.y0, row.y1))) for row in rows),
        key=lambda r: r[1],
    )


def _add_rollout_grids(graph: Graph, requirements: dict[str, _Requirement]) -> None:
    for _, name, area, x0, x1, y0, y1 in _rollout_ranges(graph):
        req = requirements.get(name)
        if req is None:
            continue  # reported as "no requirement" by the axis comparison
        dx, dy = req.size
        add_rollout_grid(graph, area, dx, dy, _cell_count(x1 - x0, dx), _cell_count(y1 - y0, dy))


def _cell_count(length: Decimal, size: Decimal) -> int:
    # Quantize first: 0.2 / (0.2 / 3) is 3.000...01 in decimal arithmetic.
    ratio = (length / size).quantize(Decimal("1e-9"))
    return max(1, int(ratio.to_integral_value(rounding=ROUND_CEILING)))


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


def compare_object_position(graph: Graph, result: RolloutComparison, requirements: dict[str, _Requirement]) -> AxisComparison:
    runs_by_cell: dict[tuple[str, int], list[tuple[int, bool]]] = {}
    for row in graph.query(_RUN_CELLS_QUERY, initNs={"cp": CP, "xsd": XSD}):
        name = str(graph.value(graph.value(graph.value(row.range, CP.hasTaskObject), CP.hasName), CP.nameValue))
        runs_by_cell.setdefault((name, int(row.k)), []).append((int(row.epi), bool(row.ok.toPython())))
    episodes = {int(ep_index.toPython()): bool(ok.toPython()) for ep_index, ok in _rollout_episodes(graph)}

    run_covered = {index: True for index in episodes}
    objects = []
    for _, name, _, rx0, rx1, ry0, ry1 in _rollout_ranges(graph):
        obj = ObjectComparison(name, _floats((rx0, rx1)), _floats((ry0, ry1)), name in requirements, runs=len(episodes))
        objects.append(obj)
        req = requirements.get(name)
        if req is None:
            for index in run_covered:
                run_covered[index] = False
            obj.successes_not_covered = sum(episodes.values())
            result.warnings.append(
                Finding(
                    "Warning",
                    f"'{name}' is randomized by the rollout but the spec has no requirement for it; "
                    f"all {len(episodes)} run(s) count as not covered",
                    [name],
                )
            )
            continue
        dx, dy = req.size
        obj.cell_size = _floats((dx, dy))
        obj.requirement_area = _floats((req.x0, req.x1, req.y0, req.y1))
        nw, nh = _cell_count(rx1 - rx0, dx), _cell_count(ry1 - ry0, dy)
        placed: set[int] = set()
        for j in range(nh):
            row_cells = []
            for i in range(nw):
                x0, x1 = rx0 + i * dx, (rx1 if i == nw - 1 else rx0 + (i + 1) * dx)
                y0, y1 = ry0 + j * dy, (ry1 if j == nh - 1 else ry0 + (j + 1) * dy)
                cell = RolloutCell(i, j, _floats((x0, x1)), _floats((y0, y1)))
                cell.outside_area = x0 < req.x0 or x1 > req.x1 or y0 < req.y0 or y1 > req.y1
                columns = [si for si in range(req.nw) if _overlaps(x0, x1, *req.column(si), si == req.nw - 1)]
                rows = [sj for sj in range(req.nh) if _overlaps(y0, y1, *req.row(sj), sj == req.nh - 1)]
                cell.overlapped = [req.student_cell(si, sj) for sj in rows for si in columns]
                cell.covered = not cell.outside_area and not cell.short_cells
                for index, ok in runs_by_cell.get((name, j * nw + i), []):
                    placed.add(index)
                    cell.runs += 1
                    cell.successes += ok
                    if cell.covered:
                        obj.runs_covered += 1
                        obj.successes_covered += ok
                    else:
                        obj.successes_not_covered += ok
                        run_covered[index] = False
                row_cells.append(cell)
            obj.cells.append(row_cells)
        for index in set(episodes) - placed:  # started outside the rollout range
            obj.runs_outside_range += 1
            obj.successes_not_covered += episodes[index]
            run_covered[index] = False
        if obj.sampled_cells > obj.covered_cells:
            not_covered = [c for row in obj.cells for c in row if c.sampled and not c.covered]
            runs = sum(c.runs for c in not_covered)
            failed = runs - sum(c.successes for c in not_covered)
            result.warnings.append(
                Finding(
                    "Warning",
                    f"'{name}': {len(not_covered)} of {obj.sampled_cells} sampled rollout cell(s) are not covered by "
                    f"the dataset ({runs} run(s), {failed} failed)",
                    [f"{name}/rollout_cell/{c.j * nw + c.i}" for c in not_covered],
                )
            )
    return AxisComparison(OBJECT_POSITION_AXIS, objects, run_covered)


# One comparer per generalization axis: (graph, result, requirements) -> AxisComparison.
# The graph holds the dataset, spec, rollout and rollout ranges. Groups that add a
# factor append their comparer here; a run must be covered on every axis.
ROLLOUT_AXES: list[Callable[..., AxisComparison]] = [compare_object_position]


def _rollout_episodes(graph: Graph):
    query = """
        SELECT ?i ?ok WHERE {
            ?rollout a cp:Rollout ; cp:hasEpisode ?e .
            ?e cp:hasIndex ?index ; cp:succeeded ?ok . ?index cp:indexValue ?i .
        }
    """
    return graph.query(query, initNs={"cp": CP})


def _guidance(result: RolloutComparison, requirements: dict[str, _Requirement], record_cfg: str | Path | None) -> None:
    targets: dict[tuple[str, int, int], FillTarget] = {}
    widen_boxes: dict[str, list[tuple[float, float, float, float]]] = {}
    for axis in result.axes:
        if axis.axis != OBJECT_POSITION_AXIS:
            continue
        for obj in axis.objects:
            for cell in (c for row in obj.cells for c in row if c.sampled and not c.covered):
                failed = cell.runs - cell.successes
                for sc in cell.short_cells:
                    key = (sc.object_name, sc.i, sc.j)
                    target = targets.setdefault(
                        key, FillTarget(sc.object_name, sc.i, sc.j, sc.x_range, sc.y_range, sc.episodes, sc.needed, 0, 0)
                    )
                    target.runs += cell.runs
                    target.failed += failed
                if cell.outside_area:
                    widen_boxes.setdefault(obj.object_name, []).append((*cell.x_range, *cell.y_range))
    result.to_fill = sorted(targets.values(), key=lambda t: (-t.failed, -t.needed, t.object_name, t.j, t.i))

    for name, boxes in sorted(widen_boxes.items()):
        req = requirements[name]
        dx, dy = req.size
        bx0, by0 = min(b[0] for b in boxes), min(b[2] for b in boxes)
        bx1, by1 = max(b[1] for b in boxes), max(b[3] for b in boxes)
        # Grow by whole cells so the cell size, and with it the rollout grid, stays the same.
        left = _cell_count_or_zero(req.x0 - Decimal(str(bx0)), dx)
        right = _cell_count_or_zero(Decimal(str(bx1)) - req.x1, dx)
        down = _cell_count_or_zero(req.y0 - Decimal(str(by0)), dy)
        up = _cell_count_or_zero(Decimal(str(by1)) - req.y1, dy)
        result.widen.append(
            AreaSuggestion(
                name,
                _floats((req.x0, req.x1, req.y0, req.y1)),
                _floats((req.x0 - left * dx, req.x1 + right * dx, req.y0 - down * dy, req.y1 + up * dy)),
                req.nw + left + right,
                req.nh + down + up,
            )
        )

    if record_cfg is None:
        return
    result.record_cfg = str(record_cfg)
    try:
        init = read_task_ranges(record_cfg, require_ranges=False).init_positions
    except RolloutRangeError as exc:
        result.errors.append(str(exc))
        return
    boxes: dict[str, list[tuple[float, float, float, float]]] = {}
    for t in result.to_fill:
        boxes.setdefault(t.object_name, []).append((*t.x_range, *t.y_range))
    for name, extra in widen_boxes.items():
        boxes.setdefault(name, []).extend(extra)
    for name, bs in sorted(boxes.items()):
        if name not in init:
            result.warnings.append(Finding("Warning", f"{record_cfg} has no init_state.pos for '{name}'; no pose range suggested", [name]))
            continue
        cx, cy = init[name][:2]
        result.pose_ranges[name] = {
            "x": (round(min(b[0] for b in bs) - cx, 4), round(max(b[1] for b in bs) - cx, 4)),
            "y": (round(min(b[2] for b in bs) - cy, 4), round(max(b[3] for b in bs) - cy, 4)),
        }


def _cell_count_or_zero(length: Decimal, size: Decimal) -> int:
    return 0 if length <= 0 else _cell_count(length, size)


def _cell_bounds(lo: Decimal, hi: Decimal, n: int, i: int) -> tuple[Decimal, Decimal]:
    return lo + i * (hi - lo) / n, (hi if i == n - 1 else lo + (i + 1) * (hi - lo) / n)


def _overlaps(a0: Decimal, a1: Decimal, b0: Decimal, b1: Decimal, b_last: bool) -> bool:
    """Whether [a0, a1] shares a positive length with student cell [b0, b1); a zero-width interval uses cell membership."""
    if a0 < a1:
        return max(a0, b0) < min(a1, b1)
    return b0 <= a0 and (a0 < b1 or (b_last and a0 == b1))


def _floats(values) -> tuple[float, ...]:
    return tuple(float(v) for v in values)


def format_comparison(result: RolloutComparison, max_nodes: int = 3) -> str:
    lines = [
        f"Dataset : {result.dataset}",
        f"Spec    : {result.spec}",
        f"Rollout : {result.rollout_log}" + (f"  ({result.eval_file}, seed {result.seed})" if result.eval_file else ""),
        f"Range   : {result.rollout_range}",
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
                f"has {t.episodes}, needs {t.needed} more; {t.runs} rollout run(s) nearby, {t.failed} failed"
            )
        lines.append("")
    if result.widen:
        lines.append("Rollout reached outside your area. Grow it by whole cells (same cell size), then record there:")
        for w in result.widen:
            lines.append(
                f"  {w.object_name}: area x=[{w.current[0]:.3f}, {w.current[1]:.3f}] y=[{w.current[2]:.3f}, {w.current[3]:.3f}]"
                f" -> areaXMin {w.suggested[0]:.6g} ; areaXMax {w.suggested[1]:.6g} ; areaYMin {w.suggested[2]:.6g} ;"
                f" areaYMax {w.suggested[3]:.6g} ; numCellW {w.num_cell_w} ; numCellH {w.num_cell_h}"
            )
        lines.append("")
    if result.pose_ranges:
        lines.append(f"Pose ranges covering those cells, as offsets from init_state.pos in {result.record_cfg}:")
        for name, r in result.pose_ranges.items():
            lines.append(f'  "{name}": {{"x": {r["x"]}, "y": {r["y"]}, "z": (0.0, 0.0)}},')
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
    head = f"[{obj.object_name}] rollout x={obj.x_range[0]:.3f}..{obj.x_range[1]:.3f} y={obj.y_range[0]:.3f}..{obj.y_range[1]:.3f}"
    if not obj.has_requirement:
        return [head + "  (no requirement in the spec)", ""]
    nh, nw = len(obj.cells), len(obj.cells[0])
    lines = [
        head + f"  cell {obj.cell_size[0]:.4g} x {obj.cell_size[1]:.4g} (your grid), {nw}x{nh} cells",
        "  successes/runs per rollout cell; ! = not covered by your dataset, - = no runs",
    ]
    texts = [[("-" if not c.sampled else f"{c.successes}/{c.runs}") + ("" if c.covered or not c.sampled else "!") for c in row] for row in obj.cells]
    width = max(4, *(len(t) for row in texts for t in row))
    for j in reversed(range(nh)):  # +y at the top
        lines.append(f"  y{j:<2} " + " ".join(f"{t:>{width}}" for t in texts[j]))
    lines.append("       " + " ".join(f"{'x' + str(i):>{width}}" for i in range(nw)))
    not_covered = obj.runs - obj.runs_covered
    lines.append(
        f"  covered: {obj.covered_cells}/{obj.sampled_cells} sampled cells, {obj.runs_covered}/{obj.runs} runs "
        f"({_pct(obj.runs_covered, obj.runs)})\n  success: {_rate(obj.successes_covered, obj.runs_covered)} covered vs "
        f"{_rate(obj.successes_not_covered, not_covered)} not covered"
    )
    if obj.runs_outside_range:
        lines.append(f"  {obj.runs_outside_range} run(s) started outside the rollout range")
    lines.append("")
    return lines


def _pct(n: int, d: int) -> str:
    return f"{100 * n / d:.0f}%" if d else "-"


def _rate(n: int, d: int) -> str:
    return f"{n}/{d}" if d else "0/0"
