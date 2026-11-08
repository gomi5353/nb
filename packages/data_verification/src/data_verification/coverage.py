"""Object position coverage verification: build the graph, run SHACL, summarize."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import asdict, dataclass, field
from decimal import Decimal
from importlib.resources import files
from pathlib import Path

from pyshacl import validate
from rdflib import RDF, Graph, Namespace, URIRef

from .dataset import DatasetError, read_initial_placements
from .graph import CP, add_dataset, dataset_iri, expand_cells
from .spec_template import todo_fields, value_range_errors, value_type_errors

SH = Namespace("http://www.w3.org/ns/shacl#")

# The requirements of ?dataset (bound by the caller), through its axes.
_REQUIREMENTS_QUERY = """
SELECT ?req ?name ?x0 ?x1 ?y0 ?y1 ?nx ?ny ?min WHERE {
    ?dataset cp:hasGeneralizationAxis ?axis .
    ?axis cp:hasRequirement ?req .
    ?req a cp:ObjectPositionAxisRequirement ; cp:hasTaskObject ?reqObj ;
         cp:minEpisodesPerCell ?min ; cp:hasAreaConfiguration ?area .
    ?reqObj cp:name ?name .
    ?area cp:areaXMin ?x0 ; cp:areaXMax ?x1 ; cp:areaYMin ?y0 ; cp:areaYMax ?y1 ;
          cp:hasCoverageGrid ?grid .
    ?grid cp:numCellW ?nx ; cp:numCellH ?ny .
}
"""

_REQUIRED_NAMES_QUERY = """
SELECT DISTINCT ?name WHERE {
    ?req a cp:ObjectPositionAxisRequirement ; cp:hasTaskObject ?reqObj .
    ?reqObj cp:name ?name .
}
"""


@dataclass
class Finding:
    severity: str
    message: str
    focus_nodes: list[str]

    @property
    def count(self) -> int:
        return len(self.focus_nodes)


@dataclass
class ObjectCoverage:
    object_name: str
    x_range: tuple[float, float]
    y_range: tuple[float, float]
    min_episodes_per_cell: int
    counts: list[list[int]]
    """counts[j][i]: episodes in column i (x), row j (y)."""

    @property
    def cells_covered(self) -> int:
        return sum(n >= self.min_episodes_per_cell for row in self.counts for n in row)

    @property
    def cells_total(self) -> int:
        return sum(len(row) for row in self.counts)


@dataclass
class CoverageReport:
    dataset: str
    spec: str
    num_episodes: int = 0
    errors: list[str] = field(default_factory=list)
    violations: list[Finding] = field(default_factory=list)
    warnings: list[Finding] = field(default_factory=list)
    coverage: list[ObjectCoverage] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.errors and not self.violations

    def to_dict(self) -> dict:
        result = asdict(self)
        result["passed"] = self.passed
        return result


def load_spec(spec: str | Path) -> Graph:
    """Parse a coverage spec Turtle file."""
    graph = Graph()
    graph.parse(Path(spec), format="turtle")
    return graph


def _packaged_graph(*parts: str) -> Graph:
    graph = Graph()
    graph.parse(data=files("data_verification").joinpath(*parts).read_text(), format="turtle")
    return graph


def build_verification_graph(dataset_root: str | Path, spec: str | Path) -> tuple[Graph, CoverageReport]:
    """Spec, cells and the dataset with its episodes; the dataset node is ``dataset_iri(<dataset dir name>)``."""
    root = Path(dataset_root)
    report = CoverageReport(dataset=str(root), spec=str(spec))
    graph = Graph()
    try:
        graph += load_spec(spec)
    except FileNotFoundError:
        report.errors.append(f"Spec file not found: {spec}")
        return graph, report
    except Exception as exc:  # rdflib raises several parser error types
        report.errors.append(f"Spec is not valid Turtle: {exc}")
        return graph, report
    graph.bind("cp", CP)
    todo = todo_fields(graph)
    if todo:
        report.errors.append("Spec still has TODO values: " + ", ".join(todo))
        return graph, report
    value_errors = value_type_errors(graph) or value_range_errors(graph)
    if value_errors:
        report.errors.extend(value_errors)
        return graph, report
    if not any(graph.subjects(RDF.type, CP.ObjectPositionAxisRequirement)):
        report.errors.append("Spec declares no cp:ObjectPositionAxisRequirement; nothing to check")
    expand_cells(graph)

    try:
        data = read_initial_placements(root)
    except DatasetError as exc:
        report.errors.append(str(exc))
        return graph, report

    report.num_episodes = len(data.episode_indices)
    if not data.episode_indices:
        report.errors.append("Dataset contains no episodes")
    required = {str(row.name) for row in graph.query(_REQUIRED_NAMES_QUERY, initNs={"cp": CP})}
    for name in sorted(required - set(data.object_names)):
        report.errors.append(f"Dataset has no 'object_position.{name}' column, which the spec requires")
    bad = [p for p in data.placements if not p.is_finite]
    for p in bad[:10]:
        report.errors.append(f"Episode {p.episode_index}: non-finite initial position for '{p.object_name}'")
    if len(bad) > 10:
        report.errors.append(f"... {len(bad) - 10} more non-finite initial positions")
    late = {e: f for e, f in data.first_frame_index.items() if f != 0}
    if late:
        sample = ", ".join(f"{e} (frame {f})" for e, f in list(late.items())[:5])
        report.errors.append(f"{len(late)} episode(s) do not start at frame_index 0: {sample}")

    add_dataset(graph, root.resolve().name, data)
    return graph, report


def verify_object_position_coverage(
    dataset_root: str | Path,
    spec: str | Path,
    extra_shapes: list[str | Path] | None = None,
    graph_out: str | Path | None = None,
) -> CoverageReport:
    """Check ``dataset_root`` against ``spec``; ``extra_shapes`` adds your own SHACL rules.

    ``graph_out`` also writes the data graph that SHACL validates (spec, generated
    cells, episodes, task objects and their initial positions) as Turtle, for
    inspection or SPARQL.
    """
    graph, report = build_verification_graph(dataset_root, spec)
    if graph_out is not None:  # also when empty, so an old file never looks current
        Path(graph_out).parent.mkdir(parents=True, exist_ok=True)
        graph.serialize(destination=Path(graph_out), format="turtle")
    if len(graph) == 0 or todo_fields(graph) or value_type_errors(graph) or value_range_errors(graph):
        return report  # spec missing or not filled in; SHACL would only add noise

    shapes = _packaged_graph("shapes", "object_position_coverage.ttl")
    for path in extra_shapes or []:
        try:
            shapes.parse(Path(path), format="turtle")
        except Exception as exc:
            report.errors.append(f"Extra shapes file {path} could not be loaded: {exc}")
            return report
    _, results_graph, _ = validate(
        data_graph=graph,
        shacl_graph=shapes,
        ont_graph=_packaged_graph("ontology", "object_position.ttl"),
        inference="none",
        abort_on_first=False,
    )
    grouped: dict[tuple[str, str], list[str]] = {}
    for result in results_graph.subjects(RDF.type, SH.ValidationResult):
        severity = str(results_graph.value(result, SH.resultSeverity)).rsplit("#", 1)[-1]
        message = str(results_graph.value(result, SH.resultMessage) or "")
        path = results_graph.value(result, SH.resultPath)
        if isinstance(path, URIRef) and not message.startswith("'"):  # not for inverse paths
            message = f"{path.n3(graph.namespace_manager)}: {message}"
        grouped.setdefault((severity, message), []).append(str(results_graph.value(result, SH.focusNode)))
    for (severity, message), nodes in sorted(grouped.items()):
        finding = Finding(severity, message, sorted(nodes))
        (report.violations if severity == "Violation" else report.warnings).append(finding)

    report.coverage = _coverage_tables(graph, dataset_iri(Path(dataset_root).resolve().name))
    return report


def cell_counts(graph: Graph, dataset: URIRef) -> dict[str, Counter]:
    """Requirement IRI -> Counter of ``dataset``'s episodes per cell index.

    Same cell bounds and membership rule as cp:DatasetCoverageShape, counted in
    Python: the SPARQL form takes tens of seconds on a 100-episode dataset.
    """
    positions: dict[str, list[tuple[Decimal, Decimal]]] = {}
    for episode in graph.objects(dataset, CP.hasEpisode):
        for obj in graph.objects(episode, CP.hasTaskObject):
            p = graph.value(obj, CP.hasPosition)
            x, y = graph.value(p, CP.positionX), graph.value(p, CP.positionY)
            if x is not None and y is not None:
                positions.setdefault(str(graph.value(obj, CP.name)), []).append((_dec(x), _dec(y)))
    counts: dict[str, Counter] = {}
    for row in requirement_rows(graph, dataset):
        x0, x1, y0, y1 = (_dec(v) for v in (row.x0, row.x1, row.y0, row.y1))
        nw, nh = int(row.nx), int(row.ny)
        counter = counts.setdefault(str(row.req), Counter())
        if not (x0 < x1 and y0 < y1 and nw >= 1 and nh >= 1):
            continue
        for x, y in positions.get(str(row.name), []):
            i, j = lattice_index(x, x0, x1, nw), lattice_index(y, y0, y1, nh)
            if 0 <= i < nw and 0 <= j < nh:
                counter[j * nw + i] += 1
    return counts


def cell_edge(lo: Decimal, hi: Decimal, n: int, i: int) -> Decimal:
    """The i-th cell edge of [lo, hi] split into n, as cp:DatasetCoverageShape computes it; edge n is exactly hi."""
    return hi if i == n else lo + i * (hi - lo) / n


def lattice_index(v: Decimal, lo: Decimal, hi: Decimal, n: int) -> int:
    """Index i of the interval [edge i, edge i+1) holding v, extended beyond [lo, hi]; hi belongs to cell n - 1."""
    if v == hi:
        return n - 1
    i = math.floor((v - lo) * n / (hi - lo))
    while v < cell_edge(lo, hi, n, i):  # guard against rounding at an edge
        i -= 1
    while v >= cell_edge(lo, hi, n, i + 1):
        i += 1
    return i


def _dec(literal) -> Decimal:
    return Decimal(str(literal.toPython()))


def requirement_rows(graph: Graph, dataset: URIRef):
    """One row per object position requirement of ``dataset``: req, name, area bounds, nx, ny, min."""
    return graph.query(_REQUIREMENTS_QUERY, initNs={"cp": CP}, initBindings={"dataset": dataset})


def _coverage_tables(graph: Graph, dataset: URIRef) -> list[ObjectCoverage]:
    counts = cell_counts(graph, dataset)
    tables = []
    for row in requirement_rows(graph, dataset):
        nx, ny = int(row.nx), int(row.ny)
        req_counts = counts.get(str(row.req), Counter())
        tables.append(
            ObjectCoverage(
                object_name=str(row.name),
                x_range=(float(row.x0), float(row.x1)),
                y_range=(float(row.y0), float(row.y1)),
                min_episodes_per_cell=int(row.min),
                # Cell index is row-major: k = j * nx + i.
                counts=[[req_counts[j * nx + i] for i in range(nx)] for j in range(ny)],
            )
        )
    return sorted(tables, key=lambda t: t.object_name)


def format_report(report: CoverageReport, max_nodes: int = 3) -> str:
    lines = [f"Dataset : {report.dataset}", f"Spec    : {report.spec}", f"Episodes: {report.num_episodes}", ""]
    for table in report.coverage:
        lines.append(
            f"[{table.object_name}] x={table.x_range[0]:.3f}..{table.x_range[1]:.3f}  "
            f"y={table.y_range[0]:.3f}..{table.y_range[1]:.3f}  "
            f"covered {table.cells_covered}/{table.cells_total} cells (>= {table.min_episodes_per_cell} each)"
        )
        width = max(3, *(len(str(n)) for row in table.counts for n in row))
        for j in reversed(range(len(table.counts))):  # +y at the top
            cells = " ".join(
                f"{n:>{width}}" if n >= table.min_episodes_per_cell else f"{str(n) + '!':>{width}}"
                for n in table.counts[j]
            )
            lines.append(f"  y{j:<2} {cells}")
        lines.append("       " + " ".join(f"{'x' + str(i):>{width}}" for i in range(len(table.counts[0]) if table.counts else 0)))
        lines.append("")
    for error in report.errors:
        lines.append(f"ERROR     {error}")
    for kind, findings in (("VIOLATION", report.violations), ("WARNING", report.warnings)):
        for f in findings:
            nodes = ", ".join(_short_node(n) for n in f.focus_nodes[:max_nodes])
            more = f" (+{f.count - max_nodes} more)" if f.count > max_nodes else ""
            lines.append(f"{kind:<9} {f.message}  [{nodes}{more}]")
    lines.append("")
    lines.append("RESULT: coverage spec met" if report.passed else "RESULT: coverage spec NOT met")
    return "\n".join(lines)


def _short_node(uri: str) -> str:
    """'http://taica.course/cp/data/ds/episode/3/object/cup' -> 'episode/3/object/cup'."""
    prefix = "http://taica.course/cp/data/"
    if uri.startswith(prefix):
        return uri[len(prefix) :].split("/", 1)[-1]
    return uri.rsplit("#", 1)[-1]
