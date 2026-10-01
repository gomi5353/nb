"""Object pose coverage verification: build the graph, run SHACL, summarize."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
from importlib.resources import files
from pathlib import Path

from pyshacl import validate
from rdflib import RDF, Graph, Namespace

from .dataset import DatasetError, read_initial_placements
from .graph import DV, add_dataset, expand_cells
from .spec_template import todo_fields, value_type_errors

SH = Namespace("http://www.w3.org/ns/shacl#")

# Same membership rule as dv:CoverageCellShape, without the HAVING filter.
_CELL_COUNTS_QUERY = """
SELECT ?name ?i ?j ?min (COUNT(DISTINCT ?p) AS ?n) WHERE {
    ?cell a dv:CoverageCell ; dv:ofRequirement ?req ; dv:cellI ?i ; dv:cellJ ?j ;
          dv:xMin ?x0 ; dv:xMax ?x1 ; dv:yMin ?y0 ; dv:yMax ?y1 ;
          dv:xMaxInclusive ?xi ; dv:yMaxInclusive ?yi .
    ?req dv:forObject ?obj ; dv:minEpisodesPerCell ?min .
    ?obj dv:objectName ?name .
    OPTIONAL {
        ?p dv:ofObject ?obj ; dv:x ?x ; dv:y ?y .
        FILTER (?x >= ?x0 && (?x < ?x1 || (?xi && ?x = ?x1)) &&
                ?y >= ?y0 && (?y < ?y1 || (?yi && ?y = ?y1)))
    }
}
GROUP BY ?name ?i ?j ?min
"""

_REQUIREMENTS_QUERY = """
SELECT ?name ?x0 ?x1 ?y0 ?y1 ?nx ?ny ?min WHERE {
    ?req a dv:CoverageRequirement ; dv:forObject ?obj ;
         dv:xMin ?x0 ; dv:xMax ?x1 ; dv:yMin ?y0 ; dv:yMax ?y1 ;
         dv:gridX ?nx ; dv:gridY ?ny ; dv:minEpisodesPerCell ?min .
    ?obj dv:objectName ?name .
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
    graph.bind("dv", DV)
    todo = todo_fields(graph)
    if todo:
        report.errors.append("Spec still has TODO values: " + ", ".join(todo))
        return graph, report
    type_errors = value_type_errors(graph)
    if type_errors:
        report.errors.extend(type_errors)
        return graph, report
    if not any(graph.subjects(RDF.type, DV.CoverageRequirement)):
        report.errors.append("Spec declares no dv:CoverageRequirement; nothing to check")
    expand_cells(graph)

    try:
        data = read_initial_placements(root)
    except DatasetError as exc:
        report.errors.append(str(exc))
        return graph, report

    report.num_episodes = len(data.episode_indices)
    if not data.episode_indices:
        report.errors.append("Dataset contains no episodes")
    required = {str(graph.value(obj, DV.objectName)) for obj in graph.objects(None, DV.forObject)}
    for name in sorted(required - set(data.object_names)):
        report.errors.append(f"Dataset has no 'object_pose.{name}' column, which the spec requires")
    bad = [p for p in data.placements if not p.is_finite]
    for p in bad[:10]:
        report.errors.append(f"Episode {p.episode_index}: non-finite initial pose for '{p.object_name}'")
    if len(bad) > 10:
        report.errors.append(f"... {len(bad) - 10} more non-finite initial poses")
    late = {e: f for e, f in data.first_frame_index.items() if f != 0}
    if late:
        sample = ", ".join(f"{e} (frame {f})" for e, f in list(late.items())[:5])
        report.errors.append(f"{len(late)} episode(s) do not start at frame_index 0: {sample}")

    add_dataset(graph, root.resolve().name, data)
    return graph, report


def verify_object_pose_coverage(
    dataset_root: str | Path, spec: str | Path, extra_shapes: list[str | Path] | None = None
) -> CoverageReport:
    """Check ``dataset_root`` against ``spec``; ``extra_shapes`` adds your own SHACL rules."""
    graph, report = build_verification_graph(dataset_root, spec)
    if len(graph) == 0 or todo_fields(graph) or value_type_errors(graph):
        return report  # spec missing or not filled in; SHACL would only add noise

    shapes = _packaged_graph("shapes", "object_pose_coverage.ttl")
    for path in extra_shapes or []:
        try:
            shapes.parse(Path(path), format="turtle")
        except Exception as exc:
            report.errors.append(f"Extra shapes file {path} could not be loaded: {exc}")
            return report
    _, results_graph, _ = validate(
        data_graph=graph,
        shacl_graph=shapes,
        ont_graph=_packaged_graph("ontology", "object_pose.ttl"),
        inference="none",
        abort_on_first=False,
    )
    grouped: dict[tuple[str, str], list[str]] = {}
    for result in results_graph.subjects(RDF.type, SH.ValidationResult):
        severity = str(results_graph.value(result, SH.resultSeverity)).rsplit("#", 1)[-1]
        message = str(results_graph.value(result, SH.resultMessage) or "")
        path = results_graph.value(result, SH.resultPath)
        if path is not None and not message.startswith("'"):
            message = f"{path.n3(graph.namespace_manager)}: {message}"
        grouped.setdefault((severity, message), []).append(str(results_graph.value(result, SH.focusNode)))
    for (severity, message), nodes in sorted(grouped.items()):
        finding = Finding(severity, message, sorted(nodes))
        (report.violations if severity == "Violation" else report.warnings).append(finding)

    report.coverage = _coverage_tables(graph)
    return report


def _coverage_tables(graph: Graph) -> list[ObjectCoverage]:
    counts: dict[str, Counter] = {}
    for row in graph.query(_CELL_COUNTS_QUERY, initNs={"dv": DV}):
        counts.setdefault(str(row.name), Counter())[(int(row.i), int(row.j))] = int(row.n)
    tables = []
    for row in graph.query(_REQUIREMENTS_QUERY, initNs={"dv": DV}):
        name, nx, ny = str(row.name), int(row.nx), int(row.ny)
        cell_counts = counts.get(name, Counter())
        tables.append(
            ObjectCoverage(
                object_name=name,
                x_range=(float(row.x0), float(row.x1)),
                y_range=(float(row.y0), float(row.y1)),
                min_episodes_per_cell=int(row.min),
                counts=[[cell_counts[(i, j)] for i in range(nx)] for j in range(ny)],
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
    """'urn:aicapstone:data:ds/episode/3/placement/cup' -> 'episode/3/placement/cup'."""
    if uri.startswith("urn:aicapstone:data:"):
        return uri.split("/", 1)[-1]
    return uri.rsplit("#", 1)[-1]
