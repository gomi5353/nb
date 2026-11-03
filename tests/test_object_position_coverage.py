"""Object position coverage verification on synthetic LeRobot v3 layouts (no simulator)."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from data_verification import format_report, render_spec, verify_object_position_coverage, write_spec
from data_verification.dataset import OBJECT_POSITION_KEY_PREFIX, OBJECT_POSITION_NAMES, read_initial_placements
from data_verification.spec_template import template_text

ROOT = Path(__file__).resolve().parents[1]

SPEC = """
@prefix cp:   <http://taica.course/cp/ontology#> .
@prefix spec: <http://taica.course/cp/spec/test#> .

spec:axis a cp:ObjectPositionAxis ; cp:hasRequirement spec:cube_requirement .
spec:cube a cp:TaskObject ; cp:hasName [ a cp:Name ; cp:nameValue "cube" ] .
spec:cube_requirement a cp:ObjectPositionAxisRequirement ;
    cp:hasTaskObject spec:cube ;
    cp:minEpisodesPerCell 1 ;
    cp:hasAreaConfiguration spec:cube_area .
spec:cube_area a cp:AreaConfiguration ;
    cp:areaXMin 0.0 ; cp:areaXMax 0.2 ; cp:areaYMin 0.0 ; cp:areaYMax 0.2 ;
    cp:hasCoverageGrid spec:cube_grid .
spec:cube_grid a cp:CoverageGrid ; cp:numCellW 2 ; cp:numCellH 2 .
"""

# One point per cell of the 2x2 grid over [0, 0.2]^2.
CELL_CENTERS = [(0.05, 0.05), (0.15, 0.05), (0.05, 0.15), (0.15, 0.15)]


@pytest.fixture
def spec(tmp_path: Path) -> Path:
    path = tmp_path / "spec.ttl"
    path.write_text(SPEC)
    return path


def write_dataset(root: Path, episodes: dict[int, dict[str, list[tuple[float, float]]]], frames: int = 3) -> Path:
    """episodes[episode][object] = xy per frame (padded with the last xy up to ``frames``)."""
    object_names = sorted({name for objs in episodes.values() for name in objs})
    features = {
        "observation.state": {"dtype": "float32", "shape": [9], "names": None},
        **{
            f"{OBJECT_POSITION_KEY_PREFIX}{name}": {"dtype": "float32", "shape": [3], "names": OBJECT_POSITION_NAMES}
            for name in object_names
        },
    }
    (root / "meta").mkdir(parents=True)
    (root / "meta" / "info.json").write_text(json.dumps({"codebase_version": "v3.0", "fps": 30, "features": features}))

    columns: dict[str, list] = {"episode_index": [], "frame_index": []}
    columns.update({f"{OBJECT_POSITION_KEY_PREFIX}{name}": [] for name in object_names})
    for episode, objs in episodes.items():
        for frame in range(frames):
            columns["episode_index"].append(episode)
            columns["frame_index"].append(frame)
            for name in object_names:
                xys = objs[name]
                x, y = xys[min(frame, len(xys) - 1)]
                columns[f"{OBJECT_POSITION_KEY_PREFIX}{name}"].append([x, y, 0.9])
    (root / "data" / "chunk-000").mkdir(parents=True)
    pq.write_table(pa.table(columns), root / "data" / "chunk-000" / "file-000.parquet")
    return root


def test_full_coverage_passes(tmp_path: Path, spec: Path) -> None:
    dataset = write_dataset(tmp_path / "ds", {i: {"cube": [xy]} for i, xy in enumerate(CELL_CENTERS)})
    report = verify_object_position_coverage(dataset, spec)
    assert report.passed, format_report(report)
    assert report.num_episodes == 4
    assert report.coverage[0].counts == [[1, 1], [1, 1]]


def test_empty_cell_is_a_violation(tmp_path: Path, spec: Path) -> None:
    dataset = write_dataset(tmp_path / "ds", {i: {"cube": [xy]} for i, xy in enumerate(CELL_CENTERS[:3])})
    report = verify_object_position_coverage(dataset, spec)
    assert not report.passed
    [finding] = report.violations
    assert finding.message == "'cube' cell (i=1, j=1) x=[0.1, 0.2] y=[0.1, 0.2] has 0 episode(s); needs at least 1"
    assert finding.focus_nodes == ["http://taica.course/cp/spec/test#cube_grid/cell/3"]
    assert report.coverage[0].cells_covered == 3


def test_initial_position_is_first_frame_not_later(tmp_path: Path, spec: Path) -> None:
    # Object starts in cell (1, 1) and is carried to cell (0, 0) afterwards.
    episodes = {i: {"cube": [xy]} for i, xy in enumerate(CELL_CENTERS[:3])}
    episodes[3] = {"cube": [(0.15, 0.15), (0.05, 0.05)]}
    report = verify_object_position_coverage(write_dataset(tmp_path / "ds", episodes), spec)
    assert report.passed, format_report(report)


def test_every_episode_counts_regardless_of_outcome(tmp_path: Path, spec: Path) -> None:
    # A 'success' bookkeeping column must not filter anything.
    dataset = write_dataset(tmp_path / "ds", {i: {"cube": [xy]} for i, xy in enumerate(CELL_CENTERS)})
    shard = dataset / "data" / "chunk-000" / "file-000.parquet"
    table = pq.read_table(shard)
    pq.write_table(table.append_column("success", pa.array([False] * table.num_rows)), shard)
    assert verify_object_position_coverage(dataset, spec).passed


def test_upper_edge_belongs_to_last_cell(tmp_path: Path, spec: Path) -> None:
    episodes = {i: {"cube": [xy]} for i, xy in enumerate(CELL_CENTERS[:3])}
    episodes[3] = {"cube": [(0.2, 0.2)]}
    report = verify_object_position_coverage(write_dataset(tmp_path / "ds", episodes), spec)
    assert report.passed, format_report(report)
    assert not report.warnings


def test_outside_area_is_warning_and_not_counted(tmp_path: Path, spec: Path) -> None:
    episodes = {i: {"cube": [xy]} for i, xy in enumerate(CELL_CENTERS)}
    episodes[4] = {"cube": [(0.5, 0.05)]}
    report = verify_object_position_coverage(write_dataset(tmp_path / "ds", episodes), spec)
    assert report.passed
    [warning] = report.warnings
    assert "outside the required area" in warning.message
    assert sum(map(sum, report.coverage[0].counts)) == 4


def test_missing_required_object_column(tmp_path: Path, spec: Path) -> None:
    dataset = write_dataset(tmp_path / "ds", {0: {"other": [(0.05, 0.05)]}})
    report = verify_object_position_coverage(dataset, spec)
    assert not report.passed
    assert any("object_position.cube" in e for e in report.errors)
    assert any("no initial position for required object 'cube'" in v.message for v in report.violations)


def test_dataset_without_position_columns(tmp_path: Path, spec: Path) -> None:
    root = tmp_path / "ds"
    (root / "meta").mkdir(parents=True)
    (root / "meta" / "info.json").write_text(json.dumps({"features": {"action": {"dtype": "float32", "shape": [8]}}}))
    report = verify_object_position_coverage(root, spec)
    assert not report.passed
    assert "declares no 'object_position.<name>' features" in report.errors[0]


def test_non_finite_position_is_error(tmp_path: Path, spec: Path) -> None:
    episodes = {i: {"cube": [xy]} for i, xy in enumerate(CELL_CENTERS)}
    episodes[4] = {"cube": [(float("nan"), 0.05)]}
    report = verify_object_position_coverage(write_dataset(tmp_path / "ds", episodes), spec)
    assert not report.passed
    assert any("non-finite" in e for e in report.errors)


def _fill(spec_text: str, values: dict[str, str]) -> str:
    for field, value in values.items():
        spec_text = spec_text.replace(f'{field} "TODO"', f"{field} {value}")
    return spec_text


FILLED = {
    "cp:areaXMin": "0.0", "cp:areaXMax": "0.2", "cp:areaYMin": "0.0", "cp:areaYMax": "0.2",
    "cp:numCellW": "2", "cp:numCellH": "2", "cp:minEpisodesPerCell": "1",
}


def test_raw_template_parses_and_asks_for_todos(tmp_path: Path) -> None:
    spec = tmp_path / "spec.ttl"
    spec.write_text(template_text())
    dataset = write_dataset(tmp_path / "ds", {0: {"cube": [(0.05, 0.05)]}})
    report = verify_object_position_coverage(dataset, spec)
    assert not report.passed
    assert report.errors[0].startswith("Spec still has TODO values: OBJECT_NAME: areaXMax")
    assert not report.violations  # SHACL skipped until the spec is filled in


def test_init_from_dataset_then_fill_then_check(tmp_path: Path) -> None:
    dataset = write_dataset(
        tmp_path / "ds", {i: {"cube": [xy], "lid": [(1.0, 1.0)]} for i, xy in enumerate(CELL_CENTERS)}
    )
    spec = write_spec(tmp_path / "spec.ttl", ["cube", "lid"], read_initial_placements(dataset), "ds")
    text = spec.read_text()
    assert text.count("# --- BEGIN OBJECT ---") == 2
    assert "Already in ds: 4 episode(s), initial x 0.050..0.150, y 0.050..0.150" in text

    # Check only the cube: drop the lid block, fill the cube block.
    head, cube, lid = text.split("# --- BEGIN OBJECT ---")
    spec.write_text(_fill(head + "# --- BEGIN OBJECT ---" + cube, FILLED))
    report = verify_object_position_coverage(dataset, spec)
    assert report.passed, format_report(report)
    assert [t.object_name for t in report.coverage] == ["cube"]


def test_init_refuses_to_overwrite(tmp_path: Path) -> None:
    spec = tmp_path / "spec.ttl"
    spec.write_text("# mine")
    with pytest.raises(FileExistsError):
        write_spec(spec, ["cube"])
    assert spec.read_text() == "# mine"


def test_quoted_numbers_are_rejected(tmp_path: Path) -> None:
    spec = tmp_path / "spec.ttl"
    spec.write_text(_fill(render_spec(["cube"]), {**FILLED, "cp:areaXMin": '"0.0"', "cp:numCellW": "2.5"}))
    dataset = write_dataset(tmp_path / "ds", {0: {"cube": [(0.05, 0.05)]}})
    report = verify_object_position_coverage(dataset, spec)
    assert report.errors == [
        'cube: areaXMin must be a number without quotes, got "0.0"',
        "cube: numCellW must be a whole number without quotes, got 2.5",
    ]


def test_invalid_turtle_is_reported(tmp_path: Path) -> None:
    spec = tmp_path / "spec.ttl"
    spec.write_text("this is not turtle")
    dataset = write_dataset(tmp_path / "ds", {0: {"cube": [(0.05, 0.05)]}})
    report = verify_object_position_coverage(dataset, spec)
    assert report.errors[0].startswith("Spec is not valid Turtle")


def test_empty_area_is_reported(tmp_path: Path) -> None:
    spec = tmp_path / "spec.ttl"
    spec.write_text(_fill(render_spec(["cube"]), {**FILLED, "cp:areaXMin": "0.3"}))
    dataset = write_dataset(tmp_path / "ds", {0: {"cube": [(0.05, 0.05)]}})
    report = verify_object_position_coverage(dataset, spec)
    # A spec error, reported before any cell is counted (no grid, no per-cell violations).
    assert report.errors == [
        "cube: coverage area is empty: areaXMin must be < areaXMax and areaYMin < areaYMax, got x 0.3..0.2, y 0.0..0.2"
    ]
    assert not report.violations and not report.coverage


def test_zero_cells_is_reported(tmp_path: Path) -> None:
    spec = tmp_path / "spec.ttl"
    spec.write_text(_fill(render_spec(["cube"]), {**FILLED, "cp:numCellW": "0"}))
    dataset = write_dataset(tmp_path / "ds", {0: {"cube": [(0.05, 0.05)]}})
    report = verify_object_position_coverage(dataset, spec)
    assert report.errors == ["cube: numCellW must be at least 1, got 0"]


def _module_constants(path: Path) -> dict[str, object]:
    tree = ast.parse(path.read_text())
    return {
        node.targets[0].id: ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id.startswith("OBJECT_POSITION_")
    }


def test_recorder_and_verifier_agree_on_column_format() -> None:
    recorder = _module_constants(ROOT / "packages/simulator/src/simulator/utils/lerobot_recorder.py")
    assert recorder == {"OBJECT_POSITION_KEY_PREFIX": OBJECT_POSITION_KEY_PREFIX, "OBJECT_POSITION_NAMES": OBJECT_POSITION_NAMES}


def _doc_extra_shape_example() -> str:
    doc = (ROOT / "docs" / "object_position_coverage.md").read_text()
    return doc.split("```turtle\n@prefix cp:", 1)[1].split("```", 1)[0].join(["@prefix cp:", ""])


def test_extra_shapes_example_from_docs(tmp_path: Path, spec: Path) -> None:
    dataset = write_dataset(tmp_path / "ds", {i: {"cube": [xy]} for i, xy in enumerate(CELL_CENTERS)})
    rules = tmp_path / "rules.ttl"
    rules.write_text(_doc_extra_shape_example())
    assert verify_object_position_coverage(dataset, spec, extra_shapes=[rules]).passed

    rules.write_text(_doc_extra_shape_example().replace('"0.0"', '"0.1"'))
    report = verify_object_position_coverage(dataset, spec, extra_shapes=[rules])
    assert [(v.message, v.count) for v in report.violations] == [("cp:positionY: Object starts behind the robot", 2)]


def test_graph_out_writes_positions_and_cells(tmp_path: Path, spec: Path) -> None:
    from rdflib import RDF, Graph

    from data_verification.graph import CP

    dataset = write_dataset(tmp_path / "ds", {i: {"cube": [xy]} for i, xy in enumerate(CELL_CENTERS)})
    out = tmp_path / "out" / "graph.ttl"
    report = verify_object_position_coverage(dataset, spec, graph_out=out)
    assert report.passed, format_report(report)
    graph = Graph().parse(out, format="turtle")
    assert len(list(graph.subjects(RDF.type, CP.Episode))) == 4
    assert len(list(graph.subjects(RDF.type, CP.Position))) == 4
    assert sorted(int(v) for v in graph.objects(None, CP.indexValue)) == [0, 0, 1, 1, 2, 2, 3, 3]  # 4 episodes, 4 cells
    assert len(list(graph.objects(None, CP.hasCell))) == 4
    assert (None, CP.hasDataset, None) in graph


def test_cells_are_indexed_row_major(tmp_path: Path) -> None:
    # 3 x 2 grid: cell k = j * 3 + i. Leave only cell (i=2, j=0), k=2, empty.
    spec = tmp_path / "spec.ttl"
    spec.write_text(SPEC.replace("cp:numCellW 2", "cp:numCellW 3").replace("cp:areaXMax 0.2", "cp:areaXMax 0.3"))
    points = [(0.05, 0.05), (0.15, 0.05), (0.05, 0.15), (0.15, 0.15), (0.25, 0.15)]
    report = verify_object_position_coverage(write_dataset(tmp_path / "ds", {i: {"cube": [xy]} for i, xy in enumerate(points)}), spec)
    assert report.coverage[0].counts == [[1, 1, 0], [1, 1, 1]]
    [finding] = report.violations
    assert finding.message.startswith("'cube' cell (i=2, j=0) x=[0.2, 0.3] y=[0, 0.1]")
    assert finding.focus_nodes == ["http://taica.course/cp/spec/test#cube_grid/cell/2"]


def test_area_bound_written_as_double_keeps_upper_edge(tmp_path: Path) -> None:
    spec = tmp_path / "spec.ttl"
    spec.write_text(SPEC.replace("cp:areaXMax 0.2", "cp:areaXMax 2.0e-1"))
    episodes = {i: {"cube": [xy]} for i, xy in enumerate(CELL_CENTERS[:3])}
    episodes[3] = {"cube": [(0.2, 0.15)]}
    report = verify_object_position_coverage(write_dataset(tmp_path / "ds", episodes), spec)
    assert report.passed, format_report(report)
    assert not report.warnings


def test_requirement_must_hang_off_an_axis(tmp_path: Path) -> None:
    spec = tmp_path / "spec.ttl"
    spec.write_text(SPEC.replace("cp:hasRequirement spec:cube_requirement", "cp:hasRequirement spec:other"))
    dataset = write_dataset(tmp_path / "ds", {i: {"cube": [xy]} for i, xy in enumerate(CELL_CENTERS)})
    report = verify_object_position_coverage(dataset, spec)
    assert not report.passed
    assert "Requirement is not linked from an ObjectPositionAxis (cp:hasRequirement)" in [v.message for v in report.violations]


def test_grid_cell_count_must_match_width_times_height(tmp_path: Path) -> None:
    # A hand-written extra cell makes 5 cells on a 2 x 2 grid.
    spec = tmp_path / "spec.ttl"
    spec.write_text(SPEC + "spec:cube_grid cp:hasCell spec:extra_cell .\nspec:extra_cell a cp:Cell .\n")
    dataset = write_dataset(tmp_path / "ds", {i: {"cube": [xy]} for i, xy in enumerate(CELL_CENTERS)})
    report = verify_object_position_coverage(dataset, spec)
    assert "Coverage grid has 5 cell(s); numCellW * numCellH = 4" in [v.message for v in report.violations]
