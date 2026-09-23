"""Object pose coverage verification on synthetic LeRobot v3 layouts (no simulator)."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from data_verification import format_report, render_spec, verify_object_pose_coverage, write_spec
from data_verification.dataset import OBJECT_POSE_KEY_PREFIX, OBJECT_POSE_NAMES, read_initial_placements
from data_verification.spec_template import template_text

ROOT = Path(__file__).resolve().parents[1]

SPEC = """
@prefix dv:   <urn:aicapstone:dv#> .
@prefix spec: <urn:aicapstone:spec:test#> .

spec:cube a dv:TaskObject ; dv:objectName "cube" .
spec:cube_coverage a dv:CoverageRequirement ;
    dv:targetsAxis dv:ObjectPoseAxis ;
    dv:forObject spec:cube ;
    dv:xMin 0.0 ; dv:xMax 0.2 ; dv:yMin 0.0 ; dv:yMax 0.2 ;
    dv:gridX 2 ; dv:gridY 2 ;
    dv:minEpisodesPerCell 1 .
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
            f"{OBJECT_POSE_KEY_PREFIX}{name}": {"dtype": "float32", "shape": [7], "names": OBJECT_POSE_NAMES}
            for name in object_names
        },
    }
    (root / "meta").mkdir(parents=True)
    (root / "meta" / "info.json").write_text(json.dumps({"codebase_version": "v3.0", "fps": 30, "features": features}))

    columns: dict[str, list] = {"episode_index": [], "frame_index": []}
    columns.update({f"{OBJECT_POSE_KEY_PREFIX}{name}": [] for name in object_names})
    for episode, objs in episodes.items():
        for frame in range(frames):
            columns["episode_index"].append(episode)
            columns["frame_index"].append(frame)
            for name in object_names:
                xys = objs[name]
                x, y = xys[min(frame, len(xys) - 1)]
                columns[f"{OBJECT_POSE_KEY_PREFIX}{name}"].append([x, y, 0.9, 1.0, 0.0, 0.0, 0.0])
    (root / "data" / "chunk-000").mkdir(parents=True)
    pq.write_table(pa.table(columns), root / "data" / "chunk-000" / "file-000.parquet")
    return root


def test_full_coverage_passes(tmp_path: Path, spec: Path) -> None:
    dataset = write_dataset(tmp_path / "ds", {i: {"cube": [xy]} for i, xy in enumerate(CELL_CENTERS)})
    report = verify_object_pose_coverage(dataset, spec)
    assert report.passed, format_report(report)
    assert report.num_episodes == 4
    assert report.coverage[0].counts == [[1, 1], [1, 1]]


def test_empty_cell_is_a_violation(tmp_path: Path, spec: Path) -> None:
    dataset = write_dataset(tmp_path / "ds", {i: {"cube": [xy]} for i, xy in enumerate(CELL_CENTERS[:3])})
    report = verify_object_pose_coverage(dataset, spec)
    assert not report.passed
    [finding] = report.violations
    assert "cell (i=1, j=1)" in finding.message and "has 0 episode(s)" in finding.message
    assert report.coverage[0].cells_covered == 3


def test_initial_pose_is_first_frame_not_later(tmp_path: Path, spec: Path) -> None:
    # Object starts in cell (1, 1) and is carried to cell (0, 0) afterwards.
    episodes = {i: {"cube": [xy]} for i, xy in enumerate(CELL_CENTERS[:3])}
    episodes[3] = {"cube": [(0.15, 0.15), (0.05, 0.05)]}
    report = verify_object_pose_coverage(write_dataset(tmp_path / "ds", episodes), spec)
    assert report.passed, format_report(report)


def test_every_episode_counts_regardless_of_outcome(tmp_path: Path, spec: Path) -> None:
    # A 'success' bookkeeping column must not filter anything.
    dataset = write_dataset(tmp_path / "ds", {i: {"cube": [xy]} for i, xy in enumerate(CELL_CENTERS)})
    shard = dataset / "data" / "chunk-000" / "file-000.parquet"
    table = pq.read_table(shard)
    pq.write_table(table.append_column("success", pa.array([False] * table.num_rows)), shard)
    assert verify_object_pose_coverage(dataset, spec).passed


def test_upper_edge_belongs_to_last_cell(tmp_path: Path, spec: Path) -> None:
    episodes = {i: {"cube": [xy]} for i, xy in enumerate(CELL_CENTERS[:3])}
    episodes[3] = {"cube": [(0.2, 0.2)]}
    report = verify_object_pose_coverage(write_dataset(tmp_path / "ds", episodes), spec)
    assert report.passed, format_report(report)
    assert not report.warnings


def test_outside_region_is_warning_and_not_counted(tmp_path: Path, spec: Path) -> None:
    episodes = {i: {"cube": [xy]} for i, xy in enumerate(CELL_CENTERS)}
    episodes[4] = {"cube": [(0.5, 0.05)]}
    report = verify_object_pose_coverage(write_dataset(tmp_path / "ds", episodes), spec)
    assert report.passed
    [warning] = report.warnings
    assert "outside the required region" in warning.message
    assert sum(map(sum, report.coverage[0].counts)) == 4


def test_missing_required_object_column(tmp_path: Path, spec: Path) -> None:
    dataset = write_dataset(tmp_path / "ds", {0: {"other": [(0.05, 0.05)]}})
    report = verify_object_pose_coverage(dataset, spec)
    assert not report.passed
    assert any("object_pose.cube" in e for e in report.errors)
    assert any("no initial pose for required object 'cube'" in v.message for v in report.violations)


def test_dataset_without_pose_columns(tmp_path: Path, spec: Path) -> None:
    root = tmp_path / "ds"
    (root / "meta").mkdir(parents=True)
    (root / "meta" / "info.json").write_text(json.dumps({"features": {"action": {"dtype": "float32", "shape": [8]}}}))
    report = verify_object_pose_coverage(root, spec)
    assert not report.passed
    assert "declares no 'object_pose.<name>' features" in report.errors[0]


def test_non_finite_pose_is_error(tmp_path: Path, spec: Path) -> None:
    episodes = {i: {"cube": [xy]} for i, xy in enumerate(CELL_CENTERS)}
    episodes[4] = {"cube": [(float("nan"), 0.05)]}
    report = verify_object_pose_coverage(write_dataset(tmp_path / "ds", episodes), spec)
    assert not report.passed
    assert any("non-finite" in e for e in report.errors)


def _fill(spec_text: str, values: dict[str, str]) -> str:
    for field, value in values.items():
        spec_text = spec_text.replace(f'{field} "TODO"', f"{field} {value}")
    return spec_text


FILLED = {
    "dv:xMin": "0.0", "dv:xMax": "0.2", "dv:yMin": "0.0", "dv:yMax": "0.2",
    "dv:gridX": "2", "dv:gridY": "2", "dv:minEpisodesPerCell": "1",
}


def test_raw_template_parses_and_asks_for_todos(tmp_path: Path) -> None:
    spec = tmp_path / "spec.ttl"
    spec.write_text(template_text())
    dataset = write_dataset(tmp_path / "ds", {0: {"cube": [(0.05, 0.05)]}})
    report = verify_object_pose_coverage(dataset, spec)
    assert not report.passed
    assert report.errors[0].startswith("Spec still has TODO values: OBJECT_NAME: gridX")
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
    report = verify_object_pose_coverage(dataset, spec)
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
    spec.write_text(_fill(render_spec(["cube"]), {**FILLED, "dv:xMin": '"0.0"', "dv:gridX": "2.5"}))
    dataset = write_dataset(tmp_path / "ds", {0: {"cube": [(0.05, 0.05)]}})
    report = verify_object_pose_coverage(dataset, spec)
    assert report.errors == [
        "cube: gridX must be a whole number without quotes, got 2.5",
        'cube: xMin must be a number without quotes, got "0.0"',
    ]


def test_invalid_turtle_is_reported(tmp_path: Path) -> None:
    spec = tmp_path / "spec.ttl"
    spec.write_text("this is not turtle")
    dataset = write_dataset(tmp_path / "ds", {0: {"cube": [(0.05, 0.05)]}})
    report = verify_object_pose_coverage(dataset, spec)
    assert report.errors[0].startswith("Spec is not valid Turtle")


def test_empty_region_is_reported(tmp_path: Path) -> None:
    spec = tmp_path / "spec.ttl"
    spec.write_text(_fill(render_spec(["cube"]), {**FILLED, "dv:xMin": "0.3"}))
    dataset = write_dataset(tmp_path / "ds", {0: {"cube": [(0.05, 0.05)]}})
    report = verify_object_pose_coverage(dataset, spec)
    assert any("Coverage region is empty" in v.message for v in report.violations)


def _module_constants(path: Path) -> dict[str, object]:
    tree = ast.parse(path.read_text())
    return {
        node.targets[0].id: ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id.startswith("OBJECT_POSE_")
    }


def test_recorder_and_verifier_agree_on_column_format() -> None:
    recorder = _module_constants(ROOT / "packages/simulator/src/simulator/utils/object_pose_recording.py")
    assert recorder == {"OBJECT_POSE_KEY_PREFIX": OBJECT_POSE_KEY_PREFIX, "OBJECT_POSE_NAMES": OBJECT_POSE_NAMES}


def _doc_extra_shape_example() -> str:
    doc = (ROOT / "docs" / "object_pose_coverage.md").read_text()
    return doc.split("```turtle\n@prefix dv:", 1)[1].split("```", 1)[0].join(["@prefix dv:", ""])


def test_extra_shapes_example_from_docs(tmp_path: Path, spec: Path) -> None:
    dataset = write_dataset(tmp_path / "ds", {i: {"cube": [xy]} for i, xy in enumerate(CELL_CENTERS)})  # z = 0.9
    rules = tmp_path / "rules.ttl"
    rules.write_text(_doc_extra_shape_example())
    assert verify_object_pose_coverage(dataset, spec, extra_shapes=[rules]).passed

    rules.write_text(_doc_extra_shape_example().replace('"0.5"', '"1.0"'))
    report = verify_object_pose_coverage(dataset, spec, extra_shapes=[rules])
    assert [(v.message, v.count) for v in report.violations] == [("dv:z: Object starts below the table", 4)]
