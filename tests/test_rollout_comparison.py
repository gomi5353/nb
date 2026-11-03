"""Rollout-vs-dataset coverage comparison on synthetic logs, eval files and datasets (no simulator)."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest
from data_verification import compare_rollout, format_comparison, rollout_comparison
from data_verification.rollout_comparison import AxisComparison
from data_verification.rollout_log import (
    OBJECT_POSITION_FACTOR,
    ROLLOUT_LOG_VERSION,
    RolloutLogError,
    read_rollout_log,
)
from data_verification.rollout_range import (
    RolloutRangeError,
    file_sha256,
    read_task_ranges,
    write_range_file,
)
from test_object_position_coverage import write_dataset

ROOT = Path(__file__).resolve().parents[1]

# Student grid: 2x2 cells of 0.1 m over [0, 0.2]^2, at least 1 episode each.
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

CELL_CENTERS = [(0.05, 0.05), (0.15, 0.05), (0.05, 0.15), (0.15, 0.15)]

EVAL_FILE = """
SPREAD = 0.1

class SceneCfg:
    cube: RigidObjectCfg = RigidObjectCfg(
        prim_path="{{ENV_REGEX_NS}}/cube",
        init_state=RigidObjectCfg.InitialStateCfg(pos=({cx}, {cy}, 0.9), rot=(1.0, 0.0, 0.0, 0.0)),
    )

class EnvCfg:
    def __post_init__(self):
        domain_randomization(
            self,
            random_options=[
                randomize_objects_apart(
                    pose_ranges={{"cube": {{"x": ({x0}, {x1}), "y": ({y0}, {y1}), "z": (0.0, 0.0)}}}},
                    min_distance=0.1,
                ),
            ],
        )
"""


@pytest.fixture
def spec(tmp_path: Path) -> Path:
    path = tmp_path / "spec.ttl"
    path.write_text(SPEC)
    return path


def write_eval(tmp_path: Path, center=(0.1, 0.1), x=(-0.1, 0.1), y=(-0.1, 0.1)) -> tuple[Path, Path]:
    """An eval file randomizing 'cube' over center + (x, y), and its generated range file."""
    eval_file = tmp_path / "cube_eval.py"
    eval_file.write_text(EVAL_FILE.format(cx=center[0], cy=center[1], x0=x[0], x1=x[1], y0=y[0], y1=y[1]))
    range_file, _ = write_range_file(eval_file, tmp_path / "ranges" / "cube_eval.ttl")
    return eval_file, range_file


def write_log(path: Path, eval_file: Path, runs: list[tuple[float, float, str]], extra: dict | None = None) -> Path:
    """runs: (x, y, status) of 'cube' per episode."""
    path.parent.mkdir(parents=True, exist_ok=True)
    header = {"type": "header", "version": ROLLOUT_LOG_VERSION, "eval_file": str(eval_file), "eval_sha256": file_sha256(eval_file), "seed": 42}
    lines = [header]
    for i, (x, y, status) in enumerate(runs):
        positions = {"cube": [x, y, 0.9], **(extra or {})}
        lines.append({"type": "episode", "episode": i, "status": status, "factors": {OBJECT_POSITION_FACTOR: positions}})
    path.write_text("".join(json.dumps(r) + "\n" for r in lines))
    return path


def full_dataset(tmp_path: Path, points=CELL_CENTERS) -> Path:
    return write_dataset(tmp_path / "ds", {i: {"cube": [xy]} for i, xy in enumerate(points)})


def run(tmp_path, spec, runs, dataset=None, **eval_kwargs):
    eval_file, range_file = write_eval(tmp_path, **eval_kwargs)
    log = write_log(tmp_path / "rollout" / "run" / "episodes.jsonl", eval_file, runs)
    return compare_rollout(dataset or full_dataset(tmp_path), spec, log, range_file)


def cube(result):
    return result.axes[0].objects[0]


# ---- Range extraction ---------------------------------------------------------


@pytest.mark.parametrize(
    "eval_file, expected",
    [
        ("eval/cup_stacking_eval.py", {"blue_cup": ("8.8", "9.0", "6.38", "6.58"), "pink_cup": ("9.0", "9.2", "6.38", "6.58")}),
        ("eval/cutlery_arrangement_eval.py", {"fork": ("7.0", "7.1", "3.0", "3.1"), "knife": ("6.9", "7.0", "3.1", "3.2")}),
        (
            "eval/toy_blocks_collection_eval.py",
            {
                "blue_block": ("3.3", "3.36", "0.79", "0.85"),
                "green_block": ("3.42", "3.48", "0.79", "0.85"),
                "red_block": ("3.36", "3.42", "0.69", "0.75"),
            },
        ),
    ],
)
def test_ranges_of_the_public_eval_files(eval_file: str, expected: dict) -> None:
    task = read_task_ranges(ROOT / eval_file)
    got = {r.name: tuple(str(v) for v in (*r.x_range, *r.y_range)) for r in task.ranges}
    assert got == expected


def test_range_file_parses_with_decimal_bounds(tmp_path: Path) -> None:
    from data_verification.graph import CP
    from rdflib import Graph

    path, task = write_range_file(ROOT / "eval/cup_stacking_eval.py", tmp_path / "cup.ttl")
    graph = Graph().parse(path)
    bounds = sorted(str(v.datatype).rsplit("#", 1)[-1] for v in graph.objects(None, CP.areaXMin))
    assert bounds == ["decimal", "decimal"]
    assert {str(v) for v in graph.objects(None, CP.sourceSha256)} == {task.sha256}


def test_uniform_randomizer_constants_and_post_init_override(tmp_path: Path) -> None:
    path = tmp_path / "task.py"
    path.write_text(
        "POS = (1.0, 2.0, 0.5)\n"
        "class Scene:\n"
        "    knife: RigidObjectCfg = RigidObjectCfg(init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)))\n"
        "class Env:\n"
        "    def __post_init__(self):\n"
        "        self.scene.knife.init_state.pos = POS\n"
        "        randomize_object_uniform('knife', pose_range={'x': (-0.05, 0.05), 'y': (0.0, 0.1)})\n"
    )
    (r,) = read_task_ranges(path).ranges
    assert (r.name, [str(v) for v in (*r.x_range, *r.y_range)]) == ("knife", ["0.95", "1.05", "2.0", "2.1"])


def test_unreadable_pose_range_is_an_error(tmp_path: Path) -> None:
    path = tmp_path / "task.py"
    path.write_text(
        "class Scene:\n"
        "    cube: RigidObjectCfg = RigidObjectCfg(init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)))\n"
        "randomize_objects_apart(pose_ranges={'cube': {'x': (-W, W)}}, min_distance=0.1)\n"
    )
    with pytest.raises(RolloutRangeError, match="not a literal"):
        read_task_ranges(path)


# ---- Comparison ---------------------------------------------------------------


def test_aligned_grid_full_coverage(tmp_path: Path, spec: Path) -> None:
    runs = [(x, y, "succeeded") for x, y in CELL_CENTERS]
    result = run(tmp_path, spec, runs)
    assert result.passed, format_comparison(result)
    assert result.fully_covered
    obj = cube(result)
    assert [[c.runs for c in row] for row in obj.cells] == [[1, 1], [1, 1]]
    assert all(len(c.overlapped) == 1 for row in obj.cells for c in row)  # touching edges do not overlap
    assert result.runs_in_distribution == 4 and not result.warnings


def test_rollout_cell_overlapping_four_student_cells(tmp_path: Path, spec: Path) -> None:
    # Rollout range [0.05, 0.15]^2: one rollout cell straddling all 4 student cells; one of them is empty.
    dataset = full_dataset(tmp_path, CELL_CENTERS[:3])
    result = run(tmp_path, spec, [(0.1, 0.1, "failed")], dataset=dataset, x=(-0.05, 0.05), y=(-0.05, 0.05))
    (cell,) = [c for row in cube(result).cells for c in row]
    assert len(cell.overlapped) == 4
    assert not cell.covered and [(c.i, c.j) for c in cell.short_cells] == [(1, 1)]
    (target,) = result.to_fill
    assert (target.i, target.j, target.needed, target.runs, target.failed) == (1, 1, 1, 1, 1)
    assert result.runs_in_distribution == 0


def test_clipped_last_column_outside_area_suggests_whole_cells(tmp_path: Path, spec: Path) -> None:
    # Rollout x in [0, 0.25]: columns [0,.1], [.1,.2], [.2,.25] (clipped); the last is outside the area.
    runs = [(0.05, 0.05, "succeeded"), (0.22, 0.05, "failed")]
    result = run(tmp_path, spec, runs, x=(-0.1, 0.15))
    obj = cube(result)
    assert len(obj.cells[0]) == 3 and obj.cells[0][2].x_range == pytest.approx((0.2, 0.25))
    last = obj.cells[0][2]
    assert last.outside_area and not last.covered and last.runs == 1
    (widen,) = result.widen
    assert widen.suggested == pytest.approx((0.0, 0.3, 0.0, 0.2)) and (widen.num_cell_w, widen.num_cell_h) == (3, 2)
    assert obj.runs_covered == 1 and result.runs_in_distribution == 1


def test_unsampled_cells_are_not_judged(tmp_path: Path, spec: Path) -> None:
    dataset = full_dataset(tmp_path, CELL_CENTERS[:1])  # only cell (0, 0) has data
    result = run(tmp_path, spec, [(0.05, 0.05, "succeeded")], dataset=dataset)
    assert result.fully_covered and not result.to_fill
    assert cube(result).sampled_cells == 1 and cube(result).covered_cells == 1


def test_randomized_object_without_requirement(tmp_path: Path, spec: Path) -> None:
    eval_file, range_file = write_eval(tmp_path)
    text = range_file.read_text().replace('"cube"', '"ball"')
    range_file.write_text(text)
    log = write_log(tmp_path / "rollout" / "r" / "episodes.jsonl", eval_file, [], extra=None)
    log.write_text(
        log.read_text()
        + json.dumps({"type": "episode", "episode": 0, "status": "succeeded", "factors": {OBJECT_POSITION_FACTOR: {"ball": [0.1, 0.1, 0.9]}}})
        + "\n"
    )
    result = compare_rollout(full_dataset(tmp_path), spec, log, range_file)
    assert result.passed, format_comparison(result)
    assert not cube(result).has_requirement and result.runs_in_distribution == 0
    assert any("no requirement" in w.message for w in result.warnings)


def test_run_outside_rollout_range_is_warned_and_not_covered(tmp_path: Path, spec: Path) -> None:
    result = run(tmp_path, spec, [(0.05, 0.05, "succeeded"), (0.5, 0.5, "failed")])
    assert cube(result).runs_outside_range == 1 and result.runs_in_distribution == 1
    assert any("outside its rollout range" in w.message for w in result.warnings)


def test_aborted_episodes_are_skipped(tmp_path: Path, spec: Path) -> None:
    result = run(tmp_path, spec, [(0.05, 0.05, "succeeded"), (0.15, 0.05, "aborted")])
    assert (result.runs, result.aborted) == (1, 1)


def test_every_axis_must_cover_a_run(tmp_path: Path, spec: Path, monkeypatch) -> None:
    def odd_runs_uncovered(graph, result, requirements):
        return AxisComparison("lighting", [], {i: i % 2 == 0 for i in range(result.runs)})

    monkeypatch.setattr(rollout_comparison, "ROLLOUT_AXES", [*rollout_comparison.ROLLOUT_AXES, odd_runs_uncovered])
    runs = [(x, y, "succeeded") for x, y in CELL_CENTERS]
    result = run(tmp_path, spec, runs)
    assert [a.axis for a in result.axes] == ["object_position", "lighting"]
    assert result.runs_in_distribution == 2 and not result.fully_covered


def test_record_cfg_pose_range_covers_cells_to_fill(tmp_path: Path, spec: Path) -> None:
    dataset = full_dataset(tmp_path, CELL_CENTERS[:3])
    eval_file, range_file = write_eval(tmp_path)
    log = write_log(tmp_path / "rollout" / "r" / "episodes.jsonl", eval_file, [(0.15, 0.15, "failed")])
    record_cfg = tmp_path / "record_cfg.py"
    record_cfg.write_text(EVAL_FILE.format(cx=0.1, cy=0.1, x0=-0.01, x1=0.01, y0=-0.01, y1=0.01))
    result = compare_rollout(dataset, spec, log, range_file, record_cfg=record_cfg)
    assert result.pose_ranges == {"cube": {"x": (0.0, 0.1), "y": (0.0, 0.1)}}
    assert '"cube": {"x": (0.0, 0.1)' in format_comparison(result)


# ---- Range file freshness and log format -------------------------------------------


def test_missing_range_file_says_how_to_generate(tmp_path: Path, spec: Path) -> None:
    eval_file, _ = write_eval(tmp_path)
    log = write_log(tmp_path / "rollout" / "r" / "episodes.jsonl", eval_file, [(0.05, 0.05, "succeeded")])
    result = compare_rollout(full_dataset(tmp_path), spec, log, tmp_path / "missing.ttl")
    assert not result.passed
    assert "No rollout range file" in result.errors[0] and f"rollout-range --eval {eval_file}" in result.errors[0]


def test_changed_eval_file_asks_to_regenerate(tmp_path: Path, spec: Path) -> None:
    eval_file, range_file = write_eval(tmp_path)
    log = write_log(tmp_path / "rollout" / "r" / "episodes.jsonl", eval_file, [(0.05, 0.05, "succeeded")])
    eval_file.write_text(eval_file.read_text().replace("(-0.1, 0.1)", "(-0.2, 0.2)"))
    result = compare_rollout(full_dataset(tmp_path), spec, log, range_file)
    assert "changed since" in result.errors[0] and "Regenerate it" in result.errors[0]


def test_rollout_run_with_a_different_eval_file_is_an_error(tmp_path: Path, spec: Path) -> None:
    eval_file, range_file = write_eval(tmp_path)
    log = write_log(tmp_path / "rollout" / "r" / "episodes.jsonl", eval_file, [(0.05, 0.05, "succeeded")])
    log.write_text(log.read_text().replace(file_sha256(eval_file), "0" * 64))
    result = compare_rollout(full_dataset(tmp_path), spec, log, range_file)
    assert "randomization ranges may differ" in result.errors[0]


def test_log_rejects_duplicate_episodes(tmp_path: Path) -> None:
    eval_file, _ = write_eval(tmp_path)
    log = write_log(tmp_path / "episodes.jsonl", eval_file, [(0.05, 0.05, "succeeded")])
    log.write_text(log.read_text() + log.read_text().splitlines(keepends=True)[1])
    with pytest.raises(RolloutLogError, match="appears twice"):
        read_rollout_log(log)


def _module_constants(path: Path, prefix: str) -> dict[str, object]:
    tree = ast.parse(path.read_text())
    return {
        node.targets[0].id: ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id in prefix.split()
    }


def test_rollout_log_writer_and_reader_agree() -> None:
    names = "ROLLOUT_LOG_VERSION OBJECT_POSITION_FACTOR"
    writer = _module_constants(ROOT / "packages/simulator/src/simulator/utils/rollout_log.py", names)
    assert writer == {"ROLLOUT_LOG_VERSION": ROLLOUT_LOG_VERSION, "OBJECT_POSITION_FACTOR": OBJECT_POSITION_FACTOR}


def test_rollout_writes_log_after_each_reset() -> None:
    source = (ROOT / "scripts" / "rollout.py").read_text()
    assert 'os.path.join("outputs", "rollout", f"{time.strftime(' in source, "every rollout writes outputs/rollout/<timestamp>.jsonl"
    reset = source.index("obs_dict, _ = env.reset()\n        needs_reset = True")
    capture = source.index("start_positions = object_positions(env.scene")
    step = source.index("env.step(actions)")
    assert reset < capture < step, "initial positions must be read after the round's reset and before stepping"
