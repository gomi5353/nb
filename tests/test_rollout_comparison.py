"""Rollout-vs-dataset coverage comparison on synthetic logs, task files and datasets (no simulator)."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest
from data_verification import compare_rollout, format_comparison, rollout_comparison
from data_verification.graph import CP
from data_verification.rollout_comparison import AxisComparison
from data_verification.rollout_log import (
    OBJECT_POSITION_FACTOR,
    ROLLOUT_LOG_VERSION,
    RolloutLogError,
    read_rollout_log,
)
from rdflib import RDF, Graph
from test_object_position_coverage import write_dataset

ROOT = Path(__file__).resolve().parents[1]

# Student grid: 2x2 cells of 0.1 m over [0, 0.2]^2, at least 1 episode each.
SPEC = """
@prefix cp:   <http://taica.course/cp/ontology#> .
@prefix spec: <http://taica.course/cp/spec/test#> .

spec:axis a cp:ObjectPositionAxis ; cp:hasRequirement spec:cube_requirement .
spec:cube a cp:TaskObject ; cp:name "cube" .
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

@pytest.fixture
def spec(tmp_path: Path) -> Path:
    path = tmp_path / "spec.ttl"
    path.write_text(SPEC)
    return path


def write_log(path: Path, runs: list[tuple[float, float, str]], extra: dict | None = None, header: dict | None = None) -> Path:
    """runs: (x, y, status) of 'cube' per run."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [{"type": "header", "version": ROLLOUT_LOG_VERSION, "eval_file": "eval/cube_eval.py", "eval_sha256": "0" * 64, "seed": 42, **(header or {})}]
    for i, (x, y, status) in enumerate(runs):
        positions = {"cube": [x, y, 0.9], **(extra or {})}
        lines.append({"type": "episode", "episode": i, "status": status, "factors": {OBJECT_POSITION_FACTOR: positions}})
    path.write_text("".join(json.dumps(r) + "\n" for r in lines))
    return path


def full_dataset(tmp_path: Path, points=CELL_CENTERS) -> Path:
    return write_dataset(tmp_path / "ds", {i: {"cube": [xy]} for i, xy in enumerate(points)})


def run(tmp_path, spec, runs, dataset=None, **kwargs):
    log = write_log(tmp_path / "rollout" / "run.jsonl", runs)
    return compare_rollout(dataset or full_dataset(tmp_path), spec, log, **kwargs)


def cube(result):
    return result.axes[0].objects[0]


# ---- Comparison ---------------------------------------------------------------


def test_every_run_in_a_covered_cell_is_in_distribution(tmp_path: Path, spec: Path) -> None:
    runs = [(x, y, "succeeded") for x, y in CELL_CENTERS]
    result = run(tmp_path, spec, runs)
    assert result.passed, format_comparison(result)
    assert result.fully_covered
    assert [[c.runs for c in row] for row in cube(result).cells] == [[1, 1], [1, 1]]
    assert result.runs_in_distribution == 4 and not result.warnings


def test_run_in_a_short_cell_asks_to_record_there(tmp_path: Path, spec: Path) -> None:
    dataset = full_dataset(tmp_path, CELL_CENTERS[:3])  # cell (1, 1) has no episode
    result = run(tmp_path, spec, [(0.15, 0.15, "failed"), (0.05, 0.05, "succeeded")], dataset=dataset)
    cell = cube(result).cells[1][1]
    assert cell.runs == 1 and not cell.covered
    (target,) = result.to_fill
    assert (target.i, target.j, target.needed, target.runs, target.failed) == (1, 1, 1, 1, 1)
    assert result.runs_in_distribution == 1


def test_area_upper_edge_and_inner_edges_follow_the_check(tmp_path: Path, spec: Path) -> None:
    # x = 0.2 is the area maximum (last column); x = 0.1 is an inner edge (second column).
    result = run(tmp_path, spec, [(0.2, 0.2, "succeeded"), (0.1, 0.0, "succeeded")])
    cells = cube(result).cells
    assert cells[1][1].runs == 1 and cells[0][1].runs == 1
    assert result.fully_covered


def test_runs_outside_the_area_suggest_whole_cells(tmp_path: Path, spec: Path) -> None:
    runs = [(0.05, 0.05, "succeeded"), (0.22, 0.05, "failed"), (-0.01, 0.15, "failed")]
    result = run(tmp_path, spec, runs)
    obj = cube(result)
    assert [(c.i, c.j, c.runs) for c in obj.outside] == [(2, 0, 1), (-1, 1, 1)]
    assert obj.outside[0].x_range == pytest.approx((0.2, 0.3))
    (widen,) = result.widen
    assert widen.suggested == pytest.approx((-0.1, 0.3, 0.0, 0.2)) and (widen.num_cell_w, widen.num_cell_h) == (4, 2)
    assert obj.runs_covered == 1 and result.runs_in_distribution == 1


def test_unsampled_cells_are_not_judged(tmp_path: Path, spec: Path) -> None:
    dataset = full_dataset(tmp_path, CELL_CENTERS[:1])  # only cell (0, 0) has data
    result = run(tmp_path, spec, [(0.05, 0.05, "succeeded")], dataset=dataset)
    assert result.fully_covered and not result.to_fill
    assert cube(result).sampled_cells == 1 and cube(result).covered_cells == 1


def test_varying_object_without_requirement_is_warned_not_compared(tmp_path: Path, spec: Path) -> None:
    log = write_log(tmp_path / "run.jsonl", [(0.05, 0.05, "succeeded"), (0.15, 0.15, "succeeded")])
    lines = [json.loads(line) for line in log.read_text().splitlines()]
    for i, line in enumerate(lines[1:]):
        line["factors"][OBJECT_POSITION_FACTOR]["ball"] = [0.1 * i, 0.0, 0.9]
    log.write_text("".join(json.dumps(r) + "\n" for r in lines))
    result = compare_rollout(full_dataset(tmp_path), spec, log)
    assert result.passed, format_comparison(result)
    assert [o.object_name for o in result.axes[0].objects] == ["cube"] and result.fully_covered
    assert any("'ball'" in w.message and "no requirement" in w.message for w in result.warnings)


def test_aborted_runs_are_kept_but_not_counted(tmp_path: Path, spec: Path) -> None:
    graph_out = tmp_path / "graph.ttl"
    result = run(tmp_path, spec, [(0.05, 0.05, "succeeded"), (0.15, 0.05, "aborted"), (0.15, 0.15, "failed")], graph_out=graph_out)
    assert (result.runs, result.successes, result.aborted) == (2, 1, 1)
    graph = Graph().parse(graph_out)
    assert {str(s) for s in graph.objects(None, CP.status)} == {"succeeded", "failed", "aborted"}
    (rollout,) = graph.subjects(RDF.type, CP.Rollout)
    assert (graph.value(rollout, CP.totalEpisodes).toPython(), graph.value(rollout, CP.successEpisodes).toPython()) == (2, 1)


def test_rollout_has_its_training_dataset_and_runs_are_not_episodes(tmp_path: Path, spec: Path) -> None:
    graph_out = tmp_path / "graph.ttl"
    dataset = full_dataset(tmp_path, CELL_CENTERS[:1])
    result = run(tmp_path, spec, [(0.15, 0.15, "succeeded")] * 3, dataset=dataset, graph_out=graph_out)
    graph = Graph().parse(graph_out)
    (rollout,) = graph.subjects(RDF.type, CP.Rollout)
    (ds,) = graph.subjects(RDF.type, CP.Dataset)
    assert graph.value(rollout, CP.hasDataset) == ds
    assert len(list(graph.subjects(RDF.type, CP.Run))) == 3 and len(list(graph.subjects(RDF.type, CP.Episode))) == 1
    # Three runs in cell (1, 1) do not count as dataset coverage there.
    assert cube(result).cells[1][1].episodes == 0 and result.runs_in_distribution == 0


def test_every_axis_must_cover_a_run(tmp_path: Path, spec: Path, monkeypatch) -> None:
    def odd_runs_uncovered(graph, rollout, result, requirements):
        return AxisComparison("lighting", [], {i: i % 2 == 0 for i in range(result.runs)})

    monkeypatch.setattr(rollout_comparison, "ROLLOUT_AXES", [*rollout_comparison.ROLLOUT_AXES, odd_runs_uncovered])
    runs = [(x, y, "succeeded") for x, y in CELL_CENTERS]
    result = run(tmp_path, spec, runs)
    assert [a.axis for a in result.axes] == ["object_position", "lighting"]
    assert result.runs_in_distribution == 2 and not result.fully_covered


def test_checkpoint_trained_on_another_dataset_is_warned(tmp_path: Path, spec: Path) -> None:
    checkpoint = tmp_path / "train" / "checkpoints" / "last" / "pretrained_model"
    checkpoint.mkdir(parents=True)
    log = write_log(tmp_path / "run.jsonl", [(0.05, 0.05, "succeeded")], header={"checkpoint": str(checkpoint)})
    dataset = write_dataset(tmp_path / "me" / "ds", {i: {"cube": [xy]} for i, xy in enumerate(CELL_CENTERS)})
    (checkpoint / "train_config.json").write_text(json.dumps({"dataset": {"repo_id": "someone/other"}}))
    result = compare_rollout(dataset, spec, log)
    assert result.passed and any("trained on 'someone/other'" in w.message for w in result.warnings)
    (checkpoint / "train_config.json").write_text(json.dumps({"dataset": {"repo_id": "me/ds"}}))
    result = compare_rollout(dataset, spec, log)
    assert not any("trained on" in w.message for w in result.warnings)


def test_all_aborted_is_an_error(tmp_path: Path, spec: Path) -> None:
    result = run(tmp_path, spec, [(0.05, 0.05, "aborted")])
    assert not result.passed and "no finished run" in result.errors[0]


# ---- Log format ---------------------------------------------------------------


def test_log_rejects_duplicate_episodes(tmp_path: Path) -> None:
    log = write_log(tmp_path / "episodes.jsonl", [(0.05, 0.05, "succeeded")])
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


def _list_attr(path: Path, name: str) -> list[str]:
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.AnnAssign) and getattr(node.target, "id", None) == name:
            return ast.literal_eval(node.value)
    return []


def _randomized_names(path: Path) -> set[str]:
    """Objects whose start pose a reset event randomizes (randomize_objects_apart / randomize_object_uniform)."""
    names: set[str] = set()
    for call in (n for n in ast.walk(ast.parse(path.read_text())) if isinstance(n, ast.Call)):
        func = call.func.attr if isinstance(call.func, ast.Attribute) else getattr(call.func, "id", None)
        kwargs = {k.arg: k.value for k in call.keywords}
        if func == "randomize_objects_apart":
            node = kwargs.get("pose_ranges") or call.args[0]
            names |= {ast.literal_eval(k) for k in node.keys}
        elif func == "randomize_object_uniform":
            names.add(ast.literal_eval(kwargs.get("name") or call.args[0]))
    return names


TASK_FILES = [
    "eval/cup_stacking_eval.py",
    "eval/cutlery_arrangement_eval.py",
    "eval/toy_blocks_collection_eval.py",
    "packages/simulator/src/simulator/tasks/cup_stacking/cup_stacking_env_cfg.py",
    "packages/simulator/src/simulator/tasks/cutlery_arrangement/cutlery_arrangement_env_cfg.py",
    "packages/simulator/src/simulator/tasks/toy_blocks_collection/toy_blocks_collection_env_cfg.py",
]


@pytest.mark.parametrize("task_file", TASK_FILES)
def test_only_randomized_objects_are_tracked(task_file: str) -> None:
    path = ROOT / task_file
    tracked, fixed = _list_attr(path, "tracked_object_names"), _list_attr(path, "fixed_object_names")
    assert set(tracked) == _randomized_names(path), "tracked_object_names must list exactly the randomized objects"
    assert not set(tracked) & set(fixed)


def test_fixed_poses_in_header_are_ignored(tmp_path: Path, spec: Path) -> None:
    log = write_log(tmp_path / "run.jsonl", [(0.05, 0.05, "succeeded")])
    lines = [json.loads(line) for line in log.read_text().splitlines()]
    pose = {"pos": [7.0, 2.9, 0.74], "rot": [1.0, 0.0, 0.0, 0.0]}
    lines[0].update(robot_base=pose, fixed_objects={"plate": pose})
    log.write_text("".join(json.dumps(r) + "\n" for r in lines))
    result = compare_rollout(full_dataset(tmp_path), spec, log)
    assert result.passed and [o.object_name for o in result.axes[0].objects] == ["cube"]


def test_rollout_header_has_robot_base_and_fixed_objects() -> None:
    source = (ROOT / "scripts" / "rollout.py").read_text()
    assert '"robot_base": asset_pose(env.scene, "robot")' in source
    assert "fixed_object_names" in source
    reset = source.index("obs_dict, _ = env.reset()\n        rollout_log = open_rollout_log(")
    assert reset < source.index("env.step(actions)"), "the header poses must be read after the first reset"


def test_inverted_spec_area_is_reported(tmp_path: Path) -> None:
    spec = tmp_path / "spec.ttl"
    spec.write_text(SPEC.replace("cp:areaXMin 0.0 ; cp:areaXMax 0.2", "cp:areaXMin 0.2 ; cp:areaXMax 0.0"))
    result = run(tmp_path, spec, [(0.05, 0.05, "succeeded")])
    assert not result.passed
    assert any(e.startswith("cube: coverage area is empty") for e in result.errors), result.errors


# ---- Rollout shapes on hand-written graphs --------------------------------------


def _rollout_errors(runs: list[tuple[int, str, list[str]]]) -> list[str]:
    """runs = (index, status, object names); every object at (0.05, 0.05)."""
    turtle = ["@prefix cp: <http://taica.course/cp/ontology#> .", "@prefix d: <http://test/> .",
              "d:ds a cp:Dataset .", "d:r a cp:Rollout ; cp:hasDataset d:ds ;"
              f" cp:successEpisodes {sum(s == 'succeeded' for _, s, _ in runs)} ;"
              f" cp:totalEpisodes {sum(s != 'aborted' for _, s, _ in runs)} ."]
    for n, (index, status, names) in enumerate(runs):
        turtle.append(f'd:r cp:hasRun d:run{n} . d:run{n} a cp:Run ; cp:index \"{index}\"^^<http://www.w3.org/2001/XMLSchema#nonNegativeInteger> ; cp:status "{status}" .')
        for m, name in enumerate(names):
            turtle.append(
                f'd:run{n} cp:hasTaskObject d:o{n}_{m} . d:o{n}_{m} a cp:TaskObject ; cp:name "{name}" ; '
                f"cp:hasPosition [ a cp:Position ; cp:positionX 0.05 ; cp:positionY 0.05 ] ."
            )
    graph = Graph().parse(data="\n".join(turtle), format="turtle")
    result = rollout_comparison.RolloutComparison(dataset="ds", spec="spec", rollout_log="r")
    rollout_comparison._validate_rollout(graph, result)
    return result.errors


def test_rollout_shape_needs_a_finished_run() -> None:
    assert "Rollout has no finished run (all aborted)" in _rollout_errors([(0, "aborted", ["cube"])])
    assert "Rollout has no finished run (all aborted)" not in _rollout_errors([(0, "aborted", []), (1, "failed", ["cube"])])


def test_rollout_shape_run_index_and_object_names_are_unique() -> None:
    errors = _rollout_errors([(0, "failed", ["cube", "cube"]), (0, "succeeded", ["cube"])])
    assert "Rollout has more than one run with index 0" in errors
    assert "Run has more than one task object named 'cube'" in errors
