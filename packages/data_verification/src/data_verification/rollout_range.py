"""Read an evaluation task file's object position randomization and write it as rollout ranges.

The task file is parsed with ``ast``, never imported, so no simulator is needed.
Recognized, as written in ``eval/*.py`` and the recording task cfgs:

* initial positions: ``<name>: RigidObjectCfg = RigidObjectCfg(init_state=...InitialStateCfg(pos=...))``
  in a class body, and ``self.scene.<name>.init_state.pos = ...`` overrides;
* randomization: ``randomize_objects_apart(pose_ranges={name: {"x": ..., "y": ...}})``
  and ``randomize_object_uniform("<name>", pose_range={"x": ..., "y": ...})``.

Values may be literals or module-level constants. Anything else is an error, so a
range is never silently guessed.
"""

from __future__ import annotations

import ast
import hashlib
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from urllib.parse import quote

from rdflib import RDF, Graph

from .graph import CP

RANGE_DIR = Path("outputs/rollout_ranges")
RANGE_NS = "http://taica.course/cp/rollout-range/"
REGENERATE_CMD = "uv run python scripts/verify_object_position_coverage.py rollout-range --eval {eval_file}"


class RolloutRangeError(ValueError):
    """The task file's object position randomization could not be read."""


@dataclass(frozen=True)
class ObjectRange:
    name: str
    init_xy: tuple[float, float]
    x_offsets: tuple[float, float]
    y_offsets: tuple[float, float]

    @property
    def x_range(self) -> tuple[Decimal, Decimal]:
        return _bounds(self.init_xy[0], self.x_offsets)

    @property
    def y_range(self) -> tuple[Decimal, Decimal]:
        return _bounds(self.init_xy[1], self.y_offsets)


@dataclass(frozen=True)
class TaskRanges:
    source_file: str
    sha256: str
    init_positions: dict[str, tuple[float, float, float]]
    ranges: list[ObjectRange]


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def default_range_path(eval_file: str | Path) -> Path:
    return RANGE_DIR / f"{Path(eval_file).stem}.ttl"


def regenerate_hint(eval_file: str | Path) -> str:
    return REGENERATE_CMD.format(eval_file=eval_file)


def read_task_ranges(task_file: str | Path, require_ranges: bool = True) -> TaskRanges:
    """Initial positions and object position randomization ranges of ``task_file``."""
    path = Path(task_file)
    try:
        tree = ast.parse(path.read_text(), filename=str(path))
    except FileNotFoundError:
        raise RolloutRangeError(f"Task file not found: {path}") from None
    except SyntaxError as exc:
        raise RolloutRangeError(f"{path} is not valid Python: {exc}") from None
    constants = _module_constants(tree)
    init = _initial_positions(tree, constants, path)
    offsets = _pose_ranges(tree, constants, path)
    if require_ranges and not offsets:
        raise RolloutRangeError(
            f"{path}: no object position randomization found "
            "(randomize_objects_apart(pose_ranges=...) or randomize_object_uniform(name, pose_range=...))"
        )
    ranges = []
    for name, (xs, ys) in offsets.items():
        if name not in init:
            raise RolloutRangeError(f"{path}: '{name}' is randomized but its init_state.pos was not found")
        ranges.append(ObjectRange(name, init[name][:2], xs, ys))
    return TaskRanges(str(path), file_sha256(path), init, sorted(ranges, key=lambda r: r.name))


def render_range_ttl(task: TaskRanges) -> str:
    """Turtle for ``task``'s ranges: one cp:ObjectPositionRolloutRange per randomized object."""
    hint = regenerate_hint(task.source_file)
    lines = [
        f"# Rollout ranges of {task.source_file}, generated; do not edit.",
        "# Regenerate after changing the file's object positions or randomization:",
        f"#   {hint}",
        "",
        "@prefix cp: <http://taica.course/cp/ontology#> .",
        f"@prefix rr: <{RANGE_NS}{quote(Path(task.source_file).stem, safe='')}#> .",
        "",
    ]
    for r in task.ranges:
        (x0, x1), (y0, y1) = r.x_range, r.y_range
        lines += [
            f"rr:{r.name}_range a cp:ObjectPositionRolloutRange ;",
            f'    cp:sourceFile "{task.source_file}" ;',
            f'    cp:sourceSha256 "{task.sha256}" ;',
            f"    cp:hasTaskObject rr:{r.name} ;",
            f"    cp:hasAreaConfiguration rr:{r.name}_area .",
            "",
            f'rr:{r.name} a cp:TaskObject ; cp:hasName [ a cp:Name ; cp:nameValue "{r.name}" ] .',
            "",
            f"# init_state.pos ({r.init_xy[0]}, {r.init_xy[1]}) + pose range x {r.x_offsets}, y {r.y_offsets}",
            f"rr:{r.name}_area a cp:AreaConfiguration ;",
            f"    cp:areaXMin {_turtle_decimal(x0)} ; cp:areaXMax {_turtle_decimal(x1)} ;",
            f"    cp:areaYMin {_turtle_decimal(y0)} ; cp:areaYMax {_turtle_decimal(y1)} .",
            "",
        ]
    return "\n".join(lines)


def write_range_file(eval_file: str | Path, out: str | Path | None = None) -> tuple[Path, TaskRanges]:
    task = read_task_ranges(eval_file)
    path = Path(out) if out is not None else default_range_path(eval_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_range_ttl(task))
    return path, task


def range_file_sha256(graph: Graph) -> set[str]:
    """The cp:sourceSha256 values of the rollout ranges in ``graph``."""
    return {
        str(sha)
        for rng in graph.subjects(RDF.type, CP.ObjectPositionRolloutRange)
        for sha in graph.objects(rng, CP.sourceSha256)
    }


def _bounds(center: float, offsets: tuple[float, float]) -> tuple[Decimal, Decimal]:
    return tuple(Decimal(str(round(center + o, 6))) for o in offsets)  # type: ignore[return-value]


def _turtle_decimal(value: Decimal) -> str:
    text = f"{value:f}"
    return text if "." in text else f"{text}.0"


def _module_constants(tree: ast.Module) -> dict[str, object]:
    constants: dict[str, object] = {}
    for node in tree.body:
        target, value = None, None
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            target, value = node.targets[0].id, node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value is not None:
            target, value = node.target.id, node.value
        if target is None:
            continue
        try:
            constants[target] = _value(value, constants)
        except ValueError:
            pass
    return constants


def _value(node: ast.AST, constants: dict[str, object]) -> object:
    """Literal value of ``node``; names resolve to module constants."""
    if isinstance(node, ast.Name) and node.id in constants:
        return constants[node.id]
    if isinstance(node, (ast.Tuple, ast.List)):
        return tuple(_value(e, constants) for e in node.elts)
    if isinstance(node, ast.Dict) and None not in node.keys:
        return {_value(k, constants): _value(v, constants) for k, v in zip(node.keys, node.values)}
    try:
        return ast.literal_eval(node)
    except ValueError:
        raise ValueError(ast.unparse(node)) from None


def _keyword(call: ast.Call, name: str) -> ast.AST | None:
    return next((k.value for k in call.keywords if k.arg == name), None)


def _func_name(call: ast.Call) -> str:
    func = call.func
    return func.attr if isinstance(func, ast.Attribute) else func.id if isinstance(func, ast.Name) else ""


def _xyz(node: ast.AST, constants: dict, path: Path, name: str) -> tuple[float, float, float]:
    try:
        pos = _value(node, constants)
    except ValueError as exc:
        raise RolloutRangeError(f"{path}: init_state.pos of '{name}' is not a literal: {exc}") from None
    if not (isinstance(pos, tuple) and len(pos) == 3 and all(isinstance(v, (int, float)) for v in pos)):
        raise RolloutRangeError(f"{path}: init_state.pos of '{name}' is not (x, y, z): {pos!r}")
    return tuple(float(v) for v in pos)  # type: ignore[return-value]


def _initial_positions(tree: ast.Module, constants: dict, path: Path) -> dict[str, tuple[float, float, float]]:
    positions: dict[str, tuple[float, float, float]] = {}
    for cls in (n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)):
        for node in cls.body:
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                continue
            target = node.targets[0] if isinstance(node, ast.Assign) else node.target
            if not isinstance(target, ast.Name) or not isinstance(node.value, ast.Call):
                continue
            init_state = _keyword(node.value, "init_state")
            pos = _keyword(init_state, "pos") if isinstance(init_state, ast.Call) else None
            if pos is not None:
                positions[target.id] = _xyz(pos, constants, path, target.id)
    # Overrides in __post_init__: self.scene.<name>.init_state.pos = (...)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        chain = _attribute_chain(node.targets[0])
        if len(chain) == 5 and chain[:2] == ["self", "scene"] and chain[3:] == ["init_state", "pos"]:
            positions[chain[2]] = _xyz(node.value, constants, path, chain[2])
    return positions


def _attribute_chain(node: ast.AST) -> list[str]:
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return parts[::-1]


def _pose_ranges(tree: ast.Module, constants: dict, path: Path) -> dict[str, tuple[tuple[float, float], tuple[float, float]]]:
    found: dict[str, tuple[tuple[float, float], tuple[float, float]]] = {}

    def add(name: object, pose_range_node: ast.AST) -> None:
        try:
            pose_range = _value(pose_range_node, constants)
        except ValueError as exc:
            raise RolloutRangeError(f"{path}: pose range of '{name}' is not a literal: {exc}") from None
        if not isinstance(name, str) or not isinstance(pose_range, dict):
            raise RolloutRangeError(f"{path}: cannot read the pose range of {name!r}")
        if name in found:
            raise RolloutRangeError(f"{path}: '{name}' is randomized more than once")
        found[name] = (_offsets(pose_range, "x", name, path), _offsets(pose_range, "y", name, path))

    for call in (n for n in ast.walk(tree) if isinstance(n, ast.Call)):
        func = _func_name(call)
        if func == "randomize_objects_apart":
            node = _keyword(call, "pose_ranges") or (call.args[0] if call.args else None)
            if not isinstance(node, ast.Dict) or None in node.keys:
                raise RolloutRangeError(f"{path}: randomize_objects_apart pose_ranges must be a literal dict")
            for key, value in zip(node.keys, node.values):
                add(_value(key, constants), value)
        elif func == "randomize_object_uniform":
            name_node = _keyword(call, "name") or (call.args[0] if call.args else None)
            range_node = _keyword(call, "pose_range") or (call.args[1] if len(call.args) > 1 else None)
            if name_node is None or range_node is None:
                raise RolloutRangeError(f"{path}: randomize_object_uniform needs a name and a pose_range")
            add(_value(name_node, constants), range_node)
    return found


def _offsets(pose_range: dict, axis: str, name: str, path: Path) -> tuple[float, float]:
    lo, hi = pose_range.get(axis, (0.0, 0.0))
    if not all(isinstance(v, (int, float)) for v in (lo, hi)) or lo > hi:
        raise RolloutRangeError(f"{path}: pose range {axis} of '{name}' must be (low, high) numbers, got {(lo, hi)!r}")
    return float(lo), float(hi)
