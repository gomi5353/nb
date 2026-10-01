"""Create a coverage spec skeleton from the packaged template."""

from __future__ import annotations

from decimal import Decimal
from importlib.resources import files
from pathlib import Path

from .dataset import DatasetPlacements
from .graph import DV

BEGIN = "# --- BEGIN OBJECT ---\n"
END = "# --- END OBJECT ---\n"
PLACEHOLDER = "OBJECT_NAME"


def template_text() -> str:
    return files("data_verification").joinpath("templates", "object_pose_coverage_spec.ttl").read_text()


def render_spec(object_names: list[str], observed: DatasetPlacements | None = None, dataset_label: str = "") -> str:
    """Template with one object block per name, values still "TODO".

    With ``observed``, each block starts with a comment giving the range of
    initial placements already in the dataset. That comment is information
    only; the requirement values stay for the user to choose.
    """
    text = template_text()
    head, rest = text.split(BEGIN, 1)
    block, tail = rest.split(END, 1)
    blocks = []
    for name in object_names:
        note = _observed_note(name, observed, dataset_label) if observed is not None else ""
        blocks.append(BEGIN + note + block.replace(PLACEHOLDER, name) + END)
    return head + "\n".join(blocks) + tail


def write_spec(path: str | Path, object_names: list[str], observed: DatasetPlacements | None = None, dataset_label: str = "") -> Path:
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"{path} already exists; not overwriting")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_spec(object_names, observed, dataset_label))
    return path


def _observed_note(name: str, observed: DatasetPlacements, dataset_label: str) -> str:
    points = [p for p in observed.placements if p.object_name == name and p.is_finite]
    if not points:
        return f"# Not recorded in {dataset_label or 'the dataset'} (no object_pose.{name} values).\n"
    xs, ys = [p.x for p in points], [p.y for p in points]
    return (
        f"# Already in {dataset_label or 'the dataset'}: {len(points)} episode(s), initial "
        f"x {min(xs):.3f}..{max(xs):.3f}, y {min(ys):.3f}..{max(ys):.3f}. For reference only.\n"
    )


def todo_fields(graph) -> list[str]:
    """'<object>: <property>' for every value still set to "TODO"."""
    fields = []
    for subject, predicate, value in graph:
        if str(value) == "TODO":
            obj = graph.value(subject, DV.forObject)
            name = graph.value(obj, DV.objectName) if obj is not None else None
            fields.append(f"{name or subject}: {predicate.split('#')[-1]}")
    return sorted(fields)



_NUMBER_FIELDS = ("xMin", "xMax", "yMin", "yMax")
_WHOLE_NUMBER_FIELDS = ("gridX", "gridY", "minEpisodesPerCell")


def value_type_errors(graph) -> list[str]:
    """Requirement values that are not numbers (e.g. still quoted)."""
    from rdflib import RDF

    errors = []
    for req in graph.subjects(RDF.type, DV.CoverageRequirement):
        obj = graph.value(req, DV.forObject)
        name = graph.value(obj, DV.objectName) if obj is not None else req
        for field in _NUMBER_FIELDS + _WHOLE_NUMBER_FIELDS:
            for value in graph.objects(req, DV[field]):
                python_value = value.toPython()
                whole = field in _WHOLE_NUMBER_FIELDS
                ok = isinstance(python_value, int) if whole else isinstance(python_value, (int, float, Decimal))
                if isinstance(python_value, bool) or not ok:
                    kind = "a whole number" if whole else "a number"
                    shown = f'"{value}"' if isinstance(python_value, str) else str(value)
                    errors.append(f"{name}: {field} must be {kind} without quotes, got {shown}")
    return sorted(errors)
