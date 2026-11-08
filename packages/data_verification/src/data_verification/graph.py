"""Turn a coverage spec, a dataset's initial positions and a rollout log into one RDF graph."""

from __future__ import annotations

from decimal import Decimal
from urllib.parse import quote

from rdflib import RDF, XSD, Graph, Literal, Namespace, URIRef

from .dataset import DatasetPlacements
from .rollout_log import RolloutLog

CP = Namespace("http://taica.course/cp/ontology#")

AREA_BOUNDS = (CP.areaXMin, CP.areaXMax, CP.areaYMin, CP.areaYMax)


def dataset_iri(dataset_name: str) -> URIRef:
    return URIRef(f"http://taica.course/cp/data/{quote(dataset_name, safe='')}")


def expand_cells(graph: Graph) -> None:
    """Store area bounds as decimals and add a cp:Cell per grid cell, indexed row-major.

    Grids with a missing or invalid numCellW/numCellH are skipped here; the
    SHACL grid shape reports them.
    """
    # Positions are decimals rounded to 6 places; a bound written as a double
    # (8.8e0) would compare against them as 8.800000000000000710.
    for area in set(graph.subjects(RDF.type, CP.AreaConfiguration)):
        for prop in AREA_BOUNDS:
            values = list(graph.objects(area, prop))
            value = values[0].toPython() if len(values) == 1 else None
            if isinstance(value, (int, float, Decimal)) and not isinstance(value, bool):
                graph.set((area, prop, Literal(Decimal(str(value)), datatype=XSD.decimal)))

    for grid in set(graph.subjects(RDF.type, CP.CoverageGrid)):
        try:
            nw, nh = int(graph.value(grid, CP.numCellW)), int(graph.value(grid, CP.numCellH))
        except (TypeError, ValueError):
            continue
        if nw < 1 or nh < 1:
            continue
        base = str(grid) if isinstance(grid, URIRef) else f"urn:cp:grid:{grid}"
        for k in range(nw * nh):
            cell = URIRef(f"{base}/cell/{k}")
            graph.add((cell, RDF.type, CP.Cell))
            graph.add((grid, CP.hasCell, cell))
            graph.add((cell, CP["index"], _index(k)))


def add_dataset(graph: Graph, dataset_name: str, data: DatasetPlacements) -> URIRef:
    """Add the dataset with every axis of the spec, its episodes and the objects' finite initial positions."""
    dataset = dataset_iri(dataset_name)
    graph.add((dataset, RDF.type, CP.Dataset))
    axes = set(graph.subjects(CP.hasRequirement, None))
    axes |= set(graph.subjects(RDF.type, CP.GeneralizationAxis)) | set(graph.subjects(RDF.type, CP.ObjectPositionAxis))
    for axis in axes:
        graph.add((dataset, CP.hasGeneralizationAxis, axis))

    for episode_index in data.episode_indices:
        episode = URIRef(f"{dataset}/episode/{episode_index}")
        graph.add((episode, RDF.type, CP.Episode))
        graph.add((episode, CP["index"], _index(episode_index)))
        graph.add((dataset, CP.hasEpisode, episode))

    for placement in data.placements:
        if not placement.is_finite:
            continue
        episode = URIRef(f"{dataset}/episode/{placement.episode_index}")
        _add_task_object(graph, episode, placement.object_name, placement.x, placement.y)
    return dataset


def add_rollout(graph: Graph, log: RolloutLog, dataset: URIRef) -> URIRef:
    """Add the rollout of ``log`` with its training ``dataset``, its runs, their status and initial positions."""
    rollout = URIRef(f"http://taica.course/cp/rollout/{quote(log.name, safe='')}")
    graph.add((rollout, RDF.type, CP.Rollout))
    graph.add((rollout, CP.hasDataset, dataset))
    graph.add((rollout, CP.successEpisodes, _index(log.successes)))
    graph.add((rollout, CP.totalEpisodes, _index(log.finished)))
    for r in log.runs:
        run = URIRef(f"{rollout}/run/{r.index}")
        graph.add((run, RDF.type, CP.Run))
        graph.add((run, CP["index"], _index(r.index)))
        graph.add((run, CP.status, Literal(r.status)))
        graph.add((rollout, CP.hasRun, run))
        for name, (x, y) in r.object_positions.items():
            _add_task_object(graph, run, name, x, y)
    return rollout


def _add_task_object(graph: Graph, owner: URIRef, name: str, x: float, y: float) -> None:
    obj = URIRef(f"{owner}/object/{quote(name, safe='')}")
    position = URIRef(f"{obj}/position")
    graph.add((obj, RDF.type, CP.TaskObject))
    graph.add((obj, CP.name, Literal(name)))
    graph.add((obj, CP.hasPosition, position))
    graph.add((position, RDF.type, CP.Position))
    graph.add((position, CP.positionX, _decimal(x)))
    graph.add((position, CP.positionY, _decimal(y)))
    graph.add((owner, CP.hasTaskObject, obj))


def _index(value: int) -> Literal:
    return Literal(value, datatype=XSD.nonNegativeInteger)


def _decimal(value: float) -> Literal:
    # Recorded positions are float32; 6 places keep micrometres and drop float noise.
    return Literal(Decimal(str(round(value, 6))), datatype=XSD.decimal)
