"""Turn a coverage spec and a dataset's initial positions into one RDF graph."""

from __future__ import annotations

from decimal import Decimal
from urllib.parse import quote

from rdflib import RDF, XSD, BNode, Graph, Literal, Namespace, URIRef

from .dataset import DatasetPlacements

CP = Namespace("http://taica.course/cp/ontology#")

AREA_BOUNDS = (CP.areaXMin, CP.areaXMax, CP.areaYMin, CP.areaYMax)


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
            graph.add((cell, CP.hasIndex, _index(URIRef(f"{cell}/index"), k, graph)))


def add_dataset(graph: Graph, dataset_name: str, data: DatasetPlacements) -> URIRef:
    """Add the dataset, its episodes and the objects' finite initial positions; link it from every axis."""
    dataset = URIRef(f"http://taica.course/cp/data/{quote(dataset_name, safe='')}")
    graph.add((dataset, RDF.type, CP.Dataset))
    for axis in set(graph.subjects(RDF.type, CP.ObjectPositionAxis)):
        graph.add((axis, CP.hasDataset, dataset))

    # One Name node per object, shared by its TaskObject in every episode.
    names = {}
    for name in data.object_names:
        names[name] = URIRef(f"{dataset}/name/{quote(name, safe='')}")
        graph.add((names[name], RDF.type, CP.Name))
        graph.add((names[name], CP.nameValue, Literal(name)))

    for episode_index in data.episode_indices:
        episode = URIRef(f"{dataset}/episode/{episode_index}")
        graph.add((episode, RDF.type, CP.Episode))
        graph.add((episode, CP.hasIndex, _index(URIRef(f"{episode}/index"), episode_index, graph)))
        graph.add((dataset, CP.hasEpisode, episode))

    for placement in data.placements:
        if not placement.is_finite:
            continue
        episode = URIRef(f"{dataset}/episode/{placement.episode_index}")
        obj = URIRef(f"{episode}/object/{quote(placement.object_name, safe='')}")
        position = URIRef(f"{obj}/position")
        graph.add((obj, RDF.type, CP.TaskObject))
        graph.add((obj, CP.hasName, names[placement.object_name]))
        graph.add((obj, CP.hasPosition, position))
        graph.add((position, RDF.type, CP.Position))
        graph.add((position, CP.positionX, _decimal(placement.x)))
        graph.add((position, CP.positionY, _decimal(placement.y)))
        graph.add((episode, CP.hasTaskObject, obj))
    return dataset


def _index(node: URIRef | BNode, value: int, graph: Graph) -> URIRef | BNode:
    graph.add((node, RDF.type, CP.Index))
    graph.add((node, CP.indexValue, Literal(value, datatype=XSD.nonNegativeInteger)))
    return node


def _decimal(value: float) -> Literal:
    # Recorded positions are float32; 6 places keep micrometres and drop float noise.
    return Literal(Decimal(str(round(value, 6))), datatype=XSD.decimal)
