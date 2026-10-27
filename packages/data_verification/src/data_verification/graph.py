"""Turn a coverage spec and a dataset's initial placements into one RDF graph."""

from __future__ import annotations

from decimal import Decimal
from urllib.parse import quote

from rdflib import RDF, XSD, Graph, Literal, Namespace, URIRef

from .dataset import DatasetPlacements

CP = Namespace("http://taica.course/cp/ontology#")


def expand_cells(graph: Graph) -> None:
    """Store requirement bounds as doubles and add a cp:CoverageCell per grid cell.

    Requirements with missing or invalid bounds/grid are skipped here; the
    SHACL requirement shape reports them.
    """
    for req in list(graph.subjects(RDF.type, CP.CoverageRequirement)):
        # Turtle writes 0.2 as xsd:decimal, and rdflib compares Decimal("0.2")
        # exactly against the double 0.2000000000000000111, which would put a
        # placement on the boundary outside. Store bounds as doubles.
        for prop in (CP.xMin, CP.xMax, CP.yMin, CP.yMax):
            values = list(graph.objects(req, prop))
            if len(values) == 1 and isinstance(values[0].toPython(), (int, float, Decimal)):
                graph.set((req, prop, _double(float(values[0]))))
        try:
            x0, x1, y0, y1 = (float(graph.value(req, p)) for p in (CP.xMin, CP.xMax, CP.yMin, CP.yMax))
            nx, ny = int(graph.value(req, CP.gridX)), int(graph.value(req, CP.gridY))
        except (TypeError, ValueError):
            continue
        if nx < 1 or ny < 1 or x0 >= x1 or y0 >= y1:
            continue
        dx, dy = (x1 - x0) / nx, (y1 - y0) / ny
        for i in range(nx):
            for j in range(ny):
                cell = URIRef(f"{req}/cell/{i}_{j}")
                graph.add((cell, RDF.type, CP.CoverageCell))
                graph.add((cell, CP.ofRequirement, req))
                graph.add((cell, CP.cellI, Literal(i)))
                graph.add((cell, CP.cellJ, Literal(j)))
                # Last column/row reuses the requirement's exact upper bound.
                graph.add((cell, CP.xMin, _bound(x0 + i * dx)))
                graph.add((cell, CP.xMax, _bound(x1 if i == nx - 1 else x0 + (i + 1) * dx)))
                graph.add((cell, CP.yMin, _bound(y0 + j * dy)))
                graph.add((cell, CP.yMax, _bound(y1 if j == ny - 1 else y0 + (j + 1) * dy)))
                graph.add((cell, CP.xMaxInclusive, Literal(i == nx - 1)))
                graph.add((cell, CP.yMaxInclusive, Literal(j == ny - 1)))


def add_dataset(graph: Graph, dataset_name: str, data: DatasetPlacements) -> URIRef:
    """Add the dataset, its episodes and their finite initial placements."""
    dataset = URIRef(f"http://taica.course/cp/data/{quote(dataset_name, safe='')}")
    graph.add((dataset, RDF.type, CP.Dataset))

    objects = {str(graph.value(obj, CP.objectName)): obj for obj in graph.subjects(RDF.type, CP.TaskObject)}
    for name in data.object_names:
        if name not in objects:
            # Recorded but not required by the spec: still described, never checked.
            objects[name] = URIRef(f"{dataset}/object/{quote(name, safe='')}")
            graph.add((objects[name], RDF.type, CP.TaskObject))
            graph.add((objects[name], CP.objectName, Literal(name)))

    for episode_index in data.episode_indices:
        episode = URIRef(f"{dataset}/episode/{episode_index}")
        graph.add((episode, RDF.type, CP.Episode))
        graph.add((episode, CP.episodeIndex, Literal(episode_index)))
        graph.add((dataset, CP.hasEpisode, episode))

    for placement in data.placements:
        if not placement.is_finite:
            continue
        episode = URIRef(f"{dataset}/episode/{placement.episode_index}")
        node = URIRef(f"{episode}/placement/{quote(placement.object_name, safe='')}")
        graph.add((node, RDF.type, CP.ObjectPlacement))
        graph.add((node, CP.ofObject, objects[placement.object_name]))
        graph.add((node, CP.x, _double(placement.x)))
        graph.add((node, CP.y, _double(placement.y)))
        graph.add((node, CP.z, _double(placement.z)))
        graph.add((episode, CP.hasInitialPlacement, node))
    return dataset


def _double(value: float) -> Literal:
    return Literal(float(value), datatype=XSD.double)


def _bound(value: float) -> Literal:
    # Grid edges are derived by arithmetic; drop float noise (6.4799999999999995 -> 6.48).
    return _double(round(value, 9))
