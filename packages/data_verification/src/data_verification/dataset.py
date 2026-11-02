"""Read initial object placements from a local LeRobot v3 dataset.

Only ``meta/info.json`` and the ``data/**/*.parquet`` shards are read; videos
are never decoded. The initial placement of an object in an episode is its
``object_position.<name>`` value in that episode's first recorded frame (lowest
``frame_index``), whatever the collection pipeline was and whether the
episode succeeded or failed.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

# Keep in sync with simulator.utils.lerobot_recorder.
OBJECT_POSITION_KEY_PREFIX = "object_position."
OBJECT_POSITION_NAMES = ["x", "y", "z"]


class DatasetError(ValueError):
    """The dataset cannot be read for object position verification."""


@dataclass(frozen=True)
class InitialPlacement:
    episode_index: int
    object_name: str
    x: float
    y: float
    z: float

    @property
    def is_finite(self) -> bool:
        return all(math.isfinite(v) for v in (self.x, self.y, self.z))


@dataclass(frozen=True)
class DatasetPlacements:
    episode_indices: list[int]
    object_names: list[str]
    placements: list[InitialPlacement]
    first_frame_index: dict[int, int]


def read_initial_placements(dataset_root: str | Path) -> DatasetPlacements:
    root = Path(dataset_root)
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        raise DatasetError(f"Not a LeRobot dataset: {info_path} does not exist")
    features = json.loads(info_path.read_text()).get("features", {})

    position_keys = sorted(k for k in features if k.startswith(OBJECT_POSITION_KEY_PREFIX))
    if not position_keys:
        raise DatasetError(
            f"meta/info.json declares no '{OBJECT_POSITION_KEY_PREFIX}<name>' features; "
            "record with simulator.utils.lerobot_recorder.LeRobotRecorderManager, not leisaac's recorder"
        )
    for key in position_keys:
        shape = list(features[key].get("shape", []))
        if shape != [len(OBJECT_POSITION_NAMES)]:
            raise DatasetError(f"Feature '{key}' has shape {shape}; expected [{len(OBJECT_POSITION_NAMES)}]")

    shards = sorted((root / "data").glob("**/*.parquet"))
    if not shards:
        raise DatasetError(f"No parquet shards under {root / 'data'}")
    columns = ["episode_index", "frame_index", *position_keys]
    tables = []
    for shard in shards:
        missing = sorted(set(columns) - set(pq.read_schema(shard).names))
        if missing:
            raise DatasetError(f"{shard.relative_to(root)} is missing columns: {', '.join(missing)}")
        tables.append(pq.read_table(shard, columns=columns))
    table = pa.concat_tables(tables)

    episodes = table.column("episode_index").to_numpy()
    frames = table.column("frame_index").to_numpy()
    order = np.lexsort((frames, episodes))
    episode_ids, first = np.unique(episodes[order], return_index=True)
    first_rows = table.take(pa.array(order[first])).to_pydict()

    object_names = [key[len(OBJECT_POSITION_KEY_PREFIX) :] for key in position_keys]
    placements = []
    for row, episode_index in enumerate(first_rows["episode_index"]):
        for key, name in zip(position_keys, object_names):
            position = first_rows[key][row]
            placements.append(
                InitialPlacement(
                    episode_index=int(episode_index),
                    object_name=name,
                    x=float("nan") if position is None else float(position[0]),
                    y=float("nan") if position is None else float(position[1]),
                    z=float("nan") if position is None else float(position[2]),
                )
            )
    return DatasetPlacements(
        episode_indices=[int(e) for e in episode_ids],
        object_names=object_names,
        placements=placements,
        first_frame_index={int(e): int(f) for e, f in zip(first_rows["episode_index"], first_rows["frame_index"])},
    )
