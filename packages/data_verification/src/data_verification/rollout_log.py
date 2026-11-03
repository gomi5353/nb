"""Read the per-episode rollout log written by ``scripts/rollout.py`` (``outputs/rollout/<timestamp>.jsonl``).

Format: ``packages/simulator/src/simulator/utils/rollout_log.py``. Keep
``ROLLOUT_LOG_VERSION`` and ``OBJECT_POSITION_FACTOR`` in sync with it.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

ROLLOUT_LOG_VERSION = 1
OBJECT_POSITION_FACTOR = "object_position"
STATUSES = ("succeeded", "failed", "aborted")


class RolloutLogError(ValueError):
    """The rollout log is missing or malformed."""


@dataclass
class RolloutEpisode:
    index: int
    succeeded: bool
    factors: dict[str, dict]

    @property
    def object_positions(self) -> dict[str, tuple[float, float]]:
        """name -> (x, y) of the object's initial position."""
        return {name: (float(p[0]), float(p[1])) for name, p in self.factors.get(OBJECT_POSITION_FACTOR, {}).items()}


@dataclass
class RolloutLog:
    path: Path
    header: dict
    episodes: list[RolloutEpisode] = field(default_factory=list)
    aborted: int = 0

    @property
    def name(self) -> str:
        """Run name: the log's file name without extension (outputs/rollout/<timestamp>.jsonl)."""
        return self.path.stem


def read_rollout_log(path: str | Path) -> RolloutLog:
    path = Path(path)
    try:
        lines = path.read_text().splitlines()
    except FileNotFoundError:
        raise RolloutLogError(f"Rollout log not found: {path}") from None
    records = []
    for number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise RolloutLogError(f"{path}:{number}: not valid JSON: {exc}") from None
    if not records or records[0].get("type") != "header":
        raise RolloutLogError(f"{path}: the first line must be the rollout header")
    header = records[0]
    if header.get("version") != ROLLOUT_LOG_VERSION:
        raise RolloutLogError(f"{path}: log version {header.get('version')!r}, expected {ROLLOUT_LOG_VERSION}")
    log = RolloutLog(path, header)
    seen = set()
    for number, record in enumerate(records[1:], 2):
        status, index = record.get("status"), record.get("episode")
        if record.get("type") != "episode" or status not in STATUSES or not isinstance(index, int):
            raise RolloutLogError(f"{path}: record {number} is not an episode with a status of {', '.join(STATUSES)}")
        if index in seen:
            raise RolloutLogError(f"{path}: episode {index} appears twice")
        seen.add(index)
        if status == "aborted":
            log.aborted += 1
            continue
        episode = RolloutEpisode(index, status == "succeeded", record.get("factors", {}))
        for name, (x, y) in episode.object_positions.items():
            if not (math.isfinite(x) and math.isfinite(y)):
                raise RolloutLogError(f"{path}: episode {index}: non-finite position for '{name}'")
        log.episodes.append(episode)
    return log
