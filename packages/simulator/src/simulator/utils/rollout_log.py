"""Per-episode rollout log for comparing rollout coverage with dataset coverage.

``scripts/rollout.py`` writes one JSON object per line:

* the first line is the header: ``{"type": "header", "version": 1, "eval_file": ..., "eval_sha256": ..., ...,
  "robot_base": {"pos": [x, y, z], "rot": [w, x, y, z]}, "fixed_objects": {"<name>": {"pos": ..., "rot": ...}, ...}}``;
* then one line per episode: ``{"type": "episode", "episode": 0, "status": "succeeded" | "failed" | "aborted",
  "factors": {"object_position": {"<name>": [x, y, z], ...}}}``.

Positions are each object's root position right after the reset that starts the
episode, relative to the env origin, metres: the same frame as the dataset's
``object_position.<name>`` columns. ``aborted`` episodes (manual reset, app
closed) have no outcome and are skipped by the comparison.

``robot_base`` and ``fixed_objects`` (the task's ``fixed_object_names``, such as
the plate) are the poses of what does not move between episodes, read once
after the first reset in env 0, same frame, rotation as a ``(w, x, y, z)``
quaternion. They are for reference only: fixed objects have no dataset column
and the comparison ignores both.

``factors`` is keyed by generalization axis. A group that adds a factor logs its
value under its own key next to ``object_position`` (see
``docs/adding_coverage_factor.md``).

Read back by ``data_verification.rollout_log``; keep ``ROLLOUT_LOG_VERSION`` and
``OBJECT_POSITION_FACTOR`` in sync with it.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROLLOUT_LOG_VERSION = 1
OBJECT_POSITION_FACTOR = "object_position"


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class RolloutLogWriter:
    """Append-only JSON Lines writer; every round is flushed, so a stopped run keeps its finished episodes."""

    def __init__(self, path: str | Path, header: dict):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("w")
        self._write({"type": "header", "version": ROLLOUT_LOG_VERSION, **header})

    def write_episode(self, episode: int, status: str, factors: dict[str, dict]) -> None:
        self._write({"type": "episode", "episode": episode, "status": status, "factors": factors})

    def flush(self) -> None:
        self._file.flush()

    def close(self) -> None:
        self._file.close()

    def _write(self, record: dict) -> None:
        self._file.write(json.dumps(record) + "\n")


def object_positions(scene, env_ids, names: list[str] | None = None) -> dict[str, list[list[float]]]:
    """Root position of the rigid objects ``names`` (default: all) for ``env_ids``, relative to the env origin."""
    origins = scene.env_origins[env_ids]
    return {
        name: (obj.data.root_pos_w[env_ids] - origins).tolist()
        for name, obj in scene.rigid_objects.items()
        if not names or name in names
    }


def asset_pose(scene, name: str, env_id: int = 0) -> dict[str, list[float]]:
    """Root pose of the scene asset ``name`` in ``env_id``: position relative to the env origin and (w, x, y, z) quaternion."""
    data = scene[name].data
    return {
        "pos": (data.root_pos_w[env_id] - scene.env_origins[env_id]).tolist(),
        "rot": data.root_quat_w[env_id].tolist(),
    }
