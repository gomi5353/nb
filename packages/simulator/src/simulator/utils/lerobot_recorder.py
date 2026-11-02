"""LeRobot recorder with extra columns for offline data verification.

This is the repository's base extension of leisaac's ``LeRobotRecorderManager``.
It adds one verification factor, the object position; groups that verify further
factors append them here (see ``docs/adding_coverage_factor.md``).

Every tracked object gets its own dataset column ``object_position.<name>``:
float32 ``(3,)`` = ``[x, y, z]``, position relative to the env origin, metres.
Orientation is not recorded: the tasks randomize only object positions and keep
each object's start rotation fixed (a random rotation can tip an object off the
table, and the reset has to be retried), so the orientation carries no
information about coverage.

The ``object_position.`` prefix is deliberate. LeRobot turns every ``observation.*``
key and every ``action*`` key into a policy input/output
(``lerobot.datasets.utils.dataset_to_policy_features``); any other column is
ignored by policies. These columns are therefore safe for training and rollout,
and are read back by ``packages/data_verification``. New factors use their own
non-``observation.`` prefix, such as ``factor.<name>``, for the same reason.

A factor needs two additions: its feature spec in ``build_with_factors`` below,
and its per-frame value in ``SingleArmFrankaTaskEnvCfg.build_lerobot_frame``
(``tasks/template/single_arm_franka_cfg.py``), next to the object positions.

Record with ``LeRobotRecorderManager`` from this module, not leisaac's original
class, which writes no verification columns.

Keep ``OBJECT_POSITION_KEY_PREFIX`` / ``OBJECT_POSITION_NAMES`` in sync with
``data_verification.dataset``.
"""

from __future__ import annotations

from functools import cache
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from isaaclab.utils.datasets.episode_data import EpisodeData

OBJECT_POSITION_KEY_PREFIX = "object_position."
OBJECT_POSITION_NAMES = ["x", "y", "z"]


def object_position_features(object_names: list[str]) -> dict[str, dict]:
    """LeRobot feature specs for the ``object_position.<name>`` columns."""
    return {
        f"{OBJECT_POSITION_KEY_PREFIX}{name}": {
            "dtype": "float32",
            "shape": (len(OBJECT_POSITION_NAMES),),
            "names": list(OBJECT_POSITION_NAMES),
        }
        for name in object_names
    }


def object_positions_from_episode(episode_data: EpisodeData, dataset_features: dict) -> dict[str, np.ndarray]:
    """Latest object positions from an IsaacLab ``EpisodeData`` buffer.

    Reads the ``states`` entry written by ``PostStepStatesRecorder`` (part of
    ``ActionStateRecorderManagerCfg``), which stores env-relative root poses
    ``[x, y, z, qw, qx, qy, qz]``; only the position is kept. Only columns declared in ``dataset_features`` are returned.
    """
    rigid_states = episode_data._data.get("states", {}).get("rigid_object", {})
    frame = {}
    for key in dataset_features:
        if not key.startswith(OBJECT_POSITION_KEY_PREFIX):
            continue
        name = key[len(OBJECT_POSITION_KEY_PREFIX) :]
        frame[key] = rigid_states[name]["root_pose"][-1][:3].cpu().numpy().astype(np.float32)
    return frame


@cache
def _make_recorder_manager_class():
    import leisaac.enhance.managers.lerobot_recorder_manager as leisaac_recorder

    class LeRobotRecorderManager(leisaac_recorder.LeRobotRecorderManager):
        """leisaac's LeRobot recorder plus the verification columns.

        leisaac builds the dataset features inside ``__init__`` with no
        extension hook, so the feature builder is wrapped for the duration of
        the base constructor only. Tracked objects come from
        ``env.cfg.tracked_object_names``, which must not be empty.
        """

        def __init__(self, cfg, dataset_cfg, env):
            object_names = list(getattr(env.cfg, "tracked_object_names", []))
            if not object_names:
                raise ValueError(
                    f"{type(env.cfg).__name__}.tracked_object_names is empty: no object positions would be recorded "
                    "and the dataset could not be verified. List the task's rigid objects there."
                )
            base_build = leisaac_recorder.build_feature_from_env

            def build_with_factors(env, dataset_cfg):
                features = base_build(env, dataset_cfg)
                features.update(object_position_features(object_names))
                # Append the feature specs of further verification factors here.
                return features

            leisaac_recorder.build_feature_from_env = build_with_factors
            try:
                super().__init__(cfg, dataset_cfg, env)
            finally:
                leisaac_recorder.build_feature_from_env = base_build

    return LeRobotRecorderManager


def __getattr__(name: str):
    # Lazy so this module stays importable without leisaac/IsaacLab.
    if name == "LeRobotRecorderManager":
        return _make_recorder_manager_class()
    raise AttributeError(name)
