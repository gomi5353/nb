"""Record per-frame object poses into LeRobot datasets for offline data verification.

Every tracked object gets its own dataset column ``object_pose.<name>``:
float32 ``(7,)`` = ``[x, y, z, qw, qx, qy, qz]``, position relative to the env
origin, quaternion in IsaacLab ``(w, x, y, z)`` order.

The ``object_pose.`` prefix is deliberate. LeRobot turns every ``observation.*``
key and every ``action*`` key into a policy input/output
(``lerobot.datasets.utils.dataset_to_policy_features``); any other column is
ignored by policies. These columns are therefore safe for training and rollout,
and are read back by ``packages/data_verification``.

Keep ``OBJECT_POSE_KEY_PREFIX`` / ``OBJECT_POSE_NAMES`` in sync with
``data_verification.dataset``.
"""

from __future__ import annotations

from functools import cache
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab.utils.datasets.episode_data import EpisodeData

OBJECT_POSE_KEY_PREFIX = "object_pose."
OBJECT_POSE_NAMES = ["x", "y", "z", "qw", "qx", "qy", "qz"]


def object_pose_features(object_names: list[str]) -> dict[str, dict]:
    """LeRobot feature specs for the ``object_pose.<name>`` columns."""
    return {
        f"{OBJECT_POSE_KEY_PREFIX}{name}": {
            "dtype": "float32",
            "shape": (len(OBJECT_POSE_NAMES),),
            "names": list(OBJECT_POSE_NAMES),
        }
        for name in object_names
    }


def read_object_poses(env: ManagerBasedEnv, object_names: list[str], env_index: int = 0) -> dict[str, np.ndarray]:
    """Current env-local object poses, keyed by dataset column.

    For custom recording pipelines: call once per recorded frame and merge the
    result into the frame dict passed to ``LeRobotDataset.add_frame``.
    """
    frame = {}
    for name in object_names:
        root_pose = env.scene[name].data.root_pose_w[env_index].clone()
        root_pose[:3] -= env.scene.env_origins[env_index]
        frame[f"{OBJECT_POSE_KEY_PREFIX}{name}"] = root_pose.cpu().numpy().astype(np.float32)
    return frame


def object_poses_from_episode(episode_data: EpisodeData, dataset_features: dict) -> dict[str, np.ndarray]:
    """Latest object poses from an IsaacLab ``EpisodeData`` buffer.

    Reads the ``states`` entry written by ``PostStepStatesRecorder`` (part of
    ``ActionStateRecorderManagerCfg``), which stores env-relative root poses.
    Only columns declared in ``dataset_features`` are returned.
    """
    rigid_states = episode_data._data.get("states", {}).get("rigid_object", {})
    frame = {}
    for key in dataset_features:
        if not key.startswith(OBJECT_POSE_KEY_PREFIX):
            continue
        name = key[len(OBJECT_POSE_KEY_PREFIX) :]
        frame[key] = rigid_states[name]["root_pose"][-1].cpu().numpy().astype(np.float32)
    return frame


@cache
def _make_recorder_manager_class():
    import leisaac.enhance.managers.lerobot_recorder_manager as leisaac_recorder

    class LeRobotRecorderManager(leisaac_recorder.LeRobotRecorderManager):
        """leisaac's LeRobot recorder plus ``object_pose.<name>`` columns.

        leisaac builds the dataset features inside ``__init__`` with no
        extension hook, so the feature builder is wrapped for the duration of
        the base constructor only. Tracked objects come from
        ``env.cfg.tracked_object_names``.
        """

        def __init__(self, cfg, dataset_cfg, env):
            base_build = leisaac_recorder.build_feature_from_env

            def build_with_object_poses(env, dataset_cfg):
                features = base_build(env, dataset_cfg)
                features.update(object_pose_features(list(getattr(env.cfg, "tracked_object_names", []))))
                return features

            leisaac_recorder.build_feature_from_env = build_with_object_poses
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
