from __future__ import annotations

import random
from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.utils.assets import NVIDIA_NUCLEUS_DIR

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

DEFAULT_HDR_TEXTURES = [
    f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Cloudy/abandoned_parking_4k.hdr",
    f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Cloudy/evening_road_01_4k.hdr",
    f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Cloudy/lakeside_4k.hdr",
    f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Indoor/autoshop_01_4k.hdr",
    f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Indoor/carpentry_shop_01_4k.hdr",
    f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Indoor/hospital_room_4k.hdr",
    f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Indoor/hotel_room_4k.hdr",
    f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Indoor/old_bus_depot_4k.hdr",
    f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Indoor/small_empty_house_4k.hdr",
    f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Indoor/surgery_4k.hdr",
    f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Studio/photo_studio_01_4k.hdr",
]


class randomize_light(ManagerTermBase):
    """Randomize a USD light (DomeLight, RectLight, ...) on each reset.

    Samples come from a private RNG seeded with ``env.cfg.seed`` and the asset
    name, so the lighting sequence is reproducible for a given seed and is not
    shifted by other consumers of the global ``random`` state (policy, other
    events). Each env in ``env_ids`` gets its own sample.
    """

    def __init__(self, cfg: EventTerm, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        asset_name = cfg.params["asset_cfg"].name
        seed = env.cfg.seed if env.cfg.seed is not None else 0
        self._rng = random.Random(f"{seed}:{asset_name}")

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | None,
        intensity_range: tuple[float, float],
        color_variation: float,
        textures: list[str],
        color_temperature_range: tuple[float, float] | None = None,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("light"),
    ):
        prims = env.scene[asset_cfg.name].prims
        # A global (non env-scoped) light has a single prim shared by all envs.
        if len(prims) == 1 or env_ids is None:
            ids = range(len(prims))
        else:
            ids = sorted(env_ids.tolist())

        rng = self._rng
        for i in ids:
            prim = prims[i]
            prim.GetAttribute("inputs:intensity").Set(rng.uniform(*intensity_range))

            offsets = [rng.uniform(-color_variation, color_variation) for _ in range(3)]
            avg = sum(offsets) / 3
            prim.GetAttribute("inputs:color").Set(tuple(max(0.0, min(1.0, 0.75 + o - avg)) for o in offsets))

            if color_temperature_range is not None:
                prim.GetAttribute("inputs:enableColorTemperature").Set(True)
                prim.GetAttribute("inputs:colorTemperature").Set(rng.uniform(*color_temperature_range))

            if textures:
                prim.GetAttribute("inputs:texture:file").Set(rng.choice(textures))


def randomize_light_conditions(
    name: str = "light",
    intensity_range: tuple[float, float] = (1500.0, 3000.0),
    color_variation: float = 0.4,
    textures: list[str] | None = None,
    color_temperature_range: tuple[float, float] | None = None,
) -> EventTerm:
    return EventTerm(
        func=randomize_light,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg(name),
            "intensity_range": intensity_range,
            "color_variation": color_variation,
            "textures": textures if textures is not None else DEFAULT_HDR_TEXTURES,
            "color_temperature_range": color_temperature_range,
        },
    )


_POSE_KEYS = ("x", "y", "z", "roll", "pitch", "yaw")


def reset_objects_apart(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    pose_ranges: dict[str, dict[str, tuple[float, float]]],
    min_distance: float,
    max_retries: int,
):
    """Reset several objects to uniform random poses that keep them apart.

    Each object is offset from its default root pose by a sample from its ``pose_ranges`` entry, like
    ``reset_root_state_uniform``. An env whose objects end up closer than ``min_distance`` (xy distance
    between root positions) is resampled, up to ``max_retries`` times. If it is still too close after
    that, the last sample is used.
    """
    names = list(pose_ranges)
    device = env.device
    num = len(env_ids)
    ranges = {
        name: torch.tensor([pose_ranges[name].get(key, (0.0, 0.0)) for key in _POSE_KEYS], device=device)
        for name in names
    }
    defaults = {name: env.scene[name].data.default_root_state[env_ids].clone() for name in names}

    def sample(count: int) -> dict[str, torch.Tensor]:
        return {
            name: math_utils.sample_uniform(ranges[name][:, 0], ranges[name][:, 1], (count, 6), device=device)
            for name in names
        }

    def too_close(offsets: dict[str, torch.Tensor], idx: torch.Tensor) -> torch.Tensor:
        xy = torch.stack([defaults[name][idx, :2] + offsets[name][:, :2] for name in names], dim=1)
        dist = torch.cdist(xy, xy)
        dist = dist + torch.eye(len(names), device=device) * min_distance
        return (dist < min_distance).flatten(1).any(dim=1)

    offsets = sample(num)
    redo = torch.nonzero(too_close(offsets, torch.arange(num, device=device))).flatten()
    for _ in range(max_retries):
        if len(redo) == 0:
            break
        resampled = sample(len(redo))
        for name in names:
            offsets[name][redo] = resampled[name]
        redo = redo[too_close(resampled, redo)]

    for name in names:
        asset = env.scene[name]
        root = defaults[name]
        positions = root[:, 0:3] + env.scene.env_origins[env_ids] + offsets[name][:, 0:3]
        rotations = math_utils.quat_from_euler_xyz(offsets[name][:, 3], offsets[name][:, 4], offsets[name][:, 5])
        orientations = math_utils.quat_mul(root[:, 3:7], rotations)
        asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
        asset.write_root_velocity_to_sim(root[:, 7:13], env_ids=env_ids)


def randomize_objects_apart(
    pose_ranges: dict[str, dict[str, tuple[float, float]]],
    min_distance: float,
    max_retries: int = 5,
) -> EventTerm:
    """Reset event: uniform random poses for several objects, resampled when two spawn too close.

    Args:
        pose_ranges: Object name -> pose range (keys ``x``, ``y``, ``z``, ``roll``, ``pitch``, ``yaw``), as offsets
            from the object's ``init_state``.
        min_distance: Smallest allowed xy distance (m) between any two of these objects' root positions.
        max_retries: How many times an env with objects too close is resampled.
    """
    return EventTerm(
        func=reset_objects_apart,
        mode="reset",
        params={"pose_ranges": pose_ranges, "min_distance": min_distance, "max_retries": max_retries},
    )
