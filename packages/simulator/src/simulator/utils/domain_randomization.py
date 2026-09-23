from __future__ import annotations

import random
from typing import TYPE_CHECKING

import torch

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
