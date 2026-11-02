import math

import gymnasium as gym
import isaaclab.sim as sim_utils
import torch
from isaaclab.assets import AssetBaseCfg, RigidObject, RigidObjectCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sim.schemas import MassPropertiesCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply_inverse
from isaaclab.utils.seed import configure_seed

from leisaac.utils.domain_randomization import domain_randomization
from leisaac.utils.general_assets import parse_usd_and_create_subassets
from simulator import ASSETS_ROOT
from simulator.assets.scenes.living_room import LIVING_ROOM_CFG, LIVING_ROOM_USD_PATH
from simulator.utils.domain_randomization import randomize_light_conditions, randomize_objects_apart
from simulator.tasks.template.single_arm_franka_cfg import (
    SingleArmFrankaObservationsCfg,
    SingleArmFrankaTaskEnvCfg,
    SingleArmFrankaTaskSceneCfg,
    SingleArmFrankaTerminationsCfg,
)

LIVING_OBJECTS_ROOT = ASSETS_ROOT / "scenes" / "living_room" / "objects"

TABLE_USD = str(ASSETS_ROOT / "scenes" / "table_0000" / "table_0000.usd")
TABLE_WORLD_POS: tuple[float, float, float] = (3.3, 1.1, 0.183)


configure_seed(42)


@configclass
class ToyBlocksCollectionEvalSceneCfg(SingleArmFrankaTaskSceneCfg):
    scene: AssetBaseCfg = LIVING_ROOM_CFG.replace(prim_path="{ENV_REGEX_NS}/Scene")
    # RectLight baked into nycu_light_collider_wooden.usd (/root/RectLight); wrap the existing prim.
    rect_light: AssetBaseCfg = AssetBaseCfg(prim_path="{ENV_REGEX_NS}/Scene/RectLight")

    table: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Scene/table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=TABLE_WORLD_POS),
        spawn=sim_utils.UsdFileCfg(usd_path=TABLE_USD),
    )

    green_block: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Scene/green_block",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(LIVING_OBJECTS_ROOT / "Bridge" / "Bridge.usd"),
            mass_props=MassPropertiesCfg(mass=0.1),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(3.45, 0.82, 0.45),
            rot=(0.707, 0.0, 0.0, 0.707),
        ),
    )

    blue_block: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Scene/blue_block",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(LIVING_OBJECTS_ROOT / "Cylinder" / "Cylinder.usd"),
            mass_props=MassPropertiesCfg(mass=0.1),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(3.33, 0.82, 0.45),
            rot=(0.707, 0.0, 0.0, 0.707),
        ),
    )

    red_block: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Scene/red_block",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(LIVING_OBJECTS_ROOT / "Triangle" / "Triangle.usd"),
            mass_props=MassPropertiesCfg(mass=0.1),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(3.39, 0.72, 0.45),
            rot=(0.26788, -0.2733, 0.65215, 0.65441),
        ),
    )

    storage_box: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Scene/storage_box",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(3.0, 0.8, 0.36348),
            rot=(0.707, 0.0, 0.0, 0.707),
        ),
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(LIVING_OBJECTS_ROOT / "Storage_Box" / "storage_box.usd"),
            mass_props=MassPropertiesCfg(mass=0.1),
        ),
    )


# Success when all three blocks lie within (x,y,z)_range of storage_box, in the box frame.
def toys_in_box(
    env,
    green_block_cfg: SceneEntityCfg,
    blue_block_cfg: SceneEntityCfg,
    red_block_cfg: SceneEntityCfg,
    storage_box_cfg: SceneEntityCfg,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    z_range: tuple[float, float],
) -> torch.Tensor:
    """Success: every block root within (x, y, z)_range of the storage box root, in the box's own frame.

    The block offset from the box root is rotated into the box frame, so the ranges follow the box
    when it is rotated (for example yawed by 90 degrees).
    """
    storage_box: RigidObject = env.scene[storage_box_cfg.name]
    box_pos = storage_box.data.root_pos_w
    box_quat = storage_box.data.root_quat_w

    done = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)

    for block_cfg in (green_block_cfg, blue_block_cfg, red_block_cfg):
        block: RigidObject = env.scene[block_cfg.name]
        offset = quat_apply_inverse(box_quat, block.data.root_pos_w - box_pos)
        for axis, (low, high) in enumerate((x_range, y_range, z_range)):
            done = torch.logical_and(done, offset[:, axis] > low)
            done = torch.logical_and(done, offset[:, axis] < high)

    return done


@configclass
class EvalTerminationsCfg(SingleArmFrankaTerminationsCfg):
    success = DoneTerm(
        func=toys_in_box,
        params={
            "green_block_cfg": SceneEntityCfg("green_block"),
            "blue_block_cfg": SceneEntityCfg("blue_block"),
            "red_block_cfg": SceneEntityCfg("red_block"),
            "storage_box_cfg": SceneEntityCfg("storage_box"),
            # Ranges in the storage box's own frame. Footprint relative to its root (measured: x -0.1525..0.1432,
            # y -0.1059..0.1032, i.e. 0.296 m x 0.209 m), shrunk by 1 cm on every side so a block must be
            # inside the walls. z: above the box root (box bottom) and below it + 11 cm (box is 0.109 m tall).
            "x_range": (-0.1425, 0.1332),
            "y_range": (-0.0959, 0.0932),
            "z_range": (0.0, 0.11),
        },
    )


@configclass
class ToyBlocksCollectionEvalEnvCfg(SingleArmFrankaTaskEnvCfg):
    scene: ToyBlocksCollectionEvalSceneCfg = ToyBlocksCollectionEvalSceneCfg(env_spacing=8.0)
    observations: SingleArmFrankaObservationsCfg = SingleArmFrankaObservationsCfg()
    terminations: EvalTerminationsCfg = EvalTerminationsCfg()
    task_description: str = "pick up the toys and place them into the storage box."
    tracked_object_names: list[str] = ["green_block", "blue_block", "red_block", "storage_box"]

    def __post_init__(self) -> None:
        super().__post_init__()
        self.episode_length_s = 60  # graded episode length for this task

        self.viewer.eye = (0.8, 0.87, 0.67)
        self.viewer.lookat = (0.4, -1.3, -0.2)
        self.dynamic_reset_gripper_effort_limit = False

        self.scene.robot.init_state.pos = (3.3, 0.35, 0.28)
        self.scene.robot.init_state.rot = (0.707, 0.0, 0.0, 0.707)

        self.scene.front.offset.pos = (3.3, 2.55, 1.1)
        self.scene.front.offset.rot = (0.0, 0.0, -0.58292, -0.81253)
        self.scene.front.offset.convention = "opengl"
        self.scene.front.spawn.focal_length = 55

        self.scene.robot.init_state.joint_pos = {
            "panda_joint1": 0.0,
            "panda_joint2": -math.pi / 4.0,
            "panda_joint3": 0.0,
            "panda_joint4": -3.0 * math.pi / 4.0,
            "panda_joint5": 0.0,
            "panda_joint6": math.pi / 2.0,
            "panda_joint7": math.pi / 4.0,
            "panda_finger_joint1": 0.04,
            "panda_finger_joint2": 0.04,
        }

        parse_usd_and_create_subassets(LIVING_ROOM_USD_PATH, self)

        domain_randomization(
            self,
            random_options=[
                # Blocks start 12 cm apart; resample when two land closer than 8 cm (they would overlap).
                randomize_objects_apart(
                    pose_ranges={
                        "green_block": {"x": (-0.03, 0.03), "y": (-0.03, 0.03), "z": (0.0, 0.0)},
                        "blue_block": {"x": (-0.03, 0.03), "y": (-0.03, 0.03), "z": (0.0, 0.0)},
                        "red_block": {"x": (-0.03, 0.03), "y": (-0.03, 0.03), "z": (0.0, 0.0)},
                    },
                    min_distance=0.08,
                ),
                randomize_light_conditions(
                    "rect_light",
                    intensity_range=(2500.0, 8000.0),
                    color_variation=0.0,
                    textures=[],
                    color_temperature_range=(3000.0, 6500.0),
                ),
            ],
        )


TASK_ID = "Public-ToyBlocksCollection-Eval-v0"

gym.register(
    id=TASK_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}:ToyBlocksCollectionEvalEnvCfg"},
)
