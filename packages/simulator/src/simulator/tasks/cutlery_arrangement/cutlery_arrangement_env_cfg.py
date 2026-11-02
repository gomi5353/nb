import math

import isaaclab.sim as sim_utils
import torch

from isaaclab.assets import AssetBaseCfg, RigidObject, RigidObjectCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sim.schemas import MassPropertiesCfg
from isaaclab.utils import configclass

from leisaac.utils.general_assets import parse_usd_and_create_subassets
from leisaac.utils.domain_randomization import domain_randomization, randomize_object_uniform
from simulator import ASSETS_ROOT
from simulator.assets.scenes.dining_room import DINING_ROOM_CFG, DINING_ROOM_USD_PATH
from simulator.utils.domain_randomization import randomize_light_conditions

from simulator.tasks.template.single_arm_franka_cfg import (
    SingleArmFrankaObservationsCfg,
    SingleArmFrankaTaskEnvCfg,
    SingleArmFrankaTaskSceneCfg,
    SingleArmFrankaTerminationsCfg,
)

DINING_OBJECTS_ROOT = ASSETS_ROOT / "scenes" / "dining_room" / "objects"

# kujiale dining_table_0000 (light wood, convexDecomposition colliders, self-contained).
# Origin is at the table's vertical CENTER (spans z +-0.355), so place at z=0.355 to
# sit the base on the floor (top ends up at ~0.71).
DININGTABLE_USD = str(
    ASSETS_ROOT / "scenes" / "dining_table_0000" / "dining_table.usd"
)
DININGTABLE_WORLD_POS: tuple[float, float, float] = (7.0, 3.5, 0.354)
DININGTABLE_WORLD_ROT: tuple[float, float, float, float] = ( 0.70711, 0.0, 0.0, -0.70711)

# Fixed plate world position. The plate sits in front of the robot with
# sufficient free space on both x sides for the two drop targets.
PLATE_WORLD_POS: tuple[float, float, float] = (7.0, 2.9, 0.74416) # (0.50, -0.40, 0.05)


@configclass
class CutleryArrangementSceneCfg(SingleArmFrankaTaskSceneCfg):
    """Scene configuration for the cutlery arrangement task."""

    scene: AssetBaseCfg = DINING_ROOM_CFG.replace(prim_path="{ENV_REGEX_NS}/Scene")

    # Standalone white table (matte, static colliders baked in). Static furniture
    # -> AssetBaseCfg (no rigid body). Tune DININGTABLE_WORLD_POS for the scene.
    diningtable: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Scene/diningtable",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=DININGTABLE_WORLD_POS,
            rot=DININGTABLE_WORLD_ROT,
        ),
        spawn=sim_utils.UsdFileCfg(
            usd_path=DININGTABLE_USD,
        ),
    )

    plate: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Scene/plate",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=PLATE_WORLD_POS,
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(DINING_OBJECTS_ROOT / "Plate" / "plate.usd"),
            mass_props=MassPropertiesCfg(mass=0.1),
        ),
    )

    knife: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Scene/knife",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(DINING_OBJECTS_ROOT / "Knife" / "knife.usd"),
            mass_props=MassPropertiesCfg(mass=0.1),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(6.8, 3.05, 0.75), rot=(0.0, 0.0, 0.0, 1.0)),
    )

    fork: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Scene/fork",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(DINING_OBJECTS_ROOT / "Fork" / "fork.usd"),
            mass_props=MassPropertiesCfg(mass=0.1),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(7.1, 3.0, 0.75), rot=(1.0, 0.0, 0.0, 0.0)),
    )


def cutlery_arranged(
    env,
    plate_cfg: SceneEntityCfg,
    fork_cfg: SceneEntityCfg,
    knife_cfg: SceneEntityCfg,
    side_offset: float,
    half_size: float,
) -> torch.Tensor:
    """Success: fork and knife each inside a square of side ``2 * half_size`` beside the plate.

    The square centres lie ``side_offset`` from the plate centre along x: the fork's at -x (the
    robot faces +y, so -x is its left; the right side of the front camera image), the knife's at +x.
    """
    plate_pos = env.scene[plate_cfg.name].data.root_pos_w
    fork_offset = env.scene[fork_cfg.name].data.root_pos_w[:, :2] - plate_pos[:, :2]
    knife_offset = env.scene[knife_cfg.name].data.root_pos_w[:, :2] - plate_pos[:, :2]

    fork_target = torch.tensor([-side_offset, 0.0], device=env.device)
    knife_target = torch.tensor([side_offset, 0.0], device=env.device)
    fork_in = torch.all(torch.abs(fork_offset - fork_target) <= half_size, dim=1)
    knife_in = torch.all(torch.abs(knife_offset - knife_target) <= half_size, dim=1)
    return torch.logical_and(fork_in, knife_in)


@configclass
class TerminationsCfg(SingleArmFrankaTerminationsCfg):
    """Termination configuration for the cutlery arrangement task."""

    success = DoneTerm(
        func=cutlery_arranged,
        params={
            "plate_cfg": SceneEntityCfg("plate"),
            "fork_cfg": SceneEntityCfg("fork"),
            "knife_cfg": SceneEntityCfg("knife"),
            "side_offset": 0.12,  # square centre to plate centre (m)
            "half_size": 0.05,  # 10 cm x 10 cm square
        },
    )


@configclass
class CutleryArrangementEnvCfg(SingleArmFrankaTaskEnvCfg):
    """Configuration for the cutlery arrangement task environment."""

    scene: CutleryArrangementSceneCfg = CutleryArrangementSceneCfg(env_spacing=8.0)
    observations: SingleArmFrankaObservationsCfg = SingleArmFrankaObservationsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    task_description: str = "place the fork on the left and knife on the right of the plate."
    tracked_object_names: list[str] = ["fork", "knife", "plate"]

    def __post_init__(self) -> None:
        super().__post_init__()
        self.episode_length_s = 40  # graded episode length for this task

        self.viewer.eye = (0.8, 0.87, 0.67)
        self.viewer.lookat = (0.4, -1.3, -0.2)
        self.dynamic_reset_gripper_effort_limit = False

        self.scene.robot.init_state.pos = (7.0, 2.4, 0.6)
        self.scene.robot.init_state.rot = (0.707, 0.0, 0.0, 0.707)


        # Per-task front camera: pos (7.05,4.5,1.6), aimed at table workspace
        # (7,3.5,~1.05). rot computed as opengl look-at quaternion (w,x,y,z).
        self.scene.front.offset.pos = (7.05, 4.77904, 1.16108)
        self.scene.front.offset.rot = (0.00747, 0.00888, 0.64418, 0.76479)
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

        parse_usd_and_create_subassets(DINING_ROOM_USD_PATH, self)

        domain_randomization(
            self,
            random_options=[
                randomize_object_uniform(
                    "knife",
                    pose_range={
                        "x": (-0.05, 0.05),
                        "y": (-0.05, 0.05),
                        "z": (0.0, 0.0),
                    },
                ),
                randomize_object_uniform(
                    "fork",
                    pose_range={
                        "x": (0.02, 0.05),
                        "y": (0.02, 0.05),
                        "z": (0.0, 0.0),
                    },
                ),
                randomize_light_conditions("light", textures=[], intensity_range=(1100, 1300)),
            ],
        )
