# Isaac Lab + LeIsaac Configuration Tutorial

A walkthrough of how a task is wired together in this repo, aimed at newcomers who have read the Isaac Lab docs but never opened our `packages/simulator/` tree.

The running example is `HCIS-CupStacking-SingleArm-v0`. Files referenced:

- `packages/simulator/src/simulator/tasks/template/single_arm_franka_cfg.py` — reusable base config for any single-arm Franka task.
- `packages/simulator/src/simulator/tasks/cup_stacking/cup_stacking_env_cfg.py` — concrete cup-stacking task built on top of the template.
- `packages/simulator/src/simulator/tasks/cup_stacking/__init__.py` — gym registration.
- `packages/simulator/src/simulator/utils/domain_randomization.py` — reset events that randomize object poses and lighting.

---

## 1. Mental model

Isaac Lab uses a **manager-based RL env** (`ManagerBasedRLEnvCfg`). A task is a tree of `@configclass` dataclasses describing:

| Manager | What it owns |
|---------|--------------|
| `scene` | USD assets, robot, sensors, lights (everything you see) |
| `observations` | Per-step obs terms (joints, images, last action, …) |
| `actions` | How `step(action)` maps onto robot joints / gripper |
| `events` | Reset/randomization callbacks |
| `rewards` | Per-step reward terms |
| `terminations` | Done conditions |
| `recorders` | What gets logged into demo episodes |

LeIsaac ([LightwheelAI/leisaac](https://github.com/LightwheelAI/leisaac)) sits on top of Isaac Lab and adds:

- Robot & scene asset configs (`leisaac.assets.*`).
- Teleop devices and an `action_process` helper.
- A LeRobot dataset handler that streams demos straight into a `lerobot` HF dataset.
- Convenience helpers like `parse_usd_and_create_subassets` to lift sub-prims out of a complex USD scene into Isaac Lab `RigidObjectCfg` entries.

Our `simulator/` package is a thin layer on top of LeIsaac that hosts project-specific task configs.

---

## 2. The single-arm Franka template

`single_arm_franka_cfg.py` is the base every Franka task should subclass. It pre-wires sensors, observations, the action setup, and the LeRobot frame builder, so a new task only has to specify scene assets and termination logic.

### 2.1 Scene (`SingleArmFrankaTaskSceneCfg`)

```python
class SingleArmFrankaTaskSceneCfg(InteractiveSceneCfg):
    scene: AssetBaseCfg = MISSING                         # subclass fills in
    robot: ArticulationCfg = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    ee_frame: FrameTransformerCfg = ...                   # end-effector + finger frames
    wrist:    TiledCameraCfg     = ...                    # 640×480 RGB on panda_hand
    front:    TiledCameraCfg     = ...                    # 640×480 RGB front view
    light = AssetBaseCfg(spawn=sim_utils.DomeLightCfg(...))
```

Key idioms:

- `{ENV_REGEX_NS}` expands per parallel env. Anything bound under `/World/...` (e.g. `front` camera) is **shared** across envs — useful for fixed external cameras, dangerous if you wanted per-env randomization.
- `ee_frame` declares three named frames so MDP terms can reference `panda_hand`, `tool_leftfinger`, `tool_rightfinger` without hardcoding prim paths.
- Cameras are `TiledCameraCfg`, which renders all envs in a single batched draw.

### 2.2 Observations (`SingleArmFrankaObservationsCfg`)

A single `policy` group exposes:

- `joint_pos`, `joint_vel`, `joint_pos_rel`, `joint_vel_rel`
- `actions` (last action returned by the env)
- `wrist`, `front` RGB images (raw uint8, `normalize=False`)
- `joint_pos_target` (most recent target sent to the controller — useful when training on action-following datasets)

`enable_corruption = True` lets `mdp` corruption terms attach noise. `concatenate_terms = False` keeps each term as its own dict entry, which is what LeRobot expects.

### 2.3 Actions

`SingleArmFrankaActionsCfg` declares `arm_action` and `gripper_action` as `MISSING`. They are filled in by `use_joint_position_actions()`, which every recording script and `scripts/rollout.py` call before creating the env, so data collection and evaluation drive the robot the same way:

```python
self.actions.arm_action = mdp.JointPositionActionCfg(
    asset_name="robot",
    joint_names=["panda_joint.*"],
    scale=1.0,
    use_default_offset=False,
)
self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
    asset_name="robot",
    joint_names=["panda_finger_joint.*"],
    open_command_expr={"panda_finger_joint.*": 0.04},
    close_command_expr={"panda_finger_joint.*": 0.0},
)
self.scene.robot.spawn.rigid_props.disable_gravity = True
```

An action is seven absolute joint targets plus one gripper command (negative closes, otherwise open). The joint gains are the `FRANKA_PANDA_CFG` defaults (stiffness 80, damping 4). Gravity is disabled on the robot so the arm holds its pose between commands. If the task sets `dynamic_reset_gripper_effort_limit = True`, the same call also adds an event that adapts the gripper effort limit every step.

### 2.4 Events, rewards, terminations

- `events.reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")` — on every `reset()`, snap everything back to its `init_state`.
- `rewards = SingleArmFrankaRewardsCfg()` — empty by default; collect demos, no shaped reward.
- `terminations.time_out = DoneTerm(func=mdp.time_out, time_out=True)` — episode caps at `episode_length_s` (15 s in the template; each HCIS task sets its graded length: cup stacking 20 s, cutlery arrangement 40 s, toy blocks collection 60 s).

### 2.5 LeRobot frame builder

`build_lerobot_frame` is the contract between Isaac Lab episode data and a LeRobot dataset row. Each call returns:

```python
{
    "action":             <last applied action>,
    "observation.state":  <last joint_pos>,
    "task":               <self.task_description>,
    "observation.images.<camera>": <rgb frame>,    # one per camera in dataset_cfg
}
```

The image keys are driven by `dataset_cfg.features` — keep `features` in sync with the cameras you declared in the scene, otherwise the recorder silently drops frames.

### 2.6 `__post_init__` defaults

```python
self.decimation = 1
self.episode_length_s = 15
self.viewer.eye = (1.4, -0.9, 1.2)
self.viewer.lookat = (2.0, -0.5, 1.0)
self.sim.physx.bounce_threshold_velocity = 0.01
self.sim.physx.friction_correlation_distance = 0.00625
self.sim.render.enable_translucency = True
self.default_feature_joint_names = [f"{j}.pos" for j in FRANKA_JOINT_NAMES]
```

`default_feature_joint_names` becomes the column ordering in LeRobot state vectors. Override `decimation` and `episode_length_s` on a per-task basis as needed.

---

## 3. The cup-stacking task

`cup_stacking_env_cfg.py` shows the minimum a new task has to provide.

### 3.1 Scene subclass

```python
class CupStackingSceneCfg(SingleArmFrankaTaskSceneCfg):
    scene: AssetBaseCfg = KITCHEN_CFG.replace(prim_path="{ENV_REGEX_NS}/Scene")
    blue_cup: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Scene/blue_cup",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(KITCHEN_OBJECTS_ROOT / "BlueCup" / "BlueCup.usd"),
            mass_props=MassPropertiesCfg(mass=0.1),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(8.95, 6.52, 1.0), rot=(1, 0, 0, 0)),
    )
    pink_cup: RigidObjectCfg = ...
```

- `KITCHEN_CFG` comes from `simulator.assets.scenes.kitchen` and brings in the kitchen USD plus its physics defaults.
- Cup USDs live under `packages/simulator/assets/scenes/kitchen/objects/`. The path is built from `simulator.ASSETS_ROOT`, which resolves to `packages/simulator/assets/` regardless of CWD.
- Quaternion convention is **(w, x, y, z)** everywhere in Isaac Lab — `(1, 0, 0, 0)` is identity.

### 3.2 Termination

```python
def blue_cup_on_top_pink_cup(env, blue_cup_cfg, pink_cup_cfg, x_range, y_range, height_threshold):
    blue = env.scene[blue_cup_cfg.name].data.root_pos_w - env.scene.env_origins
    pink = env.scene[pink_cup_cfg.name].data.root_pos_w - env.scene.env_origins
    done  = (blue[:, 0] > pink[:, 0] + x_range[0]) & (blue[:, 0] < pink[:, 0] + x_range[1])
    done &= (blue[:, 1] > pink[:, 1] + y_range[0]) & (blue[:, 1] < pink[:, 1] + y_range[1])
    done &= (blue[:, 2] > pink[:, 2] + height_threshold)
    return done
```

Notes:

- Always subtract `env.scene.env_origins` before comparing positions. Multi-env worlds are tiled in `x`/`y`, so raw `root_pos_w` is offset per env.
- Termination functions must return a boolean tensor of shape `(num_envs,)`. Combine via `&`/`|`, not Python `and`/`or`.
- `DoneTerm` takes the function plus a `params` dict. The dict is forwarded as keyword args, so `SceneEntityCfg("blue_cup")` resolves at runtime to the blue cup `RigidObject`.

### 3.3 Env subclass

```python
class CupStackingEnvCfg(SingleArmFrankaTaskEnvCfg):
    scene:        CupStackingSceneCfg               = CupStackingSceneCfg(env_spacing=8.0)
    observations: SingleArmFrankaObservationsCfg    = SingleArmFrankaObservationsCfg()
    terminations: TerminationsCfg                   = TerminationsCfg()
    task_description: str = "pick up the blue cup and place it on the pink cup."

    def __post_init__(self) -> None:
        super().__post_init__()
        self.episode_length_s = 20  # graded episode length for this task
        self.viewer.eye = (0.8, 0.87, 0.67)
        self.viewer.lookat = (0.4, -1.3, -0.2)
        self.dynamic_reset_gripper_effort_limit = False

        self.scene.robot.init_state.pos = (9.0, 6.0, 0.9)
        self.scene.robot.init_state.rot = (0.707, 0.0, 0.0, 0.707)
        self.scene.robot.init_state.joint_pos = { ... }     # home pose

        parse_usd_and_create_subassets(KITCHEN_USD_PATH, self)

        domain_randomization(self, random_options=[...])   # see 3.4
```

Things worth pointing out:

- `env_spacing=8.0` is the tile size between parallel envs. Pick something larger than the bounding radius of your scene.
- `task_description` is what gets baked into every LeRobot frame's `"task"` field — it is also the natural-language conditioning string for VLA policies. Keep it short and consistent across episodes.
- `parse_usd_and_create_subassets(KITCHEN_USD_PATH, self)` walks the kitchen USD and registers every `Xform` it finds as a Isaac Lab sub-asset. This is what lets you reference `{ENV_REGEX_NS}/Scene/blue_cup` even though the prim is buried inside the kitchen USD.
- `dynamic_reset_gripper_effort_limit = False` keeps the gripper effort limit constant. With `True`, `use_joint_position_actions()` adds an event that adapts it every step to the object nearest the gripper.

### 3.4 Randomization

The last call in `__post_init__` adds reset events that randomize the scene at every `env.reset()`:

```python
domain_randomization(
    self,
    random_options=[
        # Resample when the cups land closer than 11 cm (they would overlap).
        randomize_objects_apart(
            pose_ranges={
                "blue_cup": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (0.0, 0.0)},
                "pink_cup": {"x": (0.02, 0.05), "y": (0.02, 0.05), "z": (0.0, 0.0)},
            },
            min_distance=0.11,
        ),
        randomize_light_conditions("light", textures=[], intensity_range=(1100, 1300)),
    ],
)
```

- `domain_randomization` (from leisaac) just registers each entry as an event term.
- `randomize_objects_apart` (`simulator.utils.domain_randomization`) offsets every listed object from
  its `init_state` by a uniform sample from its `pose_ranges` entry (keys `x`, `y`, `z`, `roll`,
  `pitch`, `yaw`). If two objects end up closer than `min_distance` (xy distance between their root
  positions), that env is resampled, up to `max_retries` times (default 5). Pick `min_distance` at
  least as large as the sum of the two largest object radii, so objects never spawn inside each other.
- For a single object, leisaac's `randomize_object_uniform(name, pose_range)` does the same without the
  distance check.
- `randomize_light_conditions` changes the light intensity (and optionally color temperature) per reset.

The evaluation files in `eval/` use their own nominal poses and ranges. Read them to see which
situations your policy is graded on.

---

## 4. Gym registration

```python
# packages/simulator/src/simulator/tasks/cup_stacking/__init__.py
gym.register(
    id="HCIS-CupStacking-SingleArm-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.cup_stacking_env_cfg:CupStackingEnvCfg"},
)
```

- `entry_point` is **always** `isaaclab.envs:ManagerBasedRLEnv` for this style of task — only the `env_cfg_entry_point` changes.
- The id pattern is `HCIS-<TaskName>-<RobotKind>-v<N>`. Bump `vN` whenever you make a backwards-incompatible config change so old demo datasets stay associated with the right env.

---

## 5. Recipe: add a new single-arm Franka task

1. Create `packages/simulator/src/simulator/tasks/<my_task>/`.
2. `__init__.py`: copy the `gym.register(...)` block; point `env_cfg_entry_point` at your new module.
3. `<my_task>_env_cfg.py`:
   - Subclass `SingleArmFrankaTaskSceneCfg`; override `scene` and add per-object `RigidObjectCfg` fields.
   - Subclass `SingleArmFrankaTerminationsCfg`; add a `success` `DoneTerm` with your own predicate (return shape `(num_envs,)`, subtract `env_origins`).
   - Subclass `SingleArmFrankaTaskEnvCfg`; in `__post_init__` set viewer pose, robot home pose, `task_description`, and call `parse_usd_and_create_subassets` if you reuse a packaged USD scene.
   - Add reset events for object and light randomization (section 3.4).
4. Register the task module under `simulator.tasks` (auto-discovered when the package is imported).
5. Smoke test: run your collection script without recording (see
   [How to Record Episodes](how_to_record.md), section 4) on the new task id, in the container with
   `--enable_cameras`.

   If the env spawns and the cameras show frames, you're ready to record demos.

---

## 6. Common pitfalls

- **Prim path under the wrong namespace.** Per-env assets need `{ENV_REGEX_NS}/...`, not `/World/...`. World-fixed assets need `/World/...`. Mixing them up is the #1 cause of "I added an object but only env 0 sees it."
- **Forgot to subtract `env.scene.env_origins`.** Termination/reward predicates that pass on env 0 and silently fail on env ≥ 1 are almost always this bug.
- **Quaternion order.** Isaac Lab uses `(w, x, y, z)`. SciPy and ROS often use `(x, y, z, w)`. Convert when you copy poses from other tools.
- **LeRobot dataset feature drift.** If you add a camera to the scene but forget to add `observation.images.<name>` to `LeRobotDatasetCfg.features`, recorded demos will be missing that camera with no error.
- **Editing `dependencies/IsaacLab/` directly.** That's a submodule; commits there don't end up in this repo. Put project-specific overrides in `packages/simulator/`.

---

## 7. Where to read next

- [`standalone_env_config_export.md`](./standalone_env_config_export.md) — required reading if you self-implement an env config: how (and why) to export it as a standalone file so rollout/training can pick it up.
- `packages/simulator/src/simulator/tasks/template/single_arm_franka_cfg.py` — the file you'll subclass most often.
- [How to Record Episodes](how_to_record.md) — build the controller and recording loop for a task.
- [LightwheelAI/leisaac](https://github.com/LightwheelAI/leisaac) — upstream LeIsaac for the asset configs and the LeRobot dataset handler.
- `dependencies/IsaacLab/source/isaaclab/isaaclab/envs/manager_based_env_cfg.py` — the manager contract you're conforming to.
