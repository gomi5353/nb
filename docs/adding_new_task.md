# Adding a New Task

This guide describes how an Advanced Level group defines its own manipulation
task: how a task configuration is built, how it is registered under a task id,
and how it is tested before recording. It extends step 1 of
[Getting Started](getting_started.md). Entry Level groups use the three
predefined tasks and do not need this guide.

The running example is the cup-stacking task:

- [`template/single_arm_franka_cfg.py`](../packages/simulator/src/simulator/tasks/template/single_arm_franka_cfg.py):
  the shared base configuration that every task subclasses.
- [`cup_stacking/cup_stacking_env_cfg.py`](../packages/simulator/src/simulator/tasks/cup_stacking/cup_stacking_env_cfg.py):
  a complete task built on the template.
- [`cup_stacking/__init__.py`](../packages/simulator/src/simulator/tasks/cup_stacking/__init__.py):
  its registration.
- [`utils/domain_randomization.py`](../packages/simulator/src/simulator/utils/domain_randomization.py):
  the reset events that randomize object positions and lighting.

Every section starts with one of two labels: **Required** (something students
must do or follow) or **Background** (explanation without work items).

## 1. How a task is built

**Background**

Isaac Lab describes an environment as a tree of configuration classes
(`ManagerBasedRLEnvCfg`). Each branch configures one *manager*:

| Manager | What it owns |
|---|---|
| `scene` | USD assets, the robot, sensors, lights: everything that exists in the simulation |
| `observations` | What the environment returns at every step (joint states, camera images, ...) |
| `actions` | How the action passed to `step(action)` drives the robot joints and the gripper |
| `events` | Callbacks, mainly the randomization on every reset |
| `rewards` | Per-step rewards (empty here: imitation learning uses none) |
| `terminations` | When an episode ends: `time_out` and `success` |
| `recorders` | What the recorder captures from every step |

LeIsaac ([LightwheelAI/leisaac](https://github.com/LightwheelAI/leisaac)) adds
asset helpers, randomization helpers, and the LeRobot recorder on top of Isaac
Lab. The repository's `simulator` package adds the shared single-arm Franka
template and the three tasks.

The **template** `SingleArmFrankaTaskEnvCfg` already provides:

- the Franka robot, the `front` and `wrist` cameras (640×480 RGB, refreshed
  every 1/30 s), the end-effector frames, and a dome light;
- the observations (`joint_pos`, camera images, ...);
- the action setup `use_joint_position_actions()`: seven absolute joint
  targets plus one binary gripper command, with joint stiffness 80 and damping 4
  and gravity disabled on the robot links;
- `build_lerobot_frame`, which turns every step into a dataset row
  ([How to Record Episodes §5.4](how_to_record.md#54-what-each-frame-contains));
- `decimation = 1` (one simulation step of 1/60 s per `env.step()`) and a
  `time_out` termination.

A new task therefore only adds its scene objects, its success check, its
randomization, and a few settings.

## 2. Where the task lives

**Required**

A task is a standalone set of files in its own directory under
`packages/simulator/src/simulator/tasks/`:

```text
packages/simulator/src/simulator/tasks/<my_task>/
    __init__.py             # gym.register(...)
    <my_task>_env_cfg.py    # the task configuration
```

| Item | Pattern | Example |
|---|---|---|
| Directory | `<task_slug>` (snake_case) | `cup_stacking` |
| Configuration module | `<task_slug>_env_cfg.py` | `cup_stacking_env_cfg.py` |
| Configuration class | `<TaskName>EnvCfg` | `CupStackingEnvCfg` |
| Scene class | `<TaskName>SceneCfg` | `CupStackingSceneCfg` |
| Task id | `HCIS-<TaskName>-SingleArm-v<N>` | `HCIS-CupStacking-SingleArm-v0` |



## 3. Write the task configuration

**Required**

`<my_task>_env_cfg.py` defines three classes. Every class is a
`@configclass`; plain Python classes are not picked up by Isaac Lab.

### 3.1 Scene

The scene class subclasses `SingleArmFrankaTaskSceneCfg`, sets the background
scene, and adds one `RigidObjectCfg` per object:

```python
@configclass
class CupStackingSceneCfg(SingleArmFrankaTaskSceneCfg):
    scene: AssetBaseCfg = KITCHEN_CFG.replace(prim_path="{ENV_REGEX_NS}/Scene")
    blue_cup: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Scene/blue_cup",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(KITCHEN_OBJECTS_ROOT / "BlueCup" / "BlueCup.usd"),
            mass_props=MassPropertiesCfg(mass=0.1),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(8.95, 6.52, 1.0), rot=(1.0, 0.0, 0.0, 0.0)),
    )
    pink_cup: RigidObjectCfg = ...
```

- The attribute name (`blue_cup`) becomes the object's scene name, used as
  `env.scene["blue_cup"]` and as the dataset column `object_position.blue_cup`.
- `{ENV_REGEX_NS}` expands to each environment's own prim path. Per-environment
  objects use it; only assets shared by all environments use `/World/...`.
- Asset paths are built from `simulator.ASSETS_ROOT`
  (`packages/simulator/assets/`), which works from any working directory.
  Public 3D assets can come from the sources listed in the course
  specification.
- `init_state.pos` is relative to the environment origin, in metres.
  Quaternions are `(w, x, y, z)`; `(1, 0, 0, 0)` is no rotation.

### 3.2 Success check

The success check is a function that returns a boolean tensor of shape
`(num_envs,)`, wrapped in a `DoneTerm` named `success`:

```python
def blue_cup_on_top_pink_cup(env, blue_cup_cfg, pink_cup_cfg, xy_tolerance, z_range):
    # A difference of two positions in the same env needs no env_origins.
    offset = env.scene[blue_cup_cfg.name].data.root_pos_w - env.scene[pink_cup_cfg.name].data.root_pos_w
    done = torch.all(torch.abs(offset[:, :2]) <= xy_tolerance, dim=1)
    done &= (offset[:, 2] >= z_range[0]) & (offset[:, 2] <= z_range[1])
    return done


@configclass
class TerminationsCfg(SingleArmFrankaTerminationsCfg):
    success = DoneTerm(
        func=blue_cup_on_top_pink_cup,
        params={
            "blue_cup_cfg": SceneEntityCfg("blue_cup"),
            "pink_cup_cfg": SceneEntityCfg("pink_cup"),
            "xy_tolerance": 0.02,
            "z_range": (0.08, 0.10),
        },
    )
```

- Positions are compared after subtracting `env.scene.env_origins`; raw
  world positions differ per environment.
- Conditions are combined with `&` and `|` on tensors, not with Python `and` and
  `or`.
- `params` is passed to the function as keyword arguments; `SceneEntityCfg("blue_cup")`
  resolves to the object at runtime.
- The recorder keeps an episode only if this term is true when the episode
  ends, and rollout counts an episode as a success when it becomes true. It
  therefore defines what the task means.

### 3.3 Environment configuration

The environment class subclasses `SingleArmFrankaTaskEnvCfg`, plugs in the
scene and terminations, and sets the task-specific values:

```python
@configclass
class CupStackingEnvCfg(SingleArmFrankaTaskEnvCfg):
    scene: CupStackingSceneCfg = CupStackingSceneCfg(env_spacing=8.0)
    observations: SingleArmFrankaObservationsCfg = SingleArmFrankaObservationsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    task_description: str = "pick up the blue cup and place it on the pink cup."
    tracked_object_names: list[str] = ["blue_cup", "pink_cup"]

    def __post_init__(self) -> None:
        super().__post_init__()
        self.episode_length_s = 20                         # evaluation episode length
        self.viewer.eye = (0.8, 0.87, 0.67)                # GUI camera
        self.viewer.lookat = (0.4, -1.3, -0.2)
        self.scene.robot.init_state.pos = (9.0, 6.0, 0.9)  # robot base
        self.scene.robot.init_state.rot = (0.707, 0.0, 0.0, 0.707)
        self.scene.robot.init_state.joint_pos = {...}      # home pose
        self.scene.front.offset.pos = (...)                 # front camera aimed at the workspace
        self.scene.front.offset.rot = (...)
        parse_usd_and_create_subassets(KITCHEN_USD_PATH, self)
        domain_randomization(self, random_options=[...])   # section 3.4
```

| Setting | Purpose |
|---|---|
| `task_description` | Short sentence describing the goal. It is saved with every episode as the task text ([How to Record Episodes §5.4](how_to_record.md#54-what-each-frame-contains)) and is the instruction for language-conditioned policies. |
| `tracked_object_names` | Objects whose start position is randomized. The recorder writes their positions as `object_position.<name>` for data verification and refuses to start when the list is empty. |
| `fixed_object_names` | Objects that start at the same place in every episode (optional). They get no dataset column and are not verified; rollout logs their pose once in its log header, for reference. |
| `episode_length_s` | Time a policy gets per evaluation episode. |
| `env_spacing` | Distance between parallel environments; larger than the scene. |
| `scene.robot.init_state` | Robot base pose and home joint angles. |
| `scene.front.offset` | Pose of the front camera; it should show the whole workspace and the arm. |
| `parse_usd_and_create_subassets` | Makes objects inside a packaged USD scene addressable as `{ENV_REGEX_NS}/Scene/<name>`. Needed when the background scene already contains the objects. |
| `dynamic_reset_gripper_effort_limit` | `True` adapts the gripper force to the object nearest the gripper at every step; `False` keeps it constant. |

`__post_init__` always calls `super().__post_init__()` first, so that the
template's defaults are set before they are changed.

### 3.4 Randomization

The last call in `__post_init__` registers reset events that randomize the
scene at every `env.reset()`:

```python
domain_randomization(
    self,
    random_options=[
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

- `randomize_objects_apart` offsets every listed object from its `init_state`
  by a uniform sample from its `pose_ranges` entry (keys `x`, `y`, `z`, `roll`,
  `pitch`, `yaw`). When two objects end up closer than `min_distance` (xy
  distance of their origins), that environment is resampled, up to
  `max_retries` times (default 5). `min_distance` is at least the sum of the
  two largest object radii, so that objects never spawn inside each other.
- For a single object, leisaac's `randomize_object_uniform(name, pose_range)`
  does the same without the distance check.
- `randomize_light_conditions` changes the light intensity (and optionally the
  color temperature) at every reset.

Without randomization, every episode starts identically and the coverage check
finds all episodes in one cell.

## 4. Register the task

**Required**

Registration gives the configuration a task id, so that `gym.make`,
`parse_env_cfg`, the collection script, and rollout can find it. Two edits are
needed.

1. `<my_task>/__init__.py` registers the id:

   ```python
   import gymnasium as gym

   gym.register(
       id="HCIS-MyTask-SingleArm-v0",
       entry_point="isaaclab.envs:ManagerBasedRLEnv",
       disable_env_checker=True,
       kwargs={"env_cfg_entry_point": f"{__name__}.my_task_env_cfg:MyTaskEnvCfg"},
   )
   ```

   `entry_point` is always `isaaclab.envs:ManagerBasedRLEnv`; only
   `env_cfg_entry_point` changes.

2. [`packages/simulator/src/simulator/tasks/__init__.py`](../packages/simulator/src/simulator/tasks/__init__.py)
   imports the new directory, next to the existing tasks:

   ```python
   from . import my_task  # noqa: F401
   ```

   Tasks are not discovered automatically: without this line,
   `import simulator.tasks` does not run the `gym.register` call and the task
   id is unknown.

The version suffix `v<N>` is increased whenever a change makes old datasets
incompatible with the task (for example new objects or a different success
check), so that each dataset stays associated with the task it was recorded
in.

## 5. Verify and test the task

**Required**

All checks run inside the Isaac Lab container, because Isaac Lab modules can
only be imported after the simulator has started.

1. **Test new task configuration:**

   ```bash
   python -c "
   from isaaclab.app import AppLauncher
   app = AppLauncher(headless=True).app
   import simulator.tasks
   from isaaclab_tasks.utils import parse_env_cfg
   cfg = parse_env_cfg('HCIS-MyTask-SingleArm-v0', num_envs=1)
   print(type(cfg).__module__, type(cfg).__qualname__)
   print(cfg.task_description, cfg.tracked_object_names)
   app.close()
   "
   ```

   This should print `simulator.tasks.my_task.my_task_env_cfg`.

2. Run the group's collection script on the new task id
   without `--record` and with the GUI
   ([How to Record Episodes §6](how_to_record.md#6-run-the-collection-script)).
   The objects should appear, both cameras show the workspace, and the objects land
   at a different position after every reset.

3. **Make sure the success check works:** a test attempt that solves the task makes the
   `success` term true, and one that does not leaves it false.

After these checks, recording proceeds as for a predefined task, from step 2 of
[Getting Started](getting_started.md#5-step-2-record-demonstrations).

## 6. Evaluate on the new task

**Background**

For a separate evaluation configuration (other nominal poses, ranges, or
lighting than in recording), the group writes an evaluation file in the same
form as the files in `eval/`: a configuration class plus a `gym.register(...)`
call with a unique `TASK_ID`, passed to rollout as a path
([Rollout §5](rollout.md#5-evaluating-a-self-defined-task)).
The evaluation protocol of a self-defined task is part of the group's
reproducibility documentation.

## 7. Common pitfalls

**Background**

- **Wrong prim namespace.** Per-environment assets need `{ENV_REGEX_NS}/...`.
- **Missing `env.scene.env_origins`.** A success check that compares raw world
  positions works for environment 0 only.
- **Quaternion order.** Isaac Lab uses `(w, x, y, z)`; SciPy and ROS often use
  `(x, y, z, w)`.
- **Task not imported.** The task id is unknown because the import line in
  `simulator/tasks/__init__.py` is missing ([section 4](#4-register-the-task)).
- **New camera.** Every camera in the scene becomes a dataset column and needs
  a matching observation term; the policy contract only allows `front` and
  `wrist`.
- **Empty `tracked_object_names`.** The recorder raises a `ValueError` and
  nothing can be verified.

## 8. Checklist

**Required**

- [ ] Files at `packages/simulator/src/simulator/tasks/<my_task>/`, named by the conventions in section 2.
- [ ] Scene, terminations, and environment classes are `@configclass` subclasses of the template classes.
- [ ] Success check uses env-local positions (subtracts `env.scene.env_origins`, or compares two objects of the same env) and returns a `(num_envs,)` boolean tensor.
- [ ] `task_description`, `tracked_object_names`, and `episode_length_s` are set.
- [ ] `__post_init__` calls `super().__post_init__()` first.
- [ ] Reset randomization of object positions (and lighting) is registered.
- [ ] `gym.register` in the task's `__init__.py`, and the import line in `simulator/tasks/__init__.py`.
- [ ] `parse_env_cfg(<task id>)` loads the class from the new file.
- [ ] The scene runs, the cameras show the workspace, and the success check behaves as intended.
- [ ] The rollout contract settings are unchanged; the template and `dependencies/IsaacLab/` are not edited.
