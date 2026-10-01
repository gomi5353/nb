# How to Record Episodes

This guide shows you how to turn your controller into a pipeline that writes a
LeRobot v3 dataset. Your controller can be a scripted state machine or a person
using the keyboard. The main path uses a LeRobot recorder connected to the
Isaac Lab environment and saves each successful attempt as one dataset episode.
[Section 8](#8-advanced-option-write-the-dataset-yourself) shows a
second way, where you write the dataset yourself with LeRobot's
`LeRobotDataset` API.

Students are expected to implement the collection entry point for their task.
The entry point may use a scripted controller, keyboard teleoperation, or
another input source, but it must connect that controller to the simulator and
the LeRobot writer. Use this guide when designing the collection behaviour and
checking the resulting dataset.

> **Run on:** a Linux machine with an NVIDIA GPU, inside the Isaac Lab
> container (see [Getting Started](getting_started.md)). Isaac Sim rendering
> and the required camera observations are not supported by the host-only
> `uv` environment. Recording always needs `--enable_cameras`.

## 0. Choose the collection design

For one of the HCIS tasks, the recommended workflow is:

1. Read the task configuration and identify its scene objects, cameras,
   action space, and success predicate.
2. Implement a controller/state machine and a recording loop in your own
   collection script.
3. Connect the loop to a LeRobot v3 writer and record one test episode.

Use keyboard teleoperation when the demonstration should be driven by a
person. A custom recorder is appropriate when the required episode logic or
metadata is not provided by your base template.

## 1. Know your task

You have already picked one task. Here is what the environment gives you:

| | Cup stacking | Cutlery arrangement | Toy blocks collection |
|---|---|---|---|
| Task id | `HCIS-CupStacking-SingleArm-v0` | `HCIS-CutleryArrangement-SingleArm-v0` | `HCIS-ToyBlocksCollection-SingleArm-v0` |
| Objects (`env.scene[name]`) | `blue_cup`, `pink_cup` | `plate`, `fork`, `knife` | `green_block`, `blue_block`, `red_block`, `storage_box` |
| Task text saved in the dataset | `pick up the blue cup and place it on the pink cup.` | `place the fork on the left and knife on the right of the plate.` | `pick up the toys and place them into the storage box.` |
| Built-in success check | Blue cup within 5 cm of the pink cup in x and y, and more than 10 cm higher | Fork and knife both within 15 cm (xy) of the plate; fork on the +x side of the plate, knife on the −x side | Every block within 12 cm (x, y) and 8 cm (z) of the box |
| Defined in | [`cup_stacking_env_cfg.py`](../packages/simulator/src/simulator/tasks/cup_stacking/cup_stacking_env_cfg.py) | [`cutlery_arrangement_env_cfg.py`](../packages/simulator/src/simulator/tasks/cutlery_arrangement/cutlery_arrangement_env_cfg.py) | [`toy_blocks_collection_env_cfg.py`](../packages/simulator/src/simulator/tasks/toy_blocks_collection/toy_blocks_collection_env_cfg.py) |

The shared robot, camera, action, and timing defaults are defined in
[`single_arm_franka_cfg.py`](../packages/simulator/src/simulator/tasks/template/single_arm_franka_cfg.py).
The task-specific front-camera pose, object initial poses, randomization ranges,
and success predicate are in the three task config files above. Keep the
controller separate from dataset-writing code so that the task logic can be
tested without recording.

The shared config uses one simulation step per control step
(`decimation = 1`) and a 15-second environment timeout. Your collection loop
may disable that timeout and end an episode when the controller reports
`is_episode_done`; therefore every custom controller must include its own
phase/episode limit.

Every reset shifts the objects by a few centimetres and changes the light
intensity. Your controller must read object positions at the start of every
episode. Never hard-code them.

The built-in success check only looks at object positions, at a single moment.
It doesn't know whether the gripper is still holding something. In cup stacking,
for example, the check can already pass while the blue cup is still in the
gripper, hovering above the pink cup. Decide what "done" means for your
demonstrations (for example: object released, arm moved clear, check still
passes) and make your script enforce it.

## 2. How the recorder works

The recorder runs inside `env.step()` and `env.reset()`. You never call it
directly, except `finalize()` at the very end.

```text
env.reset()                  a new, empty episode buffer starts
    │
env.step(action)             repeated
    ├─ before physics:       observation and action are captured
    ├─ physics runs
    └─ after physics:        frame t = (what the robot saw, what it was told to do)
                             is added to the buffer
                             (the first 5 steps of every episode are skipped)
    │
env.reset()                  the recorder reads the termination term "success"
    ├─ True  → episode is written to disk   (save_episode)
    └─ False → episode is thrown away       (clear_episode_buffer)
    │
recorder_manager.finalize()  files are closed and the dataset can be loaded
```

This design has some consequences you need to plan for (source:
`leisaac/enhance/managers/lerobot_recorder_manager.py`):

1. **One frame per `env.step()`.** If you don't step, nothing is recorded.
2. **The first 5 steps after every reset are skipped.** Camera images from
   before the reset can still show up there. Make your controller hold still
   for those steps.
3. **Keep or drop is decided at the next reset**, by the `success` termination
   term. Any call to `env.reset()` ends the current episode.
4. **Only environment 0 is recorded.** Always use `num_envs=1`.
5. **Export mode must be `EXPORT_SUCCEEDED_ONLY`** (or its resume variant).
   Failed episodes never reach the disk.
6. **`finalize()` is mandatory.** Without it the parquet files have no footer
   and the dataset can't be loaded.
7. **The dataset goes to `$HF_LEROBOT_HOME/<repo_id>`** (by default
   `~/.cache/huggingface/lerobot/<repo_id>`). If that folder already exists,
   creating the dataset fails with `FileExistsError`.

### 2.1 Use the repository's `LeRobotRecorderManager`

Import the recorder from this repository, **not** from leisaac:

```python
from simulator.utils.object_pose_recording import LeRobotRecorderManager       # use this
# from leisaac.enhance.managers.lerobot_recorder_manager import LeRobotRecorderManager  # not this
```

[`object_pose_recording.py`](../packages/simulator/src/simulator/utils/object_pose_recording.py)
defines a modified version of leisaac's `LeRobotRecorderManager`. Episode
handling is the same; the difference is that it also records object poses,
which the [object-pose coverage check](object_pose_coverage.md) needs:

- At construction it adds one `object_pose.<name>` feature,
  float32 `(7,)` = `[x, y, z, qw, qx, qy, qz]`, for every name in the task's
  `tracked_object_names` (for example `blue_cup`, `pink_cup` in cup stacking).
- Every frame, `build_lerobot_frame` fills those columns with the object pose
  relative to the env origin.
- The coverage checker reads each episode's **first recorded frame**, so these
  columns tell it where every object started in that episode.

The `object_pose.` prefix is ignored by LeRobot policies. Only `observation.*`
and `action*` keys become policy inputs and outputs, so these columns do not
affect training or rollout. If you use leisaac's original class, recording
still works, but the columns are missing and the coverage check can't run.

The object poses come from the `states` buffer that Isaac Lab's
`ActionStateRecorderManagerCfg` writes. That config is the task default
(`env_cfg.recorders`). Keep it, and only change its export mode.

leisaac's own state-machine data generation script shows how the recorder is
wired in:
[`scripts/datagen/state_machine/generate.py#L153`](https://github.com/LightwheelAI/leisaac/blob/24d3bcd3f1e4585740fc79921782c41617237812/scripts/datagen/state_machine/generate.py#L153).
Below is the same pattern reduced to what an HCIS task needs. App launch and
argument parsing are left out, and `controller` is the controller from
[section 4](#4-implement-the-collection-pipeline).

```python
import gymnasium as gym
import torch
from isaaclab.managers import DatasetExportMode, TerminationTermCfg
from isaaclab_tasks.utils import parse_env_cfg
from leisaac.enhance.datasets.lerobot_dataset_handler import LeRobotDatasetCfg
from leisaac.enhance.managers import EnhanceDatasetExportMode

import simulator.tasks  # noqa: F401  registers the HCIS-* tasks
from simulator.utils.object_pose_recording import LeRobotRecorderManager


def set_success(env, success: bool):
    """Overwrite the "success" term. The recorder reads it at the next env.reset()."""
    fill = torch.ones if success else torch.zeros
    env.termination_manager.set_term_cfg(
        "success",
        TerminationTermCfg(func=lambda env: fill(env.num_envs, dtype=torch.bool, device=env.device)),
    )
    env.termination_manager.compute()


# 1. Configure the env before it is created.
env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
env_cfg.terminations.time_out = None  # the controller decides when an attempt ends
env_cfg.terminations.success = TerminationTermCfg(  # starts False; set_success() flips it
    func=lambda env: torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
)
env_cfg.recorders.dataset_export_mode = (  # keep env_cfg.recorders itself, see above
    EnhanceDatasetExportMode.EXPORT_SUCCEEDED_ONLY_RESUME if args.resume
    else DatasetExportMode.EXPORT_SUCCEEDED_ONLY
)
env = gym.make(args.task, cfg=env_cfg).unwrapped

# 2. Replace Isaac Lab's default (HDF5) recorder with the LeRobot one.
del env.recorder_manager
env.recorder_manager = LeRobotRecorderManager(
    env_cfg.recorders,
    LeRobotDatasetCfg(repo_id=args.dataset_repo_id, fps=30),
    env,
)

# 3. Recording loop: step, then judge the attempt and reset.
controller.setup(env)
try:
    with torch.inference_mode():
        env.reset()
        controller.reset()
        while env.recorder_manager.exported_successful_episode_count < args.num_episodes:
            controller.pre_step(env)
            env.step(controller.get_action(env))  # the recorder adds one frame here
            controller.advance()
            if controller.is_episode_done:
                set_success(env, controller.check_success(env))
                env.reset()  # saved if success is True, discarded otherwise
                set_success(env, False)
                controller.reset()
finally:
    env.recorder_manager.finalize()  # mandatory, see rule 6
    env.close()
```

Notes on the loop:

- `set_success(env, True)` must be followed directly by `env.reset()`. If you
  call `env.step()` while the term is `True`, the env terminates and
  auto-resets on its own.
- The episode is exported in `env.reset()`, not in `finalize()`. Reset after
  the last attempt too, or that episode is lost.
- `exported_successful_episode_count` only counts episodes saved in the
  current session. When resuming, add the number that was already in the
  dataset.

### What each frame contains

The recorder builds the dataset features from the environment. It uses
`build_feature_from_env` and `SingleArmFrankaTaskEnvCfg.build_lerobot_frame` in
[`single_arm_franka_cfg.py`](../packages/simulator/src/simulator/tasks/template/single_arm_franka_cfg.py).

| Feature | Shape / type | Content |
|---|---|---|
| `observation.state` | `(9,)` float32 | Measured joint positions: `panda_joint1..7` (rad), `panda_finger_joint1..2` (m) |
| `action` | `(8,)` float32, names `dim_0..dim_7` | `dim_0..dim_6`: target positions for `panda_joint1..7` (rad, absolute). `dim_7`: gripper command (≥ 0 open, < 0 close) |
| `observation.images.front` | 480×640×3 video | Front camera |
| `observation.images.wrist` | 480×640×3 video | Wrist camera |
| `task` → `task_index` | string | `env_cfg.task_description` |
| `object_pose.<name>` | `(7,)` float32, names `x, y, z, qw, qx, qy, qz` | Pose of each `tracked_object_names` object, relative to the env origin. Only written by the repository recorder ([section 2.1](#21-use-the-repositorys-lerobotrecordermanager)) |

These features match the simulator-side LeRobot frame builder in
[`single_arm_franka_cfg.py`](../packages/simulator/src/simulator/tasks/template/single_arm_franka_cfg.py).
The object-pose coverage checker needs the `object_pose.<name>` columns. See
[`object_pose_coverage.md`](object_pose_coverage.md) for how it uses them.
Don't add cameras to the scene, change the image size, or change the action
space unless you also update the dataset contract and downstream training
code. Any camera in the scene becomes a dataset feature.

## 3. What your controller must output

Each step, your controller returns one action: a `torch.float32` tensor of
shape `(1, 8)` on `env.device`.

- `action[:, 0:7]`: **absolute** target angles for `panda_joint1..7`. These
  are target angles, not changes from the current angle.
- `action[:, 7]`: gripper command. Use `+1.0` to open and `-1.0` to close.

Most people plan motions as gripper poses, not joint angles. To convert a pose
into joint targets, use differential inverse kinematics (IK). You can copy
these two helpers:

```python
import isaaclab.utils.math as math_utils
import torch
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg

GRIPPER_OPEN, GRIPPER_CLOSE = 1.0, -1.0


def hold_action(env, gripper: float = GRIPPER_OPEN) -> torch.Tensor:
    """Action that keeps the arm exactly where it is."""
    robot = env.scene["robot"]
    arm_ids, _ = robot.find_joints("panda_joint.*")
    action = torch.zeros(env.num_envs, 8, device=env.device)
    action[:, :7] = robot.data.joint_pos[:, arm_ids]
    action[:, 7] = gripper
    return action


class HandPoseToJoints:
    """Differential IK: desired panda_hand pose (world frame) -> 7 arm joint targets."""

    def __init__(self, env):
        self.robot = env.scene["robot"]
        self.arm_ids, _ = self.robot.find_joints("panda_joint.*")
        self.hand_idx = self.robot.find_bodies("panda_hand")[0][0]
        self.ik = DifferentialIKController(
            DifferentialIKControllerCfg(command_type="pose", ik_method="dls", use_relative_mode=False),
            num_envs=env.num_envs,
            device=env.device,
        )

    def __call__(self, target_pos_w: torch.Tensor, target_quat_w: torch.Tensor) -> torch.Tensor:
        data = self.robot.data
        # express the current and the desired hand pose in the robot base frame
        hand_pos_b, hand_quat_b = math_utils.subtract_frame_transforms(
            data.root_pos_w, data.root_quat_w, data.body_pos_w[:, self.hand_idx], data.body_quat_w[:, self.hand_idx]
        )
        target_pos_b, target_quat_b = math_utils.subtract_frame_transforms(
            data.root_pos_w, data.root_quat_w, target_pos_w, target_quat_w
        )
        # fixed-base robot: PhysX Jacobians skip the root body, hence "- 1"
        jacobian_w = self.robot.root_physx_view.get_jacobians()[:, self.hand_idx - 1, :, self.arm_ids]
        base_rot = math_utils.matrix_from_quat(math_utils.quat_inv(data.root_quat_w))
        jacobian_b = torch.cat([base_rot @ jacobian_w[:, :3], base_rot @ jacobian_w[:, 3:]], dim=1)

        self.ik.set_command(torch.cat([target_pos_b, target_quat_b], dim=-1))
        return self.ik.compute(hand_pos_b, hand_quat_b, jacobian_b, data.joint_pos[:, self.arm_ids])
```

Tips:

- Quaternions in Isaac Lab are `(w, x, y, z)`.
- The `panda_hand` origin sits about 10 cm behind the point between the
  fingertips. `env.scene["ee_frame"].data.target_pos_w[:, 0]` gives you that
  fingertip point.
- Move the target only a few millimetres per step: about 4 mm per step, which
  is 0.24 m/s at 60 steps per second. If you jump the target far away, the arm
  jerks, and jerky demonstrations are hard for a policy to imitate.

## 4. Implement the collection pipeline

Implement a controller and recording loop for the selected task. Keep the
controller independent from dataset-writing code so it can be tested first
without producing a dataset. The controller must provide:

| Member | Meaning |
|---|---|
| `setup(env)` | One-time calibration and lookup of robot/body indices. |
| `reset()` | Clear per-episode phase and timer state. |
| `pre_step(env)` | Optional hook before action generation. |
| `get_action(env)` | Return one `(num_envs, 8)` action tensor. |
| `advance()` | Advance the phase/timer after `env.step(action)`. |
| `is_episode_done` | Become `True` when the attempt should be judged. |
| `check_success(env)` | Return the final task success decision. |

Recommended implementation order:

1. Read the selected task config and list every scene object needed by the
   controller. Read poses from `env.scene[name].data.root_pos_w` after each
   reset; do not copy the initial pose constants into the controller.
2. Implement one safe phase at a time: approach, grasp, transport, release,
   retreat, and verification. Bound every phase with a step counter.
3. Test with `--record` omitted first. Confirm that the arm reaches the target,
   the gripper command has the intended sign, and `check_success(env)` agrees
   with the task's `success` term.
4. Connect the controller to the recording loop and run one or two recorded
   episodes with a new dataset id.
5. Inspect the saved dataset before collecting the full set. Only then collect
   the requested number of successful episodes and run the coverage check.

### 4.1 Scripted controller (state machine)

A skeleton. Fill in your own phases.

```python
import torch


class MyStateMachine:
    def setup(self, env):
        # Resolve robot/body indices and perform one-time calibration here.
        self.robot = env.scene["robot"]

    def reset(self):
        self.phase = 0
        self.phase_step = 0
        self._episode_done = False

    def pre_step(self, env):
        # Read randomized object poses here or in get_action().
        pass

    def get_action(self, env):
        # The action is seven absolute Panda joint targets plus one gripper
        # command: +1 opens and -1 closes.
        action = torch.zeros(env.num_envs, 8, device=env.device, dtype=torch.float32)
        arm_ids, _ = self.robot.find_joints("panda_joint.*")
        action[:, :7] = self.robot.data.joint_pos[:, arm_ids]
        action[:, 7] = 1.0
        return action

    def advance(self):
        self.phase_step += 1
        # Replace this with bounded approach/grasp/lift/place/release phases.
        if self.phase_step >= 10:
            self._episode_done = True

    def check_success(self, env):
        # Implement the task-specific final-state predicate here.
        return False

    @property
    def is_episode_done(self):
        return self._episode_done
```

Good habits for scripted demonstrations:

- Give each phase a step limit. If a phase takes too long, set
  `_episode_done = True` and let `check_success(env)` return `False` instead
  of letting it run indefinitely.
- Set `is_episode_done` shortly after the task is done (object released, small
  retreat). Motion after that point teaches the policy to keep moving once the
  job is finished.
- Add some variation, for example small random offsets in approach height or
  timing. A thousand identical trajectories teach less than a hundred varied
  ones.

### 4.2 Keyboard teleoperation

For keyboard teleoperation, implement an input adapter that converts keyboard
events into the same `(num_envs, 8)` action tensor used by the scripted
controller. Document the key bindings in your own collection entry point.
Keep input handling, episode status, and dataset writing as separate parts so
the same recorder can be used with a scripted controller.

Keyboard teleoperation needs the GUI (don't pass `--headless`) and a real-time
rate limiter. Without a rate limiter, the simulation runs as fast as it can and
the arm moves too fast to control.

Do not step the simulation while waiting for the operator to start. Begin
recording after the scene is stable, and discard practice or calibration
motion. If object-pose coverage is required, add the `object_pose.*` features
to the same frame before writing it.

## 5. Run it

```bash
# scripted controller: headless is fine and faster
python <your_collection_script>.py \
    --task HCIS-CupStacking-SingleArm-v0 \
    --num_envs 1 --device cuda --enable_cameras \
    --record --dataset_repo_id <hf_user>/cup-stacking-demos \
    --dataset_fps 30 --num_episodes 50

# keyboard controller: needs the GUI and real-time speed
python <your_collection_script>.py \
    --task HCIS-CupStacking-SingleArm-v0 \
    --input keyboard --num_envs 1 --device cuda \
    --enable_cameras --record \
    --dataset_repo_id <hf_user>/cup-stacking-demos \
    --dataset_fps 30 --num_episodes 20
```

Run inside the Isaac Lab environment. The option names above are illustrative;
make your own script expose equivalent options. Replace the task id with one
of the three registered tasks. The collector should report success/failure and
keep only successful episodes in the training dataset.

## 6. Choosing `fps`

The simulator advances at 60 Hz (`sim.dt = 1/60`, `decimation = 1`). Use a
dataset FPS of 30 to match the camera update period and the current collection
convention. Keep this value unchanged unless you also
change the recorder/evaluation contract. It is dataset metadata, not a command
to change the physics step rate.

Two things to know:

- Both cameras are 640×480 RGB and refresh every 1/30 s (`update_period`).
- Don't change `sim.dt` or `decimation` to get a different rate. Evaluation
  runs your task with its default settings and applies one policy action per
  `env.step()`, so your data must be recorded at that same step rate.

## 7. Recording over several sessions

To add episodes to a dataset finalized earlier, configure the writer in
resume/append mode instead of create mode. Use the same dataset id, feature
definitions, image size, and FPS as the first session. Only resume a finalized
dataset; copy the folder first so a failed session cannot damage existing
episodes. If the writer does not support resume, create a new dataset and keep
the sessions separate.

## 8. Advanced option: write the dataset yourself

Instead of the recorder manager, you can call LeRobot directly. You write more
code, but you control everything: which frames are kept, when, and why. There
is no hidden 5-frame skip and no success term to manage.

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from simulator import FRANKA_JOINT_NAMES
from simulator.utils.object_pose_recording import object_pose_features, read_object_poses

IMAGE = {"dtype": "video", "shape": (480, 640, 3), "names": ["height", "width", "channels"]}
OBJECTS = ["blue_cup", "pink_cup"]  # use the names from the selected task config
FEATURES = {  # base features produced by the repository recorder
    "observation.state": {"dtype": "float32", "shape": (9,), "names": [f"{j}.pos" for j in FRANKA_JOINT_NAMES]},
    "action": {"dtype": "float32", "shape": (8,), "names": [f"dim_{i}" for i in range(8)]},
    "observation.images.front": IMAGE,
    "observation.images.wrist": IMAGE,
}
FEATURES.update(object_pose_features(OBJECTS))

env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
env_cfg.use_teleop_device("keyboard")
task_success = env_cfg.terminations.success
env_cfg.terminations.success = None   # no automatic resets: you decide when an attempt ends
env_cfg.terminations.time_out = None
env_cfg.recorders = None              # no Isaac Lab recorder at all
env = gym.make(args.task, cfg=env_cfg).unwrapped

dataset = LeRobotDataset.create(
    repo_id=args.repo_id, fps=30, robot_type="franka_panda", features=FEATURES
)

try:
    with torch.inference_mode():
        obs, _ = env.reset()
        controller.reset(env)
        step = 0
        while dataset.num_episodes < args.num_episodes:
            action = controller.act(env)
            if action is None:  # e.g. teleop operator has not started yet
                env.sim.render()
                continue
            step += 1
            if step > 5:  # skip frames that may still show the pre-reset scene
                policy_obs = obs["policy"]  # observation from BEFORE this action
                dataset.add_frame({
                    "observation.state": policy_obs["joint_pos"][0].cpu().numpy(),
                    "observation.images.front": policy_obs["front"][0].cpu().numpy(),
                    "observation.images.wrist": policy_obs["wrist"][0].cpu().numpy(),
                    "action": action[0].cpu().numpy(),
                    "task": env_cfg.task_description,
                    **read_object_poses(env, OBJECTS),
                })
            obs, *_ = env.step(action)

            if controller.finished or controller.gave_up or step >= args.max_steps:
                if controller.finished and bool(task_success.func(env, **task_success.params)[0]):
                    dataset.save_episode()
                else:
                    dataset.clear_episode_buffer()
                obs, _ = env.reset()
                controller.reset(env)
                step = 0
finally:
    dataset.finalize()
    env.close()
    simulation_app.close()
```

The key rule: add the frame with the observation from **before** the action,
then step. If you pair an observation with the action from a different step,
the policy learns the wrong cause and effect.

## 9. Check your dataset

1. **Load it and look at the numbers:**

   ```python
   from lerobot.datasets.lerobot_dataset import LeRobotDataset

   ds = LeRobotDataset("<hf_user>/cup-stacking-demos")
   print(ds.num_episodes, ds.num_frames, ds.fps)
   print(list(ds.features))
   ```

2. **Watch a few episodes** with the
   [LeRobot Dataset Visualizer](lerobot_dataset_visualizer.md). Check that the
   first frame shows the freshly reset scene, the last frame shows the task
   done, both cameras show the right view, and the state curves move when the
   arm moves.

3. **Check object-pose coverage** if your dataset was recorded with the
   repository recorder:

   ```bash
   uv run python scripts/verify_object_pose_coverage.py init \
       --dataset ~/.cache/huggingface/lerobot/<hf_user>/cup-stacking-demos \
       --out specs/cup_stacking_coverage.ttl
   # Fill the TODO values in the generated Turtle file, then:
   uv run python scripts/verify_object_pose_coverage.py check \
       --dataset ~/.cache/huggingface/lerobot/<hf_user>/cup-stacking-demos \
       --spec specs/cup_stacking_coverage.ttl
   ```

   See [Object Pose Coverage](object_pose_coverage.md) for the meaning of the
   report. This check is a coverage diagnostic; it does not replace visual
   inspection or LeRobot loading.

## 10. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `FileExistsError` right at start | The dataset folder already exists | Pick a new `repo_id`, delete the old folder, or [resume](#7-recording-over-several-sessions) |
| Recorder rejects the export mode | The writer was constructed before its successful-only/resume mode was configured | Set the writer mode before creating the recorder |
| Dataset won't load / parquet footer error | `finalize()` never ran (crash, `kill -9`, closed terminal) | Keep `finalize()` in `finally`; stop with Ctrl+C |
| Every attempt is reported as failed | `check_success(env)` is `False` or the state machine ends too early | Print the object poses and compare the predicate with the task config in section 1 |
| Error or empty episode right after a saved one | The next episode was not reset or the success state was not cleared | After every `env.reset()`, set the success term back to `False` (`set_success(env, False)` in [section 2.1](#21-use-the-repositorys-lerobotrecordermanager)) |
| First frames show the previous episode's scene | Controller moved during the skipped steps | Hold still for at least 5 steps after reset |
| Camera errors / images missing | Started without `--enable_cameras` | Add the flag |
| Missing `object_pose.*` features | leisaac's `LeRobotRecorderManager` was imported instead of the repository's, or `tracked_object_names` is empty | Import from `simulator.utils.object_pose_recording` ([section 2.1](#21-use-the-repositorys-lerobotrecordermanager)) and check the task config |
| `KeyError: '<object name>'` on the first recorded frame | `env_cfg.recorders` was replaced, so the `states` buffer is missing | Keep the default `ActionStateRecorderManagerCfg` and change only its export mode |
| Dataset cannot be opened after an interrupted run | The writer was not finalized | Stop with Ctrl+C when possible, resume only a finalized dataset, and use a new repo id for a corrupted one |
| Keyboard does nothing | Running `--headless`, or your input adapter is still waiting for its start key | Run with the GUI and press the start key your script defines |

## 11. Checklist

- [ ] `num_envs=1`, `--enable_cameras`.
- [ ] Controller reads object poses fresh every episode.
- [ ] Controller holds still for at least 5 steps after each reset.
- [ ] Each episode is a single attempt that ends shortly after the task is done.
- [ ] Your definition of "done" is stricter than, or equal to, the built-in check, and your script enforces it.
- [ ] Failed, stuck, or interrupted attempts are dropped, never saved.
- [ ] Dataset FPS is kept consistent across all sessions (recommended: 30).
- [ ] `finalize()` runs at the end of every session.
- [ ] The dataset loads in LeRobot and the object-pose coverage check passes when coverage is required.
