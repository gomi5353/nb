# How to Record Episodes

This guide describes how students build the data collection pipeline for their
task: a Python script that drives the simulated robot through demonstrations
and saves them as a LeRobot v3 dataset. That dataset is later used to train
the policy ([LeRobot Training Procedure](lerobot_training.md)) and is checked
with the [object-position coverage](object_position_coverage.md) tool.

It extends steps 1 and 2 of [Getting Started](getting_started.md) and assumes
that step 0 (setup) is complete, so that a shell inside the Isaac Lab container
is available.

> **Run on:** a Linux machine with an NVIDIA GPU, inside the Isaac Lab
> container (see [Getting Started](getting_started.md)). The host `uv`
> environment cannot run Isaac Sim, and recording always needs the camera
> images, so every recording command needs `--enable_cameras`.

Every section starts with one of two labels:

- **Required**: the section describes something that students must implement
  or follow. The [checklist](#11-checklist) at the end collects all of these
  requirements.
- **Background**: the section explains how a component works. It contains no
  work items, but the required sections depend on it.

## 1. Key terms

**Background**

General terms (host, container, environment, episode, controller, collection
entry point, recorder, dataset) are defined in
[Getting Started §2](getting_started.md#2-key-terms). This guide adds:

| Term | Meaning in this guide |
|---|---|
| Task configuration | The Python class that defines an Isaac Lab environment: scene, cameras, actions, randomization, and success check ([section 3](#3-the-task-configuration)). |
| State machine | One way to write a *scripted* controller: the task is split into phases (approach, grasp, lift, ...) and the controller switches from one phase to the next when a condition is met ([section 4.4](#44-scripted-controller-state-machine)). |
| Recording loop | The loop in the collection script that repeatedly asks the controller for an action, calls `env.step()`, and resets the environment when an attempt is over ([section 5.3](#53-the-collection-entry-point)). |
| LeRobot v3 dataset format | The file layout that LeRobot reads for training: per-frame numbers (joint states, actions, ...) in Parquet tables under `data/`, camera images as MP4 videos under `videos/`, and a description of all columns and episodes under `meta/` (see the [LeRobot v3 format](https://huggingface.co/docs/lerobot/main/en/lerobot-dataset-v3#format-design)). |
| Frame | One row of the dataset: what the robot observed and the action it was given at one `env.step()`. |

The word *controller* in this guide never means the robot's low-level joint
controller (the PD actuators in Isaac Lab, with stiffness 80 and damping 4,
which stay fixed), nor an inverse kinematics (IK) solver that a controller may
call ([section 4.3](#43-inverse-kinematics-helpers)).

## 2. What students implement

**Required**

For the task selected by the group, students write **one collection entry
point**: a Python script in their copy of the repository, for example
`scripts/collect_cup_stacking.py`. The name and location are free. The script
contains, or imports, two parts:

1. **A data-collection controller** for the task
   ([section 4](#4-the-data-collection-controller)). It may be a scripted
   state machine, keyboard teleoperation, or another input source, as long as
   it provides the interface in [section 4.1](#41-controller-interface).
2. **A recording loop** that connects the controller to the Isaac Lab
   environment and to the repository's `LeRobotRecorderManager`
   ([section 5.3](#53-the-collection-entry-point)). A complete template of
   this part is given in this guide.

The result is a LeRobot v3 dataset of successful demonstrations that contains
the `object_position.*` columns and has passed the checks in
[section 9](#9-check-the-dataset). How many episodes to record, and which
object positions they must cover, is the group's decision.

What the repository provides, and what students do with it:

| Component | File | Students |
|---|---|---|
| Task configurations | `packages/simulator/src/simulator/tasks/<task>/<task>_env_cfg.py` | Read; may modify, aligned with rollout ([section 2.1](#21-modifying-the-environment-configuration)) |
| Shared robot, camera, and action settings | `packages/simulator/src/simulator/tasks/template/single_arm_franka_cfg.py` | Read; may modify, aligned with rollout ([section 2.1](#21-modifying-the-environment-configuration)) |
| LeRobot recorder | `packages/simulator/src/simulator/utils/lerobot_recorder.py` | Import and use; extend to record further verification factors ([Adding a Coverage Factor §4](adding_coverage_factor.md#4-record-the-factor-in-lerobot-v3)) |
| Evaluation files and rollout | `eval/`, `scripts/rollout.py` | Use later for evaluation |
| Coverage check | `scripts/verify_object_position_coverage.py` | Run on every dataset |
| Controller | none | **Implement** |
| Collection entry point with the recording loop | none (template in [section 5.3](#53-the-collection-entry-point)) | **Implement** |

The collection script itself changes a few settings at runtime, on its own
copy of the configuration ([section 5.3](#53-the-collection-entry-point):
time-out, success term, export mode). Groups may also change the configuration
files; [section 2.1](#21-modifying-the-environment-configuration) lists what
must stay aligned with rollout.

A recommended order of work:

1. Read the task configuration ([section 3](#3-the-task-configuration)).
2. Implement the controller one phase at a time and test it without recording
   ([section 4](#4-the-data-collection-controller)).
3. Write the collection entry point and record one or two test episodes into a
   new dataset ([section 5](#5-the-recorder-and-the-recording-loop) and
   [section 6](#6-run-the-collection-script)).
4. Check the test dataset, including the `object_position.*` columns and the
   coverage tool's `init` step ([section 9](#9-check-the-dataset)).
5. Record the full dataset and run all checks again.

### 2.1 Modifying the environment configuration

Every environment configuration may be modified for data collection: the task
configuration files, the shared template, or a new configuration class of the
group's own. Examples are wider or narrower object-position randomization, other
lighting, extra or different object start positions, a stricter success check, or
a longer episode for recording. Such changes are part of the group's
data-collection design and are reported with it.

Rollout does not use the recording configuration. It builds the environment
from the evaluation file in `eval/`, and the final ranking uses a private
evaluation configuration built on the repository's original template. A
modification therefore never reaches the graded evaluation; it only changes
the data. Every modification must stay aligned with rollout in two ways.

**The rollout contract must not change.** These settings define what the
policy receives and sends during rollout
([Rollout §4](rollout.md#4-policy-input-and-output-contract)).
A dataset recorded with different values trains a policy that cannot run in
evaluation, or runs with a different meaning:

| Setting | Required value | Defined in |
|---|---|---|
| Robot | Franka Panda, same joints and gripper | template |
| Action space | `use_joint_position_actions()`: 7 absolute joint targets + 1 gripper command, stiffness 80, damping 4 | template |
| Control rate | `sim.dt = 1/60`, `decimation = 1`, one action per `env.step()` | template |
| Joint state | `observation.state`: the 9 joint positions in the contract order | template |
| Cameras | exactly `front` and `wrist`, RGB, 640×480 | template (`front` pose per task) |
| Frame builder | `build_lerobot_frame`, unchanged keys and shapes | template |

**Differences from the evaluation environment are deliberate.** Everything
else may differ between recording and evaluation, but each difference is a
distribution shift that the policy must bridge. The relevant values are
compared side by side: the task configuration against the evaluation file
`eval/<task>_eval.py` (object start positions and randomization ranges, lighting
range, robot start pose, `front` camera pose, success check, episode length).
Recording a wider range than the evaluation file is usually helpful, because
the private evaluation uses other positions. Recording a narrower range, a moved
camera, or a different robot start pose usually hurts, and the coverage check
([Object Position Coverage](object_position_coverage.md)) only measures what the
specification asks for.


## 3. The task configuration

**Background**

Each task is defined by a configuration class in its own file. The
configuration is the description from which Isaac Lab builds the environment;
students should read it to learn what their controller has to work with.

| | Cup stacking | Cutlery arrangement | Toy blocks collection |
|---|---|---|---|
| Task id | `HCIS-CupStacking-SingleArm-v0` | `HCIS-CutleryArrangement-SingleArm-v0` | `HCIS-ToyBlocksCollection-SingleArm-v0` |
| Configuration file | [`cup_stacking_env_cfg.py`](../packages/simulator/src/simulator/tasks/cup_stacking/cup_stacking_env_cfg.py) | [`cutlery_arrangement_env_cfg.py`](../packages/simulator/src/simulator/tasks/cutlery_arrangement/cutlery_arrangement_env_cfg.py) | [`toy_blocks_collection_env_cfg.py`](../packages/simulator/src/simulator/tasks/toy_blocks_collection/toy_blocks_collection_env_cfg.py) |
| Scene objects (`env.scene[name]`) | `blue_cup`, `pink_cup` | `plate`, `fork`, `knife` | `green_block`, `blue_block`, `red_block`, `storage_box` |
| Success check (`success` term) | Blue cup within 2 cm of the pink cup in x and in y, and 8–10 cm higher | Fork and knife each inside a 10 cm × 10 cm square beside the plate, the square centred 12 cm from the plate centre: fork at smaller x (the robot's left, the right side of the front image), knife at larger x | Every block inside the storage box footprint (0.296 m × 0.209 m) shrunk by 1 cm on each side, measured in the box's own frame (the region turns with the box), and 0–11 cm above the box root (the box bottom) |
| Task text (`task_description`) | `pick up the blue cup and place it on the pink cup.` | `place the fork on the left and knife on the right of the plate.` | `pick up the toys and place them into the storage box.` |
| Episode time-out | 20 s | 40 s | 60 s |

The settings shared by all three tasks (robot, cameras, action space, timing)
are in
[`single_arm_franka_cfg.py`](../packages/simulator/src/simulator/tasks/template/single_arm_franka_cfg.py).
The concepts in the configuration:

- **Scene objects** are the rigid bodies that the robot manipulates. A script
  reads an object's current position with
  `env.scene[name].data.root_pos_w` (world frame, metres) and its orientation
  with `env.scene[name].data.root_quat_w`.
- **Cameras** are the two simulated RGB cameras whose images the policy will
  see: `front`, a fixed camera facing the table, and `wrist`, mounted on the
  robot hand. Both produce 640×480 images. Every camera in the scene becomes a
  dataset column, so adding, removing, or renaming a camera breaks the rollout
  contract ([section 2.1](#21-modifying-the-environment-configuration)).
- **Action space** is the format of the action passed to `env.step()`: eight
  numbers per step, seven absolute joint angles for the arm and one gripper
  command ([section 4.2](#42-the-action)). It is set by
  `env_cfg.use_joint_position_actions()`, which the collection script and
  `rollout.py` both call so that recording and evaluation control the robot in
  the same way.
- **Success check** (success predicate) is the `success` termination term: a
  function that looks at the object positions and returns `True` when the task
  is solved. The recorder uses it to decide which episodes are kept
  ([section 5.1](#51-how-the-recorder-works)).
- **Task text** is a short sentence that describes the task. The recorder
  saves it with every episode ([section 5.4](#54-what-each-frame-contains)).

Facts about the tasks that a controller has to handle:

- One `env.step()` is one simulation step of 1/60 s (`decimation = 1`).
- Every reset moves the objects by a few centimetres and changes the light
  intensity. A controller must read the object positions at the start of every
  episode and must never hard-code them.
- The time-out equals the length of an evaluation episode. Evaluation stops a
  policy at that time, so demonstrations should finish well within it. The
  collection script disables the time-out and lets the controller end each
  attempt ([section 5.3](#53-the-collection-entry-point)); therefore every
  controller needs its own step limit per phase and per episode.
- The success check looks only at object positions, at a single moment. It
  does not know whether the gripper still holds an object. In cup stacking,
  for example, the check can already pass while the blue cup is still in the
  gripper, held at required height above the pink cup. Each group decides what "done" means
  for its demonstrations (for example: object released, arm moved clear, check
  still passes), and the controller's `check_success(env)` enforces it.

## 4. The data-collection controller

The controller is the part that encodes *how* the task is solved. A
**state machine** is not a separate component next to the controller: it is
one common way to implement a scripted controller. The relation is:

| Controller type | How it decides the action | Uses a state machine? |
|---|---|---|
| Scripted | A program computes the action from the object positions | Usually: one phase per sub-task |
| Keyboard teleoperation | A person presses keys; an input adapter turns them into actions | No |
| Motion planner | A planner computes a collision-free trajectory | Possibly, to sequence the planner calls |

Students implement **one** controller of any type. For a scripted controller,
the state machine *is* the controller. Section 4.4 gives a state-machine
skeleton and section 4.5 describes keyboard teleoperation.

### 4.1 Controller interface

**Required**

The recording loop in [section 5.3](#53-the-collection-entry-point) calls the
controller through the following members. Every controller provides them, so
that the same loop works for any controller type.

| Member | Meaning |
|---|---|
| `setup(env)` | One-time setup after the environment is created, such as looking up robot joint and body indices. |
| `reset()` | Clear the per-episode state (phase, timers) at the start of every episode. |
| `pre_step(env)` | Optional hook before the action is computed, for example to read sensors. |
| `get_action(env)` | Return one action tensor of shape `(num_envs, 8)` ([section 4.2](#42-the-action)). |
| `advance()` | Advance the phase and timers after `env.step(action)`. |
| `is_episode_done` | Property that becomes `True` when the attempt is over and should be judged. |
| `check_success(env)` | Return the group's final decision whether the attempt succeeded. |

The controller contains no dataset code. This keeps the controller testable
without recording: a test run steps the environment and prints the result, and
nothing is written to disk.

### 4.2 The action

**Required**

At each step the controller returns one action: a `torch.float32` tensor of
shape `(1, 8)` on `env.device`.

- `action[:, 0:7]`: **absolute** target angles (radians) for `panda_joint1..7`.
  These are target angles, not changes from the current angle.
- `action[:, 7]`: gripper command. `+1.0` opens and `-1.0` closes.

### 4.3 Inverse kinematics helpers

**Background**

Motions are easier to plan as gripper poses than as joint angles. Differential
inverse kinematics (IK) converts a desired hand pose into joint targets. The
following two helpers may be copied into a controller:

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

`HandPoseToJoints` is an IK solver that a controller calls internally. Isaac
Lab names its solver class `DifferentialIKController`, but it is not the
data-collection controller of this guide.

Practical notes:

- Quaternions in Isaac Lab are ordered `(w, x, y, z)`.
- The `panda_hand` origin sits about 10 cm behind the point between the
  fingertips. `env.scene["ee_frame"].data.target_pos_w[:, 0]` gives that
  fingertip point.
- The target should move only a few millimetres per step: about 4 mm per step
  is 0.24 m/s at 60 steps per second. A target that jumps far away makes the
  arm jerk, and jerky demonstrations are hard for a policy to imitate.

### 4.4 Scripted controller (state machine)

**Required** if the group uses a scripted controller.

A skeleton; the phases are task-specific and left to the group:

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
        # Read randomized object positions here or in get_action().
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

A suggested development order:

1. List every scene object that the controller needs. Read the positions from
   `env.scene[name].data.root_pos_w` after each reset; never copy the initial
   position constants from the configuration file.
2. Implement one phase at a time: approach, grasp, transport, release,
   retreat, and verification. Bound every phase with a step counter.
3. Test without recording (`--record` omitted,
   [section 6](#6-run-the-collection-script)). Confirm that the arm reaches
   the target, that the gripper command has the intended sign, and that
   `check_success(env)` agrees with the task's `success` term.

Good practice for scripted demonstrations:

- Give each phase a step limit. A phase that takes too long sets
  `_episode_done = True`, and `check_success(env)` returns `False`, instead
  of running indefinitely.
- Set `is_episode_done` shortly after the task is done (object released, small
  retreat). Motion after that point teaches the policy to keep moving once the
  job is finished.
- Add variation, for example small random offsets in approach height or
  timing. A thousand identical trajectories teach less than a hundred varied
  ones.

### 4.5 Keyboard teleoperation

**Required** if the group uses keyboard teleoperation.

A keyboard controller consists of an input adapter that converts key events
into the same `(num_envs, 8)` action tensor, plus the members of
[section 4.1](#41-controller-interface). The key bindings are documented in
the group's collection script. Input handling and episode status stay separate
from the recorder, so that the recording loop in
[section 5.3](#53-the-collection-entry-point) works unchanged.

Keyboard teleoperation needs the GUI (no `--headless`) and a real-time rate
limiter. Without a rate limiter, the simulation runs as fast as it can and the
arm moves too fast to control. The simulation is not stepped while the script
waits for the operator to start; recording begins after the scene is stable,
and practice or calibration motion is discarded.

## 5. The recorder and the recording loop

### 5.1 How the recorder works

**Background**

The recorder is the component that writes the LeRobot v3 dataset. Students do
not write a dataset writer of their own: they attach the repository's
`LeRobotRecorderManager` to the environment, and it captures frames and saves
episodes by itself. It runs inside `env.step()` and `env.reset()`; the script
calls it directly only once, `finalize()` at the very end.

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
    ├─ True  → episode is written to disk
    └─ False → episode is thrown away
    │
recorder_manager.finalize()  files are closed and the dataset can be loaded
```

Consequences that the recording loop has to respect (source:
`leisaac/enhance/managers/lerobot_recorder_manager.py`):

1. **One frame per `env.step()`.** Without a step, nothing is recorded.
2. **The first 5 steps after every reset are skipped.** Camera images from
   before the reset can still appear there, so the controller holds still for
   those steps.
3. **Keep or drop is decided at the next reset**, by the `success` termination
   term. Any call to `env.reset()` ends the current episode.
4. **Only failed episodes are dropped, and they are dropped always.** The
   recorder accepts only the export mode `EXPORT_SUCCEEDED_ONLY` (or its
   resume variant); failed episodes never reach the disk.
5. **Only environment 0 is recorded.** Recording always uses `num_envs=1`.
6. **`finalize()` is mandatory.** Without it the Parquet files have no footer
   and the dataset cannot be loaded.
7. **The dataset goes to `$HF_LEROBOT_HOME/<repo_id>`**, by default
   `~/.cache/huggingface/lerobot/<repo_id>`. In the container this folder is
   mounted from `datasets/<repo_id>` on the host. If the folder already
   exists, creating the dataset fails with `FileExistsError`.

### 5.2 The repository's recorder and object positions

**Required**

The recorder is imported from this repository, **not** from leisaac:

```python
from simulator.utils.lerobot_recorder import LeRobotRecorderManager       # use this
# from leisaac.enhance.managers.lerobot_recorder_manager import LeRobotRecorderManager  # not this
```

[`lerobot_recorder.py`](../packages/simulator/src/simulator/utils/lerobot_recorder.py)
defines a subclass of leisaac's `LeRobotRecorderManager`. It is the
repository's base extension for data verification: it adds one verification
factor, the object position. A group that verifies further factors appends them to
this module ([Adding a Coverage Factor §4](adding_coverage_factor.md#4-record-the-factor-in-lerobot-v3)),
so that one recorder writes all verification columns. The two classes compare
as follows:

| | leisaac's `LeRobotRecorderManager` | Repository's `LeRobotRecorderManager` |
|---|---|---|
| Frame capture, skipped steps, keep/drop rule, `finalize()` | As in [section 5.1](#51-how-the-recorder-works) | Identical (inherited) |
| Robot state, action, camera images, task text | Recorded | Recorded (identical) |
| Position of every task object (`object_position.<name>`) | Not recorded | Recorded |
| Usable for the coverage check | No | Yes |

**Why object positions are recorded.** A policy can only imitate situations that
appear in its demonstrations. Every reset places the objects at a random
position, and evaluation does the same. If a region of the table has few
demonstrations, the policy has little to imitate there and is likely to fail.
The [object-position coverage check](object_position_coverage.md) finds such regions:
it reads where each object started in every episode, divides the table area
into cells, and reports cells with too few episodes. Robot joint states and
camera images do not state the object positions directly, so the positions are
recorded as separate columns.

Only the position is recorded, not the orientation, because position is the primary variable used to measure spatial coverage in the current experimental setup. Object orientations remain fixed across episodes, as randomizing them could cause objects to start in unstable configurations (e.g., tipped over) or fall off the table, requiring environment resets and additional simulation attempts. Since the orientation does not vary across episodes, it provides no additional coverage information ([Object Position Coverage §1](object_position_coverage.md#1-what-the-check-does-and-why)).

**How they are recorded.** The recorder:

- declares one `object_position.<name>` column, float32 `(3,)` =
  `[x, y, z]`, for every name in the task configuration's
  `tracked_object_names`, the objects whose start position is randomized
  (already set for the three tasks; the fixed `plate` and `storage_box` are in
  `fixed_object_names` and get no column);
- fills these columns in every frame with the object position relative to the
  environment origin;
- raises a `ValueError` when it is created for a task whose
  `tracked_object_names` is empty. This matters only for an Advanced Level
  task that the group defines itself.

The coverage check uses the **first recorded frame** of each episode, which is
where every object started. The columns do not affect training or rollout:
LeRobot policies use only the `observation.*` and `action*` columns as inputs
and outputs.

A dataset recorded with leisaac's original class has no object positions and
cannot be verified, so it must be recorded again. Writing frames with a custom
LeRobot writer is not supported for the same reason.

The object positions are read from the `states` buffer that Isaac Lab's
`ActionStateRecorderManagerCfg` fills. That configuration is the task default
(`env_cfg.recorders`); the collection script keeps it and changes only its
export mode.

### 5.3 The collection entry point

**Required**

The template below is a complete collection entry point except for the
controller. It follows the same pattern as leisaac's own state-machine data
generation script
([`scripts/datagen/state_machine/generate.py#L153`](https://github.com/LightwheelAI/leisaac/blob/24d3bcd3f1e4585740fc79921782c41617237812/scripts/datagen/state_machine/generate.py#L153)),
reduced to what an HCIS task needs. `MyStateMachine` is the controller from
[section 4.4](#44-scripted-controller-state-machine); any controller with the
interface of [section 4.1](#41-controller-interface) can be used instead.

```python
"""Collection entry point: record demonstrations into a LeRobot v3 dataset."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Record demonstrations into a LeRobot v3 dataset.")
parser.add_argument("--task", required=True, help="Task id, e.g. HCIS-CupStacking-SingleArm-v0.")
parser.add_argument("--record", action="store_true", help="Write a dataset; without it nothing is saved.")
parser.add_argument("--dataset_repo_id", default=None, help="<hf_user>/<dataset name>.")
parser.add_argument("--num_episodes", type=int, default=1, help="Successful episodes to record.")
parser.add_argument("--resume", action="store_true", help="Append to a finalized dataset.")
AppLauncher.add_app_launcher_args(parser)  # adds --device, --headless, --enable_cameras, ...
args = parser.parse_args()

# Isaac Sim must be running before Isaac Lab, leisaac, or the tasks are imported.
simulation_app = AppLauncher(vars(args)).app

import gymnasium as gym
import torch
from isaaclab.managers import DatasetExportMode, TerminationTermCfg
from isaaclab_tasks.utils import parse_env_cfg
from leisaac.enhance.datasets.lerobot_dataset_handler import LeRobotDatasetCfg
from leisaac.enhance.managers import EnhanceDatasetExportMode

import simulator.tasks  # noqa: F401  registers the HCIS-* tasks
from simulator.utils.lerobot_recorder import LeRobotRecorderManager

from my_controller import MyStateMachine  # the group's controller (section 4)


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
env_cfg.use_joint_position_actions()  # same robot and action setup as rollout; required
env_cfg.terminations.time_out = None  # the controller decides when an attempt ends
env_cfg.terminations.success = TerminationTermCfg(  # starts False; set_success() flips it
    func=lambda env: torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
)
if not args.record:
    env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_NONE  # test run: write nothing
elif args.resume:
    env_cfg.recorders.dataset_export_mode = EnhanceDatasetExportMode.EXPORT_SUCCEEDED_ONLY_RESUME
else:
    env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_ONLY
env = gym.make(args.task, cfg=env_cfg).unwrapped

# 2. Replace Isaac Lab's default (HDF5) recorder with the LeRobot one.
if args.record:
    del env.recorder_manager
    env.recorder_manager = LeRobotRecorderManager(
        env_cfg.recorders,
        LeRobotDatasetCfg(repo_id=args.dataset_repo_id, fps=60),  # see section 7
        env,
    )

# 3. Recording loop: step, then judge the attempt and reset.
controller = MyStateMachine()
controller.setup(env)
successes = attempts = 0
try:
    with torch.inference_mode():
        env.reset()
        controller.reset()
        while successes < args.num_episodes and simulation_app.is_running():
            controller.pre_step(env)
            env.step(controller.get_action(env))  # the recorder adds one frame here
            controller.advance()
            if controller.is_episode_done:
                success = bool(controller.check_success(env))
                attempts += 1
                successes += success
                print(f"attempt {attempts}: {'success' if success else 'failure'} ({successes}/{args.num_episodes})")
                set_success(env, success)
                env.reset()  # saved if success is True, discarded otherwise
                set_success(env, False)
                controller.reset()
finally:
    if args.record:
        env.recorder_manager.finalize()  # mandatory, see section 5.1
    env.close()
    simulation_app.close()
```

Notes on the loop:

- `set_success(env, True)` is followed directly by `env.reset()`. A call to
  `env.step()` while the term is `True` makes the environment terminate and
  reset on its own.
- The episode is written in `env.reset()`, not in `finalize()`. The loop
  resets after the last attempt too; otherwise that episode is lost.
- `successes` counts only the episodes saved in the current session. When
  resuming, the number already in the dataset is added by hand.
- The script imports its controller from a module next to it
  (`my_controller.py` above). Keeping the controller in its own file also lets
  it be tested on its own.

### 5.4 What each frame contains

**Background**

The recorder builds the dataset columns from the environment, using
`build_feature_from_env` and `SingleArmFrankaTaskEnvCfg.build_lerobot_frame` in
[`single_arm_franka_cfg.py`](../packages/simulator/src/simulator/tasks/template/single_arm_franka_cfg.py).

| Feature | Shape / type | Content |
|---|---|---|
| `observation.state` | `(9,)` float32 | Measured joint positions: `panda_joint1..7` (rad), `panda_finger_joint1..2` (m) |
| `action` | `(8,)` float32, names `dim_0..dim_7` | `dim_0..dim_6`: target positions for `panda_joint1..7` (rad, absolute). `dim_7`: gripper command (≥ 0 open, < 0 close) |
| `observation.images.front` | 480×640×3 video | Front camera |
| `observation.images.wrist` | 480×640×3 video | Wrist camera |
| `task_index` | int64 | Index of the task text, see below |
| `object_position.<name>` | `(3,)` float32, names `x, y, z` | Position of each `tracked_object_names` object, relative to the env origin ([section 5.2](#52-the-repositorys-recorder-and-object-positions)) |

**Where the task text is stored.** Every frame is handed to LeRobot with the
task text from the configuration's `task_description`. LeRobot does not repeat
the sentence in every row. It stores each distinct sentence once in
`meta/tasks.parquet`, together with its number, and writes only that number in
the `task_index` column of `data/`. The episode table
`meta/episodes/.../*.parquet` also lists the task text of each episode in its
`tasks` column. All episodes of one task share the same sentence, so
`task_index` is `0` everywhere. When the dataset is loaded, LeRobot looks the
number up again and returns the sentence as `task`. Language-conditioned
policies (for example pi0 or SmolVLA) use it as their instruction; Diffusion
Policy and ACT ignore it.

The scene, the image size, and the action space are not changed: every camera
in the scene becomes a dataset feature, and evaluation feeds the policy
exactly the features listed above.

## 6. Run the collection script

**Required**

The script is run from the repository root, inside the container. The
repository folder is mounted into the container, so a script written on the
host is available there immediately.

```bash
# test run: controller only, nothing is saved
python scripts/collect_cup_stacking.py \
    --task HCIS-CupStacking-SingleArm-v0 \
    --device cuda --enable_cameras --num_episodes 3

# scripted controller: headless is faster
python scripts/collect_cup_stacking.py \
    --task HCIS-CupStacking-SingleArm-v0 \
    --device cuda --enable_cameras --headless \
    --record --dataset_repo_id ${HF_USER}/cup-stacking-demos \
    --num_episodes 50

# keyboard controller: needs the GUI and real-time speed
python scripts/collect_cup_stacking.py \
    --task HCIS-CupStacking-SingleArm-v0 \
    --input keyboard --device cuda --enable_cameras \
    --record --dataset_repo_id ${HF_USER}/cup-stacking-demos \
    --num_episodes 20
```

The script name and the `--input` option are examples; the template in
[section 5.3](#53-the-collection-entry-point) defines the other options. The
task id is replaced with that of the group's task. The script reports
success or failure for every attempt, and only successful attempts are saved.

## 7. Choosing `fps`

**Required**

Robot state and action are recorded every simulation step (60 Hz). Camera
images refresh at 30 Hz to save rendering cost, so each image appears on two
consecutive frames. One frame is recorded per `env.step()`; frames are never
subsampled.

The dataset `fps` is set to **60**, so that frame timestamps match simulation
time. The `fps` value is dataset metadata: it does not change the physics step
rate or how often frames are written.

- Both cameras are 640×480 RGB and refresh every 1/30 s (`update_period`).
- `sim.dt` and `decimation` are not changed to obtain a different rate.
  Evaluation runs the task with its default settings and applies one policy
  action per `env.step()`, so the data must be recorded at that same step rate.

## 8. Recording over several sessions

**Background**

To add episodes to a dataset finalized earlier, the script is run with
`--resume`, which sets the export mode to
`EnhanceDatasetExportMode.EXPORT_SUCCEEDED_ONLY_RESUME`
([section 5.3](#53-the-collection-entry-point)). The task, dataset id,
`tracked_object_names`, and `fps` must be the same as in the first session.
Only a finalized dataset can be resumed; copying the folder first protects the
existing episodes from a failed session.

## 9. Check the dataset

**Required**

1. **Load it and look at the numbers:**

   ```python
   from lerobot.datasets.lerobot_dataset import LeRobotDataset

   ds = LeRobotDataset("<hf_user>/cup-stacking-demos")
   print(ds.num_episodes, ds.num_frames, ds.fps)
   print(list(ds.features))
   ```

   The feature list contains one `object_position.<name>` entry per tracked
   object.

2. **Watch a few episodes** with the
   [LeRobot Dataset Visualizer](lerobot_dataset_visualizer.md). The first
   frame shows the freshly reset scene, the last frame shows the finished
   task, both cameras show the right view, and the state curves move when the
   arm moves.

3. **Run the object-position coverage check** on every dataset:

   ```bash
   uv run python scripts/verify_object_position_coverage.py init \
       --dataset datasets/<hf_user>/cup-stacking-demos \
       --out specs/cup_stacking_coverage.ttl
   # Fill the TODO values in the generated Turtle file, then:
   uv run python scripts/verify_object_position_coverage.py check \
       --dataset datasets/<hf_user>/cup-stacking-demos \
       --spec specs/cup_stacking_coverage.ttl
   ```

   These commands run on the host, from the repository root: `datasets/` on
   the host is the container's `~/.cache/huggingface/lerobot/`.
   [Object Position Coverage](object_position_coverage.md) explains the report. The
   check is a coverage diagnostic; it does not replace visual inspection or
   loading the dataset in LeRobot.

## 10. Troubleshooting

**Background**

| Symptom | Cause | Fix |
|---|---|---|
| `FileExistsError` right at start | The dataset folder already exists | Choose a new `repo_id`, delete the old folder, or [resume](#8-recording-over-several-sessions) |
| Recorder rejects the export mode | `env_cfg.recorders.dataset_export_mode` was not set before the recorder was created | Set it to `EXPORT_SUCCEEDED_ONLY` (or the resume variant) before creating the recorder |
| Dataset will not load / Parquet footer error | `finalize()` never ran (crash, `kill -9`, closed terminal) | Keep `finalize()` in `finally`; stop with Ctrl+C; resume only a finalized dataset, and use a new repo id for a corrupted one |
| Every attempt is reported as failed | `check_success(env)` returns `False`, or the controller ends the attempt too early | Print the object positions and compare them with the success check in [section 3](#3-the-task-configuration) |
| Error or empty episode right after a saved one | The success state was not cleared after the reset | After every `env.reset()`, set the success term back to `False` (`set_success(env, False)`, [section 5.3](#53-the-collection-entry-point)) |
| First frames show the previous episode's scene | The controller moved during the skipped steps | Hold still for at least 5 steps after each reset |
| Camera errors / images missing | Started without `--enable_cameras` | Add the flag |
| Missing `object_position.*` features | leisaac's `LeRobotRecorderManager` was imported instead of the repository's | Import from `simulator.utils.lerobot_recorder` ([section 5.2](#52-the-repositorys-recorder-and-object-positions)) and record again |
| `ValueError: ... tracked_object_names is empty` | The task configuration tracks no objects (only possible for a self-defined task) | Set `tracked_object_names` to the task's rigid objects |
| `KeyError: '<object name>'` on the first recorded frame | `env_cfg.recorders` was replaced, so the `states` buffer is missing | Keep the default `ActionStateRecorderManagerCfg` and change only its export mode |
| Keyboard does nothing | Running `--headless`, or the input adapter still waits for its start key | Run with the GUI and press the start key that the script defines |

## 11. Checklist

**Required**

- [ ] One collection entry point and one controller implemented.
- [ ] Any configuration change keeps the rollout contract, and its differences from `eval/<task>_eval.py` are known and reported ([section 2.1](#21-modifying-the-environment-configuration)).
- [ ] `num_envs=1` and `--enable_cameras`.
- [ ] `env_cfg.use_joint_position_actions()` is called before the environment is created.
- [ ] The controller reads object positions fresh every episode.
- [ ] The controller holds still for at least 5 steps after each reset.
- [ ] Each episode is a single attempt that ends shortly after the task is done.
- [ ] The group's definition of "done" is stricter than, or equal to, the built-in check, and `check_success(env)` enforces it.
- [ ] Every phase and every episode has a step limit.
- [ ] The recorder is imported from `simulator.utils.lerobot_recorder`, not from leisaac.
- [ ] Dataset `fps` is 60 in every session.
- [ ] `finalize()` runs at the end of every session.
- [ ] `meta/info.json` lists one `object_position.<name>` feature per tracked object.
- [ ] The dataset loads in LeRobot, a few episodes were inspected visually, and the object-position coverage check was run on it.
