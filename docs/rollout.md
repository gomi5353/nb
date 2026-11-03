# Rollout (Policy Evaluation)

This guide describes how a trained policy is evaluated in the simulator with
`scripts/rollout.py`, and which input and output features a submitted policy
must use. It extends step 5 of [Getting Started](getting_started.md) and
assumes that setup (step 0) is complete and that a checkpoint exists
([LeRobot Training](lerobot_training.md)).

Every section starts with one of two labels: **Required** (something students
must do or follow) or **Background** (explanation without work items).

## 1. What rollout does

**Background**

Rollout runs **inside the Isaac Lab container**, because it starts the
simulator. It builds the environment from an evaluation task file, loads the
checkpoint, and runs a number of episodes (rounds). In each episode the policy
receives the camera images and joint state, predicts actions, and the
environment executes them until the task's success check passes or the episode
time runs out. At the end, rollout prints the success rate over all rounds. If
something goes wrong, for example an invalid flag value, it prints the error
and exits with a non-zero code.

The evaluation task files in `eval/` define the same tasks as the recording
task ids, with the evaluation's own nominal object positions, randomization
ranges, and episode length. The final ranking uses a separate private
configuration with different object positions, so a policy that only works on
the public file's exact positions is not enough.

Rollout never reads the recording task configuration (`packages/simulator/src/simulator/tasks/<task>/<task>_env_cfg.py`). A dataset recorded under a modified configuration must still produce a policy that
fits [section 4](#4-policy-input-and-output-contract)
([How to Record Episodes §2.1](how_to_record.md#21-modifying-the-environment-configuration)).

| Task | Eval file | Episode length |
|---|---|---|
| Cup stacking | `eval/cup_stacking_eval.py` | 20 s |
| Cutlery arrangement | `eval/cutlery_arrangement_eval.py` | 40 s |
| Toy blocks collection | `eval/toy_blocks_collection_eval.py` | 60 s |

## 2. Get the checkpoint

**Required**

A checkpoint trained on the same machine is used directly: the repository
folder is mounted in the container, so
`outputs/train/<job_name>/checkpoints/last/pretrained_model` works on both
sides.

A checkpoint on the Hugging Face Hub is downloaded first:

```bash
hf download ${HF_USER}/<policy_name> --local-dir checkpoints/<policy_name> --repo-type model
```

`--revision <tag>` selects a tagged version, for example `--revision v1`.

## 3. Run rollout

**Required**

Inside the container, from the repository root:

```bash
python scripts/rollout.py \
    --task=eval/cup_stacking_eval.py \
    --policy_type=lerobot-diffusion \
    --policy_checkpoint_path=outputs/train/<job_name>/checkpoints/last/pretrained_model \
    --device=cuda \
    --enable_cameras \
    --eval_rounds=50
```

Besides the success rate, every rollout writes a per-episode log to
`outputs/rollout/<timestamp>.jsonl` (no flag needed): each episode's initial
object positions and whether it succeeded. Comparing it with the dataset shows which
rollout positions the dataset does not cover and where to record next
([Object Position Coverage §8](object_position_coverage.md#8-compare-the-dataset-with-a-rollout)).
At start, rollout also prints a note when the eval file's rollout range file is
missing or out of date ([§8.1](object_position_coverage.md#81-generate-the-rollout-range-once)).

### Flag reference

| Flag | Description |
|---|---|
| `--task` | The task to evaluate on: an eval file under `eval/` (see the table above). It also accepts a registered task id or a `module:Class` reference ([section 5](#5-evaluating-a-self-defined-task)). |
| `--policy_type` | **Required.** `lerobot-<type>`, where `<type>` is the `--policy.type` used in training. One of `lerobot-diffusion`, `lerobot-act`, `lerobot-smolvla`, `lerobot-pi0`, `lerobot-pi05`, `lerobot-vqbet`, `lerobot-tdmpc`, `lerobot-sac`, `lerobot-groot`, `lerobot-xvla`. |
| `--policy_checkpoint_path` | **Required.** The checkpoint's `pretrained_model` folder (the one containing `config.json` and `model.safetensors`). |
| `--policy_action_horizon` | Number of actions executed from each inference before the policy runs again. Overrides `n_action_steps` from the checkpoint, which is used by default. It must be positive and at most the number of actions one inference predicts: `horizon - n_obs_steps + 1` for Diffusion Policy, `chunk_size` for ACT. ACT with temporal ensembling only accepts `1`. The values in use are printed at start. |
| `--policy_language_instruction` | Language instruction for language-conditioned policies. |
| `--device=cuda` | Runs simulation and inference on the GPU. |
| `--enable_cameras` | Enables camera rendering. Required, because the policy needs images. |
| `--eval_rounds` | Number of evaluation episodes. `0` (the default) runs without a time limit until success or a manual reset. |
| `--episode_length_s` | Episode length in seconds. By default the task's graded length is used; the value in use is printed at start. |
| `--seed` | Seed of the object-position and lighting randomization (default `42`). A fixed seed compares policies on the same episodes. |
| `--num_envs` | Number of environments simulated in parallel; each round evaluates this many episodes at once. |
| `--headless` | Runs without the GUI window. |

## 4. Policy input and output contract

**Required**

Rollout feeds the policy exactly the features below and expects the action
below. A checkpoint submitted for evaluation must use only these input
features and predict only this output. The dataset may contain other columns
(for example `object_position.*`); LeRobot ignores them because they are neither
`observation.*` nor `action`.

| Key | Direction | dtype | Shape | Meaning |
|---|---|---|---|---|
| `observation.state` | Input | float32 | `(9,)` | Measured joint positions in the order `panda_joint1..7, panda_finger_joint1, panda_finger_joint2`; arm in rad, fingers in m. |
| `observation.images.front` | Input | image/video | `(480, 640, 3)` | RGB uint8, height × width × channels, from the fixed front camera. |
| `observation.images.wrist` | Input | image/video | `(480, 640, 3)` | RGB uint8, from the wrist camera. |
| `action` | Output | float32 | `(8,)` | `[0:7]` absolute joint-position targets in rad; `[7]` gripper: `< 0` closes, `≥ 0` opens (±1 in recorded data). |

These are the features that the repository recorder writes
([How to Record Episodes §5.4](how_to_record.md#54-what-each-frame-contains)),
so a policy trained with default settings on such a dataset satisfies the
contract. Groups may experiment with other features, but the submitted
checkpoint must follow this table.

## 5. Evaluating a self-defined task

**Background**

`--task` accepts three forms, resolved by `simulator.tasks.external.resolve_task`:

| Form | Example |
|---|---|
| Registered task id | `--task HCIS-CupStacking-SingleArm-v0` |
| Path to a `.py` file that registers a task when imported | `--task eval/cup_stacking_eval.py` |
| `module:Class` reference | `--task my_pkg.tasks.eval:MyEvalEnvCfg` |

The eval files use the second form. An Advanced Level group that defines its
own task ([Adding a New Task](adding_new_task.md)) can evaluate on its task id
directly, or write its own evaluation file in the same form as the files in
`eval/`: a task configuration class plus a `gym.register(...)` call with a
unique `TASK_ID`.
