# LeRobot Rollout (Policy Evaluation)

Run a trained policy in the Isaac Lab simulator to evaluate robot performance.

## Prerequisites

1. **Linux machine with Nvidia GPU** — verify with `nvidia-smi`. Isaac Lab requires a Linux host with an Nvidia driver.
2. **Docker installed** — the simulator runs inside a container.
3. **Repository cloned** — if you haven't already:
   ```bash
   git clone https://github.com/gomi5353/supercoolcourseproject
   cd supercoolcourseproject
   ```

## Step 1: Launch Isaac Lab Container

From the repository root, pick the target for your machine:

```bash
make launch-isaaclab               # local machine
make launch-isaaclab-glowsai-4090  # GlowsAI RTX 4090
make launch-isaaclab-glowsai-l40s  # GlowsAI L40S
```

This builds and starts the Isaac Lab Docker container. On success, your terminal drops into the container shell.

All remaining steps run **inside the container**.

## Step 2: Get Your Trained Policy

If you trained on the same machine, use the checkpoint directly:
`<output_dir>/checkpoints/last/pretrained_model` (the repository is mounted in the container, so
paths under it work in both places).

Otherwise, download it from the Hugging Face Hub:

```bash
export HF_USER=<your-huggingface-username>
hf download ${HF_USER}/<repo_id> --local-dir <download_path>
```

Replace:
- `<repo_id>` — the policy repository name you used during training (e.g., `my_policy`)
- `<download_path>` — where to save on disk (e.g., `checkpoints/my_policy`)

If you tagged a specific version during upload, add `--revision <tag>` (e.g., `--revision v1`).

For how to upload a trained policy, see [LeRobot Training — After Training](lerobot_training.md#after-training).

## Step 3: Run Rollout

```bash
python scripts/rollout.py \
    --task=<eval_task_file> \
    --policy_type=lerobot-<policy_type> \
    --policy_checkpoint_path=<checkpoint_path> \
    --device=cuda \
    --enable_cameras \
    --eval_rounds=<num_rounds>
```

At the end it prints the success rate over all rounds. If something goes wrong (for example an
invalid flag value), the script prints the error and exits with a non-zero code.

### Flag Reference

| Flag | Description |
|------|-------------|
| `--task` | Path to an evaluation task file under `eval/` (see table below). |
| `--policy_type` | **Required.** `lerobot-<type>`, where `<type>` is the `--policy.type` you trained with. One of `lerobot-diffusion`, `lerobot-act`, `lerobot-smolvla`, `lerobot-pi0`, `lerobot-pi05`, `lerobot-vqbet`, `lerobot-tdmpc`, `lerobot-sac`, `lerobot-groot`, `lerobot-xvla`. |
| `--policy_checkpoint_path` | **Required.** Path to the checkpoint's `pretrained_model` directory (the folder that contains `config.json` and `model.safetensors`). |
| `--policy_action_horizon` | Optional. Number of actions executed from each policy inference before the policy runs again. Overrides `n_action_steps` from the checkpoint; by default the checkpoint value is used. It must be positive and at most the number of actions one inference predicts: `horizon - n_obs_steps + 1` for diffusion, `chunk_size` for ACT. ACT with temporal ensembling only accepts `1`. The values in use are printed at start. |
| `--policy_language_instruction` | Optional. Language instruction for language-conditioned policies. |
| `--device=cuda` | Run simulation and inference on GPU. |
| `--enable_cameras` | Enable camera rendering for visual observations. Required. |
| `--eval_rounds` | Number of evaluation episodes to roll out. |
| `--episode_length_s` | Optional. Episode length in seconds. By default the task's graded length is used (see table below); the value in use is printed at start. |
| `--seed` | Seed for object-pose and lighting randomization (default `42`). Keep it fixed to compare policies on the same episodes. |
| `--headless` | Run without the GUI window. |

### Available Eval Task Files

| Task | Eval file | Episode length |
|------|-----------|----------------|
| Cup stacking | `eval/cup_stacking_eval.py` | 20 s |
| Cutlery arrangement | `eval/cutlery_arrangement_eval.py` | 40 s |
| Toy blocks collection | `eval/toy_blocks_collection_eval.py` | 60 s |

### Example

```bash
python scripts/rollout.py \
    --task=eval/cup_stacking_eval.py \
    --policy_type=lerobot-diffusion \
    --policy_checkpoint_path=checkpoints/my_policy \
    --device=cuda \
    --enable_cameras \
    --eval_rounds=50
```
