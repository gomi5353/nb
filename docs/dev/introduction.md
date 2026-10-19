# Developer Introduction

Setup and usage details for working in this monorepo.

## Where to run what

| Stage | Environment | Why |
|-------|-------------|-----|
| Recording demonstrations (your collection script) | **Docker container** | Isaac Sim / Isaac Lab / Vulkan stack pinned in image |
| Rollout (policy inference in sim) | **Docker container** | Same Isaac Sim runtime as recording |
| Training (`lerobot-train`, `accelerate launch`) | **Host machine** | Container adds I/O + GPU passthrough overhead — train natively for throughput |
| Coverage check (`scripts/verify_object_pose_coverage.py`) | **Host machine** | Reads the dataset files only; no simulator needed |

Use `make launch-isaaclab` (or a GlowsAI target) for container work. Run training and the coverage
check against the host `uv` env (`uv sync`, then `lerobot-train ...`).

## Layout

- `packages/simulator/` — task configs and helpers on top of Isaac Lab and LeIsaac (`simulator.tasks`, `simulator.utils`)
- `packages/data_verification/` — dataset coverage checker (RDF/OWL/SHACL)
- `eval/` — evaluation task files used by `scripts/rollout.py`
- `scripts/` — `rollout.py` (policy evaluation), `verify_object_pose_coverage.py` (coverage check)
- `tests/` — host-side tests (`make test`)
- `dependencies/` — vendored submodules (Isaac Lab, etc.)
- `datasets/`, `checkpoints/`, `outputs/` — runtime artifacts (not tracked)

## Isaac Lab / Isaac Sim

Wraps upstream Isaac Lab with project task configs in `packages/simulator/`. New to the task config
layout? See the [Isaac Lab configuration tutorial](../isaaclab_leisaac_tutorial.md): it walks through
the single-arm Franka template, the cup-stacking task, and a recipe for adding a new task.

### Docker installation

CUDA 12.8 / Ubuntu 22.04 image. Installs Isaac Sim 5.1.0, Isaac Lab (submodule), the `simulator`
package, and LeRobot 0.4.4 (the same version as the host).

Driven by `Makefile`. Image tag set via `IMAGE` (default `leisaac-isaaclab:latest`), Dockerfile via `DOCKERFILE`.

| Target | Purpose |
|--------|---------|
| `make submodules` | Init/update git submodules (`dependencies/IsaacLab`, etc.) |
| `make submodules-pull` | Pull latest submodule revisions |
| `make install` | `submodules` + `uv sync` (host workspace install) |
| `make install-dev` | `submodules` + `uv sync --extra dev` |
| `make build-isaaclab` | Init submodules, build Docker image |
| `make launch-isaaclab` | Build, launch container with GPU + X11 + workspace bind-mount, NVIDIA Vulkan ICD probe |
| `make launch-isaaclab-glowsai-4090` / `-l40s` | Same, for GlowsAI machines (VNC display :1) |
| `make fix-perms` | Give files the container wrote under `datasets/` and `outputs/` back to your host user |
| `make check-isaaclab-gpu` | Verify NVIDIA Vulkan ICD, `nvidia-smi`, GLU/Xt libs, torch CUDA visibility inside image |
| `make test` | Run host-side tests |

Typical first-run flow:

```bash
make launch-isaaclab
```

It initializes the submodules and builds the image first. The Isaac Lab submodule must be
initialized before the build; the `Dockerfile` fails fast otherwise.

The container mounts the repository at `/workspace/aicapstone` and the host's `datasets/` at
`~/.cache/huggingface/lerobot` (LeRobot's default dataset folder), so a dataset recorded in the
container appears in `datasets/<repo_id>` on the host.

### Usage (run inside the container)

1. **Pick a task.** Task configs are in `packages/simulator/src/simulator/tasks/`.
2. **Record.** Write your own collection script (controller + recording loop); see
   [How to Record Episodes](../how_to_record.md).
3. **Evaluate.** Run `scripts/rollout.py` on an `eval/*.py` file; see
   [LeRobot Rollout](../lerobot_rollout.md).

### LeRobot & Hugging Face Hub workflow

Training runs on the **host machine** (training inside the container is significantly slower).

For the full training procedure, flag reference, and multi-GPU instructions, see [LeRobot training procedure](../lerobot_training.md).

Quick workflow:

1. **Train (single GPU, host).** `lerobot-train` with `--policy.type`, `--dataset.repo_id`, `--dataset.root=datasets/<repo_id>`, `--output_dir`, `--policy.device`, etc.
2. **Train (multi-GPU, host).** `accelerate launch --multi_gpu --num_processes=N $(which lerobot-train) <args>`.
3. **Upload datasets or checkpoints (optional).** `hf upload <repo> <local-dir> [--repo-type dataset] [--revision <tag>]`.
4. **Download checkpoints on another machine.** `hf download <model-repo> --local-dir <dir> --revision <tag>`.

Inspect datasets via the [LeRobot dataset visualizer](../lerobot_dataset_visualizer.md) to spot-check
episodes, action traces, and camera framing before training.

## Rollout (run inside the container)

Run a trained policy in sim. Entry: `scripts/rollout.py`. Required flags: `--task`, `--policy_type`,
`--policy_checkpoint_path`, `--enable_cameras`. See [LeRobot Rollout](../lerobot_rollout.md) for all flags.

Example:

```bash
python scripts/rollout.py \
    --task=eval/cup_stacking_eval.py \
    --policy_type=lerobot-diffusion \
    --policy_checkpoint_path=outputs/train/my_run/checkpoints/last/pretrained_model \
    --device=cuda \
    --enable_cameras \
    --eval_rounds=10
```
