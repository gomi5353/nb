# Getting Started

End-to-end guide for the Physical AI imitation-learning pipeline: recording demonstrations in simulation with your own controller, then training and evaluating a robot manipulation policy.

## Overview

This project builds imitation-learning policies for robot manipulation tasks (cup stacking, cutlery arrangement, toy blocks collection). The pipeline has four stages:

1. **Record** demonstrations in Isaac Lab with a controller you write (scripted, keyboard, etc.)
2. **Train** a policy model with LeRobot
3. **Evaluate** the trained policy in the simulator (rollout)
4. **Verify and diagnose** whether the dataset covers the conditions needed by the task

## Where to run what

| Stage | Environment | Why |
|-------|-------------|-----|
| Data Collection, Rollout | GPU machine, inside the Isaac Lab container | Isaac Sim / Isaac Lab / Vulkan stack pinned in image |
| Training, Coverage check | GPU machine, host (`uv` environment) | Docker adds I/O overhead — train natively for throughput |

> **GlowsAI VNC terminal:** When using the VNC desktop terminal on GlowsAI, switch to the `glows` user first:
> ```bash
> su - glows
> ```

For details on the repo layout and developer setup, see [Developer Introduction](dev/introduction.md).

## Prerequisites

- **Python 3.11** — required by the project (pinned in `.python-version`)
- **[uv](https://docs.astral.sh/uv/)** — Python package manager used for this workspace
- **Git** — with submodule support
- **Docker** — required for simulation steps (Isaac Lab runs in a container)
- **Linux machine with Nvidia GPU + driver** — required for simulation and training. Verify with `nvidia-smi`.
- **Hugging Face account** — datasets and model checkpoints are stored on the Hub. Create a token at <https://huggingface.co/docs/hub/en/security-tokens>.

> **No local GPU?** Training and simulation require a Linux machine with an Nvidia GPU.
> See [Cloud GPU Setup with GlowsAI](https://docs.google.com/presentation/d/1xPHm3gNNvLbX7aOWnY_xW5dfVLEa9XiBuskMbRsgxI8/edit?usp=sharing) for instructions on using cloud GPU instances.

### Installing prerequisites

```bash
# Docker
sudo apt-get update
sudo apt-get install -y ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin

# Add your user to the docker group (avoids needing sudo for docker commands)
sudo usermod -aG docker $USER
newgrp docker
```

Verify installations:

```bash
docker --version
```

## Installation

```bash
git clone https://github.com/gomi5353/supercoolcourseproject
cd supercoolcourseproject

# Install the host Python environment (training, coverage check)
uv sync
source .venv/bin/activate
```

The Isaac Lab container is built the first time you launch it (Step 1).

### Hugging Face login

```bash
hf auth login --token <YOUR_HF_TOKEN>
export HF_USER=<your-huggingface-username>
```

Set `HF_USER` each terminal session — commands throughout this project reference it.

---

## Step 1: Record Demonstrations in Simulation

> **Run on: GPU machine, inside Docker container.**  Requires Linux + Nvidia GPU + Docker (see Prerequisites).

### Launch the Isaac Lab container

| Target | Machine |
|--------|---------|
| `make launch-isaaclab` | Local machine with an Nvidia GPU |
| `make launch-isaaclab-glowsai-4090` | GlowsAI RTX 4090 (VNC display :1) |
| `make launch-isaaclab-glowsai-l40s` | GlowsAI L40S (VNC display :1) |

The first launch initializes the git submodules and builds the image. All remaining commands in this
step run **inside the container**, from the repository root. The container is a new shell, so run
`export HF_USER=<your-huggingface-username>` there too.

### Write your collection script and record

You write the collection entry point for your task: a controller (a scripted state machine,
keyboard teleoperation, or another input source) connected to the simulator and to a LeRobot
dataset writer. [How to Record Episodes](how_to_record.md) explains the recorder, the action format,
a controller skeleton, and how to check the result. Decide yourself how many episodes you need and
which object positions they must cover.

Tasks:

| Task | Task id |
|------|---------|
| Cup stacking | `HCIS-CupStacking-SingleArm-v0` |
| Cutlery arrangement | `HCIS-CutleryArrangement-SingleArm-v0` |
| Toy blocks collection | `HCIS-ToyBlocksCollection-SingleArm-v0` |

A dataset with repo id `${HF_USER}/<repo_id>` recorded inside the container is saved to
`~/.cache/huggingface/lerobot/${HF_USER}/<repo_id>/`, which is mounted from
`datasets/${HF_USER}/<repo_id>/` on the host.

### Inspect and upload the dataset

Check a few episodes with the [LeRobot Dataset Visualizer](lerobot_dataset_visualizer.md). Then, on
the host, upload the dataset if you want to train from the Hub (you can also train from the local
folder, see Step 2):

```bash
hf upload ${HF_USER}/<repo_id> datasets/${HF_USER}/<repo_id> --repo-type dataset
```

---

## Step 2: Train a Policy

> **Run on: GPU machine, host (not inside Docker).**

It uses your dataset from Step 1 and produces a trained policy checkpoint.

**Quick start:**

```bash
uv sync && source .venv/bin/activate

lerobot-train \
  --dataset.repo_id=${HF_USER}/<repo_id> \
  --dataset.root=datasets/${HF_USER}/<repo_id> \
  --policy.type=diffusion \
  --output_dir=<your-output-dir> \
  --job_name=cupstacking \
  --policy.device=cuda \
  --wandb.enable=true \
  --policy.repo_id=${HF_USER}/my_policy
```

Training takes a few hours to over a day depending on dataset size and GPU.

For the complete flag reference, multi-GPU training, troubleshooting, and upload instructions, see [LeRobot Training Procedure](lerobot_training.md).

---

## Step 3: Evaluate in Simulation (Rollout)

> **Run on: GPU machine, inside Docker container.** Same Isaac Lab environment as Step 1.

Loads your trained policy and runs it in the simulator to evaluate performance.

```bash
# Inside the container
python scripts/rollout.py \
    --task=<eval_task_file> \
    --policy_type=lerobot-<policy_name> \
    --policy_checkpoint_path=<path/to/checkpoint> \
    --device=cuda \
    --enable_cameras \
    --eval_rounds=<num_rounds>
```

Each episode runs for the task's graded length (cup stacking 20 s, cutlery arrangement 40 s, toy
blocks collection 60 s).

Placeholders:

| Placeholder | Meaning |
|-------------|---------|
| `<eval_task_file>` | Path to an evaluation task file under `eval/` (see table below) |
| `<policy_name>` | LeRobot policy type, e.g. `diffusion`, `act`, `pi0` (the flag becomes `lerobot-diffusion`, etc.). Required |
| `<path/to/checkpoint>` | Local path to the `pretrained_model` directory of your checkpoint |
| `<num_rounds>` | Number of evaluation episodes to roll out |

Pick `--task` from the evaluation task files under `eval/`:

| Task | Eval file |
|------|-----------|
| Cup stacking | `eval/cup_stacking_eval.py` |
| Cutlery arrangement | `eval/cutlery_arrangement_eval.py` |
| Toy blocks collection | `eval/toy_blocks_collection_eval.py` |

For the full procedure including model download and flag reference, see [LeRobot Rollout (Policy Evaluation)](lerobot_rollout.md).

---

## Step 4: Data Verification — Why Did the Policy Fail?

A failed rollout does not automatically mean that the model architecture or training strategy is the
problem. Instead, first inspect whether the training dataset covers the conditions that
appeared during evaluation. To help you verify your collected data, we provide **object-pose coverage**
as a complete example from Star-Gen taxonomy. It records each object's pose in the [LeRobot v3 dataset](https://huggingface.co/docs/lerobot/main/en/lerobot-dataset-v3#format-design), describes the coverage
requirement in RDF/OWL, and uses SHACL to report which spatial cells have too few
episodes. Follow [Object Pose Coverage](object_pose_coverage.md) to record the
required columns, write a Turtle specification, and run the checker.

Coverage is a diagnostic, not proof of the cause: an uncovered evaluation region
is evidence of a possible data-distribution gap, while complete coverage does not
guarantee that a policy will succeed.

If your hypothesis concerns a different factor, you can also select it from Star-Gen taxonomy or from another paper, and follow
[Adding a New Dataset Coverage Factor](adding_coverage_factor.md). It explains the
full extension path from recording a LeRobot feature through RDF graph mapping,
OWL vocabulary, SHACL counting rules, reports, and tests.

---

## See Also

| Document | Description |
|----------|-------------|
| [Developer Introduction](dev/introduction.md) | Repo layout, environment setup, where to run what |
| [Isaac Lab + LeIsaac Configuration Tutorial](isaaclab_leisaac_tutorial.md) | Configuring Isaac Lab with LeIsaac |
| [How to Record Episodes](how_to_record.md) | Controller, recording loop, and dataset checks |
| [LeRobot Dataset Visualizer](lerobot_dataset_visualizer.md) | Inspecting datasets before training |
| [LeRobot Checkpoint Format](lerobot-model-format.md) | Understanding model checkpoint structure |
| [Standalone Env Config Export](standalone_env_config_export.md) | Exporting environment configs as standalone files |
| [Object Pose Coverage](object_pose_coverage.md) | Check whether initial object poses cover a required spatial region |
| [Adding a New Dataset Coverage Factor](adding_coverage_factor.md) | Extend verification to lighting, viewpoint, orientation, or another factor |
