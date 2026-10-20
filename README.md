# Physical AI

Imitation learning for robot manipulation in simulation. You pick one of three tasks, build your own
data collection for it in Isaac Lab, train a policy with LeRobot, and evaluate the policy with the
provided rollout script. A coverage checker helps you find out whether your dataset covers the
situations the evaluation uses.

> **Platform:** Linux with an NVIDIA GPU. Simulation runs in a Docker container; training runs on the host.

For a complete step-by-step walkthrough, see [Getting Started](docs/getting_started.md).

## Pipeline

| Step | Where | Guide |
|------|-------|-------|
| 1. Record demonstrations with your own controller | Isaac Lab container | [How to Record Episodes](docs/how_to_record.md) |
| 2. Train a policy with `lerobot-train` | Host | [LeRobot Training Procedure](docs/lerobot_training.md) |
| 3. Evaluate the policy with `scripts/rollout.py` | Isaac Lab container | [LeRobot Rollout (Policy Evaluation)](docs/lerobot_rollout.md) |
| 4. Check whether your data covers the evaluation conditions | Host | [Object Pose Coverage](docs/object_pose_coverage.md) |

## Tasks

| Task | Task id (for recording) | Eval file (for rollout) | Episode length |
|------|-------------------------|-------------------------|----------------|
| Cup stacking | `HCIS-CupStacking-SingleArm-v0` | `eval/cup_stacking_eval.py` | 20 s |
| Cutlery arrangement | `HCIS-CutleryArrangement-SingleArm-v0` | `eval/cutlery_arrangement_eval.py` | 40 s |
| Toy blocks collection | `HCIS-ToyBlocksCollection-SingleArm-v0` | `eval/toy_blocks_collection_eval.py` | 60 s |

## Quick start

1. **Clone the repository**

   ```bash
   git clone https://github.com/gomi5353/supercoolcourseproject
   cd supercoolcourseproject
   ```

2. **Install the host environment** (used for training and the coverage check)

   ```bash
   uv sync
   source .venv/bin/activate
   ```

   This makes `hf`, `lerobot-train`, and the other installed commands available in your terminal.

3. **Log in to Hugging Face** and set your username

   Create an access token at <https://huggingface.co/docs/hub/en/security-tokens>, then:

   ```bash
   hf auth login --token <YOUR_HF_TOKEN>
   export HF_USER=<your-huggingface-username>
   ```

   Commands throughout this project use `${HF_USER}`. Set it in every new terminal.

4. **Launch the Isaac Lab container** (used for recording and rollout)

   ```bash
   make launch-isaaclab               # local machine
   make launch-isaaclab-glowsai-4090  # GlowsAI RTX 4090
   make launch-isaaclab-glowsai-l40s  # GlowsAI L40S
   ```

   The first run initializes the git submodules and builds the image. On success, the shell drops you inside the container, in the
   repository root. Datasets you record inside the container are saved to `datasets/` on the host.

Then follow [How to Record Episodes](docs/how_to_record.md) to build your collection script.

## Documentation

| Document | Description |
|----------|-------------|
| [Getting Started](docs/getting_started.md) | End-to-end walkthrough |
| [How to Record Episodes](docs/how_to_record.md) | Build a controller and recording loop that writes a LeRobot dataset |
| [LeRobot Dataset Visualizer](docs/lerobot_dataset_visualizer.md) | Inspect a dataset before training |
| [LeRobot Training Procedure](docs/lerobot_training.md) | Train an imitation-learning policy |
| [LeRobot Checkpoint Format](docs/lerobot-model-format.md) | What a trained checkpoint contains |
| [LeRobot Rollout (Policy Evaluation)](docs/lerobot_rollout.md) | Run a trained policy in the simulator |
| [Object Pose Coverage](docs/object_pose_coverage.md) | Check whether initial object poses cover a required region |
| [Adding a New Dataset Coverage Factor](docs/adding_coverage_factor.md) | Extend the coverage check to another factor |
| [Isaac Lab + LeIsaac Configuration Tutorial](docs/isaaclab_leisaac_tutorial.md) | How a task config is put together |
| [Standalone Env Config Export](docs/standalone_env_config_export.md) | Export your own env config as a standalone file |
| [Developer Introduction](docs/dev/introduction.md) | Repo layout and Makefile targets |

## License

MIT — see [LICENSE](LICENSE).
