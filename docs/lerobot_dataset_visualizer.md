# LeRobot Dataset Visualizer

This guide describes how recorded episodes are inspected visually before
training. It extends step 3 of [Getting Started](getting_started.md).

Every section starts with one of two labels: **Required** (something students
must do or follow) or **Background** (explanation without work items).

## 1. Purpose

**Background**

The visualizer shows the recorded camera videos together with per-frame plots
of `observation.state` and `action`, and an episode picker. It is used to
check episode boundaries, the continuity of the motion, and the camera
framing before GPU time is spent on training. It only reads datasets; it does
not record or edit them.

Two versions exist:

| Version | Where | Needs |
|---|---|---|
| Hosted Space, <https://huggingface.co/spaces/lerobot/visualize_dataset> | Browser | The dataset uploaded to the Hugging Face Hub |
| `lerobot-dataset-viz` | Host `uv` environment | Only the local dataset folder |

## 2. Inspect a few episodes

**Required**

### 2.1 Local viewer

On the host, with the `uv` environment active:

```bash
lerobot-dataset-viz \
    --repo-id ${HF_USER}/<dataset_name> \
    --root datasets/${HF_USER}/<dataset_name> \
    --episode-index 0
```

It opens the same plots and videos in a local rerun.io viewer. This version
needs no upload and suits quick checks while the collection script is still
being developed.

### 2.2 Hosted Space

The dataset is uploaded first, from the host:

```bash
hf upload ${HF_USER}/<dataset_name> datasets/${HF_USER}/<dataset_name> --repo-type dataset
```

Then either `${HF_USER}/<dataset_name>` is entered in the Space, or the dataset
is opened by a direct link:

```text
https://huggingface.co/spaces/lerobot/visualize_dataset?dataset=<hf_user>/<dataset_name>&episode=<N>
```

Public datasets open without signing in; private datasets require a Hub
account with read access. A link can be shared with the rest of the group.

## 3. What to look for

**Required**

| Panel | Source | What a good dataset shows |
|---|---|---|
| Video player | `observation.images.front`, `observation.images.wrist` | Correct framing, smooth playback. The first frame shows the freshly reset scene; the last frame shows the finished task. |
| State plot | `observation.state` (Franka joint positions) | Continuous trajectories within joint limits; the curves move when the arm moves. |
| Action plot | `action` (joint targets and gripper command) | Close to the state curves; gripper open and close switches where the task needs them. |
| Episode picker | `episode_index` | The expected number of episodes, with plausible lengths. |
| Frame scrubber | `frame_index`, `timestamp` | About `60 × episode seconds` frames (60 fps, one frame per simulation step; each camera image repeats on two consecutive frames). |

The visualizer reads the features from the dataset's `meta/info.json`; a
feature that the recorder did not declare does not appear. The
`object_position.*` columns are not displayed; they are checked with
[Object Position Coverage](object_position_coverage.md).

## 4. Common issues

**Background**

| Symptom | Cause |
|---|---|
| "Dataset not found" in the Space | Typo in the repo id, or a private dataset while not signed in. Opening the dataset's Hub page shows whether it is visible. |
| Video panel blank while the plots load | Videos failed to encode during recording, or were not uploaded. The local folder should contain `videos/observation.images.*/chunk-*/file-*.mp4`. |
| A camera is missing | The dataset was not recorded with the repository's recorder, or the scene was changed ([How to Record Episodes §5.4](how_to_record.md#54-what-each-frame-contains)). |
| Episode starts with the previous episode's scene | The controller moved during the 5 skipped steps after a reset ([How to Record Episodes §5.1](how_to_record.md#51-how-the-recorder-works)). |
| Episode ends early or late | The controller's `is_episode_done` or `check_success` misfires; compare it with the task's success check ([How to Record Episodes §3](how_to_record.md#3-the-task-configuration)). |

## 5. References

- Hosted visualizer: <https://huggingface.co/spaces/lerobot/visualize_dataset>
- LeRobot documentation: <https://huggingface.co/docs/lerobot>
