# Object Pose Coverage: Check Your Own Dataset

At evaluation time your policy is run with objects starting at poses it has never seen. A policy usually handles only the starting poses it saw often enough in training. This tool shows you **where in the workspace your dataset actually has episodes**, measured against a requirement **you** write.

- It is a **self-check**, not part of grading. We evaluate your trained policy, not your dataset. You decide the region, the resolution and how many episodes are enough.
- It follows the **Object Pose (POSE)** axis of STAR-Gen, [A Taxonomy for Evaluating Generalist Robot Manipulation Policies](https://stargen-taxonomy.github.io/): same task, same objects, different object poses.

Contents:

1. [Workflow](#1-workflow)
2. [Record object poses](#2-record-object-poses)
3. [Write your spec](#3-write-your-spec)
4. [Run the check and read the report](#4-run-the-check-and-read-the-report)
5. [Troubleshooting](#5-troubleshooting)
6. [What the check does not cover](#6-what-the-check-does-not-cover)
7. [How it works: RDF, OWL, SHACL](#7-how-it-works-rdf-owl-shacl)

---

## 1. Workflow

```
record episodes ──▶ init spec ──▶ fill in TODOs ──▶ check ──▶ collect where cells are empty ──┐
      ▲  (object_pose.* columns)                                                             │
      └──────────────────────────────────────────────────────────────────────────────────────┘
```

| Step | Where it runs | Command / code |
|---|---|---|
| Record, with object poses | Isaac Sim container | your pipeline + section 2 |
| Create a spec skeleton | host (`uv` env) | `verify_object_pose_coverage.py init` |
| Fill in the requirement | any editor | section 3 |
| Check | host (`uv` env) | `verify_object_pose_coverage.py check` |

The check reads only `meta/info.json` and `data/**/*.parquet`. It never starts the simulator and never decodes video, so it runs in seconds.

---

## 2. Record object poses

### 2.1 The column

| Key | dtype | shape | Content |
|---|---|---|---|
| `object_pose.<name>` | float32 | `(7,)` | `[x, y, z, qw, qx, qy, qz]` |

- **Position:** metres, relative to the env origin (env-local frame). This is the frame used by `init_state.pos` in the task cfgs, so the numbers you see there and in your spec are directly comparable.
- **Orientation:** quaternion `(w, x, y, z)`, IsaacLab order. Not used by the check yet, but recorded so you can analyse it yourself.
- **Every frame.** The check uses the first frame of each episode, and a full trajectory is useful for your own debugging.
- **Name** = the object's scene name (`blue_cup`, `fork`, ...). Keep the `object_pose.` prefix exactly.

These columns are **not policy inputs**: LeRobot only feeds `observation.*` keys and images to a policy. Your training and rollout do not change.

Which objects are recorded is set by `tracked_object_names` in the env cfg:

| Task | Default `tracked_object_names` |
|---|---|
| Cup stacking | `blue_cup`, `pink_cup` |
| Cutlery arrangement | `fork`, `knife`, `plate` |
| Toy blocks collection | `green_block`, `blue_block`, `red_block`, `storage_box` |

Change the list on your env cfg (`env_cfg.tracked_object_names = [...]`) if you want other rigid objects recorded.

### 2.2 Option A: leisaac's LeRobot recorder

Replace the leisaac import with the one from `simulator`. It behaves the same and also declares and writes the `object_pose.*` columns:

```python
# from leisaac.enhance.managers.lerobot_recorder_manager import LeRobotRecorderManager
from simulator.utils.object_pose_recording import LeRobotRecorderManager

env.recorder_manager = LeRobotRecorderManager(env_cfg.recorders, dataset_cfg, env)
```

Things to know about this recorder (they come from leisaac, not from the coverage tool):

- It only accepts `DatasetExportMode.EXPORT_SUCCEEDED_ONLY` and **discards episodes whose `success` term was false**. If you also want failed episodes in your dataset, write frames yourself (option B).
- It drops the first 5 steps of every episode, so frame 0 is about step 6. Objects spawned slightly above the table may have dropped onto it by then; in our test they shifted by a few millimetres sideways. That is fine for coverage.

### 2.3 Option B: your own frame writer

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from simulator.utils.object_pose_recording import object_pose_features, read_object_poses

names = env.cfg.tracked_object_names
dataset = LeRobotDataset.create(
    repo_id=...,
    fps=...,
    robot_type="franka_panda",
    features={**your_features, **object_pose_features(names)},
)

# each recorded frame, read at the same moment as the frame's observation:
frame = {...}                                  # your observation.state, images, action, task
frame.update(read_object_poses(env, names))    # adds object_pose.<name> for env 0
dataset.add_frame(frame)
```

`read_object_poses(env, names, env_index=i)` reads another env when you record from several parallel envs.

### 2.4 Quick self-test

After recording a few episodes:

```bash
uv run python -c "import json; f=json.load(open('datasets/<repo_id>/meta/info.json'))['features']; print([k for k in f if k.startswith('object_pose.')])"
```

You should see one key per tracked object.

---

## 3. Write your spec

A spec is a small text file ([Turtle](https://www.w3.org/TR/turtle/) format) that says, per object: *"the initial positions in my dataset should cover this rectangle, split into this grid, with at least this many episodes per cell."*

### 3.1 Create the skeleton

From a dataset (object names are taken from its `object_pose.*` columns):

```bash
uv run python scripts/verify_object_pose_coverage.py init \
    --dataset datasets/<repo_id> \
    --out specs/my_coverage.ttl
```

Before you have data, name the objects yourself:

```bash
uv run python scripts/verify_object_pose_coverage.py init \
    --objects blue_cup,pink_cup --out specs/my_coverage.ttl
```

`init` never overwrites an existing file. The raw template is also at `packages/data_verification/src/data_verification/templates/object_pose_coverage_spec.ttl` if you prefer to copy it by hand (replace `OBJECT_NAME`).

### 3.2 One object block

```turtle
# Already in my_dataset: 40 episode(s), initial x 8.930..9.050, y 6.400..6.560. For reference only.
spec:blue_cup a dv:TaskObject ;
    dv:objectName "blue_cup" .

spec:blue_cup_coverage a dv:CoverageRequirement ;
    dv:targetsAxis dv:ObjectPoseAxis ;
    dv:forObject spec:blue_cup ;
    dv:xMin "TODO" ; dv:xMax "TODO" ;
    dv:yMin "TODO" ; dv:yMax "TODO" ;
    dv:gridX "TODO" ; dv:gridY "TODO" ;
    dv:minEpisodesPerCell "TODO" .
```

| Field | Meaning | Type |
|---|---|---|
| `dv:objectName` | Object to check; must match `object_pose.<name>` in the dataset | text in quotes (already filled) |
| `dv:xMin`, `dv:xMax`, `dv:yMin`, `dv:yMax` | Rectangle, in metres, env-local frame, where you want initial positions covered | number, **no quotes** (`8.80`) |
| `dv:gridX`, `dv:gridY` | Number of cells along x and along y | whole number, no quotes (`3`) |
| `dv:minEpisodesPerCell` | Episodes needed in each cell | whole number, no quotes (`2`) |

- Delete a block to stop checking that object. Copy a block to check another one.
- Leave `dv:targetsAxis dv:ObjectPoseAxis` and `dv:forObject` as they are.
- The `# Already in ...` line shows the range your dataset has now. It is information only; the rectangle you write should describe where you *want* coverage.

### 3.3 Choosing the values

These are your decisions. Some questions that help:

**Rectangle (`xMin`..`yMax`)**
- Where could the object plausibly start at test time? The public eval file shows one nominal pose per object (`init_state.pos`) and how it is randomized (`pose_range`); the test uses different poses. Look at the table, the robot's reach and the cameras' view.
- Can the robot reach and grasp the object over the whole rectangle? Covering unreachable corners wastes collection time.
- The rectangle is per object. Objects of the same task can have different rectangles.

**Grid (`gridX`, `gridY`)**
- Cell size = `(xMax - xMin) / gridX`. A useful scale is "how far can the object move before the grasp has to change?" For small objects that is a few centimetres; for large or forgiving objects more.
- More cells show gaps in more detail but need more episodes: the minimum dataset size is `gridX × gridY × minEpisodesPerCell` per object.

**Minimum per cell (`minEpisodesPerCell`)**
- `1` only says "at least one example exists here". Higher values ask for repeated examples in every region.
- Check the budget: 3 × 3 cells × 3 episodes = 27 episodes per object, spread evenly. Random sampling needs more than that to fill every cell.

**Example (values are illustrative only, not a recommendation):**

```turtle
    dv:xMin 0.40 ; dv:xMax 0.70 ;
    dv:yMin -0.15 ; dv:yMax 0.15 ;
    dv:gridX 3 ; dv:gridY 3 ;
    dv:minEpisodesPerCell 2 .
```

= a 30 cm × 30 cm square split into nine 10 cm cells, each needing 2 episodes.

---

## 4. Run the check and read the report

```bash
uv run python scripts/verify_object_pose_coverage.py check \
    --dataset datasets/<repo_id> \
    --spec specs/my_coverage.ttl
```

Example output:

```
Episodes: 31

[blue_cup] x=8.800..9.100  y=6.290..6.590  covered 6/9 cells (>= 2 each)
  y2   0!  0!  0!
  y1    5   3   5
  y0    5   5   7
        x0  x1  x2

VIOLATION 'blue_cup' cell (i=0, j=2) x=[8.8, 8.9] y=[6.49, 6.59] has 0 episode(s); needs at least 2  [blue_cup_coverage/cell/0_2]
WARNING   Initial pose of 'blue_cup' at (9.5, 6.4) is outside the required region; it does not count toward coverage  [episode/30/placement/blue_cup]

RESULT: coverage spec NOT met
```

How to read it:

- **Grid:** each number is the number of episodes whose object started in that cell. `!` = below your minimum. Columns are x (`x0` = `xMin` side), rows are y, **+y at the top**.
- **VIOLATION:** a requirement is not met. Each empty or thin cell is one line with its exact bounds. That is where to collect next.
- **WARNING:** allowed but worth knowing. An initial position outside your rectangle does not count toward any cell. `[episode/30/...]` tells you which episode.
- **ERROR:** the check could not run properly (missing column, TODO left in the spec, ...). See section 5.
- **Every episode counts**, whether the attempt succeeded or failed, because coverage is about which starting scenes the dataset contains.

| Option | Effect |
|---|---|
| `--json report.json` | Full report (counts per cell, all findings) as JSON, for your own plots or scripts |
| `--strict` | Treat warnings as failures |
| `--extra-shapes my_rules.ttl` | Also run your own SHACL rules (section 7); repeatable |

Exit code: `0` = spec met, `1` = not met or error. Use it in your own scripts, e.g. to stop collecting once coverage is reached.

---

## 5. Troubleshooting

| Message | Cause | Fix |
|---|---|---|
| `meta/info.json declares no 'object_pose.<name>' features` | Dataset recorded without object poses | Section 2; re-record |
| `Dataset has no 'object_pose.<name>' column, which the spec requires` | Spec block for an object that was not recorded, or a typo in `dv:objectName` | Fix the name, delete the block, or add the object to `tracked_object_names` |
| `Spec still has TODO values: ...` | Unfilled fields | Fill them in (section 3.2) |
| `... must be a number without quotes, got "8.8"` | Quoted number | Write `8.8`, not `"8.8"` |
| `... must be a whole number without quotes, got 2.5` | Fraction in `gridX`/`gridY`/`minEpisodesPerCell` | Use a whole number |
| `Spec is not valid Turtle: ...` | Syntax error, usually a missing `;` or `.` | Compare with the template: `;` between fields of one block, `.` at the end of a block |
| `Coverage region is empty` | `xMin >= xMax` or `yMin >= yMax` | Swap the values |
| `episode(s) do not start at frame_index 0` | Your writer's frame numbering starts later | Start `frame_index` at 0 for every episode |
| `non-finite initial pose` | NaN/inf written as a pose | Check where `read_object_poses` is called |
| Everything lands in one cell | Objects never moved between episodes | Randomize object poses on reset in your pipeline |

---

## 6. What the check does not cover

- **Only x and y position.** Orientation (yaw) and height are recorded but not checked.
- **Each object separately.** It does not check combinations, e.g. "blue cup left *while* pink cup right", or the distance between objects.
- **Only the first frame.** It says nothing about trajectory quality, grasp success or timing.
- **Rectangles only.** For other shapes, use several requirements for the same object, or extend the rules (section 7).

Good coverage raises the chance that your policy generalizes to new poses; it does not guarantee it.

---

## 7. How it works: RDF, OWL, SHACL

You do not need this section to use the tool.

| Piece | In one sentence | File (under `packages/data_verification/src/data_verification/`) |
|---|---|---|
| **RDF** | Data stored as `subject → predicate → object` facts, e.g. `episode/3/placement/blue_cup → dv:x → 8.91` | built from your dataset by `graph.py` |
| **OWL** | The vocabulary: kinds of things (`Episode`, `ObjectPlacement`, `CoverageRequirement`, `CoverageCell`) and the properties linking them | `ontology/object_pose.ttl` |
| **Your spec** | Your requirement, written as RDF in that vocabulary | your `.ttl` file |
| **SHACL** | Rules: each rule picks nodes (every episode, every cell, ...) and states what must hold. The coverage rule is a SPARQL query counting initial placements inside each cell | `shapes/object_pose_coverage.ttl` |

What `check` does:

1. Loads your spec and splits each rectangle into `CoverageCell`s.
2. Converts each episode's first-frame positions into `ObjectPlacement` facts.
3. Runs the SHACL rules over everything with [`pyshacl`](https://github.com/RDFLib/pySHACL).
4. Prints each failed rule as one VIOLATION/WARNING line, plus the count grid.

To add your own rules (for example "initial z must be above the table"), write SHACL shapes in a separate `.ttl` file and pass it with `--extra-shapes`. The packaged shapes in `shapes/object_pose_coverage.ttl` are commented and are a good starting point. A minimal example:

```turtle
@prefix dv:  <urn:aicapstone:dv#> .
@prefix sh:  <http://www.w3.org/ns/shacl#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

dv:AboveTableShape a sh:NodeShape ;
    sh:targetClass dv:ObjectPlacement ;
    sh:property [ sh:path dv:z ; sh:minInclusive "0.5"^^xsd:double ;
                  sh:message "Object starts below the table" ] .
```
