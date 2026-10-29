# Object Pose Coverage

This guide describes the object-pose coverage check, the data-verification
tool of the course project. It extends steps 3 and 6 of
[Getting Started](getting_started.md): every dataset is checked before training,
and the report is used again when a trained policy fails.

Every section starts with one of two labels: **Required** (something students
must do or follow) or **Background** (explanation without work items).

## 1. What the check does and why

**Background**

Imitation-learning policies tend to perform well only in situations that the
demonstrations represent well. If evaluation places objects where the dataset
has few or no episodes, the policy may fail because of missing data rather than
because of the model.

Object pose is the factor checked here because it is the main thing that
changes between episodes: every reset places the objects at a random position,
while the scene, robot, cameras, and task stay the same. Object poses can also
be read directly from the simulator, without labeling or estimation.

The check answers one question: **where on the table does the dataset actually
have episodes, measured against a requirement that the group writes?** The
group states, per object, a rectangle of the table, a grid that divides it into
cells, and a minimum number of episodes per cell. The check counts the episodes
whose object *started* in each cell and reports the cells below the minimum.

```text
record ──▶ init spec ──▶ fill in the values ──▶ check ──▶ collect where cells are thin ──┐
  ▲  (object_pose.* columns)                                                             │
  └──────────────────────────────────────────────────────────────────────────────────────┘
```

| Step | Runs in | Command |
|---|---|---|
| Record, with object poses | Isaac Lab container | The group's collection script with the repository recorder ([How to Record Episodes §5.2](how_to_record.md#52-the-repositorys-recorder-and-object-poses)) |
| Create a specification skeleton | Host `uv` environment | `verify_object_pose_coverage.py init` |
| Fill in the requirement | Any editor | [Section 3](#3-write-the-specification) |
| Check | Host `uv` environment | `verify_object_pose_coverage.py check` |

The check reads only `meta/info.json` and `data/**/*.parquet`. It never starts
the simulator and never decodes video, so it runs in seconds.

The values of the requirement (region, grid, minimum) are the group's decision.
Running the check is required for every dataset; the report is evidence for the
group's own analysis, not a grade.

## 2. The object-pose columns

**Background**

The check needs the `object_pose.*` columns. Only the repository's
`LeRobotRecorderManager` writes them; [How to Record Episodes
§5.2](how_to_record.md#52-the-repositorys-recorder-and-object-poses) explains
the recorder. Each column has this format:

| Key | dtype | Shape | Content |
|---|---|---|---|
| `object_pose.<name>` | float32 | `(7,)` | `[x, y, z, qw, qx, qy, qz]` |

- **Position**: metres, relative to the environment origin. This is the frame
  of `init_state.pos` in the task configurations, so those numbers and the
  numbers in a specification are directly comparable.
- **Orientation**: quaternion `(w, x, y, z)`. Not used by the default check, but
  recorded for the group's own analysis.
- **Every frame** is recorded. The check uses the first recorded frame of each
  episode, which is where the objects started.
- **Name**: the object's scene name (`blue_cup`, `fork`, ...).

The recorded objects are those in `tracked_object_names` of the task
configuration:

| Task | `tracked_object_names` |
|---|---|
| Cup stacking | `blue_cup`, `pink_cup` |
| Cutlery arrangement | `fork`, `knife`, `plate` |
| Toy blocks collection | `green_block`, `blue_block`, `red_block`, `storage_box` |

Details of how the recorder captures the poses:

- It skips the first 5 steps of every episode, so the first recorded frame is
  about step 6. Objects spawned slightly above the table may have settled by
  then; in tests they moved by a few millimetres, which does not matter for
  coverage.
- The pose in frame *t* is read after that frame's `env.step()`, while the
  observation is read before it, so the pose is one simulation step (1/60 s)
  later. Objects at rest do not move in that time.
- Only successful episodes are saved, so the check counts the starting
  positions of successful demonstrations.

A quick test after recording a few episodes:

```bash
uv run python -c "import json; f=json.load(open('datasets/<repo_id>/meta/info.json'))['features']; print([k for k in f if k.startswith('object_pose.')])"
```

The output lists one key per tracked object. An empty list means that the
dataset was recorded with leisaac's recorder and has to be recorded again.

## 3. Write the specification

**Required**

A specification is a small text file in [Turtle](https://www.w3.org/TR/turtle/)
format. For each object, it says: *"the initial positions in the dataset should
cover this rectangle, split into this grid, with at least this many episodes
per cell."*

### 3.1 Create the skeleton

From a dataset (object names are taken from its `object_pose.*` columns):

```bash
uv run python scripts/verify_object_pose_coverage.py init \
    --dataset datasets/<repo_id> \
    --out specs/my_coverage.ttl
```

Before any data exists, the objects are named directly:

```bash
uv run python scripts/verify_object_pose_coverage.py init \
    --objects blue_cup,pink_cup --out specs/my_coverage.ttl
```

`init` never overwrites an existing file. The raw template is
`packages/data_verification/src/data_verification/templates/object_pose_coverage_spec.ttl`;
it can also be copied by hand, replacing `OBJECT_NAME`.

### 3.2 One object block

```turtle
# Already in my_dataset: 40 episode(s), initial x 8.930..9.050, y 6.400..6.560. For reference only.
spec:blue_cup a cp:TaskObject ;
    cp:objectName "blue_cup" .

spec:blue_cup_coverage a cp:CoverageRequirement ;
    cp:targetsAxis cp:ObjectPoseAxis ;
    cp:forObject spec:blue_cup ;
    cp:xMin "TODO" ; cp:xMax "TODO" ;
    cp:yMin "TODO" ; cp:yMax "TODO" ;
    cp:gridX "TODO" ; cp:gridY "TODO" ;
    cp:minEpisodesPerCell "TODO" .
```

| Field | Meaning | Type |
|---|---|---|
| `cp:objectName` | Object to check; matches `object_pose.<name>` in the dataset | Text in quotes (already filled) |
| `cp:xMin`, `cp:xMax`, `cp:yMin`, `cp:yMax` | Rectangle, in metres and in the environment frame, where initial positions should be covered | Number, **no quotes** (`8.80`) |
| `cp:gridX`, `cp:gridY` | Number of cells along x and along y | Whole number, no quotes (`3`) |
| `cp:minEpisodesPerCell` | Episodes needed in each cell | Whole number, no quotes (`2`) |

- Deleting a block stops checking that object; copying a block checks another
  one.
- `cp:targetsAxis cp:ObjectPoseAxis` and `cp:forObject` stay as generated.
- The `# Already in ...` comment shows the range that the dataset has now. It
  is information only: the rectangle describes where coverage is *wanted*, so
  copying the current range would hide exactly the gaps the check is meant to
  find.

### 3.3 Choosing the values

These values are the group's decision. The following questions help.

**Rectangle (`xMin` .. `yMax`)**

- Where could the object plausibly start at test time? The public eval file
  shows each object's nominal pose (`init_state.pos`) and its randomization
  range (`pose_range`); the private test uses different poses. The table, the
  robot's reach, and the camera views also limit the region.
- Can the robot reach and grasp the object everywhere in the rectangle?
  Covering unreachable corners wastes collection time.
- The rectangle is set per object; objects of the same task can have different
  rectangles.

**Grid (`gridX`, `gridY`)**

- Cell size = `(xMax - xMin) / gridX`. A useful scale is how far the object can
  move before the grasp has to change: a few centimetres for small objects,
  more for large or forgiving ones.
- More cells show gaps in more detail but need more episodes: the minimum
  dataset size is `gridX × gridY × minEpisodesPerCell` per object.

**Minimum per cell (`minEpisodesPerCell`)**

- `1` only says that at least one example exists there. Higher values require
  repeated examples in every region.
- Budget: 3 × 3 cells × 3 episodes = 27 episodes per object if perfectly
  spread. Random object placement needs more than that to fill every cell.

**Example** (illustrative values, not a recommendation):

```turtle
    cp:xMin 0.40 ; cp:xMax 0.70 ;
    cp:yMin -0.15 ; cp:yMax 0.15 ;
    cp:gridX 3 ; cp:gridY 3 ;
    cp:minEpisodesPerCell 2 .
```

This is a 30 cm × 30 cm square split into nine 10 cm cells, each needing 2
episodes.

## 4. Run the check and read the report

**Required**

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

- **Grid**: each number is the count of episodes whose object started in that
  cell; `!` marks a count below the minimum. Columns are x (`x0` is the `xMin`
  side), rows are y, with **+y at the top**.
- **VIOLATION**: a requirement is not met. Each thin or empty cell is one line
  with its exact bounds; these are the places to collect next.
- **WARNING**: allowed but worth knowing. An initial position outside the
  rectangle does not count toward any cell; `[episode/30/...]` names the
  episode.
- **ERROR**: the check could not run (missing column, `TODO` left in the
  specification, ...). See [section 5](#5-troubleshooting).

| Option | Effect |
|---|---|
| `--json report.json` | Writes the full report (counts per cell, all findings) as JSON, for plots or scripts |
| `--no-graph-out` | Skips writing the RDF graph that SHACL validates. By default `check` writes it (specification, generated cells, every episode and initial placement) to `outputs/object_pose_coverage/<dataset>.ttl`, for inspection or SPARQL queries |
| `--strict` | Treats warnings as failures |
| `--extra-shapes my_rules.ttl` | Also runs the group's own SHACL rules ([section 7](#7-how-it-works-rdf-owl-shacl)); repeatable |

Exit code: `0` when the specification is met, `1` when it is not or when an
error occurs. A collection script can use it, for example, to stop collecting
once coverage is reached.

## 5. Troubleshooting

**Background**

| Message | Cause | Fix |
|---|---|---|
| `meta/info.json declares no 'object_pose.<name>' features` | Dataset recorded with leisaac's recorder instead of the repository's | Record again with the repository recorder ([How to Record Episodes §5.2](how_to_record.md#52-the-repositorys-recorder-and-object-poses)) |
| `Dataset has no 'object_pose.<name>' column, which the spec requires` | Block for an object that was not recorded, or a typo in `cp:objectName` | Fix the name or delete the block |
| `Spec still has TODO values: ...` | Unfilled fields | Fill them in ([section 3.2](#32-one-object-block)) |
| `... must be a number without quotes, got "8.8"` | Quoted number | Write `8.8`, not `"8.8"` |
| `... must be a whole number without quotes, got 2.5` | Fraction in `gridX`, `gridY`, or `minEpisodesPerCell` | Use a whole number |
| `Spec is not valid Turtle: ...` | Syntax error, usually a missing `;` or `.` | Compare with the template: `;` between the fields of one block, `.` at the end of a block |
| `Coverage region is empty` | `xMin >= xMax` or `yMin >= yMax` | Swap the values |
| `episode(s) do not start at frame_index 0` | Dataset edited or written by another tool | Record with the repository recorder |
| `non-finite initial pose` | NaN or infinity written as a pose (an object fell out of the scene) | Inspect that episode; drop it or fix the reset |
| Every episode lands in one cell | Objects never moved between episodes | Check that the task's reset randomization is active |

## 6. What the check does not cover

**Background**

- **Only x and y position.** Orientation and height are recorded but not
  checked.
- **Each object separately.** Combinations ("blue cup left *while* pink cup
  right") and distances between objects are not checked.
- **Only the first frame.** Trajectory quality, grasp success, and timing are
  not checked.
- **Rectangles only.** Other shapes need several requirements for the same
  object, or extra rules ([section 7](#7-how-it-works-rdf-owl-shacl)).

Good coverage raises the chance that a policy generalizes to new poses; it
does not guarantee it. Other factors can be added by following
[Adding a New Dataset Coverage Factor](adding_coverage_factor.md).

## 7. How it works: RDF, OWL, SHACL

**Background**

This section is not needed to use the tool.

| Piece | In one sentence | File (under `packages/data_verification/src/data_verification/`) |
|---|---|---|
| **RDF** | Data stored as `subject → predicate → object` facts, e.g. `episode/3/placement/blue_cup → cp:x → 8.91` | Built from the dataset by `graph.py` |
| **OWL** | The vocabulary: kinds of things (`Episode`, `ObjectPlacement`, `CoverageRequirement`, `CoverageCell`) and the properties linking them | `ontology/object_pose.ttl` |
| **Specification** | The group's requirement, written as RDF in that vocabulary | The group's `.ttl` file |
| **SHACL** | Rules: each rule selects nodes (every episode, every cell, ...) and states what must hold. The coverage rule is a SPARQL query that counts the initial placements inside each cell | `shapes/object_pose_coverage.ttl` |

What `check` does:

1. Loads the specification and splits each rectangle into `CoverageCell`s.
2. Converts each episode's first-frame positions into `ObjectPlacement` facts.
3. Runs the SHACL rules over everything with
   [`pyshacl`](https://github.com/RDFLib/pySHACL).
4. Prints each failed rule as one VIOLATION or WARNING line, plus the count
   grid.

Additional rules (for example "the initial z must be above the table") are
written as SHACL shapes in a separate `.ttl` file and passed with
`--extra-shapes`. The packaged shapes in `shapes/object_pose_coverage.ttl` are
commented and are a good starting point. A minimal example:

```turtle
@prefix cp:  <http://taica.course/cp/ontology#> .
@prefix sh:  <http://www.w3.org/ns/shacl#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

cp:AboveTableShape a sh:NodeShape ;
    sh:targetClass cp:ObjectPlacement ;
    sh:property [ sh:path cp:z ; sh:minInclusive "0.5"^^xsd:double ;
                  sh:message "Object starts below the table" ] .
```
