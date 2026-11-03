# Object Position Coverage

This guide describes the object-position coverage check, the data-verification
tool of the course project. It extends steps 3 and 6 of
[Getting Started](getting_started.md): every dataset is checked before training,
and the report is used again when a trained policy fails. After a rollout, the
dataset is also compared with the positions the rollout actually used
([section 8](#8-compare-the-dataset-with-a-rollout)).

Every section starts with one of two labels: **Required** (something students
must do or follow) or **Background** (explanation without work items).

## 1. What the check does and why

**Background**

Imitation-learning policies tend to perform well only in situations that the
demonstrations represent well. If evaluation places objects where the dataset
has few or no episodes, the policy may fail because of missing data rather than
because of the model.

Object position is the factor checked here because it is the main thing that
changes between episodes: every reset places the objects at a random position,
while the scene, robot, cameras, and task stay the same. Object positions can also
be read directly from the simulator, without labeling or estimation.

**Why position and not the full pose.** The coverage check considers an object's 
position only, rather than its full pose (position and orientation) for the following reasons:

- **Position is the primary variable of interest.** The tasks randomize only the position. 
  Each object starts with a fixed rotation, so the orientation is the same in every episode 
  and says nothing about coverage.
- **Keeping orientation fixed improves episode reliability.** Randomizing initial orientations 
  could cause objects such as cups or cutlery to start in unstable configurations, potentially 
  leading them to tip over or fall off the table. Such episodes may become unusable, requiring 
  environment resets and additional simulation attempts, thereby wasting computational resources 
  on episodes that are never recorded.
- **Position allows straightforward spatial discretization.** Coverage is measured on the table 
  plane using the (x, y) coordinates. Positions in metres can be directly mapped onto a spatial 
  grid, whereas orientations represented by quaternions require more complex discretization and 
  have representational ambiguities (e.g., two quaternions can represent the same rotation).

Groups that randomize object orientations in their own experiments can incorporate 
orientation as an additional coverage factor ([Adding a Coverage Factor](adding_coverage_factor.md)).

The check answers one question: **where on the table does the dataset actually
have episodes, measured against a requirement that the group writes?** The
group states, per object, a rectangle of the table, a grid that divides it into
cells, and a minimum number of episodes per cell. The check counts the episodes
whose object *started* in each cell and reports the cells below the minimum.

```text
record ──▶ init spec ──▶ fill in the values ──▶ check ──▶ collect where cells are thin ──┐
  ▲  (object_position.* columns)                                                         │
  └──────────────────────────────────────────────────────────────────────────────────────┘
```

| Step | Runs in | Command |
|---|---|---|
| Record, with object positions | Isaac Lab container | The group's collection script with the repository recorder ([How to Record Episodes §5.2](how_to_record.md#52-the-repositorys-recorder-and-object-positions)) |
| Create a specification skeleton | Host `uv` environment | `verify_object_position_coverage.py init` |
| Fill in the requirement | Any editor | [Section 3](#3-write-the-specification) |
| Check | Host `uv` environment | `verify_object_position_coverage.py check` |

The check reads only `meta/info.json` and `data/**/*.parquet`. It never starts
the simulator and never decodes video, so it runs in seconds.

The values of the requirement (area, grid, minimum) are the group's decision.
Running the check is required for every dataset; the report is evidence for the
group's own analysis, not a grade.

## 2. The object-position columns

**Background**

The check needs the `object_position.*` columns. Only the repository's
`LeRobotRecorderManager` writes them; [How to Record Episodes
§5.2](how_to_record.md#52-the-repositorys-recorder-and-object-positions) explains
the recorder. Each column has this format:

| Key | dtype | Shape | Content |
|---|---|---|---|
| `object_position.<name>` | float32 | `(3,)` | `[x, y, z]` |

- **Position**: metres, relative to the environment origin. This is the frame
  of `init_state.pos` in the task configurations, so those numbers and the
  numbers in a specification are directly comparable.
- **No orientation**: the start rotation is fixed in every task
  ([section 1](#1-what-the-check-does-and-why)).
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

Details of how the recorder captures the positions:

- It skips the first 5 steps of every episode, so the first recorded frame is
  about step 6. Objects spawned slightly above the table may have settled by
  then; in tests they moved by a few millimetres, which does not matter for
  coverage.
- The position in frame *t* is read after that frame's `env.step()`, while the
  observation is read before it, so the position is one simulation step (1/60 s)
  later. Objects at rest do not move in that time.
- Only successful episodes are saved, so the check counts the starting
  positions of successful demonstrations.

A quick test after recording a few episodes:

```bash
uv run python -c "import json; f=json.load(open('datasets/<repo_id>/meta/info.json'))['features']; print([k for k in f if k.startswith('object_position.')])"
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

From a dataset (object names are taken from its `object_position.*` columns):

```bash
uv run python scripts/verify_object_position_coverage.py init \
    --dataset datasets/<repo_id> \
    --out specs/my_coverage.ttl
```

Before any data exists, the objects are named directly:

```bash
uv run python scripts/verify_object_position_coverage.py init \
    --objects blue_cup,pink_cup --out specs/my_coverage.ttl
```

`init` never overwrites an existing file. The raw template is
`packages/data_verification/src/data_verification/templates/object_position_coverage_spec.ttl`;
it can also be copied by hand, replacing `OBJECT_NAME`.

### 3.2 One object block

The specification declares one object-position axis at the top
(`spec:object_position_axis a cp:ObjectPositionAxis .`). Each object block adds
one requirement to it and describes the object, the area, and the grid:

```turtle
# Already in my_dataset: 40 episode(s), initial x 8.930..9.050, y 6.400..6.560. For reference only.
spec:object_position_axis cp:hasRequirement spec:blue_cup_requirement .

spec:blue_cup_requirement a cp:ObjectPositionAxisRequirement ;
    cp:hasTaskObject spec:blue_cup ;
    cp:hasAreaConfiguration spec:blue_cup_area ;
    cp:minEpisodesPerCell "TODO" .

spec:blue_cup a cp:TaskObject ;
    cp:hasName [ a cp:Name ; cp:nameValue "blue_cup" ] .

spec:blue_cup_area a cp:AreaConfiguration ;
    cp:hasCoverageGrid spec:blue_cup_grid ;
    cp:areaXMin "TODO" ; cp:areaXMax "TODO" ;
    cp:areaYMin "TODO" ; cp:areaYMax "TODO" .

spec:blue_cup_grid a cp:CoverageGrid ;
    cp:numCellW "TODO" ; cp:numCellH "TODO" .
```

| Field | Meaning | Type |
|---|---|---|
| `cp:nameValue` | Object to check; matches `object_position.<name>` in the dataset | Text in quotes (already filled) |
| `cp:areaXMin`, `cp:areaXMax`, `cp:areaYMin`, `cp:areaYMax` | Area (a rectangle), in metres and in the environment frame, where initial positions should be covered | Number, **no quotes** (`8.80`) |
| `cp:numCellW`, `cp:numCellH` | Number of cells along x (width) and along y (height) | Whole number, no quotes (`3`) |
| `cp:minEpisodesPerCell` | Episodes needed in each cell | Whole number, no quotes (`2`) |

- Deleting a block stops checking that object; copying a block checks another
  one.
- The links (`cp:hasRequirement`, `cp:hasTaskObject`, `cp:hasAreaConfiguration`,
  `cp:hasCoverageGrid`) stay as generated.
- The `# Already in ...` comment shows the range that the dataset has now. It
  is information only: the area describes where coverage is *wanted*, so
  copying the current range would hide exactly the gaps the check is meant to
  find.

### 3.3 Choosing the values

These values are the group's decision. The following questions help.

**Area (`areaXMin` .. `areaYMax`)**

- Where could the object plausibly start at test time? The public eval file
  shows each object's nominal position (`init_state.pos`) and its randomization
  range (`pose_range`); the private test uses different poses. The table, the
  robot's reach, and the camera views also limit the area.
- Can the robot reach and grasp the object everywhere in the area?
  Covering unreachable corners wastes collection time.
- The area is set per object; objects of the same task can have different
  areas.

**Grid (`numCellW`, `numCellH`)**

- Cell width = `(areaXMax - areaXMin) / numCellW`, cell height =
  `(areaYMax - areaYMin) / numCellH`. A useful scale is how far the object can
  move before the grasp has to change: a few centimetres for small objects,
  more for large or forgiving ones.
- More cells show gaps in more detail but need more episodes: the minimum
  dataset size is `numCellW × numCellH × minEpisodesPerCell` per object.

**Minimum per cell (`minEpisodesPerCell`)**

- `1` only says that at least one example exists there. Higher values require
  repeated examples in every region.
- Budget: 3 × 3 cells × 3 episodes = 27 episodes per object if perfectly
  spread. Random object placement needs more than that to fill every cell.

**Example** (illustrative values, not a recommendation):

```turtle
    cp:minEpisodesPerCell 2 .                    # in the requirement
    cp:areaXMin 0.40 ; cp:areaXMax 0.70 ;        # in the area
    cp:areaYMin -0.15 ; cp:areaYMax 0.15 .
    cp:numCellW 3 ; cp:numCellH 3 .              # in the grid
```

This is a 30 cm × 30 cm square split into nine 10 cm cells, each needing 2
episodes.

## 4. Run the check and read the report

**Required**

```bash
uv run python scripts/verify_object_position_coverage.py check \
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

VIOLATION 'blue_cup' cell (i=0, j=2) x=[8.8, 8.9] y=[6.49, 6.59] has 0 episode(s); needs at least 2  [blue_cup_grid/cell/6]
WARNING   Initial position of 'blue_cup' at (9.5, 6.4) is outside the required area; it does not count toward coverage  [episode/30/object/blue_cup]

RESULT: coverage spec NOT met
```

How to read it:

- **Grid**: each number is the count of episodes whose object started in that
  cell; `!` marks a count below the minimum. Columns are x (`x0` is the
  `areaXMin` side), rows are y, with **+y at the top**.
- **VIOLATION**: a requirement is not met. Each thin or empty cell is one line
  with its exact bounds; these are the places to collect next. The node name
  at the end (`blue_cup_grid/cell/6`) is the cell's row-major index
  `k = j × numCellW + i`.
- **WARNING**: allowed but worth knowing. An initial position outside the
  area does not count toward any cell; `[episode/30/...]` names the
  episode.
- **ERROR**: the check could not run (missing column, `TODO` left in the
  specification, ...). See [section 5](#5-troubleshooting).

| Option | Effect |
|---|---|
| `--json report.json` | Writes the full report (counts per cell, all findings) as JSON, for plots or scripts |
| `--no-graph-out` | Skips writing the RDF graph that SHACL validates. By default `check` writes it (specification, generated cells, every episode with its task objects and their initial positions) to `outputs/object_position_coverage/<dataset>.ttl`, for inspection or SPARQL queries |
| `--strict` | Treats warnings as failures |
| `--extra-shapes my_rules.ttl` | Also runs the group's own SHACL rules ([section 7](#7-how-it-works-rdf-owl-shacl)); repeatable |

Exit code: `0` when the specification is met, `1` when it is not or when an
error occurs. A collection script can use it, for example, to stop collecting
once coverage is reached.

## 5. Troubleshooting

**Background**

| Message | Cause | Fix |
|---|---|---|
| `meta/info.json declares no 'object_position.<name>' features` | Dataset recorded with leisaac's recorder instead of the repository's | Record again with the repository recorder ([How to Record Episodes §5.2](how_to_record.md#52-the-repositorys-recorder-and-object-positions)) |
| `Dataset has no 'object_position.<name>' column, which the spec requires` | Block for an object that was not recorded, or a typo in `cp:nameValue` | Fix the name or delete the block |
| `Spec still has TODO values: ...` | Unfilled fields | Fill them in ([section 3.2](#32-one-object-block)) |
| `... must be a number without quotes, got "8.8"` | Quoted number | Write `8.8`, not `"8.8"` |
| `... must be a whole number without quotes, got 2.5` | Fraction in `numCellW`, `numCellH`, or `minEpisodesPerCell` | Use a whole number |
| `Spec is not valid Turtle: ...` | Syntax error, usually a missing `;` or `.` | Compare with the template: `;` between the fields of one block, `.` at the end of a block |
| `Coverage area is empty` | `areaXMin >= areaXMax` or `areaYMin >= areaYMax` | Swap the values |
| `Requirement is not linked from an ObjectPositionAxis` | The `spec:object_position_axis cp:hasRequirement ...` line of a block was deleted or names another requirement | Restore the line from the template |
| `episode(s) do not start at frame_index 0` | Dataset edited or written by another tool | Record with the repository recorder |
| `non-finite initial position` | NaN or infinity written as a position (an object fell out of the scene) | Inspect that episode; drop it or fix the reset |
| Every episode lands in one cell | Objects never moved between episodes | Check that the task's reset randomization is active |

## 6. What the check does not cover

**Background**

- **Only x and y position.** Orientation is not recorded; height (z) is
  recorded but not checked.
- **Each object separately.** Combinations ("blue cup left *while* pink cup
  right") and distances between objects are not checked.
- **Only the first frame.** Trajectory quality, grasp success, and timing are
  not checked.
- **Rectangles only.** Other shapes need several requirements for the same
  object, or extra rules ([section 7](#7-how-it-works-rdf-owl-shacl)).

Good coverage raises the chance that a policy generalizes to new positions; it
does not guarantee it. Other factors can be added by following
[Adding a New Dataset Coverage Factor](adding_coverage_factor.md).

## 7. How it works: RDF, OWL, SHACL

**Background**

This section is not needed to use the tool.

| Piece | In one sentence | File (under `packages/data_verification/src/data_verification/`) |
|---|---|---|
| **RDF** | Data stored as `subject → predicate → object` facts, e.g. `episode/3/object/blue_cup/position → cp:positionX → 8.91` | Built from the dataset by `graph.py` |
| **OWL** | The vocabulary: kinds of things and the properties linking them (below) | `ontology/object_position.ttl` |
| **Specification** | The group's requirement, written as RDF in that vocabulary | The group's `.ttl` file |
| **SHACL** | Rules: each rule selects nodes (every episode, every cell, ...) and states what must hold. The coverage rule is a SPARQL query that counts the initial positions inside each cell | `shapes/object_position_coverage.ttl` |

The vocabulary, as a tree (dotted links ┄ are added by the verifier; the rest
is written in the specification):

```text
ObjectPositionAxis                      (a GeneralizationAxis)
├─ hasRequirement ─▶ ObjectPositionAxisRequirement     (an AxisRequirement; minEpisodesPerCell)
│                    ├─ hasTaskObject ─▶ TaskObject ─ hasName ─▶ Name (nameValue)
│                    └─ hasAreaConfiguration ─▶ AreaConfiguration (areaXMin .. areaYMax)
│                                               └─ hasCoverageGrid ─▶ CoverageGrid (numCellW, numCellH)
│                                                                     └┄ hasCell ┄▶ Cell ─ hasIndex ─▶ Index (k)
└┄ hasDataset ┄▶ Dataset
                 └┄ hasEpisode ┄▶ Episode
                                  ├┄ hasIndex ┄▶ Index (episode_index)
                                  └┄ hasTaskObject ┄▶ TaskObject
                                                      ├┄ hasName ┄▶ Name (nameValue)
                                                      └┄ hasPosition ┄▶ Position (positionX, positionY)
```

- A requirement's object and the episodes' objects are all `TaskObject`s; the
  check matches them by `nameValue`. Only the episodes' objects have a
  `Position`.
- A cell stores only its index `k`, row-major: column `i = k mod numCellW`,
  row `j = ⌊k / numCellW⌋`. The rule computes the cell's bounds from the area
  and the grid. A position on an inner cell edge belongs to the upper cell;
  the area's upper edges belong to the last column and row.
- Positions are stored as decimals rounded to 6 places (micrometres).

What `check` does:

1. Loads the specification and adds `numCellW × numCellH` indexed `Cell`s to
   each grid.
2. Converts each episode's first-frame positions into `TaskObject` and
   `Position` facts, and links the dataset from the axis.
3. Runs the SHACL rules over everything with
   [`pyshacl`](https://github.com/RDFLib/pySHACL).
4. Prints each failed rule as one VIOLATION or WARNING line, plus the count
   grid.

Additional rules (for example "the object must start in front of the robot") are
written as SHACL shapes in a separate `.ttl` file and passed with
`--extra-shapes`. The packaged shapes in `shapes/object_position_coverage.ttl` are
commented and are a good starting point. A minimal example:

```turtle
@prefix cp:  <http://taica.course/cp/ontology#> .
@prefix sh:  <http://www.w3.org/ns/shacl#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

cp:InFrontOfRobotShape a sh:NodeShape ;
    sh:targetClass cp:Position ;
    sh:property [ sh:path cp:positionY ; sh:minInclusive "0.0"^^xsd:decimal ;
                  sh:message "Object starts behind the robot" ] .
```

## 8. Compare the dataset with a rollout

**Required**

`check` compares the dataset with the group's own requirement. After a policy
has been evaluated, `compare` asks a second question: **did the dataset cover
the positions the rollout actually used?** It reads where each rollout episode
(run) started and whether it succeeded, and shows the runs that started where
the dataset is thin, together with the cells to record next. Repeating
record → train → rollout → compare closes those gaps one round at a time.

### 8.1 Generate the rollout range (once)

The eval file randomizes each object over a rectangle: its `init_state.pos`
plus the `pose_range` offsets. `rollout-range` reads those rectangles from the
eval file, without starting the simulator, and stores them in a range file:

```bash
uv run python scripts/verify_object_position_coverage.py rollout-range \
    --eval eval/cup_stacking_eval.py
```

```
Wrote outputs/rollout_ranges/cup_stacking_eval.ttl from eval/cup_stacking_eval.py (SHA-256 da1ed82706e9):
  blue_cup: x 8.8..9.0  y 6.38..6.58
  pink_cup: x 9.0..9.2  y 6.38..6.58
```

The range file records the eval file's SHA-256. When the file is missing, or
the eval file has changed since it was generated, both `scripts/rollout.py`
(a `NOTE` at start) and `compare` (an `ERROR`) print the command to run again.

### 8.2 Run the rollout

Every rollout ([Rollout §3](rollout.md#3-run-rollout)) writes
`outputs/rollout/<timestamp>.jsonl`, with no flag needed: one line per episode with each
object's position right after the reset (env-local frame, the same frame as
the dataset columns) and the outcome. Episodes stopped by a manual reset are
marked `aborted` and are not counted.

### 8.3 Compare

```bash
uv run python scripts/verify_object_position_coverage.py compare \
    --dataset datasets/<repo_id> \
    --spec specs/my_coverage.ttl \
    --rollout-log outputs/rollout/<timestamp>.jsonl \
    --record-cfg packages/simulator/src/simulator/tasks/cup_stacking/cup_stacking_env_cfg.py
```

How the comparison works, per object:

- **Rollout grid.** The object's rollout range is split into cells of **the
  same size as the cells of the group's grid** for that object, starting at
  the range minimum; the last column and row are clipped to the range. Each
  rollout cell counts the runs that started in it and how many succeeded.
  Rollout cells have no minimum: they only record what the rollout did.
- **Covered.** Because the cell sizes are equal, a rollout cell overlaps at
  most 2 × 2 = 4 cells of the group's grid (cells that only share an edge do
  not count). A rollout cell is covered when it lies inside the group's area
  and **every** group cell it overlaps has at least `minEpisodesPerCell`
  episodes. Only rollout cells with at least one run are judged.
- **In distribution.** A run is in distribution when, for every randomized
  object, the rollout cell it started in is covered.

**Example.** The cup-stacking eval file randomizes `blue_cup` over x 8.8–9.0,
y 6.38–6.58 and `pink_cup` over x 9.0–9.2, y 6.38–6.58
([section 8.1](#81-generate-the-rollout-range-once)). The group's
specification and dataset are:

| Object | Area in the specification | Grid | Minimum | Episodes per cell of the group's grid |
|---|---|---|---|---|
| `blue_cup` | x 8.8–9.0, y 6.4–6.6 | 2 × 2 (cells of 0.1 × 0.1 m) | 3 | (i=0, j=0): 5, (i=1, j=0): 3, (i=0, j=1): 3, **(i=1, j=1): 1** |
| `pink_cup` | x 9.0–9.2, y 6.38–6.58 | 2 × 2 (cells of 0.1 × 0.1 m) | 3 | 3 in every cell |

`blue_cup`'s area starts at y 6.4, so it misses the bottom 2 cm of the rollout
range, and its cell (i=1, j=1) has fewer than 3 episodes. A rollout with 20
episodes then gives:

```
Dataset : datasets/<repo_id>
Spec    : specs/my_coverage.ttl
Rollout : outputs/rollout/20261103-142530.jsonl  (eval/cup_stacking_eval.py, seed 42)
Range   : outputs/rollout_ranges/cup_stacking_eval.ttl
Runs    : 20 finished, 8 succeeded

[blue_cup] rollout x=8.800..9.000 y=6.380..6.580  cell 0.1 x 0.1 (your grid), 2x2 cells
  successes/runs per rollout cell; ! = not covered by your dataset, - = no runs
  y1   4/5 1/5!
  y0  1/4! 2/6!
         x0   x1
  covered: 1/4 sampled cells, 5/20 runs (25%)
  success: 4/5 covered vs 4/15 not covered

[pink_cup] rollout x=9.000..9.200 y=6.380..6.580  cell 0.1 x 0.1 (your grid), 2x2 cells
  successes/runs per rollout cell; ! = not covered by your dataset, - = no runs
  y1   3/5  2/5
  y0   1/5  2/5
         x0   x1
  covered: 4/4 sampled cells, 20/20 runs (100%)
  success: 8/20 covered vs 0/0 not covered

In distribution (covered on every axis: object_position): 5/20 runs (25%)
  success 4/5 in distribution vs 4/15 outside

Record more episodes here (2 more at least; cells with failed rollout runs first):
  blue_cup cell (i=1, j=1) x=[8.900, 9.000] y=[6.500, 6.600] centre (8.950, 6.550): has 1, needs 2 more; 5 rollout run(s) nearby, 4 failed

Rollout reached outside your area. Grow it by whole cells (same cell size), then record there:
  blue_cup: area x=[8.800, 9.000] y=[6.400, 6.600] -> areaXMin 8.8 ; areaXMax 9 ; areaYMin 6.3 ; areaYMax 6.6 ; numCellW 2 ; numCellH 3

Pose ranges covering those cells, as offsets from init_state.pos in packages/simulator/src/simulator/tasks/cup_stacking/cup_stacking_env_cfg.py:
  "blue_cup": {"x": (-0.15, 0.05), "y": (-0.14, 0.08), "z": (0.0, 0.0)},

WARNING   'blue_cup': 3 of 4 sampled rollout cell(s) are not covered by the dataset (15 run(s), 11 failed)  [blue_cup/rollout_cell/0, blue_cup/rollout_cell/1, blue_cup/rollout_cell/3]

RESULT: the dataset does NOT cover every rollout run; see above
```

The last three blocks answer three questions:

- **Record more episodes here**: *Where is data missing?* Cells of the group's
  own grid to record more episodes in, without changing the specification.
- **Grow your area**: *Is the area too small?* The rollout placed objects
  outside the area of the specification; the area has to be enlarged first.
- **Pose ranges**: *How should the objects be placed in the next recording?*
  The randomization range to set in the recording task so that new episodes
  land in the cells above.

How to read it:

| Part | Meaning | In the example above |
|---|---|---|
| **Runs** | Episodes of the rollout that ended with success or failure. Episodes stopped by a manual reset (`aborted`) are not counted. | The log has 20 episodes, none aborted; 8 succeeded. |
| **Header** `[blue_cup] rollout x=… cell … NxM cells` | The object's rollout range (from the range file), cut into cells of the same size as the group's cells, starting at the range minimum. | Cell size = (9.0 − 8.8) / 2 = 0.1 by (6.6 − 6.4) / 2 = 0.1, from the specification. The range is 0.2 × 0.2, so 0.2 / 0.1 = 2 cells each way. Rollout cell x0 is x 8.8–8.9, x1 is 8.9–9.0, y0 is y 6.38–6.48, y1 is 6.48–6.58. |
| **Grid** | `successes/runs` of the runs that started in each rollout cell, +y at the top as in `check`. `!` = the cell has runs and is not covered; `-` = no runs (not judged). A cell is covered when it lies inside the group's area and **every** group cell it overlaps has at least `minEpisodesPerCell` episodes. | `4/5` (x0, y1): 5 runs started in x 8.8–8.9, y 6.48–6.58, 4 succeeded. It overlaps group cells (0, 0) with 5 and (0, 1) with 3 episodes, both ≥ 3: covered. `1/5!` (x1, y1) overlaps (1, 0) with 3 and (1, 1) with only 1: not covered. `1/4!` and `2/6!` (row y0) start at y 6.38, below `areaYMin` 6.4: not inside the area, not covered. All `pink_cup` cells overlap group cells with 3 episodes: covered. |
| **covered** | Covered rollout cells / rollout cells with runs, and the runs in covered cells / all runs. | `blue_cup`: only (x0, y1) is covered, so 1/4 cells and 5/20 runs. `pink_cup`: 4/4 cells, 20/20 runs. |
| **success** | Success rate of the runs in covered cells compared with the runs in uncovered cells. | `blue_cup`: 4/5 in (x0, y1) vs 1 + 1 + 2 = 4 of 5 + 4 + 6 = 15 elsewhere. `pink_cup`: all 20 runs covered, so 8/20 vs 0/0. |
| **In distribution** | Runs whose starting cell is covered for **every** randomized object at once, with their success rate against the rest. A much lower rate outside points to missing data rather than the model. | A run counts only if both cups started in covered cells. All `pink_cup` cells are covered, so this equals `blue_cup`'s 5/20; success 4/5 (80 %) inside vs 4/15 (27 %) outside. |
| **Record more episodes here** — *where is data missing?* | Cells of the group's grid that are below the minimum and overlap a rollout cell with runs. Sorted by failed runs, then by missing episodes. `N more at least` is the sum of `needs … more` over the listed cells; `nearby` counts the runs in the uncovered rollout cells that overlap the cell. Recording there needs no change to the specification. | Group cell (i=1, j=1), x 8.9–9.0, y 6.5–6.6, has 1 episode; the minimum is 3, so it needs 2 more (the only listed cell, hence `2 more at least`). It overlaps only rollout cell (x1, y1): 5 runs, 5 − 1 = 4 failed. Row y0 also overlaps group row j=0, but those cells have 5 and 3 episodes, so they are not listed. |
| **Grow your area** — *is the area too small?* | Runs started in rollout cells that reach outside the group's area. The suggested area is enlarged by whole cells, so the cell size (and the rollout grid) stays the same. After updating the specification, run `check` again to identify new cells that need episodes to meet the minimum requirement. | Row y0 reaches down to 6.38, 0.02 below `areaYMin` 6.4. 0.02 is rounded up to one cell of 0.1: `areaYMin` 6.4 − 0.1 = 6.3 and `numCellH` 2 + 1 = 3. x already spans 8.8–9.0, so `areaXMin`, `areaXMax` and `numCellW` stay. |
| **Pose ranges** — *how to place the objects in the next recording?* | Shown with `--record-cfg`. The bounding box of every cell above (cells to record more in and rollout cells outside the area), minus the object's `init_state.pos` in the recording task cfg. Pasting it into the task's reset randomization ([How to Record Episodes §2.1](how_to_record.md#21-modifying-the-environment-configuration)) allows the next recording to sample object positions within the identified regions. It does not guarantee that every cell will be sampled or receive enough episodes. The box can include cells that are already covered. | The cells are x 8.9–9.0, y 6.5–6.6 (to record more) and x 8.8–9.0, y 6.38–6.48 (outside the area); together x 8.8–9.0, y 6.38–6.6. `cup_stacking_env_cfg.py` places `blue_cup` at (8.95, 6.52): x 8.8 − 8.95 = −0.15 to 9.0 − 8.95 = 0.05, y 6.38 − 6.52 = −0.14 to 6.6 − 6.52 = 0.08. `pink_cup` needs nothing, so it is not listed. |
| **WARNING** | One line per object with uncovered rollout cells. The names in brackets are the rollout cells, row-major `k = j × N + i`, as in `check`. | `blue_cup`: cells 0 (x0, y0), 1 (x1, y0) and 3 (x1, y1) are uncovered; 4 + 6 + 5 = 15 runs, 3 + 4 + 4 = 11 failed. |
| **ERROR** | The comparison could not run (stale range file, missing requirement, ...); see the table below. | None. |
| **RESULT** | Whether every run started in covered cells. | 15 runs did not, so NOT. |

| Option | Effect |
|---|---|
| `--rollout-range FILE` | Range file to use (default: `outputs/rollout_ranges/<eval file stem>.ttl` of the eval file named in the log) |
| `--record-cfg FILE` | Also prints pose ranges for the recording task cfg |
| `--json comparison.json` | Writes the full comparison (all cells, runs, suggestions) as JSON |
| `--no-graph-out` | Skips writing the RDF graph (default `outputs/object_position_coverage/<dataset>__rollout_<run>.ttl`) |
| `--strict` | Exits `1` when the dataset does not cover every run |

Exit code: `0` when the comparison ran (warnings list the uncovered runs), `1`
on an error such as a stale range file, or on warnings with `--strict`.

| Message | Cause | Fix |
|---|---|---|
| `No rollout range file ... Generate it: ...` | First comparison for this eval file | Run the printed `rollout-range` command |
| `... changed since ... was generated. Regenerate it: ...` | The eval file's positions or ranges were edited | Run the printed command |
| `The rollout ran ... with SHA-256 ..., but ... was generated from ...` | The rollout used a different version of the eval file than the range file | Run the rollout again with the current eval file, or generate the range from the file the rollout used |
| `... is randomized by the rollout but the spec has no requirement for it` | No block for that object in the specification | Add a block ([section 3.2](#32-one-object-block)) |
| `... starts at (...), outside its rollout range` | Should not happen; the reset samples inside the range | Check that the eval file and the range file match |

The comparison uses the public eval file only. The final ranking uses a
separate configuration with different positions
([Rollout §1](rollout.md#1-what-rollout-does)), so covering the public range
is necessary but not sufficient. As in `check`, only x and y are compared, and
each object separately.

### 8.4 How it works

**Background**

The comparison extends the vocabulary of [section 7](#7-how-it-works-rdf-owl-shacl)
with a rollout branch (dotted links ┄ are added by the verifier):

```text
Rollout
├┄ hasRolloutRange ┄▶ ObjectPositionRolloutRange       (a RolloutRange; sourceFile, sourceSha256)
│                     ├─ hasTaskObject ─▶ TaskObject ─ hasName ─▶ Name (nameValue)
│                     └─ hasAreaConfiguration ─▶ AreaConfiguration (areaXMin .. areaYMax)
│                                                └┄ hasCoverageGrid ┄▶ CoverageGrid (cellSizeX, cellSizeY, numCellW, numCellH)
│                                                                      └┄ hasCell ┄▶ Cell ─ hasIndex ─▶ Index (k)
└┄ hasEpisode ┄▶ Episode (succeeded)
                 ├┄ hasIndex ┄▶ Index
                 └┄ hasTaskObject ┄▶ TaskObject ─ hasName / hasPosition ─▶ …
```

- The range file holds the `ObjectPositionRolloutRange` nodes. `compare`
  creates one `Rollout` from the log, links it to every range, and gives each
  range's area a grid with the group's cell size.
- Ranges, requirements and episodes meet only through the objects'
  `nameValue`.
- SHACL rules in `shapes/rollout_comparison.ttl` check the rollout side (one
  outcome per episode, a position for every randomized object, well-formed
  ranges). A SPARQL query assigns each run to its rollout cell; the overlap
  with the group's cells is computed in `rollout_comparison.py`.

