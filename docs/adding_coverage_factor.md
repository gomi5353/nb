# Adding a New Dataset Coverage Factor

This guide describes how to extend the dataset verifier for other factors that a group wants to investigate. 
Examples are lighting, camera viewpoint, object orientation, appearance, background, and motion speed. It
extends step 6 of [Getting Started](getting_started.md).

It assumes familiarity with the complete worked example in
[Object Pose Coverage](object_pose_coverage.md).

Every section starts with one of two labels: **Required** (something that an
extension must do) or **Background** (explanation without work items).

## 1. Decide between a rule and a coverage factor

**Background**

| Question | Extension |
|---|---|
| Is every value valid or within a safety limit? | A SHACL rule |
| Does each episode contain the required metadata? | A SHACL rule |
| Are there enough episodes across several ranges or categories? | A coverage factor |

A validation rule checks individual facts. A coverage factor defines bins or
categories, counts episodes in each one, and compares those counts with a
requirement.

For example, "light intensity must be finite" is a rule. "Each of three
lighting ranges must contain at least five episodes" is a coverage requirement.
A rule alone needs no code: it is a SHACL file passed with `--extra-shapes`
([Object Pose Coverage §7](object_pose_coverage.md#7-how-it-works-rdf-owl-shacl)).
The rest of this guide is about coverage factors.

## 2. Write a factor contract first

**Required**

The factor is defined in writing before any code:

```text
Factor: scene light intensity
Dataset key: factor.light_intensity
dtype and shape: float32, (1,)
unit: lux
sampling: every recorded frame
episode value: first recorded frame
required domain: 300 to 900 lux
bins: 3 equal-width bins
minimum: 5 episodes per bin
```

The contract specifies:

- the exact physical or semantic meaning;
- unit and coordinate frame, if applicable;
- dtype and shape;
- when the value is sampled;
- how frame values become one episode-level value;
- numeric bins or categorical values;
- what counts as sufficient coverage.

The episode aggregation is a deliberate choice. Initial scene factors normally
use the first frame. Motion factors may need a minimum, maximum, mean, or
trajectory summary.

Verification-only metadata uses a key such as `factor.<name>`. A key starting
with `observation.` is used only when the factor is intentionally a policy
input, because LeRobot exposes `observation.*` features to the policy.

## 3. The extension path

**Background**

```text
collection script
   │ records factor.<name>
   ▼
LeRobot v3 metadata and Parquet
   │ reader selects/aggregates one value per episode
   ▼
Python data model
   │ graph builder creates facts
   ▼
RDF data + OWL vocabulary + Turtle requirement
   │ SHACL validates and counts bins
   ▼
coverage report
```

OWL describes what the facts mean, while SHACL validates them. Neither can
recover a factor that was never recorded in the dataset.

## 4. Record the factor in LeRobot v3

**Required**

The repository's `LeRobotRecorderManager` in
[`lerobot_recorder.py`](../packages/simulator/src/simulator/utils/lerobot_recorder.py)
is the base for every verification factor: it already adds the
`object_pose.*` columns, and a new factor is appended next to them
([How to Record Episodes §5.2](how_to_record.md#52-the-repositorys-recorder-and-object-poses)).
Recording continues with that class, so the object poses stay in the dataset
next to the new factor.

1. Declare the feature in the recorder's feature dictionary
   (`build_with_factors` in `lerobot_recorder.py`):

   ```python
   features["factor.light_intensity"] = {
       "dtype": "float32",
       "shape": (1,),
       "names": ["lux"],
   }
   ```

2. Read the value from the environment.
3. Add it in `build_lerobot_frame()` in
   [`single_arm_franka_cfg.py`](../packages/simulator/src/simulator/tasks/template/single_arm_franka_cfg.py),
   next to the object poses, at the same simulation step as the other frame
   values:

   ```python
   frame["factor.light_intensity"] = np.asarray([light_lux], dtype=np.float32)
   ```

One short episode is recorded and its raw output inspected before any RDF work:

```bash
jq '.features["factor.light_intensity"]' datasets/<repo_id>/meta/info.json

uv run python - <<'PY'
import pyarrow.parquet as pq

table = pq.read_table(
    "datasets/<repo_id>/data/chunk-000/file-000.parquet",
    columns=["episode_index", "frame_index", "factor.light_intensity"],
)
print(table.slice(0, 5))
PY
```

This separates recording failures from verifier failures.

## 5. Read and aggregate the column

**Required**

[`dataset.py`](../packages/data_verification/src/data_verification/dataset.py)
is the example. The extension adds a data class and a reader for the factor.
The reader:

1. Verifies the feature declaration in `meta/info.json`.
2. Verifies that the column exists in every Parquet shard.
3. Groups rows by `episode_index`.
4. Applies the aggregation from the contract.
5. Reports missing, null, `NaN`, or infinite values with episode numbers.

Parquet row order is not guaranteed. The object-pose reader sorts by
`episode_index` and `frame_index` before selecting the initial value.

## 6. Describe the factor with RDF and OWL

**Required**

A clearly named ontology goes under
`packages/data_verification/src/data_verification/ontology/`, such as
`lighting.ttl`. A minimal vocabulary:

```turtle
@prefix cp:   <http://taica.course/cp/ontology#> .
@prefix owl:  <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix xsd:  <http://www.w3.org/2001/XMLSchema#> .

cp:LightingCondition a owl:Class .
cp:hasLightingCondition a owl:ObjectProperty ;
    rdfs:domain cp:Episode ;
    rdfs:range cp:LightingCondition .
cp:lightIntensityLux a owl:DatatypeProperty ;
    rdfs:domain cp:LightingCondition ;
    rdfs:range xsd:double .
cp:LightingCoverageRequirement a owl:Class .
```

OWL defines the concepts and relationships; executable constraints belong in
SHACL.

## 7. Map dataset values into the RDF graph

**Required**

[`graph.py`](../packages/data_verification/src/data_verification/graph.py) is the
example. The graph builder creates one factor node for every episode, for
example:

```turtle
data:episode_7 a cp:Episode ;
    cp:hasLightingCondition data:episode_7_lighting .

data:episode_7_lighting a cp:LightingCondition ;
    cp:lightIntensityLux 612.0 .
```

Extraction stays in Python and validation in RDF/SHACL, so the generated graph
can be inspected without starting Isaac Sim.

## 8. Create a requirement template

**Required**

A template goes under
`packages/data_verification/src/data_verification/templates/`:

```turtle
@prefix cp:   <http://taica.course/cp/ontology#> .
@prefix spec: <http://taica.course/cp/spec/lighting#> .

spec:lighting a cp:LightingCoverageRequirement ;
    cp:minLux 300.0 ;
    cp:maxLux 900.0 ;
    cp:numberOfBins 3 ;
    cp:minEpisodesPerBin 5 .
```

Requirement values come from the expected evaluation conditions, the robot's
limitations, and the collection budget. Copying the minimum and maximum
already found in the dataset would hide the missing region that the check is
meant to detect.

## 9. Add bins and SHACL constraints

**Required**

`expand_cells()` in
[`graph.py`](../packages/data_verification/src/data_verification/graph.py) turns a
two-dimensional object-pose region into cells. The extension adds an
equivalent expansion function:

- numeric factor: intervals;
- two-dimensional factor: cells;
- categorical factor: one bin per required category.

Shapes go under `packages/data_verification/src/data_verification/shapes/`. They
check:

- required specification fields and datatypes;
- valid minimum, maximum, bin count, and episode minimum;
- exactly one valid episode-level value where appropriate;
- at least the required number of episodes in every bin;
- values outside the required domain.

[`object_pose_coverage.ttl`](../packages/data_verification/src/data_verification/shapes/object_pose_coverage.ttl)
is the SPARQL counting example. Numeric bins use `[min, max)` and include the
upper edge only in the final bin; otherwise one episode may be counted twice.

## 10. Add a report and command

**Required**

[`verify_object_pose_coverage.py`](../scripts/verify_object_pose_coverage.py) is
the example; a distinct factor gets its own command:

```bash
uv run python scripts/verify_lighting_coverage.py init \
    --dataset datasets/<repo_id> --out specs/lighting.ttl

uv run python scripts/verify_lighting_coverage.py check \
    --dataset datasets/<repo_id> --spec specs/lighting.ttl
```

Conventions:

- exit `0` when the requirements are met and `1` otherwise;
- display counts per bin, not only pass or fail;
- identify malformed and out-of-range episodes;
- support JSON output for plots or automation;
- never modify the input dataset.

If two factors later share substantial behavior, common binning code is
refactored after both are tested. Different factors often have different
units, aggregations, and boundary semantics.

## 11. Test the complete path

**Required**

Tests use synthetic LeRobot v3 fixtures as in
[`test_object_pose_coverage.py`](../tests/test_object_pose_coverage.py) and cover
at least:

1. Every bin meets the minimum.
2. One bin is empty.
3. Values lie exactly on boundaries.
4. A value is outside the required domain.
5. The feature declaration or Parquet column is missing.
6. The shape or datatype is wrong.
7. A value is null, `NaN`, or infinite.
8. Episodes and frames are out of order.
9. The Turtle requirement contains missing or invalid values.
10. CLI exit codes and JSON output are correct.

One small simulator collection is also run. Synthetic data proves the checker,
but not that the recorder reads the intended simulator quantity.

## 12. Definition of done

**Required**

Another student can:

- understand the factor, unit, sampling, and aggregation;
- collect it in a LeRobot v3 dataset;
- inspect the raw Parquet values;
- write a coverage requirement;
- run one documented command;
- see actionable coverage gaps;
- reproduce passing and failing tests;
- train normally without the verification metadata becoming a policy input.

Coverage is evidence, not proof. A gap can support a hypothesis about a policy
failure, but complete coverage does not guarantee policy success.
