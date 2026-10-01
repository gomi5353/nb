# Adding a New Dataset Coverage Factor

This guide shows you how to extend the dataset verifier when object pose is not
the factor you want to investigate. Examples include lighting, camera viewpoint,
object orientation, appearance, background, and motion speed.

Before extending it, run the complete worked example in
[Object Pose Coverage](object_pose_coverage.md).

## 1. Decide whether you need a rule or coverage

| Question | Extension |
|---|---|
| Is every value valid or within a safety limit? | A SHACL rule |
| Does each episode contain the required metadata? | A SHACL rule |
| Do I have enough episodes across several ranges or categories? | A coverage factor |

A validation rule checks individual facts. A coverage factor defines bins or
categories, counts episodes in each one, and compares those counts with a
requirement.

For example, “light intensity must be finite” is a rule. “Each of three lighting
ranges must contain at least five episodes” is a coverage requirement.

## 2. Write a factor contract first

Define what you intend to measure before writing code:

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

Your contract must specify:

- the exact physical or semantic meaning;
- unit and coordinate frame, if applicable;
- dtype and shape;
- when the value is sampled;
- how frame values become one episode-level value;
- numeric bins or categorical values;
- what counts as sufficient coverage.

Choose the episode aggregation deliberately. Initial scene factors normally use
the first frame. Motion factors may need a minimum, maximum, mean, or trajectory
summary.

Use a key such as `factor.<name>` for verification-only metadata. Do not use
`observation.<name>` unless the factor is intentionally a policy input, because
LeRobot may expose `observation.*` features to the policy.

## 3. Understand the extension path

```text
collector
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

OWL describes what facts mean, while SHACL validates them. Neither can recover a
factor that was never recorded in the dataset.

## 4. Record the factor in LeRobot v3

Declare the feature when creating the dataset:

```python
features["factor.light_intensity"] = {
    "dtype": "float32",
    "shape": (1,),
    "names": ["lux"],
}
```

Write it with every frame:

```python
frame["factor.light_intensity"] = np.asarray([light_lux], dtype=np.float32)
dataset.add_frame(frame)
```

When using the repository's LeIsaac recorder, follow
[`object_pose_recording.py`](../packages/simulator/src/simulator/utils/object_pose_recording.py):

1. Add the feature to the recorder's feature dictionary.
2. Read the value from the environment.
3. Add it in `build_lerobot_frame()` at the same simulation step as the other
   frame values.

Collect one short episode and inspect the raw output before touching RDF:

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

Use
[`dataset.py`](../packages/data_verification/src/data_verification/dataset.py)
as the example. Add a data class and reader for your factor. The reader should:

1. Verify the feature declaration in `meta/info.json`.
2. Verify the column exists in every Parquet shard.
3. Group rows by `episode_index`.
4. Apply the aggregation from your contract.
5. Report missing, null, `NaN`, or infinite values with episode numbers.

Do not assume Parquet row order. The object-pose reader sorts by
`episode_index` and `frame_index` before selecting the initial value.

## 6. Describe it with RDF and OWL

Create a clearly named ontology under
`packages/data_verification/src/data_verification/ontology/`, such as
`lighting.ttl`. A minimal vocabulary might be:

```turtle
@prefix dv:   <urn:aicapstone:dv#> .
@prefix owl:  <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix xsd:  <http://www.w3.org/2001/XMLSchema#> .

dv:LightingCondition a owl:Class .
dv:hasLightingCondition a owl:ObjectProperty ;
    rdfs:domain dv:Episode ;
    rdfs:range dv:LightingCondition .
dv:lightIntensityLux a owl:DatatypeProperty ;
    rdfs:domain dv:LightingCondition ;
    rdfs:range xsd:double .
dv:LightingCoverageRequirement a owl:Class .
```

Use OWL to define concepts and relationships. Put executable constraints in
SHACL.

## 7. Map dataset values into the RDF graph

Follow [`graph.py`](../packages/data_verification/src/data_verification/graph.py).
Create one factor node for every episode, for example:

```turtle
data:episode_7 a dv:Episode ;
    dv:hasLightingCondition data:episode_7_lighting .

data:episode_7_lighting a dv:LightingCondition ;
    dv:lightIntensityLux 612.0 .
```

Keep extraction in Python and validation in RDF/SHACL. This lets you inspect the
generated graph without starting Isaac Sim.

## 8. Create the student's requirement template

Add a template under
`packages/data_verification/src/data_verification/templates/`:

```turtle
@prefix dv:   <urn:aicapstone:dv#> .
@prefix spec: <urn:aicapstone:spec:lighting#> .

spec:lighting a dv:LightingCoverageRequirement ;
    dv:minLux 300.0 ;
    dv:maxLux 900.0 ;
    dv:numberOfBins 3 ;
    dv:minEpisodesPerBin 5 .
```

Choose requirements from expected evaluation conditions, robot limitations, and
the collection budget. Do not simply use the minimum and maximum already found
in the dataset; that can hide the missing region you are trying to detect.

## 9. Add bins and SHACL constraints

`expand_cells()` in [`graph.py`](../packages/data_verification/src/data_verification/graph.py)
turns a two-dimensional object-pose region into cells. Add an equivalent
expansion function:

- numeric factor: create intervals;
- two-dimensional factor: create cells;
- categorical factor: create one bin per required category.

Add shapes under `packages/data_verification/src/data_verification/shapes/`.
They should check:

- required specification fields and datatypes;
- valid minimum, maximum, bin count, and episode minimum;
- exactly one valid episode-level value where appropriate;
- at least the required number of episodes in every bin;
- values outside the required domain.

Use
[`object_pose_coverage.ttl`](../packages/data_verification/src/data_verification/shapes/object_pose_coverage.ttl)
as the SPARQL counting example. For numeric boundaries, use `[min, max)` and
include the upper edge only in the final bin. Otherwise one episode may be
counted twice.

## 10. Add a report and command

Follow
[`verify_object_pose_coverage.py`](../scripts/verify_object_pose_coverage.py),
but give a distinct factor an explicit command:

```bash
uv run python scripts/verify_lighting_coverage.py init \
    --dataset datasets/<repo_id> --out specs/lighting.ttl

uv run python scripts/verify_lighting_coverage.py check \
    --dataset datasets/<repo_id> --spec specs/lighting.ttl
```

Keep these conventions:

- exit `0` when requirements are met and `1` otherwise;
- display counts per bin, not only pass/fail;
- identify malformed and out-of-range episodes;
- support JSON output for plots or automation;
- never modify the input dataset.

If two factors later share substantial behavior, refactor common binning code
after both are tested. Different factors often have different units,
aggregations, and boundary semantics.

## 11. Test the complete path

Use synthetic LeRobot v3 fixtures as in
[`test_object_pose_coverage.py`](../tests/test_object_pose_coverage.py). Test at
least:

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

Also run one small simulator collection. Synthetic data proves the checker, but
does not prove that the recorder reads the intended simulator quantity.

## 12. Definition of done

Another student should be able to:

- understand the factor, unit, sampling, and aggregation;
- collect it in a LeRobot v3 dataset;
- inspect the raw Parquet values;
- write a coverage requirement;
- run one documented command;
- see actionable coverage gaps;
- reproduce passing and failing tests;
- train normally without accidentally making verification metadata a policy
  input.

Coverage is evidence, not proof. A gap can support a hypothesis about policy
failure, but complete coverage does not guarantee policy success.
