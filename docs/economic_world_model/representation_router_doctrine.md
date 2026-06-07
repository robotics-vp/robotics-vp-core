# Representation Router Doctrine

## Core Doctrine

Evidence sources are not a hierarchy. Real observation, simulation, geometry,
generated video, human operator input, and prior replay are complementary ways
to obtain usable evidence. None is sovereign in the abstract.

The routing question is task/economic/context dependent:

Which source gives the best usable evidence for this task under the current
economic, uncertainty, time, compute, battery, and failure-cost constraints?

## Source Classes

The deployment router currently recognizes:

- `real_observation`: direct observation from a real environment or robot-era
  capture when actually available
- `simulation`: simulator-produced evidence when a simulator source and its
  sufficiency are real for the task
- `geometry`: analytic or structured geometric evidence, often cheap and
  useful when the task is shape, reachability, visibility, or constraint
  dominated
- `generated_video`: generated or rendered video evidence, useful only when its
  compute, time, battery, and sufficiency tradeoff is justified
- `human_operator_input`: operator or reviewer evidence, especially valuable
  for high-value, high-uncertainty, high-failure-cost tasks
- `prior_replay`: existing replay or receipt-backed evidence, often the right
  source for low-uncertainty or low-value repetition
- `unavailable`: an honest non-source when no declared source is available and
  sufficient

## Routing Rules

The router must preserve:

- source availability as a hard input, not an inference
- source-specific sufficiency as a hard gate
- task economics as normalized local inputs
- source lineage and functional contribution
- rejected-source reasons
- deterministic tie-break order for exact score ties
- input and receipt SHAs for replay

The deterministic tie-break order is only an exact-score replay rule. It is not
an ontology, a promotion rule, or a claim that one source class is generally
better than another.

## Lineage And Functional Contribution

Material provenance and functional contribution must remain separate. A source
can contribute real grounding, semantic compression, counterfactual coverage,
geometry constraints, operator judgment, or replay economy. Those contributions
must not be collapsed into flat source buckets such as "real vs sim vs video."

Generated video, simulation, geometry, human input, and prior replay are
complementary evidence forms. A later datapack may use several at once, but the
router's first job is to decide which source is economically justified for the
current task and to record why other sources were rejected.

## Honest Unavailability

Missing real data, calibrated simulator assets, provider runtimes, generated
video capacity, operator review, or replay coverage must be represented as
`unavailable` or as source-specific rejection reasons. The router must not
pretend missing providers, hardware, ROS2, Unitree, Isaac, RunPod, or GPU-backed
loops exist.

An unavailable receipt is a useful artifact. It tells the Economic WM that the
next economically justified action may be data collection, calibration, asset
creation, operator review, or no action rather than pretending an evidence
source exists.

## Authority Boundary

The router is a CPU-only deployment-evidence selection layer. It does not train
models, write weights, run providers, execute hardware, mutate reward or
controller math, promote policies, or claim runtime availability. Its output is
a deterministic receipt and a typed source decision for downstream planning and
audit.
