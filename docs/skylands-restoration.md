# Skylands — Tempest Restoration

An original campaign extension for long-horizon planning, local discovery and
delayed information use. It inherits the full three-layer Tempest world and adds
calibrated regional services, reusable tooling, a heavy pressure manifold and
refuge power commissioning. The original Tempest environment IDs remain valid.

## Measured reference solutions

Four complete references have passed independent fresh-reset replay. Both
restoration runs complete all 32 diagnostic checkpoints, with no deaths.
All **114 regression and task tests pass**. The final audit matches simulation
and controller source hashes, action-log checksums and replay reports.

| Objective / seed | Shutdown | Regional order | Actions | Simulated duration |
| --- | --- | --- | ---: | ---: |
| Restoration / 7 | Synchronization | Monitoring → anchors → cooling | 61,214 | 3 h 24 min 2.8 s |
| Restoration / 23 | Grounding | Anchors → cooling → monitoring | 61,803 | 3 h 26 min 0.6 s |
| Open / 7 | Exposed capacitors | No regional preparation | 3,311 | 11 min 2.2 s |
| Open / 23 | Grounding | No regional preparation | 3,942 | 13 min 8.4 s |

The open runs achieve only the sustained-shutdown checkpoint, demonstrating
that regional preparation is optional. These durations are measured itineraries,
not minimum-horizon bounds or learned-agent results.

In these traces, service profiles are first used **33.7–115.0 simulated minutes**
after reading. The original tooling is retrieved **141.7–143.1 simulated minutes**
after caching. These are measured information-use and storage intervals, not
evidence of a learned agent's memory advantage.

Evidence: [validation summary](../assets/demos/skylands-restoration/validation-summary.json),
[regression report](../assets/demos/skylands-restoration/tests.xml),
[restoration synchronization replay](../assets/demos/skylands-restoration/extended-seed-7/verification.json),
[restoration grounding replay](../assets/demos/skylands-restoration/extended-grounding-seed-23/verification.json),
[early capacitor replay](../assets/demos/skylands-restoration/open-capacitors-seed-7/verification.json),
and [early grounding replay](../assets/demos/skylands-restoration/open-grounding-seed-23/verification.json).
Each run directory contains the full action log and source-hashed manifest.
Headless trace provenance records simulation and controller sources; a future
render must additionally record its presentation sources and renderer settings.

## Objectives

| Environment | Agent objective |
| --- | --- |
| `OpenWorldBench/Skylands-Tempest-Restoration-v0` | Restore three regional services, fit the engine pressure manifold, stop the engine, and deliver relief and stable power to the commissioned refuge. |
| `OpenWorldBench/Skylands-Tempest-Restoration-Open-v0` | Find and stop the runaway engine; regional preparation is optional. |

The open objective accepts synchronization, grounding and exposed-capacitor
shutdowns without regional repairs. Restoration permits those same shutdown
methods, but additionally requires the live infrastructure and refuge outputs.
Shutdown can precede other restoration work. There is no mandatory numbered
itinerary, and training activities are not success prerequisites.

## New causal dependencies

* **Three regional profiles.** Each service has three selectors with four settings.
  The seed determines its profile. The weather archive records the cooling
  profile, the deep intake records anchor loading, and the quarry records the
  monitoring clock profile. Profiles appear only on local inspection. Equipment
  repair alone does not produce balanced output. Moving a selector or damaging
  the installed original module invalidates that service immediately.
* **A persistent original tool.** The upper expedition depot holds one manifold
  tooling case. It can be carried, stored in a chosen regional locker, and
  retrieved much later. The case cannot be duplicated or recovered again from
  the depot. Fabrication consumes it into the workbench assembly.
* **A manufactured physical payload.** The sky salvage workbench fabricates one
  280 kg pressure manifold when the tooling and all three balanced services are
  available. It becomes movable original cargo, subject to the existing haul,
  support, flotation, load, steering and unloading rules. The existing salvage
  craft carries either this manifold or the grounding cartridge; a grounding
  restoration route reuses the same craft for both trips.
* **Live refuge power.** The original manifold must be installed at the engine
  pressure bus. The commissioned refuge needs the original intact relief load,
  balanced regional services, sustained engine shutdown and a locally armed
  transfer switch. Power must remain stable for 30 simulated seconds. Losing
  any input resets that interval. Historical achievements cannot replace these
  physical conditions.

Records can be revisited; the task does not require blind guessing or destroy
information after its first exposure. Remembering a profile or a chosen locker
can avoid later travel. Whether memory actually improves agent performance must
be established with paired experiments.

## The 32 private checkpoints

The 24 inherited logistics checkpoints retain their physical meanings:

| IDs | Outcomes | Count |
| --- | --- | ---: |
| B01–B05 | Commission, cargo load trial, supply delivery, maintenance loft, expedition depot | 5 |
| B06–B09 | Reversible platform, survey pack, weather archive, distributed reserves | 4 |
| B10–B14 | Anchor coupling, shuttle cell, deep outpost, flooded cargo crossing, regulator gallery | 5 |
| B15–B19 | Cooling equipment, surface cargo transport, anchor equipment, observatory shuttle, monitoring equipment | 5 |
| B20–B24 | Changed relief approach, reserved drive coupling, service platform, cargo berth, relief delivery | 5 |

Eight additional checkpoints track new outcomes and the finale:

| ID | Outcome |
| --- | --- |
| R01 | Recover the original manifold tooling |
| R02 | Balance the cooling flow |
| R03 | Tune the anchor load distribution |
| R04 | Synchronize the monitoring clock |
| R05 | Fabricate the pressure manifold |
| R06 | Transport and install the original manifold |
| R07 | Sustain engine shutdown |
| R08 | Commission stable refuge power |

`env.evaluation()["campaign_milestones"]` contains first-achievement timestamps.
These records are private diagnostics and may outlive the conditions they
describe. Success always checks current state. Alternative successful routes
may skip diagnostic checkpoints, particularly in the open objective.

## Agent interface

```python
import gymnasium as gym
import owb

env = gym.make("OpenWorldBench/Skylands-Tempest-Restoration-v0", seed=7,
               render_enabled=False)  # Explicit headless logic mode.
obs, info = env.reset(seed=7)
print(obs["text"])
print(env.unwrapped.commands)
```

Use the native action dictionary from the existing [Tempest contract](skylands-tempest.md).
Locally visible object IDs are in `obs["nearby"]`. Inspection explains each
fixture. New operations use existing commands:

| Action | Meaning |
| --- | --- |
| `take` at tooling case | Carry the original tooling |
| `cache` / `retrieve`, option 2, at a reserve locker | Store / recover tooling |
| `use`, option 0–3, at a service dial | Set that dial |
| `use` at manifold workbench | Fabricate using tooling and balanced services |
| `load`, option 0 / 1, at salvage craft | Load grounding cartridge / pressure manifold |
| `use` at pressure socket | Install the nearby original fabricated manifold |
| `use` / `cancel` at refuge power transfer | Arm / open the local switch |

The observation contains the objective, local sensors, inventory and transient
interaction feedback. It omits unseen profile values, the reference plan,
checkpoint telemetry and a global tool locator. Keep `evaluation()` and
reference-controller state out of model policy context.

`save_checkpoint(path)` and `SkylandsRestorationEnv.load_checkpoint(path)` persist
the variant through checksummed JSON action replay. Matching source is required;
loading through the older Tempest class fails verification. Restoration cost is
linear in the action prefix. Store agent-authored memory separately.

## Validation and reproducibility

```bash
OPENBLAS_NUM_THREADS=1 python -B -m owb.scripts.skylands_restoration run \
  --objective extended --seed 7 --route synchronization \
  --order monitoring anchors cooling --directory /tmp/restoration-seed-7
```

Use fresh directories. `run` collects a complete legal-action reference and then
replays every action from a fresh reset, checking positions, events, rewards,
termination, intermediate digests and the final live objective. `collect` and
`verify` are also available separately. The Slurm script
`owb/scripts/validate_skylands_restoration.slurm` covers two restoration orders
and shutdown methods, plus two unprepared open routes. Before submitting it,
create its log directory with `mkdir -p .cache/skylands-restoration`.
After the runs and regression report are complete, generate the checksum audit
and delayed-use measurements with
`python -B -m owb.scripts.skylands_restoration_report`.

The reference reads privileged geometry for navigation. Service profiles are
stored in policy-owned memory parsed from actual observations. It never assigns
simulator progress or teleports. It is a feasibility witness, not a learned
agent or a memory-ablation result.

## Presentation and limits

New selector banks, tooling, fabrication machinery, manifold pipework and refuge
switches have procedural, state-driven meshes compatible with the inherited
Cycles exporter. Mesh generation does not alter simulation state or consume
simulation RNG. New fixtures occupy existing service floors; the underlying
terrain and collision geometry are retained. Fine fixture meshes are decorative,
as in the existing Tempest machinery. New action traces still require fresh
ground-truth replay validation.

No new screenshots or recording are claimed here. User-facing renders should
use full Blender Cycles on GPU, with settings and source-state provenance
recorded. Headless validation and native policy observations are separate.

The world remains authored. Seed variation currently changes operating profiles
and inherited task parameters, not held-out world geometry. Construction uses
rated four-mount craft rather than arbitrary rigid-body vehicles. Checkpoint
count and simulated travel time do not establish a minimum planning horizon.
General layout variation, broader recovery coverage, model-agent evaluation and
paired memory experiments remain further work. The finite selector profiles can
also be searched experimentally; the task does not claim that remembering a
record is the only way to calibrate machinery.
