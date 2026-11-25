# Wildlands reference demonstrations

**Working-memory update (2026-11-24):** the videos, traces and preview images on
this page predate the working-memory probes `WM-WA07`, `WM-WB11` and `WM-WB15`
and do not replay on the current code. New replay-verified traces that include
the probes are in [`assets/evaluation/working-memory/wildlands/`](../assets/evaluation/working-memory/wildlands/).
They have not been rendered. The Wildlands checkpoint bank was also collected
before the probes and must be re-collected before checkpoint starts can restore
from it. See [Working-memory probes](#working-memory-probes).

**Current status:** both complete action traces are independently replay-verified,
and both walkthroughs are rendered in Blender Cycles (OptiX, 48 samples) with the
[presentation landscape](rendering.md#presentation-landscape), with every encoded
frame decoded and checked. Renderer settings and source hashes are recorded in each
course's `summary.json`.

An earlier [single Cycles checkpoint preview](../assets/demos/wildlands-reference/render-preview.png)
and its [state/render receipt](../assets/demos/wildlands-reference/render-preview.json)
predate the presentation landscape.

The prepared recordings demonstrate valid solutions to **compact executable versions**
of the [12- and 24-milestone task designs](wildlands-long-horizon-tasks.md).
They use a privileged, scripted controller. They do not establish that an agent
learned a reusable skill or retained it over a long horizon.

## Recordings and evidence

The [chapter player](../assets/demos/wildlands-reference/index.html) pairs each
milestone with instructions and its intended memory role. The films are
**checkpoint walkthroughs**: selected real simulator states held for three
seconds, with action captions and mechanism gauges. They are not continuous
gameplay recordings. Every intervening primitive action is preserved in the logs.

| Course | Milestones | Primitive actions | Simulated time | Video |
| --- | --- | --- | --- | --- |
| A: The Frostbound Pass | 12/12 | 1,018 | 203.6 seconds | [MP4](../assets/demos/wildlands-reference/task-a-seed-7/walkthrough.mp4) |
| B: The Valley Before the Storm | 24/24 | 1,791 | 358.2 seconds | [MP4](../assets/demos/wildlands-reference/task-b-seed-7/walkthrough.mp4) |

Both runs use seed 7 and complete without a death. These are valid reference
solutions, not optimality certificates or estimates of benchmark difficulty.

For each course, the artifact directory contains:

- `actions.jsonl`: every action, before/after position, reward, events, and a
  state hash at each completed milestone;
- `reference-manifest.json`: initial/final state hashes, source hashes,
  simulated duration, event counts, and implementation limitations;
- `verification.json`: an independent reset-and-replay check of all positions,
  events, rewards, milestone hashes, and final success;
- `chapters.json`: video timestamps and corresponding action/simulation times;
- `summary.json`: renderer settings, source hashes, video hash, and frame count.

The recorder replays and checks the trace again before rendering each state.
Scene capture is checked not to change the simulation. Assembly checks cached
image hashes and decodes every MP4 frame before publishing the video.

## Working-memory probes

The compact courses implement the three Wildlands probes of the
[working-memory contract](working-memory-probes.md) as extra physical
requirements of A07, B11 and B15 ([task notes](wildlands-long-horizon-tasks.md#working-memory-probes)).
Positions are world coordinates (x east, y up, z south); y = 30 is chest height
at course level.

| Probe | Source (cue) | Use site | Straight line | Reference path from cue to first attempt | First-attempt delay |
| --- | --- | --- | ---: | --- | --- |
| `WM-WA07` | Dispatch board on the depot access road, (130.5, 30, 58.5) | Keypad on the locker at the end of the stores warehouse aisles, (119.05, 33.2, 63.5) | 12.9 m | 72.3 m on foot: about 11 m to the dock, a 3 m climb, about 61 m of aisles | 119 decisions, 23.8 s |
| `WM-WB11` | Yard manifest at the yard gate, (35.5, 30, 62.5) | Six numbered bays south of the archive court; doors at x 28.5–33.5, (·, 30, 107.05) | 44.8 m | 57.2 m on foot, by way of the yard office | 88 decisions, 17.6 s (solved at 93, 18.6 s) |
| `WM-WB15` | Weather board against the east face of the station base, (95.5, 30, 96.0) | Pitch dial on the deck, (92.6, 34.2, 95.4–96.6), beside the colour-tagged rotor | 5.1 m | 42.3 m including the B14 vent, launch and glide (4 m climb) | 64 decisions, 12.8 s |

Delays are measured from the most recent cue read to the first judged entry or
opened bay. In both seed-7 reference runs each cue was read once and every first
attempt was correct. The reference is a privileged controller, so these delays
describe the course layout, not agent performance.

What makes the use site out of sight:

- **Stores warehouse** (A07): a closed brick building with a plank roof on a
  three-block stone loading dock, x 104–120, z 56–64, floor at y = 32. Its only
  doorway is in the east wall at (120, 32, 57), above a ladder on the dock face;
  the rest of the dock face leads up the wall to the closed roof. Inside,
  floor-to-ceiling plank racks at z 58, 60 and 62, each open at one end, make four
  aisles from the doorway to the locker at the far end of the last one. The
  locker moved here from the open court.
- **Coupling store** (B11): a brick block at x 27–34, z 104–106. Its six plank
  bay doors face south, away from the manifest. Opening a bay removes its door
  voxel. Bay labels (`pump fittings`, `engine fittings`, `couplings`, each used
  twice) are assigned from the probe's layout generator, independently of the contents.
- **Wind deck** (B15): the dial and rotor stand on the raised deck. A two-block
  brick parapet along the deck's east edge (x 93, y 33–34, z 93–99) hides them
  from every spot within reading reach of the board. The blade-number-to-colour
  assignment is drawn from the probe's layout generator and is never the board's
  listing order.

`WM-WA07` stays above ten seconds for any legal traversal, because the horse cannot
cover the indoor part. A galloping horse moves at 3× walking speed (10.8 against 3.6 m/s), so a fully
rideable route that takes 20–25 s on foot could be ridden in about 8–9 s. Riders
cannot climb or operate fixtures, a jump (apex 1.44 m, plus 0.65 m step-up) cannot
clear the 3 m dock, and the horse's own route planner finds no way up. The lower
bound from the dispatch board, in 0.2-second decisions, is:

- 4 decisions to reach the dock: one to mount, then a straight gallop over the
  6.05 m from the edge of reading reach. Sprinting there instead takes 5.
- 5 decisions for the climb: jump until the rise slows to climb speed, then climb at 1.9 m/s (1.17 s).
- 46 decisions through the aisles: the taut shortest path of the agent's 0.35 m
  radius body around the rack ends is 58.1 m, at a sprint of 6.4 m/s.
- 4 decisions to key the code.

That totals 59 decisions, or 11.8 s, against 23.8 s for the walking reference. A speed-up meal
(×1.22 on walking, sprinting and riding, not on climbing) would lower the bound
to 10.2 s. Course A has no speed-up food. `tests/test_wildlands_working_memory.py`
recomputes this bound from the current geometry and physics constants.

Tests check that no ray from the source, or from any standing spot within reading
reach of it, reaches a use-site fixture. They also check that the cue is absent
from policy observations (including the model runner's observation view) once the
agent leaves reach of the source.

The probes changed collision geometry and the A07, B11 and B15 mechanisms, so
the traces were re-collected. They match the current-code checkpoint-bank traces
exactly through A06 and B10. Course A now takes 1,205 actions (241.0 simulated
seconds) and course B 2,651 actions (530.2 seconds), both complete without a
death. The controller is `WorkingMemoryReferencePolicy` in
[`owb/scripts/wildlands_reference.py`](../owb/scripts/wildlands_reference.py).
It extends the unchanged base policy, parses each cue from the observation text
right after inspecting its source, and keeps the cue in its own memory. It also
climbs the stores dock and walks the warehouse aisles in and out.

```bash
python -m owb.scripts.wildlands_reference --verify assets/evaluation/working-memory/wildlands/task-a-seed-7
python -m owb.scripts.wildlands_reference --verify assets/evaluation/working-memory/wildlands/task-b-seed-7
```

Each `reference-manifest.json` records `working_memory` (the per-probe summary)
and `working_memory_layout` (positions, drawn content and cue text, evaluator-only).
`verification.json` repeats the summary from the independent replay.

## World, creatures and skills

The course is collision geometry, not presentation:

- **Organic ground.** Roads meander between the same junctions with varying
  width; pads are irregular; earth banks slope into the natural terrain instead
  of vertical walls ([`owb/world/organic.py`](../owb/world/organic.py)). Two
  stretches stay straight because mechanisms are authored on exact cells: the
  flooded return road and the storm-bridge plank.
- **Water is held by ground.** Every water block must have ground or water
  below it and on all four sides; generation drains anything else and asserts
  containment. The flooded road is a shallow dip flush with the road (draining
  leaves mud), and a river runs from the northern hills to the sea, stepping
  down over rock sills and passing under three roads on earthen bridges.
- **Creatures.** Monsters camp around dens in lowland they cannot walk out of
  onto the course; by day most rest in their dens (one per camp keeps watch),
  and all come out after dusk, with stalkers only at night. Deer, boar, goats
  and birds graze the lowlands; fish keep to deeper water
  ([`owb/world/fauna.py`](../owb/world/fauna.py)). Courses start in the
  afternoon (A 0.66, B 0.55) so each crosses dusk; nights are cold, and the
  agent puts on a packed warm doublet.
- **Riding is a learnable skill** ([`owb/sim/riding.py`](../owb/sim/riding.py)).
  A horse must be calmed with fruit (`feed`) before it accepts a rider (`ride`);
  sprinting near an unbonded horse spooks it. Distance ridden unlocks faster
  gaits (walk, trot at 80 m, canter at 250 m, gallop on the horse's stamina).
  Mounted riders cannot climb, glide, swim or operate fixtures (`dismount`).
  Once bonded, a horse answers a `whistle` from up to 150 m away on the same
  level: it plans a dry walking route (hopping single-block steps) and
  canters to where the rider whistled, giving up if it is blocked.
- **Apple trees** ([`owb/sim/orchard.py`](../owb/sim/orchard.py)) drop ripe
  apples on a deterministic schedule; striking a trunk shakes up to two loose.
  An apple restores half a heart.

Riding and the orchard are opt-in environment features, so other environments
and their published traces are unchanged. Walkthroughs hold glide, riding and
dusk moments briefly in addition to the milestone checkpoints.

## What is physically executed

The reference controller uses the public action dictionary. It does not teleport
the actor, modify the terrain, or write completion flags. Local interactions
enforce proximity, visibility, prerequisite, and inventory checks.

Counterweights use native magnetic bodies, gravity, collision, and supported-mass
measurements. Held bodies do not count as supported loads. Incorrect release
trips the brake; authored gate/lift actuators change collision geometry only
after valid support and stable operation.

Thermal stages use native jumping, gliding, and landing, with authored updraft
fields controlled by the vent, heat, and safety state. The quarry shaft requires
cooling before ignition and launch. Flow is a discrete network with routing,
inlet interlocks, pressure trips, and stable-time checks, rather than a general
fluid simulation. Cooking uses native ingredients and food effects.

The storm physically removes the wooden bridge. The controller takes the stone
detour to the same depot. Final operation removes the flooded-road obstruction;
the agent then walks through that road and reports completion. Final success
also requires that the relevant machinery remains operational.

## Scope relative to the research design

Travel, regional geography, and the teaching fixtures are compact. The course
does not implement the full mountain/coast/tower geography, long retention
intervals, a separate animated teacher, varied transfer geometry/materials, or
controlled memory ablations. Some family-symbol and location cues are authored;
the oracle reads privileged state and cannot serve as a memory-learning result.

The recordings demonstrate the intended **support/release**, **safe updraft**,
and **route-before-feed** procedures and their composition. A scientific memory
evaluation still needs the full-duration environment and comparisons between
memory conditions, held-out instances, and appropriate diagnostic baselines.

## Rendering and reproduction

Presentation uses Blender 4.5.4 Cycles (OptiX), 48 samples, OpenImageDenoise,
six maximum bounces, and AgX. The scene is rendered at its native 960 × 540 resolution and
placed without upscaling in a 1280 × 720 caption layout. Videos use H.264 Baseline,
`yuv420p`, fast-start metadata, and 10 encoded frames per second; checkpoint
images are repeated for reading. Rendering does not replace policy observations.

Render from the repository root on the Slurm login node:

```bash
bash owb/scripts/regenerate_wildlands.sh
```

The script moves any previous frame cache and published videos aside, then
submits two dependent jobs:

1. [`render_wildlands.slurm`](../owb/scripts/render_wildlands.slurm): six
   array tasks on `gpu_requeue` (RTX A6000, RTX 6000 Pro, or A40), one per
   milestone range balanced by checkpoint frames (A: 1–5, 6–12; B: 1–5, 6–13,
   14–18, 19–24). Each task replays its trace from the start with full checks,
   then renders its checkpoint states with Cycles OptiX in one persistent
   Blender process ([`cycles_batch.py`](../owb/render/cycles_batch.py)). Each
   frame starts from Blender's factory settings and runs the unchanged scene
   builder, so it matches a fresh Blender process per frame. Python exports the
   next state while Blender renders the previous one. Preempted tasks requeue
   and resume from verified saved frames.
2. [`assemble_wildlands.slurm`](../owb/scripts/assemble_wildlands.slurm): on a
   CPU node, replay-verifies both traces, encodes and decodes both walkthroughs
   from the verified frames, and builds the chapter player.

Blender is staged on node-local disk by
[`stage_blender.py`](../owb/scripts/stage_blender.py) from a checksummed copy
of the official archive in `.cache/render-tools` (downloaded once), and the
large per-frame scene snapshots stay on node-local disk. Render records include
the Blender version, device, denoiser, and source hashes, including the
presentation-landscape code.

The default frame cache is `.cache/wildlands-cycles`; `OWB_FRAME_CACHE` may point
to a persistent scratch directory. There is no automatic GPU-to-CPU fallback.

```bash
# Collect new runs into a new directory, then independently replay them.
python -m owb.scripts.wildlands_reference --collect /path/to/new-runs --task a b

# Verify an existing supplied run without Blender.
python -m owb.scripts.wildlands_reference --verify assets/demos/wildlands-reference/task-a-seed-7
python -m owb.scripts.wildlands_reference --verify assets/demos/wildlands-reference/task-b-seed-7

# Render the complete shorter course, or split it into nonoverlapping ranges.
python -m owb.scripts.wildlands_reference \
  --render-chunk assets/demos/wildlands-reference/task-a-seed-7 \
  --cache /path/to/frame-cache --milestones 1 12 --checkpoints --samples 48 --threads 4

# Assemble only after every milestone has a completed render index.
python -m owb.scripts.wildlands_reference \
  --assemble assets/demos/wildlands-reference/task-a-seed-7 --cache /path/to/frame-cache
```

For a denser recording, omit `--checkpoints` and select a sampling `--stride`
(use 1 to capture every primitive action). Existing videos are never overwritten.
Completed frames can resume only when trace hashes, source files, and render
settings match; saved frames are rechecked against the replayed state.

Blender is resolved through `OWB_BLENDER`, `PATH`, or `.owb-render.local.json`.
The H.264 encoder is resolved through `IMAGEIO_FFMPEG_EXE`, `PATH`, or the local
configuration. Tests and replay verification run headlessly and do not need
Cycles. Relevant automated checks cover remote-control rejection, supported
mass and brake ordering, flow isolation/recovery, atomic part installation,
hot-plume rejection, final live-system checks, and presentation isolation.
