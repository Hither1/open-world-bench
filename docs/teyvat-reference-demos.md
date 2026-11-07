# Teyvat reference demonstration videos

These recordings show successful, executable **compact reference courses** for
the 12- and 24-milestone task designs. Frames come from the native Teyvat
simulator. A privileged reference controller supplies ordinary movement,
interaction, party-swap, skill, and attack actions. The videos are not generated
storyboards and do not claim that a learned agent has solved the tasks.

The complete walkthroughs linked below were regenerated with **Blender 4.5.4
Cycles on GPU**, updated procedural assets, and explicit party-swap cues. The
scene renders at its native 864×486 resolution inside a 1280×720 annotated video,
using 48 samples per pixel. Both saved action traces are unchanged. The course
itself remains the compact prototype described below.

| Course | Video | Evidence |
| --- | --- | --- |
| Task A: Relay Keeper's Apprenticeship, 12 milestones | [H.264 MP4](../videos/teyvat-12-milestones.mp4) | [Chapters](../assets/demos/teyvat-reference-cycles/task-a-seed-7/chapters.json) · [Verification](../assets/demos/teyvat-reference-cycles/task-a-seed-7/verification.json) |
| Task B: Expedition of Three Arts, 24 milestones | [H.264 MP4](../videos/teyvat-24-milestones.mp4) | [Chapters](../assets/demos/teyvat-reference-cycles/task-b-seed-7/chapters.json) · [Verification](../assets/demos/teyvat-reference-cycles/task-b-seed-7/verification.json) |

Both use seed 7. The display includes the current milestone, reference action,
party health/stamina, simulator time, and mechanism state. Navigation is sampled
and sped up; explicit event pauses make interactions readable. The full action
logs preserve every decision at the native 0.2-second decision interval.
For embedded viewers, open the H.264 MP4 copies in `videos/`. They use
Constrained Baseline H.264, `yuv420p`, and fast-start metadata. Every frame was
decoded successfully. [Playback verification](../videos/playback-verification.json)
records the encoding options, render settings, toolchain, and source/output
hashes. The new evidence packages are in `assets/demos/teyvat-reference-cycles/`.
The original CPU recordings in `assets/demos/teyvat-reference/` retain their
recorded hashes and original codecs as archival artifacts.

| Recording | Completed | Native actions | Simulated time | Edited video |
| --- | --- | --- | --- | --- |
| Task A, seed 7 | 12 / 12 | 1,118 | 223.6 s | 60.5 s |
| Task B, seed 7 | 24 / 24 | 3,038 | 607.6 s | 166.0 s |

Both traces complete without player deaths. Task B includes 16 vent launches
and 16 confirmed gliding landings. Video duration includes the event pauses.

The visible character changes are party swaps by the same controller: Kestrel
(Ember), Maren (Tide), and Wren (Gale). The runs contain 17 and 44 swaps,
respectively. The previous videos skipped the exact moment of four swaps in each
run and did not explain the transitions. The regenerated MP4s show every swap.

These recordings keep the whole party visible, highlight the active member, and
show every successful swap with a labeled before/after pause (at least 0.4 s
before and 1.2 s after at the default frame rate). Both recording paths use the
same frame schedule. These holds repeat actual replay states; they do not add
simulated actions or an invented animation. The course still uses the
three-character party; a single-character ability system would be a separate
mechanics change requiring fresh reference traces.

The updated overlay can be inspected in these Cycles frames from the first swap:
[before, action 45](../assets/design/teyvat-party-continuity/step-0045.png) and
[after, action 46](../assets/design/teyvat-party-continuity/step-0046.png).
Their [metadata](../assets/design/teyvat-party-continuity/preview.json) records
render settings, source/trace hashes, and the full 1,118-action replay check.

## What is implemented

- Persistent relay stages, seeded element-role assignments, overload, an
  interruption that preserves priming, and single-/double-ring rules.
- Sentinels with shield, exposed, wind-up, and vulnerable phases; interruption,
  one training reformation, and physical attacks against the exposed core.
- Charged vent launches gated by the pulse, native gliding and stamina,
  and two measured landings per crossing.
- Mission items, atomic exchanges, delivery/access conditions, a real submerged
  pickup, ordinary camp combat, and a moving service lift.
- All 12 or 24 milestone accomplishments, recorded through actual world actions.

The controller reads private task state to choose a known solution. It does not
teleport the character, edit inventory, set completion flags, refill cooldowns,
or kill enemies directly. Mechanisms change state inside the simulator in
response to legal actions. Successful recording requires no player deaths.

## What these recordings do not establish

The authored course is a compact plateau with shortened travel. It is not the
complete island-scale layout in the design.
Ferry and causeway repair are represented by item exchanges and access
accomplishments, rather than full animated transport structures. The service
lift and vent flights do move the native character in simulation.

Procedural assets distinguish relay roles and natural materials, but the full
designed appearance changes between training and transfer sites remain incomplete.
Sentinel trials demonstrate the phase procedure without active sentinel damage;
ordinary camp enemies retain damage and pursuit. First-pass training is delivered
as NPC instructions and practice, not a separately animated teacher. Role and
state annotations in the video are explanatory overlays for the viewer.

Designed fresh-context boundaries are recorded as events. They are not applied
to a language model or evaluated as memory experiments in this oracle rollout.
Long-delay calibration, learned-skill evaluation, independent held-out layouts,
full-world access gates, and robust reference planning across many seeds remain
future work. These are ground-truth demonstrations **for this executable
prototype**, not benchmark results for the full proposed task suite.

## Verification and reproduction

Each course directory contains:

- `actions.jsonl`: every native action, before/after positions, simulator time,
  reference explanation, and emitted world events.
- `reference-manifest.json`: seed, controller identity, completion predicates,
  milestone completion times, initial/final hashes, and limitations.
- `summary.json`: successful recording, replay result, event counts, and duration.
- `chapters.json`: video timestamps mapped to milestone IDs and action starts.
- `frames.json`: every rendered state's action step, state hash, party transition,
  and corresponding encoded-frame interval.
- `verification.json`: independent replay checks for every action's position
  and events, all milestone hashes, final state, and source/artifact hashes.
- `preview.png` and `preview.gif`: actual frames from the recorded run.

The recorder first solves the course without rendering. A new environment then
replays that action log, checks milestone and final state hashes, and records
the simulator frames. Video files keep a temporary name until replay and final
frame decoding pass. A separate verifier replays again and compares every logged
position and event, not just the milestone counter.

Verify the supplied recordings without rendering or model API calls:

```bash
python -B -m owb.scripts.teyvat_reference --verify assets/demos/teyvat-reference-cycles/task-a-seed-7
python -B -m owb.scripts.teyvat_reference --verify assets/demos/teyvat-reference-cycles/task-b-seed-7
```

Re-record into a new directory (requires Blender, FFmpeg, Pillow, OpenCV for
decoding checks, and DejaVu fonts or Matplotlib's bundled fonts):

```bash
python -B -m owb.scripts.teyvat_reference \
  --task a b --seed 7 --render-stride 8 --fps 10 \
  --output /tmp/teyvat-reference-new
```

Cycles is the default: 48 samples per pixel, with no automatic CPU fallback.
Set `OWB_BLENDER` or use `--blender /path/to/blender` if needed. New videos use
H.264 MP4 directly. `--renderer cpu` is available only as an explicit diagnostic
choice. See the [rendering guide](rendering.md) for local tool configuration.

For execution environments with process time limits, use
`--render-chunk RECORDING --milestones FIRST LAST --cache CACHE` to render disjoint
intervals from an existing verified action trace. Then use
`--assemble RECORDING --cache CACHE` on that unpublished recording directory.
Each chunk replays from reset and checks state hashes; assembly requires complete,
contiguous milestone coverage and the verified final hash. Use the same frame
stride, frame rate, renderer, and render settings for every chunk. Old CPU cache
entries require an explicit `--renderer cpu` when assembled; they are never
presented as Cycles frames.

Chunk renders can resume from their existing cache. Each frame records its
replay-state and image hashes; the cache also records source hashes and all
render settings. Changed sources, traces, settings, or images are rejected.
`--max-render-frames N` optionally limits new frames in one invocation; repeat
the same command to continue. Assembly requires every chunk to be complete and
checks the saved image hashes again. Set `OWB_CYCLES_DEVICE=CUDA` to use an
allocated GPU with the same scene and sample count.

The script refuses to overwrite an existing course directory. Mechanism and
controller regression checks are in `tests/test_teyvat_reference.py`; the full-trace
replay of both recordings is `tests/test_reference_replay.py`. The full
[task design](teyvat-long-horizon-tasks.md) remains the specification for the
eventual memory benchmark.

## Code layout

| Part | Location |
| --- | --- |
| Env (sim configuration, `CourseWorld`/`CourseCombat` input hooks, `state_digest`) | `owb/sims/teyvat/reference_tasks.py` (`ReferenceTaskEnv`) |
| Course runtime and ledger (`Course`, `verify()`) | `owb/tasks/teyvat/course.py` |
| Mechanisms (relay, sentinel/fight, crossing, stations, dive, lift) | `owb/tasks/teyvat/mechanisms/` |
| Placement (`setup: teyvat.reference_course`) | `owb/tasks/teyvat/setup.py` |
| Milestone specs | `owb/tasks/specs/teyvat/reference_course_{a,b}.yaml` |
| Privileged reference controller | `owb/agents/oracle/teyvat_reference.py` (`ReferencePolicy`) |
| Recording / verification / render CLI | `owb/scripts/teyvat_reference.py` |

`open_world_bench.reference_tasks`, `owb.sims.teyvat.reference_tasks.Course` and
`owb.sims.teyvat.reference_policy` remain importable. See docs/tasks.md, "Reference courses".
