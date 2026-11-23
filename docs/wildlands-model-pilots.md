# Wildlands model pilots

The compact 12/24-milestone courses have a separate observation-only model runner.
This evaluates model actions, not the scripted reference controller.

```bash
python -u -B -m owb.scripts.wildlands_model \
  --task a b --seed 7 --transport codex \
  --account-home ~/.codex-roboharness04 \
  --codex-executable .cache/codex-sol/node_modules/.bin/codex \
  --model gpt-6.1-sol --effort medium \
  --max-calls 200 --max-actions 6000 --wall-seconds 3600 \
  --memory-chars 6000 --output runs/sol-wildlands-pilot
```

The initial run is stored in
[`runs/sol-wildlands-20261105-pilot-01`](../runs/sol-wildlands-20261105-pilot-01).
It requests the exact [GPT-6.1 Sol model](https://developers.openai.com/api/docs/models/gpt-6.1-sol)
through the saved Codex login, with no model fallback. The API preflight returned
`credit_balance_exhausted`; the Codex CLI 0.166.0 live access preflight passed.
These are Codex-runtime results, distinct from a direct Responses API baseline.

The [active-run manifest](../runs/sol-wildlands-20261105-pilot-01/active-runs.json)
points to the latest continuation for each task. Use those task folders when
reading final scores; the original attempt remains available for auditing.

Each task gets independent, initially empty model-authored memory and six recent
text/sensor/action records. The model receives the current native 320 × 180 RGB
observation, named proprioception, inventory, survival sensors, visited cells,
local control IDs/symbols and inspection text. It receives no reference actions,
fixture coordinates, evaluator events, hidden milestone predicates or simulator
handle. CLI tools and repository access are disabled.

The output schema includes the course's interaction `target` and magnetic
`vector` fields. The model may repeat `none` or `drag` for up to 30 native
actions; other commands execute once. Every primitive action advances 0.2
simulated seconds. Repetition stops at milestone completion or episode end.
This is action repetition, not an oracle navigation or puzzle-solving macro.
Model decisions and primitive actions are reported separately. World time does
not advance during inference.

`--stop-after-milestones N` ends an episode with `milestone_target_reached`
after N milestones; the scope recorded in `config.json` says so. Runs from
2026-11-21 onward retry up to three consecutive transient Codex failures (each
counts against the decision budget, with no state or memory reset), matching the
Teyvat pilots. The same runner drives the Skylands campaign; see
[Skylands model pilots](skylands-model-pilots.md).

From `runs/sol-wildlands-20261121-first3-r2` onward, the observation also reports
`holding_body`, the character's own magnetic-grip state. Earlier runs gave no
indication of a held body, and `grip` fails while one is held.

Runs collected after 2026-11-22 also get interaction feedback and visible fixtures.
Earlier observation-only runs had neither:

- **Feedback line.** After `inspect`, `operate`, `grip`, `drag`, `drop`,
  `install` or a targeted `interact`, the text reports what visibly happened, in
  player terms. Examples: `Last interaction: operate 3:leaf -> the brake trips;
  the support plate reads 0 kg`, `-> the brake locks`, `-> nothing happens`, and
  `-> out of reach or blocked from here`. The line stays until the next
  interaction. Lift trips and gate openings during later steps are appended to
  it. Evaluator reason codes are paraphrased; hidden constants are not revealed.
  The plate reading is the same figure the lift plaque shows.

  Earlier runs got no outcome at all. In `-first3-r2` Wildlands A, the brake
  worked 19 times, release tripped 12 times on a 0 kg load, and `operate` on the
  install socket or the plaque did nothing 17 times. To the model all of these looked identical, so it
  concluded it was out of range.
- **Visible fixtures in the policy frame.** Controls are posts painted with their
  symbol colour; plaques are shorter, broader posts. Orange stakes mark the
  corners of each counterweight support plate. The slab (and crates) render as
  boxes instead of spheres. Before this, controls had no visual form, the plate
  was indistinguishable from the grass, and the slab looked like a boulder.

These are policy-observation changes only. Dynamics, events and digests are
unchanged: the `-first3-r2` traces still replay-verify on the new code. Runs
from now on also hash the policy-view renderer sources.

`--privileged` runs a labelled privileged-state upper bound (true roles,
positions, mechanism state, rules and step feedback; no actions). See
[Skylands model pilots](skylands-model-pilots.md#privileged-state-condition-upper-bound).

`--recovery zelda` (the default since 2026-11-22) enables void-out and three
continues ([rules](wildlands.md)). It adds the rule to the control guide and
`observation.recovery` (deaths, void-outs, continues left), and reports the same
counts in `status.json` and `summary.json`. `--recovery permadeath` keeps the
earlier rule and states it in the guide. Runs without a recorded `recovery`
used permadeath, and replay verification applies each run's recorded rule.

The episodes run concurrently, each bounded by 200 model decisions, 6,000
primitive actions and one wall-clock hour by default. A request already underway
can extend the wall-clock bound by at most its timeout (90 seconds). The Codex
transport cannot enforce a per-response output-token cap. Independent replay
verification runs after each episode, outside its model decision budget.

Each `task-*-seed-7` folder contains:

- live `status.json`, initial `config.json`, and the exact `control-guide.txt`;
- policy images in `observations/`, exact model inputs in `requests/`, and model
  outputs, response IDs, usage and latency in `responses/`;
- every primitive action, position, reward and evaluator event in `actions.jsonl`;
- terminal `summary.json` and independent action-replay `verification.json`.

`results.json` is written once both episodes finish. Infrastructure failures
before any action set `score_available=false`; they are not zero-milestone model
results. Model memory is fallible text and never counts as completion evidence.
Completion comes exclusively from the simulator's milestone predicates.

Task A's first attempt stopped on decision 3 after a Codex transport error, with
one completed milestone. Its continuation is in
[`runs/sol-wildlands-20261105-resume-a-01`](../runs/sol-wildlands-20261105-resume-a-01).
The `--resume PRIOR_TASK_FOLDER --task a --output NEW_FOLDER` option verifies and
replays the prior actions, restores only the model's recorded memory/history,
and preserves all original budgets, including the original wall-clock start.
Only transport-interrupted episodes may resume; a scored model failure or
budget exhaustion cannot silently receive another attempt. The original files
remain unchanged, and the continuation records its parent summary hash.

These are single-seed pilots of the compact courses, not a completed evaluation
of the full [long-horizon task designs](wildlands-long-horizon-tasks.md). Native
policy views have limited mechanism visibility: local interaction IDs appear in
text while some presentation-only control props are absent from native RGB.
Failure diagnostics in evaluator events are not passed to the model. These
limitations can confound navigation and manipulation performance. The pilot
alone establishes neither a memory/no-memory advantage nor held-out skill transfer.
