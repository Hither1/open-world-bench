# Teyvat model pilots

The compact Teyvat courses have an observation-only model runner, separate from
the scripted reference demos. Task A has 12 milestones; Task B has 24.

```bash
python -u -B -m owb.scripts.teyvat_model \
  --task a b --seed 7 --transport codex \
  --account-home ~/.codex-roboharness04 \
  --codex-executable .cache/codex-sol/node_modules/.bin/codex \
  --model gpt-6.1-sol --effort medium \
  --max-calls 200 --max-actions 6000 --wall-seconds 3600 \
  --memory-chars 6000 --output runs/sol-teyvat-20261105-pilot-01
```

Each episode has independent, initially empty model-authored memory (6,000
characters) and six recent text/sensor/action records. Inputs are the current
native 384 × 216 policy image, proprioception, party vector, quest vector,
quest instruction and local feedback. There is no simulator handle, evaluator
state, inventory from evaluator info, privileged target, rule table, reference
action, repository access or external tool access in the model prompt.

The model selects camera-relative movement, camera rotation, native buttons and
party swaps. It can repeat movement, waiting, attacks or held jump for up to 30
primitive actions. Interact, skill, burst and swaps execute once. Changed local
feedback, milestone completion and episode termination interrupt repetition.
Each primitive action advances 0.2 simulated seconds. World time pauses during
inference. No navigation or puzzle-solving macro generates model actions.

The two episodes run concurrently, each bounded by 200 model calls, 6,000 native
actions and one hour. A pending call may extend that hour by its 90-second
timeout. Transient Codex failures retry up to three consecutive failed calls,
with each failed call consuming the original budget and no state or memory
reset. Model unavailability or quota errors stop the episode. Codex transport
cannot enforce an output-token cap. The exact model ID is requested without a
fallback; these are Codex-runtime results, not direct API results.

The pilot lives in [`runs/sol-teyvat-20261105-pilot-01`](../runs/sol-teyvat-20261105-pilot-01).
Each task folder contains live `status.json`, `config.json`, `control-guide.txt`,
policy images, requests, responses, every native action and evaluator event,
and terminal `summary.json` plus independent `verification.json`. The evaluator
events are logged for auditing and never passed back into model observations.
Replay checks every action's position, reward, events and state digest, then
checks the final completed milestone list. `results.json` appears when both
runs finish. A transport failure before any action is not a zero-milestone model
score (`score_available=false`).

These are pilots on the compact demonstration courses, not the full long-horizon
task designs. Native observations do not expose local entity labels or valid
quest target coordinates. Several diagnostic cues are available only in private
evaluator state. Navigation and mechanism visibility can therefore limit scores
before long-horizon memory is meaningfully tested. No result from this pilot
alone establishes a memory advantage or reusable skill transfer; that requires
paired memory ablations and held-out tasks. Policy images are inexpensive native
observations, separate from Blender Cycles user-facing presentations. This runner
does not modify the task terrain, collision geometry or simulation mechanics.
