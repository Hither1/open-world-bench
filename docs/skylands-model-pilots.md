# Skylands model pilots

Observation-only model runs on the full-scale Skylands campaign courses
(`SkylandsCampaignEnv`). The runner reuses the
[Wildlands model protocol](wildlands-model-pilots.md); only the action schema,
the control guide and one allowlisted observation field differ.

```bash
python -u -B -m owb.scripts.skylands_model \
  --task a b --seed 7 --transport codex \
  --account-home ~/.codex-roboharness04 \
  --codex-executable .cache/codex-sol/node_modules/.bin/codex \
  --model gpt-6.1-sol --effort medium \
  --max-calls 200 --max-actions 6000 --wall-seconds 3600 --timeout 90 \
  --memory-chars 6000 --stop-after-milestones 3 --output runs/sol-skylands-pilot
```

Each campaign world uses about 10 GB and is generated before the wall clock
starts. Run it in a Slurm job; the login-node memory cgroup kills it.

## Interface

- **Actions:** `move`, `look`, `buttons`, `command` (the campaign's full command
  tuple, including riding commands), `target` (0–511), `option` (0–3), `part`
  (0–8), `build_move` (3 values, which also steer a driven craft), `slot`,
  `ingredients`, `repeat`, and `memory`. Only `none` may repeat (1–30 times).
- **Observation:** the current native 320 × 180 RGB frame, proprioception,
  survival sensors, inventory, the visited-cell map, build state, visible
  construction parts and the environment's text (objective, latest inspection
  text, satchel). It also includes `local_targets`: target IDs within 4.5 m and
  within 6 m that are in line of sight, as `id:role`, plus whether the character is
  holding a part or towing cargo. Parts already mounted or held are left out of this list.
  The first run (`runs/sol-skylands-20261121-first3`) still listed mounted
  parts at their original yard positions, which led the model to retry picking
  them. Runs from `-first3-r2` onward use the corrected list.
- From 2026-11-22 the text also carries a feedback line for the latest
  interaction, for example `Last interaction: mount 12 -> part mounted` or
  `-> the fan jams`. Refusals are grouped into perceivable categories: out of
  reach, missing item, does not fit, nothing happens, or "it won't work yet".
  This is because many evaluator reasons name the missing procedure step. Fan
  strikes report how the fan sounds, not the charge. Earlier runs had no
  interaction feedback at all.
- Roles are coarse physical categories (control, part, raft, cargo…). Target
  owners, part kinds, positions, cache bindings, milestone state and evaluator
  events are not supplied. Part kinds, ratings and craft charge come only from
  `inspect`.
- The [control guide](../owb/bench/baselines/skylands_model.py) describes
  command mechanics, not task procedures. The fan-striking rhythm, mount slot
  assignments and routes have to be discovered.

As in the Wildlands runner, up to three consecutive transient Codex failures are
retried. Each failed call counts against the decision budget, and state and
memory are not reset. With `--stop-after-milestones N`, an episode ends with
`milestone_target_reached` once N milestones are complete. Every episode is
replayed independently from reset; the replay must reproduce every position,
event and reward, plus the milestone and final digests.

These are single-seed pilots, not a memory ablation or a claim of held-out
skill transfer.

[Checkpoint starts and the perception-oracle condition](checkpoint-starts.md) add segment-level evaluation from verified reference states (`--start-milestone`, `--checkpoint-memory`) and an intermediate `--condition perception_oracle`, so scores are not floored by the first milestones.

## Privileged-state condition (upper bound)

`--privileged` (in both `owb.scripts.wildlands_model` and
`owb.scripts.skylands_model`) adds `observation.privileged`, built by
[`privileged_view.py`](../owb/bench/baselines/privileged_view.py). For the
current objective it reveals the location, clue, true roles and world positions
of controls, bodies, parts, craft and cargo, plus the mechanism state. That
includes the Wildlands support zone and load, and the Skylands craft slots,
charge, route and destination. It also reveals a short rule for the milestone
kind, including hidden constants such as the fan-strike window, and the events
and failure reasons from the previous decision. It never supplies reference
actions or plans: the model still chooses every primitive action.

Each run's `config.json` records `information_condition: privileged_state` and
`observation_only: false`. Report these runs separately. They measure control
and execution given full task knowledge, not discovery, perception or memory.
