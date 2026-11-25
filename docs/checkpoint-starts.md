# Checkpoint starts and the perception-oracle condition

Observation-only model pilots score at the floor. Through 2026-11-23, GPT-6.1 Sol
completed 0 of 12 or 24 Teyvat milestones, 1–3 Wildlands milestones and one
Skylands milestone. The model fails before most of the task has been reached, so
these scores say almost nothing about the later milestones, which are the ones
that depend on memory. Two tools in the shared Wildlands/Skylands runner fix this
without changing any task, terrain or mechanic:

1. **Checkpoint starts** begin an episode after K milestones, from a verified
   reference state. Every segment of a task can be measured, even when a model
   cannot get through the earlier ones.
2. **The perception-oracle condition** sits between native observations and the
   privileged upper bound. It separates "cannot see or locate things" from
   "cannot work out what to do or remember it".

Both are labelled in each episode's `config.json` (`checkpoint`,
`information_condition`). Report them separately from full observation-only
episodes.

## Information conditions

| Condition | `--condition` | The model additionally receives |
| --- | --- | --- |
| Native | `observation_only` | Nothing: native RGB, sensors, inventory, local control IDs, interaction feedback |
| Perception oracle | `perception_oracle` | `observation.perceived`: what a perfect detector reports for the current frame |
| Perfect recall | `perfect_recall` | Perception oracle plus `observation.recalled`: every fixture shown so far (last-seen position) and every text read |
| Goal location | `goal_location` | Perfect recall plus `observation.objective_location` (position only) |
| Privileged state | `privileged_state` (alias `--privileged`) | Current-objective location, true roles, rules, hidden constants, failure reasons |

`perfect_recall` and `goal_location` are the upper rungs of the
[memory ladder](memory-ladder.md), which separates remembering from using memory
to decide where to go. Every condition now also writes evaluator-only search
metrics (`search.json`).

`perceived` lists entities whose anchor point is inside the policy camera's
frustum (the same third-person camera, 75° horizontal FOV), at most 48 m away,
with clear voxel line of sight. It returns at most 24 entities, nearest first.
Each entry has the interaction ID (if any), a **visual** category, world
position, distance, and bearing from the character's heading (positive = right):

- **Wildlands:** `post` or `plaque` with its painted colour, `movable block` with its size,
  `orange-staked ground plate` with its corners, and `creature` with its species.
  These are exactly the props the policy renderer draws
  (`WildlandsReferenceEnv._extra_entities`).
- **Skylands:** each target's coarse category (`control`, `part`, `raft`, `cargo`, `cache`…),
  the same vocabulary as `local_targets`, plus creatures. Part kinds,
  ratings and charge are still only available through `inspect`.

It never includes what a control does (brake/release/inlet…), which objective an
entity belongs to, where the objective is, mechanism state, rules, or why an
interaction failed. Entities leave the list when they leave the frame, so the
agent still has to remember where things are.

A perfect detector at 48 m exceeds what the 320 × 180 frame resolves: distant
posts are only a pixel or two wide. That is intended. The condition is an
oracle, not a model of a real detector. The Skylands policy frame does not yet
draw impostors for parts, rafts or controls, the gap that Wildlands closed on
2026-11-22. In Skylands, then, this condition also adds objects that the native
image does not show.

## Checkpoint banks

A bank is a privileged reference trace collected **on the current simulator
code**, together with `checkpoints.json`, an index of the replay prefix after
each milestone:

```bash
python -B -m owb.scripts.checkpoint_bank collect --world wildlands --task a --seed 7
sbatch --export=ALL,TASK=a owb/scripts/checkpoint_bank.slurm   # Skylands campaign, ~10 GB
```

Banks live in `assets/evaluation/checkpoint-bank/WORLD/task-T-seed-S/`. Each
`checkpoints.json` entry records the number of reference actions to replay, the
milestone digest, the simulated time, and `segment_actions`, the number of
actions the reference needed for the next milestone. Treat `segment_actions` as
a yardstick for difficulty and budget, not an expected model length.

The committed demo traces in `assets/demos/` no longer replay on the current
code. Creature spawning changed after they were recorded: the agent's positions
still match, but NPC event streams diverge at action 1–3. They remain valid
records of the code that produced the videos, but cannot seed checkpoints. A
bank refuses to restore if its initial digest differs from the episode's
(`Bank was collected from a different initial state`). Re-collect the bank
whenever environment code changes the digest.

| Bank | Milestones | Reference actions | Reference actions per segment |
| --- | ---: | ---: | --- |
| Wildlands A, seed 7 | 12 | 999 | 1, 102, 76, 125, 67, 72, 41, 43, 166, 64, 27, 215 |
| Wildlands B, seed 7 | 24 | 2,535 | 1, then 25–250 (median 100) |

## Running checkpoint starts

```bash
python -u -B -m owb.scripts.wildlands_model --task a b \
  --start-milestone 0 3 6 9 --stop-after-milestones 3 \
  --condition perception_oracle --checkpoint-memory empty \
  --transport codex --account-home ~/.codex-roboharness04 \
  --codex-executable .cache/codex-sol/node_modules/.bin/codex \
  --model gpt-6.1-sol --effort medium --output runs/NAME
```

The same flags work for `owb.scripts.skylands_model`.
`owb/scripts/model_sweep.slurm` runs a sweep in Slurm and then writes the report.

- `--start-milestone K [K ...]` runs one episode per K, concurrently, in
  `task-T-seed-S-from-KK[-clues]/`. Before any model call, the runner replays
  the bank prefix through the world's `checked_step`, which checks every
  position, event and reward plus each milestone digest.
- **Budgets count only the model's own actions.** `--max-actions`,
  `--max-calls`, `--stop-after-milestones` and the reported `primitive_actions`
  all start at the checkpoint. `actions.jsonl` holds only model actions.
  Verification replays the bank prefix and then those actions.
- The model never sees the prefix, its actions, explanations or digests. It
  inherits only the world, inventory and satchel. A line added to the control
  guide says it is taking over partway through.
- `--checkpoint-memory` sets the model's memory at the start:
  - `empty`: nothing.
  - `clues`: the verbatim clue texts the reference read during the prefix
    (`clue_exposed` events). These are texts a player sees, not evaluator
    state. This is an "ideal recall" condition.

  The gap between `empty` and `clues` at the same checkpoint is a direct
  measurement of how much the next segment depends on earlier information.
  This is the first piece of the memory ablations proposed in the task designs.

## Reading results

```bash
python -B -m owb.scripts.checkpoint_report runs/NAME [runs/OTHER ...]
```

This writes `checkpoint-report.md` and `.json`. Per episode it reports:

- `gained`: milestones the model itself completed after its start.
- whether it completed the next milestone.
- model and reference actions.

Per (world, task, model, condition, memory) group it reports the
next-milestone rate and mean milestones gained. Episodes that did not verify on
replay, or that stopped before acting, are listed but not scored.

Interpretation notes:

- Milestone 1 in both Wildlands courses is a one-action inspect, so start 0
  inflates the next-milestone rate. Compare starts after it.
- A checkpoint-start result describes one segment, entered at the reference's
  position and route. It is not a full-episode success rate, and its
  difficulty depends on where the reference left the character.
- Each bank is a single seed and covers a single reference route. Held-out
  seeds need their own banks.
- Teyvat has a separate runner (`owb.scripts.teyvat_model`) and is not wired
  in yet. The lemon tart has no model runner.
