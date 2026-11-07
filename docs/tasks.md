# Task specifications

Status: step 5, first half, of the [architecture proposal](architecture-proposal.md) §2.3.
Tasks are declared as data (`owb/tasks/schema.py`), scored by an evaluator that only
reads events and a read-only state view (`owb/tasks/evaluator.py`), and attached to a
common-schema environment by a wrapper (`owb/tasks/env.py`). Setup hooks and mechanism
modules are not part of this step (see [Not covered yet](#not-covered-yet)).

| Module | Contents |
| --- | --- |
| `owb.tasks.schema` | `Predicate`, `Budget`, `Milestone`, `TaskSpec`; `load(path)` / `dump(spec, path)` for `.yaml` and `.json` |
| `owb.tasks.evaluator` | `StateView` protocol, `State` (plain implementation), `MilestoneEvaluator` |
| `owb.tasks.registry` | `available_tasks(sim=None)`, `load_task(id)` over `owb/tasks/specs/**` |
| `owb.tasks.env` | `TaskEnv(benchmark_env, spec)`, `TeyvatStateView`, `NullStateView` |
| `owb.tasks.minecraft` | `MinecraftTask`, `MilestoneTracker`: the old API, now loaded from `specs/minecraft/*.yaml` |
| `owb.tasks.teyvat` | `questgraph_to_spec(graph, ...)`, `builtin_questline_spec(seed)` |

## File format

One task per file, YAML or JSON with the same structure. The shipped specs live in
`owb/tasks/specs/<sim>/<id>.yaml` and are packaged with the wheel. A complete example,
the first three milestones of the Minecraft *Mine Diamond from Scratch* task plus a
made-up ordered milestone to show the remaining fields:

```yaml
id: mine_diamond_from_scratch      # unique across the registry
version: 1                         # bump when scoring changes
sim: minecraft                     # TaskEnv refuses to run it on another sim
title: Mine Diamond from Scratch
instruction: "Starting with nothing, obtain a diamond by gathering wood, crafting tools, mining ores, and smelting iron."
budget:                            # all optional; omitted = not declared
  steps: 12000
  # sim_seconds: 3600
  # model_calls: 2000
  # tokens: 5000000
setup: null                        # registered set-up hook name (not implemented yet)
setup_params:                      # free-form, recorded with the run
  world_seed: 0
milestones:
- id: mine_log
  title: Mine log
  stage: basic survival            # display grouping only
  reward: 1.0                      # optional, default 0
  done: {kind: stat_increase, stat: mine_block, objects: [log, oak_log, birch_log, spruce_log]}
- id: crafting_table
  title: Craft crafting table
  stage: basic survival
  reward: 1.0
  done: {kind: stat_increase, stat: craft_item, objects: [crafting_table]}
- id: wooden_pickaxe
  title: Craft wooden pickaxe
  stage: stone age
  reward: 1.0
  done: {kind: stat_increase, stat: craft_item, objects: [wooden_pickaxe]}
- id: show_off                     # illustration, not part of the shipped task
  title: Show the pickaxe at the village at dusk
  stage: stone age
  requires: [wooden_pickaxe]       # milestone DAG; must be acyclic
  role: transfer                   # free annotation, except "optional" and "gate"
  hidden_from_agent: true          # default; TaskEnv never shows milestones anyway
  done:
    kind: all
    preds:
    - {kind: reach, position: [120.0, 64.0, -40.0], radius: 8}
    - {kind: altitude, min_height: 60, time_window: [0.7, 0.85]}
```

Predicates are written flat (`{kind: ..., <args>}`); the nested form
`{kind: ..., args: {...}}` is accepted too. Unknown kinds or arguments are errors, and so
are duplicate milestone ids, `requires` / `expires_after` that name unknown ids, and
`requires` cycles. `dump` writes the flat form and omits arguments left at their default.

### Milestone fields

| Field | Meaning |
| --- | --- |
| `id`, `title`, `stage` | Identity, display text, display grouping. |
| `done` | The completion predicate. |
| `requires` | Ids that must have fired on an **earlier** update before this one is eligible. |
| `expires_after` | Optional. Once all of these fired on an earlier update, this milestone can no longer fire. Needed for Teyvat optional objectives (below). |
| `role` | Free text (`learning`, `interference`, `transfer`, `composition`, ... from the long-horizon designs) with two reserved values: `optional` (counts in progress, not needed for success) and `gate` (a bookkeeping milestone: needed for success, not counted in progress). |
| `hidden_from_agent` | Default `true`. Recorded for leakage audits; nothing in this step shows milestones to the agent. |
| `reward` | Sparse reward annotation. `TaskEnv` does not change the env reward (`RewardShaper` is future work). |

### Predicate kinds

Events are read duck-typed: either an object with `.type` and `.get(key)` (the Teyvat
`Event`) or a dict with a `"type"` key (`BenchmarkEnv` `info["events"]`). State comes from a
`StateView` with `position` (x, y, z, y up), `time_of_day` in [0, 1), `inventory`
(item -> count) and `stats` (statistic -> object -> cumulative value); any field may be
`None`, and a predicate that needs a missing field is false.

| Kind | Arguments (defaults) | Holds when |
| --- | --- | --- |
| `stat_increase` | `stat`, `objects` | The cumulative `stats[stat][obj]` increased since the previous update. The deltas of `objects` are summed in order and the predicate holds as soon as the running sum is positive (MineStudio `RewardsCallback`). Values present at reset count for nothing. |
| `event` | `type`, `targets` (any), `item` (any), `count` (1), `distinct` (false) | `count` events of `type` (with `entity` in `targets`, `item` equal) have occurred since the milestone became eligible; with `distinct`, `count` different entities. |
| `sequence` | `targets`, `type` (`activate`) | Events of `type` on `targets` arrived in the listed order; an event on a target out of order resets the progress. |
| `inventory_has` | `item`, `count` (1) | `inventory[item] >= count` now. |
| `deliver` | `item`, `targets`, `count` (1), `type` (`talk`) | A `type` event on one of `targets` happens this step while the inventory before the hand-over (previous state plus this step's `collect` events) holds `count`. The sim does the deduction. |
| `reach` | `position`, `radius` (6) | Horizontal (x-z) distance to `position` is within `radius`. |
| `altitude` | `min_height`, `position` (none), `radius` (6), `time_window` (none) | `y >= min_height`, optionally within `radius` of `position` and inside the time-of-day window (which may wrap past midnight, e.g. `[0.9, 0.1]`). |
| `dive` | `position`, `radius` (6), `margin` (1) | Within `radius` of `position` horizontally and `y <= position.y + margin`. |
| `all` / `any` | `preds` | Every / some child holds. All children are evaluated each update, so stateful children keep counting; an empty `all` is always true. |

`event` and `sequence` are stateful and latch once satisfied; the state-based kinds are
evaluated on the current step only, so `all([reach, altitude])` needs both at once.

### Evaluation semantics

`MilestoneEvaluator(spec)`: `reset(state)`, then `update(events, state, step)` after every
step, which returns the ids that fired on that update in spec order.

- A milestone is **eligible** on an update when everything in `requires` fired on an
  earlier update and its `expires_after` set (if any) has not all fired on an earlier
  update. A milestone that fires never unlocks a dependent within the same update, so the
  result does not depend on the order milestones are listed in.
- Only eligible milestones are evaluated, and `event` / `sequence` progress starts when the
  milestone first becomes eligible. `stat_increase` always compares with the previous
  update, so an increase while a milestone was not yet eligible is not credited later.
- Each milestone fires at most once; `achieved` maps id -> step.
- `progress` = fired / total over milestones whose role is not `gate`. `success` = every
  milestone whose role is not `optional` fired.
- `summary()` gives counts, progress, success and `achieved`; `snapshot()` is the full
  JSON-serialisable evaluator state (counters, sequences, previous stats and inventory).

## TaskEnv

```python
import owb
from owb.tasks.env import TaskEnv
from owb.tasks.teyvat import builtin_questline_spec

env = TaskEnv(owb.make_env("teyvat", seed=3), builtin_questline_spec(3))
obs, info = env.reset(seed=3)
obs, reward, terminated, truncated, info = env.step(action)
info["task"]   # {"id", "progress", "success", "fired": [...], "achieved": {id: step}}
```

On reset the wrapper builds a fresh evaluator over the sim's `StateView` (Teyvat:
`TeyvatStateView`, which reads the character position, the world's time of day and the quest
engine's inventory, the only inventory Teyvat has; it has no statistics). Wildlands and
Skylands get `NullStateView` (all `None`) for now. On every step it feeds `info["events"]`
and the view. If `spec.instruction` is non-empty it replaces `observation["instruction"]`;
otherwise the native instruction is kept. Nothing else in the observation, reward,
termination, or the native `info` changes (tests/test_task_env.py compares against the bare
env). The evaluator is private to the wrapper; `task_summary()` / `task_snapshot()` are for
the harness and recorder, not the agent.

## Minecraft

`owb/tasks/specs/minecraft/{mine_diamond_from_scratch,ender_dragon}.yaml` hold the two
MCU-AgentBeats tasks with their milestone order and alias object lists unchanged (checked
against a digest of the pre-port Python definitions, and against `../MFM` configs when a
checkout is present). Every milestone is a `stat_increase` on `mine_block`, `craft_item`,
`pickup` or `kill_entity` with no `requires`, `budget.steps` is the one-life budget and
`setup_params.world_seed` the MCU seed. `MINECRAFT_TASKS`, `MinecraftTask` (now with
`from_spec` / `to_spec`), `minecraft_task` and `MilestoneTracker` keep their public
behaviour; `MilestoneTracker` is a thin wrapper that feeds MineStudio's info dicts to a
`MilestoneEvaluator` as `stats`.

## Teyvat questline

`questgraph_to_spec(graph, id=..., instruction=...)` turns a `QuestGraph` into a spec with
one milestone per objective (same id, description as title, `stage` = `<quest>:<index>`,
objective reward):

| ObjKind | Predicate |
| --- | --- |
| `TALK` | `event(type=talk, targets)` |
| `COLLECT` | `inventory_has(item, count)` |
| `ACTIVATE` / `OPEN_CHEST` / `DEFEAT` | `event(type, targets, count, distinct=true)` |
| `SEQUENCE` | `sequence(type=activate, targets)` |
| `REACH` | `reach(position, radius)` |
| `ALTITUDE` | `altitude(min_height, position, radius, time_window)` |
| `DIVE` | `dive(position, radius)` |
| `DELIVER` | `deliver(item, count, targets)` |

Stage order and quest prerequisites become `requires`: a stage-k objective requires the
non-optional objectives of stage k-1; a first-stage objective requires the last stage of
every prerequisite quest. Optional objectives get `role: optional` and `expires_after` their
stage's required objectives, since `QuestEngine` stops checking a stage once it advances. A
stage made only of optional objectives (the Sunken Ledger's guards) advances on the first
update it is checked; it becomes a `gate` milestone (`all: []`) that later stages require.
Quest rewards and item `grants` are not milestones: grants are simulator state and reach the
evaluator through the inventory.

`builtin_questline_spec(seed)` builds the common small `teyvat` env at `seed` and maps its
graph (`id: teyvat.windborne_ledger`). The spec's positions are only valid for that seed and
variant. It is generated, not a file, so `available_tasks()` does not list it.

tests/test_task_teyvat.py feeds the same event and state stream to `QuestEngine.consume` and
`MilestoneEvaluator.update`, both random streams over a hand-made graph that uses every
ObjKind and scripted runs of the built-in questline to completion, and asserts that the same
objectives complete at the same steps with the same completion fraction and success. Known
divergence, not exercised by the built-in questline: `QuestEngine` updates its inventory
while it walks the quests, so a grant or delivery earlier in the same step is visible to a
later quest's `COLLECT` check. The evaluator reads the post-step inventory instead.

## Not covered yet

- **Setup / placement hooks.** `setup` and `setup_params` are recorded but nothing runs
  them. Tasks run in the world the native env builds, which is why the Teyvat spec must be
  generated per seed.
- **Mechanism modules.** The reference courses' relay, sentinel, vent, counterweight and
  flow-network state machines (`owb/sims/teyvat/reference_tasks.py`,
  `owb/sims/voxel/env/wildlands_reference.py`) still live in env subclasses. Their milestones need
  mechanism events or a `state_equals` predicate over mechanism state, neither of which
  exists yet.
- **Wildlands and Skylands.** Wildlands has no victory predicate and the evaluator gets no
  state from it (`NullStateView`); Skylands keeps its native `success` / `task_progress`.
- **Parts of the 12/24-milestone designs** (docs/teyvat-long-horizon-tasks.md,
  docs/wildlands-long-horizon-tasks.md) the schema cannot express yet:
  - conditions held for a duration ("hold valid flow for five simulated seconds") and
    final live-world predicates that must still hold at the end of the episode;
  - crediting legitimate early work: a predicate is only evaluated once its milestone is
    eligible, while the Wildlands design asks to keep evidence from early exploration;
  - fresh-context checkpoints and memory schedules (before A08/A12, B11/B21/B24), which
    belong to the agent harness, not to milestones;
  - attempt boundaries, first-attempt vs assisted success, assistance and help logging,
    unnecessary-action costs, and the other per-probe measures in the scores tables;
  - item transactions (consume on atomic installation) beyond `deliver`'s talk-and-hold
    check, and seeded rule/layout assignments (they belong to setup);
  - budgets as ranges (the Wildlands pilot budgets); `Budget` holds one value per resource.
  The milestone DAGs themselves (chains, the B09-B11 / B13-B18 branches and their joins),
  memory-role tags and budgets are expressible with `requires`, `role` and `budget`.
