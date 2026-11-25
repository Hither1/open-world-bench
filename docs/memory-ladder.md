# Memory ladder: remembering vs. using memory to explore

Status: implemented 2026-11-24 in the shared Wildlands/Skylands model runner
(`owb.scripts.wildlands_model`, `owb.scripts.skylands_model`). No model results yet.

When a model fails a later milestone, there are two different explanations. It
may not have **remembered** what it saw and read. Or it may remember enough but
fail to **use** that memory to decide where to go: returning to a known place,
following a landmark clue, or searching efficiently. The earlier tools do not
separate these:

- `--checkpoint-memory clues` returns clue *texts* only, not where things were seen.
- `privileged_state` reveals location, roles, rules, mechanism state and failure
  reasons all at once.
- The working-memory probes' use sites are reached by ordinary travel, with no search.

## Conditions

Each rung adds one kind of information to the rung below. Everything else stays
the same: the action schema, budgets, the model's own 6,000-character memory
note, the six-step history, and the visited-cell map.

| Rung | `--condition` | Adds |
| --- | --- | --- |
| 0 | `observation_only` | Native RGB, sensors, text, visited-cell map |
| 1 | `perception_oracle` | `perceived`: perfect detector on the current frame ([checkpoint-starts.md](checkpoint-starts.md)) |
| 2 | `perfect_recall` | `recalled`: every fixture any frame shown to the model has contained, with the position where it was last seen and the decisions it was first and last seen, plus every text read, verbatim and in order |
| 3 | `goal_location` | `objective_location`: the current objective's site position (Skylands: also its layer) |

What each gap measures, all from the same checkpoints:

- **Rung 1 to 2 is the memory gap.** Remembering stops being the agent's job.
- **Rung 2 to 3 is the memory-guided exploration gap.** Nothing has to be
  remembered, and the only thing added is where to go. A large gap here means the
  model has the information but cannot turn it into a route or a search.
- A small rung-2 score that rung 3 does not lift means the bottleneck is after
  arrival: procedure, mechanism operation or control. The search metrics below
  confirm this directly.

`privileged_state` is off the ladder. Report it separately as before.

### What `recalled` contains

- Built from the frames the model was actually shown, one per decision.
  Intermediate frames of a repeated action are not included. Recall never
  includes world state the model did not observe. Movable blocks, rafts and cargo
  keep their *last seen* position. Creatures are excluded.
- Entries use the perception oracle's vocabulary: ID, visual category, colour or
  size. They never include roles, owners, milestones or mechanism state.
  Bearings and distances are dropped, because they refer to a pose the agent has
  since left.
- Texts are everything the text observation showed because something was read:
  clues, inspections, and working-memory cues.
- At a checkpoint start, `--checkpoint-memory clues` also adds everything the
  reference operator's camera showed and read during the prefix, on every
  prefix frame, marked `before takeover`. With `empty`, recall holds only the
  model's own experience. `config.json` records `recall_includes_prefix`.
- There is no size limit. It does not count against `--memory-chars`.
- `perfect_recall` makes the working-memory probes trivial, because the cue text
  is recalled. Do not report WM scores from these conditions.

### What `objective_location` contains

A position only. In Wildlands it is the milestone site `[x, ground, z]`. In
Skylands it is the site position and layer. It does not say which nearby fixture
belongs to the objective, what to do there, or what any clue means. Some
milestones involve more than one place, such as hauling cargo between docks or
driving a craft to a destination. For those, only the site where the milestone
is operated is given.

## Search metrics (every condition, evaluator-only)

`search.json` is written beside `summary.json` in every condition, and a summary
is copied into `summary.json["search"]`. The model never sees it. Each milestone
segment the model plays is split at **arrival**: the first primitive action at
which any of the objective's own interaction targets is within interaction reach
(3.2 m in Wildlands, 4.5 m in Skylands, with line of sight). An objective with no
targets uses 6 m from its site instead.

| Field | Meaning |
| --- | --- |
| `kind` | `known` (an objective target was in a frame the model saw before the segment began), `seen_by_earlier_operator` (only in the checkpoint prefix), `unseen` |
| `search_actions` / `execution_actions` | Primitive actions before and after arrival |
| `first_in_frame_decision` | First decision whose frame contained an objective target |
| `start_distance_m`, `path_to_reach_m` | Straight-line start distance vs. horizontal path walked before arrival; `reach_efficiency` in the summary is their ratio (1 = direct) |
| `revisit_fraction` | Share of 4 m × 8 m grid-cell *entries* that re-entered a cell the model had already visited in this episode |
| `first_targeted` | First targeted command in the segment and whether it hit an objective target |
| `wrong_targeted_before_reach` | Targeted commands on other fixtures before arrival |
| `reference_path_m` | The bank reference's path for the same segment, when a bank for this initial state exists; `path_ratio_to_reference` compares completed segments |

Read them by `kind`. For example, long search on a `known` segment is a failure
to *return*, and long search on an `unseen` segment is exploration cost. Arrival
uses the objective's own targets, so a model that reaches a decoy fixture first
has not arrived.

## Running

```bash
for c in perception_oracle perfect_recall goal_location; do
  sbatch owb/scripts/model_sweep.slurm wildlands --task a --seed 7 --start-milestone 2 5 8 \
    --condition $c --checkpoint-memory clues --max-calls 60 --output runs/ladder-$c-clues \
    --transport codex --account-home ~/.codex-roboharness04 \
    --codex-executable .cache/codex-sol/node_modules/.bin/codex --model gpt-6.1-sol --effort medium
done
python -B -m owb.scripts.checkpoint_report runs/ladder-*-clues
```

`checkpoint_report` adds the handed segment's `kind`, reach, and search and
execution actions to each episode row. It also prints a **ladder table** with one
row per (world, task, model, memory) and these columns:
- next-milestone rate and reach rate for each rung;
- the memory gap (`perfect_recall − perception_oracle`);
- the exploration gap (`goal_location − perfect_recall`).

Gaps are printed only when both conditions cover the same start set.

Interpretation notes:

- Run `clues` and `empty` separately. `perfect_recall` with `empty` is perfect
  memory of the model's own experience only. At a checkpoint start that is
  usually very little, so the rung-2 gain is small there by construction.
- Classify milestones before reading the gap. The question is decided on
  `seen_by_earlier_operator`/`known` segments and on clue-guided ones (for
  example, the Wildlands A01 landmark clue that leads to the cache, or Skylands
  B08 evidence that leads to B12). On `unseen` segments with no clue, rung 2
  cannot help, and rung 3 removes plain exploration.
- Single-seed, single-route checkpoint results describe segments, not full
  episodes. The checkpoint-start caveats apply.
- As of 2026-11-24, the committed Wildlands A seed-7 and Skylands A seed-7 banks
  predate the working-memory change and no longer restore on current code
  (`Bank was collected from a different initial state`). Re-collect banks before
  running checkpoint starts (`owb.scripts.checkpoint_bank collect`; Skylands in
  Slurm). Without a matching bank, `reference_path_m` is omitted.
