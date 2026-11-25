# Minecraft: long-horizon tasks

Status: two existing tasks, adopted unchanged from MCU-AgentBeats as run in the MFM
paper (*Multimodal Filesystem Memory for Long-Horizon Embodied Agents*, `../MFM`).
They can be run today through MFM's harness, but not yet through `owb.make_env`. The
milestone specification and the scorer are in [`owb/tasks/minecraft.py`](../owb/tasks/minecraft.py).

| Task | Milestones | Budget of one life | What it measures |
| --- | --- | --- | --- |
| A: Mine Diamond from Scratch | 10 | 12,000 steps | Getting through the tech tree from nothing: wood, stone, iron, smelting and diamond, with all crafting done by hand in the inventory GUI. |
| B: Ender Dragon | 24 | 36,000 steps | The whole game: Task A's chain, then obsidian, the Nether, blazes and endermen, eyes of ender, a stronghold and The End. |

The milestone counts differ from the 12 and 24 of the
[Wildlands](wildlands-long-horizon-tasks.md) and [Teyvat](teyvat-long-horizon-tasks.md)
designs. They are kept as
published so that scores stay directly comparable with MCU-AgentBeats and the MFM paper.
Do not renumber them, merge them or add to them.

## Substrate

The tasks run on the real game through MineStudio (the MineRL engine) under
MCU-AgentBeats, a fork of CraftJarvis MCU. Minecraft and the engine are installed by
the user; OWB distributes only the task specification. This is unlike the other OWB
environments, which are clean-room simulators, and the README's environment paragraph
will need a note before this row ships.

| | Contract |
| --- | --- |
| Observation | An RGB game frame, plus the state a human player can read off the HUD, the F3 screen and the pause-menu Statistics screen: `player_pos`, `location_stats`, `inventory`, `equipped_items`, `isGuiOpen`, `health`, `food_level`, and the cumulative statistics `mine_block`, `craft_item`, `pickup`, `kill_entity`, `break_item`, `use_item`, `damage_dealt`. |
| Action | MineStudio's raw action space: `forward back left right jump sneak sprint attack use drop inventory hotbar.1-9`, and `camera` as `[yaw, pitch]` in degrees. The engine has no `craft`, `smelt` or `place` verb, so crafting and smelting go through the GUI by cursor. |
| Withheld | Off-screen mob positions, the scorer's per-milestone completion bits and the scoring configuration itself. An agent may know the public milestone list, but it must work out its own progress from the statistics. |
| Built-in procedures | None. No arm gets an aim, walk, craft, mine, tree, shaft or portal macro. The only procedures an agent may run are ones it wrote itself, made of the same raw actions. |

## Task A: Mine Diamond from Scratch — 10 milestones

**Instruction:** "Starting with nothing, obtain a diamond by gathering wood, crafting
tools, mining ores, and smelting iron." Default world seed 0, empty starting inventory.

| ID | Stage | Milestone | Fires when this statistic increases |
| --- | --- | --- | --- |
| A01 | Basic survival | `mine_log` | `mine_block`: `log`, `oak_log`, `birch_log`, `spruce_log` |
| A02 | Basic survival | `crafting_table` | `craft_item`: `crafting_table` |
| A03 | Stone age | `wooden_pickaxe` | `craft_item`: `wooden_pickaxe` |
| A04 | Stone age | `mine_cobblestone` | `mine_block`: `cobblestone` |
| A05 | Stone age | `stone_pickaxe` | `craft_item`: `stone_pickaxe` |
| A06 | Iron age | `mine_iron_ore` | `mine_block`: `iron_ore`, `deepslate_iron_ore` |
| A07 | Iron age | `furnace` | `craft_item`: `furnace` |
| A08 | Iron age | `iron_ingot` | `pickup`: `iron_ingot` |
| A09 | Diamond | `iron_pickaxe` | `craft_item`: `iron_pickaxe` |
| A10 | Diamond | `diamond` | `pickup`: `diamond` |

## Task B: Ender Dragon — 24 milestones

**Instruction:** "Starting from nothing, defeat the Ender Dragon by gathering resources,
exploring the Nether, finding a stronghold, and conquering The End." Default world seed
0, empty starting inventory.

| ID | Stage | Milestone | Fires when this statistic increases |
| --- | --- | --- | --- |
| B01 | Wood and tools | `mine_log` | `mine_block`: any of the seven `*_log` types |
| B02 | Wood and tools | `crafting_table` | `craft_item`: `crafting_table` |
| B03 | Wood and tools | `wooden_pickaxe` | `craft_item`: `wooden_pickaxe` |
| B04 | Stone age | `mine_cobblestone` | `mine_block`: `cobblestone` |
| B05 | Stone age | `stone_pickaxe` | `craft_item`: `stone_pickaxe` |
| B06 | Iron age | `mine_iron_ore` | `mine_block`: `iron_ore`, `deepslate_iron_ore` |
| B07 | Iron age | `furnace` | `craft_item`: `furnace` |
| B08 | Iron age | `iron_ingot` | `pickup`: `iron_ingot` |
| B09 | Iron age | `iron_pickaxe` | `craft_item`: `iron_pickaxe` |
| B10 | Diamond age | `mine_diamond_ore` | `mine_block`: `diamond_ore`, `deepslate_diamond_ore` |
| B11 | Diamond age | `diamond` | `pickup`: `diamond` |
| B12 | Diamond age | `diamond_pickaxe` | `craft_item`: `diamond_pickaxe` |
| B13 | Diamond age | `diamond_sword` | `craft_item`: `diamond_sword` |
| B14 | Nether preparation | `mine_obsidian` | `mine_block`: `obsidian` |
| B15 | Nether preparation | `flint_and_steel` | `craft_item`: `flint_and_steel` |
| B16 | Nether | `netherrack` | `pickup`: `netherrack` |
| B17 | Nether | `kill_blaze` | `kill_entity`: `blaze` |
| B18 | Nether | `blaze_rod` | `pickup`: `blaze_rod` |
| B19 | Nether | `kill_enderman` | `kill_entity`: `enderman` |
| B20 | Nether | `ender_pearl` | `pickup`: `ender_pearl` |
| B21 | Eye of ender | `blaze_powder` | `craft_item`: `blaze_powder` |
| B22 | Eye of ender | `eye_of_ender` | `craft_item`: `ender_eye` |
| B23 | The End | `end_stone` | `pickup`: `end_stone` |
| B24 | The End | `kill_ender_dragon` | `kill_entity`: `ender_dragon` |

The stage names describe the intended order, but the scorer does not enforce it. For
example, B16 counts the item and not the dimension: in MFM, a life on the default world
earned it from netherrack in a ruined portal on the Overworld surface.

## Scoring

The scoring rule is that of MineStudio's `RewardsCallback`, which MCU-AgentBeats'
`MilestoneTrackerCallback` extends:

- A milestone fires once, on the first step at which its statistic increases for any
  of its listed objects.
- Only increases count. Statistics present at reset, and items already in the starting
  inventory, earn nothing.
- The task score is the percentage of milestones fired, reported as, for example,
  `54.17 (13/24)`.

`owb.tasks.MilestoneTracker` applies this rule:

```python
from owb.tasks import MilestoneTracker, minecraft_task

tracker = MilestoneTracker(minecraft_task("ender_dragon"))
tracker.reset(info)                    # info from the environment reset
for step in range(1, budget + 1):
    ...
    fired = tracker.update(info, step)
print(tracker.summary())               # {'milestones_achieved': 13, 'score': 54.17, ...}
```

The tracker is for the evaluator only. Its results must not be fed back to the agent.
[`tests/test_minecraft_tasks.py`](../tests/test_minecraft_tasks.py) checks that the
specification is identical to MFM's task YAML when a `../MFM` checkout is present.

## Where memory enters

Unlike the designed tasks for the other environments, these tasks have no planted
probes. The milestones reward progress, not recall. Memory is tested through three
features of the setup:

- **The horizon.** 12,000 and 36,000 raw actions are far beyond any context window. A
  single crafting recipe takes dozens of cursor moves, and the stronghold search alone
  can take thousands of steps.
- **Spatial and episodic recall within a life.** The agent must find its way back to
  its crafting table, furnace, shaft, a lava lake, a portal and a fortress. It must also
  remember what killed it last time and where it respawns.
- **Lives.** A life is one run of one budget. A chain of lives replays the same world
  seed, and each life starts from the memory of the lives before it (MFM's
  `--seed-memory` for MFM arms and `--seed-log` for the logbook arm). This matches
  OWB's *cross-session persistence* axis. In MFM, the dragon was not killed in a single
  life: the 16/24 in the paper came from lives 2 and 3 of one chain, after life 1
  reached 10/24 from nothing. Report every life of a chain, together with what it
  carried.

### Diagnostic starts (Task B only)

MFM also defines Ender Dragon starts that skip the early game. They keep the same 24
rules but begin with a kit and further on in the game. They are diagnostics, not the
task, and their scores are not comparable with a from-scratch life.

| MCU task | Start | Milestones credited before step 1 |
| --- | --- | --- |
| `ender_dragon_half` | Iron-age kit, 5 diamonds dropped at the agent's feet | B01–B10 |
| `ender_dragon_resume` | Diamond gear, flint and steel and 12 obsidian, at the spawn | B01–B15 |
| `ender_dragon_fortress` | The same kit plus armour and a shield, on a Nether fortress bridge near a blaze spawner | B01–B16 (settled) |

The credit is recorded only in an agent-side key (`given_milestones`). The scorer
ignores it and counts what the agent earns during the run.

## Reference results (MFM)

The full tables, with model, effort and commit, are in `../MFM/README.md`. Highlights:

| Task | Arm | Result |
| --- | --- | --- |
| Diamond, 12,000 | MFM, gpt-6-astra high | 9/10 in each of three lives |
| Diamond, 40,000 | MFM, claude-opus-5-5 high | 10/10 |
| Ender Dragon, 100,000 per life | MFM, paper chain | Life 1: 10/24; lives 2 and 3: 16/24 |
| Ender Dragon, fortress start (staged: the search is handed over) | MFM, claude-opus-5-5 high | 19/24 distinct: B01–B16 settled, plus B17, B18 and B21 earned |

## Running

Until OWB has a `minecraft` adapter, run the tasks with MFM's harness. From
`../MFM/MCU`, after its setup (`mcu_worldmodel/SETUP.md`: JDK 8, `xvfb-run` and the
engine):

```bash
bash scripts/run-arm.sh --arm diamond_s0 --memory mfm \
    --task mine_diamond_from_scratch --world-seed 0 \
    --green-port 9053 --purple-port 9063
bash scripts/run-arm.sh --arm dragon_l1 --memory mfm \
    --task ender_dragon --world-seed 0 --green-port 9054 --purple-port 9064
```

`--memory` selects the arm (`direct`, `logbook`, `vista`, `mfm` or `mfm-matched`), and
`--steps` overrides the budget. The result is written to `logs/<arm>_result.json`.

## Implementation fit

A `minecraft` entry in `owb.make_env` would be a `NativeAdapter` around a MineStudio
simulator. It would need to do four things:

1. Map the OWB shared action schema onto the raw button and camera space, or expose
   the raw space as a native action. The shared schema has no cursor, and the tasks
   cannot be completed without the GUI.
2. Pass through only the observation fields listed under Substrate, and never
   `milestones`.
3. Run `MilestoneTracker` in the adapter, and put its summary in the episode summary
   rather than in the observations.
4. Accept a world seed, a step budget and a carried-memory directory, so that the
   recorder can run chains of lives.
