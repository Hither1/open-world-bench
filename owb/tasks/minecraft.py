"""Minecraft long-horizon tasks: Mine Diamond from Scratch and Ender Dragon.

The two tasks are taken unchanged from MCU-AgentBeats, as used by the MFM paper
(*Multimodal Filesystem Memory for Long-Horizon Embodied Agents*). Each milestone below
corresponds one to one, in the same order, to an entry of the task's
`milestone_reward_cfg` in `MCU/MCU_benchmark/task_configs/tasks/<name>/*.yaml`.
`tests/test_minecraft_tasks.py` checks this against a neighbouring `../MFM` checkout
when one is present.

`MilestoneTracker` implements the same scoring rule as MineStudio's `RewardsCallback`,
which the benchmark scorer extends. A milestone fires once, the first time the
cumulative statistic it names (`mine_block`, `craft_item`, `pickup` or `kill_entity`)
increases between two steps for any of its objects. The rule does not enforce order.
Items in the starting inventory and statistics present at reset count for nothing. The
score is the fraction of milestones that have fired.

Agents must not receive the tracker's results. In MFM the scorer's per-milestone bits
are kept off the wire, and an agent that wants its progress works it out from the same
statistics (see docs/minecraft-long-horizon-tasks.md).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

EVENTS = ("mine_block", "craft_item", "pickup", "kill_entity")


@dataclass(frozen=True)
class Milestone:
    identity: str
    event: str
    objects: tuple[str, ...]
    stage: str


@dataclass(frozen=True)
class MinecraftTask:
    name: str                 # the MCU task category, e.g. `--task ender_dragon`
    title: str
    text: str                 # the instruction the agent receives, verbatim from MCU
    milestones: tuple[Milestone, ...]
    step_budget: int          # default budget of one life (MFM `scripts/run-arm.sh`)
    world_seed: int = 0

    @property
    def identities(self) -> tuple[str, ...]:
        return tuple(m.identity for m in self.milestones)


def _m(stage: str, event: str, identity: str, *objects: str) -> Milestone:
    assert event in EVENTS, event
    return Milestone(identity, event, tuple(objects) or (identity,), stage)


MINE_DIAMOND_FROM_SCRATCH = MinecraftTask(
    name="mine_diamond_from_scratch",
    title="Mine Diamond from Scratch",
    text=("Starting with nothing, obtain a diamond by gathering wood, crafting tools, "
          "mining ores, and smelting iron."),
    step_budget=12_000,
    milestones=(
        _m("basic survival", "mine_block", "mine_log",
           "log", "oak_log", "birch_log", "spruce_log"),
        _m("basic survival", "craft_item", "crafting_table"),
        _m("stone age", "craft_item", "wooden_pickaxe"),
        _m("stone age", "mine_block", "mine_cobblestone", "cobblestone"),
        _m("stone age", "craft_item", "stone_pickaxe"),
        _m("iron age", "mine_block", "mine_iron_ore", "iron_ore", "deepslate_iron_ore"),
        _m("iron age", "craft_item", "furnace"),
        _m("iron age", "pickup", "iron_ingot"),
        _m("diamond", "craft_item", "iron_pickaxe"),
        _m("diamond", "pickup", "diamond"),
    ),
)

ENDER_DRAGON = MinecraftTask(
    name="ender_dragon",
    title="Ender Dragon",
    text=("Starting from nothing, defeat the Ender Dragon by gathering resources, "
          "exploring the Nether, finding a stronghold, and conquering The End."),
    step_budget=36_000,
    milestones=(
        _m("wood and tools", "mine_block", "mine_log", "log", "oak_log", "birch_log",
           "spruce_log", "jungle_log", "acacia_log", "dark_oak_log"),
        _m("wood and tools", "craft_item", "crafting_table"),
        _m("wood and tools", "craft_item", "wooden_pickaxe"),
        _m("stone age", "mine_block", "mine_cobblestone", "cobblestone"),
        _m("stone age", "craft_item", "stone_pickaxe"),
        _m("iron age", "mine_block", "mine_iron_ore", "iron_ore", "deepslate_iron_ore"),
        _m("iron age", "craft_item", "furnace"),
        _m("iron age", "pickup", "iron_ingot"),
        _m("iron age", "craft_item", "iron_pickaxe"),
        _m("diamond age", "mine_block", "mine_diamond_ore",
           "diamond_ore", "deepslate_diamond_ore"),
        _m("diamond age", "pickup", "diamond"),
        _m("diamond age", "craft_item", "diamond_pickaxe"),
        _m("diamond age", "craft_item", "diamond_sword"),
        _m("nether preparation", "mine_block", "mine_obsidian", "obsidian"),
        _m("nether preparation", "craft_item", "flint_and_steel"),
        _m("nether", "pickup", "netherrack"),
        _m("nether", "kill_entity", "kill_blaze", "blaze"),
        _m("nether", "pickup", "blaze_rod"),
        _m("nether", "kill_entity", "kill_enderman", "enderman"),
        _m("nether", "pickup", "ender_pearl"),
        _m("eye of ender", "craft_item", "blaze_powder"),
        _m("eye of ender", "craft_item", "eye_of_ender", "ender_eye"),
        _m("the end", "pickup", "end_stone"),
        _m("the end", "kill_entity", "kill_ender_dragon", "ender_dragon"),
    ),
)

MINECRAFT_TASKS: dict[str, MinecraftTask] = {
    t.name: t for t in (MINE_DIAMOND_FROM_SCRATCH, ENDER_DRAGON)
}


def minecraft_task(name: str) -> MinecraftTask:
    try:
        return MINECRAFT_TASKS[name]
    except KeyError:
        raise KeyError(f"unknown Minecraft task {name!r}; "
                       f"available: {sorted(MINECRAFT_TASKS)}") from None


def _count(info: Mapping[str, Any], event: str, obj: str) -> float:
    value = (info.get(event) or {}).get(obj, 0)
    return float(np.asarray(value).item())


@dataclass
class MilestoneTracker:
    """Evaluator-side milestone ledger over MineRL's cumulative statistics.

    Call `reset(info)` with the info of the reset, then `update(info, step)` after every
    step. `info` holds the `mine_block` / `craft_item` / `pickup` / `kill_entity` dicts
    that MineStudio returns.
    """

    task: MinecraftTask
    achieved: dict[str, int] = field(default_factory=dict)   # identity -> step fired
    _prev: dict[str, dict[str, float]] = field(default_factory=dict, repr=False)

    def reset(self, info: Mapping[str, Any]) -> None:
        self.achieved = {}
        self._prev = self._snapshot(info)

    def update(self, info: Mapping[str, Any], step: int) -> list[str]:
        """Record this step's statistics and return the milestones that fired."""
        now = self._snapshot(info)
        fired = []
        for m in self.task.milestones:
            if m.identity in self.achieved:
                continue
            delta = 0.0
            for obj in m.objects:
                # Same accumulation as RewardsCallback: the deltas of the objects are
                # summed in order, and the milestone fires once the sum is positive.
                delta += now[m.event].get(obj, 0.0) - self._prev[m.event].get(obj, 0.0)
                if delta > 0:
                    self.achieved[m.identity] = step
                    fired.append(m.identity)
                    break
        self._prev = now
        return fired

    @property
    def score(self) -> float:
        return len(self.achieved) / len(self.task.milestones)

    def summary(self) -> dict[str, Any]:
        return {
            "task": self.task.name,
            "milestones_achieved": len(self.achieved),
            "total_milestones": len(self.task.milestones),
            "score": round(100 * self.score, 2),
            "achieved": dict(self.achieved),
        }

    def _snapshot(self, info: Mapping[str, Any]) -> dict[str, dict[str, float]]:
        objects: dict[str, set[str]] = {e: set() for e in EVENTS}
        for m in self.task.milestones:
            objects[m.event].update(m.objects)
        return {e: {o: _count(info, e, o) for o in objs} for e, objs in objects.items()}
