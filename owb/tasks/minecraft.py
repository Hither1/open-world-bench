"""Minecraft long-horizon tasks: Mine Diamond from Scratch and Ender Dragon.

The two tasks are taken unchanged from MCU-AgentBeats, as used by the MFM paper
(*Multimodal Filesystem Memory for Long-Horizon Embodied Agents*). The task data lives in
`owb/tasks/specs/minecraft/<name>.yaml`; each milestone there
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
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .evaluator import MilestoneEvaluator, State
from .schema import Budget, Milestone as SpecMilestone, Predicate, TaskSpec, load

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

    @classmethod
    def from_spec(cls, spec: TaskSpec) -> "MinecraftTask":
        """A Minecraft task from a spec whose milestones are all ``stat_increase``."""
        if spec.sim != "minecraft":
            raise ValueError(f"task {spec.id!r} is for sim {spec.sim!r}, not minecraft")
        milestones = []
        for m in spec.milestones:
            if m.done.kind != "stat_increase" or m.done["stat"] not in EVENTS or m.requires:
                raise ValueError(f"Minecraft milestone {m.id!r} must be an unordered "
                                 f"stat_increase over one of {EVENTS}")
            milestones.append(Milestone(m.id, m.done["stat"], tuple(m.done["objects"]), m.stage))
        return cls(name=spec.id, title=spec.title, text=spec.instruction,
                   milestones=tuple(milestones), step_budget=int(spec.budget.steps),
                   world_seed=int(spec.setup_params.get("world_seed", 0)))

    def to_spec(self) -> TaskSpec:
        """The equivalent spec (milestone titles are the identities)."""
        return TaskSpec(
            id=self.name, sim="minecraft", title=self.title, instruction=self.text,
            milestones=tuple(SpecMilestone(
                m.identity, m.identity, m.stage, reward=1.0,
                done=Predicate("stat_increase", {"stat": m.event, "objects": m.objects}))
                for m in self.milestones),
            budget=Budget(steps=self.step_budget), setup_params={"world_seed": self.world_seed})


def _m(stage: str, event: str, identity: str, *objects: str) -> Milestone:
    assert event in EVENTS, event
    return Milestone(identity, event, tuple(objects) or (identity,), stage)


_SPECS = Path(__file__).resolve().parent / "specs" / "minecraft"

# The milestone lists live in owb/tasks/specs/minecraft/<name>.yaml.
MINE_DIAMOND_FROM_SCRATCH = MinecraftTask.from_spec(load(_SPECS / "mine_diamond_from_scratch.yaml"))
ENDER_DRAGON = MinecraftTask.from_spec(load(_SPECS / "ender_dragon.yaml"))

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
    that MineStudio returns. This is a thin wrapper over
    :class:`owb.tasks.evaluator.MilestoneEvaluator` with the task's spec.
    """

    task: MinecraftTask
    achieved: dict[str, int] = field(default_factory=dict)   # identity -> step fired
    _evaluator: MilestoneEvaluator | None = field(default=None, repr=False)

    def reset(self, info: Mapping[str, Any]) -> None:
        self._evaluator = MilestoneEvaluator(self.task.to_spec())
        self._evaluator.reset(State(stats=self._snapshot(info)))
        self.achieved = self._evaluator.achieved

    def update(self, info: Mapping[str, Any], step: int) -> list[str]:
        """Record this step's statistics and return the milestones that fired."""
        if self._evaluator is None:
            raise RuntimeError("call reset(info) before update()")
        fired = self._evaluator.update((), State(stats=self._snapshot(info)), step)
        self.achieved = self._evaluator.achieved
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
