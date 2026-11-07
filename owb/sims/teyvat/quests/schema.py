"""Declarative quest data model.

A quest is an ordered list of stages; a stage is a set of objectives that must
all be satisfied before the next stage unlocks. Quests unlock through
prerequisites, which is what produces a long-horizon dependency graph rather
than a flat checklist.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from ..events import Event


class ObjKind(str, Enum):
    TALK = "talk"                # speak to an NPC
    COLLECT = "collect"          # pick up N of an item
    ACTIVATE = "activate"        # light a beacon / press a mechanism
    SEQUENCE = "sequence"        # activate mechanisms in a given order
    OPEN_CHEST = "open_chest"    # open N chests from a set
    DEFEAT = "defeat"            # defeat N enemies from a set
    REACH = "reach"              # stand inside a region
    ALTITUDE = "altitude"        # get above a height, optionally at a time of day
    DIVE = "dive"                # get below a depth at a location
    DELIVER = "deliver"          # bring an item to an NPC


@dataclass
class Objective:
    id: str
    kind: ObjKind
    description: str
    targets: tuple[str, ...] = ()            # entity / item ids this refers to
    count: int = 1                           # how many are needed
    position: tuple[float, float, float] | None = None   # waypoint hint
    radius: float = 6.0                       # for REACH / DIVE
    min_height: float = 0.0                   # for ALTITUDE
    time_window: tuple[float, float] | None = None       # time-of-day in [0,1)
    item: str | None = None                   # for COLLECT / DELIVER
    reward: float = 1.0                       # sparse reward on completion
    optional: bool = False

    def __post_init__(self) -> None:
        self.kind = ObjKind(self.kind)
        self.targets = tuple(self.targets)


@dataclass
class Stage:
    objectives: list[Objective]
    note: str = ""


@dataclass
class Quest:
    id: str
    title: str
    summary: str
    stages: list[Stage]
    prerequisites: tuple[str, ...] = ()
    reward: float = 5.0
    grants: tuple[str, ...] = ()      # item ids handed over on completion

    def __post_init__(self) -> None:
        self.prerequisites = tuple(self.prerequisites)
        self.grants = tuple(self.grants)

    @property
    def objective_count(self) -> int:
        return sum(len(s.objectives) for s in self.stages)


@dataclass
class QuestGraph:
    quests: list[Quest]

    def __post_init__(self) -> None:
        self.by_id = {q.id: q for q in self.quests}
        self._validate()

    def _validate(self) -> None:
        seen_obj: set[str] = set()
        for q in self.quests:
            for pre in q.prerequisites:
                if pre not in self.by_id:
                    raise ValueError(f"quest {q.id!r} requires unknown quest {pre!r}")
            for st in q.stages:
                if not st.objectives:
                    raise ValueError(f"quest {q.id!r} has an empty stage")
                for o in st.objectives:
                    if o.id in seen_obj:
                        raise ValueError(f"duplicate objective id {o.id!r}")
                    seen_obj.add(o.id)
        # Reject prerequisite cycles so unlocking can never deadlock.
        state: dict[str, int] = {}

        def visit(qid: str) -> None:
            if state.get(qid) == 2:
                return
            if state.get(qid) == 1:
                raise ValueError(f"cyclic quest prerequisites at {qid!r}")
            state[qid] = 1
            for pre in self.by_id[qid].prerequisites:
                visit(pre)
            state[qid] = 2

        for q in self.quests:
            visit(q.id)

    @property
    def total_objectives(self) -> int:
        return sum(q.objective_count for q in self.quests)

    @property
    def total_reward(self) -> float:
        return sum(q.reward + sum(o.reward for s in q.stages for o in s.objectives) for q in self.quests)

__all__ = [
    "ObjKind", "Objective", "Stage", "Quest", "QuestGraph", "Event",
]
