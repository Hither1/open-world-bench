"""Quest progression engine: consumes world events, emits rewards and goals."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np

from .schema import Event, ObjKind, Objective, Quest, QuestGraph


class QuestStatus(IntEnum):
    LOCKED = 0
    ACTIVE = 1
    COMPLETE = 2


@dataclass
class QuestProgress:
    status: QuestStatus = QuestStatus.LOCKED
    stage: int = 0
    done: set[str] = field(default_factory=set)              # objective ids
    counters: dict[str, int] = field(default_factory=dict)   # objective id -> progress
    hit_targets: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    sequence: dict[str, list[str]] = field(default_factory=lambda: defaultdict(list))


class QuestEngine:
    """Tracks a :class:`QuestGraph` against a stream of world events.

    The engine owns no world state of its own beyond progress, so it can be
    unit-tested by feeding it synthetic events.
    """

    def __init__(self, graph: QuestGraph, inventory: dict[str, int] | None = None):
        self.graph = graph
        self.progress: dict[str, QuestProgress] = {q.id: QuestProgress() for q in graph.quests}
        self.inventory: dict[str, int] = dict(inventory or {})
        self.log: list[str] = []
        self.completed_objectives = 0
        self.completed_quests = 0
        self._unlock()

    # ------------------------------------------------------------- unlocking
    def _unlock(self) -> list[Quest]:
        newly = []
        for q in self.graph.quests:
            p = self.progress[q.id]
            if p.status != QuestStatus.LOCKED:
                continue
            if all(self.progress[pre].status == QuestStatus.COMPLETE for pre in q.prerequisites):
                p.status = QuestStatus.ACTIVE
                newly.append(q)
                self.log.append(f"quest started: {q.title}")
        return newly

    # ------------------------------------------------------------ inspection
    @property
    def active_quests(self) -> list[Quest]:
        return [q for q in self.graph.quests if self.progress[q.id].status == QuestStatus.ACTIVE]

    @property
    def all_complete(self) -> bool:
        return all(p.status == QuestStatus.COMPLETE for p in self.progress.values())

    def current_stage_objectives(self, quest: Quest) -> list[Objective]:
        p = self.progress[quest.id]
        if p.status != QuestStatus.ACTIVE or p.stage >= len(quest.stages):
            return []
        return [o for o in quest.stages[p.stage].objectives if o.id not in p.done]

    def active_objectives(self) -> list[tuple[Quest, Objective]]:
        out = []
        for q in self.active_quests:
            for o in self.current_stage_objectives(q):
                out.append((q, o))
        return out

    def focus(self) -> tuple[Quest, Objective] | None:
        """The one objective an agent should be working on right now."""
        act = [(q, o) for q, o in self.active_objectives() if not o.optional]
        if not act:
            act = self.active_objectives()
        return act[0] if act else None

    def instruction(self) -> str:
        """Human-readable current goal, used as the language observation."""
        f = self.focus()
        if f is None:
            return "All quests complete. Explore freely."
        quest, obj = f
        prog = self.progress[quest.id]
        text = f"[{quest.title}] {obj.description}"
        if obj.count > 1:
            text += f" ({prog.counters.get(obj.id, 0)}/{obj.count})"
        return text

    def focus_target(self) -> np.ndarray | None:
        """World position of the current objective, if it has one."""
        f = self.focus()
        if f is None or f[1].position is None:
            return None
        return np.asarray(f[1].position, dtype=np.float32)

    def completion(self) -> float:
        """Fraction of all objectives completed, for monitoring long runs."""
        total = max(1, self.graph.total_objectives)
        return self.completed_objectives / total

    # ---------------------------------------------------------------- update
    def consume(self, events: list[Event], world_state: dict | None = None) -> float:
        """Apply events, advance stages, return the reward earned this step."""
        world_state = world_state or {}
        reward = 0.0
        for ev in events:
            if ev.type == "collect" and ev.get("item"):
                item = ev.get("item")
                self.inventory[item] = self.inventory.get(item, 0) + int(ev.get("amount", 1))
        for quest in list(self.active_quests):
            reward += self._update_quest(quest, events, world_state)
        self._unlock()
        return reward

    def _update_quest(self, quest: Quest, events: list[Event], world_state: dict) -> float:
        p = self.progress[quest.id]
        reward = 0.0
        if p.stage >= len(quest.stages):
            return self._finish_quest(quest)
        for obj in quest.stages[p.stage].objectives:
            if obj.id in p.done:
                continue
            if self._check(obj, p, events, world_state):
                p.done.add(obj.id)
                self.completed_objectives += 1
                reward += obj.reward
                self.log.append(f"objective done: {obj.description}")
        stage_objs = quest.stages[p.stage].objectives
        if all(o.id in p.done or o.optional for o in stage_objs):
            p.stage += 1
            if p.stage >= len(quest.stages):
                reward += self._finish_quest(quest)
            else:
                note = quest.stages[p.stage].note
                if note:
                    self.log.append(note)
        return reward

    def _finish_quest(self, quest: Quest) -> float:
        p = self.progress[quest.id]
        if p.status == QuestStatus.COMPLETE:
            return 0.0
        p.status = QuestStatus.COMPLETE
        self.completed_quests += 1
        for item in quest.grants:
            self.inventory[item] = self.inventory.get(item, 0) + 1
        self.log.append(f"QUEST COMPLETE: {quest.title}")
        return quest.reward

    # ------------------------------------------------------------- predicates
    def _check(self, obj: Objective, p: QuestProgress, events: list[Event], ws: dict) -> bool:
        k = obj.kind
        if k == ObjKind.TALK:
            return any(e.type == "talk" and e.get("entity") in obj.targets for e in events)

        if k == ObjKind.COLLECT:
            have = self.inventory.get(obj.item or "", 0)
            p.counters[obj.id] = min(have, obj.count)
            return have >= obj.count

        if k in (ObjKind.ACTIVATE, ObjKind.OPEN_CHEST, ObjKind.DEFEAT):
            etype = {ObjKind.ACTIVATE: "activate", ObjKind.OPEN_CHEST: "open_chest", ObjKind.DEFEAT: "defeat"}[k]
            for e in events:
                if e.type == etype and e.get("entity") in obj.targets:
                    p.hit_targets[obj.id].add(e.get("entity"))
            p.counters[obj.id] = len(p.hit_targets[obj.id])
            return len(p.hit_targets[obj.id]) >= obj.count

        if k == ObjKind.SEQUENCE:
            seq = p.sequence[obj.id]
            for e in events:
                if e.type != "activate":
                    continue
                ent = e.get("entity")
                if ent not in obj.targets:
                    continue
                expected = obj.targets[len(seq)] if len(seq) < len(obj.targets) else None
                if ent == expected:
                    seq.append(ent)
                elif ent in obj.targets:
                    seq.clear()           # wrong pillar: the puzzle resets
                    self.log.append("the pillars dim -- the order was wrong")
            p.counters[obj.id] = len(seq)
            return len(seq) >= len(obj.targets)

        if k == ObjKind.REACH:
            pos = ws.get("player_pos")
            if pos is None or obj.position is None:
                return False
            d = np.asarray(pos, dtype=np.float32) - np.asarray(obj.position, dtype=np.float32)
            return bool(np.linalg.norm(d[[0, 2]]) <= obj.radius)

        if k == ObjKind.ALTITUDE:
            pos = ws.get("player_pos")
            if pos is None or float(pos[1]) < obj.min_height:
                return False
            if obj.position is not None:
                d = np.asarray(pos, dtype=np.float32) - np.asarray(obj.position, dtype=np.float32)
                if float(np.linalg.norm(d[[0, 2]])) > obj.radius:
                    return False
            if obj.time_window is not None:
                t = float(ws.get("time_of_day", 0.0)) % 1.0
                lo, hi = obj.time_window
                inside = (lo <= t <= hi) if lo <= hi else (t >= lo or t <= hi)
                if not inside:
                    return False
            return True

        if k == ObjKind.DIVE:
            pos = ws.get("player_pos")
            if pos is None or obj.position is None:
                return False
            d = np.asarray(pos, dtype=np.float32) - np.asarray(obj.position, dtype=np.float32)
            return bool(np.linalg.norm(d[[0, 2]]) <= obj.radius and float(pos[1]) <= obj.position[1] + 1.0)

        if k == ObjKind.DELIVER:
            if self.inventory.get(obj.item or "", 0) < obj.count:
                return False
            for e in events:
                if e.type == "talk" and e.get("entity") in obj.targets:
                    self.inventory[obj.item] = self.inventory.get(obj.item, 0) - obj.count
                    return True
            return False

        raise ValueError(f"unhandled objective kind {k!r}")

    # ------------------------------------------------------------------ state
    def snapshot(self) -> dict:
        return {
            "completed_objectives": self.completed_objectives,
            "completed_quests": self.completed_quests,
            "completion": self.completion(),
            "inventory": dict(self.inventory),
            "instruction": self.instruction(),
            "active": [q.id for q in self.active_quests],
            "stages": {qid: pr.stage for qid, pr in self.progress.items()},
        }
