"""Evaluator-side milestone ledger for a :class:`~owb.tasks.schema.TaskSpec`.

`MilestoneEvaluator` consumes, after every step, the step's events and a read-only
:class:`StateView` of the world. It never touches the simulator and its results are not
meant for the agent.

Semantics (docs/tasks.md has the long form):

* A milestone is *eligible* on an update when every id in ``requires`` fired on an earlier
  update, and every id in ``expires_after`` (if any) had not all fired on an earlier
  update. Firing within the same update never unlocks a dependent, so the outcome does not
  depend on the order milestones are listed in. This is also how ``QuestEngine`` advances
  stages and unlocks quests.
* Only eligible milestones are evaluated. Stateful predicates (``event`` counts,
  ``sequence`` progress) start accumulating when the milestone first becomes eligible;
  ``stat_increase`` looks at the change since the previous update, whatever was eligible.
* An eligible milestone whose predicate holds fires once and records the step.
"""
from __future__ import annotations

from typing import Any, Iterable, Mapping, Protocol, runtime_checkable

import numpy as np

from .schema import Milestone, Predicate, TaskSpec


@runtime_checkable
class StateView(Protocol):
    """What predicates may read. Every field may be ``None`` when a sim lacks the data.

    ``position`` is (x, y, z) with y up; ``time_of_day`` is in [0, 1); ``inventory`` maps
    item id -> count; ``stats`` maps statistic -> object -> cumulative value.
    """

    position: Any
    time_of_day: float | None
    inventory: Mapping[str, int] | None
    stats: Mapping[str, Mapping[str, float]] | None


class State:
    """A plain :class:`StateView`, for tests and for sims that report dicts."""

    def __init__(self, position=None, time_of_day=None, inventory=None, stats=None):
        self.position, self.time_of_day = position, time_of_day
        self.inventory, self.stats = inventory, stats

    def __repr__(self) -> str:
        return (f"State(position={self.position!r}, time_of_day={self.time_of_day!r}, "
                f"inventory={self.inventory!r}, stats=...)")


def _etype(event: Any) -> Any:
    return event.get("type") if isinstance(event, Mapping) else getattr(event, "type", None)


def _eget(event: Any, key: str, default: Any = None) -> Any:
    return event.get(key, default)


def _value(x: Any) -> float:
    return float(np.asarray(x).item())


def _field(state: Any, name: str) -> Any:
    return None if state is None else getattr(state, name, None)


def _horizontal(pos: Any, target: Any) -> float:
    d = np.asarray(pos, dtype=np.float32) - np.asarray(target, dtype=np.float32)
    return float(np.linalg.norm(d[[0, 2]]))


def _walk(pred: Predicate, path: str):
    yield path, pred
    if pred.kind in ("all", "any"):
        for i, child in enumerate(pred["preds"]):
            yield from _walk(child, f"{path}/{i}")


class MilestoneEvaluator:
    def __init__(self, spec: TaskSpec):
        self.spec = spec
        self._stat_keys = sorted({(p["stat"], obj) for m in spec.milestones
                                  for _, p in _walk(m.done, m.id) if p.kind == "stat_increase"
                                  for obj in p["objects"]})
        self.reset(None)

    # ------------------------------------------------------------------ public
    def reset(self, state: StateView | None = None) -> None:
        self.achieved: dict[str, int] = {}
        self.step: int | None = None
        self._counts: dict[str, int] = {}             # predicate path -> matching events
        self._seen: dict[str, list] = {}              # predicate path -> distinct entities
        self._sequence: dict[str, list] = {}          # predicate path -> correct prefix
        self._prev_stats = self._stats(state)
        self._prev_inventory = self._inventory(state)

    def update(self, events: Iterable[Any], state: StateView | None, step: int) -> list[str]:
        """Record one step; return the ids of the milestones that fired, in spec order."""
        events = list(events or ())
        self.step = int(step)
        ctx = _Context(events, state, self._prev_stats, self._stats(state),
                       self._prev_inventory)
        done_before = set(self.achieved)
        fired = []
        for m in self.spec.milestones:
            if m.id in done_before or not self._eligible(m, done_before):
                continue
            if self._holds(m.done, m.id, ctx):
                self.achieved[m.id] = self.step
                fired.append(m.id)
        self._prev_stats = ctx.stats
        self._prev_inventory = self._inventory(state)
        return fired

    @property
    def progress(self) -> float:
        scored = [m for m in self.spec.milestones if m.scored]
        return sum(m.id in self.achieved for m in scored) / max(1, len(scored))

    @property
    def success(self) -> bool:
        return all(m.id in self.achieved for m in self.spec.milestones if m.required)

    def eligible(self) -> list[str]:
        """Ids that are eligible on the next update and have not fired."""
        done = set(self.achieved)
        return [m.id for m in self.spec.milestones if m.id not in done and self._eligible(m, done)]

    def summary(self) -> dict[str, Any]:
        scored = [m for m in self.spec.milestones if m.scored]
        return {
            "task": self.spec.id,
            "milestones_achieved": sum(m.id in self.achieved for m in scored),
            "total_milestones": len(scored),
            "progress": self.progress,
            "success": self.success,
            "achieved": dict(self.achieved),
        }

    def snapshot(self) -> dict[str, Any]:
        """JSON-serialisable evaluator state (for logs and resume)."""
        return {
            "task": self.spec.id,
            "version": self.spec.version,
            "step": self.step,
            "achieved": dict(self.achieved),
            "counts": dict(self._counts),
            "seen": {k: list(v) for k, v in self._seen.items()},
            "sequence": {k: list(v) for k, v in self._sequence.items()},
            "prev_stats": [[s, o, v] for (s, o), v in self._prev_stats.items()],
            "prev_inventory": dict(self._prev_inventory) if self._prev_inventory is not None else None,
        }

    # ---------------------------------------------------------------- internal
    def _eligible(self, m: Milestone, done: set[str]) -> bool:
        if not all(r in done for r in m.requires):
            return False
        return not (m.expires_after and all(e in done for e in m.expires_after))

    def _stats(self, state) -> dict[tuple[str, str], float]:
        stats = _field(state, "stats") or {}
        return {(s, o): _value((stats.get(s) or {}).get(o, 0.0)) for s, o in self._stat_keys}

    @staticmethod
    def _inventory(state) -> dict[str, int] | None:
        inv = _field(state, "inventory")
        return None if inv is None else {str(k): int(v) for k, v in inv.items()}

    def _holds(self, p: Predicate, path: str, ctx: "_Context") -> bool:
        k = p.kind
        if k == "all":
            # Evaluate every child so that stateful children keep accumulating.
            results = [self._holds(c, f"{path}/{i}", ctx) for i, c in enumerate(p["preds"])]
            return all(results)
        if k == "any":
            results = [self._holds(c, f"{path}/{i}", ctx) for i, c in enumerate(p["preds"])]
            return any(results)

        if k == "stat_increase":
            # MineStudio RewardsCallback: deltas of the objects summed in order; the
            # milestone fires as soon as the running sum is positive.
            delta = 0.0
            for obj in p["objects"]:
                key = (p["stat"], obj)
                delta += ctx.stats[key] - ctx.prev_stats[key]
                if delta > 0:
                    return True
            return False

        if k == "event":
            targets, item = p["targets"], p["item"]
            matches = [e for e in ctx.events if _etype(e) == p["type"]
                       and (not targets or _eget(e, "entity") in targets)
                       and (item is None or _eget(e, "item") == item)]
            if p["distinct"]:
                seen = self._seen.setdefault(path, [])
                for e in matches:
                    ent = _eget(e, "entity")
                    if ent not in seen:
                        seen.append(ent)
                return len(seen) >= p["count"]
            self._counts[path] = self._counts.get(path, 0) + len(matches)
            return self._counts[path] >= p["count"]

        if k == "sequence":
            seq = self._sequence.setdefault(path, [])
            targets = p["targets"]
            for e in ctx.events:
                if _etype(e) != p["type"]:
                    continue
                ent = _eget(e, "entity")
                if ent not in targets:
                    continue
                expected = targets[len(seq)] if len(seq) < len(targets) else None
                if ent == expected:
                    seq.append(ent)
                else:
                    seq.clear()          # a wrong target resets the order
            return len(seq) >= len(targets)

        if k == "inventory_has":
            inv = _field(ctx.state, "inventory")
            return inv is not None and int(inv.get(p["item"], 0)) >= p["count"]

        if k == "deliver":
            # Inventory before this step's own hand-over: the previous state's inventory
            # plus this step's `collect` events. The sim performs the deduction.
            if ctx.prev_inventory is None:
                return False
            have = ctx.prev_inventory.get(p["item"], 0) + sum(
                int(_eget(e, "amount", 1)) for e in ctx.events
                if _etype(e) == "collect" and _eget(e, "item") == p["item"])
            if have < p["count"]:
                return False
            return any(_etype(e) == p["type"] and _eget(e, "entity") in p["targets"]
                       for e in ctx.events)

        pos = _field(ctx.state, "position")
        if pos is None:
            return False
        if k == "reach":
            return _horizontal(pos, p["position"]) <= p["radius"]

        if k == "altitude":
            if float(pos[1]) < p["min_height"]:
                return False
            if p["position"] is not None and _horizontal(pos, p["position"]) > p["radius"]:
                return False
            if p["time_window"] is not None:
                tod = _field(ctx.state, "time_of_day")
                if tod is None:
                    return False
                t = float(tod) % 1.0
                lo, hi = p["time_window"]
                if not ((lo <= t <= hi) if lo <= hi else (t >= lo or t <= hi)):
                    return False
            return True

        if k == "dive":
            return (_horizontal(pos, p["position"]) <= p["radius"]
                    and float(pos[1]) <= p["position"][1] + p["margin"])

        raise ValueError(f"unhandled predicate kind {k!r}")


class _Context:
    __slots__ = ("events", "state", "prev_stats", "stats", "prev_inventory")

    def __init__(self, events, state, prev_stats, stats, prev_inventory):
        self.events, self.state = events, state
        self.prev_stats, self.stats, self.prev_inventory = prev_stats, stats, prev_inventory


__all__ = ["MilestoneEvaluator", "State", "StateView"]
