"""Declarative task specifications: predicates, milestones, budgets, and tasks.

A :class:`TaskSpec` is plain data. It names the simulator it runs on, the instruction the
agent receives, a DAG of :class:`Milestone` objects whose completion conditions are
serialisable :class:`Predicate` trees, and optional resource budgets. Specs load from and
dump to YAML or JSON (one task per file); see docs/tasks.md for the format and the
evaluation semantics, which live in :mod:`owb.tasks.evaluator`.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

_MISSING = object()

# kind -> {argument: default}; `_MISSING` marks a required argument.
PREDICATE_ARGS: dict[str, dict[str, Any]] = {
    "stat_increase": {"stat": _MISSING, "objects": _MISSING},
    "event": {"type": _MISSING, "targets": (), "item": None, "count": 1, "distinct": False},
    "inventory_has": {"item": _MISSING, "count": 1},
    "reach": {"position": _MISSING, "radius": 6.0},
    "altitude": {"min_height": _MISSING, "position": None, "radius": 6.0, "time_window": None},
    "dive": {"position": _MISSING, "radius": 6.0, "margin": 1.0},
    "sequence": {"targets": _MISSING, "type": "activate"},
    "deliver": {"item": _MISSING, "targets": _MISSING, "count": 1, "type": "talk"},
    "all": {"preds": _MISSING},
    "any": {"preds": _MISSING},
}
PREDICATE_KINDS = tuple(PREDICATE_ARGS)

# Milestone roles with a scoring meaning. Any other string (for example the memory roles
# "learning", "interference", "transfer", "composition" of the long-horizon designs) is an
# annotation only.
OPTIONAL_ROLE = "optional"   # counted in progress, not required for success
GATE_ROLE = "gate"           # bookkeeping milestone: required for success, not counted in progress


def _tuplify(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return tuple(_tuplify(v) for v in value)
    if isinstance(value, Mapping):
        return {str(k): _tuplify(v) for k, v in value.items()}
    return value


def _listify(value: Any) -> Any:
    if isinstance(value, Predicate):
        return value.to_dict()
    if isinstance(value, (list, tuple)):
        return [_listify(v) for v in value]
    if isinstance(value, Mapping):
        return {str(k): _listify(v) for k, v in value.items()}
    return value


@dataclass(frozen=True)
class Predicate:
    """A serialisable completion condition. ``args`` holds the kind's parameters."""

    kind: str
    args: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in PREDICATE_ARGS:
            raise ValueError(f"unknown predicate kind {self.kind!r}; choose from {PREDICATE_KINDS}")
        spec = PREDICATE_ARGS[self.kind]
        unknown = set(self.args) - set(spec)
        if unknown:
            raise ValueError(f"predicate {self.kind!r} got unknown arguments {sorted(unknown)}")
        args = {}
        for name, default in spec.items():
            if name in self.args:
                args[name] = self.args[name]
            elif default is _MISSING:
                raise ValueError(f"predicate {self.kind!r} requires argument {name!r}")
            else:
                args[name] = default
        if self.kind in ("all", "any"):
            args["preds"] = tuple(p if isinstance(p, Predicate) else Predicate.from_dict(p)
                                  for p in args["preds"])
        else:
            args = {k: _tuplify(v) for k, v in args.items()}
        self._check(args)
        object.__setattr__(self, "args", args)

    def _check(self, a: dict) -> None:
        k = self.kind
        if k == "stat_increase" and not a["objects"]:
            raise ValueError("stat_increase needs at least one object")
        if k in ("sequence", "deliver") and not a["targets"]:
            raise ValueError(f"{k} needs at least one target")
        if "count" in a and int(a["count"]) < 1:
            raise ValueError(f"{k} count must be >= 1")
        for key in ("position",):
            if a.get(key) is not None and len(a[key]) != 3:
                raise ValueError(f"{k} {key} must have three coordinates")
        if a.get("time_window") is not None and len(a["time_window"]) != 2:
            raise ValueError("altitude time_window must be (start, end) in [0, 1)")

    def __getitem__(self, key: str) -> Any:
        return self.args[key]

    def to_dict(self) -> dict:
        """Flat form: ``{"kind": ..., **args}``, omitting arguments left at their default."""
        out: dict[str, Any] = {"kind": self.kind}
        for name, default in PREDICATE_ARGS[self.kind].items():
            value = self.args[name]
            if default is _MISSING or _tuplify(default) != value:
                out[name] = _listify(value)
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Predicate":
        data = dict(data)
        kind = data.pop("kind")
        args = data.pop("args", None)
        if args is not None and data:
            raise ValueError("predicate mixes an 'args' mapping with flat arguments")
        return cls(kind, dict(args if args is not None else data))


@dataclass(frozen=True)
class Budget:
    """Resource limits for one attempt; ``None`` means unlimited / not declared."""

    steps: int | None = None
    sim_seconds: float | None = None
    model_calls: int | None = None
    tokens: int | None = None

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v is not None}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "Budget":
        return cls(**dict(data or {}))


@dataclass(frozen=True)
class Milestone:
    """One verifiable accomplishment of a task.

    It fires once, at the first update on which it is *eligible* and ``done`` holds. It is
    eligible once every id in ``requires`` fired at an earlier update, and stops being
    eligible once every id in ``expires_after`` fired at an earlier update (an empty
    ``expires_after`` never expires).
    """

    id: str
    title: str
    stage: str
    done: Predicate
    requires: tuple[str, ...] = ()
    role: str = ""
    hidden_from_agent: bool = True
    reward: float = 0.0
    expires_after: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.done, Predicate):
            object.__setattr__(self, "done", Predicate.from_dict(self.done))
        object.__setattr__(self, "requires", tuple(self.requires))
        object.__setattr__(self, "expires_after", tuple(self.expires_after))
        object.__setattr__(self, "reward", float(self.reward))

    @property
    def scored(self) -> bool:
        """Counted in progress."""
        return self.role != GATE_ROLE

    @property
    def required(self) -> bool:
        """Required for success."""
        return self.role != OPTIONAL_ROLE

    def to_dict(self) -> dict:
        out: dict[str, Any] = {"id": self.id, "title": self.title, "stage": self.stage,
                               "done": self.done.to_dict()}
        if self.requires:
            out["requires"] = list(self.requires)
        if self.expires_after:
            out["expires_after"] = list(self.expires_after)
        if self.role:
            out["role"] = self.role
        if not self.hidden_from_agent:
            out["hidden_from_agent"] = False
        if self.reward:
            out["reward"] = self.reward
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Milestone":
        data = dict(data)
        data["done"] = Predicate.from_dict(data["done"])
        return cls(**data)


@dataclass(frozen=True)
class TaskSpec:
    id: str
    sim: str
    title: str
    instruction: str
    milestones: tuple[Milestone, ...]
    budget: Budget = field(default_factory=Budget)
    setup: str | None = None
    setup_params: dict = field(default_factory=dict)
    version: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "milestones", tuple(
            m if isinstance(m, Milestone) else Milestone.from_dict(m) for m in self.milestones))
        if not isinstance(self.budget, Budget):
            object.__setattr__(self, "budget", Budget.from_dict(self.budget))
        object.__setattr__(self, "setup_params", _listify(dict(self.setup_params or {})))
        self._validate()

    def _validate(self) -> None:
        if not self.milestones:
            raise ValueError(f"task {self.id!r} has no milestones")
        ids = [m.id for m in self.milestones]
        dup = sorted({i for i in ids if ids.count(i) > 1})
        if dup:
            raise ValueError(f"task {self.id!r} has duplicate milestone ids {dup}")
        known = set(ids)
        for m in self.milestones:
            for ref in m.requires + m.expires_after:
                if ref not in known:
                    raise ValueError(f"milestone {m.id!r} refers to unknown milestone {ref!r}")
        by_id = {m.id: m for m in self.milestones}
        state: dict[str, int] = {}

        def visit(mid: str, path: tuple[str, ...]) -> None:
            if state.get(mid) == 2:
                return
            if state.get(mid) == 1:
                raise ValueError(f"task {self.id!r} has a requires cycle: {' -> '.join(path + (mid,))}")
            state[mid] = 1
            for pre in by_id[mid].requires:
                visit(pre, path + (mid,))
            state[mid] = 2

        for mid in ids:
            visit(mid, ())

    def milestone(self, mid: str) -> Milestone:
        for m in self.milestones:
            if m.id == mid:
                return m
        raise KeyError(mid)

    # ------------------------------------------------------------ serialisation
    def to_dict(self) -> dict:
        out: dict[str, Any] = {"id": self.id, "version": self.version, "sim": self.sim,
                               "title": self.title, "instruction": self.instruction}
        budget = self.budget.to_dict()
        if budget:
            out["budget"] = budget
        if self.setup is not None:
            out["setup"] = self.setup
        if self.setup_params:
            out["setup_params"] = _listify(self.setup_params)
        out["milestones"] = [m.to_dict() for m in self.milestones]
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TaskSpec":
        data = dict(data)
        data["milestones"] = tuple(Milestone.from_dict(m) for m in data.get("milestones", ()))
        data["budget"] = Budget.from_dict(data.get("budget"))
        return cls(**data)


def load(path: str | Path) -> TaskSpec:
    """Read a spec from a ``.yaml``/``.yml`` or ``.json`` file."""
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix in (".yaml", ".yml"):
        import yaml
        data = yaml.safe_load(text)
    elif path.suffix == ".json":
        data = json.loads(text)
    else:
        raise ValueError(f"unsupported task file {path.name!r}; use .yaml or .json")
    return TaskSpec.from_dict(data)


def dump(spec: TaskSpec, path: str | Path) -> Path:
    """Write ``spec`` as YAML or JSON, chosen by the file suffix."""
    path = Path(path)
    data = spec.to_dict()
    if path.suffix in (".yaml", ".yml"):
        import yaml
        text = yaml.safe_dump(data, sort_keys=False, allow_unicode=True, width=100)
    elif path.suffix == ".json":
        text = json.dumps(data, indent=2) + "\n"
    else:
        raise ValueError(f"unsupported task file {path.name!r}; use .yaml or .json")
    path.write_text(text, encoding="utf-8")
    return path


__all__ = ["Budget", "GATE_ROLE", "Milestone", "OPTIONAL_ROLE", "PREDICATE_KINDS", "Predicate",
           "TaskSpec", "dump", "load"]
