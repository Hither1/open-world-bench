"""Map a Teyvat :class:`QuestGraph` onto a :class:`TaskSpec`.

Each objective becomes one milestone with the objective's id, so that
``MilestoneEvaluator`` completes the same objectives at the same steps as
``QuestEngine.consume`` on the same events and state (tests/test_task_teyvat.py):

* stage order inside a quest and quest prerequisites become ``requires``. An objective of
  stage k requires the required objectives of stage k-1; an objective of a quest's first
  stage requires the last stage of each prerequisite quest;
* optional objectives get ``role="optional"`` (not needed for success) and
  ``expires_after`` set to their stage's required objectives, because ``QuestEngine`` stops
  checking a stage once it advances;
* a stage made only of optional objectives advances on the first update it is checked.
  It is represented by a ``role="gate"`` milestone with an always-true predicate, which
  later stages require and which closes the optional objectives' window.

Not represented: quest completion rewards and ``grants`` (item hand-overs are simulator
state, which the evaluator reads back through the inventory), and the instruction text
QuestEngine generates for the current objective.
"""
from __future__ import annotations

from owb.sims.teyvat.quests.schema import ObjKind, Objective, QuestGraph

from ..schema import GATE_ROLE, OPTIONAL_ROLE, Budget, Milestone, Predicate, TaskSpec

QUESTLINE_TITLE = "The Windborne Ledger"


def _pos(obj: Objective) -> tuple[float, float, float]:
    if obj.position is None:
        raise ValueError(f"objective {obj.id!r} ({obj.kind.value}) needs a position")
    return tuple(float(v) for v in obj.position)  # type: ignore[return-value]


def objective_predicate(obj: Objective) -> Predicate:
    """The predicate QuestEngine checks for one objective."""
    k, targets = obj.kind, tuple(obj.targets)
    if k == ObjKind.TALK:
        return Predicate("event", {"type": "talk", "targets": targets})
    if k == ObjKind.COLLECT:
        return Predicate("inventory_has", {"item": obj.item or "", "count": obj.count})
    if k in (ObjKind.ACTIVATE, ObjKind.OPEN_CHEST, ObjKind.DEFEAT):
        return Predicate("event", {"type": k.value, "targets": targets, "count": obj.count,
                                   "distinct": True})
    if k == ObjKind.SEQUENCE:
        return Predicate("sequence", {"type": "activate", "targets": targets})
    if k == ObjKind.REACH:
        return Predicate("reach", {"position": _pos(obj), "radius": float(obj.radius)})
    if k == ObjKind.ALTITUDE:
        args = {"min_height": float(obj.min_height), "radius": float(obj.radius)}
        if obj.position is not None:
            args["position"] = _pos(obj)
        if obj.time_window is not None:
            args["time_window"] = tuple(float(t) for t in obj.time_window)
        return Predicate("altitude", args)
    if k == ObjKind.DIVE:
        return Predicate("dive", {"position": _pos(obj), "radius": float(obj.radius)})
    if k == ObjKind.DELIVER:
        return Predicate("deliver", {"item": obj.item or "", "count": obj.count, "targets": targets})
    raise ValueError(f"unhandled objective kind {k!r}")


def questgraph_to_spec(graph: QuestGraph, *, id: str, sim: str = "teyvat", instruction: str = "",
                       title: str = QUESTLINE_TITLE, budget: Budget | None = None,
                       setup: str | None = None, setup_params: dict | None = None) -> TaskSpec:
    """One milestone per objective, in quest / stage / objective order."""
    per_quest: dict[str, list[Milestone]] = {}
    # quest id -> ids whose firing means its last stage is done
    finished: dict[str, tuple[str, ...]] = {}
    pending = list(graph.quests)
    while pending:   # prerequisites first; QuestGraph already rejects cycles
        quest = next(q for q in pending if all(p in finished for p in q.prerequisites))
        pending.remove(quest)
        gate: tuple[str, ...] = tuple(i for p in quest.prerequisites for i in finished[p])
        for k, stage in enumerate(quest.stages):
            label = f"{quest.id}:{k}"
            required = tuple(o.id for o in stage.objectives if not o.optional)
            stage_ms = []
            if not required:
                gate_id = f"{quest.id}:stage{k}"
                stage_ms.append(Milestone(gate_id, f"{quest.title}: stage {k + 1}", label,
                                          Predicate("all", {"preds": ()}), requires=gate,
                                          role=GATE_ROLE))
                required = (gate_id,)
            for o in stage.objectives:
                stage_ms.append(Milestone(
                    o.id, o.description, label, objective_predicate(o), requires=gate,
                    role=OPTIONAL_ROLE if o.optional else "", reward=float(o.reward),
                    expires_after=required if o.optional else ()))
            per_quest.setdefault(quest.id, []).extend(stage_ms)
            gate = required
        finished[quest.id] = gate
    # Keep the graph's own quest order in the listing (evaluation is order-free).
    milestones = [m for q in graph.quests for m in per_quest[q.id]]
    return TaskSpec(id=id, sim=sim, title=title, instruction=instruction,
                    milestones=tuple(milestones), budget=budget or Budget(), setup=setup,
                    setup_params=dict(setup_params or {}))


def builtin_questline_spec(seed: int = 0, *, instruction: str = "") -> TaskSpec:
    """The built-in questline of the common ``teyvat`` env (small variant) at ``seed``.

    Positions depend on the world, so the graph is taken from a freshly built native env,
    exactly as ``owb.make_env("teyvat", seed=seed)`` builds it. The empty default
    instruction leaves the native per-objective instruction in place under ``TaskEnv``.
    """
    from owb.bench.framework import make_env

    env = make_env("teyvat", seed=int(seed), image_size=(8, 6))
    try:
        graph = env.native.graph
    finally:
        env.close()
    return questgraph_to_spec(graph, id="teyvat.windborne_ledger", instruction=instruction,
                              setup_params={"world_seed": int(seed), "variant": "small"})
