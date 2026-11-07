"""Privileged legal-action references, including shortcuts and reordered regions."""
from __future__ import annotations

import numpy as np

from .skylands_campaign import SkylandsCampaignPolicy
from .skylands_reference import neutral


class _PlanningView:
    """Policy-local itinerary; never assign the simulator's current milestone."""
    def __init__(self, env):
        self.native = env
        self.site = None

    @property
    def current(self):
        return self.native.by_id[self.site] if self.site else None

    def __getattr__(self, name):
        return getattr(self.native, name)


class SkylandsTempestPolicy:
    def __init__(self, env, *, route="synchronization", order=("cooling", "anchors", "monitoring"), prepared=None):
        if route not in ("synchronization", "grounding", "capacitors"):
            raise ValueError("unknown shutdown route")
        if len(order) != 3 or set(order) != {"cooling", "anchors", "monitoring"}:
            raise ValueError("order must contain each region exactly once")
        self.env = env
        self.view = _PlanningView(env)
        self.base = SkylandsCampaignPolicy(self.view)
        self.route = route
        self.order = tuple(order)
        self.prepared = env.objective == "extended" if prepared is None else prepared
        self.plan = []
        if self.prepared:
            self.plan = [f"B{i:02}" for i in range(1, 12)]
            if route == "synchronization":
                self.plan.insert(self.plan.index("B09"), "record_0")
                self.plan.append("record_2")
            branches = {"cooling": ["B12", "B13", "B14", "B15"],
                        "anchors": ["B16", "B17"], "monitoring": ["B18", "B19"]}
            for branch in order:
                self.plan.extend(branches[branch])
                if branch == "cooling" and route == "synchronization":
                    self.plan.insert(self.plan.index("B13"), "record_1")
            self.plan.extend(f"B{i:02}" for i in range(20, 25))
        elif route == "synchronization":
            self.plan.extend(("record_0", "record_1", "record_2"))
        self.plan.append("plate")
        self.plan.extend(("grounding_canal", "grounding_socket") if route == "grounding" else ("lock_0", "lock_1", "lock_2"))
        self.plan.extend(("stop", "stabilize"))
        self.index = 0
        self.stage = self.plan[0]
        self.explanation = "Begin the objective-driven expedition"
        self.stage_steps = 0

    def _complete(self, name):
        e = self.env
        if name.startswith("B"):
            return e.by_id[name].state["done"]
        if name.startswith("record_"):
            return name in e.facts_seen
        if name == "plate":
            return "machine_plate" in e.facts_seen
        if name == "grounding_canal":
            c, r = e.cargo["grounding_cartridge"], e.rafts["T01"]
            return not c["parent"] and np.linalg.norm(c["pos"]-r["receiver"]) < 1
        if name == "grounding_socket":
            return e.grounding_live()
        if name.startswith("lock_"):
            return e.locks[int(name[-1])]
        if name == "stop":
            return e.engine_disabled
        return e.success

    def __call__(self, obs):
        e = self.env
        while self.index < len(self.plan)-1 and self._complete(self.plan[self.index]):
            self.index += 1
            self.base.path = []
            self.base.key = None
            self.stage_steps = 0
        self.stage = name = self.plan[self.index]
        self.stage_steps += 1
        if self.stage_steps > 25000:
            raise RuntimeError(f"Reference stalled at {name}: {e.agent.pos}; {self.explanation}")
        if name.startswith("B"):
            self.view.site = name
            action = self.base(obs)
        elif name.startswith("record_") or name == "plate":
            action = self.base.approach(e.engine_targets[name], "inspect")
        elif name == "grounding_canal":
            action = self.base.raft(e.grounding_site, "T01")
        elif name == "grounding_socket":
            cid = "grounding_cartridge"
            action = self.base.transport_cargo(cid, e.grounding_socket, e.engine_targets["socket"])
            if action is None:
                action = self.base.approach(e.engine_targets["socket"])
        elif name.startswith("lock_"):
            tid = e.engine_targets[name]
            i = int(name[-1])
            if not e._reachable(e.targets[tid]["pos"], 2.8):
                action = self.base.approach(tid, "inspect")
            elif e.engine_phase() != "venting":
                action = neutral()
                self.base.explanation = "Observe the shutters from the sheltered service bay"
            else:
                action = self.base.command("attack" if self.route == "capacitors" else "use", tid, e.lock_settings[i])
        elif name == "stop":
            action = self.base.approach(e.engine_targets["stop"])
        else:
            action = neutral()
            self.base.explanation = "Verify sustained shutdown and live regional service"
        self.explanation = self.base.explanation
        return action
