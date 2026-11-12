"""Action-only restoration reference, with policy-owned service-profile memory."""
from __future__ import annotations

import re
import numpy as np

from .skylands_tempest import SkylandsTempestPolicy
from .skylands_reference import neutral


class SkylandsRestorationPolicy(SkylandsTempestPolicy):
    def __init__(self, env, *, route="synchronization", order=("monitoring", "anchors", "cooling"), prepared=None):
        super().__init__(env, route=route, order=order, prepared=prepared)
        self.profiles = {}
        self.tool_locker = self.base.bindings["lock_coupling"]
        if self.prepared:
            preparation = self.plan[:self.plan.index("plate")]
            # All shutdown strategies need profiles for the restoration services.
            for record, after in (("record_0", "B08"), ("record_1", "B12"), ("record_2", "B11")):
                if record not in preparation:
                    preparation.insert(preparation.index(after)+1, record)
            preparation.insert(preparation.index("B05")+1, "tooling")
            preparation.insert(preparation.index("B09")+1, "cache_tooling")
            # Equipment repair and calibration are distinct. Revisit each works
            # after all records have been encountered, in the requested order.
            i = preparation.index("B20")
            preparation[i:i] = [f"calibrate_{name}_{axis}" for name in order for axis in range(3)]
            finale = self.plan[self.plan.index("plate"):-1]
            self.plan = preparation+["retrieve_tooling", "forge", "manifold_canal", "manifold_socket"]+finale+["refuge_power", "stabilize"]
            self.stage = self.plan[0]

    def _complete(self, name):
        e = self.env
        if name == "tooling":
            return e.tool_location != "depot"
        if name == "cache_tooling":
            return e.tool_location == f"locker:{self.tool_locker}"
        if name == "retrieve_tooling":
            return e.tool_location == "carried"
        if name.startswith("calibrate_"):
            _, region, axis = name.split("_")
            return region in self.profiles and e.service_dials[region][int(axis)] == self.profiles[region][int(axis)]
        if name == "forge":
            return e.manifold_manufactured
        if name == "manifold_canal":
            c, r = e.cargo["pressure_manifold"], e.rafts["T01"]
            return not c["parent"] and np.linalg.norm(c["pos"]-r["receiver"]) < 1
        if name == "manifold_socket":
            return e.manifold_live()
        if name == "refuge_power":
            return e.refuge_switches[e.destination]
        return super()._complete(name)

    def _ferry(self, cid, option):
        e, b = self.env, self.base
        r = e.rafts["T01"]
        if any(pid is None for pid in r["parts"]):
            return b.raft(e.grounding_site, "T01")
        if r["cargo"] is None:
            # Reuse the same craft for the second payload on grounding routes.
            if np.linalg.norm(r["pos"]-r["origin_pos"]) > .1:
                if not e.on_deck(r):
                    return b.go(r["pos"]+(.5, .01, .5))
                delta = r["origin_pos"]-r["pos"]
                axis = 0 if abs(delta[0]) > .1 else 2
                v = np.zeros(3); v[axis] = np.sign(delta[axis])
                return b.drive(r, v)
            if np.any(r["drive_vector"]):
                return b.command("drive", r["target"], 2)
            action = b.transport_cargo(cid, r["origin_pos"]+(-2.5, 0, 0), r["target"])
            if action is not None:
                return action
            return b.approach(r["target"], "load", option, 4.8)
        if r["cargo"] != cid:
            raise RuntimeError("Unexpected payload in salvage craft")
        return b.raft(e.grounding_site, "T01")

    def __call__(self, obs):
        for region, a, b, c in re.findall(
                r"(Cooling|Anchors|Monitoring) service profile: dials 0, 1, 2 = (\d), (\d), (\d)", obs["text"]):
            self.profiles[region.lower()] = [int(a), int(b), int(c)]
        while self.index < len(self.plan)-1 and self._complete(self.plan[self.index]):
            self.index += 1
            self.base.path = []
            self.base.key = None
            self.stage_steps = 0
        name = self.plan[self.index]
        custom = {"tooling", "cache_tooling", "retrieve_tooling", "forge", "manifold_canal",
                  "manifold_socket", "refuge_power", "grounding_canal"}
        if name not in custom and not name.startswith("calibrate_"):
            return super().__call__(obs)
        self.stage = name
        self.stage_steps += 1
        if self.stage_steps > 25000:
            raise RuntimeError(f"Restoration reference stalled at {name}: {self.env.agent.pos}")
        e, b = self.env, self.base
        if name == "tooling":
            action = b.approach(e.restoration_targets["tooling"], "take")
        elif name in ("cache_tooling", "retrieve_tooling"):
            tid = next(t for t, obj in e.targets.items() if obj["role"] == "cache" and obj["option"] == self.tool_locker)
            action = b.approach(tid, "cache" if name == "cache_tooling" else "retrieve", 2)
        elif name.startswith("calibrate_"):
            _, region, axis = name.split("_")
            if region not in self.profiles:
                raise RuntimeError(f"Reference has not read the {region} profile")
            # The monitoring console is on the original upper service deck.
            # Its historical 'landed' flag is not a substitute for returning
            # physically to the launch and ascending again on a later visit.
            launch = np.asarray(e.by_id["B19"].state["launch"])
            if region == "monitoring" and e.agent.pos[1] < launch[1]+4:
                if b.near(launch, .6):
                    action = neutral()
                    action["command"] = e.commands.index("ascend")
                    b.explanation = "Ascend through the monitoring service deck again"
                else:
                    action = b.go(launch)
            else:
                action = b.approach(e.restoration_targets[f"{region}_{axis}"], "use", self.profiles[region][int(axis)])
        elif name == "forge":
            action = b.approach(e.restoration_targets["forge"])
        elif name in ("manifold_canal", "grounding_canal"):
            action = self._ferry("pressure_manifold" if name == "manifold_canal" else "grounding_cartridge",
                                 1 if name == "manifold_canal" else 0)
        elif name == "manifold_socket":
            tid = e.restoration_targets["socket"]
            action = b.transport_cargo("pressure_manifold", e.manifold_socket, tid)
            if action is None:
                action = b.approach(tid)
        elif name == "refuge_power":
            action = b.approach(e.restoration_targets[f"refuge_{e.destination}"])
        else:
            action = neutral()
        self.explanation = b.explanation
        return action
