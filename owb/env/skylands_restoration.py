"""Tempest restoration: delayed-use tooling, calibrated services and relief power.

The 32 checkpoints are private diagnostics, never an ordered success checklist.
All new fixtures occupy existing service floors; terrain is not rewritten.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json

import numpy as np

from .skylands_tempest import GOALS as BASE_GOALS, REGIONS, SkylandsTempestEnv
from .skylands_reference import plain
from ..world.tempest import identity


SUITE = "skylands-restoration-v1"
GOALS = {
    "open": "Find and stop the runaway Tempest Engine. Regional repairs are optional.",
    "extended": (
        "Restore the cooling, anchor and sky monitoring services, fit a passive pressure manifold "
        "at the Tempest Engine, stop the engine, and deliver relief and stable power to the "
        "refuge commissioned by the landing office."
    ),
}
REGION_RECORDS = {"cooling": "record_0", "anchors": "record_1", "monitoring": "record_2"}
EXTRA_CHECKPOINTS = {
    "R01": "Recover the original manifold tooling",
    "R02": "Balance the cooling flow",
    "R03": "Tune the anchor load distribution",
    "R04": "Synchronize the monitoring clock",
    "R05": "Fabricate the pressure manifold",
    "R06": "Transport and install the original manifold",
    "R07": "Sustain engine shutdown",
    "R08": "Commission stable refuge power",
}


class SkylandsRestorationEnv(SkylandsTempestEnv):
    """Native Gymnasium task; inherited RGB sensors and separate Cycles export."""

    def __init__(self, objective="extended", seed=7, *, render_enabled=False, max_steps=180000):
        super().__init__(objective, seed, render_enabled=render_enabled, max_steps=max_steps)

    def _reset_mechanics(self):
        super()._reset_mechanics()
        rng = np.random.default_rng(identity(self._seed, SUITE+":calibration"))
        self.service_profiles = {name: [int(v) for v in rng.integers(0, 4, 3)]
                                 for name in REGION_RECORDS}
        # Unset is a real disconnected selector, distinct from setting zero.
        self.service_dials = {name: [-1, -1, -1] for name in REGION_RECORDS}
        self.tool_location = "depot"
        self.manifold_manufactured = False
        self.refuge_switches = [False, False]
        self.refuge_stable_seconds = 0.
        self.restoration_completed = []
        self.restoration_targets = {}
        depot = self.by_id["B05"]
        self._fixture("tooling", "tooling_case", depot.pos+(1, 7, 0))
        for mid, name in REGIONS.items():
            control = self.targets[self.by_id[mid].state["control"]]["pos"]
            for axis in range(3):
                self._fixture(f"{name}_{axis}", f"{name}_dial_{axis}", control+((axis-1)*1., 0, -1),
                              region=name, axis=axis)
        forge = self.grounding_site.pos
        self._fixture("forge", "manifold_workbench", forge+(0, 1, -4))
        self._add_cargo("pressure_manifold", forge+(-3, 0, -6), 280, anchored=True)
        self.manifold_socket = self.engine_pos+(-16, 0, 9)
        self._fixture("socket", "pressure_manifold_socket", self.manifold_socket+(0, 1, 0))
        for i, p in enumerate(self.refuge_receivers):
            self._fixture(f"refuge_{i}", f"refuge_{i}_power_transfer", p+(1, 1, 0), refuge=i)

    def _fixture(self, key, role, pos, **extra):
        self.restoration_targets[key] = self._target("restoration", role, pos, **extra)

    def live_regions(self):
        raw = super().live_regions()
        return {name: live and getattr(self, "service_dials", {}).get(name) ==
                getattr(self, "service_profiles", {}).get(name, "unset")
                for name, live in raw.items()}

    def manifold_live(self):
        c = self.cargo.get("pressure_manifold")
        return bool(c and self.manifold_manufactured and c["intact"] and c["anchored"]
                    and c["parent"] == "pressure_bus"
                    and np.linalg.norm(c["pos"]-self.manifold_socket) < .1)

    def relief_live(self):
        c = self.cargo["relief"]
        return bool(c["intact"] and c["anchored"] and c["parent"] == "refuge"
                    and c.get("delivered_to") == self.destination
                    and np.linalg.norm(c["pos"]-self.refuge_receivers[self.destination]) < .1)

    def refuge_power_available(self):
        return (self.engine_disabled and self.shutdown_seconds >= 5 and self.manifold_live()
                and all(self.live_regions().values()) and self.relief_live()
                and self.refuge_switches[self.destination])

    def system_operational(self):
        if not super().system_operational():
            return False
        return self.objective == "open" or (self.refuge_power_available() and self.refuge_stable_seconds >= 30.)

    def clue(self, m):
        text = super().clue(m)
        if m.id == "B01":
            text += (" The upper expedition depot holds the original manifold tooling. "
                     "The sky salvage workbench needs it and three calibrated services. "
                     "Deliver its 280 kg manifold to the engine pressure socket, then commission refuge power.")
        if m.id == "B08":
            text += (" Service profiles: cooling at this weather archive; anchor loads at the deep intake; "
                     "monitoring clocks at the quarry yard. Inspect their service records.")
        if m.id in REGIONS:
            name = REGIONS[m.id]
            source = {"cooling": "weather archive", "anchors": "deep pump intake", "monitoring": "quarry charging yard"}[name]
            text += f" Set all three {name} dials from the profile recorded at the {source}."
        return text

    def _dispatch(self, command, ingredients, action):
        tid, option = int(action["target"]), int(action["option"])
        obj = self.targets.get(tid)
        # The same assembled craft can carry either original payload. Selection
        # happens through a local action, not a reference-policy state write.
        if obj and obj["owner"] == "T01" and obj["role"] == "raft" and command == "load":
            if not self._reachable(self.rafts["T01"]["pos"]+(0, 1, 0), 6):
                return self._advance(self._fail("load from the visible craft"))
            if option not in (0, 1):
                return self._advance(self._fail("payload 0: grounding cartridge; payload 1: pressure manifold"))
            self.rafts["T01"]["cargo_id"] = "pressure_manifold" if option == 1 else "grounding_cartridge"
        if obj and obj["role"] == "cache" and command in ("cache", "retrieve") and option == 2:
            if not self._reachable(obj["pos"], 4.5):
                return self._advance(self._fail("tooling locker must be visible and within reach"))
            location = f"locker:{obj['option']}"
            if command == "cache" and self.tool_location == "carried":
                self.tool_location = location
                self.satchel["manifold_tooling"] = 0
            elif command == "retrieve" and self.tool_location == location:
                self.tool_location = "carried"
                self.satchel["manifold_tooling"] = 1
            else:
                return self._advance(self._fail("original tooling is not available here"))
            self._message("Original tooling stored." if command == "cache" else "Original tooling retrieved.")
            return self._advance([dict(type="tooling_moved", operation=command, locker=obj["option"])])
        if obj and obj["owner"] == "restoration":
            if not self._reachable(obj["pos"], 4.5):
                return self._advance(self._fail("fixture must be visible and within reach"))
            if command == "inspect":
                text = self._fixture_text(obj)
                self._message(text)
                return self._advance([dict(type="clue_exposed", site=obj["role"], text=text)])
            events = self._operate_fixture(obj, command, option)
            return self._advance(events)
        events = super()._dispatch(command, ingredients, action)
        if obj and command == "inspect" and any(e["type"] == "clue_exposed" for e in events):
            extra = ""
            for name, record in REGION_RECORDS.items():
                if tid == self.engine_targets[record]:
                    profile = ", ".join(map(str, self.service_profiles[name]))
                    extra += f" {name.capitalize()} service profile: dials 0, 1, 2 = {profile}."
            if obj["role"] == "cache":
                present = self.tool_location == f"locker:{obj['option']}"
                extra += f" Original tooling {'present' if present else 'absent'}. Option 2 stores or retrieves tooling."
            if obj["owner"] == "T01" and obj["role"] == "raft":
                extra += " Load option 0 selects the grounding cartridge; option 1 selects the 280 kg pressure manifold."
            if extra:
                self._message(self.last_text+extra)
                for event in events:
                    if event["type"] == "clue_exposed":
                        event["text"] = self.last_text
        return events

    def _fixture_text(self, obj):
        role = obj["role"]
        if "region" in obj:
            name = obj["region"]
            source = {"cooling": "weather archive", "anchors": "deep pump intake", "monitoring": "quarry charging yard"}[name]
            return (f"{name} dial {obj['axis']}: {self.service_dials[name][obj['axis']]}. "
                    f"Use option 0-3 to set it. Profile at the {source}; "
                    f"local service {'balanced' if self.live_regions()[name] else 'unbalanced or offline'}.")
        if role == "tooling_case":
            return (f"Original manifold tooling {'available' if self.tool_location == 'depot' else 'removed'}. "
                    "Take to carry; reserve lockers accept cache/retrieve option 2. Needed at the sky salvage workbench.")
        if role == "manifold_workbench":
            return ("Fabricate a 280 kg pressure manifold using the original tooling and three balanced regional services. "
                    "Use to fabricate; transport on the salvage craft and fit at the engine pressure socket.")
        if role == "pressure_manifold_socket":
            return f"Pressure bus {'connected' if self.manifold_live() else 'disconnected'}. Bring the original fabricated manifold; use to install."
        return (f"Refuge {obj['refuge']} power transfer {'armed' if self.refuge_switches[obj['refuge']] else 'open'}. "
                "Use to arm; cancel to open. Stable power requires the original relief load, balanced services, "
                "the engine pressure manifold and sustained shutdown. Commission for 30 seconds.")

    def _operate_fixture(self, obj, command, option):
        role = obj["role"]
        if role == "tooling_case" and command in ("take", "use"):
            if self.tool_location != "depot":
                return self._fail("the original tooling has already been removed")
            self.tool_location = "carried"
            self.satchel["manifold_tooling"] = 1
            return [dict(type="tooling_recovered")]
        if "region" in obj and command == "use":
            name, axis = obj["region"], obj["axis"]
            self.service_dials[name][axis] = option
            self._message(f"{name} dial {axis} set to {option}; service {'balanced' if self.live_regions()[name] else 'unbalanced or offline'}.")
            return [dict(type="service_dial_set", region=name, axis=axis, value=option)]
        if role == "manifold_workbench" and command == "use":
            if self.manifold_manufactured:
                return self._fail("the original manifold has already been fabricated")
            if self.tool_location != "carried" or not all(self.live_regions().values()):
                return self._fail("bring original tooling and balance all three regional services")
            self.tool_location = "forge"
            self.satchel["manifold_tooling"] = 0
            self.manifold_manufactured = True
            self.cargo["pressure_manifold"]["anchored"] = False
            return [dict(type="manifold_fabricated", mass=280)]
        if role == "pressure_manifold_socket" and command == "use":
            c = self.cargo["pressure_manifold"]
            if (not self.manifold_manufactured or not c["intact"] or c["parent"]
                    or np.linalg.norm(c["pos"]-self.manifold_socket) > 4):
                return self._fail("bring the original intact fabricated manifold to this socket")
            if self.hauling == "pressure_manifold":
                self.hauling = None
            c.update(parent="pressure_bus", anchored=True, pos=self.manifold_socket.copy())
            return [dict(type="manifold_installed")]
        if "refuge" in obj and command in ("use", "cancel"):
            self.refuge_switches[obj["refuge"]] = command == "use"
            return [dict(type="refuge_transfer_set", refuge=obj["refuge"], armed=command == "use")]
        return self._fail("inspect the fixture for its local controls")

    def _task_outcome(self, events):
        super()._task_outcome(events)
        self.refuge_stable_seconds = self.refuge_stable_seconds+.2 if self.refuge_power_available() else 0.
        live = self.live_regions()
        states = [self.tool_location != "depot", live["cooling"], live["anchors"], live["monitoring"],
                  self.manifold_manufactured, self.manifold_live(), self.engine_disabled and self.shutdown_seconds >= 5,
                  self.refuge_power_available() and self.refuge_stable_seconds >= 30.]
        done = {row["id"] for row in self.restoration_completed}
        for key, active in zip(EXTRA_CHECKPOINTS, states):
            if active and key not in done:
                self.restoration_completed.append(dict(id=key, step=self.steps, seconds=round(self.elapsed, 6)))
        self.success = bool(self.agent.alive and self.system_operational())
        return self.success, 10. if self.success else 0.

    def _observation(self):
        obs = super()._observation()
        obs["text"] = obs["text"].replace(BASE_GOALS[self.objective], GOALS[self.objective], 1)[:4096]
        return obs

    def _info(self, events):
        info = super()._info(events)
        visible = {"tooling_recovered", "tooling_moved", "service_dial_set", "manifold_fabricated",
                   "manifold_installed", "refuge_transfer_set"}
        info["events"].extend(e for e in events if e["type"] in visible)
        return info

    def evaluation(self):
        result = super().evaluation()
        result.update(suite=SUITE, diagnostic_checkpoint_count=32,
                      campaign_milestones=deepcopy(self.completed+self.restoration_completed),
                      tooling_location=self.tool_location, manifold_live=self.manifold_live(),
                      refuge_power_available=self.refuge_power_available(),
                      refuge_stable_seconds=self.refuge_stable_seconds)
        return result

    def state_digest(self):
        state = dict(base=super().state_digest(), suite=SUITE, profiles=self.service_profiles,
                     dials=self.service_dials, tooling=self.tool_location, manufactured=self.manifold_manufactured,
                     switches=self.refuge_switches, stable=self.refuge_stable_seconds,
                     completed=self.restoration_completed)
        return hashlib.sha256(json.dumps(plain(state), sort_keys=True).encode()).hexdigest()

    def presentation_meshes(self, batch):
        super().presentation_meshes(batch)
        from ..render.skylands_restoration import restoration_meshes
        restoration_meshes(self, batch)


def make_open(**kwargs):
    return SkylandsRestorationEnv(objective="open", **kwargs)
