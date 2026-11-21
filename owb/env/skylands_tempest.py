"""An objective-driven Skylands campaign with optional preparation and a finale.

The inherited logistics mechanisms remain physical and reversible. Milestones
are evaluator telemetry, never an ordered task list or a success prerequisite.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import string

import numpy as np
from gymnasium import spaces

from .skylands_campaign import SkylandsCampaignEnv
from .skylands_reference import Site, COMMANDS, plain
from .skylands import SkylandsEnv
from .wildlands import WildlandsEnv
from ..core import blocks as B
from ..world.tempest import author_engine, author_settlements, identity, install_building


GOALS = {
    "open": "A runaway atmospheric machine is making Aster Reach uninhabitable. Find it and stop it.",
    "extended": ("Stop the runaway atmospheric machine. Restore the cooling works, anchor works and "
                 "sky monitoring station, and deliver relief to the refuge identified in the landing office."),
}
REGIONS = {"B15": "cooling", "B17": "anchors", "B19": "monitoring"}
TITLES = {
    "B01": "Landing office", "B02": "Shipwright's load trial", "B03": "Supply inlet",
    "B04": "Maintenance loft", "B05": "Expedition depot", "B06": "Recirculating service platform",
    "B07": "Survey pack store", "B08": "Weather archive", "B09": "Regional reserve lockers",
    "B10": "Abandoned anchor foundry", "B11": "Quarry charging yard", "B12": "Deep pump intake",
    "B13": "Flooded cooling gallery", "B14": "Regulator mezzanine", "B15": "Cooling works",
    "B16": "Reed-channel freight", "B17": "Anchor works", "B18": "Observatory shuttle",
    "B19": "Sky monitoring station", "B20": "Damaged relief approach", "B21": "Relief drive housing",
    "B22": "Lock service platform", "B23": "Relief berth", "B24": "Refuge receiving dock",
}


class SkylandsTempestEnv(SkylandsCampaignEnv):
    """Full authored world. ``render_enabled=False`` is for headless logic only."""
    # Published traces predate the organic campaign world and its new mechanics.
    organic_world=False
    world_features=frozenset()
    commands=COMMANDS

    presentation_kind = "skylands_tempest"

    def __init__(self, objective="open", seed=7, *, render_enabled=False, max_steps=160000):
        if objective not in GOALS:
            raise ValueError("objective must be open or extended")
        self.objective = objective
        super().__init__("b", seed, render_enabled=render_enabled)
        self.cfg.max_steps = max_steps
        self.cfg.time_penalty = 0.
        self.action_space["target"] = spaces.Discrete(1024)
        self.observation_space["text"] = spaces.Text(max_length=4096, charset=string.printable)
        # Local actionable objects, not a catalogue of unseen world entities.
        self.observation_space["nearby"] = spaces.Box(-np.inf, np.inf, (32, 5), np.float32)
        self.action_history = []

    def _reset_mechanics(self):
        super()._reset_mechanics()
        for m in self.milestones:
            m.title = TITLES[m.id]
        self.engine_disabled = False
        self.engine_route = None
        self.locks = [False]*3
        self.capacitors = [3]*3
        rules = np.random.default_rng(identity(self._seed, "operating-rules"))
        self.lock_settings = [int(v) for v in rules.integers(0, 4, size=3)]
        self.cycle_offset = int(rules.integers(0, 80))
        self.last_strike = -20
        self.cue_until = -1
        self.last_text = ""
        self.action_history = []
        self._last_action_target = 0
        self.engine_events = []
        self.facts_seen = {}
        self.shutdown_seconds = 0.
        # Author new terrain before architecture placement and trace collection.
        forge = author_engine(self)
        author_settlements(self)
        fx, y, fz = forge
        self.grounding_site = Site("T01", "Sky salvage canal", "raft", 2, fx, fz, elevation=y)
        self._add_raft(self.grounding_site, "T01")
        r = self.rafts["T01"]
        r["cargo_id"] = "grounding_cartridge"
        self._add_cargo("grounding_cartridge", np.array(forge)+(-3, 0, -3), 240)
        # The receiver joins the engine service apron on solid ground.
        rx, _, rz = r["receiver"].astype(int)
        self._pad(self.world, (rx, y, rz), radius=3)
        self._road(self.world, 2, [(rx, rz), (int(self.engine_pos[0])-16, rz)])
        install_building(self, "sky-salvage-foundry", "foundry", (fx-45, y, fz-20))
        install_building(self, "engine-service-gantry", "gantry", (int(self.engine_pos[0])+43, y, fz))
        self.engine_targets = {}
        for i, p in enumerate(self.lock_positions):
            self.engine_targets[f"lock_{i}"] = self._target("tempest", "phase_lock", p, circuit=i)
        self.grounding_socket = self.engine_pos+(-13, 0, 11)
        self.engine_targets["socket"] = self._target("tempest", "grounding_socket", self.grounding_socket+(0, 1, 0))
        self.engine_targets["stop"] = self._target("tempest", "shutdown", self.engine_pos+(0, 1, -15))
        self.engine_targets["plate"] = self._target("tempest", "machine_plate", self.engine_pos+(-27, 1, -29))
        for i, mid in enumerate(("B08", "B12", "B11")):
            m = self.by_id[mid]
            self.engine_targets[f"record_{i}"] = self._target("tempest", "archive_record", m.pos+(3, 1, -3), circuit=i)
        self.grounding_installed = False
        self.refuge_targets = [self._target("refuge", "refuge_receiver", p+(0, 1, 0), refuge=i)
                               for i, p in enumerate(self.refuge_receivers)]
        self.world.meta["tempest_architecture"] = deepcopy(self.architecture)
        self.world.recompute_heightmaps()

    def site_specs(self):
        return super().site_specs()

    def live_regions(self):
        """Persistent completion marks cannot stand in for operating equipment."""
        out = {}
        for mid, name in REGIONS.items():
            s = self.by_id[mid].state
            live = s["installed"] and s["stable"] >= 5
            if mid == "B19":
                live = live and s.get("station_part") == "spindle"
            else:
                c = self.cargo["power_module" if mid == "B15" else "marsh_coupling"]
                live = live and c["intact"] and c["anchored"] and c["parent"] == mid
                live = live and np.linalg.norm(c["pos"]-self.by_id[mid].pos) < 6
            out[name] = bool(live)
        return out

    def engine_phase(self):
        if self.engine_disabled:
            return "stopped"
        t = (self.steps+self.cycle_offset) % 80
        if t < 20:
            return "warning"
        if t < 35:
            return "discharge"
        # Repairing the cooling works physically extends the service interval.
        return "venting" if t >= (35 if self.live_regions()["cooling"] else 60) else "pressurized"

    def clue(self, m):
        if m.id == "B01":
            return (f"Refuge {self.destination} requests relief. The weather archive is at the sky lookout. "
                    "Cooling, anchor and monitoring crews have left repair records at their works.")
        if m.id == "B08":
            x, _, z = self.engine_pos
            return (f"The Tempest Engine occupies the long industrial island southeast of this station, near {int(x)}, {int(z)}. "
                    "Cooling extends its vent interval; anchors suppress discharges; monitoring broadcasts its phase. "
                    "The sky salvage canal holds a heavy grounding cartridge.")
        if m.id == "B20":
            return "The west maintenance bridge has closed. The eastern service platform and upper gallery remain accessible."
        if m.kind in ("raft", "stage"):
            return "Hull and cargo ratings are stamped on the equipment. Mounted parts and load must remain supported; the cell supplies propulsion."
        if m.kind == "reverse":
            return "This service platform records only its recent motion. Its visible return path can be reversed while aboard."
        return m.title+". Inspect the local mechanisms and their physical connections."

    def _message(self, text):
        self.last_text = text
        self.cue_until = self.steps+30

    def _dispatch(self, command, ingredients, action):
        tid = int(action["target"])
        self._last_action_target = tid
        obj = self.targets.get(tid)
        if obj and obj["role"] == "refuge_receiver" and command in ("inspect", "use"):
            if not self._reachable(obj["pos"], 4.5):
                return self._advance(self._fail("receiver must be visible and within reach"))
            refuge = obj["refuge"]
            if command == "inspect":
                text = f"Refuge {refuge}. Receiving pad for the original relief shipment."
                self._message(text)
                return self._advance([dict(type="clue_exposed", site="refuge_receiver", text=text)])
            c = self.cargo["relief"]
            if refuge != self.destination:
                return self._advance(self._fail("this refuge is not the commissioned destination"))
            if c["parent"] or not c["intact"] or np.linalg.norm(c["pos"]-self.refuge_receivers[refuge]) > 4:
                return self._advance(self._fail("bring the original intact relief shipment to the receiving pad"))
            if self.hauling == "relief":
                self.hauling = None
            c.update(parent="refuge", anchored=True, delivered_to=refuge, pos=self.refuge_receivers[refuge].copy())
            self.by_id["B24"].state["done"] = True
            return self._advance([dict(type="cargo_delivered", cargo="relief", receiver=refuge)])
        if obj and obj["owner"] == "tempest" and command in ("inspect", "use", "attack"):
            if not self._reachable(obj["pos"], 4.5):
                return self._advance(self._fail("target must be visible and within reach"))
            role = obj["role"]
            events = []
            if command == "inspect":
                self.facts_seen[role if role != "archive_record" else f"record_{obj['circuit']}"] = self.steps
                if role == "archive_record":
                    i = obj["circuit"]
                    self.facts_seen[f"circuit_{i}"] = self.steps
                    text = f"Circuit {i}: service selector {self.lock_settings[i]}. Isolate only while the vent shutters are open."
                elif role == "machine_plate":
                    text = ("Three independent circuits feed the engine. Service selectors are recorded at the weather archive, "
                            "deep pump intake and quarry yard. Alternatively, dock the intact heavy grounding cartridge. "
                            "Exposed capacitors can also be broken during venting. Overhead service shelters block discharge.")
                elif role == "phase_lock":
                    i = obj["circuit"]
                    text = f"Circuit {i}; isolated {self.locks[i]}; capacitor integrity {self.capacitors[i]}/3; shutters {self.engine_phase()}."
                else:
                    text = "Ground the engine or isolate its three circuits before engaging the shutdown lever."
                self._message(text)
                events.append(dict(type="clue_exposed", site=role, text=text))
            elif self.engine_disabled:
                events = self._fail("engine is already stopped")
            elif role == "phase_lock":
                i = obj["circuit"]
                if self.engine_phase() != "venting":
                    events = self._fail("shutters closed; wait for venting")
                elif command == "attack":
                    if self.steps-self.last_strike < 4:
                        events = self._fail("tool recovering")
                    elif self.capacitors[i] > 0:
                        self.last_strike = self.steps
                        self.capacitors[i] -= 1
                        self.locks[i] = self.locks[i] or self.capacitors[i] == 0
                        events.append(dict(type="capacitor_struck", circuit=i, integrity=self.capacitors[i]))
                elif int(action["option"]) != self.lock_settings[i]:
                    self.agent.hurt(1., "selector_fault")
                    events = self._fail("selector fault; circuit remains energized")
                else:
                    self.locks[i] = True
                    events.append(dict(type="circuit_isolated", circuit=i))
            elif role == "grounding_socket" and command == "use":
                c = self.cargo["grounding_cartridge"]
                if not c["intact"] or c["parent"] or np.linalg.norm(c["pos"]-self.grounding_socket) > 4:
                    events = self._fail("bring the original intact cartridge to the socket")
                else:
                    self.hauling = None
                    c.update(parent="tempest_ground", anchored=True, pos=self.grounding_socket.copy())
                    self.grounding_installed = True
                    events.append(dict(type="grounding_installed", cargo="grounding_cartridge"))
            elif role == "shutdown" and command == "use":
                if not (all(self.locks) or self.grounding_live()):
                    events = self._fail("live circuits prevent shutdown")
                else:
                    self.engine_disabled = True
                    self.engine_route = "grounding" if self.grounding_live() else "capacitors" if not any(self.capacitors) else "synchronization"
                    events.append(dict(type="engine_stopped", method=self.engine_route))
            else:
                events = self._fail("inspect this local record")
            return self._advance(events)
        previous = self.last_text
        events = super()._dispatch(command, ingredients, action)
        if self.last_text != previous or any(e["type"] == "clue_exposed" for e in events):
            self.cue_until = self.steps+30
        return events

    def _operate(self, m, role, option):
        # In this task, physically reaching an alternative entrance is valid.
        # The guided course's demonstration-specific checks are not objectives.
        if m.kind in ("ascend", "ascend_station", "compose", "hoist") and role == "control":
            if self._reachable(self.targets[m.state["control"]]["pos"]):
                m.state["landed"] = True
        if m.kind == "reverse":
            mv = self.movers[m.state["mover"]]
            if m.id == "B18" and not self.satchel["sky_cell"]:
                return self._fail("shuttle equipment requires its charged service cell")
            if m.id == "B22" and not all(self.live_regions().values()):
                return self._fail("relief lock requires the regional network")
            m.state["done"] = True
            if m.id == "B07":
                self.satchel["survey"] = 1
            if m.id == "B18":
                self.satchel["spindle"] = 1
            return [dict(type="platform_landing_reached", site=m.id, reversed_distance=mv["reversed_distance"])]
        return super()._operate(m, role, option)

    def grounding_live(self):
        c = self.cargo["grounding_cartridge"]
        return (self.grounding_installed and c["intact"] and c["anchored"] and c["parent"] == "tempest_ground"
                and np.linalg.norm(c["pos"]-self.grounding_socket) < .1)

    def _task_outcome(self, events):
        # Retain inherited physical updates and private milestone telemetry.
        super()._task_outcome(events)
        # Cargo IDs remain attached to the actual load after drive/unload actions.
        # The reference course updated these only during selected dispatch paths.
        for c in self.cargo.values():
            self.targets[c["target"]]["pos"] = c["pos"]+(0, 1, 0)
        for r in self.rafts.values():
            self.targets[r["target"]]["pos"] = r["pos"]+(0, 1, 0)
        for obj in self.targets.values():
            if "mover" in obj:
                obj["pos"] = self.movers[obj["mover"]]["pos"]+(0, 1, 0)
        if not self.engine_disabled and self.engine_phase() == "discharge":
            delta = self.agent.pos-self.engine_pos
            if abs(delta[1]) < 4 and np.linalg.norm(delta[[0, 2]]) < 32:
                roof = self.world.raycast(self.agent.eye, (0, 1, 0), 6)
                if not roof.hit:
                    damage = .08 if self.live_regions()["anchors"] else .45
                    self.agent.hurt(damage, "tempest_discharge")
                    events.append(dict(type="storm_hit", damage=damage))
        self.shutdown_seconds = self.shutdown_seconds+.2 if self.engine_disabled else 0.
        self.success = bool(self.agent.alive and self.system_operational())
        self.engine_events.extend(dict(step=self.steps, **e) for e in events if e["type"] in
                                  ("circuit_isolated", "capacitor_struck", "grounding_installed", "engine_stopped", "storm_hit"))
        return self.success, 10. if self.success else 0.

    def system_operational(self):
        if not getattr(self, "engine_disabled", False) or self.shutdown_seconds < 5:
            return False
        if self.objective == "open":
            return True
        c = self.cargo["relief"]
        return (all(self.live_regions().values()) and c["intact"] and c["anchored"]
                and c["parent"] == "refuge" and c.get("delivered_to") == self.destination
                and np.linalg.norm(c["pos"]-self.refuge_receivers[self.destination]) < .1)

    def _local_targets(self):
        visible = []
        for tid, obj in self.targets.items():
            d = np.linalg.norm(obj["pos"]-self.agent.eye)
            if d <= 18 and self.world.line_of_sight(self.agent.eye, obj["pos"]):
                visible.append((d, tid, obj))
        return sorted(visible, key=lambda row: (row[0], row[1]))[:32]

    def _observation(self):
        # Bypass the guided-course text and its current-milestone disclosure.
        obs = SkylandsEnv._observation(self)
        rows = self._local_targets()
        obs["nearby"] = np.zeros((32, 5), np.float32)
        descriptions = []
        for i, (d, tid, obj) in enumerate(rows):
            obs["nearby"][i] = [1, tid, *(obj["pos"]-self.agent.pos)]
            label = obj["role"]
            if label in ("control", "lower") and obj["owner"] in self.by_id:
                label += ":"+self.by_id[obj["owner"]].title
            if label == "part":
                label += ":"+self.parts[obj["part"]]["kind"]
            if "circuit" in obj:
                label += f" {obj['circuit']}"
            if "refuge" in obj:
                label += f" {obj['refuge']}"
            if obj["role"] == "cache":
                label += f" {obj['item']} locker {obj['option']}"
            descriptions.append(f"{tid} {label} ({d:.1f}m)")
        cue = self.last_text if self.steps <= self.cue_until else ""
        nearby_engine = np.linalg.norm(self.agent.pos-self.engine_pos) < 65
        phase = f"Engine shutters: {self.engine_phase()}." if nearby_engine or self.live_regions()["monitoring"] else ""
        obs["text"] = (GOALS[self.objective]+f" Health {self.agent.health:.1f}; stamina {self.agent.stamina:.1f}. "
                       +f"Carried service items: {dict(self.satchel)}. "+phase+" "+cue
                       +" Local targets: "+"; ".join(descriptions))[:4096]
        return obs

    def _info(self, events):
        # Evaluator-only progress and remote trajectories never enter agent info.
        visible_types = {"action_failed", "clue_exposed", "part_picked", "part_mounted", "part_rotated",
                         "cargo_loaded", "cargo_delivered", "cargo_towed", "ascended", "reserves_cached",
                         "cache_retrieved", "circuit_isolated", "capacitor_struck", "grounding_installed",
                         "engine_stopped", "storm_hit", "death", "station_engaged", "gallery_serviced",
                         "platform_landing_reached", "dock_lit", "outpost_opened", "service_part_recovered",
                         "reversal_started", "reversal_cancelled", "lock_coupling_installed", "cargo_berth_opened",
                         "map_opened", "map_closed", "map_layer_selected"}
        local_events = [e for e in events if e["type"] in visible_types]
        result = WildlandsEnv._info(self, local_events)
        result.update(success=self.success, objective=self.objective)
        return result

    def evaluation(self):
        return dict(success=self.success, objective=self.objective, milestones=deepcopy(self.completed),
                    live_regions=self.live_regions(), engine_disabled=self.engine_disabled,
                    engine_route=self.engine_route, locks=list(self.locks), capacitors=list(self.capacitors),
                    grounding_live=self.grounding_live(), facts_seen=dict(self.facts_seen),
                    decisions=self.steps, simulated_seconds=self.elapsed, deaths=self.agent.deaths,
                    architecture_count=len(self.architecture), architecture_kinds=sorted({b["kind"] for b in self.architecture}))

    def step(self, action):
        result = super().step(action)
        self.action_history.append(plain(deepcopy(action)))
        return result

    def state_digest(self):
        state = dict(base=super().state_digest(), objective=self.objective, disabled=self.engine_disabled,
                     route=self.engine_route, locks=self.locks, capacitors=self.capacitors,
                     settings=self.lock_settings, cycle_offset=self.cycle_offset, last_strike=self.last_strike,
                     grounding=self.grounding_installed, shutdown_seconds=self.shutdown_seconds,
                     cue_until=self.cue_until, last_text=self.last_text, facts_seen=self.facts_seen,
                     architecture=self.architecture)
        return hashlib.sha256(json.dumps(plain(state), sort_keys=True).encode()).hexdigest()

    def save_checkpoint(self, path):
        """Portable deterministic replay checkpoint; no executable pickle payload."""
        data = dict(format="owb-tempest-replay-v1", seed=self._seed, objective=self.objective,
                    max_steps=self.cfg.max_steps, actions=self.action_history, digest=self.state_digest())
        raw = json.dumps(data, separators=(",", ":"), allow_nan=False)
        with Path(path).open("x") as stream:
            json.dump(dict(sha256=hashlib.sha256(raw.encode()).hexdigest(), payload=raw), stream)

    @classmethod
    def load_checkpoint(cls, path, *, render_enabled=False):
        wrapper = json.loads(Path(path).read_text())
        raw = wrapper["payload"]
        if hashlib.sha256(raw.encode()).hexdigest() != wrapper["sha256"]:
            raise ValueError("checkpoint checksum mismatch")
        data = json.loads(raw)
        if data["format"] != "owb-tempest-replay-v1":
            raise ValueError("unsupported checkpoint format")
        env = cls(data["objective"], data["seed"], render_enabled=False, max_steps=data["max_steps"])
        env.reset(seed=data["seed"])
        try:
            for row in data["actions"]:
                action = {k: np.asarray(v, dtype=env.action_space[k].dtype) if isinstance(v, list) else v for k, v in row.items()}
                if not env.action_space.contains(action):
                    raise ValueError("invalid checkpoint action")
                env.step(action)
            if env.state_digest() != data["digest"]:
                raise ValueError("checkpoint replay diverged; use matching source version")
            env.cfg.obs_rgb = render_enabled
            if render_enabled:
                r = env.cfg.render_cfg
                env.observation_space["rgb"] = spaces.Box(0, 255, (r.height, r.width, 3), np.uint8)
            return env
        except BaseException:
            env.close()
            raise

    def presentation_meshes(self, batch):
        from ..render.tempest import engine_meshes
        super().presentation_meshes(batch)
        engine_meshes(self, batch)

    def static_presentation_meshes(self, batch):
        from ..render.tempest import architecture_meshes
        architecture_meshes(self.architecture, batch)

    def presentation_lights(self):
        lights = super().presentation_lights()+[
            dict(pos=(self.engine_pos+(0, 19, 0)).tolist(), color=[.35, .68, 1.], power=4200., radius=4.)]
        for building in self.architecture:
            p = np.asarray(building["pos"], float)
            lights.append(dict(pos=(p+(-9.5, 6., -10.)).tolist(), color=[.95, .72, .42],
                               power=4500. if p[1] < 30 else 1800., radius=2.))
            if building["kind"] not in ("gantry", "signal_tower", "cistern"):
                for x in (-2.7, 3.7):
                    lights.append(dict(pos=(p+(x, 3.1, -7.)).tolist(), color=[1., .68, .32], power=750., radius=.3))
        return lights


def make_extended(**kwargs):
    return SkylandsTempestEnv(objective="extended", **kwargs)
