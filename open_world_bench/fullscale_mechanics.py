"""Full-scale Teyvat mechanisms: elemental relays, sentinels and vents.

These are the procedures R, G and T of the legacy designs
(``docs/teyvat-legacy-tasks.md``) at regional scale, written to be reused by
both full-scale tasks. Each unit holds private state and applies the rules;
the course decides which units are live and what completing them grants.
Units never draw from the simulation RNG.

Rules, briefly:

* **Relay (R).** Single ring (v1): drain if overloaded, prime the source,
  route the vane, confirm at the core. Double ring (v2): additionally drain
  first even when not overloaded. Wrong applications are rejected with
  categorical feedback; draining keeps completed stages. An authored
  interruption re-overloads the unit once after priming.
* **Sentinel (G).** Shield, exposed (1 s), wind-up (5 s), vulnerable (8 s),
  then the shield reforms. The learned breaker element opens the shield, the
  learned interrupter element during the wind-up opens the core. A warding
  pulse hurts within 6 m every 6 s in the shield phase; an uninterrupted
  wind-up slams within 9 m. Wrong elements trigger the pulse or the slam.
* **Vent (T).** Charged by the family element; a jump in the column on the
  safe beat carries the character up to the column top; a jump on the wrong
  beat sputters and spends the charge.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import zlib

import numpy as np

from .elements import Element

ROLES = ("source", "vane", "drain", "core")

# Sentinel timing (seconds) and damage.
EXPOSED_TIME = 1.0
WINDUP_TIME = 5.0
VULNERABLE_TIME = 8.0
PULSE_PERIOD = 6.0
PULSE_RADIUS = 6.0
PULSE_DAMAGE = 8.0
SLAM_RADIUS = 9.0
SLAM_DAMAGE = 26.0
WAKE_RADIUS = 12.0          # a sentinel holding a seeded start phase waits for the agent

# Vent timing.
BEAT_PERIOD = 4.0
BRIGHT_TIME = 2.5
CHARGE_TIME = 30.0
RISE_SPEED = 14.0
COLUMN_RADIUS = 2.0         # metres from the pad centre where a jump launches


def stable_unit(*parts) -> float:
    """Deterministic value in [0, 1) from names, for presentation offsets."""
    return (zlib.crc32("|".join(map(str, parts)).encode()) & 0xFFFFFF) / float(0x1000000)


# --------------------------------------------------------------------- relay
@dataclass
class RelayUnit:
    id: str
    label: str
    family: str                 # workshop | field | shrine | beacon
    version: int                # 1 single ring, 2 double ring
    receivers: dict             # role -> entity id
    progress: int = 0           # 0 unprimed, 1 primed, 2 routed
    overloaded: bool = False
    drained: bool = False
    active: bool = False
    interrupt: str | None = None      # "prime": overload once right after priming
    interrupted: bool = False
    applications: int = 0
    invalid: int = 0
    unnecessary_drains: int = 0
    log: list = field(default_factory=list)

    def ready_for_core(self) -> bool:
        return self.progress == 2 and not self.overloaded and (self.version == 1 or self.drained)

    def needed_role(self) -> str:
        """The next role a correct procedure acts on (reference only)."""
        if self.overloaded or (self.version == 2 and not self.drained):
            return "drain"
        return ("source", "vane", "core")[self.progress]

    def apply(self, role: str, element: Element, rules: dict) -> tuple[bool, str, dict]:
        """Apply ``element`` at ``role``. Returns (accepted, feedback, event payload)."""
        self.applications += 1
        payload = dict(relay=self.id, role=role, element=element.name.lower())
        if role == "core":
            return False, "Core: confirm by hand, not with an element.", payload
        if element != rules[role]:
            self.invalid += 1
            return False, f"{self.label} {role}: the receiver rejects that element.", dict(payload, reason="element")
        if role == "drain":
            if not self.overloaded and (self.version == 1 or self.drained):
                self.unnecessary_drains += 1
            self.overloaded = False
            self.drained = True
            return True, f"{self.label} drain: charge released; completed stages kept.", payload
        if self.overloaded:
            self.invalid += 1
            return False, f"{self.label} {role}: overloaded; nothing takes.", dict(payload, reason="overload")
        if self.version == 2 and not self.drained:
            self.invalid += 1
            return False, (f"{self.label} {role}: the inlet still holds a residual charge "
                           "(double ring)."), dict(payload, reason="residual")
        if role == "source":
            if self.progress != 0:
                self.invalid += 1
                return False, f"{self.label} source: already primed.", dict(payload, reason="stage")
            self.progress = 1
            if self.interrupt == "prime" and not self.interrupted:
                self.interrupted = True
                self.overloaded = True
                return True, f"{self.label}: primed - then a surge overloads it; priming held.", \
                    dict(payload, interrupted=True)
            return True, f"{self.label} source: primed.", payload
        if role == "vane":
            if self.progress != 1:
                self.invalid += 1
                return False, f"{self.label} vane: nothing to route yet.", dict(payload, reason="stage")
            self.progress = 2
            return True, f"{self.label} vane: routed.", payload
        return False, "", payload

    def confirm(self) -> tuple[bool, str]:
        if self.active:
            return False, f"{self.label} is already running."
        if self.ready_for_core():
            self.active = True
            return True, f"{self.label}: core confirmed; the relay runs."
        if self.version == 2 and not self.drained:
            return False, f"{self.label} core: blocked - residual charge in the inlet."
        return False, f"{self.label} core: blocked - check the lamps and the overload flag."

    def cue(self) -> str:
        lamps = ("unprimed", "primed", "routed")[self.progress] if not self.active else "active"
        ring = "double ring" if self.version == 2 else "single ring"
        return f"{self.label} ({ring}): lamps {lamps}; overload {'on' if self.overloaded else 'off'}"


# ------------------------------------------------------------------ sentinel
@dataclass
class SentinelUnit:
    id: str
    entity: str
    label: str
    hp: float
    phase: str = "shield"       # shield | exposed | windup | vulnerable
    timer: float = 0.0
    hold: bool = False          # seeded start phase waits until the agent is near
    reform_once: bool = False
    reformed: bool = False
    pulse_timer: float = PULSE_PERIOD
    cycles: int = 0
    wrong_casts: int = 0
    interrupts: int = 0
    missed_windows: int = 0
    pulses: int = 0
    slams: int = 0

    def posture(self) -> str:
        return dict(shield="shielded", exposed="open core", windup="winding up", vulnerable="cracked core")[self.phase]

    def set_phase(self, phase):
        self.phase, self.timer = phase, 0.0
        if phase == "shield":
            self.pulse_timer = PULSE_PERIOD

    def element_hit(self, element: Element, rules: tuple) -> list:
        """Rule outcome of an element landing on the sentinel: a list of (event, payload)."""
        breaker, interrupter = rules
        out = []
        if self.hold:
            self.hold = False
        if self.phase == "shield":
            if element == breaker:
                self.set_phase("exposed")
                out.append(("sentinel_exposed", {}))
            else:
                self.wrong_casts += 1
                out.append(("sentinel_pulse", dict(reason="wrong_element")))
        elif self.phase == "windup":
            if element == interrupter:
                self.interrupts += 1
                if self.reform_once and not self.reformed:
                    self.reformed = True
                    self.set_phase("shield")
                    out.append(("sentinel_reformed", {}))
                else:
                    self.set_phase("vulnerable")
                    out.append(("sentinel_interrupted", {}))
            else:
                self.wrong_casts += 1
                out.append(("sentinel_slam", dict(reason="wrong_element")))
        else:
            self.wrong_casts += 1
            out.append(("sentinel_blocked", dict(phase=self.phase)))
        return out

    def tick(self, dt: float, near: bool, distance: float) -> list:
        """Advance timers. ``near`` is whether the agent is inside WAKE_RADIUS."""
        out = []
        if self.hold:
            if not near:
                return out
            self.hold = False
            self.timer = 0.0
        self.timer += dt
        if self.phase == "shield":
            self.pulse_timer -= dt
            if self.pulse_timer <= 0.0:
                self.pulse_timer = PULSE_PERIOD
                out.append(("sentinel_pulse", dict(reason="ward")))
        elif self.phase == "exposed" and self.timer >= EXPOSED_TIME:
            self.set_phase("windup")
            out.append(("sentinel_windup", {}))
        elif self.phase == "windup" and self.timer >= WINDUP_TIME:
            self.missed_windows += 1
            out.append(("sentinel_slam", dict(reason="window_missed")))
        elif self.phase == "vulnerable" and self.timer >= VULNERABLE_TIME:
            self.cycles += 1
            self.set_phase("shield")
            out.append(("sentinel_reformed", dict(reason="window_closed")))
        return out


# ---------------------------------------------------------------------- vent
@dataclass
class VentState:
    id: str
    route: str
    leg: int
    pad: tuple
    pad_height: float
    apex: float
    requires: str | None
    offset: float               # beat phase offset (presentation-stable)
    charged: bool = False
    charged_at: float = 0.0
    launches: int = 0
    sputters: int = 0
    charges: int = 0

    def bright(self, elapsed: float) -> bool:
        return ((elapsed + self.offset) % BEAT_PERIOD) < BRIGHT_TIME

    def beat(self, elapsed: float) -> str:
        return "bright" if self.bright(elapsed) else "dark"

    def safe(self, elapsed: float, safe_beat: str) -> bool:
        return self.beat(elapsed) == safe_beat

    def in_column(self, pos) -> bool:
        return (float(np.hypot(pos[0] - self.pad[0], pos[2] - self.pad[1])) <= COLUMN_RADIUS
                and abs(float(pos[1]) - self.pad_height) < 2.5)

    def expire(self, elapsed: float) -> bool:
        if self.charged and elapsed - self.charged_at > CHARGE_TIME:
            self.charged = False
            return True
        return False
