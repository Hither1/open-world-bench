"""Teyvat working-memory probes: content, cue wording and the cue channel.

Contract: ``docs/working-memory-probes.md``. Scoring and events come from
:mod:`owb.core.working_memory`; this module only fixes what Teyvat shows and
where. Content is drawn from the probe's own generator (seeded by the episode
seed and the probe id), never from the simulation or course RNG.

Probes:

* ``WM-TA06`` (compact Task A, A06): the ferry mechanic speaks a 3-lever start
  sequence when the casing is delivered; the lever bank stands at the lift's
  remote test station.
* ``WM-TB09`` (Task B, B09): the boatman names the 3-glyph mark of the chest
  holding the coupling; six marked chests lie under water, their marks readable
  only within a few metres. Select probe with one slot.
* ``WM-TB10`` (Task B, B10): the causeway mechanic gives a 4-element gate
  pattern when kit and coupling are delivered and says to set it in reverse at
  the winch at the far end of the causeway.
"""
from __future__ import annotations

import numpy as np

from owb.core.working_memory import WorkingMemoryProbe, summaries

LEVER_COLOURS = ("red", "blue", "green", "yellow", "white")
CHEST_GLYPHS = ("anchor", "bell", "crown", "fish", "key", "lantern", "shell", "star")
PAWL_ELEMENTS = ("ember", "tide", "frost", "spark", "gale", "stone")
CHEST_COUNT = 6
CUE_REACH = 4.0          # metres: the cue clears when the agent is farther than this from the source
CUE_HEIGHT = 3.0
CUE_SECONDS = 5.0        # simulated seconds: the cue clears after this, whichever comes first
WALK_AWAY = 12.0         # metres from the panel centre: a partial entry is abandoned

TA06, TB09, TB10 = "WM-TA06", "WM-TB09", "WM-TB10"


def make_probes(task: str, seed: int, *, control: bool = False) -> dict:
    """{probe id: WorkingMemoryProbe} for compact/full-scale task 'a' or 'b'."""
    if task == "a":
        return {TA06: WorkingMemoryProbe.draw(TA06, "A06", "hold", seed, LEVER_COLOURS, 3, control=control)}
    return {TB09: WorkingMemoryProbe.draw(TB09, "B09", "select", seed, CHEST_GLYPHS, 3, control=control),
            TB10: WorkingMemoryProbe.draw(TB10, "B10", "reverse", seed, PAWL_ELEMENTS, 4, control=control)}


def chest_marks(probe: WorkingMemoryProbe, seed: int, count: int = CHEST_COUNT):
    """(marks, target index). Distractor marks share one or two glyphs, in place, with the target."""
    rng = probe.rng(seed)
    target = tuple(probe.items)
    others = [g for g in CHEST_GLYPHS if g not in target]
    marks, shared = [], [1, 2] * count
    while len(marks) < count - 1:
        keep = shared[len(marks)]
        replace = rng.choice(3, size=3 - keep, replace=False)
        mark = list(target)
        picks = rng.choice(len(others), size=len(replace), replace=False)
        for slot, k in zip(replace, picks):
            mark[int(slot)] = others[int(k)]
        mark = tuple(mark)
        if mark not in marks:
            marks.append(mark)
    index = int(rng.integers(count))
    marks.insert(index, target)
    return marks, index


def mark_text(mark) -> str:
    return "-".join(mark)


def mark_code(mark) -> int:
    """Visual version for a marked chest (stable per entity; render only)."""
    a, b, c = (CHEST_GLYPHS.index(g) for g in mark)
    return 1 + a * 64 + b * 8 + c


def mark_from_code(code: int):
    v = int(code) - 1
    return CHEST_GLYPHS[(v // 64) % 8], CHEST_GLYPHS[(v // 8) % 8], CHEST_GLYPHS[v % 8]


def cue_text(probe: WorkingMemoryProbe, compact: bool = False) -> str:
    """What the source says. The transformation, when any, is stated explicitly."""
    items = list(probe.items)
    if probe.probe == TA06:
        where = "the lift's test station" if compact else "the lift's upper test station"
        return (f"\"Casing's in. Start the lift at {where}: pull the levers {', '.join(items[:-1])} "
                f"and {items[-1]}, in that order.\"")
    if probe.probe == TB09:
        return (f"\"Six chests went down out there. The coupling is in the one marked "
                f"{mark_text(items)}. The others are empty.\"")
    if probe.probe == TB10:
        return (f"\"Kit and coupling in. The gate pattern is {', '.join(items)}. Set it on the winch at the far "
                f"end of the causeway in reverse order: last first.\"")
    raise KeyError(probe.probe)


def control_text(probe: WorkingMemoryProbe, compact: bool = False) -> str:
    """The same cue, posted at the use site (``wm_control=True`` only)."""
    where = {TA06: "Notice on the lever bank", TB09: "Tag on the buoy line", TB10: "Notice on the winch"}[probe.probe]
    return f"{where}: {cue_text(probe, compact)}"


def neutral_entry(label: str, count: int) -> str:
    """Entry feedback: what was operated and how many entries so far; never correctness."""
    return f"{label}: click ({count} set)."


def wm_event(payload: dict):
    from .events import Event
    payload = dict(payload)
    return Event(payload.pop("type"), payload)


class CueChannel:
    """The single cue currently on screen: shown only within reach and for 5 s.

    Cleared (not merely hidden) when either condition fails, so walking back
    within 5 s does not bring it back; the source must be inspected again.
    """

    def __init__(self):
        self.cue = None          # (probe id, source entity id, text, time posted)

    def post(self, probe_id, entity_id, text, now):
        self.cue = (probe_id, entity_id, text, float(now))

    def _live(self, world, char, now):
        _, eid, _, when = self.cue
        e = world.get(eid)
        d = float(np.hypot(*(e.pos - char.pos)[[0, 2]]))
        return now - when <= CUE_SECONDS + 1e-9 and d <= CUE_REACH and abs(float(e.pos[1] - char.pos[1])) <= CUE_HEIGHT

    def update(self, world, char, now):
        """Called every simulation substep: clear the cue once either condition fails."""
        if self.cue is not None and not self._live(world, char, now):
            self.cue = None

    def text(self, world, char, now):
        if self.cue is None or not self._live(world, char, now):
            return None
        _, eid, text, _ = self.cue
        return f"{world.get(eid).label}: {text}"

    def state(self):
        return None if self.cue is None else [self.cue[0], self.cue[1], self.cue[2], round(self.cue[3], 6)]


__all__ = ["make_probes", "chest_marks", "mark_text", "mark_code", "mark_from_code", "cue_text", "control_text",
           "neutral_entry", "wm_event", "CueChannel", "summaries", "LEVER_COLOURS", "CHEST_GLYPHS",
           "PAWL_ELEMENTS", "CHEST_COUNT", "CUE_REACH", "CUE_SECONDS", "WALK_AWAY", "TA06", "TB09", "TB10"]
