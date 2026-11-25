"""Shared bookkeeping for working-memory (WM) probes.

See docs/working-memory-probes.md. An environment owns the physical fixtures
(source sign/NPC, keypad, chests, dial) and decides when the agent reads the cue
or makes an entry; this module draws the per-episode content from a dedicated
generator, judges complete entries, emits the shared events, and builds the
evaluator-only summary. It never touches the simulation RNG and never reports
per-item correctness, so environments can surface its events as neutral feedback.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import zlib

import numpy as np

KINDS = ("hold", "reverse", "remap", "select")


def probe_rng(seed: int, probe_id: str) -> np.random.Generator:
    """Generator for a probe's content, independent of every simulation stream."""
    return np.random.default_rng([int(seed) & 0xFFFFFFFF, zlib.crc32(probe_id.encode())])


@dataclass
class WorkingMemoryProbe:
    """One WM probe's content and scoring state.

    kind      hold/reverse: enter `items` (or items reversed) one at a time.
              remap: enter `expected`, which the environment derives from `items`
              and a mapping visible at the use site (pass it to `set_expected`).
              select: one or more slots, each answered by choosing one container;
              `expected[i]` is the correct container for slot i.
    """

    probe: str
    milestone: str
    kind: str
    load: int
    items: tuple = ()
    expected: tuple = ()
    control: bool = False
    cue_reads: int = 0
    attempts: int = 0
    solved: bool = False
    first_attempt_correct: bool | None = None
    first_attempt_delay: dict | None = None
    solve_delay: dict | None = None
    buffer: list = field(default_factory=list)
    slot_done: list = field(default_factory=list)
    tried: set = field(default_factory=set)   # select slots that have had a first choice
    _cue_at: tuple | None = None   # (decision, seconds) of the most recent cue read

    @classmethod
    def draw(cls, probe: str, milestone: str, kind: str, seed: int, symbols, load: int,
             *, distinct: bool = True, control: bool = False) -> "WorkingMemoryProbe":
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}")
        symbols = list(symbols)
        rng = probe_rng(seed, probe)
        if distinct and load > len(symbols):
            raise ValueError("not enough distinct symbols for the load")
        picks = rng.choice(len(symbols), size=load, replace=not distinct)
        items = tuple(symbols[int(i)] for i in picks)
        expected = tuple(reversed(items)) if kind == "reverse" else items
        return cls(probe=probe, milestone=milestone, kind=kind, load=load, items=items,
                   expected=expected, control=control,
                   slot_done=[False] * (load if kind == "select" else 0))

    def rng(self, seed: int) -> np.random.Generator:
        """A second, still independent stream for layout choices tied to this probe
        (e.g. which chest carries the target mark)."""
        return probe_rng(seed, self.probe + "/layout")

    def set_expected(self, expected) -> None:
        self.expected = tuple(expected)
        if self.kind == "select":
            self.slot_done = [False] * len(self.expected)

    # -- agent interactions -------------------------------------------------
    def read_cue(self, decision: int, seconds: float) -> dict:
        self.cue_reads += 1
        self._cue_at = (int(decision), float(seconds))
        return dict(type="wm_cue_read", probe=self.probe, read_index=self.cue_reads)

    def _delay(self, decision, seconds):
        if self._cue_at is None:
            return dict(decisions=None, seconds=None)
        return dict(decisions=int(decision) - self._cue_at[0],
                    seconds=round(float(seconds) - self._cue_at[1], 6))

    def _judge(self, correct: bool, decision, seconds) -> list[dict]:
        self.attempts += 1
        delay = self._delay(decision, seconds)
        if self.first_attempt_correct is None:
            self.first_attempt_correct = bool(correct)
            self.first_attempt_delay = delay
        if correct and not self.solved:
            self.solved = True
            self.solve_delay = delay
        return [dict(type="wm_attempt", probe=self.probe, attempt=self.attempts, correct=bool(correct),
                     decisions_since_cue=delay["decisions"], seconds_since_cue=delay["seconds"],
                     cue_reads=self.cue_reads)]

    def record_attempt(self, correct: bool, decision: int, seconds: float) -> list[dict]:
        """Judge one attempt directly, for mechanisms whose entries are physical
        (e.g. knobs that only turn in order) rather than a typed sequence."""
        if self.solved:
            return []
        return self._judge(correct, decision, seconds)

    def enter(self, item, decision: int, seconds: float) -> list[dict]:
        """One sequence entry (hold/reverse/remap). A complete entry is judged and
        the panel clears either way; a solved panel ignores further input."""
        if self.kind == "select":
            raise ValueError("select probes use choose()")
        if self.solved:
            return []
        self.buffer.append(item)
        events = [dict(type="wm_entry", probe=self.probe, count=len(self.buffer))]
        if len(self.buffer) >= len(self.expected):
            correct = tuple(self.buffer) == tuple(self.expected)
            self.buffer = []
            events += self._judge(correct, decision, seconds)
        return events

    def clear(self) -> None:
        """Abandon a partial entry (e.g. the agent walked away or pressed reset)."""
        self.buffer = []

    def choose(self, slot: int, choice, decision: int, seconds: float) -> list[dict]:
        """Open one container for `slot`. Every opening of an unsolved slot is an
        attempt; the probe is solved when every slot has been answered correctly.
        `first_attempt_correct` is true only if each slot's first choice was right."""
        if self.kind != "select":
            raise ValueError("choose() is for select probes")
        if self.slot_done[slot]:
            return []
        correct = choice == self.expected[slot]
        self.attempts += 1
        delay = self._delay(decision, seconds)
        first_for_slot = slot not in self.tried
        self.tried.add(slot)
        if first_for_slot:
            ok = bool(correct) and (self.first_attempt_correct is not False)
            self.first_attempt_correct = ok
            if self.first_attempt_delay is None:
                self.first_attempt_delay = delay
        if correct:
            self.slot_done[slot] = True
        if all(self.slot_done) and not self.solved:
            self.solved = True
            self.solve_delay = delay
        return [dict(type="wm_attempt", probe=self.probe, attempt=self.attempts, correct=bool(correct),
                     slot=int(slot), decisions_since_cue=delay["decisions"],
                     seconds_since_cue=delay["seconds"], cue_reads=self.cue_reads)]

    # -- evaluator views ----------------------------------------------------
    def summary(self) -> dict:
        return dict(milestone=self.milestone, kind=self.kind, load=self.load, control=self.control,
                    solved=self.solved, attempts=self.attempts,
                    first_attempt_correct=self.first_attempt_correct, cue_reads=self.cue_reads,
                    first_attempt_delay=self.first_attempt_delay, solve_delay=self.solve_delay)

    def state(self) -> dict:
        """Complete, JSON-able state for snapshots and replay digests."""
        s = self.summary()
        s.update(probe=self.probe, items=list(self.items), expected=list(self.expected),
                 buffer=list(self.buffer), slot_done=list(self.slot_done),
                 tried=sorted(self.tried),
                 cue_at=list(self._cue_at) if self._cue_at else None)
        return s

    @classmethod
    def from_state(cls, s: dict) -> "WorkingMemoryProbe":
        p = cls(probe=s["probe"], milestone=s["milestone"], kind=s["kind"], load=s["load"],
                items=tuple(s["items"]), expected=tuple(s["expected"]), control=s["control"],
                cue_reads=s["cue_reads"], attempts=s["attempts"], solved=s["solved"],
                first_attempt_correct=s["first_attempt_correct"],
                first_attempt_delay=s["first_attempt_delay"], solve_delay=s["solve_delay"],
                buffer=list(s["buffer"]), slot_done=list(s["slot_done"]), tried=set(s.get("tried", [])))
        p._cue_at = tuple(s["cue_at"]) if s.get("cue_at") else None
        return p


def summaries(probes) -> dict:
    """{probe_id: summary} for an iterable of probes."""
    return {p.probe: p.summary() for p in probes}
