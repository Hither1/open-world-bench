import json

from owb.core.working_memory import WorkingMemoryProbe, probe_rng


def test_draw_is_deterministic_and_independent_of_simulation_rng():
    a = WorkingMemoryProbe.draw("WM-X", "A01", "hold", 7, "ABCDEF", 4)
    b = WorkingMemoryProbe.draw("WM-X", "A01", "hold", 7, "ABCDEF", 4)
    c = WorkingMemoryProbe.draw("WM-Y", "A01", "hold", 7, "ABCDEF", 4)
    assert a.items == b.items and len(set(a.items)) == 4
    assert a.items != c.items or probe_rng(7, "WM-X").random() != probe_rng(7, "WM-Y").random()


def test_reverse_sequence_scoring_and_delays():
    p = WorkingMemoryProbe.draw("WM-R", "B10", "reverse", 3, "ABCDEF", 4)
    assert p.expected == tuple(reversed(p.items))
    p.read_cue(10, 2.0)
    events = []
    for item in p.items:  # forward order is wrong for a reverse probe
        events += p.enter(item, 50, 10.0)
    assert [e["type"] for e in events][-1] == "wm_attempt" and events[-1]["correct"] is False
    assert all("correct" not in e for e in events if e["type"] == "wm_entry")
    for item in p.expected:
        events = p.enter(item, 60, 12.0)
    assert events[-1]["correct"] and p.solved and p.attempts == 2
    s = p.summary()
    assert s["first_attempt_correct"] is False
    assert s["first_attempt_delay"] == dict(decisions=40, seconds=8.0)
    assert s["solve_delay"] == dict(decisions=50, seconds=10.0)
    assert p.enter("A", 70, 13.0) == []


def test_select_two_slots_and_state_round_trip():
    p = WorkingMemoryProbe.draw("WM-S", "B11", "select", 1, range(6), 2)
    p.set_expected((4, 1))
    p.read_cue(0, 0.0)
    p.choose(0, 4, 5, 1.0)
    p.choose(1, 2, 6, 1.2)          # wrong first choice for slot 1
    assert p.first_attempt_correct is False and not p.solved
    restored = WorkingMemoryProbe.from_state(json.loads(json.dumps(p.state())))
    assert restored.state() == p.state()
    restored.choose(1, 1, 9, 2.0)
    assert restored.solved and restored.attempts == 3 and restored.choose(0, 4, 10, 2.1) == []
