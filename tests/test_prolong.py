"""PRO-LONG harness: log contents, plan validation, sandbox boundary and recorded replay."""
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest

from owb.bench.baselines.prolong import (AgentResult, MirroredCredential, ProLongError, ProLongPolicy, Sandbox,
                                        _interface)
from owb.bench.baselines.skylands_model import SkylandsModelPolicy
from owb.bench.baselines.wildlands_model import WildlandsModelPolicy
from owb.env.wildlands_reference import WildlandsReferenceEnv
from owb.scripts.prolong_model import run
from owb.scripts.wildlands_model import WILDLANDS

pytestmark = pytest.mark.skipif(shutil.which("bwrap") is None, reason="bubblewrap unavailable")
ROOT = Path(__file__).resolve().parents[1]


class FakeBackend:
    """Stands in for the CLI: each scripted turn writes actions.json in the workspace."""
    name, model, effort, timeout, auth = "fake", "fake-model", "high", 60, "none"
    executable = sys.executable
    account_home = Path("/nonexistent")
    view_hint, tools_line = "View frames", "Read"

    def __init__(self, turns):
        self.turns, self.session_id, self.prompts = list(turns), None, []

    def mounts(self, private):
        return (), (private,), {}

    def call(self, sandbox, prompt, system_prompt, transcript):
        self.prompts.append(prompt)
        self.session_id = "session-1"
        turn = self.turns.pop(0)
        if isinstance(turn, Exception):
            raise turn
        if turn is not None:
            (sandbox.workspace / "actions.json").write_text(json.dumps(turn))
        Path(transcript).write_text("")
        return AgentResult(text="Briefing.\n[PLAN]\nInspect.", session_id="session-1",
                           usage=dict(input_tokens=5, output_tokens=7), tools=["Read"], duration_seconds=0.1)


def no_render(monkeypatch):
    monkeypatch.setattr(WildlandsReferenceEnv, "render", lambda _: np.zeros((180, 320, 3), np.uint8))


def make_policy(tmp_path, turns, **kwargs):
    return ProLongPolicy(WildlandsModelPolicy(None), FakeBackend(turns), tmp_path / "p",
                         mission="Repair the intake.", **kwargs)


def test_entries_default_to_neutral_and_invalid_tail_is_discarded(tmp_path):
    policy = make_policy(tmp_path, [{"actions": [
        {"move": [1, 0], "repeat": 5}, {"command": "inspect", "target": 1},
        {"command": "teleport"}, {"command": "none"}]}], action_cap=20)
    policy.current = ({}, "frames/000000.png")
    policy.master_log.write_text("")
    entries = policy.plan()
    assert [e[1] for e in entries] == [5, 1]
    assert entries[0][2]["command"] == "none" and entries[0][2]["ingredients"] == [0] * 5
    log = policy.master_log.read_text()
    assert "[PLAN ERROR]" in log and "entry 3" in log and "1 later entries" in log
    assert "memory" not in entries[0][2]


@pytest.mark.parametrize("turn", [None, {"actions": "ACTION1"}, {"actions": [{"oracle": 1}]},
                                  {"actions": [{"command": "inspect", "repeat": 3}]}])
def test_unusable_plan_returns_nothing_and_nudges(tmp_path, turn):
    policy = make_policy(tmp_path, [turn, {"actions": [{}]}])
    policy.master_log.write_text("")
    assert policy.plan() == []
    policy.plan()
    assert "did not produce a valid" in policy.backend.prompts[1]


def test_action_cap_truncates(tmp_path):
    policy = make_policy(tmp_path, [{"actions": [{}] * 5}], action_cap=2)
    policy.master_log.write_text("")
    assert len(policy.plan()) == 2


def test_prompt_reuses_course_interface_without_memory_protocol():
    for guide in (WildlandsModelPolicy.guide, SkylandsModelPolicy.guide):
        text = _interface(guide)
        assert "memory note" not in text and "supplied schema" not in text
        assert "repeat is 1..30" in text


def test_episode_log_replay_and_milestone_flush(tmp_path, monkeypatch):
    no_render(monkeypatch)
    inspect = {"command": "inspect", "target": 1}
    backend = FakeBackend([{"actions": [inspect, {"move": [1, 0], "repeat": 3}, {}]},
                           {"actions": [{"move": [1, 0], "repeat": 2}]}])
    output = tmp_path / "episode"
    result = run("a", output, backend, course=WILDLANDS, max_calls=2, max_actions=50)
    assert result["milestones_completed"] == 1 and result["stop_reason"] == "agent_call_limit"
    assert result["primitive_actions"] == 3        # 1 inspect + flushed batch + 2 moves
    assert json.loads((output / "verification.json").read_text())["replay_verified"]
    log = (output / "prolong/history/logs.txt").read_text()
    assert log.count("[OBSERVATION]") == 3 and "[PLAN STOPPED]\n2 remaining entries" in log
    assert log.count("[ANALYSIS]") == 2
    # Evaluator events, digests and milestone ids stay outside the agent's history.
    assert "milestone_complete" not in log and "digest" not in log and '"A01"' not in log
    assert sorted(p.name for p in (output / "prolong/history/frames").iterdir()) == \
        ["000000.png", "000001.png", "000003.png"]
    assert "Read the episode log" in backend.prompts[1]
    rows = [json.loads(line) for line in (output / "actions.jsonl").read_text().splitlines()]
    assert [r["decision"] for r in rows] == [1, 2, 2] and [r["entry"] for r in rows] == [1, 1, 1]


def test_no_log_condition_has_no_history_mount(tmp_path, monkeypatch):
    no_render(monkeypatch)
    backend = FakeBackend([{"actions": [{}]}])
    output = tmp_path / "episode"
    run("a", output, backend, course=WILDLANDS, max_calls=1, max_actions=5, log_window=-1)
    assert "[CURRENT OBSERVATION]" in backend.prompts[0] and "episode log" not in backend.prompts[0]
    assert (output / "prolong/workspace/current_frame.png").is_file()
    assert "logs.txt" not in (output / "control-guide.txt").read_text()


def test_log_window_exposes_only_recent_sections(tmp_path):
    policy = make_policy(tmp_path, [{"actions": [{}]}], log_window=2)
    obs = dict(text="t", proprio=np.zeros(16), survival=np.zeros(4), inventory=np.zeros((32, 4)),
               rgb=np.zeros((4, 4, 3), np.uint8))
    for step in range(5):
        policy.observe(obs, step, f"Section {step}")
    policy.plan()
    visible = (policy.visible / "logs.txt").read_text()
    assert "Section 0" in visible and "Section 1" not in visible and "Section 4" in visible
    assert sorted(p.name for p in (policy.visible / "frames").iterdir()) == ["000000.png", "000003.png", "000004.png"]


def test_transport_errors_retry_then_stop(tmp_path, monkeypatch):
    no_render(monkeypatch)
    backend = FakeBackend([ProLongError("slow", "timeout"), {"actions": [{}]},
                           ProLongError("quota", "usage_limit")])
    result = run("b", tmp_path / "episode", backend, course=WILDLANDS, max_calls=5, max_actions=5)
    assert result["stop_reason"] == "inference_or_interface_error"
    assert result["failure"]["code"] == "usage_limit" and result["primitive_actions"] == 1


def test_sandbox_hides_repository_and_protects_history(tmp_path):
    workspace, history = tmp_path / "ws", tmp_path / "hist"
    (workspace / "history").mkdir(parents=True)
    history.mkdir()
    (history / "logs.txt").write_text("log")
    box = Sandbox(workspace, history, read_only=(Path(sys.prefix).resolve(),),
                  env=dict(PATH=f"{Path(sys.prefix).resolve() / 'bin'}:/usr/bin:/bin"))
    probe = (
        "import os,json;"
        f"r={{'repo':os.path.exists({str(ROOT)!r}),'home':os.path.exists({str(Path.home() / '.ssh')!r}),"
        "'log':open('/workspace/history/logs.txt').read(),'cwd':os.getcwd()}\n"
        "open('/workspace/notes.md','w').write('ok')\n"
        "try:\n open('/workspace/history/logs.txt','a').write('x'); r['history_writable']=True\n"
        "except OSError: r['history_writable']=False\n"
        "import numpy, PIL; print(json.dumps(r))")
    command, env = box.wrap(["python", "-I", "-c", probe])
    out = subprocess.run(command, env=env, capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    result = json.loads(out.stdout)
    assert result == dict(repo=False, home=False, log="log", cwd="/workspace", history_writable=False)
    assert (workspace / "notes.md").read_text() == "ok"


def test_credential_mirror_writes_back_only_a_sandbox_refresh(tmp_path):
    source, target = tmp_path / "auth.json", tmp_path / "private" / "auth.json"
    target.parent.mkdir()
    source.write_text("v1")
    mirror = MirroredCredential(source, target)
    mirror.pull()
    mirror.push()
    assert target.read_text() == "v1" and source.read_text() == "v1"
    target.write_text("v2")              # the CLI refreshed its token inside the sandbox
    mirror.push()
    assert source.read_text() == "v2"
    source.write_text("v3")              # another process re-logged the profile
    target.write_text("v2-stale-refresh")
    mirror.push()
    assert source.read_text() == "v3"
    mirror.pull()
    assert target.read_text() == "v3"
