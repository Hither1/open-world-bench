"""Codex transport isolation, failure handling, and CLI output parsing."""
import base64
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from owb.bench.baselines.codex_policy import CodexClient, CodexPolicyError, _run_cli


@pytest.fixture
def client(tmp_path):
    account = tmp_path / "account"
    account.mkdir()
    (account / "auth.json").write_text("{}")
    return CodexClient(account, executable="python", timeout=10)


def request():
    return dict(model="gpt-6-astra", instructions="Control the agent.",
                reasoning=dict(effort="low"), text=dict(format=dict(schema={"type": "object"})),
                input=[dict(content=[dict(type="input_text", text="sensor data"),
                       dict(type="input_image", image_url="data:image/png;base64," +
                            base64.b64encode(b"fake-image-bytes").decode())])])


def test_codex_uses_selected_account_image_schema_and_no_api_key(client, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "unused-test-key")
    monkeypatch.setenv("CODEX_API_KEY", "unused-test-key")
    original_home = os.environ.get("CODEX_HOME")
    directories = []

    def run(command, **kwargs):
        directories.append(kwargs["cwd"])
        assert kwargs["env"]["CODEX_HOME"] == str(client.account_home)
        assert "OPENAI_API_KEY" not in kwargs["env"]
        assert "CODEX_API_KEY" not in kwargs["env"]
        assert "--ignore-user-config" in command and "--ephemeral" in command
        assert "features.shell_tool=false" in command
        assert "features.multi_agent=false" in command
        assert command[command.index("--model") + 1] == "gpt-6-astra"
        assert Path(command[command.index("--image") + 1]).read_bytes() == b"fake-image-bytes"
        output = Path(command[command.index("--output-last-message") + 1])
        output.write_text('{"memory":"observed"}')
        events = [dict(type="thread.started", thread_id="codex-thread"),
                  dict(type="item.completed", item=dict(type="agent_message", text="{}")),
                  dict(type="turn.completed", usage=dict(input_tokens=150, output_tokens=25))]
        return SimpleNamespace(returncode=0, stdout="\n".join(map(json.dumps, events)), stderr="")

    monkeypatch.setattr("owb.bench.baselines.codex_policy._run_cli", run)
    result = client.create(**request())
    assert result.status == "completed" and result.id == "codex-thread"
    assert result.usage.input_tokens == 150 and result.usage.output_tokens == 25
    assert json.loads(result.output_text)["memory"] == "observed"
    assert not Path(directories[0]).exists()
    assert os.environ.get("CODEX_HOME") == original_home
    assert (client.account_home / "auth.json").read_text() == "{}"


def test_codex_model_upgrade_error(client, monkeypatch):
    event = dict(type="turn.failed", error=dict(message="Model requires a newer version of Codex"))
    monkeypatch.setattr("owb.bench.baselines.codex_policy._run_cli", lambda *args, **kwargs: SimpleNamespace(
        returncode=1, stdout=json.dumps(event), stderr="secret"))
    with pytest.raises(CodexPolicyError, match="newer Codex CLI") as error:
        client.create(**request())
    assert "secret" not in str(error.value)


def test_codex_rejects_tool_activity(client, monkeypatch):
    def run(command, **kwargs):
        Path(command[command.index("--output-last-message") + 1]).write_text("{}")
        events = [dict(type="item.completed", item=dict(type="command_execution")),
                  dict(type="turn.completed", usage={})]
        return SimpleNamespace(returncode=0, stdout="\n".join(map(json.dumps, events)), stderr="")
    monkeypatch.setattr("owb.bench.baselines.codex_policy._run_cli", run)
    with pytest.raises(CodexPolicyError, match="tool activity"):
        client.create(**request())


def test_cli_timeout_kills_launcher_and_child(tmp_path):
    # The child holds inherited stdout open; killing just the launcher would hang communicate().
    code = "import subprocess,sys,time; subprocess.Popen([sys.executable,'-c','import time; time.sleep(5)']); time.sleep(5)"
    with pytest.raises(subprocess.TimeoutExpired):
        _run_cli([sys.executable, "-c", code], input="", env=os.environ.copy(),
                 cwd=tmp_path, timeout=0.2)
