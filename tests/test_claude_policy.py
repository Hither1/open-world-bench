"""Claude login isolation, image transport, model identity, and simulator integration."""
import base64
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from owb.bench import make_env, run_episode
from owb.bench.baselines import ClaudePolicy, OpenAIPolicyConfig
from owb.bench.baselines.claude_policy import ClaudeClient, ClaudePolicyError


@pytest.fixture
def account(tmp_path):
    path = tmp_path / "claude-account"
    path.mkdir()
    (path / ".credentials.json").write_text("{}")
    return path


def request():
    return dict(model="claude-opus-5-5", instructions="Control the agent.",
                reasoning=dict(effort="low"), text=dict(format=dict(schema={"type": "object"})),
                input=[dict(content=[dict(type="input_text", text="sensor data"),
                       dict(type="input_image", image_url="data:image/png;base64," +
                            base64.b64encode(b"image-bytes").decode())])])


def completed(*, model="claude-opus-5-5", tool="StructuredOutput", error=None):
    action = dict(move=[0.5, 0], look=[0, 0], buttons=[], command="none", swap=0,
                  slot=0, ingredients=[0] * 5, part=0, build_move=[0] * 3, memory="Starting area.")
    events = [dict(type="system", subtype="init", tools=["StructuredOutput"], model=model),
              dict(type="assistant", message=dict(model=model, content=[dict(type="tool_use", name=tool)])),
              dict(type="result", subtype="success", is_error=bool(error), result=error or "",
                   structured_output=action, session_id="claude-session", usage=dict(
                       input_tokens=10, cache_read_input_tokens=80, cache_creation_input_tokens=20,
                       output_tokens=30))]
    return SimpleNamespace(returncode=int(bool(error)), stdout="\n".join(map(json.dumps, events)), stderr="")


def test_selected_login_image_schema_and_usage(account, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "unused-test-key")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://unused.invalid")
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "unused-test-token")
    original = os.environ.get("CLAUDE_CONFIG_DIR")
    directories = []

    def run(command, **kwargs):
        directories.append(kwargs["cwd"])
        assert kwargs["env"]["CLAUDE_CONFIG_DIR"] == str(account)
        assert not any(k.startswith("ANTHROPIC_") for k in kwargs["env"])
        assert "CLAUDE_CODE_OAUTH_TOKEN" not in kwargs["env"]
        assert "--safe-mode" in command and "--restricted" in command
        assert "--fallback-model" not in command and "--bare" not in command
        assert command[command.index("--tools") + 1] == ""
        assert command[command.index("--model") + 1] == "claude-opus-5-5"
        message = json.loads(kwargs["input"])
        content = message["message"]["content"]
        assert content[0] == dict(type="text", text="sensor data")
        assert base64.b64decode(content[1]["source"]["data"]) == b"image-bytes"
        return completed()

    monkeypatch.setattr("owb.bench.baselines.claude_policy._run_cli", run)
    result = ClaudeClient(account, executable="python").create(**request())
    assert result.model == "claude-opus-5-5" and result.id == "claude-session"
    assert result.usage.input_tokens == 110 and result.usage.output_tokens == 30
    assert os.environ.get("CLAUDE_CONFIG_DIR") == original
    assert not Path(directories[0]).exists()
    assert (account / ".credentials.json").read_text() == "{}"


@pytest.mark.parametrize("kwargs,code", [
    ({"model": "claude-sonnet-5-5"}, "model_mismatch"),
    ({"tool": "Bash"}, "unexpected_tools"),
    ({"error": "Your account is on hold and can't use Claude Code."}, "account_on_hold"),
])
def test_rejects_fallback_tools_and_account_failure(account, monkeypatch, kwargs, code):
    monkeypatch.setattr("owb.bench.baselines.claude_policy._run_cli", lambda *a, **k: completed(**kwargs))
    with pytest.raises(ClaudePolicyError) as error:
        ClaudeClient(account, executable="python").create(**request())
    assert error.value.code == code


@pytest.mark.parametrize("environment", ["skylands", "teyvat", "wildlands"])
def test_real_simulator_loop_with_offline_claude_response(account, monkeypatch, tmp_path, environment):
    pytest.importorskip("PIL")
    monkeypatch.setattr("owb.bench.baselines.claude_policy._run_cli", lambda *a, **k: completed())
    policy = ClaudePolicy(OpenAIPolicyConfig(model="claude-opus-5-5", max_calls=2),
                          account_home=account, executable="python")
    with make_env(environment, image_size=(32, 24), max_steps=2) as env:
        summary = run_episode(env, policy, seed=7, output_dir=tmp_path / environment)
    assert summary["steps"] == summary["policy"]["calls"] == 2
    assert summary["policy"]["type"] == "claude"
    assert summary["policy"]["input_tokens"] == 220
    assert summary["policy"]["output_token_limit_enforced"] is False
    assert "max_output_tokens" not in summary["policy"]["config"]


def test_account_hold_never_advances_environment(account, monkeypatch, tmp_path):
    pytest.importorskip("PIL")
    monkeypatch.setattr("owb.bench.baselines.claude_policy._run_cli", lambda *a, **k: completed(
        error="Your account is on hold and can't use Claude Code."))
    policy = ClaudePolicy(account_home=account, executable="python")
    output = tmp_path / "failed"
    with make_env("skylands", image_size=(32, 24), max_steps=1) as env:
        with pytest.raises(ClaudePolicyError):
            run_episode(env, policy, output_dir=output)
        assert env.steps == 0
    failure = json.loads((output / "error.json").read_text())
    assert failure["last_decision"]["error_code"] == "account_on_hold"
    assert not (output / "summary.json").exists()
