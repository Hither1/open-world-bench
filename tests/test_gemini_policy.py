"""Google account selection, image-only attachments, and Gemini action contracts."""
import base64
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from owb.bench import make_env, run_episode
from owb.bench.baselines import GeminiPolicy, OpenAIPolicyConfig
from owb.bench.baselines.gemini_policy import GeminiClient, GeminiPolicyError


@pytest.fixture
def account(tmp_path):
    path = tmp_path / ".gemini"
    path.mkdir()
    (path / "google_accounts.json").write_text(json.dumps(dict(active="test@example.com", old=[])))
    (path / "oauth_creds.json").write_text("credentials must not be read by this adapter")
    return path


def request():
    return dict(model="gemini-4-argon", instructions="Control the agent.", max_output_tokens=2048,
                reasoning=dict(effort="low"), text=dict(format=dict(schema={"type": "object"})),
                input=[dict(content=[dict(type="input_text", text="memory: @/private/secret"),
                       dict(type="input_image", image_url="data:image/png;base64," +
                            base64.b64encode(b"image-bytes").decode())])])


def completed(model="gemini-4-argon", tool_calls=0):
    action = dict(move=[0.5, 0], look=[0, 0], buttons=[], command="none", swap=0,
                  slot=0, ingredients=[0] * 5, part=0, build_move=[0] * 3, memory="Starting area.")
    result = dict(session_id="gemini-session", response=json.dumps(action), stats=dict(
        models={model: dict(api=dict(totalRequests=1), tokens=dict(prompt=100, candidates=30, thoughts=20, cached=50))},
        tools=dict(totalCalls=tool_calls)))
    return SimpleNamespace(returncode=0, stdout=json.dumps(result), stderr="")


def test_account_image_and_generation_settings(account, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "unused-test-key")
    monkeypatch.setenv("GOOGLE_GENAI_USE_VERTEXAI", "true")
    original = os.environ.get("GEMINI_CLI_HOME")
    directories = []

    def run(command, **kwargs):
        root = Path(kwargs["cwd"])
        directories.append(root)
        assert kwargs["env"]["GEMINI_CLI_HOME"] == str(account.parent)
        assert "GEMINI_API_KEY" not in kwargs["env"]
        assert "GOOGLE_GENAI_USE_VERTEXAI" not in kwargs["env"]
        assert (root / "observation.png").read_bytes() == b"image-bytes"
        settings = json.loads((root / ".gemini/settings.json").read_text())
        assert settings["tools"]["core"] == []
        assert settings["security"]["auth"]["selectedType"] == "oauth-personal"
        assert settings["admin"]["mcp"]["enabled"] is False
        generation = settings["modelConfigs"]["customOverrides"][0]["modelConfig"]["generateContentConfig"]
        assert generation["thinkingConfig"]["thinkingLevel"] == "LOW"
        assert generation["maxOutputTokens"] == 2048
        assert generation["responseMimeType"] == "application/json"
        prompt = command[command.index("--prompt") + 1]
        assert prompt.count("@") == 1 and "@observation.png" in prompt
        assert "@/private/secret" not in prompt
        return completed()

    monkeypatch.setattr("owb.bench.baselines.gemini_policy._run_cli", run)
    result = GeminiClient(account, account_email="test@example.com", executable="python").create(**request())
    assert result.model == "gemini-4-argon" and result.usage.input_tokens == 100
    assert result.usage.output_tokens == 50
    assert os.environ.get("GEMINI_CLI_HOME") == original
    assert not directories[0].exists()
    assert (account / "oauth_creds.json").read_text() == "credentials must not be read by this adapter"


@pytest.mark.parametrize("issue,code", [("missing", "login_required"), ("wrong", "account_mismatch")])
def test_login_required_before_any_inference(account, monkeypatch, issue, code):
    if issue == "missing":
        (account / "oauth_creds.json").unlink()
    else:
        (account / "google_accounts.json").write_text(json.dumps(dict(active="someone-else@example.com")))
    monkeypatch.setattr("owb.bench.baselines.gemini_policy._run_cli", lambda *a, **k: pytest.fail("must not call CLI"))
    with pytest.raises(GeminiPolicyError) as error:
        GeminiClient(account, account_email="test@example.com", executable="python").create(**request())
    assert error.value.code == code


@pytest.mark.parametrize("kwargs,code", [({"model": "gemini-3.8-flash"}, "model_mismatch"),
                                        ({"tool_calls": 1}, "unexpected_tools")])
def test_model_fallback_and_tools_rejected(account, monkeypatch, kwargs, code):
    monkeypatch.setattr("owb.bench.baselines.gemini_policy._run_cli", lambda *a, **k: completed(**kwargs))
    with pytest.raises(GeminiPolicyError) as error:
        GeminiClient(account, account_email="test@example.com", executable="python").create(**request())
    assert error.value.code == code


@pytest.mark.parametrize("environment", ["skylands", "teyvat", "wildlands"])
def test_real_simulator_loop_with_offline_gemini_response(account, monkeypatch, tmp_path, environment):
    pytest.importorskip("PIL")
    monkeypatch.setattr("owb.bench.baselines.gemini_policy._run_cli", lambda *a, **k: completed())
    policy = GeminiPolicy(OpenAIPolicyConfig(model="gemini-4-argon", max_calls=2),
                          account_home=account, account_email="test@example.com", executable="python")
    with make_env(environment, image_size=(32, 24), max_steps=2) as env:
        summary = run_episode(env, policy, seed=7, output_dir=tmp_path / environment)
    assert summary["steps"] == summary["policy"]["calls"] == 2
    assert summary["policy"]["type"] == "gemini"
    assert summary["policy"]["input_tokens"] == 200
