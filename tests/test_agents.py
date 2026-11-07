"""Model-agnostic agent layer: LLMPolicy over a harness, decoder, budgets, caps, legacy surface."""
import base64
import io
import json
from types import SimpleNamespace

import numpy as np
import pytest

import owb
from owb.agents import (ActionDecoder, Budgets, Decision, Harness, HarnessCaps, LLMPolicy,
                        LLMPolicyConfig, Part)
from owb.agents.harness import ClaudeCLIHarness, OpenAIResponsesHarness
from owb.bench import run_episode
from owb.bench.baselines import OpenAIPolicy, OpenAIPolicyConfig

pytest.importorskip("PIL")
from PIL import Image


def payload(**overrides):
    data = dict(move=[0.5, 0], look=[0, 0], buttons=[], command="none", swap=0,
                slot=0, ingredients=[0] * 5, part=0, build_move=[0] * 3, memory="Seen the start.")
    data.update(overrides)
    return data


class FakeHarness(Harness):
    caps = HarnessCaps(profile="cli-stateless", persistent_session=False, workspace=None,
                       tools=frozenset({"StructuredOutput"}), max_turns_per_decision=2)

    def __init__(self, output=None, tokens=(100, 50)):
        self.output = payload() if output is None else output
        self.tokens = tokens
        self.requests, self.opened = [], []

    def open(self, episode_seed):
        self.opened.append(episode_seed)

    def decide(self, system, parts, schema):
        self.requests.append((system, parts, schema))
        return Decision(payload=dict(self.output), model="fake-model", usage=dict(total=sum(self.tokens)),
                        latency_seconds=0.0, raw_id=f"fake-{len(self.requests)}", status="completed",
                        input_tokens=self.tokens[0], output_tokens=self.tokens[1])

    def metadata(self):
        return dict(name="fake")


@pytest.mark.parametrize("name", ["teyvat", "wildlands", "skylands"])
def test_llm_policy_steps_each_environment(name, tmp_path):
    harness = FakeHarness()
    policy = LLMPolicy(harness, config=LLMPolicyConfig(history_steps=1))
    with owb.make_env(name, image_size=(32, 24), max_steps=3) as env:
        summary = run_episode(env, policy, seed=7, output_dir=tmp_path / name)
        frame_shape = (env.config.image_size[1], env.config.image_size[0], 3)
    assert summary["steps"] == summary["policy"]["calls"] == 3
    assert summary["policy"]["input_tokens"] == 300 and summary["policy"]["output_tokens"] == 150
    assert harness.opened[-1] == 7
    system, parts, schema = harness.requests[-1]
    assert "Open-World Bench" in system
    assert [p.kind for p in parts] == ["text", "image_png_base64"]
    prompt = json.loads(parts[0].data)
    assert list(prompt) == ["step", "decision_seconds", "observation", "memory", "recent_history"]
    assert prompt["step"] == 2 and prompt["memory"] == "Seen the start." and len(prompt["recent_history"]) == 1
    image = np.array(Image.open(io.BytesIO(base64.b64decode(parts[1].data))))
    assert image.shape == frame_shape
    assert "none" in schema["properties"]["command"]["enum"]
    rows = [json.loads(s) for s in (tmp_path / name / "trajectory.jsonl").read_text().splitlines()]
    assert rows[1]["policy_decision"]["response_id"] == "fake-1"
    assert rows[1]["policy_decision"]["status"] == "completed"
    recorded = json.loads((tmp_path / name / "policy.json").read_text())
    assert recorded["privileged"] is False and recorded["harness"] == dict(name="fake")


def test_decoder_rejects_unsupported_command():
    with owb.make_env("wildlands", image_size=(32, 24), max_steps=3) as env:
        observation, _ = env.reset(seed=7)
    decoder = ActionDecoder(1600)
    with pytest.raises(ValueError, match="unsupported command"):
        decoder.decode(payload(command="teleport"), observation)
    policy = LLMPolicy(FakeHarness(payload(command="teleport")))
    with pytest.raises(ValueError, match="unsupported command"):
        policy(observation)
    assert policy.last_decision["status"] == "invalid_action" and policy.memory == ""


def test_token_budget_stops_episode(tmp_path):
    policy = LLMPolicy(FakeHarness(tokens=(100, 50)),
                       config=LLMPolicyConfig(budgets=Budgets(max_total_tokens=200)))
    with owb.make_env("skylands", image_size=(32, 24), max_steps=5) as env:
        with pytest.raises(RuntimeError, match="token budget"):
            run_episode(env, policy, seed=7, output_dir=tmp_path / "budget")
        assert env.steps == 2
    error = json.loads((tmp_path / "budget" / "error.json").read_text())
    assert error["policy"]["calls"] == 2 and error["policy"]["input_tokens"] == 200
    with pytest.raises(ValueError):
        Budgets(max_total_tokens=0)


def test_caps_reported_in_metadata(tmp_path):
    policy = LLMPolicy(FakeHarness())
    meta = policy.metadata()
    assert meta["privileged"] is False
    assert meta["caps"] == dict(profile="cli-stateless", isolation="process", persistent_session=False,
                                workspace=None, persist_workspace="none", tools=["StructuredOutput"],
                                mcp_tools=[], network="model-endpoint", max_turns_per_decision=2,
                                privileged=False)
    assert meta["config"]["budgets"] == dict(max_calls=100, max_total_tokens=None)
    api = LLMPolicy(OpenAIResponsesHarness(client=SimpleNamespace()))
    assert api.metadata()["caps"]["profile"] == "api-stateless"
    assert api.metadata()["harness"]["name"] == "openai_responses"
    account = tmp_path / "claude"
    account.mkdir()
    (account / ".credentials.json").write_text("{}")
    cli = ClaudeCLIHarness(account, executable="python")
    assert cli.caps.profile == "cli-stateless" and cli.caps.max_turns_per_decision == 3
    assert cli.caps.tools == frozenset({"StructuredOutput"}) and not cli.caps.persistent_session
    with pytest.raises(ValueError):
        HarnessCaps(profile="cloud", persistent_session=False, workspace=None,
                    tools=frozenset(), max_turns_per_decision=1)


def test_cli_harness_decide_returns_decision(tmp_path):
    account = tmp_path / "claude"
    account.mkdir()
    (account / ".credentials.json").write_text("{}")
    seen = {}

    def runner(command, **kwargs):
        seen.update(command=command, message=json.loads(kwargs["input"]))
        events = [dict(type="system", subtype="init", tools=["StructuredOutput"]),
                  dict(type="assistant", message=dict(model="claude-opus-5-5", content=[])),
                  dict(type="result", subtype="success", is_error=False, structured_output=payload(),
                       session_id="s1", usage=dict(input_tokens=4, cache_read_input_tokens=6, output_tokens=2))]
        return SimpleNamespace(returncode=0, stdout="\n".join(map(json.dumps, events)), stderr="")

    harness = ClaudeCLIHarness(account, executable="python", runner=runner)
    decision = harness.decide("sys", [Part("text", "hi"), Part("image_png_base64", "QUJD")], {"type": "object"})
    assert decision.payload == payload() and decision.raw_id == "s1" and decision.status == "completed"
    assert (decision.input_tokens, decision.output_tokens) == (10, 2)
    assert seen["message"]["message"]["content"][1]["source"]["data"] == "QUJD"
    assert seen["command"][seen["command"].index("--max-turns") + 1] == "3"


def test_openai_policy_metadata_keys_unchanged():
    client = SimpleNamespace(responses=SimpleNamespace(create=None))
    meta = OpenAIPolicy(OpenAIPolicyConfig(), client=client).metadata()
    # Pre-refactor snapshot: no privileged/caps/harness keys on the legacy names.
    assert set(meta) == {"type", "config", "calls", "input_tokens", "output_tokens", "latency_seconds"}
    assert meta["type"] == "openai"
    assert set(meta["config"]) == {"model", "reasoning_effort", "max_output_tokens", "max_calls",
                                   "history_steps", "memory_chars", "timeout", "image_detail",
                                   "decision_seconds"}
    assert isinstance(OpenAIPolicy(client=client), LLMPolicy)
