"""Offline API contract and real simulator integration for the vision policy."""
import base64
from copy import deepcopy
import io
import json
from types import SimpleNamespace

import numpy as np
import pytest

from owb.bench import COMMANDS, MODALITIES, make_env
from owb.bench.baselines import OpenAIPolicy, OpenAIPolicyConfig
from owb.bench.baselines.openai_policy import decode_action, describe_observation
from owb.bench.rollout import run_episode

pytest.importorskip("PIL")
from PIL import Image


def payload(**overrides):
    return dict(move=[0.5, 0], look=[0, 0], buttons=[], command="none", swap=0,
                slot=0, ingredients=[0] * 5, part=0, build_move=[0] * 3,
                memory="Remember the starting area.", **overrides)


class FakeClient:
    def __init__(self, output=None, status="completed"):
        self.output = payload() if output is None else output
        self.status = status
        self.requests = []
        self.responses = self

    def create(self, **kwargs):
        self.requests.append(deepcopy(kwargs))
        usage = SimpleNamespace(input_tokens=100, output_tokens=50,
                                model_dump=lambda: dict(input_tokens=100, output_tokens=50))
        return SimpleNamespace(id="resp_test", model=kwargs["model"], status=self.status,
                               output_text=json.dumps(self.output), usage=usage)


@pytest.fixture(scope="module")
def observation():
    with make_env("skylands", image_size=(32, 24), max_steps=2) as env:
        obs, _ = env.reset(seed=7)
        return obs


@pytest.mark.parametrize("name", ["teyvat", "wildlands", "skylands"])
def test_real_env_rollout_with_model_actions(name, tmp_path):
    client = FakeClient()
    policy = OpenAIPolicy(OpenAIPolicyConfig(max_calls=2), client=client)
    with make_env(name, image_size=(32, 24), max_steps=2) as env:
        summary = run_episode(env, policy, seed=7, output_dir=tmp_path / name)
    assert summary["steps"] == summary["policy"]["calls"] == 2
    assert summary["policy"]["input_tokens"] == 200
    rows = [json.loads(s) for s in (tmp_path / name / "trajectory.jsonl").read_text().splitlines()]
    assert rows[1]["policy_decision"]["response_id"] == "resp_test"
    assert rows[2]["action"]["move"] == [0.5, 0.0]
    assert (tmp_path / name / "policy.json").is_file()
    assert not (tmp_path / name / "error.json").exists()


def test_image_sensors_memory_and_request_contract(observation):
    client = FakeClient()
    policy = OpenAIPolicy(OpenAIPolicyConfig(history_steps=1, max_calls=3), client=client)
    for _ in range(3):
        policy(observation)
    request = client.requests[-1]
    assert request["model"] == "gpt-6-astra" and request["store"] is False
    assert "previous_response_id" not in request
    assert request["text"]["format"]["strict"] is True
    text_part, image_part = request["input"][0]["content"]
    prompt = json.loads(text_part["text"])
    assert prompt["memory"] == payload()["memory"]
    assert len(prompt["recent_history"]) == 1
    assert "oracle_goal_hint" not in prompt["observation"]
    assert "native" not in prompt["observation"]
    image_bytes = base64.b64decode(image_part["image_url"].split(",")[1])
    np.testing.assert_array_equal(np.array(Image.open(io.BytesIO(image_bytes))), observation["rgb"])
    with pytest.raises(RuntimeError, match="call limit"):
        policy(observation)
    assert len(client.requests) == 3
    policy.reset(7)
    assert policy.memory == "" and not policy.history and policy.calls == 0
    assert policy.metadata()["input_tokens"] == 0


@pytest.mark.parametrize("patch", [
    {"move": [float("nan"), 0]}, {"move": [1.1, 0]}, {"look": [0]},
    {"ingredients": [0.5, 0, 0, 0, 0]}, {"ingredients": [33, 0, 0, 0, 0]},
    {"swap": True}, {"slot": -1}, {"part": 9}, {"buttons": ["teleport"]},
    {"command": "skill"}, {"memory": "x" * 1601}, {"unexpected": 1},
])
def test_bad_model_controls_rejected(observation, patch):
    data = payload()
    data.update(patch)
    with pytest.raises(ValueError):
        decode_action(data, observation, 1600)


def test_masks_and_local_construction(observation):
    observation = deepcopy(observation)
    data = payload()
    data.update(command="grab", part=1, move=[0, 0])
    action = decode_action(data, observation, 1600)
    assert action["command"] == COMMANDS.index("grab")
    assert action["move"].dtype == np.float32
    assert action["buttons"].dtype == np.int8
    sensors = describe_observation(observation)
    assert sensors["construction"] and sensors["build_state"]["held_part"] == 0
    observation["modality_mask"][MODALITIES.index("construction")] = 0
    assert "construction" not in describe_observation(observation)
    observation["button_mask"][:] = 0
    data["buttons"] = ["jump"]
    with pytest.raises(ValueError, match="unsupported button"):
        decode_action(data, observation, 1600)


def test_incomplete_response_stops_before_env_step(tmp_path):
    policy = OpenAIPolicy(client=FakeClient(status="incomplete"))
    path = tmp_path / "failed"
    with make_env("skylands", image_size=(32, 24), max_steps=1) as env:
        with pytest.raises(RuntimeError, match="no completed action"):
            run_episode(env, policy, seed=7, output_dir=path)
        assert env.steps == 0
    error = json.loads((path / "error.json").read_text())
    assert error["status"] == "error" and error["policy"]["calls"] == 1
    assert error["policy"]["output_tokens"] == 50
    assert not (path / "summary.json").exists()


def test_api_failure_is_not_a_scripted_fallback(observation):
    class BrokenClient:
        def create(self, **kwargs):
            raise ConnectionError("offline")
    client = SimpleNamespace(responses=BrokenClient())
    policy = OpenAIPolicy(client=client)
    with pytest.raises(ConnectionError):
        policy(observation)
    assert policy.calls == 1 and policy.memory == ""
    assert policy.last_decision["status"] == "error"
    assert policy.last_decision["error_type"] == "ConnectionError"


def test_quota_error_records_actionable_code_without_raw_message(tmp_path):
    class QuotaError(Exception):
        code = "credit_balance_exhausted"
        status_code = 429

    class Client:
        def create(self, **kwargs):
            raise QuotaError("sensitive request details")

    policy = OpenAIPolicy(client=SimpleNamespace(responses=Client()))
    path = tmp_path / "quota"
    with make_env("skylands", image_size=(32, 24), max_steps=1) as env:
        with pytest.raises(QuotaError):
            run_episode(env, policy, output_dir=path)
        assert env.steps == 0
    raw = (path / "error.json").read_text()
    error = json.loads(raw)
    assert error["last_decision"]["error_code"] == "credit_balance_exhausted"
    assert error["last_decision"]["http_status"] == 429
    assert "sensitive request details" not in raw


@pytest.mark.parametrize("kwargs", [{"max_calls": 0}, {"history_steps": -1},
                                    {"timeout": float("nan")}, {"decision_seconds": 0}])
def test_invalid_config(kwargs):
    with pytest.raises(ValueError):
        OpenAIPolicyConfig(**kwargs)
