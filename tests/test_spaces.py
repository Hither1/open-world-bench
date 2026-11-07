"""Declarative modalities/controls: spaces, masks, descriptions and vocabularies come from one place."""
from dataclasses import dataclass
import json

from gymnasium import spaces
import numpy as np
import pytest

from owb import spaces as owb_spaces
from owb.agents.perception import describe_observation
from owb.bench import framework
from owb.bench.framework import (
    BUTTONS, COMMANDS, MODALITIES, EnvironmentSpec, NativeAdapter, make_env, neutral_action,
    register_environment, schema_vocabularies,
)
from owb.bench.rollout import RandomPolicy, run_episode
from owb.spaces import Control, Modality, SpaceRegistry
from owb.spaces.builtin import CONTROLS, MODALITY_SPECS


def test_builtin_declarations_are_the_canonical_schema_v2_order():
    assert owb_spaces.modality_names()[:len(MODALITIES)] == MODALITIES == (
        "party", "inventory", "survival", "minimap", "goal", "construction", "build_state")
    assert tuple(c.name for c in owb_spaces.controls()[:len(CONTROLS)]) == (
        "move", "look", "buttons", "command", "swap", "slot", "ingredients", "part", "build_move")
    assert owb_spaces.REGISTRY.control("buttons").vocab is BUTTONS
    assert owb_spaces.REGISTRY.control("command").vocab is COMMANDS
    adapters = {s.name: s.adapter for s in framework.environment_specs()}
    assert [m.name for m in adapters["skylands"].modality_specs] == [
        "inventory", "survival", "minimap", "construction", "build_state"]
    assert adapters["wildlands"].modalities == ("inventory", "survival", "minimap")
    declared = [m for a in adapters.values() for m in a.modality_specs]
    assert {m.name for m in declared} == set(MODALITIES)
    assert all(any(m is b for b in MODALITY_SPECS) for m in declared)


def test_env_spaces_and_neutral_action_follow_declarations():
    with make_env("wildlands", image_size=(32, 24), max_steps=2) as env:
        for modality in MODALITY_SPECS:
            assert env.observation_space[modality.name] == modality.space
            assert env.observation_space[modality.name] is not modality.space  # per-env copy
        for control in CONTROLS:
            assert env.action_space[control.name] == control.space
        action = neutral_action()
        assert list(action) == [c.name for c in CONTROLS]
        assert env.action_space.contains(action)
        obs, _ = env.reset(seed=1)
        assert obs["modality_mask"].tolist() == [0, 1, 1, 1, 0, 0, 0]
        assert len(obs["command_mask"]) == len(COMMANDS) and len(obs["swap_mask"]) == 5


def test_registry_is_idempotent_and_rejects_conflicts():
    registry = SpaceRegistry()
    a = Modality("lidar", spaces.Box(0, 1, (4,), np.float32), None, vocab=dict(lidar=("a", "b", "c", "d")))
    registry.register_modality(a)
    registry.register_modality(a)
    registry.register_modality(Modality("lidar", spaces.Box(0, 1, (4,), np.float32), None,
                                        vocab=dict(lidar=("a", "b", "c", "d"))))  # equal declaration
    assert registry.modality_names() == ("lidar",)
    with pytest.raises(ValueError, match="different declaration"):
        registry.register_modality(Modality("lidar", spaces.Box(0, 1, (5,), np.float32), None))
    with pytest.raises(ValueError, match="vocabulary keys"):
        registry.register_modality(Modality("sonar", spaces.Box(0, 1, (1,), np.float32), None, vocab=dict(lidar=())))
    with pytest.raises(ValueError, match="vocabulary keys"):
        registry.register_modality(Modality("sonar", spaces.Box(0, 1, (1,), np.float32), None, vocab=dict(commands=())))
    with pytest.raises(ValueError, match="shared field"):
        registry.register_modality(Modality("rgb", spaces.Box(0, 1, (1,), np.float32), None))
    registry.register_control(Control("brake", spaces.Discrete(2), "stop"))
    with pytest.raises(ValueError, match="different declaration"):
        registry.register_control(Control("brake", spaces.Discrete(3), "stop"))


# ------------------------------------------------------------- a new backend, declaratively

@dataclass
class _ToyConfig:
    seed: int = 0


class _ToyNative:
    """Minimal native env: a point moving on a line with a range sensor."""
    def __init__(self, config):
        self.cfg, self.size = _ToyConfig(config.seed), config.image_size
        self.x = self.t = 0.0
        self.steps = 0

    def _obs(self):
        return dict(rgb=np.zeros((self.size[1], self.size[0], 3), np.uint8),
                    ranges=np.array([self.x, 1 - self.x, 0.5], np.float32))

    def reset(self, seed=None):
        self.x = self.t = 0.0
        self.steps = 0
        return self._obs(), {}

    def step(self, action):
        self.x += 0.1 * float(action["move"][0])
        self.t += 0.2
        self.steps += 1
        return self._obs(), float(self.x), False, False, dict(events=[dict(type="tick")])

    def render(self):
        return self._obs()["rgb"]

    def close(self):
        pass


RANGES = Modality("ranges", spaces.Box(-np.inf, np.inf, (3,), np.float32),
                  lambda value, observation: {"ranges": dict(zip(("ahead", "behind", "up"), value.tolist()))},
                  vocab=dict(ranges=("ahead", "behind", "up")))


class _ToyAdapter(NativeAdapter):
    modality_specs = (RANGES,)
    control_specs = (Control("brake", spaces.Discrete(2), "1 stops the point"),)

    def translate(self, action):
        return action

    def state(self):
        n = self.native
        return (np.array([0.1, 0, 0]), np.array([1, 0, 0]), np.array([0, 0, 1]), 1.0, 1.0, 1.0, 1.0,
                True, False, False, False, False, n.t / 10)

    def project(self, observation):
        return dict(instruction="move", ranges=observation["ranges"])

    def metrics(self, info):
        return None, None, {}

    def elapsed(self):
        return self.native.t


@pytest.fixture
def isolated_registries(monkeypatch):
    monkeypatch.setattr(owb_spaces, "REGISTRY", owb_spaces.REGISTRY.copy())
    monkeypatch.setattr(framework, "_REGISTRY", dict(framework._REGISTRY))


def test_new_backend_declares_modality_without_central_edits(isolated_registries, tmp_path):
    register_environment(EnvironmentSpec("toy", lambda config, native: _ToyNative(config), _ToyAdapter, "toy"))
    assert owb_spaces.modality_names() == MODALITIES + ("ranges",)
    with make_env("toy", image_size=(8, 6), max_steps=3) as env:
        assert env.observation_space["ranges"] == RANGES.space
        assert env.observation_space["modality_mask"].n == len(MODALITIES) + 1
        assert "brake" in env.action_space.spaces and neutral_action()["brake"] == 0
        obs, _ = env.reset(seed=2)
        assert obs["modality_mask"].tolist() == [0] * len(MODALITIES) + [1]
        assert not obs["party"].any()
        described = describe_observation(obs)
        assert described["ranges"] == {"ahead": 0.0, "behind": 1.0, "up": 0.5}
        assert "inventory" not in described
        summary = run_episode(env, RandomPolicy(), seed=2, output_dir=tmp_path / "toy")
    assert summary["steps"] == 3 and summary["event_counts"]["tick"] == 3
    schema = json.loads((tmp_path / "toy" / "config.json").read_text())["schema"]
    assert schema["modalities"][-1] == "ranges" and schema["ranges"] == ["ahead", "behind", "up"]
    assert schema_vocabularies()["ranges"] == ("ahead", "behind", "up")
    # Built-in envs built after the registration expose the union; old indices do not move.
    with make_env("wildlands", image_size=(32, 24), max_steps=2) as env:
        obs, _ = env.reset(seed=1)
        assert obs["modality_mask"].tolist() == [0, 1, 1, 1, 0, 0, 0, 0]


def test_conflicting_backend_registers_nothing(isolated_registries):
    bad = Modality("ranges", spaces.Box(0, 1, (3,), np.float32), None)

    class Conflicting(_ToyAdapter):
        modality_specs = (RANGES, Modality("echo", spaces.Box(0, 1, (1,), np.float32), None), bad)

    with pytest.raises(ValueError, match="different declaration"):
        register_environment(EnvironmentSpec("bad", lambda c, n: _ToyNative(c), Conflicting, "bad"))
    assert owb_spaces.modality_names() == MODALITIES
    assert "bad" not in framework.available_environments()
