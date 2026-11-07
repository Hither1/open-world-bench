"""Golden snapshot of the common-schema and native gym interfaces (behaviour freeze).

For every name in ``owb.available_environments()`` the test builds
``make_env(name, config=BenchmarkConfig(seed=7, max_steps=150, image_size=(64, 48)))``
and runs ``run_episode`` with the seeded ``RandomPolicy`` for the whole episode.
For each native gym id registered by ``owb`` (``OpenWorldBench/Genshin-Small-v0``,
``Wildlands-Small-v0``, ``Skylands-Small-v0``) it calls ``gym.make(id)``,
``reset(seed=7)`` and takes 60 actions sampled from ``action_space`` seeded with 7.

Recorded per environment: ``repr`` of ``observation_space`` and ``action_space``,
``SCHEMA_VERSION`` (common schema only), the episode summary, and for the reset
plus every step a sha256 of the action, of the observation without ``rgb``, of
``rgb`` alone, and of ``jsonable(info)`` (which includes ``info["native"]``),
alongside reward/terminated/truncated. Splitting ``rgb`` out lets a failure say
whether simulation or only sensor rendering changed.

Hashing is canonical and exact: numpy arrays contribute dtype, shape and the
raw bytes of ``np.ascontiguousarray(a)``; Python floats contribute
``float.hex()``; dict keys are sorted; bool/int/str/None are tagged by type.
An unsupported leaf type raises instead of being hashed by ``repr``.

Snapshots are MACHINE-SPECIFIC. Raw float bytes and rendered pixels depend on
CPU architecture, numpy/BLAS build, gymnasium version (space reprs and
``Space.sample`` RNG) and Python version. They were generated on the
developer's Apple Silicon macOS machine and are meant to freeze behaviour
across a refactor on that machine, not to be portable. Regenerate deliberately,
only when a behaviour change is intended (or on a new machine), with::

    OWB_UPDATE_SNAPSHOTS=1 python -m pytest tests/test_contract_snapshot.py

and review the diff of ``tests/snapshots/`` before committing it. Observation
rendering here is the cheap sensor renderer; nothing invokes Blender.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import struct

import gymnasium as gym
import numpy as np
import pytest

import owb  # noqa: F401  (registers the native gym ids)
from owb.bench import BenchmarkConfig, available_environments, make_env
from owb.bench.framework import SCHEMA_VERSION, jsonable
from owb.bench.rollout import RandomPolicy, run_episode

SNAPSHOTS = Path(__file__).resolve().parent / "snapshots"
UPDATE = os.environ.get("OWB_UPDATE_SNAPSHOTS") == "1"
COMMON_CONFIG = dict(seed=7, max_steps=150, image_size=(64, 48))
NATIVE_IDS = ("OpenWorldBench/Genshin-Small-v0", "OpenWorldBench/Wildlands-Small-v0",
              "OpenWorldBench/Skylands-Small-v0")
NATIVE_SEED, NATIVE_STEPS = 7, 60


# -------------------------------------------------------------- hashing

def _feed(h, value):
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        h.update(b"A" + array.dtype.str.encode() + repr(array.shape).encode() + b":")
        h.update(array.tobytes())
    elif isinstance(value, np.generic):
        h.update(b"G" + value.dtype.str.encode() + b":")
        h.update(np.ascontiguousarray(value).tobytes())
    elif isinstance(value, dict):
        h.update(b"{%d:" % len(value))
        for key in sorted(value, key=str):
            _feed(h, str(key))
            _feed(h, value[key])
        h.update(b"}")
    elif isinstance(value, (list, tuple)):
        h.update(b"[%d:" % len(value))
        for item in value:
            _feed(h, item)
        h.update(b"]")
    elif value is None:
        h.update(b"N")
    elif isinstance(value, bool):
        h.update(b"T" if value else b"F")
    elif isinstance(value, int):
        h.update(b"I" + str(value).encode() + b";")
    elif isinstance(value, float):
        h.update(b"D" + struct.pack("<d", value))
    elif isinstance(value, str):
        data = value.encode()
        h.update(b"S%d:" % len(data) + data)
    else:
        raise TypeError(f"no canonical hash for {type(value).__name__}; extend _feed deliberately")


def digest(value):
    h = hashlib.sha256()
    _feed(h, value)
    return h.hexdigest()


def _hex(value):
    return float(value).hex()


def _transition(action, observation, info, reward=None, terminated=None, truncated=None):
    obs = dict(observation)
    rgb = obs.pop("rgb", None)
    row = dict(action=None if action is None else digest(action), obs=digest(obs),
               rgb=None if rgb is None else digest(rgb), info=digest(jsonable(info)))
    if reward is not None:
        row.update(reward=_hex(reward), terminated=bool(terminated), truncated=bool(truncated))
    return row


# ------------------------------------------------------------ recording

class _Recorder:
    """Transparent proxy that records what run_episode sees."""
    def __init__(self, env):
        self._env, self.rows = env, []

    def __getattr__(self, name):
        return getattr(self._env, name)

    def reset(self, **kwargs):
        observation, info = self._env.reset(**kwargs)
        self.rows.append(_transition(None, observation, info))
        return observation, info

    def step(self, action):
        result = self._env.step(action)
        observation, reward, terminated, truncated, info = result
        self.rows.append(_transition(action, observation, info, reward, terminated, truncated))
        return result


def record_common(name):
    env = make_env(name, config=BenchmarkConfig(**COMMON_CONFIG))
    try:
        proxy = _Recorder(env)
        summary = run_episode(proxy, RandomPolicy(), seed=COMMON_CONFIG["seed"])
        return dict(kind="common", environment=name, config=jsonable(COMMON_CONFIG),
                    schema_version=SCHEMA_VERSION,
                    observation_space=repr(env.observation_space), action_space=repr(env.action_space),
                    summary_digest=digest(jsonable(summary)), summary=_json(summary),
                    steps=len(proxy.rows) - 1, transitions=proxy.rows)
    finally:
        env.close()


def record_native(env_id):
    env = gym.make(env_id)
    try:
        observation, info = env.reset(seed=NATIVE_SEED)
        env.action_space.seed(NATIVE_SEED)
        rows = [_transition(None, observation, info)]
        for _ in range(NATIVE_STEPS):
            action = env.action_space.sample()
            observation, reward, terminated, truncated, info = env.step(action)
            rows.append(_transition(action, observation, info, reward, terminated, truncated))
            if terminated or truncated:
                break
        return dict(kind="native", environment=env_id, seed=NATIVE_SEED, requested_steps=NATIVE_STEPS,
                    observation_space=repr(env.observation_space), action_space=repr(env.action_space),
                    steps=len(rows) - 1, transitions=rows)
    finally:
        env.close()


def _json(value):
    """Human-readable copy of small payloads; floats survive JSON exactly."""
    return json.loads(json.dumps(jsonable(value), allow_nan=False))


# ------------------------------------------------------------ comparison

def _snapshot_path(label):
    return SNAPSHOTS / (label.replace("OpenWorldBench/", "native_").replace("/", "_") + ".json")


def check(label, actual):
    path = _snapshot_path(label)
    if UPDATE:
        SNAPSHOTS.mkdir(exist_ok=True)
        path.write_text(json.dumps(actual, indent=1, sort_keys=True, allow_nan=False) + "\n")
        return
    if not path.exists():
        pytest.fail(f"missing snapshot {path.name}; generate with OWB_UPDATE_SNAPSHOTS=1")
    expected = json.loads(path.read_text())
    actual = json.loads(json.dumps(actual, sort_keys=True, allow_nan=False))
    # Interface first, then the first diverging transition, then episode totals,
    # so a failure names the earliest observable difference.
    outcome = ("transitions", "steps", "summary", "summary_digest")
    for key in sorted((set(expected) | set(actual)) - set(outcome)):
        assert actual.get(key) == expected.get(key), f"{label}: {key} differs from snapshot"
    exp_rows, act_rows = expected["transitions"], actual["transitions"]
    for i, (e, a) in enumerate(zip(exp_rows, act_rows)):
        if e != a:
            fields = sorted(k for k in set(e) | set(a) if e.get(k) != a.get(k))
            where = "reset" if i == 0 else f"step {i}"
            pytest.fail(f"{label}: first divergence at {where} in {fields}")
    assert len(act_rows) == len(exp_rows), f"{label}: transition count differs"
    for key in outcome[1:]:
        assert actual.get(key) == expected.get(key), f"{label}: {key} differs from snapshot"


@pytest.mark.parametrize("name", available_environments())
def test_common_schema_snapshot(name):
    check(name, record_common(name))


@pytest.mark.parametrize("env_id", NATIVE_IDS)
def test_native_gym_snapshot(env_id):
    assert env_id in gym.envs.registry
    check(env_id, record_native(env_id))


def test_snapshots_cover_every_environment():
    names = {p.stem for p in SNAPSHOTS.glob("*.json")}
    expected = {_snapshot_path(n).stem for n in (*available_environments(), *NATIVE_IDS)}
    assert names == expected
