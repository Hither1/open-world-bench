"""Re-run recorded actions and check that the simulation reproduces the trajectory.

Generalises the digest replay of ``owb.scripts.teyvat_reference`` to every
``run_episode`` recording: rebuild the env from ``config.json`` (common and native
configuration, run seed), feed the ``trajectory.jsonl`` actions, and compare each
transition's ``proprio``, ``reward`` and ``events`` (optionally the full ``info``).
Like the recordings themselves, this assumes the same code and dependency versions.
"""
from __future__ import annotations

from dataclasses import asdict, fields, is_dataclass, replace
import enum
import json
from pathlib import Path

import numpy as np

from ..bench.framework import BenchmarkConfig, make_env
from ..core import jsonable

DEFAULT_FIELDS = ("proprio", "reward", "events")
FIELDS = DEFAULT_FIELDS + ("info",)


def _canonical(value):
    return json.loads(json.dumps(jsonable(value), allow_nan=False))


def as_action(env, action):
    """Common-schema action from a recorded (JSON) or live action dict."""
    return {key: np.asarray(value, dtype=env.action_space[key].dtype)
            if env.action_space[key].shape else value for key, value in action.items()}


def replay_actions(env, actions, *, seed=None):
    """Reset ``env`` (with ``seed``; None reuses its last seed) and step ``actions``; return each step's info."""
    env.reset(seed=seed)
    return [env.step(as_action(env, action))[-1] for action in actions]


def _rebuild(template, data):
    """Native config of ``template``'s types holding the recorded ``data`` (from ``asdict``)."""
    if is_dataclass(template) and not isinstance(template, type) and isinstance(data, dict):
        return replace(template, **{f.name: _rebuild(getattr(template, f.name), data[f.name])
                                    for f in fields(template) if f.init and f.name in data})
    if isinstance(template, (list, tuple)) and isinstance(data, list):
        items = [_rebuild(template[min(i, len(template) - 1)] if template else None, v) for i, v in enumerate(data)]
        if hasattr(template, "_fields"):
            return type(template)(*items)
        return type(template)(items)
    if isinstance(template, enum.Enum):
        return type(template)(data)
    if isinstance(template, np.ndarray):
        return np.asarray(data, dtype=template.dtype)
    if isinstance(template, dict) and isinstance(data, dict):
        return {k: _rebuild(template.get(k), v) for k, v in data.items()}
    if template is None and isinstance(data, (dict, list)):
        raise ValueError("cannot infer the type of a recorded native config value")
    return data


def make_recorded_env(config, *, native_config=None):
    """Env equivalent to the one recorded in ``config`` (a loaded config.json)."""
    benchmark = dict(config["benchmark"], image_size=tuple(config["benchmark"]["image_size"]))
    benchmark = BenchmarkConfig(**benchmark)
    recorded = config["native"]
    if native_config is None:
        env = make_env(config["environment"], config=benchmark)
        if _canonical(asdict(env.native.cfg)) == recorded:
            return env
        template = env.native.cfg
        env.close()
        try:
            native_config = _rebuild(template, recorded)
        except (TypeError, ValueError, KeyError) as exc:
            raise ValueError("the recorded native config could not be rebuilt; pass native_config=") from exc
    env = make_env(config["environment"], config=benchmark, native_config=native_config)
    if _canonical(asdict(env.native.cfg)) != recorded:
        env.close()
        raise ValueError("the recorded native config is not reproduced; pass the original native_config=")
    return env


def _differences(row, observation, reward, info, compare):
    out = []
    if "proprio" in compare and _canonical(observation["proprio"]) != row["proprio"]:
        out.append("proprio")
    if "reward" in compare and reward is not None and float(reward) != row["reward"]:
        out.append("reward")
    if "events" in compare and _canonical(info["events"]) != row["info"]["events"]:
        out.append("events")
    if "info" in compare and _canonical(info) != row["info"]:
        out.append("info")
    return out


def verify_run(run_dir, *, native_config=None, compare=DEFAULT_FIELDS) -> dict:
    """Replay a ``run_episode`` recording and report the first divergence, if any.

    Returns ``verified`` (bool), ``steps`` (recorded decisions), ``checked``
    (transitions compared, reset included) and ``divergence`` (None or the first
    ``{"step", "fields"}``; step 0 is the reset).
    """
    unknown = set(compare) - set(FIELDS)
    if unknown:
        raise ValueError(f"unknown comparison fields {sorted(unknown)}; choose from {FIELDS}")
    run_dir = Path(run_dir)
    config = json.loads((run_dir / "config.json").read_text())
    rows = [json.loads(line) for line in (run_dir / "trajectory.jsonl").read_text().splitlines() if line.strip()]
    if not rows or rows[0]["type"] != "reset" or any(r["type"] != "step" for r in rows[1:]):
        raise ValueError(f"{run_dir} has no reset-then-steps trajectory")
    report = dict(environment=config["environment"], seed=config["seed"], steps=len(rows) - 1,
                  compared=list(compare), checked=0, divergence=None)
    env = make_recorded_env(config, native_config=native_config)
    try:
        observation, info = env.reset(seed=config["seed"])
        ended = False
        for step, row in enumerate(rows):
            reward = None
            if step:
                if ended:
                    report["divergence"] = dict(step=step, fields=["ended_early"])
                    break
                observation, reward, terminated, truncated, info = env.step(as_action(env, row["action"]))
                ended = terminated or truncated
            report["checked"] += 1
            diff = _differences(row, observation, reward, info, compare)
            if diff:
                report["divergence"] = dict(step=step, fields=diff)
                break
    finally:
        env.close()
    report["verified"] = report["divergence"] is None
    return report
