"""TaskEnv on the common teyvat env: info["task"], instruction override, no side effects."""
import json

import numpy as np
import pytest

import owb
from owb.bench.rollout import RandomPolicy
from owb.tasks import load_task
from owb.tasks.env import TaskEnv
from owb.tasks.teyvat import builtin_questline_spec

SEED, STEPS = 5, 20
INSTRUCTION = "Restore the Windborne Ledger."


def make():
    return owb.make_env("teyvat", seed=SEED, max_steps=STEPS, image_size=(32, 24))


def rollout(env):
    obs, info = env.reset(seed=SEED)
    policy = RandomPolicy()
    policy.reset(SEED)
    out = [(obs, 0.0, False, False, info)]
    for _ in range(STEPS):
        obs, reward, terminated, truncated, info = env.step(policy(obs))
        out.append((obs, reward, terminated, truncated, info))
        if terminated or truncated:
            break
    return out


def test_task_env_reports_progress_and_overrides_instruction():
    spec = builtin_questline_spec(SEED, instruction=INSTRUCTION)
    bare, wrapped = rollout(make()), rollout(TaskEnv(make(), spec))
    assert len(bare) == len(wrapped) == STEPS + 1
    for (o0, r0, t0, u0, i0), (o1, r1, t1, u1, i1) in zip(bare, wrapped):
        # Simulation untouched: same frames, rewards, termination and native info.
        assert set(o1) == set(o0)
        assert o1["instruction"] == INSTRUCTION != o0["instruction"]
        for key in o0:
            if key != "instruction":
                np.testing.assert_array_equal(o1[key], o0[key])
        assert (r1, t1, u1) == (r0, t0, u0)
        assert {k: v for k, v in i1.items() if k != "task"} == i0
        task = i1["task"]
        assert set(task) == {"id", "progress", "success", "fired", "achieved"}
        assert task["id"] == "teyvat.windborne_ledger"
        assert 0.0 <= task["progress"] <= i0["task_progress"] + 1e-9
        assert task["success"] is False
        assert set(task["fired"]) <= set(task["achieved"])
        json.dumps(task, allow_nan=False)


def test_empty_instruction_keeps_the_native_one():
    env = TaskEnv(make(), builtin_questline_spec(SEED))
    try:
        obs, info = env.reset(seed=SEED)
        assert obs["instruction"] == env.unwrapped.native.instruction
        assert env.observation_space.contains(obs)
        assert info["task"]["progress"] == 0.0 and info["task"]["achieved"] == {}
        json.dumps(env.task_snapshot())
    finally:
        env.close()


def test_spec_for_another_sim_is_rejected():
    env = make()
    try:
        with pytest.raises(ValueError, match="minecraft"):
            TaskEnv(env, load_task("ender_dragon"))
    finally:
        env.close()
