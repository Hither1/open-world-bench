"""`TaskEnv`: run a :class:`TaskSpec` on top of a common-schema ``BenchmarkEnv``.

The wrapper owns a :class:`MilestoneEvaluator`. On every step it feeds the evaluator the
common ``info["events"]`` and a read-only :class:`StateView` of the native simulator, and
reports the result under ``info["task"]``. The observation only changes when the spec
sets an ``instruction``, which then replaces ``observation["instruction"]``; no evaluator
state ever enters the observation. Simulator behaviour, rewards and termination are
untouched. Setup hooks (placing a spec's entities) are not implemented yet: the task runs
in the world the native env builds.
"""
from __future__ import annotations

from typing import Any

import gymnasium as gym

from .evaluator import MilestoneEvaluator, StateView
from .schema import TaskSpec


class NullStateView:
    """For sims whose state the evaluator cannot read yet: every field is ``None``."""

    position = None
    time_of_day = None
    inventory = None
    stats = None


class TeyvatStateView:
    """Live view of a native ``OpenWorldEnv``: position, time of day and inventory.

    Teyvat's only inventory is the quest engine's ledger (``env.quests.inventory``). It has
    no cumulative statistics, so ``stats`` is ``None``.
    """

    def __init__(self, native):
        self._native = native

    @property
    def position(self):
        return self._native.char.pos.copy()

    @property
    def time_of_day(self) -> float:
        return float(self._native.world.time_of_day)

    @property
    def inventory(self) -> dict[str, int]:
        return dict(self._native.quests.inventory)

    stats = None


STATE_VIEWS = {"teyvat": TeyvatStateView}


def state_view(env) -> StateView:
    """The :class:`StateView` adapter for a ``BenchmarkEnv`` (``NullStateView`` if none)."""
    factory = STATE_VIEWS.get(env.name)
    return factory(env.native) if factory is not None else NullStateView()


class TaskEnv(gym.Wrapper):
    def __init__(self, env, spec: TaskSpec):
        super().__init__(env)
        base = env.unwrapped
        if spec.sim != base.name:
            raise ValueError(f"task {spec.id!r} is for sim {spec.sim!r}, not {base.name!r}")
        if spec.instruction and not self.observation_space["instruction"].contains(spec.instruction):
            raise ValueError(f"task {spec.id!r} instruction does not fit the instruction space")
        self.task_spec = spec
        self._evaluator = MilestoneEvaluator(spec)
        self._view: StateView = NullStateView()

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self._view = state_view(self.env.unwrapped)
        self._evaluator = MilestoneEvaluator(self.task_spec)
        self._evaluator.reset(self._view)
        return self._observe(observation), self._with_task(info, [])

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        fired = self._evaluator.update(info.get("events", ()), self._view, info["steps"])
        return self._observe(observation), reward, terminated, truncated, self._with_task(info, fired)

    def task_summary(self) -> dict[str, Any]:
        """Evaluator summary for the harness / recorder (never for the agent)."""
        return self._evaluator.summary()

    def task_snapshot(self) -> dict[str, Any]:
        return self._evaluator.snapshot()

    def _observe(self, observation):
        if self.task_spec.instruction:
            observation = dict(observation, instruction=self.task_spec.instruction)
        return observation

    def _with_task(self, info, fired):
        info = dict(info)
        ev = self._evaluator
        info["task"] = {"id": self.task_spec.id, "progress": ev.progress, "success": ev.success,
                        "fired": list(fired), "achieved": dict(ev.achieved)}
        return info


__all__ = ["NullStateView", "STATE_VIEWS", "TaskEnv", "TeyvatStateView", "state_view"]
