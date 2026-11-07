"""One episode loop for every common-schema env: policy, budgets, recorder, frame sink."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
import time

from ..bench.framework import EPISODE_KEYS
from ..core import SCHEMA_VERSION, event_type
from .policies import RandomPolicy
from .recorder import EpisodeRecorder
from .sinks import VideoSink


@dataclass(frozen=True)
class Budget:
    """Optional per-episode limits checked by the runner before each decision.

    ``max_model_calls`` counts ``policy.calls`` when the policy exposes it (LLM
    policies), otherwise the runner's own policy invocations. ``max_wall_clock_s``
    is measured from the start of ``run_episode`` (including reset). A hit ends the
    episode with ``termination_reason="budget"``; it is not an error.
    """
    max_model_calls: int | None = None
    max_wall_clock_s: float | None = None

    def __post_init__(self):
        if self.max_model_calls is not None and (
                type(self.max_model_calls) is not int or self.max_model_calls <= 0):
            raise ValueError("max_model_calls must be None or a positive integer")
        if self.max_wall_clock_s is not None and (
                isinstance(self.max_wall_clock_s, bool) or not isinstance(self.max_wall_clock_s, (int, float))
                or not math.isfinite(self.max_wall_clock_s) or self.max_wall_clock_s <= 0):
            raise ValueError("max_wall_clock_s must be None or a positive finite number")

    @property
    def enabled(self):
        return self.max_model_calls is not None or self.max_wall_clock_s is not None

    def limits(self):
        return dict(max_model_calls=self.max_model_calls, max_wall_clock_s=self.max_wall_clock_s)

    def exceeded(self, policy, invocations, started):
        """Name of the first exhausted budget, or None."""
        if self.max_model_calls is not None:
            calls = getattr(policy, "calls", None)
            calls = calls if type(calls) is int else invocations
            if calls >= self.max_model_calls:
                return "max_model_calls"
        if self.max_wall_clock_s is not None and time.monotonic() - started >= self.max_wall_clock_s:
            return "max_wall_clock_s"
        return None


def _summary(episode, info, counts, policy):
    summary = dict(episode, schema_version=SCHEMA_VERSION, metrics=info["metrics"], event_counts=dict(counts))
    if hasattr(policy, "metadata"):
        summary["policy"] = policy.metadata()
    return summary


def _close_all(*closers):
    """Call each close in order; every one runs even if an earlier one raises."""
    if not closers:
        return
    first, *rest = closers
    try:
        if first is not None:
            first()
    finally:
        _close_all(*rest)


def run_episode(env, policy=None, *, seed=None, output_dir=None, video=None,
                video_renderer=None, video_size=(960, 540), blender=None, samples=48, threads=8,
                frame_sink=None, max_model_calls=None, max_wall_clock_s=None) -> dict:
    """Run until termination/truncation (or an exhausted budget). Policies receive observations only.

    A policy is callable and may expose reset(seed) for episode-local memory,
    ``metadata()``/``last_decision`` for the recorder, and ``close()``, which is
    called when the episode ends (also on error). The caller owns the environment
    and can reset/reuse or close it afterward.

    ``output_dir`` records the run (``EpisodeRecorder``). ``frame_sink`` receives
    presentation frames; ``video="gif"|"mp4"`` (with the ``video_*``/Blender options)
    is shorthand for a ``VideoSink`` and requires ``output_dir``. Budgets default to
    None (unlimited); when any is set the summary also carries ``budget`` and
    ``budget_hit``.
    """
    if video is not None and output_dir is None:
        raise ValueError("video recording requires output_dir")
    if video is not None and frame_sink is not None:
        raise ValueError("pass either video= or frame_sink=, not both")
    budget = Budget(max_model_calls, max_wall_clock_s)
    policy = RandomPolicy() if policy is None else policy
    started = time.monotonic()
    try:
        return _run(env, policy, env.config.seed if seed is None else int(seed), output_dir, frame_sink,
                    budget, started, dict(video=video, renderer=video_renderer, size=video_size,
                                          blender=blender, samples=samples, threads=threads))
    finally:
        close = getattr(policy, "close", None)
        if callable(close):
            close()


def _run(env, policy, seed, output_dir, sink, budget, started, video):
    observation, info = env.reset(seed=seed)
    if hasattr(policy, "reset"):
        policy.reset(seed)
    if video["video"] is not None:
        sink = VideoSink(video.pop("video"), env, **video)
    recorder = None
    try:
        if output_dir is not None:
            recorder = EpisodeRecorder(output_dir, env, seed, **(sink.config_fields() if sink else {}))
        if sink is not None:
            sink.open(recorder.directory if recorder else None, env)
    except Exception:
        _close_all(recorder.close if recorder else None, sink.close if sink else None)
        raise
    counts, invocations = Counter(), 0
    try:
        if recorder:
            if hasattr(policy, "metadata"):
                recorder.write_policy(policy.metadata())
            recorder.record("reset", observation, info)
        if sink:
            sink.append(env, observation)
        while True:
            hit = budget.exceeded(policy, invocations, started) if budget.enabled else None
            if hit is not None:
                episode = {key: info[key] for key in EPISODE_KEYS}
                episode.update(terminated=False, truncated=True, termination_reason="budget")
                summary = _summary(episode, info, counts, policy)
                summary.update(budget=budget.limits(), budget_hit=hit)
                if recorder:
                    recorder.finish(summary)
                return summary
            action = policy(observation)
            invocations += 1
            observation, reward, terminated, truncated, info = env.step(action)
            counts.update(event_type(event) for event in info["events"])
            if recorder:
                recorder.record("step", observation, info, action=action, reward=reward,
                                policy_decision=getattr(policy, "last_decision", None))
            if sink:
                sink.append(env, observation)
            if terminated or truncated:
                summary = _summary(info["episode"], info, counts, policy)
                if budget.enabled:
                    summary.update(budget=budget.limits(), budget_hit=None)
                if recorder:
                    recorder.finish(summary)
                return summary
    except Exception as exc:
        if recorder:
            # Do not persist raw API exception text (may include request headers or images).
            error = dict(error_type=type(exc).__name__, steps=env.steps, status="error")
            if hasattr(policy, "metadata"):
                error["policy"] = policy.metadata()
                error["last_decision"] = getattr(policy, "last_decision", None)
            recorder.fail(error)
        raise
    finally:
        _close_all(sink.close if sink else None, recorder.close if recorder else None)
