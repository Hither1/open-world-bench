"""Execute a manifest: one ``run_episode`` per env x seed, same layout as benchmark.py.

Each recorded run directory ``<output_dir>/<sim>-<seed>/`` holds what
``owb.eval.EpisodeRecorder`` writes (config.json, trajectory.jsonl, summary.json,
policy.json, error.json) plus ``manifest.json`` (``{"manifest": Manifest.to_dict(),
...}``) and, with a task, ``task.json`` (the evaluator summary). Existing run
directories are never overwritten: all are checked before the first episode.
"""
from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
import json
from pathlib import Path
import sys

from ..core import SCHEMA_VERSION, jsonable
from .agents import build_policy, oracle
from .manifest import ManifestError


class RunError(RuntimeError):
    """A model/harness failure reported as a one-line message (exit status 1), not a traceback."""


@dataclass
class Episode:
    sim: str
    seed: int
    run_dir: Path | None
    summary: dict
    privileged: bool
    task: dict | None = None


def plan(manifest):
    """``(sim, seed, run_dir or None)`` for every run, in execution order."""
    record = manifest.record.trajectory
    return [(sim, seed, manifest.run_dir(sim, seed) if record else None)
            for sim in manifest.env.sim for seed in manifest.task.seeds]


def preflight(manifest, *, allow_privileged=False):
    """Checks that must pass before any env is built; raises ManifestError."""
    if manifest.privileged and not allow_privileged:
        raise ManifestError("privileged run refused (" + "; ".join(manifest.privileged_reasons)
                            + "); pass --allow-privileged to run it as a non-benchmark reference")
    if manifest.agent.oracle is not None:
        spec = oracle(manifest.agent.oracle)
        if not spec.runnable:
            raise ManifestError(f"agent.policy: oracle {spec.name!r} drives its native reference-course env, "
                                f"not the common-schema env built by owb run; use {spec.script}")
    for _, _, run_dir in plan(manifest):
        if run_dir is not None and run_dir.exists():
            raise ManifestError(f"recording already exists: {run_dir}")


def _check_overrides(template, data, where):
    """Reject native_config keys that are not fields of the native config dataclass."""
    if not isinstance(data, dict):
        return
    if is_dataclass(template):
        names = {f.name for f in fields(template) if f.init}
        unknown = sorted(set(data) - names)
        if unknown:
            raise ManifestError(f"{where}: unknown native config field(s) {unknown}")
        for key, value in data.items():
            _check_overrides(getattr(template, key), value, f"{where}.{key}")


def native_config(manifest, sim, bench):
    """The native config for ``sim``: ``make_env``'s own, with ``env.native_config[sim]`` applied."""
    overrides = (manifest.env.native_config or {}).get(sim)
    if not overrides:
        return None
    from ..bench.framework import make_env
    from ..eval.replay import _rebuild
    with make_env(sim, config=bench) as probe:
        template = probe.native.cfg
    _check_overrides(template, overrides, f"env.native_config.{sim}")
    try:
        return _rebuild(template, overrides)
    except (TypeError, ValueError, KeyError) as exc:
        raise ManifestError(f"env.native_config.{sim}: {exc}") from None


def _task_env(env, spec):
    from ..tasks.env import TaskEnv

    class RecordedTaskEnv(TaskEnv):
        """TaskEnv exposing the BenchmarkEnv attributes the runner and recorder read."""
        name = property(lambda self: self.env.unwrapped.name)
        config = property(lambda self: self.env.unwrapped.config)
        native = property(lambda self: self.env.unwrapped.native)
        steps = property(lambda self: self.env.unwrapped.steps)

        def schema_vocabularies(self):
            return self.env.unwrapped.schema_vocabularies()

    return RecordedTaskEnv(env, spec)


def _write_json(path, value):
    path.write_text(json.dumps(jsonable(value), indent=2, allow_nan=False) + "\n")


def _model_error(manifest, exc, run_dir):
    """benchmark.py's user-facing message for a model failure, or None to re-raise."""
    harness = manifest.agent.harness
    if manifest.agent.policy != "llm":
        return None
    from ..agents.harness import ClaudePolicyError, CodexPolicyError, GeminiPolicyError
    if isinstance(exc, (CodexPolicyError, ClaudePolicyError, GeminiPolicyError)):
        return str(exc)
    if harness.kind != "openai_api":
        return None
    try:
        from openai import OpenAIError
    except ImportError:
        return None
    if not isinstance(exc, OpenAIError):
        return None
    code = getattr(exc, "code", None)
    if code in ("credit_balance_exhausted", "insufficient_quota"):
        message = "OpenAI API credits/quota exhausted. Fund the API project or configure a funded OPENAI_API_KEY."
    else:
        message = f"OpenAI request failed: {type(exc).__name__} (code={code})."
    return message + f" See {run_dir / 'error.json'}." if run_dir is not None else message


def run_one(manifest, sim, seed, run_dir, *, argv=None) -> Episode:
    from ..bench.framework import BenchmarkConfig, make_env
    from ..eval.runner import run_episode
    env_cfg, record, budget = manifest.env, manifest.record, manifest.budget
    bench = BenchmarkConfig(seed=seed, variant=env_cfg.variant, image_size=tuple(env_cfg.image_size),
                            max_steps=manifest.max_steps, decision_seconds=env_cfg.decision_seconds,
                            include_goal_hint=env_cfg.include_goal_hint, render_quality=env_cfg.render_quality)
    spec = manifest.task_spec()
    task = None
    with make_env(sim, config=bench, native_config=native_config(manifest, sim, bench)) as base:
        env = _task_env(base, spec) if spec is not None else base
        policy = build_policy(manifest, env)
        try:
            summary = run_episode(env, policy, seed=seed, output_dir=run_dir, video=record.video,
                                  video_renderer=record.video_renderer, video_size=tuple(record.video_size),
                                  blender=record.blender, samples=record.samples, threads=record.threads,
                                  max_model_calls=budget.max_model_calls,
                                  max_wall_clock_s=budget.max_wall_clock_s)
            if spec is not None:
                task = env.task_summary()
        except Exception as exc:
            message = _model_error(manifest, exc, run_dir)
            if message is None:
                raise
            raise RunError(message) from exc
        finally:
            if run_dir is not None and run_dir.is_dir():
                _write_json(run_dir / "manifest.json", dict(
                    schema_version=SCHEMA_VERSION, manifest=manifest.to_dict(), environment=sim, seed=seed,
                    privileged=manifest.privileged, privileged_reasons=list(manifest.privileged_reasons),
                    argv=list(sys.argv if argv is None else argv)))
                if task is not None:
                    _write_json(run_dir / "task.json", task)
    return Episode(sim, seed, run_dir, summary, manifest.privileged, task)


def run_manifest(manifest, *, allow_privileged=False, on_episode=None, argv=None) -> list[Episode]:
    """Run every env x seed of ``manifest``; ``on_episode(Episode)`` is called after each."""
    preflight(manifest, allow_privileged=allow_privileged)
    episodes = []
    for sim, seed, run_dir in plan(manifest):
        episode = run_one(manifest, sim, seed, run_dir, argv=argv)
        episodes.append(episode)
        if on_episode is not None:
            on_episode(episode)
    return episodes


__all__ = ["Episode", "RunError", "native_config", "plan", "preflight", "run_manifest", "run_one"]
