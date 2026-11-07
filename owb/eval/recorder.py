"""Run-directory recorder: JSON/JSONL only. It never renders.

Presentation output (GIF/MP4, Cycles) is a separate ``FrameSink`` run by
``run_episode``; the recorder only notes its ``video``/``presentation`` metadata
in ``config.json``.
"""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

from ..core import SCHEMA_VERSION, jsonable


def _schema(env):
    if hasattr(env, "schema_vocabularies"):
        return env.schema_vocabularies()
    from ..bench.framework import schema_vocabularies
    return schema_vocabularies()


class EpisodeRecorder:
    """Writes config.json, trajectory.jsonl, summary.json, error.json and policy.json.

    Refuses to overwrite a run directory. ``video`` and ``presentation`` are recorded
    as given (the frame sink's format and renderer settings, or None).
    """
    def __init__(self, directory, env, seed, video=None, *, presentation=None):
        self.log = None
        self.env = env
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        config = dict(schema_version=SCHEMA_VERSION, environment=env.name, seed=seed,
                      benchmark=asdict(env.config), native=asdict(env.native.cfg),
                      schema=_schema(env),
                      video=video, decision_seconds=env.config.decision_seconds,
                      presentation=presentation)
        try:
            self._write_json("config.json", config)
            self.log = (self.directory / "trajectory.jsonl").open("w")
        except Exception:
            self.close()
            raise

    def _write_json(self, name, value):
        (self.directory / name).write_text(json.dumps(jsonable(value), indent=2, allow_nan=False) + "\n")

    def write_policy(self, metadata):
        self._write_json("policy.json", metadata)

    def record(self, kind, observation, info, *, action=None, reward=None, policy_decision=None):
        row = dict(type=kind, info=info, proprio=observation["proprio"], instruction=observation["instruction"])
        if action is not None:
            row.update(action=action, reward=reward)
        if policy_decision is not None:
            row["policy_decision"] = policy_decision
        self.log.write(json.dumps(jsonable(row), allow_nan=False) + "\n")
        self.log.flush()

    def finish(self, summary):
        self._write_json("summary.json", summary)

    def fail(self, error):
        self._write_json("error.json", error)

    def close(self):
        if self.log is not None:
            self.log.close()
