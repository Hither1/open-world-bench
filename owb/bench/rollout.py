"""Compatibility module: the runner, recorder and reference policies live in ``owb.eval``.

``run_episode``, ``RandomPolicy`` and ``DemoPolicy`` are re-exported unchanged.
``EpisodeRecorder`` here keeps the old combined signature (JSON recording plus an
optional GIF/MP4) by pairing ``owb.eval.EpisodeRecorder`` with a ``VideoSink``.
"""
from __future__ import annotations

from ..core import SCHEMA_VERSION, event_type, jsonable  # noqa: F401  (formerly defined/imported here)
from ..eval import recorder as _recorder
from ..eval.policies import DemoPolicy, RandomPolicy
from ..eval.runner import run_episode
from ..eval.sinks import VideoSink
from .framework import BUTTONS, COMMANDS, MODALITIES, PROPRIO_NAMES, neutral_action  # noqa: F401

__all__ = ["DemoPolicy", "EpisodeRecorder", "RandomPolicy", "run_episode"]


class EpisodeRecorder(_recorder.EpisodeRecorder):
    """JSONL transitions plus optional GIF/MP4; refuses to overwrite a run (legacy signature)."""
    def __init__(self, directory, env, seed, video=None, *, video_renderer=None,
                 video_size=(960, 540), blender=None, samples=48, threads=8):
        if video not in (None, "gif", "mp4"):
            raise ValueError("video must be None, 'gif', or 'mp4'")
        self.sink = (VideoSink(video, env, renderer=video_renderer, size=video_size, blender=blender,
                               samples=samples, threads=threads) if video else None)
        try:
            super().__init__(directory, env, seed, **(self.sink.config_fields() if self.sink else {}))
        except Exception:
            if self.sink is not None:
                self.sink.close()
            raise
        try:
            if self.sink is not None:
                self.sink.open(self.directory, env)
        except Exception:
            self.close()
            raise

    def record(self, kind, observation, info, **kwargs):
        super().record(kind, observation, info, **kwargs)
        if self.sink is not None:
            self.sink.append(self.env, observation)

    def close(self):
        try:
            if self.sink is not None:
                self.sink.close()
        finally:
            super().close()
