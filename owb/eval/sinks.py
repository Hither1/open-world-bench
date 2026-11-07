"""Optional presentation output for ``run_episode``, kept apart from recording.

A ``FrameSink`` receives the live env and the policy observation after the reset
and after every step. It must not step or otherwise change the simulation.
``VideoSink`` is the GIF/MP4 writer behind ``run_episode(video=...)``: Teyvat and
Wildlands default to Blender Cycles (``owb.render.presentation``), other worlds
write the policy RGB observation.
"""
from __future__ import annotations


class FrameSink:
    """Lifecycle: ``open(directory, env)`` once, ``append`` per transition, ``close`` once.

    ``config_fields()`` returns the ``video``/``presentation`` entries recorded in
    config.json (both None by default). ``directory`` is the run directory, or None
    when the episode is not recorded.
    """
    def config_fields(self) -> dict:
        return {}

    def open(self, directory, env) -> None:
        pass

    def append(self, env, observation) -> None:
        raise NotImplementedError

    def close(self) -> None:
        pass


class VideoSink(FrameSink):
    """``episode.gif`` / ``episode.mp4`` at the decision interval.

    The presentation renderer is created on construction (so a missing Blender fails
    before the run directory exists); the file is created by ``open``.
    """
    def __init__(self, video, env, *, renderer=None, size=(960, 540), blender=None, samples=48, threads=8):
        if video not in ("gif", "mp4"):
            raise ValueError("video must be None, 'gif', or 'mp4'")
        try:
            import imageio.v2 as imageio
        except ImportError as exc:
            raise ImportError('video recording requires pip install -e ".[video]"') from exc
        self._imageio = imageio
        self.video, self.size, self.writer, self.presentation = video, size, None, None
        if renderer is None and env.name in ("teyvat", "wildlands"):
            renderer = "cycles"
        if renderer is not None:
            from ..render.presentation import PresentationRenderer
            self.presentation = PresentationRenderer(backend=renderer, width=size[0], height=size[1],
                                                     blender=blender, samples=samples, threads=threads)

    def config_fields(self):
        return dict(video=self.video, presentation=self.presentation.metadata if self.presentation else None)

    def open(self, directory, env):
        if directory is None:
            raise ValueError("video recording requires output_dir")
        if self.video == "gif":
            self.writer = self._imageio.get_writer(directory / "episode.gif", mode="I",
                                                   duration=env.config.decision_seconds * 1000, loop=0)
        else:
            from ..render.video import H264Writer
            size = self.size if self.presentation else env.config.image_size
            self.writer = H264Writer(directory / "episode.mp4", size=size, fps=1 / env.config.decision_seconds)

    def append(self, env, observation):
        frame = self.presentation.render(env) if self.presentation else observation["rgb"]
        self.writer.append_data(frame)

    def close(self):
        try:
            if self.writer is not None:
                self.writer.close()
        finally:
            if self.presentation is not None:
                self.presentation.close()
