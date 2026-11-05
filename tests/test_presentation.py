"""Presentation defaults, failure behavior, observation isolation, and playback."""
import hashlib
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from owb.render.presentation import PresentationRenderer


def test_missing_cycles_is_an_error_not_a_cpu_fallback(monkeypatch):
    monkeypatch.setattr("owb.render.presentation.shutil.which", lambda _: None)
    with pytest.raises(RuntimeError, match="no automatic fallback"):
        PresentationRenderer()
    with PresentationRenderer(backend="cpu") as renderer:
        assert renderer.metadata["backend"] == "cpu"


def test_failed_cycles_render_is_reported(monkeypatch):
    monkeypatch.setattr("owb.render.presentation.export_scene", lambda env, path, **kw: path.write_bytes(b"snapshot"))
    def fail(command, stdout, **kwargs):
        stdout.write("Renderer test failure")
        return SimpleNamespace(returncode=1)
    monkeypatch.setattr("owb.render.presentation.subprocess.run", fail)
    with PresentationRenderer(blender=sys.executable) as renderer:
        with pytest.raises(RuntimeError, match="Renderer test failure"):
            renderer.render(object())


def test_recorder_defaults_to_cycles_and_preserves_policy_rgb(monkeypatch, tmp_path):
    pytest.importorskip("imageio")
    Image = pytest.importorskip("PIL.Image")
    from open_world_bench import EnvConfig, WorldConfig, TerrainConfig, RenderSettings
    from owb import make_env
    from owb.bench.rollout import run_episode
    seen = []
    class RenderSpy:
        def __init__(self, **settings):
            seen.append(settings)
            self.metadata = settings
        def render(self, env):
            return np.full((54, 96, 3), 123, np.uint8)
        def close(self):
            pass
    monkeypatch.setattr("owb.render.presentation.PresentationRenderer", RenderSpy)
    cfg = EnvConfig(world=WorldConfig(terrain=TerrainConfig(size=512, resolution=33), prop_density=0),
                    render=RenderSettings(march_steps=16, shadow_steps=0, draw_props=False, ground_cover=False))
    policy_shapes = []
    def policy(observation):
        from owb.bench import neutral_action
        policy_shapes.append(observation["rgb"].shape)
        return neutral_action()
    with make_env("teyvat", native_config=cfg, image_size=(16, 12), max_steps=1) as env:
        run_episode(env, policy, output_dir=tmp_path / "run", video="gif", video_size=(96, 54))
    assert seen[0]["backend"] == "cycles"
    assert policy_shapes == [(12, 16, 3)]
    config = json.loads((tmp_path / "run/config.json").read_text())
    assert config["presentation"]["backend"] == "cycles"
    with Image.open(tmp_path / "run/episode.gif") as image:
        assert image.size == (96, 54)
        np.testing.assert_array_equal(np.asarray(image.convert("RGB")), 123)


def test_chunk_assembly_rejects_legacy_cpu_frames_by_default(tmp_path):
    from owb.scripts.teyvat_reference import assemble_chunks
    recording, cache = tmp_path / "run", tmp_path / "cache"
    recording.mkdir(); cache.mkdir()
    (recording / "reference-manifest.json").write_text(json.dumps({"verification":{"required":1},"final_digest":"done"}))
    (recording / "actions.jsonl").write_text("")
    piece = cache / "run-01-01"
    piece.mkdir()
    (piece / "index.json").write_text(json.dumps({"first":1,"last":1,"fps":10,"stride":8,
        "trace_sha256":hashlib.sha256(b"").hexdigest(),"checkpoint":"done","frames":[]}))
    with pytest.raises(AssertionError, match="different renderer"):
        assemble_chunks(recording, cache)
    assert not list(recording.glob("*.mp4"))


def test_h264_recording_has_compatible_codec_and_fast_start(tmp_path):
    cv2 = pytest.importorskip("cv2")
    from owb.render.video import H264Writer, resolve_ffmpeg
    try:
        resolve_ffmpeg()
    except RuntimeError:
        pytest.skip("FFmpeg is not installed")
    path = tmp_path / "test.mp4"
    writer = H264Writer(path, size=(64, 48), fps=10)
    try:
        for color in (30, 120, 220):
            writer.append_data(np.full((48, 64, 3), color, np.uint8))
    finally:
        writer.close()
    cap = cv2.VideoCapture(str(path))
    try:
        assert cap.isOpened() and int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 3
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        assert "".join(chr((fourcc >> (8*i)) & 255) for i in range(4)) in ("avc1", "h264", "H264")
        for _ in range(3):
            assert cap.read()[0]
    finally:
        cap.release()
    data = path.read_bytes()
    assert 0 < data.index(b"moov") < data.index(b"mdat")
