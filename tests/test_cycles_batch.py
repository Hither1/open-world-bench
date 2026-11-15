"""Headless checks of the persistent Cycles process protocol (no Blender needed)."""
import sys
import textwrap

import numpy as np
import pytest

from owb.render import cycles_batch

FAKE_BLENDER = textwrap.dedent("""\
    #!{python}
    # Stand-in for Blender: answers --version and the batch job protocol.
    import json, os, sys
    from PIL import Image
    if "--version" in sys.argv:
        print("Blender test-double"); sys.exit(0)
    args = sys.argv[sys.argv.index("--") + 1:]
    ack = os.fdopen(int(args[args.index("--ack-fd") + 1]), "w", buffering=1)
    for line in sys.stdin:
        job = json.loads(line)
        if os.environ.get("FAKE_BLENDER_FAIL") == str(job["id"]):
            print("simulated Cycles failure", flush=True)
            ack.write(json.dumps(dict(id=job["id"], ok=False, error="boom")) + "\\n"); sys.exit(1)
        with open(job["scene"]) as f: value = int(f.read())
        Image.new("RGB", (960, 540), (value, value, value)).save(job["output"])
        ack.write(json.dumps(dict(id=job["id"], ok=True, build_seconds=0.0, render_seconds=0.0)) + "\\n")
""")


@pytest.fixture
def fake(tmp_path, monkeypatch):
    path = tmp_path / "blender"
    path.write_text(FAKE_BLENDER.format(python=sys.executable))
    path.chmod(0o755)

    def export(env, scene, **kwargs):
        scene.write_text(str(env.value))
    monkeypatch.setattr(cycles_batch, "export_scene", export)
    return str(path)


class State:
    def __init__(self, value):
        self.value = value


def test_frames_return_in_submission_order_from_one_process(fake):
    with cycles_batch.CyclesBatchRenderer(blender=fake, device="OPTIX") as renderer:
        tickets = [renderer.submit(State(v)) for v in (10, 20, 30)]
        process = renderer._process
        frames = [renderer.result(t) for t in tickets]
        assert renderer._process is process
        assert renderer.metadata["process"] == "persistent" and renderer.metadata["device"] == "OPTIX"
    assert [int(f[0, 0, 0]) for f in frames] == [10, 20, 30]
    assert all(f.shape == (540, 960, 3) for f in frames)


def test_results_must_be_collected_in_order(fake):
    with cycles_batch.CyclesBatchRenderer(blender=fake, device="CUDA") as renderer:
        first, second = renderer.submit(State(1)), renderer.submit(State(2))
        with pytest.raises(ValueError, match="submission order"):
            renderer.result(second)
        renderer.result(first)


def test_failed_frame_is_reported_with_log(fake, monkeypatch):
    monkeypatch.setenv("FAKE_BLENDER_FAIL", "1")
    with cycles_batch.CyclesBatchRenderer(blender=fake, device="CUDA") as renderer:
        assert np.all(renderer.render(State(5)) == 5)
        ticket = renderer.submit(State(6))
        with pytest.raises(RuntimeError, match="simulated Cycles failure"):
            renderer.result(ticket)
