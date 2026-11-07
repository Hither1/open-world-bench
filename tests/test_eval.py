"""owb.eval: recorded runs replay exactly, budgets end episodes, sinks and recorder are separate."""
import json

import numpy as np
import pytest

import owb
from owb.bench import make_env, neutral_action
from owb.bench.rollout import run_episode as legacy_run_episode
from owb.eval import (
    Budget, DemoPolicy, FrameSink, RandomPolicy, make_recorded_env, replay_actions, run_episode, verify_run,
)


def arena():
    return owb.EnvConfig(
        world_cfg=owb.WorldConfig(preset="arena", size_x=32, size_z=32, size_y=40, animal_density=0),
        sim_cfg=owb.SimConfig(weather_enabled=False, wind_enabled=False),
        render_cfg=owb.RenderConfig(max_dist=40, shadows=False, max_steps=64, quality="fast"),
    )


def rows(path):
    return [json.loads(line) for line in (path / "trajectory.jsonl").read_text().splitlines()]


def test_compat_names_are_the_eval_objects():
    from owb.bench import rollout
    assert legacy_run_episode is run_episode and rollout.DemoPolicy is DemoPolicy
    assert issubclass(rollout.EpisodeRecorder, owb.eval.EpisodeRecorder)


@pytest.mark.parametrize("name", ["teyvat", "wildlands", "skylands"])
def test_random_run_replays_and_verifies(name, tmp_path):
    with make_env(name, image_size=(32, 24), max_steps=30) as env:
        summary = run_episode(env, RandomPolicy(), seed=11, output_dir=tmp_path / name)
        recorded = rows(tmp_path / name)
        infos = replay_actions(env, [r["action"] for r in recorded[1:]], seed=11)
    assert summary["steps"] == 30 and len(infos) == 30
    assert infos == [r["info"] for r in recorded[1:]]
    report = verify_run(tmp_path / name, compare=("proprio", "reward", "events", "info"))
    assert report["verified"] and report["checked"] == 31 and report["divergence"] is None


def test_verify_rebuilds_a_custom_native_config_and_reports_divergence(tmp_path):
    path = tmp_path / "arena"
    with make_env("wildlands", native_config=arena(), image_size=(32, 24), max_steps=12) as env:
        run_episode(env, RandomPolicy(), seed=3, output_dir=path)
    assert verify_run(path)["verified"]
    lines = (path / "trajectory.jsonl").read_text().splitlines()
    row = json.loads(lines[5])
    row["action"]["move"] = [-v if v else 1.0 for v in row["action"]["move"]]
    lines[5] = json.dumps(row)
    (path / "trajectory.jsonl").write_text("\n".join(lines) + "\n")
    report = verify_run(path)
    assert not report["verified"] and report["divergence"]["step"] == 5
    assert "proprio" in report["divergence"]["fields"]


def test_teyvat_native_config_round_trips(tmp_path):
    from owb.sims.teyvat import EnvConfig, RenderSettings, TerrainConfig, WorldConfig
    cfg = EnvConfig(world=WorldConfig(terrain=TerrainConfig(size=512, resolution=65), prop_density=0),
                    render=RenderSettings(march_steps=16, shadow_steps=0, draw_props=False))
    with make_env("teyvat", native_config=cfg, image_size=(32, 24), max_steps=6) as env:
        run_episode(env, DemoPolicy(), seed=5, output_dir=tmp_path / "t")
    config = json.loads((tmp_path / "t" / "config.json").read_text())
    with make_recorded_env(config) as env:
        assert env.native.cfg.world.terrain.resolution == 65
    assert verify_run(tmp_path / "t")["verified"]


def test_model_call_budget_ends_episode_and_still_verifies(tmp_path):
    with make_env("skylands", image_size=(32, 24), max_steps=10) as env:
        summary = run_episode(env, RandomPolicy(), seed=7, output_dir=tmp_path / "b", max_model_calls=3)
        assert env.steps == 3
    assert summary["termination_reason"] == "budget" and summary["budget_hit"] == "max_model_calls"
    assert summary["steps"] == 3 and summary["truncated"] and not summary["terminated"]
    assert summary["budget"] == dict(max_model_calls=3, max_wall_clock_s=None)
    assert json.loads((tmp_path / "b" / "summary.json").read_text()) == summary
    assert len(rows(tmp_path / "b")) == 4
    assert verify_run(tmp_path / "b")["verified"]


def test_wall_clock_budget_and_policy_call_counter():
    class Counting:
        calls = 0

        def __call__(self, observation):
            self.calls += 2  # e.g. a policy that retries every model request once
            return neutral_action()

    with make_env("skylands", image_size=(32, 24), max_steps=10) as env:
        summary = run_episode(env, Counting(), seed=7, max_model_calls=4)
        assert summary["steps"] == 2 and summary["budget_hit"] == "max_model_calls"
        summary = run_episode(env, RandomPolicy(), seed=7, max_wall_clock_s=1e-9)
        assert summary["steps"] == 0 and summary["budget_hit"] == "max_wall_clock_s"
        unbounded = run_episode(env, RandomPolicy(), seed=7, max_model_calls=50)
        plain = run_episode(env, RandomPolicy(), seed=7)
    assert unbounded["budget_hit"] is None and unbounded["termination_reason"] == "time_limit"
    assert "budget" not in plain and "budget_hit" not in plain
    assert {k: v for k, v in unbounded.items() if not k.startswith("budget")} == plain
    for bad in (dict(max_model_calls=0), dict(max_model_calls=True), dict(max_wall_clock_s=float("inf"))):
        with pytest.raises(ValueError):
            Budget(**bad)


def test_policy_close_is_called_on_success_and_error(tmp_path):
    class Closing(RandomPolicy):
        closed = 0

        def close(self):
            self.closed += 1

    class Failing(Closing):
        def __call__(self, observation):
            raise RuntimeError("boom")

    with make_env("skylands", image_size=(32, 24), max_steps=2) as env:
        ok = Closing()
        run_episode(env, ok, seed=1)
        bad = Failing()
        with pytest.raises(RuntimeError, match="boom"):
            run_episode(env, bad, seed=1, output_dir=tmp_path / "err")
    assert ok.closed == 1 and bad.closed == 1
    assert json.loads((tmp_path / "err" / "error.json").read_text())["status"] == "error"


def test_frame_sink_is_independent_of_recording(tmp_path):
    class Collect(FrameSink):
        def __init__(self):
            self.frames, self.opened, self.closed = [], [], False

        def open(self, directory, env):
            self.opened.append(directory)

        def append(self, env, observation):
            self.frames.append(observation["rgb"].copy())

        def close(self):
            self.closed = True

    with make_env("skylands", image_size=(32, 24), max_steps=3) as env:
        sink = Collect()
        run_episode(env, RandomPolicy(), seed=2, frame_sink=sink)
        assert sink.opened == [None] and sink.closed and len(sink.frames) == 4
        recorded = Collect()
        run_episode(env, RandomPolicy(), seed=2, frame_sink=recorded, output_dir=tmp_path / "r")
        with pytest.raises(ValueError, match="either video"):
            run_episode(env, frame_sink=Collect(), video="gif", output_dir=tmp_path / "x")
    assert recorded.opened == [tmp_path / "r"]
    np.testing.assert_array_equal(np.stack(recorded.frames), np.stack(sink.frames))
    assert sorted(p.name for p in (tmp_path / "r").iterdir()) == ["config.json", "summary.json", "trajectory.jsonl"]
    config = json.loads((tmp_path / "r" / "config.json").read_text())
    assert config["video"] is None and config["presentation"] is None
