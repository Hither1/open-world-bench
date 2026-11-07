"""`owb` CLI: manifest validation/round trip, run -> replay --verify, listings, privileged refusal."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from owb.cli import agents as cli_agents
from owb.cli.main import main
from owb.cli.manifest import Manifest, ManifestError, apply_overrides, read

EXAMPLES = Path(__file__).resolve().parents[1] / "configs" / "examples"
TINY = ["--set", "env.image_size=[32,24]"]
RECORDER_CONFIG_KEYS = {"schema_version", "environment", "seed", "benchmark", "native", "schema", "video",
                        "decision_seconds", "presentation"}


def base(**sections):
    data = dict(run=dict(name="t"), env=dict(sim="skylands", image_size=[32, 24]),
                agent=dict(policy="random"), budget=dict(max_steps=5))
    for name, value in sections.items():
        data[name] = value
    return data


def llm(**harness):
    return dict(policy="llm", harness=dict(dict(kind="claude_cli", account_home="~/.claude"), **harness))


# -- manifest --------------------------------------------------------------------------
def test_examples_load_validate_and_round_trip(tmp_path):
    names = {p.stem for p in EXAMPLES.glob("*.yaml")}
    assert {"random-smoke", "demo-skylands", "claude-cli-teyvat", "openai-wildlands"} <= names
    for path in sorted(EXAMPLES.glob("*.yaml")):
        manifest = Manifest.load(path)
        data = manifest.to_dict()
        assert Manifest.from_dict(data) == manifest, path
        assert json.loads(json.dumps(data)) == data
        assert not manifest.privileged
        as_json = tmp_path / f"{path.stem}.json"
        as_json.write_text(json.dumps(data))
        assert Manifest.load(as_json) == manifest
    smoke = Manifest.load(EXAMPLES / "random-smoke.yaml")
    assert smoke.env.sim == ("teyvat", "wildlands", "skylands") and smoke.max_steps == 20
    claude = Manifest.load(EXAMPLES / "claude-cli-teyvat.yaml")
    assert claude.agent.harness.caps()["profile"] == "cli-stateless"
    assert Manifest.load(EXAMPLES / "openai-wildlands.yaml").agent.harness.caps()["profile"] == "api-stateless"


def test_defaults_and_shorthands():
    m = Manifest.from_dict(dict(base(), run="relay/notes", task=dict(seeds=3)))
    assert m.run.name == "relay/notes" and m.task.seeds == (3,)
    assert m.output_dir == Path("runs/relay/notes") and m.run_dir("skylands", 3).name == "skylands-3"
    assert m.env.variant == "small" and m.env.decision_seconds == 0.2 and m.record.video is None
    h = Manifest.from_dict(base(agent=llm(), budget=dict(max_steps=5, max_model_calls=5))).agent.harness
    assert (h.resolved_model, h.resolved_executable) == ("claude-opus-5-5", "claude")


@pytest.mark.parametrize("data, match", [
    (dict(base(), extra={}), "unknown section"),
    ({k: v for k, v in base().items() if k != "agent"}, "agent: missing required section"),
    (base(env=dict(sim="minecraft")), "env.sim"),
    (base(env=dict(sim=["teyvat", "teyvat"])), "only once"),
    (base(env=dict(sim="teyvat", fov=90)), r"env: unknown key\(s\) \['fov'\]"),
    (base(env=dict(sim="teyvat", image_size=[0, 24])), "env.image_size"),
    (base(env=dict(sim="teyvat", native_config=dict(skylands={}))), "env.native_config"),
    (base(budget={}), "budget.max_steps"),
    (base(budget=dict(max_steps=-1)), "budget.max_steps"),
    (base(task=dict(id="nope")), "task.id: .*unknown task"),
    (base(task=dict(id="ender_dragon")), "task.id: .*minecraft"),
    (base(task=dict(seeds=[1, 1])), "task.seeds"),
    (base(agent=dict(policy="oracle")), "agent.policy"),
    (base(agent=dict(policy="random", history_steps=3)), "only applies to policy llm"),
    (base(agent=dict(policy="llm")), "agent.harness: is required"),
    (base(agent=llm(), budget=dict(max_steps=5)), "budget.max_model_calls"),
    (base(budget=dict(max_steps=5, max_total_tokens=10)), "budget.max_total_tokens"),
    (base(agent=llm(kind="claude_cli", account_home=None), budget=dict(max_steps=5, max_model_calls=5)),
     "account_home: is required"),
    (base(agent=llm(kind="gemini_cli", account_home="~/.gemini"), budget=dict(max_steps=5, max_model_calls=5)),
     "account_email"),
    (base(agent=llm(kind="gemini_cli", account_home="~/.gemini", account_email="a@b", effort="max"),
          budget=dict(max_steps=5, max_model_calls=5)), "agent.harness.effort"),
    (base(agent=llm(max_output_tokens=10), budget=dict(max_steps=5, max_model_calls=5)), "not enforced"),
    (base(agent=llm(tools=["read", "write"]), budget=dict(max_steps=5, max_model_calls=5)), "fixed caps tools"),
    (base(agent=llm(profile="cli-workspace"), budget=dict(max_steps=5, max_model_calls=5)), "fixed caps profile"),
    (base(agent=llm(network="allowlist"), budget=dict(max_steps=5, max_model_calls=5)), "fixed caps network"),
    (base(agent=llm(persist_workspace="run"), budget=dict(max_steps=5, max_model_calls=5)),
     "fixed caps persist_workspace"),
    (base(agent=llm(network="everywhere"), budget=dict(max_steps=5, max_model_calls=5)), "agent.harness.network"),
    (base(record=dict(trajectory=False, video="gif")), "record.video"),
    (base(record=dict(video_renderer="eevee")), "record.video_renderer"),
])
def test_validation_errors(data, match):
    with pytest.raises(ManifestError, match=match):
        Manifest.from_dict(data)


def test_declared_caps_matching_the_harness_are_accepted():
    m = Manifest.from_dict(base(agent=llm(profile="cli-stateless", tools=["StructuredOutput"], network="model-endpoint",
                                          persist_workspace="none", max_turns_per_decision=3),
                                budget=dict(max_steps=5, max_model_calls=5)))
    assert m.agent.harness.tools == ("StructuredOutput",)


def test_overrides():
    data = apply_overrides(read(EXAMPLES / "random-smoke.yaml"),
                           ["env.image_size=[32, 24]", "budget.max_steps=3", "task.seeds=[1,2]", "run=other",
                            "agent.harness.kind=openai_api"])
    assert data["env"]["image_size"] == [32, 24] and data["budget"]["max_steps"] == 3
    assert data["task"]["seeds"] == [1, 2] and data["run"] == "other"
    for bad in ("nokey", "bogus.key=1", "env..sim=x"):
        with pytest.raises(ManifestError, match="--set"):
            apply_overrides({}, [bad])


# -- owb run / replay -------------------------------------------------------------------
@pytest.fixture(scope="module")
def smoke(tmp_path_factory):
    out = tmp_path_factory.mktemp("cli") / "smoke"
    args = ["run", str(EXAMPLES / "random-smoke.yaml"), *TINY, "--set", f"record.output_dir={out}"]
    return out, args, main(args)


def test_run_writes_recordings_and_manifest(smoke, capsys):
    out, args, code = smoke
    assert code == 0
    expected = Manifest.load(EXAMPLES / "random-smoke.yaml", ["env.image_size=[32,24]", f"record.output_dir={out}"])
    for name in ("teyvat", "wildlands", "skylands"):
        run_dir = out / f"{name}-7"
        assert {"config.json", "trajectory.jsonl", "summary.json", "manifest.json"} <= {p.name for p in run_dir.iterdir()}
        config = json.loads((run_dir / "config.json").read_text())
        assert set(config) == RECORDER_CONFIG_KEYS  # existing config.json keys unchanged
        assert config["benchmark"]["image_size"] == [32, 24] and config["benchmark"]["max_steps"] == 20
        summary = json.loads((run_dir / "summary.json").read_text())
        assert summary["steps"] == 20 and summary["termination_reason"] == "time_limit"
        recorded = json.loads((run_dir / "manifest.json").read_text())
        assert recorded["manifest"] == expected.to_dict()
        assert (recorded["environment"], recorded["seed"], recorded["privileged"]) == (name, 7, False)
        assert recorded["argv"] == ["owb", *args]


def test_run_matches_direct_run_episode(smoke, tmp_path):
    import owb
    from owb.eval import RandomPolicy, run_episode
    out, _, _ = smoke
    with owb.make_env("skylands", seed=7, max_steps=20, image_size=(32, 24)) as env:
        run_episode(env, RandomPolicy(), seed=7, output_dir=tmp_path / "direct")
    for name in ("config.json", "trajectory.jsonl", "summary.json"):
        assert (out / "skylands-7" / name).read_bytes() == (tmp_path / "direct" / name).read_bytes()


def test_replay_verify_and_refuse_overwrite(smoke, capsys):
    out, args, _ = smoke
    capsys.readouterr()
    for name in ("teyvat", "wildlands", "skylands"):
        assert main(["replay", str(out / f"{name}-7"), "--verify"]) == 0
        report = json.loads(capsys.readouterr().out)
        assert report["verified"] and report["checked"] == 21
    assert main(["replay", str(out / "skylands-7")]) == 0
    assert json.loads(capsys.readouterr().out)["summary"]["steps"] == 20
    before = (out / "teyvat-7" / "trajectory.jsonl").read_bytes()
    assert main(args) == 2
    assert "recording already exists" in capsys.readouterr().err
    assert (out / "teyvat-7" / "trajectory.jsonl").read_bytes() == before


def test_run_prints_one_line_per_episode_and_a_table(tmp_path, capsys):
    manifest = tmp_path / "m.json"
    manifest.write_text(json.dumps(base(task=dict(seeds=[1, 2]))))
    assert main(["run", str(manifest), "--seed", "4", "--set", f"record.output_dir={tmp_path / 'o'}"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].startswith("skylands-4: steps=5 ") and "privileged=false" in lines[0]
    assert lines[2].split()[:2] == ["run", "steps"] and lines[3].split()[:2] == ["skylands-4", "5"]
    assert sorted(p.name for p in (tmp_path / "o").iterdir()) == ["skylands-4"]


def test_native_config_overrides(tmp_path, capsys):
    data = base(env=dict(sim="skylands", image_size=[32, 24],
                         native_config=dict(skylands=dict(time_penalty=0.01, render_cfg=dict(max_dist=40)))),
                record=dict(output_dir=str(tmp_path)))
    path = tmp_path / "m.json"
    path.write_text(json.dumps(data))
    assert main(["run", str(path)]) == 0
    native = json.loads((tmp_path / "skylands-0" / "config.json").read_text())["native"]
    assert native["time_penalty"] == 0.01 and native["render_cfg"]["max_dist"] == 40
    assert native["render_cfg"]["width"] == 32  # make_env's own settings still apply
    assert main(["replay", str(tmp_path / "skylands-0"), "--verify"]) == 0
    data["env"]["native_config"]["skylands"] = dict(render_cfg=dict(bogus=1))
    data["record"]["output_dir"] = str(tmp_path / "bad")
    path.write_text(json.dumps(data))
    assert main(["run", str(path)]) == 2
    assert "env.native_config.skylands.render_cfg: unknown native config field(s) ['bogus']" in capsys.readouterr().err


def test_task_id_wraps_task_env(tmp_path, monkeypatch):
    from owb.tasks import registry
    from owb.tasks.teyvat import builtin_questline_spec
    spec = builtin_questline_spec(3)
    real = registry.load_task
    monkeypatch.setattr(registry, "load_task", lambda task_id: spec if task_id == spec.id else real(task_id))
    path = tmp_path / "m.json"
    path.write_text(json.dumps(base(env=dict(sim="teyvat", image_size=[32, 24]),
                                    task=dict(id=spec.id, seeds=[3]), record=dict(output_dir=str(tmp_path)))))
    assert main(["run", str(path)]) == 0
    run_dir = tmp_path / "teyvat-3"
    task = json.loads((run_dir / "task.json").read_text())
    assert task["task"] == spec.id and 0.0 <= task["progress"] <= 1.0
    rows = [json.loads(line) for line in (run_dir / "trajectory.jsonl").read_text().splitlines()]
    assert all(row["info"]["task"]["id"] == spec.id for row in rows)
    assert main(["replay", str(run_dir), "--verify"]) == 0


# -- listings ---------------------------------------------------------------------------
def test_tasks_list_and_show(capsys):
    assert main(["tasks", "list"]) == 0
    out = capsys.readouterr().out
    assert "ender_dragon" in out and "mine_diamond_from_scratch" in out and "minecraft" in out
    assert main(["tasks", "show", "ender_dragon"]) == 0
    assert "id: ender_dragon" in capsys.readouterr().out
    assert main(["tasks", "show", "nope"]) == 2


def test_agents_and_envs_list(capsys):
    assert main(["agents", "list"]) == 0
    out = capsys.readouterr().out
    for kind in ("openai_api", "claude_cli", "codex_cli", "gemini_cli"):
        assert kind in out
    assert "cli-workspace (not implemented)" in out
    oracle_rows = {line.split()[0]: line.split()[1] for line in out.splitlines()
                   if line.split() and line.split()[0] in {o.name for o in cli_agents.oracles()}}
    assert oracle_rows and set(oracle_rows.values()) == {"true"}
    assert {"teyvat_reference", "wildlands_reference"} <= set(oracle_rows)
    assert main(["envs", "list"]) == 0
    assert {"teyvat", "wildlands", "skylands"} <= {line.split()[0] for line in capsys.readouterr().out.splitlines()}


def test_render_prints_the_cycles_command(smoke, capsys):
    out, _, _ = smoke
    assert main(["render", str(out / "teyvat-7"), "--samples", "64"]) == 0
    text = capsys.readouterr().out
    assert "python -m owb.scripts.render_blender --env teyvat --seed 7" in text and "--samples 64" in text
    assert main(["render", str(out / "skylands-7")]) == 2
    assert "no Blender Cycles scene exporter" in capsys.readouterr().err


# -- privileged policies ----------------------------------------------------------------
@pytest.fixture
def test_oracle():
    from owb.eval import RandomPolicy
    spec = cli_agents.OracleSpec("test_oracle", "reads env state", "tests:none",
                                 factory=lambda env, manifest: RandomPolicy())
    cli_agents.register_oracle(spec)
    yield spec
    cli_agents.unregister_oracle(spec.name)


def test_privileged_policies_need_the_flag_and_are_marked(tmp_path, capsys, test_oracle):
    path = tmp_path / "m.json"
    path.write_text(json.dumps(base(agent=dict(policy="oracle(test_oracle)"),
                                    record=dict(output_dir=str(tmp_path / "o")))))
    assert main(["run", str(path)]) == 2
    assert "privileged run refused" in capsys.readouterr().err
    assert not (tmp_path / "o").exists()
    assert main(["run", str(path), "--allow-privileged"]) == 0
    assert "privileged=true" in capsys.readouterr().out
    run_dir = tmp_path / "o" / "skylands-0"
    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["policy"]["privileged"] is True and summary["policy"]["type"] == "test_oracle"
    assert json.loads((run_dir / "policy.json").read_text())["privileged"] is True
    assert json.loads((run_dir / "manifest.json").read_text())["privileged"] is True


def test_builtin_oracles_and_goal_hint_are_refused(tmp_path, capsys):
    path = tmp_path / "m.json"
    path.write_text(json.dumps(base(agent=dict(policy="oracle(teyvat_reference)"))))
    assert main(["run", str(path)]) == 2
    assert "privileged run refused" in capsys.readouterr().err
    assert main(["run", str(path), "--allow-privileged"]) == 2
    assert "owb.scripts.teyvat_reference" in capsys.readouterr().err
    path.write_text(json.dumps(base(env=dict(sim="skylands", include_goal_hint=True))))
    assert main(["run", str(path)]) == 2
    assert "include_goal_hint" in capsys.readouterr().err
    path.write_text(json.dumps(base(agent=dict(policy="oracle(missing)"))))
    assert main(["run", str(path), "--allow-privileged"]) == 2
    assert "unknown oracle" in capsys.readouterr().err


# -- benchmark.py translation -----------------------------------------------------------
def benchmark_args(**kw):
    args = dict(env=["teyvat"], seed=7, steps=12, size=(320, 240), variant="small", quality="fast", policy="demo",
                codex_account_home=None, codex_executable="codex", claude_account_home=None,
                claude_executable="claude", gemini_account_home=None, gemini_account_email=None,
                gemini_executable="gemini", decision_seconds=0.2, model=None, reasoning_effort="low",
                max_output_tokens=2048, max_model_calls=100, history_steps=6, api_timeout=60.0, video=None,
                video_renderer=None, blender=None, samples=48, threads=8, video_size=(960, 540),
                output=Path("/tmp/x"))
    return SimpleNamespace(**dict(args, **kw))


def test_benchmark_flags_translate_to_a_manifest():
    from owb.scripts.benchmark import manifest_from_args
    m = manifest_from_args(benchmark_args())
    assert (m.agent.policy, m.budget.max_steps, m.budget.max_model_calls) == ("demo", 12, None)
    assert m.env.image_size == (320, 240) and m.task.seeds == (7,) and m.output_dir == Path("/tmp/x")
    m = manifest_from_args(benchmark_args(policy="claude", claude_account_home=Path("~/.claude")))
    h = m.agent.harness
    assert (h.kind, h.model, h.account_home, h.executable, h.max_output_tokens) == (
        "claude_cli", "claude-opus-5-5", "~/.claude", "claude", None)
    assert m.budget.max_model_calls == 100 and m.agent.history_steps == 6
    h = manifest_from_args(benchmark_args(policy="gemini", gemini_account_home=Path("~/.gemini"),
                                          gemini_account_email="a@b.c")).agent.harness
    assert (h.kind, h.model, h.account_email, h.max_output_tokens) == ("gemini_cli", "gemini-4-argon", "a@b.c", 2048)
    h = manifest_from_args(benchmark_args(policy="openai", model="gpt-x", video="gif")).agent.harness
    assert (h.kind, h.model, h.timeout) == ("openai_api", "gpt-x", 60.0)
