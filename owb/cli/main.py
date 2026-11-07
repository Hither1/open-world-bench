"""``owb``: run, replay, render and inspect Open-World Bench from one entry point.

    owb run configs/examples/random-smoke.yaml [--seed N ...] [--set key=value ...]
    owb replay RUN_DIR [--verify]
    owb tasks list | owb tasks show ID
    owb agents list
    owb envs list
    owb render RUN_DIR [render_blender options]

See docs/cli.md for the manifest schema.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


def _fmt(value):
    if value is None:
        return "-"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _table(headers, rows):
    rows = [[_fmt(v) for v in row] for row in rows]
    widths = [max(len(h), *(len(r[i]) for r in rows)) if rows else len(h) for i, h in enumerate(headers)]
    lines = ["  ".join(h.ljust(w) for h, w in zip(headers, widths))]
    lines += ["  ".join(v.ljust(w) for v, w in zip(row, widths)) for row in rows]
    return "\n".join(line.rstrip() for line in lines)


def _episode_line(episode):
    s = episode.summary
    line = (f"{episode.sim}-{episode.seed}: steps={s['steps']} return={_fmt(s['episode_return'])} "
            f"success={_fmt(s['success'])} progress={_fmt(s['task_progress'])} "
            f"reason={s['termination_reason']}")
    if s.get("budget_hit"):
        line += f" budget_hit={s['budget_hit']}"
    if episode.task is not None:
        line += f" task_progress={_fmt(episode.task.get('progress'))}"
    line += f" privileged={_fmt(episode.privileged)}"
    return line + (f" -> {episode.run_dir}" if episode.run_dir is not None else "")


# -- run ---------------------------------------------------------------------------------
def cmd_run(args):
    from .manifest import Manifest, ManifestError
    from .run import RunError, run_manifest
    overrides = list(args.set or [])
    if args.seed:
        overrides.append("task.seeds=" + json.dumps(args.seed))
    try:
        manifest = Manifest.load(args.manifest, overrides)
    except (ManifestError, OSError) as exc:
        return _error(f"invalid manifest {args.manifest}: {exc}")
    if manifest.privileged and args.allow_privileged:
        print("warning: privileged run (" + "; ".join(manifest.privileged_reasons)
              + "); results are not benchmark-agent results", file=sys.stderr)
    try:
        episodes = run_manifest(manifest, allow_privileged=args.allow_privileged,
                                on_episode=lambda e: print(_episode_line(e), flush=True),
                                argv=["owb", *args.argv])
    except ManifestError as exc:
        return _error(str(exc))
    except RunError as exc:
        return _error(str(exc))
    print()
    print(_table(["run", "steps", "return", "success", "progress", "reason", "privileged"],
                 [[f"{e.sim}-{e.seed}", e.summary["steps"], e.summary["episode_return"], e.summary["success"],
                   e.summary["task_progress"], e.summary["termination_reason"], e.privileged]
                  for e in episodes]))
    return 0


# -- replay ------------------------------------------------------------------------------
def cmd_replay(args):
    run_dir = Path(args.run_dir)
    if not (run_dir / "config.json").is_file():
        return _error(f"{run_dir} is not a recorded run (no config.json)")
    if not args.verify:
        config = json.loads((run_dir / "config.json").read_text())
        out = dict(run_dir=str(run_dir), environment=config["environment"], seed=config["seed"],
                   benchmark=config["benchmark"], video=config.get("video"),
                   presentation=config.get("presentation"))
        for name in ("summary", "error", "manifest", "task"):
            path = run_dir / f"{name}.json"
            out[name] = json.loads(path.read_text()) if path.is_file() else None
        if out["manifest"] is not None:
            out["manifest"] = dict(privileged=out["manifest"].get("privileged"),
                                   run=out["manifest"]["manifest"]["run"])
        print(json.dumps(out, indent=2))
        return 0
    from ..eval.replay import DEFAULT_FIELDS, verify_run
    try:
        report = verify_run(run_dir, compare=tuple(args.fields or DEFAULT_FIELDS))
    except ValueError as exc:
        return _error(str(exc))
    print(json.dumps(report, indent=2))
    return 0 if report["verified"] else 1


# -- tasks / agents / envs ---------------------------------------------------------------
def cmd_tasks(args):
    from ..bench.framework import available_environments
    from ..tasks.registry import available_tasks, load_task, task_path
    if args.action == "list":
        rows = []
        for task_id in available_tasks(args.sim):
            spec = load_task(task_id)
            rows.append([spec.id, spec.sim, spec.sim in available_environments(), len(spec.milestones), spec.title])
        print(_table(["id", "sim", "env", "milestones", "title"], rows))
        return 0
    if not args.id:
        return _error("tasks show needs a task id")
    try:
        spec = load_task(args.id)
    except KeyError as exc:
        return _error(exc.args[0])
    import yaml
    print(f"# {task_path(args.id)}")
    print(yaml.safe_dump(spec.to_dict(), sort_keys=False, allow_unicode=True, width=100), end="")
    return 0


def cmd_agents(args):
    from ..agents.harness.base import PROFILES
    from .agents import harness_rows, oracles
    print("policies: random, demo (reference, observation-only), llm (agent.harness), oracle(<name>)\n")
    print("harness kinds (agent.harness.kind):")
    print(_table(["kind", "profile", "default model", "executable", "tools", "network", "max_turns"],
                 [[kind, caps["profile"], model, exe, ",".join(caps["tools"]) or "none", caps["network"],
                   caps["max_turns_per_decision"]] for kind, model, exe, caps in harness_rows()]))
    used = {caps["profile"] for *_, caps in harness_rows()}
    print("\nprofiles (docs/agent-sandbox.md §4.2): "
          + ", ".join(p + ("" if p in used else " (not implemented)") for p in PROFILES))
    print("\noracles (agent.policy: oracle(<name>); need --allow-privileged):")
    print(_table(["name", "privileged", "owb run", "description", "run with"],
                 [[o.name, o.privileged, "yes" if o.runnable else "no", o.description, o.script]
                  for o in oracles()]))
    return 0


def cmd_envs(args):
    from ..bench.framework import environment_specs
    print(_table(["name", "implemented", "description"],
                 [[s.name, s.implemented, s.description] for s in environment_specs()]))
    return 0


# -- render ------------------------------------------------------------------------------
def cmd_render(args):
    """``render_blender`` renders an env's reset snapshot; it does not read run directories."""
    import shlex
    run_dir = Path(args.run_dir)
    if not (run_dir / "config.json").is_file():
        return _error(f"{run_dir} is not a recorded run (no config.json)")
    config = json.loads((run_dir / "config.json").read_text())
    name, bench = config["environment"], config["benchmark"]
    if name not in ("teyvat", "wildlands"):
        return _error(f"{name} has no Blender Cycles scene exporter; its recordings use the native "
                      "observation renderer (see docs/rendering.md)")
    command = ["python", "-m", "owb.scripts.render_blender", "--env", name, "--seed", str(config["seed"]),
               "--variant", bench["variant"], "--output", str(run_dir / "blender"), *args.extra]
    print("owb.scripts.render_blender does not accept a run directory yet: it exports and renders the\n"
          "reset state of a freshly built env with Blender Cycles (no CPU fallback), not the recorded\n"
          "trajectory. The reset snapshot of this run's env and seed is rendered by:\n")
    print("  " + " ".join(shlex.quote(part) for part in command))
    print("\nTo render every recorded decision, re-run the manifest with record.video: gif|mp4\n"
          "(Cycles by default for Teyvat/Wildlands).")
    return 0


def _error(message):
    print(f"owb: error: {message}", file=sys.stderr)
    return 2


def parser():
    p = argparse.ArgumentParser(prog="owb", description="Open-World Bench command line (docs/cli.md).")
    sub = p.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run a manifest (one run directory per env x seed)")
    run.add_argument("manifest", type=Path, help="run manifest (.yaml/.yml/.json)")
    run.add_argument("--seed", type=int, nargs="+", help="override task.seeds")
    run.add_argument("--set", action="append", metavar="KEY=VALUE",
                     help="override a manifest key, e.g. --set budget.max_steps=20 (value parsed as YAML)")
    run.add_argument("--allow-privileged", action="store_true",
                     help="permit oracle policies / privileged observations (marked privileged: true)")
    run.set_defaults(func=cmd_run)

    replay = sub.add_parser("replay", help="show a recorded run, or re-simulate it with --verify")
    replay.add_argument("run_dir")
    replay.add_argument("--verify", action="store_true", help="replay the actions and compare the trajectory")
    replay.add_argument("--fields", nargs="+", choices=("proprio", "reward", "events", "info"),
                        help="fields compared by --verify (default: proprio reward events)")
    replay.set_defaults(func=cmd_replay)

    tasks = sub.add_parser("tasks", help="list or show shipped task specs")
    tasks.add_argument("action", choices=("list", "show"))
    tasks.add_argument("id", nargs="?")
    tasks.add_argument("--sim", help="only tasks for this sim (list)")
    tasks.set_defaults(func=cmd_tasks)

    agents = sub.add_parser("agents", help="list policies, harness kinds, profiles and oracles")
    agents.add_argument("action", choices=("list",))
    agents.set_defaults(func=cmd_agents)

    envs = sub.add_parser("envs", help="list environments")
    envs.add_argument("action", choices=("list",))
    envs.set_defaults(func=cmd_envs)

    render = sub.add_parser("render", help="how to render a run with Blender Cycles")
    render.add_argument("run_dir")
    render.add_argument("extra", nargs=argparse.REMAINDER, help="extra owb.scripts.render_blender options")
    render.set_defaults(func=cmd_render)
    return p


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    args = parser().parse_args(argv)
    args.argv = argv
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
