"""Run the PRO-LONG coding-agent baseline on the Wildlands / Skylands milestone courses.

Same courses, observation allowlist, action interface, budgets and outputs as
``owb.scripts.wildlands_model``; the policy is PRO-LONG's log-plus-coding-agent
harness instead of one schema-constrained model response per decision. See
docs/prolong-baseline.md.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import time

from owb.bench.baselines.prolong import ClaudeCodeBackend, CodexBackend, ProLongError, ProLongPolicy
from owb.env.wildlands_reference import plain
from owb.scripts.skylands_model import SKYLANDS
from owb.scripts.wildlands_model import WILDLANDS, save, verify_run
from owb.scripts.wildlands_reference import ROOT, sha

COURSES = {"wildlands": WILDLANDS, "skylands": SKYLANDS}
EXTRA_SOURCES = ("owb/bench/baselines/prolong.py", "owb/scripts/prolong_model.py")
TRANSIENT = ("timeout", "agent_call_failed")


def run(task, directory, backend, *, course, seed=7, max_calls=200, max_actions=6000, wall_seconds=3600,
        action_cap=20, log_window=None, stop_after=None, memory_chars=6000):
    directory.mkdir(parents=True, exist_ok=False)
    for name in ("requests", "responses"):
        (directory / name).mkdir()
    env = course.make_env(task, seed)
    obs, _ = env.reset(seed=seed)
    base = course.policy(None, memory_chars=memory_chars)    # allowlist, schema and decoder only
    policy = ProLongPolicy(base, backend, directory / "prolong", mission=course.missions[task],
                           action_cap=action_cap, log_window=log_window, max_calls=max_calls)
    sources = {p: sha(ROOT / p) for p in course.sources + EXTRA_SOURCES}
    config = dict(
        world=course.world, task=task, seed=seed, baseline="prolong", backend=backend.name,
        model=backend.model, reasoning_effort=backend.effort, executable=backend.executable,
        account_home=str(backend.account_home), auth=backend.auth,
        log_condition="full" if log_window is None else "no_log" if log_window == -1 else f"last_{log_window}",
        max_agent_calls=max_calls, action_cap_per_call=action_cap, max_primitive_actions=max_actions,
        max_wall_seconds=wall_seconds, max_action_repeat=30, seconds_per_action=.2,
        request_timeout_seconds=backend.timeout, max_consecutive_transport_errors=3,
        output_token_limit_enforced=False, sandbox="bubblewrap: workspace rw, history ro; repository not mounted; "
        "network unrestricted", observation_renderer="native policy renderer; no presentation video",
        observation_resolution=[env.cfg.render_cfg.width, env.cfg.render_cfg.height],
        observation_only=True, information_condition="observation_only", oracle_supplied=False,
        initial_digest=env.state_digest(), sources=sources, mission=course.missions[task],
        started_utc=datetime.now(timezone.utc).isoformat(), stop_after_milestones=stop_after,
        reference="https://github.com/alexisfox7/PRO-LONG (arXiv:2607.20064)",
        scope=("Single-seed PRO-LONG pilot; agent calls are batches of up to action_cap entries, so the call "
               "budget is not comparable to the single-action baseline's decision budget."
               + (f" Scored on the first {stop_after} milestones only." if stop_after else "")))
    save(directory / "config.json", config)
    (directory / "control-guide.txt").write_text(policy.system_prompt)
    started = time.monotonic()
    reason, failure, errors = "running", None, 0

    def status(state):
        return dict(status=state, task=task, model=backend.model, backend=backend.name,
                    agent_calls=policy.calls, model_decisions=policy.calls, primitive_actions=env.steps,
                    milestones_completed=len(env.completed), required=len(env.milestones),
                    completed=env.completed, current_objective=env.current.title if env.current else None,
                    elapsed_wall_seconds=time.monotonic() - started, simulated_seconds=env.elapsed,
                    input_tokens=policy.input_tokens, output_tokens=policy.output_tokens,
                    cost_usd=policy.cost_usd, session_id=backend.session_id)

    save(directory / "status.json", status("running"))
    obs = course.observe(env, dict(obs, rgb=env.render().copy()), [])
    policy.observe(obs, env.steps, "Initial state | primitive actions 0")
    try:
        with (directory / "actions.jsonl").open("w") as stream:
            while not env._ended:
                if policy.calls >= max_calls: reason = "agent_call_limit"; break
                if env.steps >= max_actions: reason = "primitive_action_limit"; break
                if time.monotonic() - started >= wall_seconds: reason = "wall_time_limit"; break
                if stop_after and len(env.completed) >= stop_after: reason = "milestone_target_reached"; break
                call = policy.calls + 1
                try:
                    entries = policy.plan()
                    errors = 0
                except ProLongError as error:
                    # Transient failures retry against the call budget; the session and workspace persist.
                    errors += 1
                    if errors >= 3 or error.code not in TRANSIENT:
                        raise
                    continue
                finally:
                    if policy.last_request is not None:
                        save(directory / "requests" / f"{call:04}.json", policy.last_request)
                    if policy.last_response is not None:
                        save(directory / "responses" / f"{call:04}.json", policy.last_response)
                for index, (action, repeat, entry) in enumerate(entries, 1):
                    begin, events = env.steps, []
                    stop = False
                    for _ in range(min(repeat, max_actions - env.steps)):
                        milestone, before = env.current, env.agent.pos.copy()
                        obs, reward, terminated, truncated, info = env.step(action)
                        row = dict(step=env.steps, decision=call, entry=index,
                                   milestone=milestone.id if milestone else None, action=plain(action),
                                   position_before=plain(before), position_after=plain(env.agent.pos),
                                   seconds=env.elapsed, events=plain(info["events"]), reward=reward)
                        events.extend(info["events"])
                        completed = any(e["type"] == "milestone_complete" for e in info["events"])
                        if completed:
                            row["digest"] = env.state_digest()
                        stream.write(json.dumps(row, allow_nan=False) + "\n")
                        stream.flush()
                        if terminated or truncated or completed:
                            stop = True
                            break
                    stop = stop or env.steps >= max_actions
                    obs = course.observe(env, dict(obs, rgb=env.render().copy()), events)
                    policy.observe(obs, env.steps,
                                   f"Call {call} | Entry {index}/{len(entries)} | primitive actions {env.steps}\n\n"
                                   f"[ENTRY] {json.dumps(entry, separators=(',', ':'))}\n"
                                   f"executed {env.steps - begin} of {repeat} repeats")
                    if stop:
                        if index < len(entries):
                            policy.note("PLAN STOPPED", f"{len(entries) - index} remaining entries of call {call} "
                                        "were discarded (milestone completed, episode end or action budget)")
                        break
                save(directory / "status.json", status("running"))
                print(f"{course.world} {task.upper()} call {call}/{max_calls}: {len(entries)} entries, "
                      f"{len(env.completed)}/{len(env.milestones)} milestones, {env.steps} actions", flush=True)
            if env._ended:
                reason = "success" if env.success else "death" if not env.agent.alive else "environment_time_limit"
    except Exception as error:
        reason = "inference_or_interface_error"
        failure = dict(type=type(error).__name__, code=getattr(error, "code", None))
        if isinstance(error, ProLongError):
            failure["reason"] = str(error)
        save(directory / "error.json", failure)
    finally:
        summary = dict(**status("finished"), stop_reason=reason, success=env.success, failure=failure,
                       health=env.agent.health, final_digest=env.state_digest(),
                       actions_sha256=sha(directory / "actions.jsonl"), policy=policy.metadata(),
                       score_available=env.steps > 0)
        save(directory / "summary.json", summary)
        save(directory / "status.json", summary)
        env.close()
    verify_run(directory, course)
    print(json.dumps(dict(world=course.world, task=task, stop_reason=reason,
                          milestones=summary["milestones_completed"], required=summary["required"],
                          score_available=summary["score_available"])), flush=True)
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--world", choices=sorted(COURSES), default="wildlands")
    p.add_argument("--task", nargs="+", choices=("a", "b"), default=["a"])
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--backend", choices=("claude-code", "codex"), default="claude-code")
    p.add_argument("--account-home", type=Path, required=True,
                   help="CLI profile: CLAUDE_CONFIG_DIR with .credentials.json, or CODEX_HOME with auth.json")
    p.add_argument("--executable", help="CLI executable (default: claude or codex on PATH)")
    p.add_argument("--model", help="default: claude-opus-5-5 (claude-code) or gpt-6.1-sol (codex)")
    p.add_argument("--effort", choices=("low", "medium", "high", "xhigh", "max"), default="high")
    p.add_argument("--max-calls", type=int, default=200, help="agent calls (each returns a batch of entries)")
    p.add_argument("--action-cap", type=int, default=20, help="entries per call (PRO-LONG default 20)")
    p.add_argument("--max-actions", type=int, default=6000)
    p.add_argument("--wall-seconds", type=float, default=4 * 3600)
    p.add_argument("--timeout", type=float, default=1800, help="seconds per agent call")
    p.add_argument("--log-window", type=int,
                   help="omit for the full log (PRO-LONG); N>0 keeps the last N entry sections; -1 = no-log baseline")
    p.add_argument("--stop-after-milestones", type=int)
    args = p.parse_args()
    if min(args.max_calls, args.max_actions, args.wall_seconds, args.timeout, args.action_cap,
           args.stop_after_milestones or 1) <= 0:
        p.error("Limits must be positive")
    if args.log_window is not None and (args.log_window == 0 or args.log_window < -1):
        p.error("--log-window must be -1 or positive")
    if len(set(args.task)) != len(args.task):
        p.error("Each task may appear only once")
    for task in args.task:
        if (args.output / f"task-{task}-seed-{args.seed}").exists():
            p.error("Output task directory already exists")
    course = COURSES[args.world]

    def worker(task):
        if args.backend == "claude-code":
            backend = ClaudeCodeBackend(args.account_home, executable=args.executable or "claude",
                                        model=args.model or "claude-opus-5-5", effort=args.effort,
                                        timeout=args.timeout)
        else:
            backend = CodexBackend(args.account_home, executable=args.executable or "codex",
                                   model=args.model or "gpt-6.1-sol", effort=args.effort, timeout=args.timeout)
        return run(task, args.output / f"task-{task}-seed-{args.seed}", backend, course=course, seed=args.seed,
                   max_calls=args.max_calls, max_actions=args.max_actions, wall_seconds=args.wall_seconds,
                   action_cap=args.action_cap, log_window=args.log_window, stop_after=args.stop_after_milestones)

    with ThreadPoolExecutor(max_workers=len(args.task)) as pool:
        results = list(pool.map(worker, args.task))
    save(args.output / "results.json", results)


if __name__ == "__main__":
    main()
