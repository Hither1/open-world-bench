"""Run the VISTA harness as a baseline on the Wildlands/Skylands milestone courses.

VISTA (https://github.com/joshhhhhan/VISTA) drives Claude Code or Codex CLI inside
its sandboxed player container. The model acts only through `play`; VISTA adds
`inspect`, `read_pixels`, `history` and GUIDE/WORKING notes. Logs match
`owb.scripts.wildlands_model` and are replay-verified the same way.

    python -u -B -m owb.scripts.vista_model --world wildlands --task a \
      --backend codex --auth-file ~/.codex-roboharness04/auth.json \
      --model gpt-6.1-sol --effort medium --output runs/vista-wildlands-pilot
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_VISTA = ROOT / '.cache' / 'vista'
VISTA_COMMIT = '30542e943d21922ad73f91e55888067dbec765bd'
OWN_SOURCES = ('owb/bench/baselines/vista_course.py', 'owb/scripts/vista_model.py')


def import_vista(source=DEFAULT_VISTA):
    """Put a VISTA checkout on sys.path without installing its pinned dependencies."""
    source = Path(source).resolve()
    if not (source / 'src' / 'vista').is_dir():
        raise SystemExit(f'No VISTA checkout at {source}; clone '
                         f'https://github.com/joshhhhhan/VISTA there at {VISTA_COMMIT}')
    if str(source / 'src') not in sys.path:
        sys.path.insert(0, str(source / 'src'))
    git = lambda *a: subprocess.run(['git', '-C', str(source), *a], capture_output=True,
                                    text=True).stdout.strip()
    return dict(source=str(source), commit=git('rev-parse', 'HEAD') or None,
                dirty=bool(git('status', '--porcelain')), expected_commit=VISTA_COMMIT)


def adapt_container_runtime():
    """Make VISTA's `docker run` work under rootless Podman on lab storage.

    VISTA runs the player as `--user <uid>:<gid>`. Under rootless Podman that maps to
    a subordinate UID that cannot write the run directory, and the user namespace
    drops the supplementary group that grants access to /n/holylfs06/LABS/...
    Container UID 0 is the invoking host user under rootless Podman (unprivileged on
    the host, all capabilities still dropped), so use `--user 0:0` plus
    `--group-add keep-groups`. Every other sandbox flag is unchanged. VISTA's socket
    directories come from tempfile, so TMPDIR must be short for AF_UNIX paths.
    Returns the recorded adaptation, or None under real Docker.
    """
    import os
    import tempfile
    os.environ['TMPDIR'] = '/tmp'
    tempfile.tempdir = None
    version = subprocess.run(['docker', '--version'], capture_output=True, text=True)
    if 'podman' not in (version.stdout + version.stderr).lower():
        return dict(container_runtime=version.stdout.strip(), tmpdir='/tmp', changes=[])
    from vista.backends.claude.runner import ClaudeCodeRunner
    from vista.backends.codex.runner import DockerCodexRunner

    def wrap(method):
        def command(self, *args, **kwargs):
            out = list(method(self, *args, **kwargs))
            user = out.index('--user')
            out[user + 1] = '0:0'
            out[out.index('run') + 1:out.index('run') + 1] = ['--group-add', 'keep-groups']
            return out
        command.__wrapped__ = method
        return command
    for runner in (ClaudeCodeRunner, DockerCodexRunner):
        if not hasattr(runner._docker_command, '__wrapped__'):
            runner._docker_command = wrap(runner._docker_command)
    return dict(container_runtime=version.stdout.strip(), tmpdir='/tmp',
                changes=['--user 0:0 (rootless Podman maps it to the invoking user)',
                         '--group-add keep-groups'])


def check_transport(backend, binary):
    """VISTA's version/image preflight, tolerating Podman's inspect template.

    VISTA reads `docker image inspect --format '{{json .Id}}'`; Podman rejects that
    template and reports IDs without the `sha256:` prefix (which `podman run` still
    accepts). Only that one call is translated; the pinned-version check is VISTA's.
    """
    import types
    from vista.backends import preflight

    def run(command, *args, **kwargs):
        if command[:3] == ['docker', 'image', 'inspect'] and command[-1] == '{{json .Id}}':
            out = subprocess.run([*command[:-1], '{{.Id}}'], *args, **kwargs)
            image = out.stdout.strip()
            image = image if image.startswith('sha256:') else 'sha256:' + image
            return subprocess.CompletedProcess(out.args, out.returncode, json.dumps(image),
                                               out.stderr)
        return subprocess.run(command, *args, **kwargs)
    original = preflight.subprocess
    preflight.subprocess = types.SimpleNamespace(run=run)
    try:
        return preflight.check_transport(backend, binary)
    finally:
        preflight.subprocess = original


def course_for(world, privileged=False):
    if world == 'wildlands':
        from owb.scripts.wildlands_model import WILDLANDS as plain, WILDLANDS_PRIVILEGED as upper
    else:
        from owb.scripts.skylands_model import SKYLANDS as plain, SKYLANDS_PRIVILEGED as upper
    return upper if privileged else plain


def play_loop(session, runtime, backend):
    """Prompt until the episode ends; the model plays many decisions per turn."""
    from vista.core.contracts import ObservationBundle
    from vista.core.profiles import Phase
    from vista.core.runtime import BudgetExhausted
    from owb.bench.baselines.vista_course import SCREEN_PROMPT
    started, redirect = False, ''
    try:
        while not session.finished and not runtime.terminal:
            phase = Phase.STEP if started else Phase.INITIAL
            lead = redirect if started else f'Mission: {session.task.objective}'
            observation = session.current
            blocks = tuple(b for b in (lead, SCREEN_PROMPT, observation, session.current_text)
                           if isinstance(b, ObservationBundle) or b.strip())
            task_input = runtime.prepare_input(phase, observation=observation,
                                               public_text=lead, blocks=blocks)
            before = session.decisions
            result = backend.run(task_input)
            started = True
            redirect = ''
            if runtime.terminal or result.returncode != 0 or runtime.failure:
                return
            if session.decisions == before:
                # The player ended its turn without acting; redirect, bounded by final_answers.
                reply = runtime.submit(result.final_message)
                if not runtime.terminal:
                    redirect = reply.text
    except BudgetExhausted:
        return


def run(task, directory, backend_factory, *, course, backend, model, effort, seed=7,
        max_calls=200, max_actions=6000, wall_seconds=3600, stop_after=None,
        tool_calls=None, backend_segments=None, vista_info=None, transport_provenance=None):
    from vista.core.runtime import TaskRuntime
    from owb.bench.baselines.vista_course import CourseSession, make_experiment, make_profile
    from owb.scripts.wildlands_model import save, verify_run
    from owb.scripts.wildlands_reference import sha
    directory.mkdir(parents=True, exist_ok=False)
    (directory / 'observations').mkdir()
    env = course.make_env(task, seed)
    obs, _ = env.reset(seed=seed)
    sources = {p: sha(ROOT / p) for p in (*course.sources, *OWN_SOURCES)}
    revision = hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()[:16]
    resolution = [env.cfg.render_cfg.width, env.cfg.render_cfg.height]
    config = dict(
        world=course.world, task=task, seed=seed, model=model, harness='vista',
        transport=f'vista-{backend}', reasoning_effort=effort, vista=vista_info,
        transport_provenance=transport_provenance, max_model_decisions=max_calls,
        max_primitive_actions=max_actions, max_wall_seconds=wall_seconds,
        max_tool_calls=tool_calls, max_backend_segments=backend_segments,
        max_action_repeat=30, seconds_per_action=.2,
        observation_renderer='native policy renderer; no presentation video',
        observation_resolution=resolution, observation_only=course.condition == 'observation_only',
        information_condition=course.condition, oracle_supplied=False,
        memory='VISTA GUIDE.md/WORKING.md plus retrievable history of every returned frame',
        initial_digest=env.state_digest(), sources=sources, benchmark_revision=revision,
        started_utc=datetime.now(timezone.utc).isoformat(), mission=course.missions[task],
        stop_after_milestones=stop_after,
        scope=('Single-seed VISTA harness pilot; one play call is one model decision.'
               + (f' Scored on the first {stop_after} milestones only.' if stop_after else '')))
    save(directory / 'config.json', config)
    session = runtime = backend_session = None
    failure = None
    started = time.monotonic()
    try:
        with (directory / 'actions.jsonl').open('w') as stream:
            session = CourseSession(course, task, env, obs, stream, directory,
                                    max_decisions=max_calls, max_actions=max_actions,
                                    wall_seconds=wall_seconds, stop_after=stop_after)
            profile = make_profile(course, backend=backend)
            (directory / 'control-guide.txt').write_text(profile.prompts.system)
            experiment = make_experiment(profile, course, task, backend=backend, model=model,
                                         effort=effort, seed=seed, revision=revision,
                                         resolution=resolution, max_decisions=max_calls,
                                         tool_calls=tool_calls, backend_segments=backend_segments)
            runtime = TaskRuntime(session, experiment, artifact_dir=directory / 'vista' / 'artifacts')
            backend_session = backend_factory(runtime, directory / 'vista' / 'transport')
            play_loop(session, runtime, backend_session)
            if runtime.failure:
                failure = dict(type='vista_runtime', reason=runtime.failure)
    except Exception as error:
        failure = dict(type=type(error).__name__, reason=str(error)[:500])
        (directory / 'error-traceback.txt').write_text(traceback.format_exc())
    finally:
        if runtime is not None:
            try:
                runtime.close()
            except Exception as error:
                failure = failure or dict(type='cleanup_failure', reason=str(error)[:500])
    reason = session.stop_reason if session is not None else 'not_started'
    if failure is not None and reason == 'running':
        reason = 'inference_or_interface_error'
    elif reason == 'running':
        reason = 'vista_budget_or_turn_limit'
    summary = dict(
        status='finished', task=task, model=model, harness='vista', stop_reason=reason,
        model_decisions=session.decisions if session else 0,
        rejected_actions=session.rejected if session else 0,
        primitive_actions=env.steps, milestones_completed=len(env.completed),
        required=len(env.milestones), completed=env.completed, success=env.success,
        current_objective=env.current.title if env.current else None,
        elapsed_wall_seconds=time.monotonic() - started, simulated_seconds=env.elapsed,
        vista_usage=dict(runtime.usage) if runtime else {},
        backend_usage=backend_session.usage_reports if backend_session else [],
        failure=failure, health=env.agent.health, final_digest=env.state_digest(),
        actions_sha256=sha(directory / 'actions.jsonl'), score_available=env.steps > 0)
    save(directory / 'summary.json', summary)
    env.close()
    verify_run(directory, course)
    print(json.dumps(dict(world=course.world, task=task, stop_reason=reason,
                          milestones=summary['milestones_completed'],
                          required=summary['required'])), flush=True)
    return summary


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--vista-source', type=Path, default=DEFAULT_VISTA)
    known, _ = p.parse_known_args(argv)
    vista_info = import_vista(known.vista_source)
    from vista.backends.launch import add_backend_arguments, prepare_backend, validate_backend_arguments

    p.add_argument('--world', choices=('wildlands', 'skylands'), default='wildlands')
    p.add_argument('--task', choices=('a', 'b'), default='a')
    p.add_argument('--seed', type=int, default=7)
    p.add_argument('--privileged', action='store_true',
                   help='Labelled upper-bound condition with evaluator-side state')
    add_backend_arguments(p)
    p.add_argument('--model', required=True)
    p.add_argument('--effort', required=True)
    p.add_argument('--max-calls', type=int, default=200)
    p.add_argument('--max-actions', type=int, default=6000)
    p.add_argument('--wall-seconds', type=float, default=3600)
    p.add_argument('--tool-calls', type=int, help='Bound on all tool calls incl. inspect/history')
    p.add_argument('--backend-segments', type=int)
    p.add_argument('--stop-after-milestones', type=int)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args(argv)
    if args.backend is None:
        p.error('--backend is required')
    validate_backend_arguments(args, p.error)
    course = course_for(args.world, args.privileged)
    directory = args.output / f'{args.world}-task-{args.task}-seed-{args.seed}'
    if directory.exists():
        p.error(f'{directory} already exists')
    container = adapt_container_runtime()
    factory, provenance = prepare_backend(args, p.error, check=check_transport)
    provenance = dict(provenance, container_adaptation=container)
    run(args.task, directory, factory, course=course, backend=args.backend, model=args.model,
        effort=args.effort, seed=args.seed, max_calls=args.max_calls,
        max_actions=args.max_actions, wall_seconds=args.wall_seconds,
        stop_after=args.stop_after_milestones, tool_calls=args.tool_calls,
        backend_segments=args.backend_segments, vista_info=vista_info,
        transport_provenance=provenance)


if __name__ == '__main__':
    main()
