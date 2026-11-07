"""Run and record any common-schema environment: python -m owb.scripts.benchmark.

Deprecated: use ``owb run <manifest.yaml>`` (docs/cli.md). The flags below are
translated into the equivalent manifest (``manifest_from_args``) and executed by
the same ``owb.cli.run.run_manifest``.
"""
import argparse
import json
from pathlib import Path
import sys

from ..bench import available_environments
from ..render.presentation import add_render_arguments


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", nargs="+", choices=available_environments(), default=list(available_environments()))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--size", type=int, nargs=2, metavar=("WIDTH", "HEIGHT"), default=(160, 120))
    parser.add_argument("--variant", choices=("small", "full"), default="small")
    parser.add_argument("--quality", choices=("fast", "presentation"), default="fast")
    parser.add_argument("--policy", choices=("random", "demo", "openai", "codex", "claude", "gemini"), default="demo")
    parser.add_argument("--codex-account-home", type=Path, help="Saved Codex account directory (requires auth.json)")
    parser.add_argument("--codex-executable", default="codex")
    parser.add_argument("--claude-account-home", type=Path, help="Saved Claude account directory (requires .credentials.json)")
    parser.add_argument("--claude-executable", default="claude")
    parser.add_argument("--gemini-account-home", type=Path, help="Saved Gemini .gemini directory")
    parser.add_argument("--gemini-account-email", help="Required Google account identity")
    parser.add_argument("--gemini-executable", default="gemini")
    parser.add_argument("--decision-seconds", type=float, default=0.2)
    parser.add_argument("--model", help="Defaults to claude-opus-5-5 for Claude, gemini-4-argon for Gemini, gpt-6-astra otherwise")
    parser.add_argument("--reasoning-effort", choices=("low", "medium", "high", "xhigh", "max"), default="low")
    parser.add_argument("--max-output-tokens", type=int, default=2048, help="Per-response cap for openai and gemini policies")
    parser.add_argument("--max-model-calls", type=int, default=100, help="Per-episode request cap")
    parser.add_argument("--history-steps", type=int, default=6)
    parser.add_argument("--api-timeout", type=float, default=60)
    parser.add_argument("--video", choices=("gif", "mp4"))
    add_render_arguments(parser, backend_flag="--video-renderer", default=None)
    parser.add_argument("--video-size", type=int, nargs=2, default=(960, 540), metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.steps <= 0 or min(*args.size, *args.video_size, args.samples, args.threads) <= 0:
        parser.error("steps and image dimensions must be positive")
    if len(set(args.env)) != len(args.env):
        parser.error("each environment must appear only once")
    for name in args.env:
        if (args.output / f"{name}-{args.seed}").exists():
            parser.error(f"recording already exists: {args.output / f'{name}-{args.seed}'}")
    model_policy = args.policy in ("openai", "codex", "claude", "gemini")
    if model_policy:
        if args.steps > args.max_model_calls:
            parser.error("--steps must not exceed --max-model-calls for a model policy")
        if args.policy == "codex" and args.codex_account_home is None:
            parser.error("--policy codex requires --codex-account-home")
        if args.policy == "claude" and args.claude_account_home is None:
            parser.error("--policy claude requires --claude-account-home")
        if args.policy == "gemini":
            if args.gemini_account_home is None or not args.gemini_account_email:
                parser.error("--policy gemini requires --gemini-account-home and --gemini-account-email")
            if args.reasoning_effort not in ("low", "medium", "high"):
                parser.error("Gemini 4 Argon supports low, medium, or high reasoning effort")
    from ..cli.manifest import ManifestError
    from ..cli.run import RunError, run_manifest
    try:
        manifest = manifest_from_args(args)
    except ManifestError as exc:
        parser.error(str(exc))
    print("DeprecationWarning: python -m owb.scripts.benchmark is deprecated; use `owb run <manifest.yaml>` "
          "(see docs/cli.md)", file=sys.stderr, flush=True)
    try:
        run_manifest(manifest, on_episode=lambda episode: print(json.dumps(episode.summary, indent=2)))
    except ManifestError as exc:
        parser.error(str(exc))
    except RunError as exc:
        parser.exit(1, str(exc) + "\n")


HARNESS_KINDS = {"openai": "openai_api", "codex": "codex_cli", "claude": "claude_cli", "gemini": "gemini_cli"}


def manifest_from_args(args):
    """The ``owb run`` manifest equivalent to these benchmark.py arguments (docs/cli.md)."""
    from ..cli.manifest import Manifest
    agent = dict(policy=args.policy)
    budget = dict(max_steps=args.steps)
    if args.policy in HARNESS_KINDS:
        kind = HARNESS_KINDS[args.policy]
        harness = dict(kind=kind, effort=args.reasoning_effort, timeout=args.api_timeout,
                       model=args.model or {"claude": "claude-opus-5-5", "gemini": "gemini-4-argon"}.get(
                           args.policy, "gpt-6-astra"))
        if kind in ("openai_api", "gemini_cli"):
            harness["max_output_tokens"] = args.max_output_tokens
        if kind != "openai_api":
            harness["account_home"] = str(getattr(args, f"{args.policy}_account_home"))
            harness["executable"] = getattr(args, f"{args.policy}_executable")
        if kind == "gemini_cli":
            harness["account_email"] = args.gemini_account_email
        agent = dict(policy="llm", harness=harness, history_steps=args.history_steps)
        budget["max_model_calls"] = args.max_model_calls
    return Manifest.from_dict(dict(
        run=dict(name=f"benchmark-{args.policy}"),
        env=dict(sim=list(args.env), variant=args.variant, image_size=list(args.size),
                 decision_seconds=args.decision_seconds, render_quality=args.quality),
        task=dict(seeds=[args.seed]), agent=agent, budget=budget,
        record=dict(output_dir=str(args.output), video=args.video, video_renderer=args.video_renderer,
                    video_size=list(args.video_size), samples=args.samples, threads=args.threads,
                    blender=args.blender)))

if __name__ == "__main__":
    main()
