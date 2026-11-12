"""Run and record any common-schema environment: python -m owb.scripts.benchmark."""
import argparse
import json
from pathlib import Path

from ..bench import available_environments, make_env
from ..bench.rollout import DemoPolicy, RandomPolicy, run_episode
from ..render.presentation import add_render_arguments


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", nargs="+", choices=available_environments(), default=list(available_environments()))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--task", choices=("summit", "cleansing", "desert"),
                        help="Goal-first Teyvat campaign; requires --env teyvat")
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
    if args.task is not None and args.env != ["teyvat"]:
        parser.error("--task requires --env teyvat")
    if args.steps <= 0 or min(*args.size, *args.video_size, args.samples, args.threads) <= 0:
        parser.error("steps and image dimensions must be positive")
    if len(set(args.env)) != len(args.env):
        parser.error("each environment must appear only once")
    for name in args.env:
        if (args.output / f"{name}-{args.seed}").exists():
            parser.error(f"recording already exists: {args.output / f'{name}-{args.seed}'}")
    policy_config = None
    if args.policy in ("openai", "codex", "claude", "gemini"):
        from ..bench.baselines import OpenAIPolicy, OpenAIPolicyConfig
        try:
            policy_config = OpenAIPolicyConfig(
                model=args.model or {"claude": "claude-opus-5-5", "gemini": "gemini-4-argon"}.get(args.policy, "gpt-6-astra"),
                reasoning_effort=args.reasoning_effort,
                max_output_tokens=args.max_output_tokens, max_calls=args.max_model_calls,
                history_steps=args.history_steps, timeout=args.api_timeout,
                decision_seconds=args.decision_seconds)
        except ValueError as exc:
            parser.error(str(exc))
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
    for name in args.env:
        if args.policy == "codex":
            from ..bench.baselines import CodexPolicy
            policy = CodexPolicy(policy_config, account_home=args.codex_account_home,
                                 executable=args.codex_executable)
        elif args.policy == "claude":
            from ..bench.baselines import ClaudePolicy
            policy = ClaudePolicy(policy_config, account_home=args.claude_account_home,
                                  executable=args.claude_executable)
        elif args.policy == "gemini":
            from ..bench.baselines import GeminiPolicy
            policy = GeminiPolicy(policy_config, account_home=args.gemini_account_home,
                                  account_email=args.gemini_account_email, executable=args.gemini_executable)
        else:
            policy = (OpenAIPolicy(policy_config) if policy_config is not None else
                  DemoPolicy() if args.policy == "demo" else RandomPolicy())
        with make_env(name, seed=args.seed, max_steps=args.steps, variant=args.variant,
                      image_size=tuple(args.size), render_quality=args.quality,
                      decision_seconds=args.decision_seconds, task=args.task) as env:
            try:
                summary = run_episode(env, policy, seed=args.seed,
                                      output_dir=args.output / f"{name}-{args.seed}", video=args.video,
                                      video_renderer=args.video_renderer, video_size=tuple(args.video_size),
                                      blender=args.blender, samples=args.samples, threads=args.threads)
            except Exception as exc:
                if policy_config is None:
                    raise
                from ..bench.baselines.codex_policy import CodexPolicyError
                from ..bench.baselines.claude_policy import ClaudePolicyError
                from ..bench.baselines.gemini_policy import GeminiPolicyError
                if isinstance(exc, (CodexPolicyError, ClaudePolicyError, GeminiPolicyError)):
                    parser.exit(1, str(exc) + "\n")
                if args.policy in ("codex", "claude", "gemini"):
                    raise
                from openai import OpenAIError
                if not isinstance(exc, OpenAIError):
                    raise
                code = getattr(exc, "code", None)
                if code in ("credit_balance_exhausted", "insufficient_quota"):
                    message = "OpenAI API credits/quota exhausted. Fund the API project or configure a funded OPENAI_API_KEY."
                else:
                    message = f"OpenAI request failed: {type(exc).__name__} (code={code})."
                parser.exit(1, message + f" See {args.output / f'{name}-{args.seed}' / 'error.json'}.\n")
            print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
