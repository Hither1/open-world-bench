"""Run and record any common-schema environment: python -m owb.scripts.benchmark."""
import argparse
import json
from pathlib import Path

from ..bench import available_environments, make_env
from ..bench.rollout import DemoPolicy, RandomPolicy, run_episode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", nargs="+", choices=available_environments(), default=list(available_environments()))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--size", type=int, nargs=2, metavar=("WIDTH", "HEIGHT"), default=(160, 120))
    parser.add_argument("--variant", choices=("small", "full"), default="small")
    parser.add_argument("--quality", choices=("fast", "presentation"), default="fast")
    parser.add_argument("--policy", choices=("random", "demo", "openai", "codex"), default="demo")
    parser.add_argument("--codex-account-home", type=Path, help="Saved Codex account directory (requires auth.json)")
    parser.add_argument("--codex-executable", default="codex")
    parser.add_argument("--decision-seconds", type=float, default=0.2)
    parser.add_argument("--model", default="gpt-6-astra")
    parser.add_argument("--reasoning-effort", choices=("low", "medium", "high", "xhigh", "max"), default="low")
    parser.add_argument("--max-output-tokens", type=int, default=2048, help="Per-response cap for --policy openai only")
    parser.add_argument("--max-model-calls", type=int, default=100, help="Per-episode request cap")
    parser.add_argument("--history-steps", type=int, default=6)
    parser.add_argument("--api-timeout", type=float, default=60)
    parser.add_argument("--video", choices=("gif", "mp4"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.steps <= 0 or min(args.size) <= 0:
        parser.error("steps and image dimensions must be positive")
    if len(set(args.env)) != len(args.env):
        parser.error("each environment must appear only once")
    for name in args.env:
        if (args.output / f"{name}-{args.seed}").exists():
            parser.error(f"recording already exists: {args.output / f'{name}-{args.seed}'}")
    policy_config = None
    if args.policy in ("openai", "codex"):
        from ..bench.baselines import OpenAIPolicy, OpenAIPolicyConfig
        try:
            policy_config = OpenAIPolicyConfig(
                model=args.model, reasoning_effort=args.reasoning_effort,
                max_output_tokens=args.max_output_tokens, max_calls=args.max_model_calls,
                history_steps=args.history_steps, timeout=args.api_timeout,
                decision_seconds=args.decision_seconds)
        except ValueError as exc:
            parser.error(str(exc))
        if args.steps > args.max_model_calls:
            parser.error("--steps must not exceed --max-model-calls for a model policy")
        if args.policy == "codex" and args.codex_account_home is None:
            parser.error("--policy codex requires --codex-account-home")
    for name in args.env:
        if args.policy == "codex":
            from ..bench.baselines import CodexPolicy
            policy = CodexPolicy(policy_config, account_home=args.codex_account_home,
                                 executable=args.codex_executable)
        else:
            policy = (OpenAIPolicy(policy_config) if policy_config is not None else
                  DemoPolicy() if args.policy == "demo" else RandomPolicy())
        with make_env(name, seed=args.seed, max_steps=args.steps, variant=args.variant,
                      image_size=tuple(args.size), render_quality=args.quality,
                      decision_seconds=args.decision_seconds) as env:
            try:
                summary = run_episode(env, policy, seed=args.seed,
                                      output_dir=args.output / f"{name}-{args.seed}", video=args.video)
            except Exception as exc:
                if policy_config is None:
                    raise
                from ..bench.baselines.codex_policy import CodexPolicyError
                if isinstance(exc, CodexPolicyError):
                    parser.exit(1, str(exc) + "\n")
                if args.policy == "codex":
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
