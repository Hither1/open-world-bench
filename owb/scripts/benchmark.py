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
    parser.add_argument("--policy", choices=("random", "demo"), default="demo")
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
    for name in args.env:
        with make_env(name, seed=args.seed, max_steps=args.steps, variant=args.variant,
                      image_size=tuple(args.size)) as env:
            policy = DemoPolicy() if args.policy == "demo" else RandomPolicy()
            summary = run_episode(env, policy, seed=args.seed,
                                  output_dir=args.output / f"{name}-{args.seed}", video=args.video)
            print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
