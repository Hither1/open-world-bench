"""Headless rollout or turn-based terminal play: python -m owb.scripts.wildlands."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..env import DISCRETE_ACTIONS, make_small
from ..core.config import RenderConfig


def write_frame(path, frame):
    """Portable pixmap output requires no optional image libraries."""
    height, width = frame.shape[:2]
    with path.open("wb") as file:
        file.write(f"P6\n{width} {height}\n255\n".encode("ascii"))
        file.write(frame.tobytes())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--output", type=Path, default=Path("/tmp/wildlands.ppm"))
    parser.add_argument("--interactive", action="store_true", help="enter named actions at a terminal")
    parser.add_argument("--gif", type=Path, help="optional recording; requires the video extra")
    parser.add_argument("--renderer", choices=("landscape", "voxel"), default="landscape")
    parser.add_argument("--quality", choices=("fast", "balanced", "high"), default="balanced")
    parser.add_argument("--supersample", type=int, choices=(1, 2, 3), default=1,
                        help="samples per image axis; 2 gives smoother foliage at extra CPU cost")
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=180)
    parser.add_argument("--first-person", action="store_true", help="use the agent's eye camera")
    args = parser.parse_args()
    if args.steps <= 0:
        parser.error("--steps must be positive")
    if min(args.width, args.height) <= 0:
        parser.error("image dimensions must be positive")
    if args.output.suffix.lower() != ".ppm":
        parser.error("--output must have a .ppm extension")
    if args.gif:
        import imageio.v2 as imageio

    env = make_small(action_space="discrete", obs_rgb=False, max_steps=args.steps,
                     render_cfg=RenderConfig(style=args.renderer, quality=args.quality, supersample=args.supersample,
                                             width=args.width, height=args.height,
                                             max_dist=100, third_person=not args.first_person,
                                             cam_distance=5.5, cam_pitch_offset=-0.22, hud=True))
    frames, totals = [], {}
    try:
        _, info = env.reset(seed=args.seed)
        write_frame(args.output, env.render())
        if args.interactive:
            print("Commands:", ", ".join(DISCRETE_ACTIONS))
            print("Append a repeat count (e.g. forward 5); inventory lists slots; quit exits.")
            print(f"View refreshes at {args.output}")
        # Harvest pepper, select slot 3, cook, eat, then traverse the landscape.
        intro = ["interact", "next_slot", "next_slot", "next_slot", "cook", "eat"]
        while env.steps < args.steps:
            step = env.steps
            if args.interactive:
                try:
                    raw = input("wildlands> ").strip().lower().split()
                except (EOFError, KeyboardInterrupt):
                    break
                if not raw:
                    continue
                if raw[0] == "quit":
                    break
                if raw[0] == "inventory":
                    for slot, stack in enumerate(env.agent.inventory.stacks):
                        marker = "*" if slot == env.selected_slot else " "
                        print(f"{marker} {slot}: {stack.name} x{stack.count}")
                    continue
                try:
                    name = raw[0]
                    repeat = int(raw[1]) if len(raw) == 2 else 1
                    if name not in DISCRETE_ACTIONS or repeat <= 0 or len(raw) > 2:
                        raise ValueError
                except ValueError:
                    print("Use an action name and an optional positive repeat count.")
                    continue
            else:
                name = intro[step] if step < len(intro) else (
                    "turn_right" if step % 24 < 5 else "jump" if step % 5 == 0 else "forward")
                repeat = 1
            ended = False
            for _ in range(min(repeat, args.steps - env.steps)):
                _, _, terminated, truncated, info = env.step(DISCRETE_ACTIONS.index(name))
                for event in info["events"]:
                    totals[event["type"]] = totals.get(event["type"], 0) + 1
                if args.gif:
                    frames.append(env.render())
                ended = terminated or truncated
                if ended:
                    break
            if args.interactive:
                write_frame(args.output, env.render())
                print(f"health={info['health']:.1f} stamina={info['stamina']:.1f} "
                      f"weather={info['weather']} slot={env.selected_slot}")
                for event in info["events"]:
                    print(event)
            if ended:
                break
        write_frame(args.output, env.render())
        if args.gif and frames:
            imageio.mimsave(args.gif, frames, duration=1000 / env.metadata["render_fps"], loop=0)
        print(json.dumps(dict(seed=args.seed, steps=env.steps, health=info["health"],
                              stamina=info["stamina"], visited_cells=info["visited_cells"],
                              events=totals, frame=str(args.output)), indent=2))
    finally:
        env.close()


if __name__ == "__main__":
    main()
