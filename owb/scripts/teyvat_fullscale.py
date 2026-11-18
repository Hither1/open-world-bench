"""Full-scale Teyvat Task A tools.

``world-check`` builds the full-scale world for each seed, proves its region
gates and opened routes, drives the real controller through the designed glide
and the strongest climbs at every gate wall, and writes for each seed:

* ``report.json``: every stage, model parameter, controller attempt, digest;
* ``map.png``: a diagnostic top-down map (native shading, not a Cycles render).

    python -B -m owb.scripts.teyvat_fullscale world-check --seed 7 19 --output /tmp/fullscale-world
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from open_world_bench.architecture import Architecture
from open_world_bench.campaign_env import plain
from open_world_bench.fullscale_reach import check_world
from open_world_bench.fullscale_world import build_world
from open_world_bench.world_map import terrain_surface

LABELS = {
    "spawn": "village", "cache": "cache islet", "camp": "raider camp", "ferry_low": "lower ferry",
    "ferry_high": "upper ferry", "outpost": "outpost", "west_edge": "west edge", "ruin": "ruin",
    "corridor_mouth": "ravine gate", "relay_1": "field relay 1", "relay_2": "field relay 2",
    "summit_base": "service lift", "summit_top": "summit beacon",
    "junction_1": "reed post 1", "junction_2": "reed post 2", "junction_3": "reed post 3",
}


def diagnostic_map(world, report, path: Path, scale: int = 2) -> None:
    from PIL import Image, ImageDraw, ImageFont

    rgb, (x0, z0, x1, z1) = terrain_surface(world.terrain, world.props, Architecture(world.buildings), 512)
    image = Image.fromarray(np.ascontiguousarray(rgb.transpose(1, 0, 2)))   # rows run north to south
    image = image.resize((image.width * scale, image.height * scale), Image.NEAREST)
    draw = ImageDraw.Draw(image)
    px = lambda x, z: ((x - x0) / (x1 - x0) * image.width, (z - z0) / (z1 - z0) * image.height)
    font = ImageFont.load_default(size=14)
    for lift in world.lifts:
        draw.line([px(*lift.bottom), px(*lift.top)], fill=(250, 210, 40), width=4)
    for gate in world.gates:
        (gx, _, gz), (sx, _, sz) = gate["position"], gate["size"]
        draw.rectangle([px(gx - sx / 2, gz - sz / 2), px(gx + sx / 2, gz + sz / 2)], outline=(220, 40, 40), width=4)
    for name, label in LABELS.items():
        x, z = world.layout.anchors[name]
        u, v = px(x, z)
        draw.ellipse([u - 5, v - 5, u + 5, v + 5], fill=(255, 255, 255), outline=(20, 20, 20), width=2)
        draw.text((u + 8, v - 8), label, fill=(255, 255, 255), font=font, stroke_width=2, stroke_fill=(20, 20, 20))
    status = "all checks pass" if report["ok"] else "CHECKS FAILED"
    caption = (f"Diagnostic top-down map (native shading, not a Cycles render) - seed {world.seed} - {status}. "
               "Yellow: lifts. Red: ravine gate.")
    draw.rectangle([0, image.height - 26, image.width, image.height], fill=(20, 20, 20))
    draw.text((8, image.height - 21), caption, fill=(240, 240, 240), font=font)
    image.save(path)


def world_check(seeds, output: Path) -> bool:
    output.mkdir(parents=True, exist_ok=False)
    summary, ok = [], True
    for seed in seeds:
        started = time.time()
        world = build_world(seed)
        report = check_world(world)
        report["seconds"] = round(time.time() - started, 1)
        directory = output / f"seed-{seed}"
        directory.mkdir()
        (directory / "report.json").write_text(json.dumps(plain(report), indent=2, allow_nan=False) + "\n")
        diagnostic_map(world, report, directory / "map.png")
        ok &= bool(report["ok"])
        summary.append(dict(seed=seed, ok=bool(report["ok"]), digest=report["world_digest"],
                            stages={s["name"]: s["ok"] for s in report["stages"]},
                            max_climb=report["climbs"]["max_gained"], seconds=report["seconds"]))
        print(f"seed {seed}: {'ok' if report['ok'] else 'FAILED'} in {report['seconds']} s")
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return ok


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("world-check", help="prove gates and routes; write reports and diagnostic maps")
    check.add_argument("--seed", type=int, nargs="+", default=[7, 19])
    check.add_argument("--output", type=Path, required=True, help="new directory for reports")
    args = parser.parse_args(argv)
    if args.command == "world-check":
        raise SystemExit(0 if world_check(args.seed, args.output) else 1)


if __name__ == "__main__":
    main()
