"""State-driven original service fixtures, compatible with the Cycles exporter."""
import numpy as np

from open_world_bench.render.assets import METAL, BARK
from .tempest import COPPER, IRON, PATINA, wheel, pipe


def restoration_meshes(env, batch):
    batch.current_material = METAL
    for key, tid in env.restoration_targets.items():
        obj = env.targets[tid]
        p = obj["pos"]
        if "region" in obj:
            region, axis = obj["region"], obj["axis"]
            batch.box(p+(0, -.35, .12), (.65, 1.3, .4), PATINA)
            wheel(batch, p+(0, .15, -.16), .25, phase=env.service_dials[region][axis]*np.pi/2)
            color = (.18, .8, .4) if env.live_regions()[region] else (.85, .28, .08)
            batch.box(p+(0, .5, -.13), (.18, .10, .06), color, emissive=.5)
        elif key == "forge":
            batch.box(p+(0, -.25, 0), (2.4, .22, 1.3), IRON)
            for x in (-1., 1.):
                for z in (-.5, .5):
                    batch.box(p+(x, -.65, z), (.12, .8, .12), IRON)
            pipe(batch, [p+(-.8, 0, 0), p+(-.8, 1., 0), p+(.8, 1., 0), p+(.8, 0, 0)], .12)
        elif key == "socket":
            pipe(batch, [p+(-.8, -1, .3), p+(-.8, .7, .3), p+(.8, .7, .3), p+(.8, -1, .3)], .16)
            wheel(batch, p+(0, .5, 0), .4, phase=.8 if env.manifold_live() else 0.)
        elif "refuge" in obj:
            batch.box(p, (.7, 1.8, .5), PATINA)
            armed = env.refuge_switches[obj["refuge"]]
            batch.tube(p+(0, 0, -.3), p+(.25 if armed else -.25, .4, -.5), .07, COPPER)
    if env.tool_location in ("depot", "carried", "forge"):
        if env.tool_location == "carried":
            p = env.agent.pos+(0, .6, .3)
        else:
            key = "tooling" if env.tool_location == "depot" else "forge"
            p = env.targets[env.restoration_targets[key]]["pos"]+(0, -.3, 0)
        batch.current_material = BARK
        batch.box(p, (.8, .45, .5), (.30, .19, .10))
        batch.current_material = METAL
        for x in (-.27, .27):
            batch.box(p+(x, 0, 0), (.06, .48, .53), COPPER)
        batch.tube(p+(-.15, .3, 0), p+(.15, .3, 0), .035, IRON)
    # Distinguish the original pressure manifold from the existing wooden cargo.
    c = env.cargo["pressure_manifold"]
    p = c["pos"]
    for z in (-.45, .45):
        pipe(batch, [p+(-.6, 1.3, z), p+(-.6, 1.9, z), p+(.6, 1.9, z), p+(.6, 1.3, z)], .14)
    wheel(batch, p+(0, 1.65, -.7), .3)
