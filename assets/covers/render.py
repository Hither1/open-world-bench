"""Run inside Blender: standard Cycles scene with a recorded cover exposure."""
import json
from pathlib import Path
import sys

import bpy

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "owb/render"))
import blender_scene

standard_setup = blender_scene.setup_scene


def setup_cover(*args, **kwargs):
    scene = standard_setup(*args, **kwargs)
    # Display exposure only: preserve all scene lights, materials and geometry.
    scene.view_settings.exposure = 0.65
    return scene


blender_scene.setup_scene = setup_cover
blender_scene.main()
scene = bpy.context.scene
output = Path(sys.argv[sys.argv.index("--output") + 1])
preferences = bpy.context.preferences.addons["cycles"].preferences
report = dict(engine=scene.render.engine, blender_version=bpy.app.version_string,
    device=scene.cycles.device, compute_device=preferences.compute_device_type,
    accelerators=[d.name for d in preferences.devices if d.use],
    samples=scene.cycles.samples, denoising=scene.cycles.use_denoising,
    max_bounces=scene.cycles.max_bounces,
    width=scene.render.resolution_x, height=scene.render.resolution_y,
    resolution_percentage=scene.render.resolution_percentage,
    view_transform=scene.view_settings.view_transform, look=scene.view_settings.look,
    exposure=scene.view_settings.exposure, transparent=scene.render.film_transparent,
    output_format=scene.render.image_settings.file_format,
    output_color_mode=scene.render.image_settings.color_mode)
output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
