# Blender rendering and Godot asset handoff

OWB can now export a live **Teyvat or Wildlands** scene and render it in Blender.
The Python simulator continues to own actions, physics, rewards, and observations.
Blender Cycles is the default for Teyvat and Wildlands presentation output,
including screenshots and recordings. Godot is not yet a live environment
backend. CPU presentation requires an explicit option; missing or failing
Blender never causes an automatic quality downgrade.

The export includes terrain, visible procedural props, characters and their
current poses, entities, water, the camera, and directional lighting. Wildlands
terrain is meshed from the landscape renderer's density field, preserving caves
and overhangs. Architecture retains its voxel geometry. Teyvat uses more detailed
tessellation and full prop models at the positions selected by its renderer.

Blender adds path-traced sunlight, indirect lighting, contact shadows, reflections,
water shading, smooth normals, antialiasing, and denoising. This improves the
rendering of the existing original assets. Detailed anime characters, authored
textures, animations, and larger environmental art sets remain separate art work;
changing the renderer alone cannot supply them.

Teyvat's procedural asset set adds seeded shape
variation, individual foliage, distinct task fixtures, and material groups for
stone, bark, foliage, metal, cloth, skin, and terrain. These groups survive scene
export; Blender assigns their different surface responses during rendering.

## Presentation landscape

Compact reference courses are small authored layouts, so the Cycles
presentation frames them with a landscape. The code is in
[`owb/render/dressing.py`](../owb/render/dressing.py). The simulated terrain,
collision, task entities, and action traces are unchanged; none of this
geometry collides or appears in policy RGB observations.

| Layer | What it adds | Where |
| --- | --- | --- |
| Horizon ridges | Two rings of ridged terrain beyond the walkable world. The near ring has gaps and a beach line; the far ring is taller with snow. | Teyvat (course and island), Wildlands, Skylands |
| Distance haze | Camera-distance blend toward the horizon colour, so far ridges read as layered silhouettes. Subjects within about 45–90 m are unaffected. | All Cycles scenes |
| Set pieces | Locations rather than a scatter. Each task site gets a role-specific vignette (ruined walls framing a relay court, a standing-stone cairn at a survey point, a stair-and-terrace ruin by a lift, a broken colonnade by a sentinel). Ambient clusters group trees, bushes, and grass around rock outcrops, with regional tree species. | Teyvat course, Wildlands course, Skylands full-scale surface |
| Ground detail | A world-space texture (path wear, soil, paving, exposed rock) applied to upward-facing terrain, with noise-broken edges. Relay courts get irregular flagstones; the route between sites is a worn path. | Same as set pieces |
| Meadow carpet | Short grass instanced by geometry nodes within 40 m of the camera, clumped by a world-space noise shared with the ground tone, and kept off paths, paving, and rock. | Teyvat course |
| Gate trim | Bases, banded shafts, capitals, and a cornice wrapped around the Wildlands courtyard gate voxels; the voxels remain the collision geometry. | Wildlands course |
| Occluder fading | Tall objects between the presentation camera and the character are faded for camera rays only; their shadows and bounce light are unchanged. | Teyvat presentation view, Wildlands |

Set pieces stay outside protected corridors: every straight route between task
entities (the same corridors architecture placement keeps clear), entity
positions, buildings, courtyards, and roads. The Teyvat corridor margin is
6.5 m because the reference controller strays up to about 5 m from straight
routes during combat and turns.
[`tests/test_presentation_landscape.py`](../tests/test_presentation_landscape.py)
checks that every position in the committed Teyvat and Wildlands reference
traces is more than 1 m from every set-piece footprint. The measured minimum
is 2.5 m.

Layouts are pure functions of the world identity and a layout version. They
use local hash noise and identity-seeded generators, never the simulation RNG,
so pieces keep their appearance as the camera moves. Snapshots record the
layout summary under `presentation_landscape` (`collision: false`,
`observation: false`) together with the ground-detail texture and its world
mapping.

The haze is an emission shader mixed into every material, so materials whose
only emission is haze use `emission_sampling = "NONE"`. Otherwise Cycles treats
every hazed triangle as a light: in the Skylands full-scale tasks that made the
~70 million baked grass triangles light-tree emitters, and some frames exceeded
128 GB of host memory. Rays still gather the haze, so the image is unchanged
(re-rendered frames differ by at most one 8-bit level); genuinely glowing
materials keep light sampling.

Shared look settings in [`blender_scene.py`](../owb/render/blender_scene.py):
AgX with Medium High Contrast at exposure 0 (previously +0.5, which washed out
the ground), sun strength × 4.0, sky strength 0.62. Teyvat terrain is graded to
a deeper, less saturated meadow with restrained dry patches. This removes most
of the strong green bounce in cast shadows. Blender also accepts
`--device METAL` (Apple GPUs) alongside `CPU`, `CUDA`, and `OPTIX`; an
unavailable device is still an error, never a silent fallback.

Weather comes from the simulated climate, recorded per snapshot as
`metadata["weather"]` (kind, lightning flash, simulated seconds, seed, wind).
Rain and thunderstorms dim the sun to 0.4× / 0.24× and the sky to 0.88× / 0.7×,
pull the distance haze in, and darken and gloss every surface material, with
sheen pooling on upward-facing ground. Snow and fog thin the light and thicken
the haze. Camera-facing streaks (rain) or small hexagonal flakes (snow) fill a
26 m radius around the camera, skipping the nearest 2–3.5 m. They are seen only by
camera rays and cast no shadows. Positions are hashed from world cells, the seed
and simulated time, never drawn from the simulation RNG, so a drop keeps its
identity as the camera moves and falls coherently between video frames. Clear
weather leaves every frame exactly as before. Both the per-frame renderer and
the persistent full-scale renderer apply it through `blender_scene.apply_weather`.

Matched before/after Cycles renders of the committed traces, with their
settings, are in [`assets/design/landscape-refresh`](../assets/design/landscape-refresh/):
[Teyvat A03 before](../assets/design/landscape-refresh/teyvat-a03-before.png) /
[after](../assets/design/landscape-refresh/teyvat-a03-after.png),
[Teyvat A10](../assets/design/landscape-refresh/teyvat-a10-compare.png),
[Wildlands A04](../assets/design/landscape-refresh/wildlands-a04-compare.png), and
[Skylands](../assets/design/landscape-refresh/skylands-a-step1500-compare.png).
The Skylands full-scale camera looks steeply down at the traveler, so most of its
change (ridges and set pieces) is visible only in wider views.

Not changed: the compact courses remain compact prototypes on the same
terrain. A shaped valley, occluding terrain, or colliding landmarks inside the
walkable area would change the task and need new ground-truth traces. Videos
and previews committed before this change are not re-rendered automatically.

## Render both environments

Install the repository and an [official Blender release](https://www.blender.org/download/).
The pipeline was tested with **Blender 4.5.4 LTS**, using CPU Cycles without a display
or GPU. Blender bundles its own Python and NumPy; OWB does not need to be installed
into Blender's Python environment.

```bash
pip install -e .
python -B -m owb.scripts.render_blender \
  --env teyvat wildlands --seed 7 --view showcase \
  --size 960 540 --samples 48 --threads 8 \
  --blender /path/to/blender --output /tmp/owb-blender-demo
```

Use a new output directory for each run. If `blender` is on your PATH, omit
`--blender`. You can also set `OWB_BLENDER` or save a machine-local executable
path in the git-ignored `.owb-render.local.json` at the repository root:

```json
{"blender": "/path/to/blender", "ffmpeg": "/path/to/ffmpeg"}
```

`ffmpeg` is optional when it is on PATH or supplied by `imageio-ffmpeg` from the
video extra. Each environment exported by `render_blender` gets:

| File | Purpose |
| --- | --- |
| `scene.npz` | Numeric scene snapshot plus a JSON manifest; no pickle or external assets. |
| `render.png` | Blender Cycles still image. |
| `scene.blend` | Editable Blender scene, including lighting and procedural water. |
| `scene.glb` | Portable meshes, vertex colors, materials, camera, and directional light. |

The default `--view agent` preserves the camera and state after reset.
`--view showcase` selects fixed presentation viewpoints, including Wildlands'
survey tower and daytime lighting. `--variant small --size 320 180 --samples 16`
reduces the cost of a quick preview. Full Wildlands exports can occupy hundreds
of megabytes because the mesh preserves the complete terrain and its caves.
Allow several minutes and a few GB of memory for a full scene. Use a local
scratch directory for these generated scene files. If a job's time limit cuts
off rendering after `scene.blend` has been saved, open that file in Blender to
render without rebuilding the mesh.

## Capture a state from the shared agent pipeline

```python
from owb import make_env
from owb.bench import neutral_action
from owb.render.scene_export import export_scene

with make_env("teyvat", seed=7) as env:  # also accepts "wildlands"
    obs, info = env.reset(seed=7)
    obs, reward, terminated, truncated, info = env.step(neutral_action())
    export_scene(env, "/tmp/teyvat-step-1.npz", width=960, height=540)
```

Exporting reads the current state without advancing simulation or consuming its
random generator. Captured arrays are detached from the live world. Native
Teyvat and Wildlands environments and Gymnasium wrappers are also accepted.

On a machine without Blender, use `--export-only` to write the NPZ files. Then
render a snapshot on another machine with the standalone script from this repo:

```bash
blender --background --factory-startup --python-exit-code 1 \
  --python owb/render/blender_scene.py -- \
  --scene /tmp/teyvat-step-1.npz --output /tmp/teyvat-step-1.png \
  --blend /tmp/teyvat-step-1.blend --glb /tmp/teyvat-step-1.glb \
  --samples 48 --threads 8
```

Snapshots are camera-specific presentation artifacts, not savegames. The prop
selection still uses the native renderer's visibility and population limits.
The mesh includes more terrain than an agent can see, so it should not be supplied
to a policy that is being evaluated on RGB-only observations. Presentation frames
are separate from policy RGB observations and never advance the simulation.
Skylands export is explicitly rejected until its construction objects and layered
lighting have a dedicated adapter.

## Record Cycles video

The shared episode recorder defaults to Cycles for Teyvat and Wildlands:

```bash
python -B -m owb.scripts.benchmark --env teyvat --variant full \
  --seed 7 --steps 20 --video mp4 --output /tmp/teyvat-cycles-recording
```

The default presentation size is 960×540 with 48 samples per pixel. Set
`--video-size WIDTH HEIGHT`, `--samples`, `--threads`, and `--blender` as needed.
`--size` and `--quality` continue to configure policy observations, independently
of the recording. Video frames are captured from each actual simulator state;
camera, positions, collisions, actions, and rewards are not altered by rendering.
This is offline rendering and can take substantially longer than simulation.
MP4 output uses H.264, `yuv420p`, and fast-start metadata for viewer compatibility.
The recording's `config.json` includes the presentation backend and settings.

Cycles presentation supports an explicit GPU device through
`OWB_CYCLES_DEVICE=CUDA`, `OWB_CYCLES_DEVICE=OPTIX`, or `OWB_CYCLES_DEVICE=METAL`. The default hardware device
is `CPU`; all three use the same full Cycles scene, materials, and sample count.
The selected device is recorded in presentation metadata. An unavailable GPU
raises an error instead of switching devices. The standalone Blender script
accepts the equivalent `--device CPU|CUDA|OPTIX|METAL` option.

The reference task recorder (`owb.scripts.teyvat_reference`), screenshot script,
and native Teyvat `render_hd()` also default to Cycles. Reference frames are
rendered at 864×486 inside a 1280×720 annotated layout, without upscaling a
low-resolution observation. Reference summaries and render caches record the
backend and settings, and assembly rejects a mixture of render settings. Legacy
CPU caches cannot be assembled as Cycles recordings.

For an explicitly requested CPU diagnostic, use `--video-renderer cpu` on the
benchmark runner, `--renderer cpu` on the reference or screenshot scripts, or
`render_hd(renderer="cpu")`. The Wildlands terminal script defaults to Cycles;
its existing `--renderer landscape` and `--renderer voxel` select CPU output.
Other environments retain their native recording path because their scenes do
not yet have a Cycles exporter.

The compact reference task geometry remains compact. A renderer change does
not replace that course with the full island, and older videos are not
automatically re-rendered. See the [reference demo scope](teyvat-reference-demos.md).

## Use the assets in Godot

Copy a generated `scene.glb` into a Godot 4 project and open its imported scene.
Godot recommends glTF 2.0 and supports both `.glb` and `.gltf`; it can also import
`.blend` files by invoking Blender. See the
[official Godot import guide](https://docs.godotengine.org/en/stable/tutorials/assets_pipeline/importing_3d_scenes/available_formats.html).

The GLB contains Y-up geometry, normals, vertex colors, PBR material parameters,
the captured camera, and a directional light. Add a `WorldEnvironment` in Godot
for sky, ambient lighting, fog, and tonemapping. Blender's world shader,
procedural water bump, and Cycles lighting result are not baked into the GLB,
so the two engines will not produce identical images. GLB structure and vertex
color export have been checked; interactive Godot rendering has not been tested.

A live Godot backend would next need a state-update and RGB-capture bridge to
the existing Python simulator, plus synchronized camera and entity identifiers.
The same shared action and observation contract can then remain in place for
both environments. Moving physics into Godot would be a separate change with
different determinism and benchmark implications.
