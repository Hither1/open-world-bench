# Teyvat asset diversity refresh

Cycles now renders a richer procedural asset set. The earlier repetition came
from shared cone/sphere models and one generic surface material; changing the
renderer alone could not add the missing shapes or surface detail.

## Same scene and camera

| Earlier assets | Updated assets |
| --- | --- |
| [Full-world preview](../assets/design/teyvat-environment-refresh/world-before.png) | [Updated full-world preview](../assets/screenshots/teyvat-diverse-cycles.png) |
| [Relay replay preview](../assets/design/teyvat-environment-refresh/relay-before.png) | [Updated relay replay preview](../assets/design/teyvat-environment-refresh/relay-after.png) |

The full-world pair uses seed 7, the same action, avatar position, camera, and
960×540 Cycles settings. The relay pair uses action 138 from the existing Task A
trace and the same annotated layout. Neither comparison changes the terrain
or moves objects to improve the composition.
The [world-state metadata](../assets/design/teyvat-environment-refresh/world-state.json)
and [render source hashes](../assets/design/teyvat-environment-refresh/render-source-hashes.json)
record the state and implementation used for the refreshed views.

## What changed

- Trees have branching trunks, root flares, asymmetric crowns, individual folded
  leaves, and stable variation in height, spread, lean, and color. Broadleaf
  trees have four crown families; pines use needle sprays, and bamboo has
  segmented stems and narrow leaves.
- Rocks have irregular fractured surfaces and moss-colored upper faces.
  Grass uses curved narrow blades; shrubs have woody stems and separate leaves.
- Waypoints have stone supports and metal/crystal assemblies. Chests have
  timber planks, straps, clasps, and rivets. NPCs vary in clothing, hair,
  accessories, skin tone, and stature.
- Relay parts are one architectural family. Each stands on a stepped square
  foundation with corner kerbs. Sources are banded, gently tapering steles whose
  capital holds a cut crystal in a bronze cradle; drains are sunken basins with a
  bronze grate; vanes are bronze masts stepped into a stone collar; cores are
  squat drums holding an armillary gimbal around a floating crystal. Source
  bands visibly retain the single-/double-ring distinction required by the task
  rules. Vent shelves use a separate ground-level design. (The earlier source
  pillar, a tapered shaft under a large glowing cone, read as a giant pencil.)
- Cycles uses separate stone, bark, foliage, metal, cloth, skin, and terrain
  materials with different roughness, fine surface normals, and color variation.
  Leaves transmit some light; terrain includes world-anchored earth patches.

These are original procedural meshes and materials. No external model or texture
downloads are required. The style is more natural and detailed, but characters
and buildings are still procedural; this is not a photorealistic asset library.
The compact reference course also remains a compact test arena.

## Stability and verification

Prop appearance uses a stable hash of its kind, position, and orientation.
Entity appearance uses world seed and persistent entity ID, so moving actors
retain their appearance. Local random generators do not advance simulation RNG.
Cached templates are transformed without mutation.

Both observation and presentation renderers use the new asset families. Cycles
uses denser foliage and tessellation. Presentation remains the default for
user-facing recordings; see the [rendering guide](rendering.md).

Interaction locations, collision volumes, task state, and controls are
unchanged. Both saved reference trajectories replayed successfully after the
update, comparing every logged position/event and all milestone hashes:

- [Task A: 1,118 actions, 12 milestones](../assets/design/teyvat-environment-refresh/task-a-replay-verification.json)
- [Task B: 3,038 actions, 24 milestones](../assets/design/teyvat-environment-refresh/task-b-replay-verification.json)

Targeted tests cover stable variation, distinct role/version cues, template
immutability, material export, scene isolation, task mechanisms, and recording.
RGB observations intentionally look different, so this is a visual environment
revision; previous learned-agent scores should not be assumed to transfer.
The [regenerated Cycles walkthroughs](teyvat-reference-demos.md) now show the
updated assets. The original CPU recordings remain archived separately.

Blender-specific procedural material nodes remain in Cycles/.blend output.
The GLB handoff retains geometry, vertex colors, and portable material properties;
matching all procedural surface detail in another engine requires baking or
recreating those shaders.

## Task architecture

Teyvat now places six original building families around task locations:

- Timber lodges with plaster panels, exposed framing, shutters, tiled gables,
  chimneys, and planked doors.
- Stone windmills with timber upper stories and framed canvas sails.
- Open courtyard pavilions with curved jade roofs, raised corners, brackets,
  and hanging lanterns.
- Market stalls with striped canvas, counters, timber posts, and pottery.
- Ruined gateways with masonry piers, carved lintels, and a sun-arch crown.
- Crenellated watchtowers with staggered masonry and narrow shuttered windows.

Reference courses use task roles to choose nearby architecture: lessons receive
lodges, exchanges receive stalls, relay sites receive pavilions, wind crossings
receive windmills, and combat sites receive ruins. Additional survey/cache sites
mix the families. Seed 7 places 17 buildings in Task A and 42 in Task B. The
regular island also places architecture near its quest NPCs and landmarks when
terrain and existing props leave suitable space (nine buildings for seed 7).
Stone foundations extend to the sampled ground on gentle slopes; they do not
raise, lower, or smooth the heightfield.

Placement happens at reset and is independent of both simulation and course RNG.
Persistent site IDs and world seed determine appearance. The planner checks
footprint heights, water, solid vegetation, nearby buildings, interaction zones,
and travel segments before accepting a site. It does not flatten terrain, move
entities, relocate objectives, or choose buildings based on the camera.

Structural boxes are shared by visual meshes, player collision, and camera
occlusion. Pavilions and gateways have open passages; houses and towers have
solid exteriors. Roofs, sails, and facade ornaments are decorative: this update
does not add enterable house interiors, rooftop standing, or building climbing.
Both the observation renderer and Cycles receive the architecture. Presentation
snapshots also record all building identities, transforms, and appearance seeds.
This changes RGB observations and adds obstacles away from the protected task
corridors; it is an environment revision, not a claim that prior model scores
transfer unchanged.

The saved seed-7 reference action traces are revalidated against the new collision
geometry. Evidence is stored separately from the existing walkthroughs:
[Task A verification](../assets/design/teyvat-architecture/task-a-seed-7/verification.json)
and [Task B verification](../assets/design/teyvat-architecture/task-b-seed-7/verification.json).
The older videos have not been regenerated by this architecture update.

To create four actual-course presentation views with GPU Cycles:

```bash
OPENBLAS_NUM_THREADS=1 python -m owb.scripts.teyvat_architecture \
  --output /tmp/teyvat-architecture --task b --seed 7 \
  --device CUDA --size 1280 720 --samples 64
```

The output contains camera-specific NPZ scenes, PNG renders, Blender logs, and a
manifest with renderer settings, source hashes, building identities, and image
hashes. Only presentation cameras change; task state and terrain remain exact.
`--export-only` writes scenes without rendering. The render command requires an
available CUDA or OptiX GPU and never falls back to CPU rendering.

## Presentation landscape

Cycles presentation renders of the compact reference course also add a
presentation-only landscape, described in the
[rendering guide](rendering.md#presentation-landscape): horizon ridges beyond
the plateau, set pieces grouped around each task site, worn paths, paved relay
courts, and meadow grass near the camera. None of it collides or appears in
policy observations, and the course terrain is unchanged.
