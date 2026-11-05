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
- Relay sources are caged cells, drains are low basins with channels, vanes are
  wind-driven structures, and cores are ring-mounted consoles. Source inlets
  visibly retain the single-/double-ring distinction required by the task rules.
  Vent shelves use a separate ground-level design.
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
The older walkthrough videos still show the earlier assets.

Blender-specific procedural material nodes remain in Cycles/.blend output.
The GLB handoff retains geometry, vertex colors, and portable material properties;
matching all procedural surface detail in another engine requires baking or
recreating those shaders.
