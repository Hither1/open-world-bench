# Wildlands exploration and survival

Wildlands connects the voxel world, character controller, climate, creatures,
inventory, cooking, and NumPy renderer into a Gymnasium environment. All content
is original. It runs without a GPU or display server.

For the shared Genshin/Zelda agent interface and recorder, use `owb.make_env()`;
see [the common framework guide](framework.md). The APIs below are native Wildlands.

## Run

From the repository root, after `pip install -e ".[dev]"`:

```bash
python -m owb.scripts.wildlands --seed 7 --steps 100 --output /tmp/wildlands.ppm
python -m owb.scripts.wildlands --interactive --steps 3000
python -m pytest -q tests/test_wildlands.py
```

The demo gathers a pepper, cooks it, eats the dish, and explores. The PPM image
can be opened with an image viewer. Interactive mode takes named actions such as
`forward 5`, `turn_right`, `climb`, `glide`, `interact`, `next_slot`, `cook`, and
`eat`; `inventory` lists slots and `quit` exits. Each command advances simulation
time and refreshes the image. This is turn-based terminal control, not a real-time
window. Add `--gif /tmp/wildlands.gif` with the `video` extra to record frames.

On a quota-constrained cluster, use `PYTHONDONTWRITEBYTECODE=1` and
`python -m pytest -p no:cacheprovider` to avoid creating Python/test caches.

## Agent interface

```python
import gymnasium as gym
import numpy as np
import owb  # registers both Wildlands environments
from owb.env import COMMANDS

env = gym.make("OpenWorldBench/Wildlands-Small-v0")
obs, info = env.reset(seed=7)
action = {
    "move": np.array([0.0, 0.0], dtype=np.float32),
    "look": np.array([0.0, 0.0], dtype=np.float32),
    "buttons": np.zeros(5, dtype=np.int8),
    "command": COMMANDS.index("interact"),
    "slot": 0,
    "ingredients": np.zeros(5, dtype=np.int64),
}
obs, reward, terminated, truncated, info = env.step(action)
frame = env.render()
env.close()
```

`OpenWorldBench/Wildlands-v0` uses a 256 x 80 x 256 world and 128 x 128 images.
`OpenWorldBench/Wildlands-Small-v0` uses a 96 x 64 x 96 world and 96 x 72 images.
Pass an `owb.EnvConfig` as `cfg` to customize world, simulation, and rendering.
Set `obs_rgb=False` for faster nonvisual rollouts; `render()` still works.

The default Dict action has these fields:

| Field | Meaning |
| --- | --- |
| `move` | `[forward, strafe]`, each in [-1, 1], relative to agent yaw |
| `look` | `[yaw rate, pitch rate]`, each in [-1, 1]; positive yaw turns right |
| `buttons` | jump, sprint, climb, glide, crouch/dive, in that order |
| `command` | index into `COMMANDS`: none, interact, eat, cook, equip, attack, stow |
| `slot` | zero-based inventory stack to eat/equip; also the selected slot |
| `ingredients` | five entries: 0 means unused; 1..32 selects inventory slot 0..31 |

Cooking can repeat a slot to use multiple units, up to five total ingredients.
It requires a reachable cooking pot. Failed cooking leaves the inventory intact.
Interaction gathers the nearest reachable resource or chest; otherwise it lights
a nearby campfire. Locked trial chests remain locked. `stow` unequips metal gear.
One command executes per decision, while movement runs for `frame_skip` physics
ticks (default: two 0.1-second ticks). Jump presses execute once within that repeat.
Invalid actions raise `ValueError` without advancing the simulation.

For a simpler policy use `action_space="discrete"` and index `DISCRETE_ACTIONS`.
`next_slot`/`prev_slot` select inventory stacks; `cook` uses one selected ingredient.
Slots shift when a stack is consumed; read the next observation before selecting.

## Observations and episode semantics

| Key | Contents |
| --- | --- |
| `rgb` | optional uint8 H x W x 3 view |
| `depth`, `semantic` | optional float32 distance in metres / uint8 class IDs |
| `proprio` | 25 body values; names in `owb.entities.agent.PROPRIO_NAMES` |
| `survival` | time, day, weather index, wind x/z, ambient temperature, elapsed seconds, selected slot |
| `inventory` | 32 stack rows, columns in `owb.env.INVENTORY_COLUMNS`; item ID is one-based in `ITEM_IDS`, 0 is empty |
| `minimap` | north-up, agent-centred visited-cell map: 0 unknown, biome ID + 1 visited, 255 agent |
| `text` | optional local weather, vitals and inventory description |

Inventory rows include cooked-dish effects and duration, equipment slots and
remaining durability. `info` reports action failures, damage, discoveries,
inventory descriptions, active effects with remaining duration, and visit counts.
The minimap records visited cells only, persists during the episode, and resets
with the episode. It does not reveal unseen resources or landmarks.

Death returns `terminated=True`; reaching `max_steps` returns `truncated=True`.
Call `reset()` after either. This is an exploration sandbox with no quest victory
condition. Milestone mode rewards each new landmark visit once; dense mode adds
`exploration_bonus` for each new map cell. Sparse mode has no discovery reward.
All modes apply configured `time_penalty` and `death_penalty` (both default to 0).
The initial camp visit is recorded on reset and does not pay reward.

Explicitly resetting with the same seed reproduces terrain, entities, weather RNG,
and action outcomes. A reset without a seed replays the last seed. `cfg.seed`
(or the explicit reset seed) determines the episode world seed. Each environment
owns its RNG and creature/body ID counters. Returned arrays and info are copies.

## Current gameplay and limits

Walking, jumping, sprinting, climbing (with rain penalties), gliding, swimming,
drowning, fall damage, day/night weather, temperature exposure, campfire warmth,
food effects, gathering, chest looting, melee combat, and weapon wear are connected.
Every default reset provides a small clearing with a pot, unlit campfire, nearby
ingredients, a glider, wooden sword, and apples. Set `starter_camp=False` to use
the untouched procedural spawn. Dishes grant temporary protection; ordinary armor
defense reduces physical hits, while thermal resistance handles cold and heat.

This iteration does not implement quest/puzzle completion, crafting actions, rune
controls, ranged combat, NPC trading/dialogue, hunger, automatic respawn, or saved
cross-session checkpoints. Enabling hunger/respawn raises an explicit error.
## Visual style

`RenderConfig(style="landscape")` is the default. It renders a continuous natural
surface, branching trees with layered crowns and individual folded leaves,
wind-bent grass, modeled camp props, and an articulated traveler with cloth,
boots, hair, scarf, satchel, sword and glider. World-space material variation adds
soil patches, moss colors and weathered rock. Sunlight uses terrain and mesh
shadows, with local ambient occlusion at contact points. Water has animated
ripples, sun highlights and depth-dependent color. The HUD uses hearts and a
stamina ring.

`RenderConfig(quality="balanced")` is the default presentation tier. `"fast"`
reduces foliage and omits material detail, mesh shadows and contact shading;
`"high"` adds more leaves and grass plus a finer shadow map. Foliage detail also
scales with camera distance and output resolution. These tiers affect rendered
vegetation geometry, so visual depth/semantic images can differ between tiers.
They do not alter collision geometry, actions, or the simulation random stream.
Toggle `shadows` or `ambient_occlusion` independently; those only affect RGB.
`supersample=2` smooths fine foliage edges at additional CPU and memory cost.

The terminal demo defaults to a 320 x 180 third-person view. Agent observation
dimensions and the default first-person agent camera retain their existing
settings. Choose `--first-person`, `--width`, `--height`, or `--renderer voxel`
for the demo; configure `RenderConfig` directly for policies.

```bash
python -B -m owb.scripts.wildlands --interactive --steps 3000
python -B -m owb.scripts.wildlands --interactive --quality high --supersample 2 --width 640 --height 360
python -B -m owb.scripts.screenshots --environment wildlands --compare-wildlands
python -B -m owb.scripts.screenshots --environment wildlands --wildlands-view woodland --quality high --size 640 360
```

The two screenshot backends use identical seed, camera, time of day, and HUD.
Screenshot captures default to high quality and 2x supersampling. `--wildlands-view`
selects the survey-tower overlook, starter camp, or a ground-level woodland view;
these are fixed viewpoints in the generated seed-7 world.
These images are actual engine output. No generated artwork is composited over
observations. Both backends return RGB, ray-distance depth, and semantic buffers.
Landscape supersampling averages RGB subpixels; depth/semantics use the same centre subpixel.
Rendering does not consume the simulation RNG.

Physics still uses the existing voxel grid. Natural surfaces are rounded within
roughly one voxel of that grid; foliage silhouettes are approximate meshes.
Buildings retain their block geometry. Caves and terrain edits are rendered from
current occupancy rather than a top-down heightmap, and caches refresh on world
edits and resets. `style="voxel"` remains available for exact terrain debugging.

The assets are still procedural. This pass improves the native renderer's
materials, vegetation, lighting and silhouettes; it does not provide the detailed
authored characters and environment assets of a finished game. CPU rendering is
not real-time at presentation resolutions. Use `quality="fast"`, small agent
frames, or disable RGB for faster rollouts. The optional
[Blender presentation pipeline](rendering.md) is separate from live observations.
