# Wildlands exploration and survival

Wildlands connects the voxel world, character controller, climate, creatures,
inventory, cooking, and NumPy renderer into a Gymnasium environment. All content
is original. It runs without a GPU or display server.

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
The renderer uses voxel terrain and simple entity impostors; this is a gameplay
and agent-interface milestone, not a finished art pass.
