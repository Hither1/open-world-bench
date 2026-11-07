# Shared environment and rollout framework

Use `owb.make_env("teyvat")` for the Genshin-inspired environment and
`owb.make_env("wildlands")` or `owb.make_env("skylands")` for the Zelda-inspired
environments. All expose the
same Gymnasium spaces, control conventions, capability masks, and episode
summaries. Adapters preserve the separate simulators and their native APIs.

`owb.make_env("skylands")` runs the three-layer construction course; see the
[Skylands guide](skylands.md). `owb.available_environments()` lists all three.
The common interface is now **schema version 2**: construction commands, `part`,
`build_move`, `construction`, and `build_state` have been added. Use
`neutral_action()` to initialize every field. Version 1 action dictionaries and
recordings need the new action fields filled with zero; native Wildlands and
Teyvat APIs retain their existing schemas.

## One pipeline

```python
import owb
from owb.bench import BenchmarkConfig, RandomPolicy, run_episode

config = BenchmarkConfig(seed=7, max_steps=100, image_size=(96, 72))
for name in owb.available_environments():
    with owb.make_env(name, config=config) as env:
        print(run_episode(env, RandomPolicy(), seed=7))
```

The common configuration sets seed, `small`/`full` variant, image size `(width,
height)`, decision limit, decision duration, and optional goal hints. Keyword
overrides work too: `owb.make_env("teyvat", max_steps=100)`.

Decisions advance 0.2 simulated seconds by default: six Teyvat physics ticks or
two Wildlands ticks. `decision_seconds` must be a positive integer multiple of
the native physics timestep. Death can end a decision early. Native movement
speeds, camera rates, and gameplay remain different.

Advanced settings use `native_config=owb.sims.teyvat.EnvConfig(...)` for Teyvat
or `native_config=owb.EnvConfig(...)` for Wildlands/Skylands. The factory copies it; common
seed, resolution, horizon, timing, and required interface modes take precedence.
The variant selects defaults only without a native configuration. `env.native`
exposes the simulator for debugging.

`render_quality="presentation"` enables 2x supersampling and stronger terrain/
shadow sampling for policy observations. `"fast"` keeps lower-cost observations;
neither setting changes physics, tasks, or the output tensor shape. Teyvat uses
original procedural adventurer meshes, layered foliage, softer cel lighting,
atmospheric color, and fine outlines. It remains a CPU research renderer rather
than a production game engine with authored animation and materials.
User-facing Teyvat/Wildlands recordings use Blender Cycles independently of
these sensor settings; see the [rendering guide](rendering.md).

## Controls

`neutral_action()` returns a new valid no-op dictionary. Controls are:

| Field | Meaning |
| --- | --- |
| `move` | float32 `[forward, right]` in `[-1, 1]`, relative to horizontal view direction. |
| `look` | float32 `[turn right, look up]` in `[-1, 1]`. |
| `buttons` | Six bits: jump, sprint, climb, glide, dive, crouch. |
| `command` | Index in `COMMANDS`: none, interact, attack, skill, burst, eat, cook, equip, stow, grab, place, rotate, ascend. |
| `swap` | 0 keeps the character; 1–4 selects a party slot. |
| `slot` | Inventory slot 0–31. |
| `ingredients` | Five entries: 0 unused, 1–32 selects inventory slot 0–31. |
| `part`, `build_move` | Part ID 0–8 and three floats for grid manipulation in Skylands; see its guide. |

`command_mask`, `button_mask`, and `swap_mask` describe backend support, not
current cooldown/proximity readiness. Unsupported but well-formed requests consume
a decision and emit `action_unavailable`; supported movement still executes.
Malformed actions fail before advancing. One command executes per decision.
Teyvat swaps before skill logic, and translates glide/dive requests to its jump
control only in the relevant movement states. Native APIs retain richer native
simultaneous-button combinations.

## Observations and batching

| Field | Common shape and semantics |
| --- | --- |
| `rgb` | uint8 `(height, width, 3)`. |
| `proprio` | `(12,)`: forward/right/up velocity in m/s, health/stamina fractions, grounded/airborne/swimming/climbing/gliding flags, time-of-day sine/cosine. |
| `instruction` | Up to 1024 printable characters: quest or local survival description. |
| `party` | `(30,)` Teyvat party state; see README for layout. |
| `inventory` | `(32, 16)` Wildlands stacks; `owb.sims.voxel.env.INVENTORY_COLUMNS` and `ITEM_IDS` define the vocabulary. |
| `survival` | `(8,)` Wildlands weather/time state; names in `owb.sims.voxel.env.SURVIVAL_NAMES`. |
| `minimap` | `(32, 32)` Wildlands visited-cell map. |
| `goal` | `(15,)` Teyvat native objective vector; disabled by default. |
| `construction`, `build_state` | `(8,9)` locally observable parts and `(6,)` held cursor/layer state in Skylands. |
| `modality_mask` | Seven bits: party, inventory, survival, minimap, goal, construction, build_state. Unsupported modalities are zero-filled. |
| Capability masks | Commands, buttons, and party slots supported by the backend. |

Constants `PROPRIO_NAMES`, `MODALITIES`, `BUTTONS`, and `COMMANDS` are exported by
`owb.bench`. `include_goal_hint=True` explicitly enables oracle objective direction.
Common proprioception excludes absolute world coordinates. Native diagnostics in
`info["native"]` may contain privileged state; the runner gives policies only
observations. This interface alone does not establish a fair memory benchmark.

```python
import gymnasium as gym
import owb

envs = gym.vector.SyncVectorEnv([
    lambda: owb.make_env("teyvat"),
    lambda: owb.make_env("wildlands"),
])
try:
    obs, info = envs.reset(seed=7)
    # obs["rgb"]: (2, 72, 96, 3)
    obs, reward, terminated, truncated, info = envs.step(envs.action_space.sample())
finally:
    envs.close()
```

Random space samples may include unsupported requests; useful policies should
honor masks. Returned arrays and info are detached from simulator state.

## Episodes and evaluation

Reset before stepping and after termination or truncation. Reusing a seed
reproduces trajectories and action-space samples; reset without a seed reuses the
last seed. Nonempty reset options are rejected.

Common info reports schema version, environment, seed, steps, simulated seconds,
decision duration, return, health, stamina, success, task progress, terminal flags,
termination reason, structured events, backend metrics, and native diagnostics.
Ended episodes include an `episode` summary. Success/death takes precedence over
a coincident time limit, so terminal flags are not both true.

Teyvat success means questline completion, and its default respawn is preserved.
Wildlands has no victory predicate, so success and task progress are `None`.
Skylands succeeds after building the crossing, reaching the far island, and
interacting with the beacon; its progress is 0, 0.5 (bridge), or 1 (success).
A time limit is truncation, not success. Metrics preserve quest counts versus
visited cells/landmarks. Rewards stay native; do not average them across backends
as if they represented equal task difficulty.

Use `success_defined` and `task_progress_defined` when aggregating vectorized
results. Some Gymnasium versions coerce `None` to `False` or `NaN` while batching;
these explicit validity flags preserve the distinction between an unsupported
metric and a failed task. They are also included in episode summaries.

Policies are callables receiving observations and may implement `reset(seed)` to
clear episode-local memory. The caller owns/closes the environment. `RandomPolicy`
is seeded and capability-aware. `DemoPolicy` demonstrates party skills,
gathering/cooking and traversal, or the fixed Skylands bridge course; it is not
a general task-solving baseline.

`OpenAIPolicy` in `owb.bench.baselines` uses RGB and named sensor observations to
choose structured actions with GPT-6-Astra. Run it with `--policy openai`; see the
[embodied agent guide](embodied-agent.md) for setup, request limits, and memory.
Policies may expose `metadata()` and `last_decision` for recorder diagnostics.
`ClaudePolicy` uses the same observation/action contract through a saved Claude
Code login; select `--policy claude --model claude-opus-5-5` and pass
`--claude-account-home` for the intended profile.
`GeminiPolicy` selects a saved Gemini CLI Google login; use `--policy gemini
--model gemini-4-argon`, `--gemini-account-home ~/.gemini`, and
`--gemini-account-email` to identify and verify the intended account.

## Recorded demos

Example artifacts: [shared pipeline recording](../assets/demos/shared-framework.gif)
and [Teyvat renderer update](../assets/demos/teyvat-style.png).

```bash
pip install -e ".[dev,video]"
PYTHONDONTWRITEBYTECODE=1 python -m owb.scripts.benchmark \
  --env teyvat wildlands --seed 7 --steps 60 --size 256 192 \
  --policy demo --video gif --output /tmp/owb-shared-demo
```

Teyvat and Wildlands videos default to Blender Cycles at 960×540 and 48 samples.
Configure Blender using `--blender`, `OWB_BLENDER`, or the local configuration
described in the [rendering guide](rendering.md). `--video-size` and `--samples`
configure the video. `--size` and `--quality` configure policy observations.
An explicit `--video-renderer cpu` selects diagnostic CPU output. There is no
automatic fallback when Blender fails. Other worlds retain their native video
renderer until they have a Cycles scene exporter.

Each `ENV-SEED/` contains `config.json` (schema, common/native configurations,
actual seed, presentation renderer/settings, and field vocabularies), `trajectory.jsonl` (reset and action/reward/
event records with proprioception/instruction), `summary.json`, and optional
`episode.gif` or `episode.mp4`. Videos render the reset and decision states at
the configured decision interval. Cycles frames are presentation output, not a
copy of the policy's RGB observation. Existing run directories are never overwritten.
Omit `--video` for JSON-only recording. Python callers can use
`run_episode(..., output_dir=path, video="gif")`.

Logs are action/event traces, not full observation datasets or simulator
checkpoints. Reconstruct omitted tensors with the same code/dependency versions,
configuration, seed, and actions. GIF is intended for short previews; use MP4 with
the video extra for longer recordings.

## Package layout

| Package | Contents |
| --- | --- |
| `owb.sims.teyvat` | Teyvat simulator: terrain, world, party, elements, character, camera, quests, sensor renderers (`render/`), `env.OpenWorldEnv`, reference course and privileged reference policy. |
| `owb.sims.voxel` | Voxel simulator behind Wildlands and Skylands: `core`, `world`, `physics`, `sim`, `entities`, `env` (`WildlandsEnv`, `SkylandsEnv`, `WildlandsReferenceEnv`) and the observation renderers in `render/` (`make_renderer`, raycast, landscape, shadows). |
| `owb.bench` | Common `BenchmarkEnv`, adapters, rollout and baselines. |
| `owb.tasks` | Evaluator-side task definitions. |
| `owb.render` | Presentation only: scene export, Blender Cycles, video encoding. |
| `owb.scripts` | `python -m owb.scripts.*` entry points. |

The old import paths still work but are deprecated: `open_world_bench` and its
submodules alias `owb.sims.teyvat`; `owb.core`, `owb.world`, `owb.physics`,
`owb.sim`, `owb.entities`, `owb.env` and the moved `owb.render.*` sensor modules
alias `owb.sims.voxel.*`. `owb/_compat.py` registers the canonical module objects
under the old names (so `open_world_bench.env.OpenWorldEnv is
owb.sims.teyvat.env.OpenWorldEnv`) and issues one `DeprecationWarning` per old
prefix. Gymnasium ids are unchanged; their entry points name the canonical modules.

## Adding another environment

Implement a native factory `(BenchmarkConfig, native_config)` and a `NativeAdapter`
subclass supplying control translation, physical state, observation projections,
capabilities, elapsed time, events, and task metrics. Register an `EnvironmentSpec`
using `register_environment()`. New backends can be implemented this
way without changing the recorder. Implemented names cannot be
silently replaced. New mechanics may require a new schema version rather than
reusing an existing action with different semantics.

Run contracts and native regressions with:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider
```
