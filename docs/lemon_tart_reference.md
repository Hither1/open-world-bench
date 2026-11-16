# Lemon tart reference trajectory

The video [lemon-tart-reference.mp4](../videos/lemon-tart-reference.mp4) is an **older version** of the task. It finishes the previous nine milestones: containers moved into the oven or onto the stove, a baking sheet used as a second bake, and the lemon placed on the tart pan. The current procedure is [mobile-manipulation-tasks.md](mobile-manipulation-tasks.md). A robot trajectory for that procedure is not recorded yet.

The controller below is still the one that produced the old video. It emits only the PandaOmron's own 12-dimensional actions. It does not teleport the base, write object or door joints, or set the knobs directly.

## What the old run did

Script: `owb/scripts/lemon_tart_reference.py`. Seed 1, one continuous run from `reset`.

```bash
python -m owb.scripts.lemon_tart_reference --seed 1 --horizon 24000 \
    --actions videos/lemon-tart-reference-actions.npz
```

`videos/lemon-tart-reference.json`: all nine old milestones, `success` true, 22402 control steps, about 18 minutes 40 seconds of simulated time. Three storage opens during restore, zero wrong opens. Four repeats of the same command latched every milestone on the same step.

Replay uses a new environment and `env.step` only. The action log has to stay float64. A float32 copy reaches zero milestones.

```bash
python -m owb.scripts.lemon_tart_reference \
    --replay videos/lemon-tart-reference-actions.npz --horizon 24000
```

| Old milestone | Step |
| --- | --- |
| `ingredients_out` | 5329 |
| `crust_baked` | 8660 |
| `knobs_zeroed` | 9628 |
| `curd_started` | 10136 |
| `crumble_baked` | 12474 |
| `assembled` | 12899 |
| `ingredients_restored` | 17868 |
| `tools_washed` and `served` | 22402 |

The default horizon of 20000 cuts this run off at 7/9, so the recording uses `--horizon 24000`. Two attempts to shorten it broke a grasp and were not kept.

The video is H.264, one frame every 4 control steps, played at 20 fps, so it is 4x. The camera follows the robot. MuJoCo's offscreen renderer is used because RoboCasa has no Cycles exporter. On-screen captions in that file are Chinese. They describe the old nine milestones.

`videos/lemon-tart-state-walkthrough.mp4` is not this trajectory. It writes poses directly so each old milestone can be seen. Its captions are also Chinese.

## Why the kitchen was changed

These edits are in `owb/tasks/lemon_tart_env.py`. They stay relevant for the new procedure.

- `LemonTart.evaluator_state()` returns homes and the restore-open counts. The policy info does not.
- Only reachable drawers are homes. The PandaOmron torso only moves up, and the gripper bottoms out near 0.29 m. A drawer whose handle is below 0.30 m is not a home. In layout 001 that drops the bottom drawer and leaves exactly four. Seed 1's homes therefore changed: flour in `stack_1_main_group_2`, sugar in `_3`, whisk in `_4`, bowl in `stack_4_main_group_2`, saucepan in the dishwasher.
- Dishwasher objects sit on the upper rack, `rack1`. The lower rack is at 0.22 m behind a door that opens only about 35 degrees.
- Objects in the dishwasher no longer overlap.
- Flat dishes are propped on the rack's front wall. The rack has no solid floor, so a flat plate lying on the tines cannot be picked up again.

## What the old controller actually had to do

The action is `HYBRID_MOBILE_BASE` from `robosuite/controllers/config/robots/default_pandaomron.json`. Build it with `env.robots[0].create_action_vector(action_dict)`. Do not guess the indices. After the first `reset`, `env.action_dim` is 12. The last dimension is `base_mode` and is not inside the part slices.

| Key | Meaning | How to fill one step |
| --- | --- | --- |
| `right` | Right-arm OSC pose, 6 numbers, a delta in the base frame | Position delta clamped to ±5 cm, orientation delta to ±0.5 rad, then divided by that limit into `[-1, 1]`. Zero holds the current goal |
| `right_gripper` | Gripper, 1 number | `-1` opens, `+1` closes. It is incremental: 0.2 per step, so a full open or close takes many steps |
| `torso` | Torso joint position, a delta | Zero holds the current height |
| `base` | Base joint velocity | Forward, sideways, yaw. Zero stops. Commands below about 0.2 are inside a dead zone |
| `base_mode` | Last dimension | `+1` while the base moves, so the arm goal tracks the base. The old script sends `+1` even for arm-only motion and reads the controller's own goal before sending the next delta |

Control is 20 Hz. A saturated arm step is about 1 m/s before `ramp_ratio=0.2`, which makes it slower. A world target has to be converted into the base frame before it becomes a `right` delta.

Facts the first plan got wrong, and the old run had to follow:

- Drawer and dishwasher handles sit under the counter lip. An open gripper spans about 20 cm, so a pinch from above hits the counter. Drawers are hooked from behind with closed fingertips and pulled by driving the base backward.
- Pulling a drawer, pushing the rack, and closing the dishwasher require the arm to hold its pose while the base moves.
- The base controller's sign for yaw since reset is inverted. The script pre-rotates the command by twice that yaw.
- OSC has no orientation constraint. At full extension the elbow can lock against the joint limit. `unlock_elbow` reaches out and comes back. Knob and tray stances stay close to the oven.
- A horizontal reach only works when the target is at least 0.6 m above the arm base. Lower targets need the wrist angled down past 60 degrees.
- The oven tray has no handle. It is dragged out by the fingertips and pushed back by its front edge.
- The time knob decays on its own, so it is turned last.
- "Hand away" in the tracker is a gripper more than 0.25 m from the object.
- The baking sheet is about 7 by 11 cm. Lying flat on the rack tines it falls below the rack region and does not count as inside the dishwasher. The old run stood it inside the saucepan. The current spec does not use a baking sheet.
- The bowl and whisk were not used while making the crust. The old scorer did not require that. The current spec does.

## Known limits of the old run

- Only seed 1. Other seeds change object orientation.
- The first egg grasp misses once and the retry succeeds.
- The bowl leans in the rack.
- At the end the oven door is still open, the tray is out, and the knobs are left at the second bake. The old scorer ignores that.

## Separate clips

These are robot actions, not the full task, and they are not scored by the nine milestones or by the 21-milestone spec.

- `videos/lemon-tart-mix.mp4`: bowl, flour, and sugar out of drawers. A sugar cube is placed in the bowl. The flour bag is one rigid body, tipped to 117 degrees over the bowl, and nothing comes out. Seed 1, 4857 steps.
- `videos/lemon-tart-mix-usable.mp4`, plus the shorter flour, sugar, and egg clips: with `usable_ingredients=True`, those ingredients pour as grains. The same usable run cuts the butter and the lemon.
- `videos/lemon-tart-oven-lock.mp4` and `videos/lemon-tart-oven-knobs.mp4`: the optional lock and knob order. The door locks again when it closes. A knob turns up only after the earlier knobs in that episode's order are on.

On-screen captions in all of these files are Chinese. The task write-up is the English document linked above.
