# Teyvat full-scale tasks: design for Task A

Status: **phase 1 implemented** (world, solid-terrain physics, gate proofs); see
[Phase 1 results](#phase-1-results). A labelled [route preview](#route-preview)
walks the whole journey with native movement and lifts. Milestone mechanics,
NPCs, items and the reference solution are not implemented yet. Task lengths
and times are planning estimates, not measured rollouts.

This document extends the 12- and 24-milestone [legacy Teyvat designs](teyvat-legacy-tasks.md)
from their [compact reference courses](teyvat-reference-demos.md) to full regional scale.
It follows the path Skylands took from its compact testbed to its
[full-scale campaigns](skylands-campaign-demos.md). In that case,
`SkylandsCampaignEnv` subclasses the compact environment, keeps its milestones and
local mechanics, and replaces the world. The milestone IDs, count, and order stay the same.

Working-memory probe `WM-TA06` (A06) is specified in the
[working-memory probe contract](working-memory-probes.md) and implemented in the
compact course only. It is not implemented at full scale, because full-scale Task A
has no milestone mechanics yet.

Task A, *The Relay Keeper's Apprenticeship*, is specified here in full. Task B,
*The Expedition of Three Arts*, has its own design and seed-7 implementation in
[teyvat-full-scale-task-b.md](teyvat-full-scale-task-b.md) (see
[Task B outlook](#task-b-outlook)).

| | Compact course (today) | Full-scale Task A (this design) |
| --- | --- | --- |
| Milestones | 12, linear | 12, linear, same IDs and predicates |
| World | 640 m flat plateau, no vegetation, sites on a 46 m serpentine grid | 1,280 m authored island, seven regions, climbing, swimming, gliding |
| Reference length | 1,118 actions, 3.7 min, 1.09 km | **≈ 13,500 actions, ≈ 45 min, ≈ 7 km** (estimate) |
| Walking share | 93 % (1,042 of 1,118 actions) | ≈ 57 % (estimate) |
| Spatial memory | None tested | Two location probes (cache at A07, ruin at A10); world map as an evaluation condition |
| Next-step guidance | `Step i/n: ...` appended to the instruction | None. Diegetic leads from NPCs, plaques, and signposts |
| Ferry lift, gates | Item exchange and access receipts | Moving lift, visible seals with matching collision |

The existing suites are not changed. `teyvat-campaign-v1` (`summit`, `cleansing`,
`desert`) remains the short, branching suite. The compact course remains the
mechanics regression testbed.

## Why scaling alone is not enough

In the compact Task A trace (`assets/demos/teyvat-reference-cycles/task-a-seed-7`),
local mechanics take about 76 actions. A relay repair takes three or four casts, and
each lesson or exchange is a single interaction. If the course were only stretched to
regional scale, the task would become almost entirely walking. The legacy and
Skylands designs both rule this out: *"Avoid empty walking or forced waiting."*

The full-scale task therefore gains length from three sources:

1. **Traversal with decisions.** Cliffs bound what can be climbed on one stamina
   bar, marsh channels require swimming, glides need enough height, and gates
   restrict routes.
2. **Mechanisms at site scale.** Relay receivers are spread across a structure.
   Camps and ruins have real enemies. Lifts actually move.
3. **Designed returns.** The supply cache is learned early and revisited after
   intervening work. The summit cap lies in a lowland ruin that is described early
   and needed late. See [Spatial memory probes](#spatial-memory-probes).

## World layout

- **Extent:** geography scale 2.0, a 1,280 m square island, with land inside about 560 m of the center.
- **Heightfield:** resolution 513, giving 2.5 m cells. The finer grid is needed for
  readable cliffs and ledges. Heights are authored directly, so procedural
  terracing cannot add unintended rest ledges.
- **Relief:** lowland 1–17 m, plateau 58 m, ridge 104 m, summit dome to about 110 m,
  beacon pillar top 156 m. Snow line 125 m; tree line 85 m. Each tier is set so that
  its wall exceeds the proven climb ceiling from the tier below (see
  [Phase 1 results](#phase-1-results)).
- **Lighting:** fixed time of day in v1, so appearance stays consistent for perception.

```text
                                   N
  +---------------------------------------------------------------+
  |                     (7) Summit beacon  156 m         A12      |
  |                         | service lift (cap socket)  A11      |
  |   (5) Signal relay 2         Signal ridge                     |
  |        A09        .........  (5) Signal relay 1  A08          |
  |      glide down            [corridor gate]                    |
  |      the west edge      (4) Highland outpost  A07             |
  |         \                  Upper ferry landing                |
  |==========\======== escarpment, ~48 m sheer ==================|
  |~~~~~~~~~~~\~~~~~~~~~~~~~~~~~ river ~~~~~~~~~~~~~~~~~~~~~~~~~~~|
  |  (6) Old ruin A10   (3) Lower ferry  A06     (2) Raider camp  |
  |      western gully                               A05          |
  |  (1) Windrest village        Reedmarsh junctions A04          |
  |      training yard A01-A03            ~~ (2b) Cache islet ~~  |
  +---------------------------------------------------------------+
                                   S
```

| # | Region | Approx. center (x, z) m | Elevation | Terrain and structures |
| --- | --- | --- | --- | --- |
| 1 | Windrest village | (-400, 430) | 8–15 m | Meadow; timber houses and windmills; training yard with three relays; keeper; respawn statue. |
| 2 | Reedmarsh | (100, 420) | 0–3 m | Shallow channels with swims of at most 40 m and reed islets; three reed-post junctions. |
| 2b | Cache islet | (310, 470) | 2 m | Broken windmill landmark, unique silhouette; small hooded signal lamp, unlit at start and visible only within about 40 m; sealed cache. |
| 2c | Raider camp | (390, 180) | 10–17 m | Eastern hills; palisade and watchtower; 3–4 ordinary raiders with native damage. |
| 3 | Lower ferry | (60, 155) | 8 m | River's south bank; mechanic; cable-lift tower and cabin; respawn statue. |
| 4 | Highland outpost | (-90, -60) | 58 m | Plateau above the escarpment; upper landing at (58, 93); quartermaster; respawn statue. |
| 5 | Signal ridge | (-40, -270) | 104 m | Upland behind a ~46 m wall, reached only up a 130 m ravine ramp (22°) closed by a sealed gate; two field relays, stone-mast family. |
| 6 | Old ruin | (-470, 205) | ~8 m | Lowland western gully below the escarpment, north of the village. Its broken arch tower is visible from the training yard. Relay-powered vault seal; 2–3 ruin guards. |
| 7 | Summit | (0, -455) | 104–156 m | Dome rising to about 110 m; final 46 m sheer pillar; service lift; summit beacon. |

Each seed jitters region centers by up to ±18 m, as campaign-v1 does. Local
mechanism geometry keeps fixed dimensions.

### Traversal physics

Cliffs are not barriers under the shared Teyvat controller
(`open_world_bench/character.py`). Measured on test walls:

- Walking into any slope from 53° to vertical climbs it with no stamina; a
  vertical 80 m wall takes 1.1 s. The heightfield acts only as a floor, and the
  slide on over-steep ground pushes along `-normal`, which points uphill.
- A climber who runs out of stamina lands on the face, recovers stamina there,
  and keeps climbing.
- Updraft can be cast while hanging. Hanging idle while it cools down costs about
  28 stamina; each cast refunds 25 and lifts 3.5 m, so any wall can be scaled.

These affect every Teyvat task. The shared controller is left unchanged so that
existing tasks and their evidence keep their behavior. The full-scale task uses
`FullScaleController` (`open_world_bench/fullscale_physics.py`) instead:

| Rule | Effect |
| --- | --- |
| Solid terrain | A move is rejected if the ground ahead rises more than the step height (0.55 m), or rises at all onto ground steeper than 50°. |
| No standing on steep ground | Slides go downhill; no jumping and no stamina recovery there. |
| Stamina-limited climbing | 9 stamina/s in every direction; the hold lasts until the face tops out, then the climber pulls up at most 1 m onto walkable ground. |
| Ground-only launch skills | Updraft and other launch skills fire only from walkable ground. |
| Surface-consistent slopes | Slopes come from the gradient of the same bilinear surface as the ground height, so the foot of a cliff is standable and its face is climbable. |

Under these rules one climb gains at most 24.6 m: a 21.1 m stamina bound plus a
3.5 m launch start. Proofs use 30.8 m (×1.25). The strongest real climb measured
on any gate wall gained 23.0 m.

| Ability | Native parameters | Consequence for level design |
| --- | --- | --- |
| Climb | See above | Gate walls are 46 m or taller and stand above the proven climb ceiling of the tier below by at least 10 m. |
| Fall | Damage above 22 m/s impact | Falls from about 10 m start to hurt. |
| Glide | 8.5 m/s forward, 3.2 m/s descent; 5.5 stamina/s | About 57 m of drop and 150 m of range per bar. The plateau-to-lowland glide is a legal one-way route; the real controller lands 37 m from the ruin. |
| Swim | 2.7 m/s; stamina recovers while floating | Swimming distance is effectively unlimited. Shores on required routes are gentle. |
| Run / sprint | 5.4 / 8.2 m/s | The estimates below assume 4.5 m/s effective average speed. |

### Gates and locks

Linear order is enforced by the world, not by a hidden step counter visible to
the agent. Region gates are physical: cliffs, lifts, and a gate with collision
that matches its visual geometry, using `GateCollisions` from campaign-v1. Item
locks (the cache and the vault) are interaction locks: the container refuses to
open until its fact holds. They are implemented with the task logic in phase 2.

| Gate or lock | Visible cue | Opens when | Designed probe it protects |
| --- | --- | --- | --- |
| Escarpment | Continuous ~48 m cliff rising straight out of the river | Ferry lift operational (A06) | Lowland and highland work stay in order. |
| Cache seal | Hooded signal lamp on the cache windmill, visible only within about 40 m | Upper-landing test lever pulled (A06) | Supplies cannot be taken at A04, so A07 requires a return to a remembered place. The lamp cannot act as a distant beacon. |
| Corridor gate | Sealed gate filling the 12 m ravine mouth from floor to rim (50 m tall), set 7 m into both walls | Supplies delivered (A07) | The first relay-transfer probe (A08) comes after the intervening work. |
| Ruin vault seal | Dark seal glyph on the vault door, readable within about 10 m | Both field relays active (A09) | The cap cannot be taken early, so A10 requires a return to a place described at A04. |
| Summit pillar | 46 m sheer pillar; lift socket missing its cap | Cap installed (A11) | Summit work stays last. |

**Bypass validation is required and implemented.** `open_world_bench/fullscale_reach.py`
proves that each gated region is unreachable before its opening fact and that
each opened route works; see [Phase 1 results](#phase-1-results). This is a
test, not a visual check. Changing terrain or collision invalidates prior
ground truth, as `AGENTS.md` requires.

## Task A: The Relay Keeper's Apprenticeship — 12 milestones

**Goal shown to the agent (constant for the whole episode):**
*"Restore Windrest's signal route and bring the summit beacon online. The relay
keeper can show you how the old machinery works."*

The procedure under test is **R**, diagnosing and commissioning an elemental relay,
as specified in the legacy design. Each seed samples a permutation of ember, tide,
and gale as the source, vane, and drain elements. The keeper teaches it at A01.
The default party is Kestrel, Maren, and Wren, so all three elements are available.

At full scale, a relay installation spreads its four receivers (source, vane, drain,
core) over a 12–20 m footprint. At least one receiver is elevated 4–8 m and reached
by stairs or a climbable face. Role markings are consistent everywhere. Positions,
orientation, and family appearance change between installations:

- training relays use a timber frame;
- field relays use stone masts;
- the summit beacon has a differently shaped, tiered silhouette with the same role markings.

| ID | Milestone | Full-scale content | Completion predicate | Est. path | Est. time |
| --- | --- | --- | --- | ---: | ---: |
| A01 | Complete the keeper's guided relay repair | Keeper lesson in the village yard, then training relay 1 | Relay 1 active | 70 m | 1.8 min |
| A02 | Repair a second training relay independently | Keeper powers relay 2: rearranged roles, no stepwise text | Relay 2 active | 120 m | 1.2 min |
| A03 | Recover an overloaded training relay | Relay 3 starts overloaded; one scripted interruption after priming | Relay 3 active | 210 m | 2.0 min |
| A04 | Survey the marsh supply route | Register three reed-post junctions across the marsh channels, then the cache windmill. The scout also describes the old ruin (needed at A10) | Three posts and the cache registered | 1,210 m | 5.5 min |
| A05 | Recover the ferry-lift casing | Raider camp in the eastern hills; fighting or sneaking are both legal | Casing in pack | 440 m | 4.1 min |
| A06 | Repair the ferry lift | Deliver the casing; the mechanic installs it; ride the cable lift up 50 m; pull the upper test lever, which lights the cache lamp | Lever pulled with the lift operational | 470 m | 3.6 min |
| A07 | Supply the highland outpost | Glide or ride down, return to the remembered cache (**spatial probe S1**), ride the lift up, deliver to the quartermaster; the corridor gate opens | Supplies delivered | 1,440 m | 7.0 min |
| A08 | Restore the first field relay | Unseen stone-mast relay on the ridge; no recipe or skill hint (**first delayed R probe**) | Field relay 1 active | 370 m | 2.1 min |
| A09 | Stabilize the second field relay | Starts overloaded and partly primed; skip completed stages | Field relay 2 active | 470 m | 2.7 min |
| A10 | Recover the summit service cap | Glide down the western escarpment to the lowland ruin described at A04, whose vault the field relays now power; get past the ruin guards (**spatial probe S2**) | Cap in pack | 740 m | 5.2 min |
| A11 | Reopen summit access | Walk back to the lower ferry, ride the lift up, climb the switchbacks, install the cap in the pillar socket, ride the service lift 47 m | Character on the work platform | 1,400 m | 8.2 min |
| A12 | Commission the summit beacon | Tiered beacon, same role markings; one state-preserving interruption | Beacon active **and** both field relays still active | 20 m | 1.6 min |
| | **Total** | | | **≈ 6.96 km** | **≈ 45 min** |

At the 0.2 s decision interval, 45 minutes is about **13,500 native actions**. The target
range is 30–60 minutes (9,000–18,000 actions), the same as Skylands Task A. Skylands A
measured 57.9 minutes after implementation.

**Estimate method.** Straight-line distances between the region centers are multiplied
by a terrain factor of 1.1–1.45, then divided by an effective speed of 4.5 m/s. Mechanism
time is added per milestone: 45–180 s, covering lessons, receiver positioning, skill
cooldowns of 6–9 s, combat, and lift rides. These are planning numbers. Final budgets
are set from legal reference rollouts.

### How the agent learns what to do next

There is no step pointer. A short lead is delivered in the world after each receipt.
It names landmarks and directions, never coordinates.

| After | Lead source | Lead content (paraphrased) |
| --- | --- | --- |
| Reset | Constant goal; the keeper stands in view of spawn | Talk to the keeper. |
| A01, A02 | Keeper | The next practice relay in the yard is powered. |
| A03 | Keeper | Help the marsh scout register the reed posts east of the village. Note the quartermaster's cache, a broken windmill on an islet. |
| A04 | Cache plaque; scout | The cache is sealed until the ferry signal burns. The lower-ferry mechanic says raiders in the eastern hills took the lift casing. The old summit keepers kept spare lift caps in the vault under the broken arch tower, in the western gully below the cliffs north of the village; its seal has been dead for years. |
| A05 | Item description | The ferry-lift casing is for the lower-ferry mechanic. |
| A06 | Mechanic | The signal is lit. Bring the quartermaster's supplies from the cache to the highland outpost. (No location is repeated.) |
| A07 | Quartermaster | The corridor gate is open. Restore both relays on the signal ridge, then the summit. |
| A09 | Relay service plaque | The summit service lift lacks its cap. The ruin vault the marsh scout described now has power. (No location is repeated.) |
| A10 | Item description | Summit service-lift cap. |

Dialogue stays in `local_text` while the agent is within about 4 m of the speaker,
and for 5 s afterwards. Speakers repeat their latest lead when spoken to again.
This is **assisted recall** and is logged with its travel cost, so forgetting
shows up as a measurable detour rather than making the task unsolvable.
Signposts at major junctions show region names within about 6 m.

### Progression rules

- Receipts are issued strictly in A01–A12 order.
- World changes made early persist and are credited when the chain reaches them.
  Examples: defeated raiders, a registered reed post, the casing taken early.
- An object owned by a later milestone responds with diegetic feedback, such as
  "The mechanic has no part to install", never with step numbers. The compact
  course's "not part of the current step" message is removed.
- Where early access would remove a designed probe, a visible lock prevents it,
  as listed in [Gates and locks](#gates-and-locks).
- The keeper powers the training relays one at a time. A02 and A03 therefore
  follow the lesson.

### Memory schedule

Times are taken from the estimate above and will be replaced by measured values.

| Exposure | Probe | Estimated delay | Intervening work |
| --- | --- | --- | --- |
| R taught and practiced, A01–A03 (ends at about 5 min) | A08 (starts at about 25 min) | ≈ 20 min, ≈ 6,000 actions | A04–A07: survey, camp, ferry repair, supply run. No relay practice. |
| R partial reuse at A09 (ends at about 30 min) | A12 (starts at about 43 min) | ≈ 13 min | A10–A11: ruin and summit ascent |
| Cache registered at A04 (about 10 min) | S1, return during A07 (about 20 min) | ≈ 10 min | A05–A06 |
| Ruin described at A04 (about 10 min) | S2, A10 (starts at about 30 min) | ≈ 20 min, ≈ 6,000 actions | A05–A09 |

Fresh agent contexts start before A08 and A12, preserving only the memory the
evaluation condition permits. This is unchanged from the legacy design. The
environment emits these boundaries as events. The model runner (`owb/scripts/teyvat_model.py`)
does not implement them yet; adding that support is part of phase 6 below.

## Spatial memory probes

Relay skill R remains the primary test. Two location probes are added so that
spatial memory is measured in its own right, not only as a side effect of travel.

| Probe | Place | Learned | Used | Delay | Withheld at use |
| --- | --- | --- | --- | --- | --- |
| S1 | Cache windmill on the marsh islet | Visited and registered at A04 | A07, after the mechanic's lead | ≈ 10 min | Location in the lead; a lamp visible from afar |
| S2 | Ruin vault under the broken arch tower | Tower visible from the training yard at A01–A03; described by the scout at A04 | A10, after the relay plaque's lead | ≈ 20 min | Location in the lead; the vault's powered state from afar |

The design rules that keep these probes honest:

- **No distant beacons.** The cache lamp is small and hooded, visible only within
  about 40 m, with no beam, glow, or smoke above the terrain. The vault seal is readable
  only within about 10 m. Landmarks themselves (the windmill and the arch tower)
  stay visible from a distance, because recognizing a remembered landmark is part
  of the skill.
- **Leads at use time do not repeat locations.** The A06 and A09 leads name the
  target ("the cache", "the ruin vault the scout described") without directions.
- **Early access is locked.** The cache seal and the vault seal stop the agent from
  collecting either item before its probe ([Gates and locks](#gates-and-locks)).
- **Recall can be assisted.** Re-asking the scout repeats the A04 directions.
  This is logged as assisted recall, not counted as unaided success.

### Spatial metrics

Each probe starts when its use-time lead is delivered and ends when the agent first
comes within interaction range of the target. All metrics are computed from the
action log and are reported separately from task success.

| Metric | Definition |
| --- | --- |
| Path efficiency | Shortest feasible path length ÷ actual path length. The shortest path comes from the reachability planner, under the same traversal rules as the agent. 1.0 is perfect. |
| Initial heading error | Angle between the agent's displacement over its first 50 m of travel and the shortest path's direction over the same 50 m |
| Time to target | Simulated seconds from the lead to arrival |
| Assisted recall | Number of times the agent asks the scout (or another source) again before arrival, and the extra distance this costs |
| Unaided success | Arrived without assisted recall and within 3× the shortest-path length |

The reference controller's runs record the shortest-path length for each probe on
each seed, so these metrics can be computed for any later policy.

### World map as an evaluation condition

The fog-of-war world map (`EnvConfig.world_map`, on by default in every OWB
environment) paints explored terrain together with building footprints
(`open_world_bench/world_map.py`). The cache windmill and the arch tower therefore
appear on the map after the agent has seen them. The map works as external spatial
memory.

Every spatial-probe evaluation therefore runs in two conditions:

- **map-off:** `world_map=False`. **The primary spatial-memory scores come from this condition.**
- **map-on:** the default interface, reported alongside for comparison.

Task success is reported for both conditions. The map shows no quest markers in either.

## Observation, actions, reward, and termination

These follow the campaign-v1 contract described in
[teyvat-long-horizon-tasks.md](teyvat-long-horizon-tasks.md#observation-controls-and-scoring).

- **Observations:** `rgb`, `proprio`, `party`, the constant goal as `instruction`, and
  `local_text`. `local_text` contains the pack, current dialogue or plaque text, recent
  mechanism feedback (relay progress and overload, for example), and up to five
  visible labels within 24 m, with terrain and building occlusion. `task` is a zero vector.
- **Withheld:** step text, coordinates, element-role tables, future leads, the
  milestone table, and quest markers. The fog-of-war world map is an evaluation
  condition, described in [World map as an evaluation condition](#world-map-as-an-evaluation-condition).
- **Actions:** native movement, look, jump/glide, sprint/climb, interact, attack,
  skill, burst, and party swap. There is no navigation macro or teleport. Version 1
  has **no fast travel**.
- **Reward:** +1 at A12, zero otherwise. Receipts and diagnostics go to `info`
  for scoring and are never shown to the policy.
- **Death:** a party wipe respawns the team at the nearest discovered statue
  (village, lower ferry, upper landing, outpost), with world facts and the pack
  preserved. Deaths are counted. Today `World._respawn` picks the nearest of any
  `WAYPOINT` entity. The full-scale task restricts this to discovered statues.
- **Budget:** `max_steps` is provisionally 60,000, about 4.5 times the estimate, and will
  be set after the reference rollouts.

## Assets and presentation

New or extended assets:

- the cable ferry lift (tower, cable, and cabin);
- the summit service lift and pillar socket;
- three relay families (timber training frame, stone field mast, tiered summit beacon);
- the cache windmill with its signal lamp;
- reed posts and signposts;
- the raider palisade camp;
- the ruin vault;
- statues.

Appearance is keyed to stable entity identity and never consumes simulation RNG.
Collision for lifts and seals matches their visible geometry.

User-facing walkthroughs use **Blender Cycles**, following the Skylands recording
contract: labeled travel timelapse, milestone chapters, recorded settings, and
playback verification. Policy observations keep the inexpensive native renderer.
Validation runs use headless zero-RGB observations, as campaign-v1 does.

## Phase 1 results

Implemented in new modules only; no existing file changed.

| Module | Role |
| --- | --- |
| `open_world_bench/fullscale_physics.py` | `FullScaleController`, `FullScaleCombat`, `climb_height_bound` |
| `open_world_bench/fullscale_world.py` | Seeded layout, authored terrain, scenery, structures, ravine gate, lifts, region masks, world digest |
| `open_world_bench/fullscale_reach.py` | Gate proofs on the bilinear surface (`over` and `under` models) and `check_world` |
| `open_world_bench/fullscale_routes.py` | Real-controller checks: the west-edge glide and the strongest climb at every gate wall |
| `owb/scripts/teyvat_fullscale.py` | `world-check`: per-seed `report.json` and a diagnostic top-down `map.png` |
| `tests/test_teyvat_fullscale_world.py` | 17 tests: physics rules, world contents, proof models against the controller, gates for seeds 7 and 19 |

**How the proofs work.** Each 2.5 m square of the bilinear ground surface is a
graph node. For each one the validator computes exactly which part is walkable
(the gradient is separable, so that part is convex) and its height range. Two
models bracket the real controller:

- `over` is more capable than the controller. It swims without limit, falls
  anywhere lower, glides 1.25 times farther than a full stamina bar allows at
  1.35 times the real glide ratio, and climbs 30.8 m from where it stands. A climb started mid-air gets
  only the climbing part, 26.4 m. Climbs can also move sideways for free within
  that distance, but stamina never refreshes on a face, in a fall, or in a
  glide. A region `over` cannot reach is unreachable.
- `under` is less capable: it walks only on gentle ground, wades through
  gentle shallows, rides lifts, and never climbs, falls or glides. A region
  `under` reaches is reachable.
- The real controller then flies the designed glide and makes the strongest
  climb at 17 points along the ridge wall and around the pillar.

A toy-wall test confirms that the validator agrees with the controller. Both open
a 15 m wall, and both keep a 45 m wall sealed.

| Seed | Before ferry: highest `over` point (plateau rim ≥ 57 m) | After ferry (ridge rim ≥ 103 m) | After ravine gate (pillar top 156 m) | Strongest real climb | West glide |
| --- | ---: | ---: | ---: | ---: | --- |
| 7 | 31.7 m | 92.5 m | 141.9 m | 22.0 m | lands 37 m from the ruin |
| 19 | 31.7 m | 92.5 m | 142.1 m | 23.0 m | lands, ruin reached on foot |
| 3, 101, 2026 | 31.7 m | 92.5 m | ≤ 142.3 m | ≤ 22.7 m | lands, ruin reached on foot |

Every sealed region has zero reachable patches before its fact, and every site
behind each gate is reachable after it. Each seed takes about 45 s:

```bash
python -B -m owb.scripts.teyvat_fullscale world-check --seed 7 19 --output /tmp/fullscale-world
```

Known limits of this phase: the terrain is still geometric (straight-ish river
and walls, a sandy marsh), and structures reuse existing building kinds. Natural
shaping and the dedicated assets follow with the task content.

## Route preview

Before the task mechanics exist, a **route preview** shows the whole Task A
journey on the phase-1 world. It is labelled as a preview in the video and in
every artifact. It is not a task solution.

- `open_world_bench/fullscale_env.py`: `TeyvatFullScaleEnv`, the full-scale
  world as a native `OpenWorldEnv` with the solid-terrain controller, the
  ravine gate's collision, and two working lifts. Interacting at a lift end
  rides it once its fact is open; the cabin carries the character up beside
  the lower end, across a level cable, and down onto the upper end. Each
  milestone is a list of checkpoints that completes when the character
  arrives at them in order. Arriving at the lower ferry opens `ferry_open`,
  at the quartermaster `corridor_open`, and at the service-lift socket
  `cap_installed`, the places where the task logic will open them. Site
  fixtures (relays, NPCs, chests, posts) are static.
- `open_world_bench/fullscale_nav.py`: shortest routes on the phase-1 `under`
  graph (gentle ground, shallows, open water), with solid props and a
  clearance margin removed. Planned routes cross only ground the proofs
  already accept.
- `open_world_bench/fullscale_policy.py`: `RoutePreviewPolicy`, a privileged
  controller that plans with the anchors, lift paths, and open facts, and acts
  only through native actions (camera-relative movement, look, sprint, jump to
  glide, interact to board). It rests at the west edge until stamina is full
  before the glide, because a partial bar can run out mid-air.
- `owb/scripts/teyvat_fullscale_preview.py`: `collect` runs the agent and
  replays every action in a fresh env, comparing positions, events, rewards,
  and the state digest after every action. `render` replays again, renders
  sampled states with Blender Cycles on one or more GPUs (resumable from a
  hashed frame cache), and assembles a captioned 1600 × 900 walkthrough.

| Seed | Actions | Simulated time | Travel | HP | Re-plans |
| --- | ---: | ---: | ---: | --- | ---: |
| 7 | 5,445 | 18.2 min | 6.23 km | full throughout | 0 |
| 19 | 5,339 | 17.8 min | 6.07 km | full throughout | 0 |

Without the relay, combat, and item mechanics, the route takes about 18
minutes, against the 45-minute estimate for the full task; the difference is
mechanism time and the slower walking of a real agent.

The video follows the [Skylands campaign](skylands-campaign-demos.md) format:
a labelled timelapse of real replay states (walking 16×, lift rides 10×, the
glide 4×), with event holds, a milestone list, a route map, and captions
around the native 1280 × 720 Cycles scene (Blender 4.5.4 LTS, OptiX, 64
samples, denoised). Lift towers, cables, and cabins are presentation geometry;
the gate box matches its collision.

```bash
python -B -m owb.scripts.teyvat_fullscale_preview collect --seed 7 --output runs/fullscale-preview/seed-7
python -B -m owb.scripts.teyvat_fullscale_preview render runs/fullscale-preview/seed-7 \
  --cache .cache/teyvat-fullscale-preview/seed-7 --gpus 0 1 --per-gpu 2
```

`tests/test_teyvat_fullscale_preview.py` checks that lifts refuse to run
before their fact, carry the character without harm, that the ravine route
appears only with the gate open, and that a full seed-7 run completes and
replays exactly.

## Implementation plan

New modules, so that the existing suites stay untouched:

| Module | Role |
| --- | --- |
| `open_world_bench/fullscale_tasks.py` | `FullScaleCourse(Course)`: authored site table, leads, locks, diegetic feedback, receipts, verification |
| `open_world_bench/fullscale_env.py` | `TeyvatFullScaleEnv`: terrain authoring, regions, scenery, gate collision, campaign-style `local_text` |
| `open_world_bench/fullscale_policy.py` | Privileged reference controller with a heightfield route planner (walk, climb, swim, glide costs and stamina feasibility). The campaign-v1 planner only makes straight-line detours around buildings. |
| `owb/scripts/teyvat_fullscale.py` | Extend `world-check` with record, verify, and render, reusing the `teyvat_reference.py` machinery |
| `tests/test_teyvat_fullscale.py` | Contract tests |

Shared interface: `owb.make_env("teyvat", task="apprenticeship", variant="full")`.
This requires extending the allowed `BenchmarkConfig.task` values. `variant="small"`
selects geography scale 1.0 and is for tests only. The suite version is `teyvat-fullscale-v1`.

**Compatibility constraint.** `owb/scripts/teyvat_campaign.py --verify` refuses
to replay campaign-v1 evidence if any source file recorded in its manifest has changed. That
covers every `open_world_bench/**/*.py` file plus `owb/render/scene_export.py`.
Prefer adding modules. If a shared file must change, for example to add Cycles
assets in `scene_export.py`, re-record and re-verify the campaign-v1 and compact
evidence in the same change, and confirm that their state digests are unchanged.

| Phase | Work | Exit criterion |
| --- | --- | --- |
| 1 ✅ | Terrain, regions, structures, gates; reachability validator | Every gated region is provably unreachable before its fact; no unintended rest ledges on gate cliffs |
| 2 | Course logic, leads, locks, observation contract | Unit tests for predicates, locks, privacy, and early-credit rules |
| 3 | Reference planner and rollouts for seeds 7 and 19 | Both complete 12/12 without deaths; independent replay matches every action; shortest-path lengths recorded for S1 and S2 |
| 4 | Shared interface registration, documentation, measured lengths | `make_env` smoke run; this document updated with measured numbers |
| 5 | Cycles walkthrough with chapters | Playback-verified MP4 and recorded settings |
| 6 | Evaluation harness | Fresh-context boundaries in the model runner; spatial metrics computed from action logs; map-on and map-off conditions; a budgeted pilot |

## Validation checklist

- Reference completes A01–A12 with seeds 7 and 19. Actions are native only, and the controller is checked for simulator-state mutation.
- An independent reset-and-replay matches positions, events, rewards, and state digests for every action.
- Reachability proofs for the escarpment, the cache seal, the corridor gate, and the summit pillar.
- Observation privacy: no step text, coordinates, role tables, or receipts in observations.
- Every milestone's lead is delivered in-world before its first required action.
- Early-credit and lock behavior: casing taken early is credited at A05; the cache stays sealed until A06; the ruin vault stays sealed until A09.
- Beacon check: from the lower ferry, the marsh junctions, and the plateau edge, renders show no cache-lamp or vault-seal pixels beyond their stated ranges.
- The A06 and A09 leads contain no directions; only the A04 lead locates the ruin.
- Measured delays for each probe in the memory schedule, recorded with the evidence.
- Measured length and walking share, compared with the estimates here.

## Defaults chosen for review

| Decision | Default in this design | Alternative |
| --- | --- | --- |
| Shared API task name | `apprenticeship` (A), later `expedition` (B) | `a` / `b` |
| Death | Respawn at the nearest discovered statue; facts preserved; counted | Terminate the episode, as campaign-v1 does |
| Fast travel | None in v1 | Statue teleport, as an optional variant |
| World map | Evaluation condition: map-off and map-on are both run, and spatial scores come from map-off | Always on, as in campaign-v1 |
| Time of day | Fixed | Day/night cycle |

## Task B outlook

Task B keeps its 24 legacy milestones and adds procedures G (sentinels) and T (vent
crossings) at regional scale. Its target is 60–120 minutes, on an island of roughly
1,600 m (geography scale 2.5). It reuses Task A's terrain authoring, gate validator,
route planner, and recording tools. Two pieces of new mechanics are needed:

- sentinels that deal active damage, which the compact course left out;
- vent routes whose shelves and landings are validated on real terrain.

The full design, its review, and the seed-7 implementation (island of 1,760 m,
geography scale 2.75) are in [teyvat-full-scale-task-b.md](teyvat-full-scale-task-b.md).
