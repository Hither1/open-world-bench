# Teyvat full-scale tasks: design for Task B

Status: **implemented for seed 7** (world, mechanisms, course, reference solution,
replay verification, Cycles walkthrough); see [Implementation results](#implementation-results-seed-7).
World proofs also pass on seeds 3, 11, 19 and 23, and the seed-19 reference completes. The
implementation was reviewed; see [Implementation review](#implementation-review). The design
tables below keep their planning estimates; measured numbers are in the results section.
Shared-interface registration and the evaluation harness are not done.

This document scales the 24-milestone [legacy Task B](teyvat-legacy-tasks.md#task-b-the-expedition-of-three-arts--24-milestones),
*The Expedition of Three Arts*, from its [compact reference course](teyvat-reference-demos.md)
to a full island. It reuses what [Task A's full-scale design](teyvat-full-scale-tasks.md)
built and validated in phase 1:

- authored heightfields;
- the solid-terrain controller (`FullScaleController`);
- gate collision;
- the `over`/`under` reachability proofs;
- the lift and route-preview machinery.

Task B adds two mechanics that Task A does not have: sentinels that deal active damage
(procedure **G**) and vent crossings on real terrain (procedure **T**).

| | Compact course (today) | Full-scale Task B (this design) |
| --- | --- | --- |
| Milestones | 24, linear | 24, linear, same IDs and predicates |
| World | 640 m flat plateau; sites on a 46 m serpentine grid | 1,760 m authored island; five height tiers, an enclosed crater with a central mesa, and 14 rock stacks reached only by vent |
| Reference length | 3,038 actions, 10.1 min (seed 7) | **≈ 22,000 actions, ≈ 73 min, ≈ 7.3 km** (estimate) |
| Sentinels | Stationary; no damage; 18 s wind-up | Telegraphed slam and warding pulse that hurt; wrong elements are punished; 5 s wind-up; 8 s vulnerable window; shield reforms |
| Vents | 3 m shelves on flat patches; any launch reaches the next shelf | Launch columns sized per leg; shelves on stacks above the climb ceiling; every uncoached route has one long level leg, so resting between legs is required |
| Next-step guidance | `Step i/n: ...` appended to the instruction | None. Leads come from NPCs, plaques, and item descriptions in the world |
| Spatial memory | None tested | Depot probe S1 (B07 → B15); world map as an evaluation condition |

Task B is independent of Task A, as in the legacy design:

- it uses a different island;
- it starts with empty memory;
- it samples its own family assignments.

The two tasks share code, not content.

## Design rules carried over from Task A

- **Physics.** `FullScaleController` rules apply unchanged: solid terrain, no standing on
  steep ground, stamina-limited climbing, and launch skills from walkable ground only.
  One climb gains at most 24.6 m; the proofs use 30.8 m.
  - A glide covers about 154 m of range and 58 m of drop per stamina bar (144 m and
    54 m after the 7-stamina jump that a vent launch needs).
  - Falls start to hurt from about 10 m.
- **Gates.** Every region gate is either:
  - a wall at least 46 m tall and at least 10 m above the proven climb ceiling of the tier below; or
  - a gate box whose collision matches its visuals, set into rock on both sides.
- **Bypass proofs.** Each gated region is proven unreachable before its opening fact
  under the `over` model, and reachable after it under the `under` model. Task B adds
  vents to both models (see [Vent proofs](#vent-proofs)).
- **Diegetic leads.** There is no step pointer and there are no coordinates in leads.
  Speakers repeat their latest lead when spoken to again; this is logged as assisted recall.
- **Rendering.** Appearance never draws from the simulation RNG. Cycles is used for every
  user-facing video, and every video is labeled and replay-verified.

## World layout

- **Extent:** geography scale 2.75, a 1,760 m square. Land lies within about 780 m of the center.
- **Crater:** center (0, -430); floor inside radius 214 m; ring wall crest (150 m) between
  radius 214 and 244 m. The Crown mesa (radius 85 m) is centered near (70, -440), and the
  spire (radius 16 m) stands at its center.
- **Heightfield:** resolution 705, giving 2.5 m cells, the same cell size as Task A.
- **Tiers:**

  | Tier | Height | Wall to the tier below |
  | --- | --- | --- |
  | Lowland | 9 m | (none) |
  | Upper valley (plateau) | 58 m | Great Scarp, 49 m |
  | Highland and crater floor | 104 m | Highland wall, 46 m |
  | Crown mesa | 150 m | 46 m above the crater floor |
  | Summit spire | 196 m | 46 m above the Crown |

  The crater rim is a closed ring at 150 m, 46 m above both the highland outside and the
  crater floor inside.

```text
                                      N  (-z)
  +--------------------------------------------------------------------+
  |                 ______ crater rim 150 m (closed ring) ______         |
  |               /   (B24) outpost stack 196 m                \       |
  |              |  shrine          ___ Crown 150 m ___          |      |
  |              |  stack (B18)   /  spire 196 m (B23) \         |      |
  |              |  150 m        | B21  B22  B23 relays |        |      |
  |              |   B19 court    \__ B20 landing ______/        |      |
  |              |      B20 vent stacks  -->                      |      |
  |               \______ [crater gate] (B17) _________________/       |
  |   highland 104 m   sky-lift top           gate sentinel, gate relay |
  |=========== highland wall 46 m ======================================|
  |   (B16) sky-lift controls, relay, powered vents ^                   |
  |   upper valley 58 m: signal station (B11)   lookout stacks (B12)   |
  |   field sentinel hollow (B13)   reserve relay (B14)                 |
  |                   Scarp Cut ramp ^                                  |
  |=========== Great Scarp 49 m ======[causeway gate] (B10)=============|
  |  arenas (B03-B04)        Mirror Lake (B09)      raider hills (B08)  |
  |  workshop (B01-B02)      depot granary (B07, B15)                   |
  |  village, spawn    wind-guide stacks (B05-B06)                      |
  +--------------------------------------------------------------------+
                                      S  (+z)
```

| Region | Approx. center (x, z) m | Height | Content |
| --- | --- | --- | --- |
| Hearthvale village | (-430, 640) | 9 | Spawn, relay keeper, timber workshop with two training relays, respawn statue |
| Warden's arenas | (-500, 420) | 9 | Two ring arenas, about 28 m across, with low walls; sentinel warden |
| Wind-guide stacks | (-150, 690) | 9 → 49–65 | Wind guide; three training stacks; two overlooks. Every stack is at least 300 m from the Great Scarp, and no lowland stack top exceeds 65 m |
| Expedition depot | (300, 665) | 9 | Tall stone granary with a crane: a unique silhouette. Quartermaster; reserve compartment that is visibly sealed; respawn statue. Placed off the axis of the Scarp Cut descent (about 35° east of it, 450 m away) so the return is not a straight walk |
| Mirror Lake | (180, 470) | -8 floor | Round lake about 140 m across; a marked chest on the bed at about 7 m depth |
| Raider hills | (430, 470) | 10–17 | Palisade camp with 3–4 ordinary raiders; chest holding the repair kit |
| Scarp Cut | (40, 300 → 170) | 9 → 58 | 12 m wide ravine ramp (21°) up the Great Scarp. Causeway gate box at its mouth; mechanic beside it |
| Signal station | (-250, 180) | 58 | Stone-mast field relay |
| Lookout stacks | (275, 190) | 58 → 98–108 | Two stacks; the observation post sits on the higher one (108 m). About 340 m from the highland wall |
| Field-sentinel hollow | (-80, 40) | 58 | Arena with a sentinel of a different appearance |
| Reserve relay | (170, 30) | 58 | Relay inside a service panel that stays locked until it receives the field seal |
| Sky-lift | (-330, 60) | 58 → 104 | Control house, a relay, and powered vents. Route: pad, then stack K1 (98 m, about 55 m from the wall), then the highland rim |
| Crater gate | (0, -172) | 104 | Notch through the south rim, 12 m wide and 30 m long, filled by a full-height gate box. Gate sentinel and gate relay outside. (The legacy name is "the guarded causeway gate".) |
| Shrine stack | (-170, -455) | 104 → 150 | Radius 14 m. High shrine platform with the shrine sentinel (arena about 20 m across) |
| Shrine court | (-60, -300) | 104 | Double-ring relay with its housing sealed; service plate |
| Crown approach | (-90, -330) → Crown | 104 → 150 | Three-leg vent chain over two stacks to the Crown landing signal, about 200 m in all |
| Crown mesa | (70, -440) | 150 | Radius about 85 m. Double-ring relay (B21), old single-ring station (B22), summit-approach relay (B23); respawn statue at the landing |
| Summit spire | (70, -440) | 196 | Radius 16 m. Reliquary and its sentinel (arena about 20 m across); powered vents at its foot on the side away from the outpost |
| Distant outpost | (-90, -585) | 104 → 196 | Radius 14 m, on the crater floor, about 215 m from the spire and 20 m inside the ring. Reached from the Crown's north-west edge over one intermediate stack (180 m). Outpost sentinel at the Crown-edge pad |

Each seed jitters the site groups by up to ±18 m, as in Task A. Stack spacing and heights
are rechecked by the proofs after jitter. A seed fails world-check if any proof fails.

### Gates and locks

Region gates are physical. Item and power locks are interaction locks: the container or
housing refuses to open until its fact holds.

| Gate or lock | Kind | Opens when | What it protects |
| --- | --- | --- | --- |
| Great Scarp + causeway gate | Wall + gate box (Task A's corridor-gate geometry) | `causeway_open` (B10) | The upper valley is not reached before the interference interval ends (B07–B10) |
| Depot reserve compartment | Interaction lock; seal glyph visible within about 10 m | `reserve_power` (B14) | Tools cannot be taken at B07, so B15 needs a return to a remembered place |
| Reserve-relay service panel | Interaction lock | Field seal in pack (B13) | B14 follows B13 |
| Highland wall | 46 m wall; the only way up is the sky-lift's powered vents | `skylift_power` (B16) | The highland and crater stay later than the upper valley |
| Crater rim + crater gate | Closed ring wall 46 m above both sides + gate box | `crater_gate_open` (B17) | All crater and Crown work follows B17. No point outside is high enough to glide over the rim |
| Shrine-court relay housing | Interaction lock | Shrine seal in pack (B18) | The double-ring adaptation (B19) follows B18 |
| Crater-gate relay housing | Interaction lock | Gate sentinel defeated | B17 is G then R |
| Crown | 46 m walls; vent chain only | (T-gated, no fact) | Reached only by vent. Proven unreachable with all vents disabled |
| Crown landing signal | Interaction lock: "the signal has no charge" | Relay B19 active | `crown_signal` cannot be lit before the B19 adaptation, even if the Crown is reached early |
| Crown relays dormant, one at a time | World state: "no charge in the inlets" | B21 on `crown_signal`; B22 on B21 active; summit-approach housing (B23) on B22 active | B21 is the first v2 reuse; B22 follows it; B23 follows B22 |
| Spire vents | Powered vents | `summit_power` (B23 relay) | The reliquary follows its relay |
| Reliquary | Interaction lock | Reliquary sentinel defeated | B23 ends with G |
| Outpost sentinel | Dormant (inert, invulnerable) | Heart in pack | B24's G cannot be done early |
| Outpost vents | Powered vents | `outpost_clear` (outpost sentinel defeated) | B24 is G then T. This also keeps the spire sealed: unpowered outpost columns would reach the spire face before `summit_power` |
| Outpost beacon socket | Interaction lock | Heart in pack | B24 ends with R |

**Why a closed crater.** A knife ridge has ends, and a glide around an end into the sea
can land low enough on the far cliff to climb out. A ring wall has no ends. The highest
ground outside the crater is the highland at 104 m, and gliding only loses height, so the
rim at 150 m cannot be crossed without entering through the gate.

### Vent stacks and spacing rules

T-gated places are **stacks**: rock columns whose flat tops are at least 31 m above any
standable ground within reach. Two rules keep vents necessary:

1. **Vertical.** A stack top is more than 30.8 m above its base (the proof climb ceiling).
2. **Horizontal.** Gliding from a stack top or a vent apex loses height. For a target top
   `H` at distance `d` from a start at height `A`, the `over` model must give
   `A + 4 − 0.278·d < H − 26.4`.
   - 0.278 is the `over` glide's drop per metre: the real glide's 0.376 divided by 1.35.
   - 26.4 m is `air_climb_budget`, the height a climber who grabs a face mid-air can gain.
   - The `over` glide is also capped at 97 patch steps: about 241 m along the axes and
     341 m on diagonals.
3. **Column height.** A column must also pass the `under` vent glide, which drops
   0.47 m per metre, keeps 1.5 m of clearance, and is capped at about 123 m. Each column
   top is therefore `max(top, pad) + 0.4706·d + 6`.
   - Unpowered vents on the lowland stand at least 300 m from the Great Scarp, and no
     lowland stack top exceeds 65 m.
   - Unpowered vents on the plateau stand about 340 m from the highland wall.

The proofs, not these hand calculations, are the authority. The rules only make the
first layout likely to pass.

## Mechanics

The party is Kestrel (ember), Maren (tide), and Wren (gale), so all three elements are available.
Each seed samples three assignments independently of the layout:

- relay roles: a permutation of ember, tide, and gale over source, vane, and drain;
- the sentinel pair: shield-breaker and interrupter, two distinct elements;
- the vent family: its charge element and its safe pulse cue (bright or dark beat).

Cast reach follows the native skills:

| Skill | Shape | Reach |
| --- | --- | --- |
| Kestrel's Cinder Lash | Front arc | 4.0 m |
| Maren's Tidecall | Field around the caster | 6.5 m |
| Wren's Updraft | Field around the caster | 5.0 m |

An element applies to the nearest receiver, sentinel, or vent pad within that reach.

### R: elemental relays (single-ring and double-ring)

Rules are as in the legacy design.

- **Indicators.** Each relay shows progress (unprimed, primed, routed, active) and an
  independent overload flag.
- **Single ring (v1).** If overloaded, drain first; then prime the source, route the vane,
  and confirm at the core.
- **Double ring (v2).** Drain first even when the relay is not overloaded. The housing's
  service plate says the inlet keeps a residual charge, and priming first returns that
  diagnostic.
- **Wrong applications.** These give categorical feedback and a reversible fault.
  Draining keeps completed stages.

At full scale, each relay spreads its four receivers over a 12–20 m footprint on level
ground. Elevated receivers on plinths are deferred: a 4–8 m plinth needs either a terrain
ramp, which 2.5 m cells cannot shape cleanly, or standable box collision, which the
controller does not have. The families look different but share role markings:

| Family | Used at |
| --- | --- |
| Timber workshop frame | B01, B02 |
| Stone field mast | B11, B14, B16, B17, B22 |
| Shrine double ring, with a distinct badge | B19, B21, B23 |
| Tiered outpost beacon | B24 |

| Relay | Version | Initial state | Interruption |
| --- | --- | --- | --- |
| B01 training | v1 | clean | none (guided) |
| B02 workshop | v1 | overloaded | one, after priming |
| B11 signal station | v1 | clean | none |
| B14 reserve | v1 | primed and overloaded | none |
| B16 sky-lift | v1 | clean | none |
| B17 crater gate | v1 | clean | none |
| B19 shrine court | **v2** | clean | none (adaptation) |
| B21 Crown east | v2 | clean | none |
| B22 old station | v1 | clean | none (scope preservation; draining is safe but logged) |
| B23 summit approach | v2 | clean | one, after priming (the taught branch) |
| B24 outpost beacon | seeded v1 or v2, badge visible | clean | one, after priming |

### G: sentinels with active damage

A sentinel is a stationary construct in an arena about 28 m across. The phases are:

| Phase | Duration | Behavior and threat | Correct response |
| --- | --- | --- | --- |
| Shield | Until broken | Emblem visible. **Warding pulse** every 6 s: 8 damage within 6 m. A wrong element triggers a pulse at once | Apply the learned shield-breaker element |
| Exposed | 1.0 s | No attack | (wait) |
| Wind-up | 5.0 s, telegraphed by a glowing core and a rising tone line in `local_text` | If it is not interrupted, or if a wrong element hits: **slam** for 26 damage within 9 m, then back to shield | Apply the learned interrupter element inside the window |
| Vulnerable | 8.0 s | Core takes ordinary attacks | Attack the core |
| (reform) | | After the vulnerable window the shield reforms; at B04 it also reforms once right after the first interrupt | Go back to diagnosis |

The breaker and interrupter belong to different characters, so their skill cooldowns do
not interact. The window has to cover one party swap (1.0 s cooldown) and the decision
interval (0.2 s).

Health is set so that a correct cycle takes one or two vulnerable windows:

| Encounter | HP |
| --- | --- |
| Training (B03, B04) | 160 |
| Field (B13, B17, B18) | 220 |
| Reliquary and outpost (B23, B24) | 260 |

**Logging.** Shield cycles, unnecessary casts, correct interrupts, and missed windows are
logged separately from defeat.

Ordinary raiders (B08) keep native pursuit and damage. Death respawns the party at the
nearest discovered statue, keeps facts and the pack, and is counted.

**Variation across encounters:**

- Later sentinels change arena, appearance, HP, orientation, and starting phase.
- B13 starts **exposed**, so shield breaking is skipped.
- B17 starts in **wind-up**.
- A sentinel that starts in a non-shield phase holds it, inert, until the agent comes within
  12 m. The full phase duration then starts. Otherwise the phase would expire before the
  agent arrived.
- Wrong elements are punished (a pulse in the shield phase, the slam in wind-up). Trying all
  three elements therefore costs health and a full cycle, so brute force is not free.
- Healing: interacting with a discovered statue restores the party, as in the source game.

### T: vent crossings

A vent is a pad 3 m across marked with the family glyph. It pulses on a 4 s cycle: 2.5 s
bright, 1.5 s dark. The steps are:

1. **Charge.** Apply the family's element within reach. The charge holds until it is
   used, or for 30 s.
2. **Launch.** Press jump inside the column on the family's **safe beat**. The character
   is carried up (kinematic, 14 m/s, AIRBORNE, no stamina cost) and stops at the column
   top with zero vertical velocity, so there is no overshoot past the apex. A jump on the
   wrong beat makes the vent sputter: the jump stays an ordinary jump, and the charge is spent. Wren's Updraft
   charges a gale vent but never counts as the launch; the launch needs its own jump.
3. **Glide** to the target shelf. This costs 5.5 stamina/s, as natively.
4. **Rest** on the shelf until stamina allows the next leg. Stamina regenerates at
   21/s after a 0.45 s delay.

A landing that misses the shelf falls to recovery ground. Every stack stands on the floor
of the tier it rises from, so a missed landing never strands the character. Powered vents
also need their relay's fact. Without it they show a dark pad and reject the charge with
"no power".

**Stamina condition.** Every uncoached route has one long level leg of 80–90 m, which costs
about 65 stamina including the launch jump. A second leg flown straight after it would run
the bar out mid-air, so the agent has to read its stamina and rest on the shelf.

| Route | Legs | Vents | Purpose |
| --- | ---: | --- | --- |
| B05 guided | 2 | unpowered | Coached acquisition of T |
| B06 second overlook | 3 | unpowered | Uncoached; rearranged legs, one long |
| B12 lookout | 2 | unpowered | Delayed T transfer; one long leg |
| B16 sky-lift | 2 | powered (`skylift_power`) | R then T composition; region gate |
| B18 shrine approach | 2 | unpowered | T then G |
| B20 Crown approach | 3 | unpowered | Extended crossing; T-gated region; one long leg |
| B23 spire | 1 | powered (`summit_power`) | R, T, G composition |
| B24 outpost | 2 | powered (`outpost_clear`) | G, T, R from saved memory |

Legacy rule: "Validate that a scored skill probe actually requires the taught
interaction." The proofs establish that every route's destination is unreachable with
vents disabled. They do not prove that every individual leg is necessary. The reference
logs the launches it actually used, and any legal shortcut is reported, not scored as T reuse.

### Vent proofs

`fullscale_reach` gains vents as launch sources. If a pad patch is reached and its vent
is usable under the current facts, a glide starts from the column top:

- in the `over` model, from the column top plus 4 m, with the full `over` glide;
- in the `under` model, from the column top, at the `under` glide ratio over 80 % of one
  real stamina bar, landing only on standable patches.

**Checks:**

- Every T-gated destination is unreachable under `over` with all vents disabled.
- With vents enabled, every T-gated destination is reachable under `under`.
- No region behind a fact gate becomes reachable early through any vent.
- The real controller flies every leg from its column top (`fullscale_b_check.vent_flight`)
  and lands on the shelf with stamina to spare. The launch itself (charge, beat, rise) is
  exercised by the reference run and a unit test.

## Task B: The Expedition of Three Arts — 24 milestones

**Goal shown to the agent (constant):** *"Reopen the valley's expedition route, recover
the beacon heart, and bring the distant outpost online. Train with the relay keeper,
sentinel warden, and wind guide before setting out."*

| ID | Milestone | Full-scale content | Completion predicate | Est. path | Est. time |
| --- | --- | --- | --- | ---: | ---: |
| B01 | Complete the relay lesson | Keeper's spoken lesson, then the timber training relay | Relay B01 active | 80 m | 2.0 min |
| B02 | Repair the workshop's faulty relay | Rearranged roles; overloaded; one interruption after priming | Relay B02 active | 110 m | 2.0 min |
| B03 | Complete the sentinel lesson | Warden's lesson; training sentinel in arena 1 | Sentinel B03 defeated | 160 m | 2.5 min |
| B04 | Clear the second training arena | Uncoached; one authored shield reformation | Sentinel B04 defeated | 110 m | 2.5 min |
| B05 | Complete the wind guide's crossing | Guide's lesson; 2 legs to overlook 1 | Standing on overlook 1 (reachable only by vent, per the proofs) | 230 m | 2.5 min |
| B06 | Reach the second training overlook | Uncoached; 3 legs, rearranged | Standing on overlook 2 | 180 m | 3.0 min |
| B07 | Survey the expedition cache | Register the depot granary; the reserve compartment is visibly sealed | Depot registered | 470 m | 2.0 min |
| B08 | Recover the causeway repair kit | Raider camp in the eastern hills; fighting or sneaking both legal | Kit in pack | 420 m | 4.0 min |
| B09 | Retrieve the submerged ferry coupling | Hear the lake boatman on the east shore name the chest's 3-glyph mark; dive in Mirror Lake and open the chest with that mark among six on the bed at about 8 m depth (**working-memory probe `WM-TB09`**, select) | Boatman heard; coupling in pack | 260 m | 2.5 min |
| B10 | Reopen the expedition causeway | Deliver kit and coupling; the mechanic gives a 4-element gate pattern; the causeway gate opens; climb the Scarp Cut; set the pattern in reverse on the winch at its far end (**working-memory probe `WM-TB10`**, reverse) | Character on the upper valley; winch set | 460 m | 2.5 min |
| B11 | Restore the valley signal station | Stone-mast relay; no recipe (**first delayed R probe**) | Relay B11 active | 330 m | 2.5 min |
| B12 | Reach the cliff observation post | Lookout stacks, 2 legs (**delayed T probe**) | Standing at the post | 520 m | 3.5 min |
| B13 | Recover the warden's field seal | Different-looking sentinel, starts exposed (**delayed G probe**) | Seal in pack | 340 m | 2.5 min |
| B14 | Stabilize the reserve-power relay | Unlock the panel with the seal; relay primed and overloaded | Relay B14 active (`reserve_power`) | 250 m | 2.5 min |
| B15 | Retrieve the expedition service tools | Return down the Scarp Cut to the depot without a location in the lead (**spatial probe S1**) | Tools in pack | 650 m | 3.0 min |
| B16 | Open the sky-lift route | Tools open the control house; commission its relay; 2 powered legs up the highland wall | Standing on the highland | 1,050 m | 6.0 min |
| B17 | Restore the guarded crater gate | Gate sentinel (starts in wind-up); its defeat unlocks the gate relay's housing | Relay B17 active (`crater_gate_open`) | 350 m | 3.5 min |
| B18 | Recover the high shrine's access seal | 2 legs onto the shrine stack; shrine sentinel | Shrine seal in pack | 300 m | 4.0 min |
| B19 | Commission a double-ring relay | Unseal the housing; service plate; drain before priming (**adaptation**) | Relay B19 active | 200 m | 3.0 min |
| B20 | Establish the remote landing route | 3-leg crossing to the Crown; light the landing signal (needs relay B19 active) | Landing signal lit (`crown_signal`) | 250 m | 3.5 min |
| B21 | Repair another double-ring relay | New layout; no service explanation | Relay B21 active | 80 m | 2.0 min |
| B22 | Restore an older single-ring station | Uncharged v1 unit; unnecessary drain is safe but logged | Relay B22 active | 110 m | 2.0 min |
| B23 | Recover the summit beacon heart | Approach relay (v2) powers the spire vents; reliquary sentinel; its defeat opens the reliquary | Heart in pack | 100 m | 4.5 min |
| B24 | Commission a distant outpost from saved memory | Outpost sentinel at the Crown-edge pad (dormant until the heart is in the pack); its defeat powers the 2-leg vent route; install the heart; seeded-version beacon | Beacon active, heart installed, sentinel defeated | 250 m | 5.0 min |
| | **Total** | | | **≈ 7.3 km** | **≈ 73 min** |

At the 0.2 s decision interval, 73 minutes is about **22,000 actions**. The target is
60–120 minutes.

**Calibration caveat.** Task A's route preview needed 18 minutes of pure travel for 6.2 km,
which is faster than the planning speed of 4.5 m/s. The measured length of B may therefore
come in shorter than this table. The final budgets come from reference rollouts.

### How the agent learns what to do next

| After | Lead source | Lead content (paraphrased) |
| --- | --- | --- |
| Reset | Constant goal; the keeper stands in view of spawn | Talk to the relay keeper. |
| B01 | Keeper | The workshop relay by the mill keeps tripping. Fix it yourself. |
| B02 | Keeper | The sentinel warden trains at the ring arenas west of the village. |
| B03 | Warden | The second arena's sentinel reforms its shield once. Clear it. |
| B04 | Warden | The wind guide waits by the sea stacks east of the village. |
| B05 | Guide | Reach the second overlook on your own. |
| B06 | Guide | The expedition quartermaster waits at the depot: the granary with the crane, south-east of Mirror Lake. |
| B07 | Quartermaster; compartment plaque | The reserve compartment is sealed until reserve power returns from the upper valley. The causeway mechanic at the foot of the Scarp Cut needs a repair kit, which raiders in the eastern hills stole, and the ferry coupling, which sank in Mirror Lake. The boatman on the lake's east shore saw which chest it went down in. |
| B08, B09 | Item descriptions | "For the causeway mechanic." |
| B10 | Mechanic | The valley signal station up the cut has been dark. |
| B11 | Station plaque | The observation post on the eastern stacks reports the route north. |
| B12 | Observer | A field sentinel in the western hollow holds the warden's field seal. |
| B13 | Item description; observer | The seal opens the reserve relay's service panel near the eastern rim. |
| B14 | Relay plaque | Reserve power restored; the depot compartment is open. (No location is repeated.) |
| B15 | Item description | Tools for the sky-lift control house at the foot of the northern wall. |
| B16 | Plaque at the top | The crater gate to the north is shut. |
| B17 | Gatekeeper's plaque | The high shrine above the crater floor keeps the court's seal. |
| B18 | Item description | Opens the shrine-court relay housing. |
| B18 (at the housing) | Service plate | Double-ring inlets keep a residual charge. (Read on inspection; also the diagnostic for priming first.) |
| B19 | Shrine plaque | The Crown above the crater is dark. Its landing signal waits beyond the wind stacks. |
| B20 | Landing signal | The Crown's relays wake. The east relay is a double ring. |
| B21 | Crown plaque | The old station on the Crown's south side is dark. |
| B22 | Station plaque | The summit-approach housing is open. The heart lies in the reliquary on the spire. |
| B23 | Item description | The beacon heart, for the outpost beyond the Crown's north edge. |

Dialogue stays in `local_text` while the agent is within about 4 m of the speaker, and for
5 s afterwards. Working-memory cues (the boatman's mark, the mechanic's gate pattern) are
stricter: they stay only while the agent is within 4 m **and** for at most 5 s, and do not
return without a new interaction. Mechanism feedback lingers 8 s, but the cues are never
put in it.

### Working-memory probes

B09 and B10 carry two probes from the shared [working-memory probe
contract](working-memory-probes.md). Each is an extra physical requirement of the
milestone; every existing predicate still applies and the world (terrain, gates, structures,
proofs) is unchanged. The new fixtures are entities without collision, placed by offsets
from existing anchors.

| Probe | Source and cue | Use site | Load, kind |
| --- | --- | --- | --- |
| `WM-TB09` | Lake boatman, 66 m east of the lake centre on the shore: "The coupling is in the one marked *a-b-c*" (three of eight glyphs) | Six sunken chests in a 9 m ring on the lake bed; a chest's mark is readable only within 4 m. Distractor marks share one or two glyphs, in place, with the target | 3, select (one slot; the first chest opened is the first attempt; a wrong chest is empty) |
| `WM-TB10` | Causeway mechanic, on delivery of kit and coupling: a pattern of four of six elements, "set it on the winch at the far end of the causeway in reverse order" | Causeway winch with six element pawls, beyond the top of the Scarp Cut | 4, reverse |

Pawls and chests give neutral feedback ("Winch pawl (tide): click (2 set).", "Sunken chest:
empty."). Only a complete winch entry is judged; a wrong one drops the pawls back, and a
partial entry drops back when the agent walks more than 12 m away. The winch is chained
until the mechanic has given the pattern. The boatman and the mechanic repeat the cue
whenever they are asked while the probe is unsolved. `TeyvatExpeditionEnv(wm_control=True)`
also posts each cue within 12 m of its use site (diagnostic only).
`TeyvatExpeditionEnv(working_memory=False)` rebuilds the earlier course, version
`teyvat-fullscale-b-course-v2`, whose initial state digest matches the archived seed-7 run.
The course with the probes is `teyvat-fullscale-b-course-v3-wm`. Probe state is in the state
digest; events and summaries follow the contract.

| Probe | Source (x, y, z) | Use site (x, y, z) | Distance |
| --- | --- | --- | ---: |
| `WM-TB09` | Boatman (258.9, 0.6, 479.8) | Chest ring centre (196.9, −8.0, 482.8) | 62.1 m |
| `WM-TB10` | Mechanic (45.2, 9.5, 326.0) | Winch (38.2, 57.5, 156.0) | 170.1 m |

The winch is beyond the 144 m range at which the policy renderer draws entities, so the
mechanic is not drawn from it. The lake is 124 m across, so the boatman can be in view from
the lake surface; his words are not, and a chest's mark cannot be read from the shore.

### Progression rules

These follow Task A:

- **Receipts.** Receipts are issued strictly in B01–B24 order.
- **Early work.** Work done early persists and is credited when the chain reaches it. For
  example, the kit or coupling taken early, or raiders already defeated.
- **Later-milestone objects.** An object that belongs to a later milestone answers with
  diegetic feedback, never with step numbers.
- **Training units.** The keeper, warden, and guide each power their unit one at a time,
  so B02, B04, and B06 follow their lessons.
- **Crown relays.** These stay dormant until `crown_signal`. The B19 adaptation therefore
  always comes first.

### Memory schedule

| Exposure | Probe | Estimated delay | Intervening work |
| --- | --- | --- | --- |
| R, B01–B02 (ends at about 4 min) | B11 (starts at about 27 min) | ≈ 23 min, ≈ 6,900 actions | G and T training, depot, camp, dive, causeway |
| G, B03–B04 (ends at about 9 min) | B13 (starts at about 33 min) | ≈ 24 min | T training, B07–B12 |
| T, B05–B06 (ends at about 14 min) | B12 (starts at about 30 min) | ≈ 16 min | B07–B11 |
| Depot, B07 (about 16 min) | S1, B15 (starts at about 38 min) | ≈ 22 min | B08–B14 |
| Double-ring update, B19 (about 56 min) | B21 (about 63 min), B23, B24 | ≈ 7–17 min | B20 crossing |
| Single-ring rule (last v1 at B17, about 51 min) | B22 (about 65 min) | ≈ 14 min | B18–B21, which include the v2 update |

Fresh agent contexts start before **B11, B21, and B24**, preserving only the memory the
evaluation condition allows. The environment emits `designed_context_boundary` events at
those points.

## Spatial memory probe

| Probe | Place | Learned | Used | Withheld at use |
| --- | --- | --- | --- | --- |
| S1 | Depot granary (lowland) | Registered at B07 | B15, after the B14 plaque | Location in the lead; the compartment glyph is readable only within about 10 m |

The metrics are the same as Task A's: path efficiency, initial heading error, time to
target, assisted recall, and unaided success. Evaluation runs in both map-off and map-on
conditions, and the primary spatial score comes from map-off.

## Observation, actions, reward, and termination

These are as in Task A (campaign-v1 contract): `rgb`, `proprio`, `party`, the constant goal
as `instruction`, and `local_text`.

`local_text` carries:

- the pack;
- the current dialogue or plaque;
- mechanism feedback;
- up to five visible labels within 24 m.

Cues name what an agent could see, never the hidden rule:

- **Relay receivers** report their role marking (source, vane, drain, core), their progress
  lamps, the overload flag, and a ring badge (single or double).
- **Sentinels** report an emblem *glyph*, not its element, and their phase posture
  (shielded, open core, winding up, cracked core).
- **Vents** report a family glyph, charged or uncharged, and the current beat (bright or dark).

The element-to-glyph tables are private.

- **Actions:** native only. There is no navigation macro, teleport, or fast travel.
- **Reward:** +1 at B24. Receipts go to `info` only. `info` is privileged diagnostics
  (position, facts, milestone index) for logging and replay; a harness must not pass it to
  the agent. Its `instruction` field carries the constant goal, not the current milestone.
- **Death:** respawn at the nearest discovered statue, with facts and the pack preserved;
  deaths are counted. Statues are at the village, the depot, the Scarp Cut top, the
  sky-lift top, inside the crater gate, and the Crown landing. Interacting with a statue
  heals the party.
- **Budget:** `max_steps` is provisionally 90,000, about 4× the estimate. It will be set
  from the reference rollouts.

## Reference solution and walkthrough

The reference controller is privileged. It reads anchors, receiver roles, the sampled
assignments, sentinel phases, and vent beats. It acts only through native actions:

- camera-relative movement, look, sprint, and jump/glide;
- interact, attack, skill, and party swap.

It plans on the phase-1 style `under` graph with clearance, as in Task A's route preview.
It adds:

- **Relays.** Approach each receiver until it is within skill reach; swap to the
  character with the needed element; cast; confirm at the core.
- **Sentinels.** Hold a stance about 3 m from the core, inside Kestrel's 4 m reach. Swap and
  cast the breaker. Pre-swap to the interrupter during the exposed phase, and cast it on
  wind-up. Attack in the vulnerable phase. Step out of the 6 m warding pulse while skills
  cool down.
- **Vents.** Rest to full stamina; charge; jump on the safe beat; steer the glide to the
  shelf; rest again.
- **Dive.** Hold jump while swimming above the chest, and keep holding it while
  interacting at the bottom; releasing it makes the character bob up.

Every collected run is replayed in a fresh environment, and positions, events, rewards,
and the state digest must match after every action.

The walkthrough follows the Task A route-preview and Skylands formats:

- a 1600 × 900 canvas around a 1280 × 720 Blender Cycles scene (OptiX, 64 samples);
- timelapse travel (walking 16×, glides 4×, mechanisms 2–4×) with event holds;
- a 24-milestone list, a route map, captions, and recorded settings;
- decode checks.

It is labeled **REFERENCE SOLUTION — privileged controller**.

## Implementation plan

New modules keep Task A and the existing suites untouched. Task A's phase 2 can later
reuse the mechanics module.

| Module | Role |
| --- | --- |
| `open_world_bench/fullscale_b_world.py` | B island authoring: tiers, Scarp Cut, crater ring and gate, stacks, lake, structures, vents, and the world digest |
| `open_world_bench/fullscale_reach.py` (extend) | Vents as launch sources in both models; world-provided site lists for the gate checks. Task A's results must be unchanged |
| `open_world_bench/fullscale_mechanics.py` | Relays (v1/v2), sentinels with phases and damage, vents with charge/beat/launch, items, locks, leads, receipts. Written to be reusable by Task A's phase 2 |
| `open_world_bench/fullscale_b_env.py` | `TeyvatExpeditionEnv`: course, observations, digest, statues, context-boundary events |
| `open_world_bench/fullscale_b_policy.py` | Privileged reference controller |
| `owb/scripts/teyvat_fullscale_b.py` | `world-check`, `collect`, `verify`, `render` (sharing the preview script's Cycles farm and overlay) |
| `tests/test_teyvat_fullscale_b.py` | World proofs, mechanics unit tests, locks, privacy, a full replayed run |

| Phase | Work | Exit criterion |
| --- | --- | --- |
| B1 ✅ | World and vent proofs | Every gate proof passes for seed 7 (also 3, 11, 19, 23); vent destinations are unreachable with vents disabled; one real-controller flight per leg lands |
| B2 ✅ | Mechanics, course, leads, locks, observations | Unit tests for R v1/v2, G phases and damage, T charge/beat/launch, locks, early credit, privacy |
| B3 ✅ | Reference controller, collect and verify | Seed 7 completes 24/24 without deaths; replay matches every action |
| B4 ✅ | Cycles walkthrough | Playback-verified MP4 with chapters and recorded settings |
| B5 | Registration, seed 19, measured lengths, evaluation harness | `make_env` smoke run; this document updated with measured numbers |

## Validation checklist

- **Gate proofs** for every row of [Gates and locks](#gates-and-locks), for seeds 7 and 19.
- **Reference run.** It completes B01–B24. Actions are native only, with no writes to
  simulator state. An independent replay matches every action.
- **Lock behavior:**
  - the reserve compartment stays sealed until B14;
  - the service panel stays shut until the field seal;
  - the shrine housing stays shut until the shrine seal;
  - Crown relays stay dormant until `crown_signal`.
- **Early credit:** a kit or coupling taken before its milestone is credited when the
  chain reaches it.
- **Observation privacy:** no step text, coordinates, element tables, or receipts.
- **Leads:** every milestone's lead is delivered in the world before its first required
  action, and the B14 lead contains no directions.
- **Damage:** sentinel slams and pulses can be avoided by the correct procedure. The
  reference never dies and its minimum HP is recorded.
- **Vent use** is recorded per route, together with any shortcut the reference or the
  proofs find.
- **Measurements:** delays, total length, and walking share are compared with the estimates.

## Defaults chosen for review

| Decision | Default | Alternative |
| --- | --- | --- |
| Shared API task name | `expedition` | `b` |
| Island | Separate island from Task A | Reuse Task A's island with new sites |
| Crown access | T-gated only (no fact); the landing signal needs relay B19; Crown relays wake one at a time | An extra gate fact at the landing |
| Sentinel damage | Pulse 8 within 6 m; slam 26 within 9 m | Slam only |
| Vent launch | Kinematic rise to the column top on the safe beat | Physical updraft force while gliding |
| Outpost beacon version | Seeded v1 or v2, visibly badged | Always v2 |
| Death | Respawn at the nearest discovered statue; counted | Terminate |

## Implementation results (seed 7)

Status (2026-11-24): the results, trace and walkthrough video in this section predate the
working-memory probes `WM-TB09` and `WM-TB10` (see [Working-memory
probes](#working-memory-probes)). They were made on course `teyvat-fullscale-b-course-v2`,
which `TeyvatExpeditionEnv(working_memory=False)` still builds. A new seed-7 reference with
the probes, `teyvat-fullscale-b-course-v3-wm`, verified by an independent replay, is in
`assets/evaluation/working-memory/teyvat/fullscale-b/seed-7/` (`manifest.json`,
`actions.jsonl`, `verification.json`). It has not been rendered. It completes 24 / 24 in
8,364 native actions (27.9 simulated minutes, 8.99 km, no deaths or falls, lowest HP 70,
17 vent launches, no sputters):

| Probe | First attempt | Cue reads | Delay from cue to first attempt |
| --- | --- | ---: | --- |
| `WM-TB09` | correct | 1 | 118 decisions, 23.6 s |
| `WM-TB10` | correct | 1 | 164 decisions, 32.8 s |

The privileged controller reads each cue from private state; its correct first attempts
show the probes are solvable and say nothing about memory. Each cue was in `local_text`
for 3 observations (0.6 s) before the controller left reach. The world proofs are unchanged:
the probe fixtures add no terrain or collision. The reference controller gained a corner
escape: when it is stuck on ground the solid-terrain controller will not leave in the
planned direction, it steps out for six actions along a heading the controller accepts, then
replans. Without it, the run stalled for good at the foot of the Scarp Cut after B10's
exchange once the B09 route had changed. In the recorded run it fired three times (in B04,
B08 and B10).


| Module | Role |
| --- | --- |
| `open_world_bench/fullscale_b_world.py` | Island authoring, stacks, gate boxes, vents (column tops sized from the `under` glide) |
| `open_world_bench/fullscale_b_check.py` | Gate proofs per fact stage, vent-free proofs, and one real-controller flight per vent leg |
| `open_world_bench/fullscale_mechanics.py` | `RelayUnit`, `SentinelUnit`, `VentState`: the R, G and T rules |
| `open_world_bench/fullscale_b_env.py` | `TeyvatExpeditionEnv` and `ExpeditionCourse`: 24 milestones, locks, leads, statues, observations |
| `open_world_bench/fullscale_b_policy.py` | `ExpeditionReference`, the privileged reference controller |
| `open_world_bench/fullscale_sites.py` | Site compounds (colliding pieces placed around anchors) and zoned vegetation |
| `open_world_bench/render/site_assets.py`, `anime_assets.py` | Presentation meshes: site pieces, plants, far peaks, the player's character, people, sentinels |
| `owb/scripts/teyvat_fullscale_b.py` | `world-check`, `collect`, `verify`, `render` (step-range passes can share one frame cache) |
| `tests/test_teyvat_fullscale_b.py` | Rule units, locks, observation and `info` contract, same-step milestones, sentinel deaths, reaction splash, vent launch detection, a real-controller check of `strict_carry`, world proofs, and a full collect plus replay |

`fullscale_reach.reach` gained vents as launch sources and an opt-in `strict_carry`
mode, which Task B's proofs use. In the default mode a climb ceiling could travel along a
connected face indefinitely, through repeated drops from holds. On this island that let a
lookout-stack ceiling leak along the Great Scarp and the coastal cliffs up to the highland.
`strict_carry` changes three things. All three stay optimistic, so the proofs remain sound.

- **Gliding off a face.** A climber who leaves a hold at height `h` with ceiling `C` carries
  `min(C, h + budget)` with the glide. That value loses height exactly as the glide does,
  so it stays tied to one path. The default mode instead mixes the highest arrival and the
  largest ceiling, which may come from different holds.
- **Carried values.** They spread wherever any glide can pass, not only where the arrival
  height improves.
- **Along faces.** The ceiling loses half a glide step (0.14 m per metre) per patch step.
  This is a modelling charge, not a real cost: moving sideways on a face costs only
  stamina. At full stick a climber rises and moves sideways at the same time; at a stick of
  0.1 or less it pays only 0.35 of the climb cost and still moves sideways at full speed,
  about 60 m on one bar. The charge is sound because the climb budget's 1.25 margin covers
  it over every split of one stamina bar between those two ways of climbing.
  `strict_face_slack` computes the worst case from the character parameters (5.1 m for the
  default party), and `reach` refuses strict mode if it is not positive. This is one
  multi-source shortest path per iteration, with ceilings rounded up to 5 cm so that the
  fixed point is reached.

Task A's evidence keeps the default mode; its 17 world tests are unchanged.

**World proofs, seed 7** (`world-check`, 52 s with 9 processes; report, with source hashes, in
`runs/fullscale-b/world-7/report-7.json`):

| Stage | Sealed under `over` | Highest `over` point | Opened sites reachable under `under` |
| --- | --- | ---: | --- |
| start (no facts) | Plateau, highland, crater, Crown, spire, and all upper stacks | 65.6 m | All lowland sites; training stacks by vent |
| `causeway_open` | Highland, crater, Crown, spire, K1, crater stacks, outpost | 108.6 m | Upper-valley sites, lookout post by vent |
| `skylift_power` | Crater floor, Crown, spire, crater stacks, outpost | 139.2 m | Highland sites, crater gate |
| `crater_gate_open` | Spire top, outpost | 183.5 m | Crater, shrine, Crown and Crown relays |
| `summit_power` | Outpost | 196.6 m | Spire top |
| `outpost_clear` | (none) | 196.6 m | Outpost stack |

With every fact open but vents disabled, starting in the lowland, the upper valley, or on
the crater floor, no stack top, the Crown, the spire, or the highland is reachable under
`over`. All 17 vent legs were flown with the real controller from their column tops: each
landed on its shelf without damage, with 11.8–60.5 stamina to spare (the B06 level leg is the
tightest). The same proofs and flights pass on seeds 3, 11, 19 and 23.

**Reference runs.** Seed 7 is the recorded run; seed 19 checks that the reference holds on
another layout.

| Measure | Seed 7 | Seed 19 |
| --- | --- | --- |
| Result | 24 / 24 milestones, success | 24 / 24 milestones, success |
| Actions | 8,324 native actions | 7,936 native actions |
| Simulated time | 27.7 minutes | 26.5 minutes |
| Travel | 8.96 km | 8.76 km |
| Deaths | 0 | 0 |
| Falls with damage | 0 | 0 |
| Lowest HP | 70 | 68 |
| Vent launches | 17, one per leg, no sputters | 17, one per leg, no sputters |
| Route restarts | 0 | 0 |
| Relay applications rejected | 0 | 0 |
| Unnecessary drains at B22 | 0 | 0 |
| Relay interruptions handled | 3 (B02, B23, B24) | 3 (B02, B23, B24) |
| Sentinel interrupts | 11 | 9 |
| Missed wind-up windows | 0 | 0 |
| Slams | 0 | 0 |
| Replay | Every action matches | Every action matches |

Replay is an independent re-run in a fresh environment that compares positions, events,
rewards and the state digest after every action.

Action share by controller mode:

| Mode | Share |
| --- | ---: |
| Walking | 54 % |
| Vent routes (approach, rest, rise, glide legs) | 32 % |
| Descents by glider | 5 % |
| Swimming | 3 % |
| Combat | 5 % |
| Other mechanisms | 1 % |

**Measured delays against the memory schedule** (measured before the scenery pass below, which lengthened the seed-7 run by 0.3 minutes; not re-measured):

| Exposure | Probe | Planned delay | Measured delay |
| --- | --- | ---: | ---: |
| R (ends B02, 0.7 min) | B11 (8.8 min) | ≈ 23 min | 8.0 min |
| G (ends B04, 2.2 min) | B13 (11.7 min) | ≈ 24 min | 9.5 min |
| T (ends B06, 5.3 min) | B12 (9.6 min) | ≈ 16 min | 4.4 min |
| Depot (B07, 6.0 min) | S1 at B15 (13.8 min) | ≈ 22 min | 7.8 min |
| Double ring (B19, 23.2 min) | B21 (24.3 min) | ≈ 7 min | 1.1 min |
| Single ring (B17, 21.0 min) | B22 (24.7 min) | ≈ 14 min | 3.6 min |

**The reference is much shorter than the estimate.** It took 27 minutes against an estimate
of 73 minutes and a target of 60–120 minutes.

- The privileged controller sprints, never hesitates, and needs one pass per mechanism, so it
  covers 8.9 km at about 5.4 m/s.
- A learned agent will be slower, but the designed delays are only about 35 % of plan when
  measured with the reference.

If reference-measured delays should meet the schedule, the island has to grow, or the
intervening work (B07–B10, B13–B16) has to get longer. These are design decisions and have
not been made.

**Walkthrough.** The walkthrough follows the route-preview format, labeled *Reference
solution*. Vent pads, vent columns, and sentinel phase rings are presentation cues; gate
boxes match their collision. It plays walking and resting at 16×, and vent legs, glides,
and mechanisms at 4×.

The seed-7 video renders 2,283 sampled replay states of the run above (2,903 frames, 4:50). It is a
1600 × 900 canvas around 1280 × 720 Blender 4.5.4 Cycles scenes (OptiX, 64 samples,
denoised); its frame count, duration, settings and decode check are recorded in
`video-manifest.json`. Rendering runs as four step-range passes sharing one frame cache;
each pass replays from step 0 and checks the state digest at every step.

```bash
python -B -m owb.scripts.teyvat_fullscale_b world-check --seed 7 --output runs/fullscale-b/world-7
python -B -m owb.scripts.teyvat_fullscale_b collect --seed 7 --output runs/fullscale-b/seed-7
python -B -m owb.scripts.teyvat_fullscale_b render runs/fullscale-b/seed-7 \
  --cache .cache/teyvat-fullscale-b/seed-7 --gpus 0 1 --per-gpu 2
```

**Known limits of this pass:**

- **Scope.** The recorded run and the video are for seed 7. World proofs pass on seeds 3, 7,
  11, 19 and 23, and the reference completes seeds 7 and 19; other seeds' reference runs are
  untested.
- **Geometry.** The terrain is still authored geometry; shaded cliff faces read as flat.
- **Sentinels.** In observations they reuse the generic enemy figure with phase rings; the Cycles presentation has its own sentinel model.
- **`local_text` labels.** Labels are not occlusion-tested.
- **Elevated receivers.** Elevated relay receivers are deferred.
- **Crown shortcuts.** Inside the crater, some vent columns can reach a later shelf by gliding
  and then climbing. The proofs only require each vent-only place to be unreachable with
  vents disabled, so these shortcuts are logged as valid shortcuts, not scored as T reuse.

## Review changes

An independent review (Fable 5.2) of the first draft found three blockers and a set of
should-fix items. This version resolves them as follows.

| Finding | Resolution |
| --- | --- |
| Unpowered outpost columns could reach the spire face before `summit_power` | Outpost vents are powered by `outpost_clear`, the outpost sentinel's defeat |
| `crown_signal` could be lit before the B19 adaptation | The landing signal needs relay B19 active |
| B24 did not enforce G then T then R | The outpost sentinel is dormant until the heart is in the pack, and its defeat powers the vents |
| B17 did not enforce G then R | The gate relay's housing opens on the gate sentinel's defeat; the reliquary likewise opens on its sentinel's defeat |
| B22 could be done before B21 | Crown relays wake one at a time: B21, then B22, then the B23 housing |
| Spacing rule used the wrong climb budget and no glide cap | The rule now uses `air_climb_budget` (26.4 m) and states the 97-step cap |
| Columns sized from the real glide fail the `under` vent model | Columns are sized from the `under` ratio (0.4706 m per metre) plus 6 m |
| An 80 m lowland overlook could reach the Great Scarp | Lowland stacks are capped at 65 m and kept at least 300 m from the scarp |
| Launch overshoot past the apex | The launch stops at the column top with zero vertical velocity |
| T's stamina condition was never binding | Every uncoached route has one 80–90 m level leg |
| Seeded start phases expire before the agent arrives | A non-shield start phase holds until the agent is within 12 m |
| Brute-forcing elements on sentinels was free | Wrong elements trigger a pulse (shield) or the slam (wind-up) |
| A 4–5 m stance is outside Kestrel's 4 m reach | Stance is about 3 m |
| Updraft and vents interacted ambiguously | Updraft can charge a gale vent but never counts as the launch |
| Plinths cannot be shaped at 2.5 m cells | Elevated receivers are deferred; receivers stand on level ground |
| S1 return was a straight walk from the Scarp Cut | The depot moved off that axis, to (300, 665) |
| Leads that supplied a decomposition or the scope answer | The B16 and B21 leads now give only state, not a recipe |
| B23 interruption on an untaught branch | Moved to "after priming", the taught branch |
| Crater geometry missing; outpost overlapped the ring; totals off | Geometry stated; outpost moved to (-90, -585); totals corrected |

Scope for the first implementation pass follows the review: seed 7 only, fixed start
phases, no plinths, and the walkthrough rendered after replay verification. Kept without
exception:

- `over` proofs with vents disabled, and for each fact;
- one real-controller flight per route;
- sentinel damage;
- replay verification;
- renderer and settings labels.

## Implementation review

A second independent review (Fable 5.2) read the implementation in commit 649347d and
probed it with the real controller: about 24,000 differential climbs against the strict
`over` model on seeds 7 and 19, gate pushes on both gates, `world-check` on seeds 3 and
19, a seed-19 reference run, and targeted reproductions. It found the terrain seals and
`strict_carry` sound, and one blocker and two should-fix holes in the course logic. All
findings are resolved as follows.

| Severity | Finding | Resolution |
| --- | --- | --- |
| Blocker | Two milestones completing in one step (allowed by early credit, for example B08 and B09 when the coupling is fetched before the strongbox) left the quest engine a stage behind for good: no +1 and no termination | `ExpeditionQuests` advances through every stage completed in a step. Regression test |
| Should-fix | Sentinel pulses and slams, and wrong-element casts, landed after the base defeat check, so a downed member ended the episode instead of swapping out or respawning | Defeat is handled after the course tick, after casts, and after attacks. Regression test |
| Should-fix | Reaction splash bypassed the sentinel rules and could damage, even kill, an inert or shielded sentinel | Sentinels take damage only through the phase rules. Regression test |
| Should-fix | The doc justified the strict face charge as a climb cost; moving sideways on a face costs only stamina | Justification rewritten. `strict_face_slack` checks the bound from the character parameters (5.1 m of slack), and `reach` refuses strict mode without slack. A real-controller differential test was added |
| Should-fix | Test gaps: same-step milestones, sentinel deaths, launch detection, privacy of `info`, strict-carry soundness | Tests added for each |
| Minor | `under` vent glides were capped by step count, so a diagonal could exceed one stamina bar | Capped by path length (a height floor per vent) |
| Minor | `stack_b24a` was missing from the staged seal lists | Added. Seed 19 then showed it reachable under `over` from the B20 crown vent with zero margin, so the stack and the outpost were both raised 6 m (180 m and 196 m) |
| Minor | `info["instruction"]` carried the current milestone title | It carries the constant goal; `info` is documented as privileged diagnostics |
| Minor | The sputter's "2 m hop" was a no-op | Removed; a sputter is an ordinary jump that spends the charge |
| Minor | Doc versus code: launch count, B23 start state, lowland stack tops, shrine, K1, lookout, flight method, B05 predicate, depot | Doc corrected |
| Minor | The reference took four slams on seed 19 (Kestrel's strike is a forward cone, cast without facing the core) | The reference faces the core before casting at a sentinel |
| Minor | Provenance: source hashes omitted shared simulation files; the world report recorded none; the digest omitted yaw, regeneration timer, auras and cooldowns; relay role orders collided for B02/B11 and B14/B23 | Hashes widened and recorded; digest extended; role orders seeded by CRC-32 (`COURSE_VERSION` v2) |

Found while validating the fixes on more seeds:

- The sky-lift's second vent leg was 114 m, longer than the 80–90 m the spacing rules
  intend. The real flight landed with 0.7 stamina on seeds 7, 19 and 11, and with fall
  damage on seed 23. Stack K1 moved 35 m north, which splits the route into two legs of
  about 80 m; the tightest flight on any proven seed now keeps 6.5 stamina (11.8 on seed 7).
- The reference stopped steering 2.5 m from a shelf's centre. A released glider keeps its
  horizontal speed, so it overshot, landed on a stack's sloped edge twice on seed 7, slid
  off, and flew the leg again. It now steers its velocity toward a target that shrinks near
  the centre (which also brakes), and every leg lands within 1.9 m of the centre on the
  first launch.

Kept as known limits:

- A climber on the steep part of a patch classified as walkable is outside the strict
  spread. The differential climbs found 19 such states and none led to a violation.
- The environment builds any seed. Registration (phase B5) should expose only seeds whose
  `world-check` passed.

## Scenery and presentation pass

A later pass changed how the island looks. It touches the simulated world in three places,
so the proofs and reference runs above were redone on the new world (`world-check` passes
on seeds 3, 7, 11, 19 and 23; the reference completes seeds 7 and 19 with replay).

**Simulated world (changes collisions and the world digest):**

- **Site compounds.** Each site anchor gets a small compound of colliding pieces
  (granary, storehouse, workshop, well, fences, palisades, tents, standing stones, masts,
  pylons, shrine hall, crates, carts, ruins). `SiteArchitecture` adds their boxes to the
  shared building collisions; roofs are never platforms. A piece that does not fit swings
  around its anchor in 20° steps, and a piece that still does not fit is skipped and logged.
- **Vegetation.** Plants sit on a jittered 7 m grid, thinned by a grove mask, with a plant
  table per zone (orchard, lakeside, hills, plateau, highland, crater, crown). Anchors,
  compounds, trails, gate passages and stack tops are kept clear.
- **Irregular walls and stacks.** The Great Scarp and the highland wall follow a rougher
  line away from the Scarp Cut and the coasts, and stacks bulge into lobes of up to 16 %.
  A stack whose neighbour is close keeps a round plan, so no lobe covers a vent pad.

**Presentation only (Cycles recordings; observations and simulation unchanged):**

- An opt-in style, recorded as `style: anime` in the video manifest: toon shading mixed
  over the physical shader, terrain coloured from slope and height, faceted rock shading,
  a procedural cloud sky, the Standard view transform, haze, a lawn within 60 m of the
  camera, and ink outlines on figures.
- Original stylised models for plants (three detail tiers, 900 plants per frame), the
  player's character with pose animation and a glider, other people, and sentinels, plus
  presentation-only mountains out at sea.
- The observation renderer draws the site pieces but keeps its own plant and figure models.

**Known gaps in the look:** relays, vents, gates, chests and statues keep their earlier
models; there are no skill, hit or downed animations; ground paths are not drawn; water is
only recoloured; shaded cliffs have no geometric relief.
