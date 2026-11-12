> Legacy design: retained for the 12/24-step reference courses and existing artifacts.
> The current branching suite is documented in [Teyvat campaigns](teyvat-long-horizon-tasks.md).

# Teyvat: long-horizon memory and reusable skills

Status: the full task suite remains a design; [compact executable reference
courses and demonstration videos](teyvat-reference-demos.md) are now available.
The two tasks
retain exactly **12** and **24** milestones in the Genshin-inspired environment.
A milestone is a verifiable accomplishment requiring many actions, not a button
press or a claim that the agent has learned something.

The central question is: **Can an agent turn experience into a useful procedure,
retain it through intervening work, and retrieve and adapt it in a new setting?**
Remembering a location remains a secondary measure. The main tests require
conditional action selection, binding procedures to new objects, recognizing
when a procedure applies, and recovering from interruptions. Learning here means
adaptation through agent-owned memory with model weights held fixed.

| Task | Milestones | Hypothesis to test |
| --- | --- | --- |
| A: The Relay Keeper's Apprenticeship | 12 | One procedure can be retained and applied to new layouts, initial states, and interruptions after a long delay. |
| B: The Expedition of Three Arts | 24 | Several procedures can be retained, selected, composed, and revised while preserving older knowledge that remains valid. |

## Milestone graphics

Task B's complete 24-node graph shows the ordered chain, learning phases, skill
combinations, and fresh-context checkpoints on one sheet:
[PNG](../assets/design/teyvat-milestones/task-b-milestone-graph.png) ·
[editable SVG](../assets/design/teyvat-milestones/task-b-milestone-graph.svg) ·
[PDF](../assets/design/teyvat-milestones/task-b-milestone-graph.pdf).
Regenerate with `python -B assets/design/teyvat-milestones/render_graph.py`.

Each illustrated card shows the action, completion condition, and memory role.
Colors label learning, intervening work, transfer, composition, and adaptation;
fresh-context checkpoints are explicitly marked. These are designer-facing
diagrams, not observations or solution hints supplied to the agent.

| Sheet | Coverage | Exports |
| --- | --- | --- |
| Relay Keeper's Apprenticeship | A01–A12 | [PNG](../assets/design/teyvat-milestones/task-a-01-12.png) · [SVG](../assets/design/teyvat-milestones/task-a-01-12.svg) |
| Expedition of Three Arts, first half | B01–B12 | [PNG](../assets/design/teyvat-milestones/task-b-01-12.png) · [SVG](../assets/design/teyvat-milestones/task-b-01-12.svg) |
| Expedition of Three Arts, second half | B13–B24 | [PNG](../assets/design/teyvat-milestones/task-b-13-24.png) · [SVG](../assets/design/teyvat-milestones/task-b-13-24.svg) |

[Printable three-page PDF](../assets/design/teyvat-milestones/teyvat-milestone-storyboards.pdf)
· [Full graphics bundle, including 36 individual cards](../assets/design/teyvat-milestones/teyvat-milestone-graphics.zip).
The bundle also includes editable SVGs, the caption manifest, and the renderer.
Regenerate from the repository root with
`python -B assets/design/teyvat-milestones/render.py` (Matplotlib and Pillow).

## What counts as a reusable skill

Structure the tasks around **experience -> abstraction -> delay -> retrieval ->
transfer -> revision**. Memory writing is permitted throughout; it is neither a
rewarded milestone nor proof of learning. Infer competence from behavior on
previously unseen encounters.

A useful stored procedure could include applicability, visible preconditions,
object roles, conditional actions, a success signal, recovery, and supporting
experiences. For example:

> On a single-ring relay, inspect charge and progress. If overloaded, discharge
> the drain. Prime the source and route the vane with the family's learned
> elements, skipping stages already complete. Confirm at the core. If
> interrupted, inspect again and resume from the surviving state.

This is an illustrative description, not a supplied agent skill. The agent must
derive its own memory from observation. Fixed coordinates and button sequences
will not solve every instance. No scripted macro action is granted after
training: the agent continues using movement, interaction, party swaps, and
elemental controls.

## Procedures to acquire

Start with Kestrel, Maren, and Wren: ember, tide, and gale are all available.
The receivers, guardian phases, and vent trials below are new authored mechanics;
existing combat and traversal provide foundations, not complete implementations.

### R: Diagnose and commission an elemental relay

Each relay has a source, vane, drain, and core with consistent visible role
markings. Their positions, identities, orientation, and appearance vary across
sites. Indicators expose unprimed, primed, routed, or active progress, plus an
independent overload flag.

For a **single-ring** relay:

1. Inspect progress and overload.
2. If overloaded, apply the discharge element to the drain.
3. If unprimed, apply the priming element to the source.
4. If primed but not routed, apply the driving element to the vane.
5. Once routed and not overloaded, interact with the core and verify activation.

Sample the element-role assignments as a permutation of ember, tide, and gale.
They remain stable within a task's family and are taught through demonstrations
and practice. Different task instances can have different assignments; ordinary
combat and movement physics remain consistent.

Wrong applications produce categorical feedback and a reversible fault.
Draining clears overload without erasing completed stages. A scripted
interruption can introduce overload after priming; recovery requires inspecting
retained state. Teach this case before testing it. Do not reward partial action
prefixes during transfer.

Task B introduces visibly **double-ring** relays, which additionally require
initial drainage even when not overloaded. At the adaptation encounter, an
inspectable service indicator explains that the inlet retains residual charge.
Trying to prime first produces that diagnostic. The correction can be inferred
without forcing an error. Single-ring rules remain unchanged; unnecessary
drainage is safe but costs actions. This tests whether an update is stored with
the right scope.

### G: Expose and interrupt a sentinel

Learn to inspect a shield emblem, apply its learned shield-breaking element,
wait for the visible exposed phase, interrupt its wind-up with the learned
second element, and finish with ordinary attacks. If its shield reforms, return
to diagnosis. If already exposed, skip shield breaking.

Sample valid emblem/element assignments within an authored sentinel family and
demonstrate every required cue/action relation. Later sentinels change arena,
appearance, health, orientation, and starting phase while retaining those rules.
The procedure exposes a damageable core, preventing brute force against the
shield from bypassing the trial. Ordinary enemies retain normal combat.

Phase windows must tolerate the declared decision interval. Record shield
cycles, unnecessary casts, and correct interrupts separately from eventual
defeat, so control difficulty does not masquerade as forgetting.

### T: Plan and execute a vent-assisted crossing

Learn to inspect a vent's family mark and pulse, charge it with the learned
element, launch during the safe pulse, land on a resting shelf, and recover
enough visible stamina for the next leg. Sample charge-element and safe-pulse
cue assignments per family; demonstrate them in a safe training route.

Transfer sites rearrange vents, shelves, orientations, and distances. The agent
must choose reachable landings and sufficient stamina from current observations,
rather than replay movement vectors. All falls have reachable recovery terrain;
controls and cues are visible before committing.

Accept any valid traversal. Validate that a scored skill probe actually requires
the taught interaction; if a legal shortcut bypasses it, count task success
without claiming evidence of skill reuse. Vent forces and pulse behavior are
bounded new mechanics, not assumed current simulator capabilities.

## Task A: The Relay Keeper's Apprenticeship — 12 milestones

**Player-facing goal:** "Restore Windrest's signal route and bring the summit
beacon online. The relay keeper can show you how the old machinery works."

One procedure is acquired early, followed by useful work that does not rehearse
it, then transfer tests with changed objects and initial states.

| ID | Milestone | Completion predicate and memory role |
| --- | --- | --- |
| A01 | Complete the keeper's guided relay repair. | Activate a single-ring relay with stepwise instruction. Record observed cues, actions, and state transitions: supported acquisition of R. |
| A02 | Repair a second training relay independently. | Activate a normal relay with rearranged role positions and no stepwise hints. Establish immediate transfer; record any return for help. |
| A03 | Recover an overloaded training relay. | Activate an initially overloaded relay that also receives one signaled interruption after priming. Diagnostic feedback teaches conditional recovery and preserved progress. |
| A04 | Survey the marsh supply route. | Visit and register marked route junctions for the quartermaster; learn a distinct cache landmark. No relay practice occurs here. |
| A05 | Recover the ferry-lift casing. | Retrieve the unique casing from an ordinary raider camp by combat or another legal approach. |
| A06 | Deliver the casing to the mechanic. | Exchange the casing for repair of the ferry lift and verify it operational. This is an NPC exchange, not an R puzzle. |
| A07 | Supply the highland outpost. | Retrieve supplies from A04's cache, cross the repaired lift route, and deliver them. This ends the interference interval and opens the signal corridor. |
| A08 | Restore the first field relay. | Activate an unseen single-ring relay with new IDs, orientation, and role positions, without a recipe or skill-selection hint. First delayed R transfer probe. |
| A09 | Stabilize the second field relay. | Activate a new relay starting overloaded and partly primed. Apply R's condition checks and skip completed stages. |
| A10 | Recover the summit service cap. | Retrieve the unique cap from a separate ruin through ordinary exploration and combat. No relay demonstration is repeated. |
| A11 | Reopen the summit service platform. | Install the cap in a visibly matching mechanical-lift socket and reach the work platform. |
| A12 | Commission the summit beacon. | Activate a differently shaped beacon bearing the same role markings, recovering from one state-preserving interruption. Verify both field relays and the summit beacon active. |

**Dependencies:** A01 through A12 form the initial chain. Supply and physical
access gates enforce major dependencies. Valid early actions remain credited
through persistent world facts; progress counters do not manufacture items.

**Memory schedule:** start fresh agent contexts before A08 and A12, retaining
only memory permitted by the evaluation condition. These are not milestones and
do not reset the world. A03 to A08 contains four intervening milestones without
R practice. A09 rehearses part of R before A12, so report delay since that reuse
as well as delay since acquisition.

A08 measures near transfer, A09 conditional transfer, and A12 transfer across
appearance with recovery. Report acquisition failures separately: an agent that
never mastered A02/A03 has not necessarily forgotten a previously learned skill.

## Task B: The Expedition of Three Arts — 24 milestones

**Player-facing goal:** "Reopen the valley's expedition route, recover the
beacon heart, and bring the distant outpost online. Train with the relay keeper,
sentinel warden, and wind guide before setting out."

Acquire a repertoire, retrieve from current cues, compose for unfamiliar goals,
and update one mechanism version without overwriting another. This is independent
of Task A: start with empty memory and independently sampled family assignments.

| ID | Milestone | Completion predicate and memory role |
| --- | --- | --- |
| B01 | Complete the relay lesson. | Commission a single-ring relay with guidance; expose R's role/element relations and progress indicators. |
| B02 | Repair the workshop's faulty relay. | Activate a rearranged unit with overload and a signaled interruption. Record independent performance and assistance: R's acquisition baseline. |
| B03 | Complete the sentinel lesson. | Defeat a demonstration sentinel with coaching; expose G's shield and interrupt rules. |
| B04 | Clear the second training arena. | Defeat a sentinel independently through one authored shield reformation. Establish G's acquisition and recovery baseline. |
| B05 | Complete the wind guide's crossing. | Traverse a coached vent route; expose T's charging, pulse, landing, and stamina conditions. |
| B06 | Reach the second training overlook. | Complete an uncoached crossing with rearranged shelves and different leg length. Establish T's acquisition baseline. |
| B07 | Survey the expedition cache. | Visit and register a depot landmark. Its reserve compartment remains visibly locked pending field-relay restoration. |
| B08 | Recover the causeway repair kit. | Retrieve a kit from an ordinary raider camp, providing interference without R, G, or T practice. |
| B09 | Retrieve the submerged ferry coupling. | Dive for a separate marked chest and recover its coupling. |
| B10 | Reopen the expedition causeway. | Deliver kit and coupling to the mechanic and cross the repaired causeway. End the shared interference interval. |
| B11 | Restore the valley signal station. | Apply R at a normal relay with rearranged roles: delayed retrieval of the earliest learned procedure. |
| B12 | Reach the cliff observation post. | Apply T to an unfamiliar vent/shelf layout. Record actual vent use and any valid shortcut separately. |
| B13 | Recover the warden's field seal. | Defeat a differently presented G-family sentinel starting in a seeded taught phase; skip shield breaking if already exposed. |
| B14 | Stabilize the reserve-power relay. | Unlock its service panel with B13's field seal, then apply R to a partly primed, overloaded unit, powering B07's reserve compartment. |
| B15 | Retrieve the expedition service tools. | Return to B07's depot without a new oracle marker and recover the tools. Standalone spatial-memory probe. |
| B16 | Open the sky-lift route. | Use the tools to access its controls, commission its relay, and traverse the powered vent route. Requires R then T without a supplied decomposition. |
| B17 | Restore the guarded causeway gate. | Defeat the sentinel holding the gate controls and commission the relay. Requires G then R, with new object bindings. |
| B18 | Recover the high shrine's access seal. | Traverse a new high-platform approach and defeat its sentinel. Requires T then G. |
| B19 | Commission a double-ring relay. | Beyond B18's seal gate, inspect its badge and residual-charge diagnostic, infer the initial-drain precondition, and activate it. Record adaptation; failure is not compulsory. |
| B20 | Establish the remote landing route. | Complete an extended T crossing and activate its destination signal. Open the next district without rehearsing the relay update. |
| B21 | Repair another double-ring relay. | Apply B19's update in a new layout without the service explanation repeated. Check initial drainage before priming on the first attempt. |
| B22 | Restore an older single-ring station. | Repair an ordinary, uncharged unit after the update. Unnecessary drainage is safe but logged, testing preservation of the rule's scope. |
| B23 | Recover the summit beacon heart. | B22's station restores access to a double-ring relay powering a vent approach to a guarded reliquary. Retrieve the heart using R, T, and G with no intermediate recipes or named skill suggestions. |
| B24 | Commission a distant outpost from saved memory. | After a fresh agent context, defeat its sentinel, traverse its vent route, and commission its beacon with the heart. Use a held-out layout and visibly marked seeded relay version; verify all three outcomes. |

**Dependencies:** B01 through B24 are ordered in this first version, making
retention delays comparable. Local routes, retries, experiments, and revisits
remain flexible. Branchable campaign order can be a later variant with measured
rather than assumed delays. Each acquired mission item must be consumed or
checked by a specified gate; optional loot cannot create fictitious dependencies.

**Memory schedule:** fresh contexts before B11, B21, and B24 preserve only the
allowed store. B24 requires G -> T -> R, unlike B23's R -> T -> G. The agent must
plan from new constraints instead of replaying the previous mission's script.

| Evidence acquired | Evaluation encounters | What must transfer |
| --- | --- | --- |
| B01–B02: R | B11, B14, B16–B17 | Role-based conditional procedure and recovery, not IDs or coordinates. |
| B03–B04: G | B13, B17–B18 | Phase diagnosis, timing, and skipping unnecessary stages. |
| B05–B06: T | B12, B16, B18, B20 | Local route planning, pulse recognition, and stamina checks. |
| B07: depot landmark | B15 | Episodic/spatial recall, separate from procedural transfer. |
| Individual skills and pairwise combinations | B23–B24 | Selection, ordering, and composition for a new goal. |
| B19: double-ring exception | B21, B23, and applicable B24 variants | Retention of an updated, scoped rule. |
| Original single-ring procedure | B22 and applicable B24 variants | Preservation of old valid knowledge and avoidance of unnecessary steps. |

## Memory, observations, and fairness

Give RGB, supported controls, current health/stamina, inventory, local state
cues, newly encountered dialogue, and a concise goal. A transfer goal says
"Restore the signal station," without a recipe or saved-skill selection hint.
Visible role markings and family badges are legitimate observations: procedural
memory must operate on observable current state.

Do not supply an automatic skill summary, full journal, old-dialogue replay,
milestone design table, future encounter list, hidden coordinates, or demonstration
script. The agent writes and retrieves its own memory under declared budgets.
Format is optional; structured procedure notes are an experimental condition,
not an evaluator-written answer bank. All conditions get the same controls
tutorial. World state, rule manifests, and evaluation logs are private.

Training sources remain revisitable. Primary transfer scores evaluate the first
attempt before new instructional help. A later solve following another lesson is
assisted eventual success; record its travel and action cost. Local state and
fault feedback remain available in every condition without printing a complete
solution.

A fresh context clears recent dialogue, actions, plans, and other transient
agent state while preserving the world and permitted memory. This can run in
one live simulator process. Cross-process resume additionally needs full state
serialization. Independent task instances clear all memory; B24 deliberately
retains it within the same learned-mechanics campaign.

Seed layouts independently of rule assignments. Hold out transfer layouts and
appearances from acquisition; hold out assignment/layout combinations across
benchmark development and evaluation. Keep rules stable except for the signaled
double-ring extension. Teach every branch required at a transfer probe, or mark
its first encounter as adaptation. An untaught rule is not a forgetting test.

## Horizon and evaluation

Log both time since acquisition and time since the last relevant demonstration,
assistance, or use: decisions, observed-context tokens, simulated time, intervening
goals, and competing procedures. Reuse refreshes memory. Milestone numbering
alone cannot establish a retention interval.

For continuous-context variants, vary interference so supporting experience
either remains inside or has left the fixed recent-context window. Scheduled
fresh-context probes establish a stricter persistent-memory test. Log the actual
context available at each probe. Delays should contain useful task work, not
idle waiting. Calibrate budgets with legal reference rollouts; simulated time
is distinct from model runtime and cost.

### Comparison conditions

Use the same model, control interface, decision interval, and transfer action
budget. Match storage and retrieval budgets where possible and report exceptions.

| Condition | Available past information | Purpose |
| --- | --- | --- |
| Recent context only | Fixed recent observation/action window; no persistent store. | Performance when old experience is unavailable. |
| Agent-authored memory | Same recent window plus bounded notes derived from experience. | Practical benefit of persistent memory. |
| Structured procedure memory | Same total budget with an optional procedure format/retrieval interface. | Test abstraction and retrieval design without supplying solutions. |
| Retrieved experience records | Same budget retaining encounter excerpts. | Compare reuse of examples with explicit procedural summaries. |
| Full-history reference | Past observations/actions within a reported larger budget. | Information-availability reference, not a compute-matched condition. |
| Correct-procedure reference | Applicable abstract procedure, without hidden coordinates or action trajectory. | Estimate remaining perception, planning, and motor-control difficulty. |

For selected probes, branch a matched checkpoint with actual memory retained,
relevant learned-procedure records removed, or the older applicable version
substituted after B19. Fix world state, recent context, instructions, and action
budget. Remove duplicate answers from all permitted stores/context. If relevant
content cannot be isolated in free-form notes, describe deletion as a general
memory ablation rather than claiming procedure-specific causality.

Use several seeds and repeated agent runs; report uncertainty and acquisition
failures separately. Matched checkpoints avoid attributing different learning
histories to memory. A convincing result needs improved behavior on unseen
instances, not fluent explanations or a growing collection of notes.

### Scores

| Measure | Definition |
| --- | --- |
| Task success / progress | Final world predicate and all required milestones; separately report unique completions divided by 12 or 24. |
| Acquisition | Independent success, attempts, help, and actions at A02–A03 and B02/B04/B06 before the delay. |
| Delayed unassisted transfer | First-attempt success on unseen encounters before a repeated lesson or complete recipe hint; report eventual assisted success separately. |
| Conditional correctness | Correct initial response to overload, retained progress, exposed shield, relay version, and current stamina. |
| Reuse efficiency | Invalid applications, unnecessary casts, resets, and excess actions against a layout-specific successful reference. Do not compare raw training/test times across different difficulty. |
| Composition | Success at B16–B18 and B23–B24 without a supplied skill order; identify component failures. |
| Adaptation and preservation | Attempts at B19, first-use correctness at B21, and overgeneralization costs at B22. |
| Memory contribution | Difference in transfer success/cost under matched interventions, with uncertainty across seeds/runs. |
| Recovery and spatial recall | Source revisits, extra travel, and cache-location errors, separate from procedural competence. |

Define attempt boundaries before evaluation: mechanism reset, combat-trial reset,
traversal fall, or abandoning a probe for help. A scripted interruption is part
of the trial, not automatic failure; appropriate recovery is the tested behavior.

Sparse reward can remain one unit per milestone plus a completion bonus. Do not
reward claimed memories, note length, skill labels, or correct action prefixes.
Score transfer quality separately; disable target-distance shaping where a goal
location is meant to be remembered. Eventual success alone does not demonstrate
that memory improved reusable skill learning.

## Implementation fit and order

| Area | Existing foundation | Required addition |
| --- | --- | --- |
| Task graph | Quest, Stage, and prerequisites in open_world_bench/quests/schema.py. | Task selection, milestone IDs, compound state predicates, phase annotations, and acquisition/transfer metrics. |
| Relays | Party swaps and elemental skills. | Targetable receivers, seeded role bindings, visible faults/progress, reusable interactions, and versioned rules. |
| Sentinels | Enemies and elemental shields. | Core exposure, interruption, shield reformation, readable phases, and family-rule assignments. |
| Traversal | Climbing, gliding, swimming, stamina, and terrain. | Vent forces, charge/pulse cues, safe shelves, recovery paths, and feasibility validation. |
| Agent memory | Recent history and episode-local notes in the [embodied baseline](embodied-agent.md). | Fresh-context boundaries preserving permitted memory, selectable conditions, budgets, and access logs. Do not delete the tested store via a full policy reset. |
| Observation contract | Native goal vectors and quest instructions; shared goal hints are off by default. | Local dialogue/demonstration delivery and mechanism cues; audit instructions, rewards, HUD, and evaluator fields for answer leakage. |
| Recovery | Event-based quests and generic interactions. | Persistent facts, reusable mechanisms, durable quest items, explicit resets, and no duplicate rewards. |
| Reproducibility | Seeded worlds and recorded events/actions. | Separate layout/rule/interruption seeds, exposure and memory-operation logs, and full snapshots for paired interventions and cross-process resume. |

For each milestone record completion predicates, physical prerequisites,
encounter family/version, acquired or needed procedure, visible cues, permitted
assistance, perturbations, reset boundaries, and outputs. Distinguish
demonstration, practice, fault feedback, source revisit, and successful reuse.
Evaluator annotations must not populate agent memory or announce solutions.

Implement R and Task A first. Validate a complete legal rollout, recovery
branches, held-out role layouts, and fresh-context comparisons before adding
G and T. Then build Task B's individual tests, compositions, and scoped update.
Use a correct-procedure reference so unsolvable sites or inadequate controls
are not counted as forgetting.

The compact reference courses demonstrate the procedures and milestone outcomes.
They do not yet implement the complete island-scale tasks or provide memory
benchmark results; see the [reference-demo scope and verification](teyvat-reference-demos.md).
