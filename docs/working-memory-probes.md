# Working-memory probes

Status: shared contract for the working-memory (WM) milestones added on
2026-11-24. Each environment document states what its implementation covers.

Before this change no milestone in Teyvat, Wildlands, Skylands, or the lemon
tart was designed to test working memory. The procedural, episodic, and spatial
probes all span minutes of intervening work. A WM probe instead asks the agent
to hold a few freshly seen items for tens of seconds, sometimes while
transforming them, and use them where the source is no longer visible.

Milestone counts and IDs do not change. Each WM probe repurposes a milestone
that previously had no memory role (intervening work or plain execution), so
the existing predicate still holds and the WM mechanism becomes an additional
physical requirement of that milestone.

## Probes

| Probe | Task | Milestone | Source (cue) | Use site | Load | Kind |
| --- | --- | --- | --- | --- | ---: | --- |
| `WM-TA06` | Teyvat A | A06 ferry lift | Mechanic's spoken start sequence | Lever bank at the lift's upper/remote station | 3 | hold |
| `WM-TB09` | Teyvat B | B09 ferry coupling | Boatman names the chest's glyph mark | Six marked chests underwater | 3 | hold (select) |
| `WM-TB10` | Teyvat B | B10 causeway | Mechanic's gate pattern | Winch at the far end, set in **reverse** | 4 | reverse |
| `WM-WA07` | Wildlands A | A07 regulator | Dispatch board at the depot entrance | Locker keypad inside the stores room | 4 | hold |
| `WM-WB11` | Wildlands B | B11 couplings | Yard manifest: bay for each coupling | Six bays out of sight of the manifest | 2 bindings | hold (select) |
| `WM-WB15` | Wildlands B | B15 wind station | Weather board at the base: pitch per blade colour | Deck dial lists blades by number; colour tags are on the blades | 3 | remap |
| `WM-SA12` | Skylands A | A12 freight lift | Dispatch placard at the loading berth | Control panel at the top station | 4 | hold |
| `WM-SB10` | Skylands B | B10 marsh coupling | Gate code at the outer fence | Keypad at the inner yard gate | 4 | hold |
| `WM-SB17` | Skylands B | B17 surface station | Wiring schematic at the entrance | Patch panel inside, wired in **reverse** | 4 | reverse |
| `WM-LT1` | Lemon tart | `knobs_zeroed` | Knob order used for the first bake | Knobs turned down in **reverse** order | 3 | reverse |
| `WM-LT2` | Lemon tart | `knobs_zeroed_again` | Same order, used again for the second bake | Knobs turned down in **reverse** order | 3 | reverse |

Teyvat full-scale Task A has no milestone mechanics yet, so `WM-TA06` is
implemented only in the compact course. Teyvat campaign-v1 has no WM probe.
`WM-LT2` belongs to the 21-milestone lemon-tart procedure; the code still scores
the previous nine milestones, so only `WM-LT1` is implemented. The lemon-tart
knobs give physical feedback per knob (a knob out of turn is held), so its
primary score is reaching `knobs_zeroed` with no blocked turn rather than a
judged complete entry.

## Contract

1. **Source only.** The cue is shown in the environment's normal local text or
   dialogue channel, only after inspecting the source within reach. It clears
   when the agent leaves reach or after 5 simulated seconds, whichever comes
   first. Visible entity labels may name the source ("dispatch board") but
   never contain its content.
2. **Out of sight.** The use site is not within visibility range of the source
   and is reached by ordinary traversal of roughly 10–60 simulated seconds.
   Travel, a lift, an ascent, or a dive is the distractor. No forced waiting.
3. **Fresh, single-use content.** Items are drawn per episode from a dedicated
   generator, `np.random.default_rng([seed, stable_hash(probe_id)])`. They never
   consume the simulation RNG and are not reused by any other milestone, so
   neither long-term memory nor earlier episodes help.
4. **The rule is observable.** For `reverse` and `remap`, the source states the
   transformation ("set the winch in the reverse of this order"). The memory
   demand is holding and transforming, not guessing a hidden rule.
5. **No per-item correctness feedback.** Entering items gives a neutral click.
   Only a complete entry is judged. A wrong complete entry resets the panel
   (or the wrong chest/bay is simply empty) and is recoverable. The source can
   always be revisited.
6. **Selection probes** (`select`) count the first container opened as the
   first attempt. Distractor marks share some, not all, items with the target.
7. **Evaluator-only scoring.** Probe state is not in the policy observation.
   Environments emit these events and a per-probe summary:

   | Event | Fields |
   | --- | --- |
   | `wm_cue_read` | `probe`, `read_index` |
   | `wm_entry` | `probe`, `count` (entries so far; never the item's correctness) |
   | `wm_attempt` | `probe`, `attempt`, `correct`, `decisions_since_cue`, `seconds_since_cue`, `cue_reads` |

   Summary (`working_memory[probe]`): `milestone`, `kind`, `load`, `solved`,
   `attempts`, `first_attempt_correct`, `cue_reads`, `first_attempt_delay`
   (`decisions`, `seconds` since the most recent cue read), `solve_delay`.
   `first_attempt_correct` is the primary WM score; report it with the delay.
8. **Matched control.** `wm_control=True` also posts the cue at the use site.
   It is a diagnostic condition, never a benchmark default. The difference in
   first-attempt success between the normal and control conditions separates
   holding information from failing to operate the mechanism.
9. **Inside the recent-context window.** A WM probe is valid only if the cue
   lies inside the agent's declared recent-context window, measured in
   decisions and observation volume. A delay beyond that window turns it into
   an episodic probe. Report the measured delay rather than assuming it.

## Effect on other measurements

`WM-TB09`, `WM-TB10`, `WM-WB11`, and `WM-SB10` sit in the intervening interval
between skill training and the delayed skill probes. They add memory load
during retention, so procedural retention results must be reported with that
interval's content. Reference traces, replay verifications, checkpoint banks,
and walkthrough videos made before this change show the previous milestones.
