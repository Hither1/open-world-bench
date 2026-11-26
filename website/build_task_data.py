"""Build the public task explorer from the repository's task specifications.

The diagrams are editorial groupings, never simulator input. Milestone tables
are read verbatim (with Markdown formatting removed); missing/duplicate IDs fail
the build instead of silently publishing an incomplete diagram.
"""
import json
import argparse
import re
from pathlib import Path
from reference_videos import catalogue

ROOT = Path(__file__).resolve().parents[1]


def plain(text):
    text = re.sub(r"\[([^]]+)\]\([^)]+\)", r"\1", text)
    return text.replace("**", "").replace("`", "").strip()


def rows(doc, prefix):
    found = []
    for line in (ROOT / "docs" / doc).read_text().splitlines():
        cells = [plain(c) for c in line.strip().strip("|").split("|")]
        if not line.startswith("|") or not re.fullmatch(prefix + r"\d{2}", cells[0]):
            continue
        if len(cells) == 4:  # Minecraft retains the original statistic predicate.
            ident, stage, title, detail = cells
            detail = f"{stage}. Credit when this statistic increases: {detail}."
            title = title.replace("_", " ").capitalize()
        else:
            ident, title, detail = cells
        found.append(dict(id=ident, title=title.rstrip("."), detail=detail))
    return found


def phase(title, ids, note, position=None):
    return dict(title=title, ids=ids.split(), note=note, position=position)


def span(prefix, start, end):
    return " ".join(f"{prefix}{i:02}" for i in range(start, end + 1))


def item(ident, title, detail):
    return dict(id=ident, title=title, detail=detail)


TASKS = []


def task(world, ident, title, doc, goal, status, phases, milestones=None,
         prefix=None, count=None, edges=None, memory=(), route=None, metric=None):
    milestones = milestones if milestones is not None else rows(doc, prefix)
    if count is not None and len(milestones) != count:
        raise ValueError(f"{ident}: expected {count} milestones, got {len(milestones)}")
    ids = [m["id"] for m in milestones]
    assigned = [i for p in phases for i in p["ids"]]
    if len(set(ids)) != len(ids) or sorted(assigned) != sorted(ids):
        raise ValueError(f"{ident}: every milestone must appear in exactly one stage")
    for i, p in enumerate(phases):
        p["id"] = f"stage-{i + 1}"
        if p["position"] is None:
            row, col = divmod(i, 4)
            p["position"] = [col if row % 2 == 0 else 3 - col, row]
    edges = edges if edges is not None else [[i, i + 1] for i in range(len(phases) - 1)]
    for edge in [*edges, *memory]:
        if not all(0 <= n < len(phases) for n in edge[:2]):
            raise ValueError(f"{ident}: invalid connection {edge}")
    TASKS.append(dict(world=world, id=ident, title=title, source=doc, goal=goal,
                      status=status, phases=phases, milestones=milestones,
                      edges=edges, memory=memory,
                      metric=metric or f"{len(milestones)} milestones",
                      route=route or "A reference itinerary. Connections show the task structure; valid exploration, retries and alternative routes remain possible."))


# Current Teyvat campaigns: genuine prerequisite branches, with stable receipt IDs.
doc = "teyvat-long-horizon-tasks.md"
ms = [item("charged_strike", "Discover a quartz-charged strike", "Crimson quartz grants one charged physical hit, expiring after 45 simulated seconds. Each shard needs two separately charged hits; quartz regenerates.")]
for level in ("lower", "upper"):
    for i in range(3):
        detail = "Land two separately charged hits while managing cold and returning to braziers."
        if level == "lower" and i == 1:
            detail += " Defeat the shard's guardian first."
        if level == "lower" and i == 2:
            detail += " Read the local inscription and solve its seeded three-element latch sequence."
        if level == "upper":
            detail += " The summit wind seal must already be released; arriving early cannot bypass it."
        ms.append(item(f"{level}_{i}", f"Thaw {'foothill' if level == 'lower' else 'summit'} shard {i + 1}", detail))
ms += [item("ascent", "Release the summit wind seal", "All three lower shards must be thawed before the wind seal console becomes usable."), item("final", "Restore Frostfall Spire", "Restore the spire after all three summit shards are thawed. Prerequisites persist across travel.")]
ps = [phase("Discover quartz", "charged_strike", "Learn the charged-hit rule locally.", [0, 1])]
ps += [phase(f"Foothill shard {i + 1}", f"lower_{i}", ["Quartz and cold", "Guardian and quartz", "Elemental latches and quartz"][i], [1, i]) for i in range(3)]
ps += [phase("Open the summit", "ascent", "All three lower shards converge here.", [2, 1])]
ps += [phase(f"Summit shard {i + 1}", f"upper_{i}", "Reuse the quartz rule at a new location.", [3, i]) for i in range(3)]
ps += [phase("Restore the spire", "final", "All three summit shards are required.", [4, 1])]
task("teyvat", "frostfall-spire", "Frostfall Spire", doc, "Restore the Frostfall Spire at the mountain summit.", "Playable branching campaign", ps, ms, count=9,
     edges=[[0, i] for i in (1, 2, 3)] + [[i, 4] for i in (1, 2, 3)] + [[4, i] for i in (5, 6, 7)] + [[i, 8] for i in (5, 6, 7)],
     memory=[[0, 6, "Reuse the quartz rule after the foothill expedition; each hit still needs a fresh charge."]],
     route="Solid lines are prerequisites. The three foothill shards can be completed in any order, as can the three summit shards after the seal opens.")

ms = [item("lens", "Recover the Echo Lens", "Use the lens at each echo statue to reveal that shrine's unique ward.")]
ps = [phase("Recover Echo Lens", "lens", "A shared tool opens five independent branches.", [0, 2])]
for i in range(5):
    ms += [item(f"shrine_{i}_ward", f"Install shrine {i + 1}'s ward", "Recover the shrine-specific ward with the lens and consume it at its matching socket. Another shrine's ward cannot substitute."),
           item(f"shrine_{i}_puzzle", "Solve the elemental pattern", "Read the seeded ember, tide and gale order from the local tablet. Receiver positions are shuffled; incorrect ordering resets the lamps but preserves the ward."),
           item(f"shrine_{i}", "Defeat the guardian and cleanse", "The solved pattern exposes the guardian. Defeat it, then interact with the altar to permanently cleanse this shrine.")]
    ps.append(phase(f"Cleanse shrine {i + 1}", f"shrine_{i}_ward shrine_{i}_puzzle shrine_{i}", "Ward · pattern · guardian", [1, i]))
ms += [item("final", "Defeat the Blightroot Heart", "All five cleansings unlock the Heart. Each of its two 80-HP phases needs a puzzle solution and combat within a 20-second exposure window. Phase two rotates the tablet order; both phases are subconditions of this one final receipt.")]
ps += [phase("Defeat the Heart", "final", "Two puzzle-and-combat phases, one final receipt.", [2, 2])]
task("teyvat", "fivefold-cleansing", "Fivefold Cleansing", doc, "End the valley blight by defeating the sealed Blightroot Heart.", "Playable branching campaign", ps, ms, count=17,
     edges=[[0, i] for i in range(1, 6)] + [[i, 6] for i in range(1, 6)],
     memory=[[1, 5, "Transfer the role-based puzzle procedure across shrines; each local tablet has its own order."], [5, 6, "Adapt the procedure again when the Heart rotates its pattern."]],
     route="Solid lines are prerequisites. All five shrines are available after the lens, in any order. All five must be cleansed to unlock the Heart.")

ms = [item("slate", "Recover the rank-one slate", "Start with rank-one clearance and six charge units. Attuning and reading each cost one unit; opening an archive costs two. Obelisks recharge the slate.")]
ps = [phase("Recover the slate", "slate", "Track both clearance and renewable charge.", [0, 1])]
for i in range(3):
    ms += [item(f"temple_{i}_puzzle", f"Solve temple {i + 1}'s circuit", "Read and solve the local circuit, accounting for slate charge."), item(f"temple_{i}", "Obtain the permission mark", "Defeat the temple guardian after its puzzle and obtain this temple's persistent permission mark.")]
    ps.append(phase(f"Temple {i + 1}", f"temple_{i}_puzzle temple_{i}", "Circuit · guardian · permission", [1, i]))
ms += [item("rank_two", "Upgrade at the central calibrator", "Return with all three permission marks to unlock rank-two clearance. Rank-one clearance cannot open the temple archives.")]
ps += [phase("Upgrade clearance", "rank_two", "Return to the central calibrator.", [2, 1])]
for i in range(3):
    ms += [item(f"temple_{i}_record", f"Revisit archive {i + 1}", "Return with rank-two clearance, spend two charge units, and collect this archive's unique chart.")]
    ps.append(phase(f"Revisit archive {i + 1}", f"temple_{i}_record", "New clearance makes an old location useful.", [3, i]))
ms += [item("atlas_key", "Assemble the cipher at camp", "The expedition camp's cipher desk requires all three unique charts."), item("zenith_puzzle", "Solve the Zenith circuit", "Use the assembled cipher to reach and solve the final circuit."), item("zenith", "Defeat the custodian", "Overcome the Zenith custodian after the circuit is solved."), item("final", "Recover the star atlas", "Recover the lost atlas with all persistent credentials and final prerequisites satisfied.")]
ps += [phase("Recover the atlas", "atlas_key zenith_puzzle zenith final", "Camp cipher · Zenith circuit · custodian", [4, 1])]
task("teyvat", "zenith-archive", "Zenith Archive", doc, "Recover the lost star atlas from the sealed Zenith Archive.", "Playable branching campaign", ps, ms, count=15,
     edges=[[0, i] for i in (1, 2, 3)] + [[i, 4] for i in (1, 2, 3)] + [[4, i] for i in (5, 6, 7)] + [[i, 8] for i in (5, 6, 7)],
     memory=[[1, 5, "Remember an archive encountered while locked and return after the clearance upgrade."], [3, 7, "Retain temple locations while managing recharge trips."]],
     route="Solid lines are prerequisites. Temple branches can be reordered. All permissions are needed for rank two, and all charts for the camp cipher.")

# Milestone-based courses. Each stage groups exact source-table IDs.
task("wildlands", "frostbound-pass", "The Frostbound Pass", "wildlands-long-horizon-tasks.md", "Repair the mountain intake, drain the flooded pass, and return to Reedbank with the road open.", "Task design · compact reference available", [
    phase("Commission & experience", span("A", 1, 3), "Learn a cache landmark, counterweights and safe updrafts."),
    phase("Independent practice", span("A", 4, 5), "Check both skills without procedural hints."),
    phase("Supply expedition", span("A", 6, 8), "Survey, obtain a regulator and recover the cached handle."),
    phase("Delayed transfer", span("A", 9, 10), "Changed weights, rotated controls and a rainy shaft."),
    phase("Reopen the pass", span("A", 11, 12), "Recall the calibration and keep the intake operational.")], prefix="A", count=12,
    memory=[[1, 3, "A04 → A09 and A05 → A10: retrieve counterweight and updraft procedures after other work."], [0, 2, "A01 → A08: recall the ranger cache's landmark."], [2, 4, "A06 → A11: recall the surveyed intake calibration."]])
task("wildlands", "valley-before-storm", "The Valley Before the Storm", "wildlands-long-horizon-tasks.md", "Restore the floodworks, divert water away from the designated refuge, and reopen its evacuation road.", "Task design · compact reference available", [
    phase("Learn & practice", span("B", 1, 7), "Counterweights, updrafts and flow routing.", [0, 1]),
    phase("Survey & supplies", span("B", 8, 12), "Remember the depot, couplings and valve-to-basin map.", [1, 1]),
    phase("Wind station", span("B", 13, 15), "Transfer support/release and safe updraft skills.", [2, 0]),
    phase("Marsh pump", "B16", "Route before feeding a larger network.", [2, 1]),
    phase("Quarry engine", "B17 B18", "Cool an unsafe plume before launching.", [2, 2]),
    phase("Update the route", "B19 B20", "A storm closes the remembered depot approach.", [3, 1]),
    phase("Restore the floodworks", span("B", 21, 24), "Combine all three skills and reach the correct refuge.", [4, 1])], prefix="B", count=24,
    edges=[[0, 1], [1, 2], [1, 3], [1, 4], [2, 5], [3, 5], [4, 5], [5, 6]],
    memory=[[0, 4, "Choose cooling before updraft traversal when the plume is unsafe; familiar appearance alone is insufficient."], [1, 5, "B08 → B19–B20: revise the old bridge route using the dated closure report."], [0, 6, "B01 → B22–B24: retain the authorized basin and designated refuge."]],
    route="After preparation, wind, marsh and quarry branches may be interleaved. All regional outputs enable the return to the depot and final floodworks.")

task("skylands", "fallen-observatory", "The Fallen Observatory", "skylands-long-horizon-tasks.md", "Recover the navigation core beneath Aster Reach and bring the sky observatory back online.", "Task design · full-scale reference available", [
    phase("Learn cargo & ascent", span("A", 1, 5), "Demonstrations followed by independent practice."),
    phase("Stage & survey", span("A", 6, 8), "Leave a return raft and locate the depths works."),
    phase("Extract the core", span("A", 9, 12), "Transfer skills to a flooded gallery and a heavy load."),
    phase("Recover the return raft", "A13", "Retrieve the raft at the dock you chose earlier."),
    phase("Restore navigation", "A14 A15", "Hoist the physical core and recall its calibration.")], prefix="A", count=15,
    memory=[[0, 2, "A05 → A09: adapt ascent to new ceiling geometry. A06 → A11: reuse cargo construction after other work."], [1, 3, "A06 → A13: retain the chosen dock and raft identity."], [1, 4, "A07 → A15: recall the receiver calibration from the survey."]])
task("skylands", "three-layer-lifeline", "The Three-Layer Lifeline", "skylands-long-horizon-tasks.md", "Restore three freight stations, then deliver the relief cargo to the refuge named in the commission.", "Task design · full-scale reference available", [
    phase("Learn & practice", span("B", 1, 7), "Cargo craft, ascent and recent-motion reversal.", [0, 1]),
    phase("Survey & cache", "B08 B09", "Read cross-layer landmarks and choose reserve caches.", [1, 1]),
    phase("Depths station", span("B", 12, 15), "Retrieve the lamp kit, ferry the module and ascend.", [2, 0]),
    phase("Marsh station", "B10 B16 B17", "Recover, carry and install the marsh coupling.", [2, 1]),
    phase("Sky station", "B11 B18 B19", "Charge a cell and reverse the shuttle's fresh trajectory.", [2, 2]),
    phase("Revise & recover", span("B", 20, 22), "Update the closed bridge route; retrieve the cached coupling.", [3, 1]),
    phase("Deliver relief", "B23 B24", "Commission the lock and barge; reach the assigned refuge.", [4, 1])], prefix="B", count=24,
    edges=[[0, 1], [1, 2], [1, 3], [1, 4], [2, 5], [3, 5], [4, 5], [5, 6]],
    memory=[[0, 4, "B07 → B18: reuse the reversal procedure with freshly generated motion history."], [1, 5, "B09 → B21: recover the exact coupling from the agent-chosen cache; update B08's obsolete approach."], [0, 6, "B01 → B24: retain the refuge and delivery-channel assignment."]],
    route="The three station branches can be interleaved. All three live outputs enable the final lock. Couplings and cells may be collected earlier if reachable.")

doc = "skylands-tempest.md"
ms = [item("discover", "Discover the engine", "Use local observations and service records to find the runaway machine. The final encounter is accessible immediately."),
      item("prepare", "Restore regional infrastructure", "Cooling widens the vent window; anchors reduce discharge damage; monitoring exposes the phase remotely. Preparation is optional for the open objective."),
      item("sync", "Synchronize the circuits", "Read the three seed-dependent settings at the sky archive, deep intake and quarry. Isolate circuits while their shutters vent."),
      item("ground", "Deliver a grounding cartridge", "Build a rated craft, secure the unique 240 kg cartridge, steer to the receiving dock, tow it to the socket and install it."),
      item("strike", "Break exposed capacitors", "Strike exposed capacitors during venting and respect tool recovery. This allows an early route without regional repairs."),
      item("shutdown", "Sustain shutdown", "Operate the shutdown lever after isolation or grounding. The open goal requires the agent alive and the engine stopped for five simulated seconds."),
      item("relief", "Complete the extended objective", "Also maintain live regional outputs with intact original components and deliver the original intact relief shipment to its commissioned refuge.")]
task("skylands", "tempest-engine", "Silence the Tempest Engine", doc, "Find and stop the runaway atmospheric machine; choose an open or extended objective.", "Playable campaign · three shutdown methods", [
    phase("Find the engine", "discover", "The encounter can be attempted immediately.", [0, 1]),
    phase("Regional preparation", "prepare", "Optional for open; required live outcomes for extended.", [1, 0]),
    phase("Synchronize", "sync", "Recall three service settings.", [2, 0]),
    phase("Ground", "ground", "Transport the original 240 kg cartridge.", [2, 1]),
    phase("Exposed capacitors", "strike", "Time strikes to the vent window.", [2, 2]),
    phase("Stop the engine", "shutdown", "Any one shutdown method can satisfy the open goal.", [3, 1]),
    phase("Extended: relief", "relief", "Regional outputs and intact delivery must also hold.", [4, 1])], ms,
    edges=[[0, 1], [0, 2], [0, 3], [0, 4], [2, 5], [3, 5], [4, 5], [1, 6], [5, 6]],
    metric="Open / extended objectives", memory=[[0, 2, "Retain service settings discovered across three regions for the synchronization route."]],
    route="Alternative routes, not a checklist: choose one shutdown method. Regional preparation changes the encounter; the extended goal also requires regional outputs and relief delivery, in either order.")

doc = "skylands-restoration.md"
ms = [item("records", "Read the regional profiles", "The weather archive records cooling, the deep intake records anchors, and the quarry records monitoring. Each service has three seed-dependent selectors."),
      item("R01", "Recover the original manifold tooling", "Retrieve the unique tooling case from the upper depot. Carry it or cache it in a chosen regional locker for later recovery."),
      item("R02", "Balance the cooling flow", "Repair cooling equipment and apply the locally discovered profile. The installed original module and settings must remain valid."),
      item("R03", "Tune the anchor load distribution", "Install the original equipment and apply the anchor profile from the deep intake. Wrong selectors invalidate output."),
      item("R04", "Synchronize the monitoring clock", "Repair monitoring equipment and apply the profile recorded at the quarry."),
      item("R05", "Fabricate the pressure manifold", "Recover the original tooling and consume it at the sky salvage workbench. All three balanced services must be available to fabricate the unique 280 kg manifold."),
      item("R06", "Transport and install the original manifold", "Load, steer, unload and install the physical manifold at the engine pressure bus. The same salvage craft can also transport the grounding cartridge on another trip."),
      item("R07", "Sustain engine shutdown", "Synchronization, grounding or exposed-capacitor shutdown all qualify. Shutdown may occur before the other restoration work."),
      item("relief", "Deliver the original relief shipment", "The intact relief cargo must reach the refuge named in the commission. The inherited logistics route provides equipment and access."),
      item("R08", "Commission stable refuge power", "Arm the local transfer switch. Balanced services, the installed original manifold, sustained shutdown and intact relief delivery must hold for 30 simulated seconds. Losing any input resets the interval.")]
task("skylands", "tempest-restoration", "Tempest Restoration", doc, "Restore regional services, install the pressure manifold, stop the engine, and deliver relief and stable refuge power.", "Playable campaign · 32 private checkpoints", [
    phase("Read & cache", "records R01", "Learn profiles and remember the tooling locker.", [0, 1]),
    phase("Cooling", "R02", "Balance the original installed equipment.", [1, 0]),
    phase("Anchors", "R03", "Apply the remote service profile.", [1, 1]),
    phase("Monitoring", "R04", "Restore and synchronize the clock.", [1, 2]),
    phase("Fabricate & transport", "R05 R06", "Recover tooling and install the heavy manifold.", [2, 1]),
    phase("Shutdown & relief", "R07 relief", "Both are live prerequisites for refuge power.", [3, 1]),
    phase("Commission refuge power", "R08", "Keep every input stable for 30 seconds.", [4, 1])], ms,
    edges=[[0, 1], [0, 2], [0, 3], [1, 4], [2, 4], [3, 4], [4, 6], [5, 6]],
    metric="32 private checkpoints · grouped overview",
    memory=[[0, 3, "Apply regional profiles learned elsewhere; repair alone does not balance a service."], [0, 4, "Recover the original tooling from the chosen locker much later for fabrication."]],
    route="This overview groups the 24 inherited logistics checkpoints and eight restoration checkpoints. Shutdown and delivery may precede other work. All inputs must hold at the final live-state check.")

for ident, title, prefix, count, groups, goal in [
    ("mine-diamond", "Mine Diamond from Scratch", "A", 10,
     [("Wood & workbench", 1, 2), ("Stone tools", 3, 5), ("Iron & smelting", 6, 8), ("Obtain a diamond", 9, 10)],
     "Starting with nothing, obtain a diamond through gathering, GUI crafting, mining and smelting."),
    ("ender-dragon", "Ender Dragon", "B", 24,
     [("Wood & stone", 1, 5), ("Iron tools", 6, 9), ("Diamond equipment", 10, 13), ("Portal preparation", 14, 15), ("Nether resources", 16, 20), ("Eyes & stronghold", 21, 22), ("Defeat the dragon", 23, 24)],
     "Starting with nothing, progress through the Nether and a stronghold to defeat the Ender Dragon.")]:
    task("minecraft", ident, title, "minecraft-long-horizon-tasks.md", goal,
         f"Existing task · {12000 if prefix == 'A' else 36000:,} steps per life",
         [phase(name, span(prefix, start, end), "Original MCU-AgentBeats milestone statistics.") for name, start, end in groups],
         prefix=prefix, count=count,
         memory=[[0, len(groups) - 1, "Carry recipes, routes and agent-authored procedures across the long journey. These tasks have no planted memory probes; milestone credit measures progress."]],
         route="A conventional progression, not enforced scorer order. Credit fires on the first statistic increase; pre-existing inventory and statistics earn no credit.")

doc = "mobile-manipulation-tasks.md"
ms = []
for line in (ROOT / "docs" / doc).read_text().splitlines():
    if re.match(r"\| `[a-z_]+` \|", line):
        ident, detail = [plain(c) for c in line.strip().strip("|").split("|")]
        ms.append(item(ident, ident.replace("_", " ").capitalize(), detail))
task("mobile-manipulation", "lemon-tart", "Lemon tart", doc, "Make a lemon tart, return ingredient containers home, wash and restore the tools, and leave the oven closed with its knobs off.", "21-milestone design · implementation incomplete", [
    phase("Read & gather", "recipe_stowed ingredients_out tools_out", "Discover the recipe, knob order and ingredient homes."),
    phase("Prepare & bake crust", "butter_cut lemon_cut crust_mixed crust_in_pan crust_loaded crust_baked", "Mix, load and use this episode's knob sequence."),
    phase("Reset & make curd", "knobs_zeroed crust_out curd_mixed curd_started", "Reverse the knob order before cooking the filling."),
    phase("Bake the tart", "tart_filled tart_loaded tart_baked knobs_zeroed_again tart_out", "Reuse the oven procedure after intervening work."),
    phase("Restore the kitchen", "ingredients_restored tools_washed tools_restored", "Return ingredients to their own storage and restore tools.")], ms, count=21,
    memory=[[0, 3, "Retrieve the episode-specific knob order for the second bake; the first bake's knob positions have been reset."], [0, 4, "Recall ingredient homes after cooking. Tool-home recall is not part of the ingredient memory score."]],
    route="The target 21-milestone procedure. The standard environment still scores the older nine milestones; the documented continuous run reaches 20/21. This diagram shows the design, not a completed evaluation.")
ms = [item("morning", "Learn in the morning", "Use the lemon-tart skill in the kitchen and make nut and lemon sauces. Learn the oven rule and restore the kitchen."),
      item("journey", "Learn the dining-room route", "The route includes a stretch not visible from either room. The second trip should use the remembered route."),
      item("update", "Read the later guest note", "A new note says the guest cannot eat nuts. Update the serving decision to lemon sauce."),
      item("relocate", "Find the moved lemon sauce", "The sauce is no longer in its morning location. Opening the old cabinet is a stale-location error."),
      item("transfer", "Use the dining-room oven", "The same procedure applies, but knob positions differ. Transfer the learned rule, not the old coordinates or leftover knob settings."),
      item("serve", "Serve and restore", "Serve only lemon sauce and return objects to their own rooms. An optional pause preserves the world while clearing recent agent context.")]
task("mobile-manipulation", "guest-dessert", "The guest's dessert", doc, "Reuse the morning's cooking skill in another room after the guest's requirements and a storage location change.", "Proposed · not implemented", [phase(m["title"], m["id"], m["detail"].split(". ")[0] + ".") for m in ms], ms,
     metric="Conceptual stages · milestone list not finalized",
     memory=[[0, 4, "Reuse the morning's oven rule with physically different knob positions."], [0, 3, "Update the sauce location instead of relying on stale storage memory."], [2, 5, "The later guest note changes which sauce may be served."]],
     route="A conceptual design: the second room and revised lemon-tart procedure are not implemented. These stages are not scored milestone IDs.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--media-output", type=Path)
    args = parser.parse_args()
    output = args.output
    catalogue(TASKS, args.media_output)
    output.write_text("// Generated by website/build_task_data.py from docs/.\nwindow.OWB_TASKS = " + json.dumps(TASKS, ensure_ascii=False, indent=2) + ";\n")
    print(f"Built {len(TASKS)} task diagrams across {len({t['world'] for t in TASKS})} environments")
