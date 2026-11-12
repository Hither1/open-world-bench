"""Export all 32 Tempest Restoration checkpoints as a milestone graph.

Run: python -B assets/design/skylands-milestones/render_graph.py
Requires the Graphviz `dot` executable. Does not import or run the simulator.
The graph summarizes physical flows and optional routes, not a required itinerary.
"""
from __future__ import annotations

import ast
import hashlib
import html
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
STEM = "tempest-restoration-milestone-graph"
SOURCES = [
    "docs/skylands-restoration.md", "docs/skylands-tempest.md",
    "owb/env/skylands_reference.py", "owb/env/skylands_campaign.py",
    "owb/env/skylands_tempest.py", "owb/env/skylands_restoration.py",
]

# Short outcome labels, with source terminology retained in the JSON manifest.
# Notes identify long-distance information links without a tangle of arrows.
GROUPS = [
    ("prepare", "1   PREPARE & DISCOVER", "#536677", "#f1f4f7", [
        ("B01", "Accept the commission", "Remember the designated refuge"),
        ("B02", "Commission the training barge", "Loaded craft / trim trial"),
        ("B03", "Deliver the practice load", "Independent cargo construction"),
        ("B04", "Open the maintenance loft", "Learn an ascent route"),
        ("B05", "Recover the depot service kit", "Lamp kit + final-lock coupling"),
        ("B06", "Restore the training platform", "Learn motion reversal"),
        ("B07", "Retrieve the survey pack", "Independent platform route"),
        ("B08", "Survey at the weather archive", "Local record: cooling profile"),
        ("B09", "Distribute expedition reserves", "Remember the two chosen caches"),
        ("R01", "Recover the original tooling", "Upper depot; may cache and retrieve"),
    ]),
    ("cooling", "2a   DEPTHS / COOLING", "#146d84", "#edf7fa", [
        ("B12", "Establish the deep outpost", "Lamp kit from B05 / cache B09"),
        ("B13", "Ferry the power module", "Loaded craft through flooded works"),
        ("B14", "Open the regulator gallery", "Reach the upper service controls"),
        ("B15", "Restore cooling equipment", "Install module; sustain output 5 s"),
        ("R02", "Balance the cooling flow", "Set dials from the B08-area record"),
    ]),
    ("anchors", "2b   SURFACE / ANCHORS", "#39753d", "#f0f7ec", [
        ("B10", "Recover the anchor coupling", "Original load from abandoned yard"),
        ("B16", "Deliver the anchor coupling", "Adapt the craft for reed channels"),
        ("B17", "Restore anchor equipment", "Install coupling; output for 5 s"),
        ("R03", "Tune anchor load distribution", "Set dials from the B12-area record"),
    ]),
    ("monitoring", "2c   SKY / MONITORING", "#6551a2", "#f4f0fb", [
        ("B11", "Charge the shuttle power cell", "Local record: monitoring profile"),
        ("B18", "Recover the sky spindle", "Power the observatory shuttle"),
        ("B19", "Restore monitoring equipment", "Upper controls; output for 5 s"),
        ("R04", "Synchronize the monitoring clock", "Set dials from the B11-area record"),
    ]),
    ("relief", "3a   RELIEF DELIVERY", "#9a641a", "#fff7e9", [
        ("B20", "Survey the changed approach", "Bridge closes after equipment repair"),
        ("B21", "Recover & fit the drive coupling", "Original part from B05 / cache B09"),
        ("B22", "Reach the lock service terrace", "Three balanced services required"),
        ("B23", "Commission the cargo lock & barge", "Fit drive; open berth; test loaded craft"),
        ("B24", "Deliver the relief shipment", "Original load at commissioned refuge"),
    ]),
    ("manifold", "3b   FABRICATE & INSTALL", "#ad5845", "#fff1ed", [
        ("R05", "Fabricate the pressure manifold", "Tooling + all 3 balanced services"),
        ("R06", "Transport & install the manifold", "280 kg original load to engine bus"),
    ]),
    ("finish", "4   ENGINE & REFUGE POWER", "#17594e", "#eaf5ef", [
        ("R07", "Sustain engine shutdown", "5 s / synchronization, grounding or capacitors"),
        ("R08", "Commission stable refuge power", "All 3 services live + relief + manifold", "Engine stopped + local switch armed / 30 s"),
    ]),
]

# Solid arrows describe physical inputs and outputs along the illustrated route.
# Dashed arrows are useful progression, not mandatory completion dependencies.
# This is a schematic: it is not the full low-level precondition graph.
EDGES = [
    ("B01", "B02", "route", "Begin cargo practice"),
    ("B02", "B03", "route", "Apply the cargo construction procedure"),
    ("B01", "B04", "route", "Begin ascent practice"),
    ("B04", "B05", "route", "Apply ascent at the depot"),
    ("B01", "B06", "route", "Begin reversal practice"),
    ("B06", "B07", "route", "Apply motion reversal"),
    ("B07", "B08", "flow", "Survey pack enables archive survey"),
    ("B05", "B09", "flow", "Service kit provides both reserve items"),
    ("B05", "R01", "route", "Tooling is at the upper expedition depot"),
    ("B09", "B12", "route", "Retrieve the cached lamp kit; caching itself is optional"),
    ("B09", "B21", "route", "Retrieve the cached drive coupling; caching itself is optional"),
    ("B12", "B13", "flow", "Released original power module"),
    ("B13", "B14", "route", "Reach the gallery after the cargo crossing"),
    ("B13", "B15", "flow", "Deliver the original module to cooling works"),
    ("B14", "B15", "flow", "Gallery opens the installation shutter"),
    ("B15", "R02", "flow", "Installed intact cooling equipment plus correct dials"),
    ("B10", "B16", "flow", "Released original anchor coupling"),
    ("B16", "B17", "flow", "Delivered coupling can be installed"),
    ("B17", "R03", "flow", "Installed intact anchor equipment plus correct dials"),
    ("B11", "B18", "flow", "Charged cell powers shuttle equipment"),
    ("B18", "B19", "flow", "Original spindle for monitoring installation"),
    ("B19", "R04", "flow", "Installed monitoring equipment plus correct dials"),
    ("B15", "B20", "flow", "All three equipment outputs trigger bridge closure"),
    ("B17", "B20", "flow", "All three equipment outputs trigger bridge closure"),
    ("B19", "B20", "flow", "All three equipment outputs trigger bridge closure"),
    ("B20", "B22", "route", "Discover the replacement approach"),
    ("B21", "B23", "flow", "Installed original drive coupling enables the cargo lock"),
    ("B22", "B23", "route", "Platform provides access; alternatives are permitted"),
    ("B23", "B24", "route", "Reference barge delivery; alternative physical routes are permitted"),
    ("R01", "R05", "flow", "Bring original tooling to the workbench; it is consumed"),
    ("R02", "R05", "flow", "Cooling must currently be balanced"),
    ("R03", "R05", "flow", "Anchors must currently be balanced"),
    ("R04", "R05", "flow", "Monitoring must currently be balanced"),
    ("R05", "R06", "flow", "Fabricated original manifold becomes physical cargo"),
    ("R06", "R08", "flow", "Intact original manifold currently installed at engine bus"),
    ("R07", "R08", "flow", "Sustained engine shutdown"),
    ("B24", "R08", "flow", "Original intact relief load currently at commissioned refuge"),
]


def literal_assignment(path, name):
    tree = ast.parse((ROOT / path).read_text())
    for stmt in tree.body:
        if isinstance(stmt, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in stmt.targets):
            return ast.literal_eval(stmt.value)
    raise ValueError(f"Missing {name} in {path}")


def main():
    base = literal_assignment("owb/env/skylands_reference.py", "B_TASK")
    extra = literal_assignment("owb/env/skylands_restoration.py", "EXTRA_CHECKPOINTS")
    places = literal_assignment("owb/env/skylands_tempest.py", "TITLES")
    expected = {f"B{i:02}": row[0] for i, row in enumerate(base, 1)} | extra
    rows = [row for *_, group in GROUPS for row in group]
    ids = [row[0] for row in rows]
    assert len(ids) == len(set(ids)) == 32 and set(ids) == set(expected)
    assert all(a in expected and b in expected and kind in ("flow", "route") for a, b, kind, _ in EDGES)
    assert len({(a, b) for a, b, *_ in EDGES}) == len(EDGES)
    # Acyclicity and coverage check for the actual rendered graph.
    remaining = set(ids)
    while remaining:
        roots = {n for n in remaining if not any(b == n and a in remaining for a, b, *_ in EDGES)}
        assert roots, "Unexpected dependency cycle"
        remaining -= roots
    assert set(ids) == {n for a, b, *_ in EDGES for n in (a, b)}

    q = json.dumps
    lines = ["digraph milestones {", 'graph [rankdir=TB, bgcolor="#ffffff", pad=0.4, nodesep=0.35, ranksep=0.48, splines=polyline, compound=true, newrank=true, outputorder=edgesfirst, fontname="Helvetica", fontsize=24, labelloc=t, labeljust=l];',
             'node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=15, margin="0.14,0.10", penwidth=1.3];',
             'edge [color="#81909a", arrowsize=0.65, penwidth=1.25, fontname="Helvetica", fontsize=10];']
    title = ('<<FONT POINT-SIZE="30" COLOR="#173329"><B>SKYLANDS / TEMPEST RESTORATION</B></FONT>'
             '<BR ALIGN="LEFT"/><FONT POINT-SIZE="17" COLOR="#52646c">32 milestone nodes across preparation, three regional services and the finale</FONT>'
             '<BR ALIGN="LEFT"/><BR ALIGN="LEFT"/><FONT POINT-SIZE="13" COLOR="#52646c">Solid arrows: physical input / output flow. Dashed arrows: optional route or practice progression.</FONT>'
             '<BR ALIGN="LEFT"/><FONT POINT-SIZE="13" COLOR="#52646c">Branches can be reordered. Shutdown may happen early. Labels identify record and cache links.</FONT>'
             '<BR ALIGN="LEFT"/><BR ALIGN="LEFT"/>>')
    lines.append(f"label={title};")
    manifest_nodes = []
    for group, title, color, fill, nodes in GROUPS:
        lines.append(f'subgraph cluster_{group} {{ label={q(title)}; fontsize=15; fontcolor={q(color)}; color={q(color + "55")}; style="rounded"; margin=18;')
        for mid, short, *notes in nodes:
            label = f'<<FONT COLOR="{color}" POINT-SIZE="13"><B>{mid}</B></FONT><BR ALIGN="LEFT"/><FONT POINT-SIZE="16"><B>{html.escape(short)}</B></FONT>'
            label += ''.join(f'<BR ALIGN="LEFT"/><FONT COLOR="#52646c" POINT-SIZE="11">{html.escape(note)}</FONT>' for note in notes)
            label += '<BR ALIGN="LEFT"/>>'
            tip = f"{mid}: {short}. " + ". ".join(notes)
            lines.append(f'{mid} [id="milestone-{mid}", label={label}, fillcolor={q(fill)}, color={q(color)}, tooltip={q(tip)}];')
            manifest_nodes.append(dict(id=mid, title=short, source_title=expected[mid], site=places.get(mid), group=group, notes=notes))
        lines.append("}")
    # Invisible edges and equal ranks control page layout only; they do not
    # appear in the diagram or count as semantic relationships in the manifest.
    lines.extend([
        '{rank=same; B08; B09; R01;}',
        '{rank=same; B12; B10; B11;}',
        '{rank=same; R02; R03; R04;}',
        '{rank=same; B20; B21; R05;}',
        '{rank=same; B24; R06; R07;}',
        'B08 -> B10 [style=invis]; B08 -> B11 [style=invis];',
        'R02 -> B20 [style=invis]; R03 -> B20 [style=invis]; R04 -> B20 [style=invis];',
    ])
    for a, b, kind, meaning in EDGES:
        style = 'style=dashed, color="#a3aeb6", penwidth=1.1' if kind == "route" else 'style=solid'
        lines.append(f'{a} -> {b} [{style}, tooltip={q(meaning)}];')
    lines.append('}')
    dot = HERE / f"{STEM}.dot"
    dot.write_text("\n".join(lines)+"\n")
    version = subprocess.run(["dot", "-V"], capture_output=True, text=True, check=True).stderr.strip()
    for ext in ("svg", "png", "pdf"):
        subprocess.run(["dot", f"-T{ext}", "-Gdpi=160", str(dot), "-o", str(HERE / f"{STEM}.{ext}")], check=True)
    svg = ET.parse(HERE / f"{STEM}.svg")
    rendered_ids = {e.attrib["id"].removeprefix("milestone-") for e in svg.iter() if e.attrib.get("id", "").startswith("milestone-")}
    assert rendered_ids == set(ids), "An expected milestone is missing from the SVG"
    manifest = dict(
        title="Skylands: Tempest Restoration", environment="OpenWorldBench/Skylands-Tempest-Restoration-v0",
        scope="Largest implemented Skylands checkpoint set: 24 inherited logistics + 8 restoration outcomes. No minimum-duration claim.",
        interpretation="Schematic of physical flows and optional routes, not exhaustive low-level preconditions or a mandatory milestone checklist. Records name information sources; reading them is not mechanically required to set correct dials.",
        final_live_conditions=["Agent alive", "All three regional services currently balanced with original intact equipment", "Original intact manifold installed at engine pressure bus", "Engine stopped for at least five seconds", "Original intact relief load at commissioned refuge", "Local refuge transfer switch armed", "All power conditions sustained for 30 seconds"],
        omitted_cross_edges=["B05 supplies lamp kit at B12 and drive coupling at B21; storage at B09 is optional", "B22 needs all three balanced services", "B23 needs installed regional equipment and the repaired B21 drive", "R08 still needs all three balanced services after R05", "B08-area record -> R02; B12-area record -> R03; B11-area record -> R04", "B01 identifies the commissioned B24 refuge"],
        sources={p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in SOURCES},
        renderer=dict(name="Graphviz dot", version=version, kind="2D vector diagram", png_dpi=160, simulation_loaded=False),
        nodes=manifest_nodes,
        edges=[dict(source=a, target=b, kind=kind, meaning=meaning) for a, b, kind, meaning in EDGES],
        validation=dict(source_ids_match=True, unique_nodes=len(ids), rendered_nodes=len(rendered_ids), edges=len(EDGES), acyclic=True),
    )
    (HERE / f"{STEM}.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(json.dumps(manifest["validation"]))


if __name__ == "__main__":
    main()
