"""Export Task B's 24 milestones as a single, ordered node graph.

Run: python -B assets/design/teyvat-milestones/render_graph.py
Requires Matplotlib. Reads the existing milestone manifest and task sources;
never loads or modifies the simulator. Exports PNG, editable SVG, PDF, and JSON.
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import textwrap

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "owb-design-matplotlib"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.path import Path as PlotPath

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
STEM = "task-b-milestone-graph"
WIDTH, HEIGHT, DPI = 2220, 1370, 180
BG, INK, MUTED, LINE = "#f7f8fa", "#172e40", "#566979", "#7b8a97"
GOLD = "#af710c"
COLORS = {
    "Learn": ("#18776e", "#edf7f4"),
    "Practice": ("#18776e", "#edf7f4"),
    "Other work": ("#637282", "#f0f3f6"),
    "Transfer": ("#2a65a4", "#edf3fb"),
    "Recall": ("#2a65a4", "#edf3fb"),
    "Compose": ("#7552a0", "#f4effa"),
    "Adapt": ("#ae6121", "#fff3e7"),
    "Preserve": ("#52753d", "#eff6e9"),
}
ROWS = [
    ("01", "Learn the three arts", "B01–B06  /  guided lessons and independent practice"),
    ("02", "Interference and delayed retrieval", "B07–B12  /  other work, then skill reuse"),
    ("03", "Recall and combine", "B13–B18  /  conditional transfer, spatial recall, and skill pairs"),
    ("04", "Adapt, preserve, and finish", "B19–B24  /  a scoped rule update and full composition"),
]


def load_nodes():
    manifest_path = HERE / "milestones.json"
    manifest = json.loads(manifest_path.read_text())
    nodes = [m for m in manifest["milestones"] if m["id"].startswith("B")]
    expected = [f"B{i:02d}" for i in range(1, 25)]
    if [m["id"] for m in nodes] != expected:
        raise ValueError("Expected all 24 Task B milestones in order, without duplicates.")

    # Compare with executable course without importing the simulator or using RNG.
    source_path = ROOT / "open_world_bench/reference_tasks.py"
    tree = ast.parse(source_path.read_text())
    build = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_build_b")
    titles = [
        stmt.value.args[0].value
        for stmt in build.body
        if isinstance(stmt, ast.Expr)
        and isinstance(stmt.value, ast.Call)
        and isinstance(stmt.value.func, ast.Attribute)
        and stmt.value.func.attr == "add"
    ]
    if titles != [n["title"] for n in nodes]:
        raise ValueError("Milestone titles/order differ from the executable reference course.")
    doc_path = ROOT / manifest["source"]
    doc_ids = re.findall(r"^\| (B\d{2}) \|", doc_path.read_text(), flags=re.MULTILINE)
    if doc_ids != expected:
        raise ValueError("Task design and graph milestone IDs differ.")
    if [n["id"] for n in nodes if n["fresh_context"]] != ["B11", "B21", "B24"]:
        raise ValueError("Fresh-context checkpoints differ from the task design.")
    return nodes, [manifest_path, doc_path, source_path]


def main():
    nodes, sources = load_nodes()
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none", "pdf.fonttype": 42})
    fig = plt.figure(figsize=(WIDTH / 100, HEIGHT / 100), facecolor=BG)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set(xlim=(0, WIDTH), ylim=(HEIGHT, 0))
    ax.axis("off")
    text_artists = []
    node_texts = []

    def text(x, y, value, size=12, color=INK, weight="normal", **kwargs):
        artist = ax.text(x, y, value, fontsize=size, color=color, weight=weight,
                         va="top", linespacing=1.25, zorder=5, **kwargs)
        text_artists.append(artist)
        return artist

    def box(x, y, w, h, fill, edge="none", lw=1, radius=13, **kwargs):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
            boxstyle=f"round,pad=0,rounding_size={radius}", facecolor=fill,
            edgecolor=edge, linewidth=lw, **kwargs))

    def arrow(points):
        path = PlotPath(points, [PlotPath.MOVETO] + [PlotPath.LINETO] * (len(points) - 1))
        ax.add_patch(FancyArrowPatch(path=path, arrowstyle="-|>", mutation_scale=15,
                                    color=LINE, linewidth=1.7, joinstyle="round", zorder=1))

    text(80, 36, "TEYVAT  /  TASK B  /  24 MILESTONES", 11, MUTED, "bold")
    text(80, 67, "The Expedition of Three Arts", 29, weight="bold")
    text(80, 121, "Reopen the expedition route, recover the beacon heart, and bring the distant outpost online.", 13, MUTED)
    text(80, 166, "R  Relay repair     G  Sentinel combat     T  Vent-assisted traversal", 12, weight="bold")
    text(1340, 166, "Follow arrows: rows alternate direction.", 12, MUTED)
    ax.plot((80, 2140), (205, 205), color="#d9e0e6", lw=1)

    positions = {}
    node_w, node_h = 300, 170
    for i, node in enumerate(nodes):
        row, col = divmod(i, 6)
        if row % 2:
            col = 5 - col
        positions[node["id"]] = (80 + 352 * col, 280 + 240 * row)

    edges = [{"source": a["id"], "target": b["id"], "kind": "task_order"}
             for a, b in zip(nodes, nodes[1:])]
    for edge in edges:
        x1, y1 = positions[edge["source"]]
        x2, y2 = positions[edge["target"]]
        if y1 == y2:
            direction = 1 if x2 > x1 else -1
            start_x = x1 + node_w + 4 if direction == 1 else x1 - 4
            end_x = x2 - 6 if direction == 1 else x2 + node_w + 6
            arrow([(start_x, y1 + node_h / 2), (end_x, y2 + node_h / 2)])
        else:
            right = x1 > WIDTH / 2
            side = x1 + node_w + 4 if right else x1 - 4
            margin = 2180 if right else 40
            end_x = x2 + node_w + 6 if right else x2 - 6
            arrow([(side, y1 + node_h / 2), (margin, y1 + node_h / 2),
                   (margin, y2 + node_h / 2), (end_x, y2 + node_h / 2)])

    for row, (number, title, subtitle) in enumerate(ROWS):
        y = 234 + row * 240
        text(80, y, number, 13, MUTED, "bold")
        text(126, y - 1, title, 16, weight="bold")
        text(770, y + 3, subtitle, 11, MUTED)

    for node in nodes:
        x, y = positions[node["id"]]
        accent, fill = COLORS[node["phase"]]
        fresh = node["fresh_context"]
        box(x, y, node_w, node_h, fill, GOLD if fresh else "#d2dce4", 1.7 if fresh else 1, zorder=2)
        ax.plot((x + 18, x + 18), (y + 20, y + 39), color=accent, lw=3, solid_capstyle="round", zorder=4)
        artists = [text(x + 31, y + 20, node["id"], 11.5, accent, "bold")]
        artists.append(text(x + node_w - 17, y + 22, node["phase"].upper(), 9, accent, "bold", ha="right"))
        title = textwrap.fill(node["title"], 22, break_long_words=False, break_on_hyphens=False)
        if len(title.splitlines()) > 2:
            raise ValueError(f"Title too long: {node['id']}")
        artists.append(text(x + 18, y + 54, title, 14, weight="bold"))
        skill = node["skills"] or ("Spatial memory" if node["id"] in {"B07", "B15"} else "Mission / access")
        artists.append(text(x + 18, y + 112, skill, 11, accent, "bold"))
        if fresh:
            artists.append(text(x + 18, y + 141, "◆  FRESH CONTEXT BEFORE", 8.4, GOLD, "bold"))
        node_texts.append((node["id"], x, y, node_w, node_h, artists))

    ax.plot((80, 2140), (1210, 1210), color="#d9e0e6", lw=1)
    legend = [("Learning / practice", "Learn"), ("Other work", "Other work"),
              ("Transfer / recall", "Transfer"), ("Composition", "Compose"),
              ("Adaptation", "Adapt"), ("Preservation", "Preserve")]
    for x, (label, phase) in zip((80, 445, 755, 1130, 1460, 1790), legend):
        color, fill = COLORS[phase]
        box(x, 1235, 18, 18, fill, color, 1.5, radius=4)
        text(x + 30, 1234, label, 11, MUTED)
    text(80, 1277, "Arrows = required milestone order. Layout is schematic, not distance or elapsed time.", 10.5, MUTED)
    text(80, 1304, "Gold outline = designed context reset before B11, B21, or B24; world state and permitted memory persist.", 10.5, MUTED)
    text(80, 1335, "Source: Teyvat long-horizon task design + compact executable reference course. Designer-facing diagram.", 9, MUTED)

    # Check actual text geometry at export size, not just character counts.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bounds = fig.bbox
    for artist in text_artists:
        b = artist.get_window_extent(renderer)
        if b.x0 < bounds.x0 or b.y0 < bounds.y0 or b.x1 > bounds.x1 or b.y1 > bounds.y1:
            raise ValueError(f"Text falls outside figure: {artist.get_text()}")
    for mid, x, y, w, h, artists in node_texts:
        left, bottom = ax.transData.transform((x + 8, y + h - 4))
        right, top = ax.transData.transform((x + w - 8, y + 4))
        for artist in artists:
            b = artist.get_window_extent(renderer)
            if b.x0 < left or b.y0 < bottom or b.x1 > right or b.y1 > top:
                raise ValueError(f"Text falls outside {mid}: {artist.get_text()}")

    for extension in ("png", "svg", "pdf"):
        fig.savefig(HERE / f"{STEM}.{extension}", dpi=DPI, facecolor=BG)
    plt.close(fig)
    data = {
        "title": "The Expedition of Three Arts",
        "task": "B",
        "scope": "Ordered task design with a compact executable reference course",
        "edge_semantics": "Required task order; not skill-reuse or physical-distance edges",
        "sources": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        "renderer": {"name": "Matplotlib", "version": matplotlib.__version__, "kind": "2D vector diagram",
                     "dpi": DPI, "png_pixels": [round(WIDTH * DPI / 100), round(HEIGHT * DPI / 100)],
                     "svg_text": "editable", "simulation_loaded": False},
        "nodes": nodes,
        "edges": edges,
        "validation": {"nodes": len(nodes), "edges": len(edges), "course_titles_match": True,
                       "design_ids_match": True, "text_within_bounds": True},
    }
    (HERE / f"{STEM}.json").write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    print(f"Exported {len(nodes)} nodes, {len(edges)} ordered edges: {HERE / STEM}.{{png,svg,pdf,json}}")


if __name__ == "__main__":
    main()
