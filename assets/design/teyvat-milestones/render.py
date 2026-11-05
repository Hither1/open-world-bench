"""Render the Teyvat milestone storyboards and individual card exports.

Run from any directory: python /path/to/assets/design/teyvat-milestones/render.py
Requires matplotlib and Pillow. No simulator state or network access is used.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import tempfile
import textwrap
import zipfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "owb-design-matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
W, H = 1600, 1300
DPI = 160
BG, INK, MUTED = "#f2efe7", "#233c46", "#62747b"
TEAL, GOLD, BLUE = "#267f79", "#c69945", "#4779ab"
PHASES = {
    "Learn": ("#267f79", "#e5f1ec"),
    "Practice": ("#267f79", "#e5f1ec"),
    "Other work": ("#877968", "#eee8de"),
    "Transfer": ("#386eaa", "#e8eff8"),
    "Recall": ("#386eaa", "#e8eff8"),
    "Compose": ("#755aa4", "#eee9f5"),
    "Adapt": ("#a96b31", "#f6eadb"),
    "Preserve": ("#527648", "#eaf0e2"),
}
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "axes.unicode_minus": False,
})
TEXT_BOXES = []

def text(ax, x, y, value, size=10, color=INK, weight="normal", ha="left", va="top", **kw):
    return ax.text(x, y, value, fontsize=size, color=color, weight=weight,
                   ha=ha, va=va, linespacing=1.22, **kw)

def line(ax, points, color=TEAL, lw=1.6, **kw):
    ax.plot(*zip(*points), color=color, lw=lw, solid_capstyle="round", **kw)

def arrow(ax, a, b, color=MUTED, lw=1.3, curve=0, **kw):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=8,
                 linewidth=lw, color=color, connectionstyle=f"arc3,rad={curve}", **kw))

def box(ax, x, y, w, h, fill, edge="none", radius=10, lw=1):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        facecolor=fill, edgecolor=edge, linewidth=lw))

def label(ax, x, y, value, width, size=9.1, color=INK, weight="normal", maxlines=2):
    wrapped = textwrap.fill(value, width=width, break_long_words=False, break_on_hyphens=False)
    if len(wrapped.splitlines()) > maxlines:
        raise ValueError(f"Too much card text: {value}")
    artist = text(ax, x, y, wrapped, size, color, weight)
    return artist

def crate(ax, x, y, s=1, color=GOLD):
    ax.add_patch(Rectangle((x-12*s,y-9*s),24*s,18*s,fill=False,edgecolor=color,lw=1.6))
    line(ax,[(x-12*s,y-9*s),(x+12*s,y+9*s)],color)
    line(ax,[(x+12*s,y-9*s),(x-12*s,y+9*s)],color)

def scene(ax, kind, cx, cy, scale=1):
    def pt(x,y): return (cx+x*scale,cy+y*scale)
    def l(points,color=TEAL,lw=1.6,**kw): line(ax,[pt(x,y) for x,y in points],color,lw,**kw)
    def a(x1,y1,x2,y2,color=MUTED,curve=0): arrow(ax,pt(x1,y1),pt(x2,y2),color,curve=curve)
    def circle(x,y,r,fill="#ffffff",edge=TEAL,lw=1.6):
        ax.add_patch(Circle(pt(x,y),r*scale,facecolor=fill,edgecolor=edge,lw=lw))
    def poly(points,fill="none",edge=TEAL,lw=1.6):
        ax.add_patch(Polygon([pt(x,y) for x,y in points],closed=True,facecolor=fill,edgecolor=edge,lw=lw))
    def tx(x,y,value,color=MUTED,size=6.6):
        text(ax,*pt(x,y),value,size*scale,color,ha="center",va="center")
    if kind.startswith("compose_"):
        order = kind.split("_")[1].upper()
        pos = [-88,0,88] if len(order)==3 else [-57,57]
        mapping = {"R":"relay","G":"sentinel","T":"wind"}
        for i,(p,k) in enumerate(zip(pos,order)):
            scene(ax,mapping[k],cx+p*scale,cy,scale*.38)
            tx(p,33,k,INK,8)
            if i<len(order)-1: a(p+33,0,pos[i+1]-33,0)
        return
    if kind.startswith("relay"):
        fault = kind == "relay_fault"
        version = 2 if kind == "relay_v2" else 1
        for x,word in [(-72,"source"),(0,"vane"),(72,"core")]:
            circle(x,0,15, "#e5f1ec" if x<72 else "#fcf6e8")
            if x==72:
                poly([(x,-9),(x+7,0),(x,9),(x-7,0)],GOLD,GOLD)
                if version==2: circle(x,0,20,"none",GOLD,1.1)
            elif x==0:
                l([(x-8,-6),(x+8,6)])
                l([(x+8,-6),(x-8,6)])
            else:
                l([(x-5,5),(x,-8),(x+5,5),(x-5,5)])
            tx(x,27,word)
        a(-53,0,-20,0); a(20,0,52,0)
        if fault:
            circle(-72,-30,9,"#f6eadb","#a96b31")
            tx(-72,-30,"!","#a96b31",8)
            tx(5,-28,"inspect • drain • resume","#a96b31",7)
        elif kind=="relay_shuffle":
            tx(0,-34,"same roles / new positions",BLUE,7)
            a(43,-21,-42,-21,BLUE,curve=.2)
        elif kind in ("relay_v2","relay_v1"):
            tx(0,-30,"double-ring: drain first" if version==2 else "single-ring: original rule",
               "#a96b31" if version==2 else TEAL,7)
    elif kind.startswith("sentinel"):
        poly([(-67,-23),(-40,-23),(-40,5),(-53,19),(-67,5)],"#edf0f2")
        l([(-59,-5),(-48,-5)],GOLD,2.3)
        a(-32,-2,-15,-2)
        poly([(-10,-15),(6,-15),(15,0),(6,15),(-10,15),(-18,0)],"#fcf3e3",GOLD)
        circle(-1,0,5,GOLD,GOLD)
        a(24,-2,43,-2)
        l([(57,-19),(48,-1),(61,-1),(50,18)],"#755aa4",2.7)
        tx(-54,31,"shield"); tx(-1,31,"expose"); tx(56,31,"interrupt")
        if kind=="sentinel_reset":
            a(36,-26,-39,-26,BLUE,curve=.35)
        if kind=="sentinel_open":
            tx(0,-30,"already exposed: skip shield",BLUE,6.8)
    elif kind.startswith("wind"):
        l([(-102,25),(-64,25),(-64,32),(-102,32)])
        l([(-12,2),(24,2),(24,9),(-12,9)])
        l([(67,-22),(109,-22),(109,-15),(67,-15)])
        for x in (-91,-82,-73):
            a(x,19,x,-14,TEAL,curve=.08)
        a(-63,-6,-15,-12,BLUE,curve=-.4)
        a(24,-13,67,-35,BLUE,curve=-.25)
        poly([(-47,-25),(-27,-33),(-7,-25),(-27,-22)],"#e2edf8",BLUE)
        l([(-27,-22),(-27,-15)],BLUE)
        tx(-83,43,"charge"); tx(5,25,"rest"); tx(87,2,"land")
        if kind=="wind_shuffle": tx(12,-48,"new shelves / new distances",BLUE,6.5)
    elif kind=="map":
        poly([(-103,-21),(-52,-31),(0,-18),(54,-30),(104,-10),(93,27),(43,19),(-10,31),(-59,20),(-103,28)],"#e9eee3","#b8c7b4",1)
        l([(-82,9),(-45,-11),(-13,8),(32,-3),(79,9)],TEAL,1.4,ls="--")
        circle(-82,9,4,TEAL,TEAL)
        poly([(-49,-26),(-43,-16),(-55,-16)],GOLD,GOLD)
        crate(ax,*pt(79,9),scale*.8)
    elif kind in ("camp","ruin","dive"):
        if kind=="camp":
            poly([(-105,20),(-78,-21),(-49,20)],"#e7ddd0","#897663")
            l([(-78,-21),(-73,20)],"#897663")
            poly([(-39,21),(-15,-10),(8,21)],"#e7ddd0","#897663")
        elif kind=="ruin":
            for x in [-91,-53,-15]:
                l([(x,23),(x,-20),(x+13,-20),(x+13,23)],"#897663",2.3)
            l([(-96,-24),(1,-24)],"#897663",2.3)
        else:
            for y in [-25,-13]:
                l([(-109,y),(-89,y-4),(-69,y),(-49,y-4),(-29,y),(-9,y-4),(11,y),(31,y-4),(51,y),(71,y-4),(91,y)],BLUE,1.3)
            a(-41,-4,-7,19,BLUE)
            circle(-44,-4,5,"#f6eadb",GOLD)
        crate(ax,*pt(75,14),scale)
        a(14,13,52,13,GOLD)
    elif kind in ("deliver","supply","bridge"):
        crate(ax,*pt(-85,8),scale)
        a(-61,8,-12,8,GOLD)
        if kind=="deliver":
            circle(8,-10,8,"#e7ddd0","#897663")
            l([(8,-2),(8,23),(-4,29),(8,23),(20,29)],"#897663")
            l([(-6,8),(8,3),(22,8)],"#897663")
            l([(52,30),(52,-27),(88,-27),(88,30)],TEAL)
            l([(45,5),(96,5)],TEAL,2.5)
            a(71,1,71,-19,TEAL)
        elif kind=="supply":
            poly([(4,-8),(43,-31),(82,-8)],"#e7ddd0","#897663")
            l([(12,-8),(12,30),(76,30),(76,-8)],"#897663")
            ax.add_patch(Rectangle(pt(35,5),15*scale,25*scale,facecolor="#f2efe7",edgecolor="#897663"))
        else:
            l([(-5,9),(102,9)],TEAL,3)
            for x in [0,31,62,93]:
                l([(x,-11),(x,27)],TEAL)
            l([(-5,-10),(102,-10)],TEAL,1)
            a(7,-26,92,-26,BLUE)
    elif kind=="lift":
        l([(-79,31),(-79,-28),(15,-28),(15,31)],TEAL,2)
        l([(-60,10),(6,10)],GOLD,3)
        for x in [-46,-12]: l([(x,-27),(x,9)],TEAL)
        a(-25,0,-25,-18,BLUE)
        l([(21,-24),(89,-24),(110,-36)],"#897663",2)
        crate(ax,*pt(70,-35),scale*.65)
    elif kind=="beacon":
        poly([(-25,28),(-18,-11),(18,-11),(25,28)],"#e3eadd")
        l([(-35,29),(35,29)],TEAL,2)
        poly([(-14,-11),(0,-35),(14,-11),(0,2)],"#fcf3e3",GOLD,2)
        for x,y in [(-42,-22),(42,-22),(0,-52)]:
            l([(x*.66,y*.66),(x,y)],GOLD,1.8)
        for x in [-85,85]:
            circle(x,14,10,"#e5f1ec")
            l([(x-4,14),(x-1,18),(x+5,9)],TEAL)
            l([(x,24),(x,32)],TEAL)
        l([(-74,16),(-32,16)],TEAL,1.2,ls="--")
        l([(32,16),(74,16)],TEAL,1.2,ls="--")
    else:
        raise ValueError(kind)

def draw_card(ax, row, x, y):
    cw, ch = 360, 298
    color,tint = PHASES[row["phase"]]
    box(ax,x,y+3,cw,ch,"#deded5",radius=13)
    box(ax,x,y,cw,ch,"#fffefb",edge="#d9dfd8",radius=13,lw=.75)
    box(ax,x+13,y+13,44,25,color,radius=5)
    text(ax,x+35,y+25.5,row["id"],9.2,"white","bold",ha="center",va="center")
    text(ax,x+68,y+18,row["phase"].upper(),8,color,"bold")
    if row["fresh_context"]:
        box(ax,x+239,y+14,107,23,"#243f49",radius=5)
        text(ax,x+292.5,y+25.5,"FRESH CONTEXT",6.9,"white","bold",ha="center",va="center")
    elif row["skills"]:
        text(ax,x+344,y+19,row["skills"],8.2,color,"bold",ha="right")
    title = label(ax,x+17,y+48,row["title"],32,12,INK,"bold")
    box(ax,x+13,y+91,cw-26,66,"#f5f7f2",radius=7)
    scene(ax,row["visual"],x+cw/2,y+123,.88)
    text(ax,x+17,y+168,"DO",7.0,MUTED,"bold")
    action = label(ax,x+50,y+167,row["action"],44,9.0,maxlines=3)
    text(ax,x+17,y+220,"DONE",7.0,MUTED,"bold")
    done = label(ax,x+60,y+219,row["done"],42,8.8,maxlines=2)
    box(ax,x+13,y+267,cw-26,23,tint,radius=5)
    mem = label(ax,x+22,y+273,row["memory"],55,7.45,color,maxlines=1)
    for artist in (title,action,done,mem):
        TEXT_BOXES.append((artist,(x+9,y+42,x+cw-9,y+ch-6),row["id"]))
    return (x,y,cw,ch)

def header(ax,rows,task,page_number,title,subtitle):
    ax.add_patch(Rectangle((0,0),W,185,facecolor=INK))
    # Original vector skyline, used as a quiet setting cue.
    for offset,alpha in [(0,.14),(21,.10)]:
        pts=[(940,185),(985,115+offset),(1020,140+offset),(1080,67+offset),
             (1130,128+offset),(1180,91+offset),(1235,145+offset),
             (1293,81+offset),(1361,143+offset),(1430,97+offset),(1540,158+offset),(1600,123+offset),(1600,185)]
        ax.add_patch(Polygon(pts,facecolor="#b3c9bc",edgecolor="none",alpha=alpha))
    text(ax,48,28,"OPEN-WORLD BENCH  /  TEYVAT",9.0,"#d5c29b","bold")
    text(ax,48,58,title,25.5,"#fffaf0","bold")
    text(ax,49,109,subtitle,10.8,"#d8e4df")
    text(ax,1551,31,f"TASK {task}  •  {page_number}",9,"#d5c29b","bold",ha="right")
    text(ax,49,151,"R  Relay repair     G  Sentinel combat     T  Wind traversal",9.0,"#e1e8e0")
    text(ax,1551,151,"Read left to right, then down  →",8.6,"#e1e8e0",ha="right")
    x0,segw,gap = 49,120.5,5
    for i,row in enumerate(rows):
        color,tint=PHASES[row["phase"]]
        x=x0+i*(segw+gap)
        box(ax,x,200,segw,16,color,radius=3)
        text(ax,x+segw/2,208,row["id"],6.7,"white","bold",ha="center",va="center")

def render_page(rows, task, page_number, title, subtitle, filename, footer):
    fig = plt.figure(figsize=(W/100,H/100),facecolor=BG)
    ax=fig.add_axes((0,0,1,1))
    ax.set(xlim=(0,W),ylim=(H,0)); ax.axis("off")
    header(ax,rows,task,page_number,title,subtitle)
    boxes=[]
    for index,row in enumerate(rows):
        col,r= index%4,index//4
        x,y=49+col*381,240+r*316
        boxes.append((row["id"],draw_card(ax,row,x,y)))
        if col<3:
            arrow(ax,(x+365,y+148),(x+376,y+148),"#b7bdb5",lw=.9)
    line(ax,[(49,1191),(1552,1191)],"#c8d0c8",.8)
    text(ax,49,1206,footer,9.6,INK,"bold")
    resets=", ".join(r["id"] for r in rows if r["fresh_context"])
    if resets:
        note=f"Fresh context before {resets}: clear recent history; retain only allowed saved memory. World state persists."
    else:
        note="Acquisition is established through behavior. Reading a lesson or writing a note is not proof of learning."
    text(ax,49,1232,note,8.8,MUTED)
    text(ax,49,1271,"DESIGNER VIEW  •  Proposed mechanics, not gameplay screenshots or agent-facing hints.",7.7,MUTED)
    text(ax,1552,1271,f"{rows[0]['id']}–{rows[-1]['id']}  /  {len(rows)} milestones",7.7,MUTED,ha="right")
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    for artist,(x1,y1,x2,y2),id_ in TEXT_BOXES:
        if artist.axes is not ax: continue
        extent=artist.get_window_extent(renderer)
        pts=ax.transData.inverted().transform([[extent.x0,extent.y0],[extent.x1,extent.y1]])
        xmin,ymin=pts.min(axis=0); xmax,ymax=pts.max(axis=0)
        if xmin<x1-.5 or xmax>x2+.5 or ymin<y1-.5 or ymax>y2+.5:
            raise ValueError(f"Text escapes {id_}: {artist.get_text()!r}")
    for ext in ("png","svg"):
        fig.savefig(HERE/f"{filename}.{ext}",dpi=DPI,facecolor=BG)
    with Image.open(HERE/f"{filename}.png") as im:
        for id_,(x,y,w,h) in boxes:
            im.crop(tuple(round(v*DPI/100) for v in (x-3,y-3,x+w+3,y+h+6))).save(HERE/"cards"/f"{id_}.png")
    return fig

def main():
    data=json.loads((HERE/"milestones.json").read_text())
    rows=data["milestones"]
    source=(ROOT/data["source"]).read_text()
    expected=re.findall(r"^\| ([AB]\d{2}) \|",source,re.M)
    assert [r["id"] for r in rows]==expected, "Manifest must match all source milestone IDs in order"
    assert {r["id"] for r in rows if r["fresh_context"]}=={"A08","A12","B11","B21","B24"}
    (HERE/"cards").mkdir(exist_ok=True)
    pages=[
        (rows[:12],"A","1 / 1","The Relay Keeper's Apprenticeship",
         "12 milestones  /  Learn one procedure, retain it, and transfer it to unfamiliar machinery.",
         "task-a-01-12","MEMORY ARC   Learn A01–A03  →  other work A04–A07  →  transfer A08–A09  →  new setting A12"),
        (rows[12:24],"B","1 / 2","The Expedition of Three Arts",
         "Milestones 01–12  /  Acquire relay, combat, and traversal skills; carry them through intervening work.",
         "task-b-01-12","MEMORY ARC   Acquire R, G, T in B01–B06  →  interference B07–B10  →  delayed use B11–B12"),
        (rows[24:],"B","2 / 2","The Expedition of Three Arts",
         "Milestones 13–24  /  Retrieve, compose, update, and apply the repertoire to a new mission.",
         "task-b-13-24","MEMORY ARC   Retrieve  →  compose pairs  →  update R v2  →  retain R v1  →  change the final skill order"),
    ]
    with PdfPages(HERE/"teyvat-milestone-storyboards.pdf",
                  metadata={"Title":"Teyvat: long-horizon memory milestone storyboards",
                            "Author":"Open-World Bench","Subject":"Task design: 12 and 24 milestones",
                            "CreationDate":None,"ModDate":None}) as pdf:
        for args in pages:
            fig=render_page(*args)
            pdf.savefig(fig,facecolor=BG)
            plt.close(fig)
    # Compact contact sheet for browsing; the full sheets retain readable text.
    thumbs=[]
    for page in pages:
        with Image.open(HERE/f"{page[5]}.png") as im:
            thumb=im.convert("RGB")
            thumb.thumbnail((800,650))
            thumbs.append(thumb)
    preview=Image.new("RGB",(2400,650),BG)
    for i,thumb in enumerate(thumbs): preview.paste(thumb,(800*i,0))
    preview.save(HERE/"overview.png")
    with zipfile.ZipFile(HERE/"teyvat-milestone-graphics.zip","w",zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(HERE.rglob("*")):
            if path.is_file() and path.suffix in {".png",".svg",".pdf",".json",".py"}:
                bundle.write(path,path.relative_to(HERE))
    print("Rendered 3 PNG/SVG sheets, a 3-page PDF, 36 individual cards, an overview, and a ZIP bundle.")
    print("Validated all 36 source milestone IDs, 5 fresh-context checkpoints, and card text bounds.")

if __name__=="__main__":
    main()
