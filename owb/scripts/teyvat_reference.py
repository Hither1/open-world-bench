"""Record verified native-action Teyvat reference-course walkthroughs.

python -B -m owb.scripts.teyvat_reference --task a b --output /tmp/teyvat-reference

The controller runs first without rendering. A fresh environment replays its
action log, verifies milestone/final hashes, and records actual simulator frames.
No model API calls, teleport controls, or progress-setting policy actions exist.
Defaults to Blender Cycles. Requires Blender, FFmpeg, Pillow, and OpenCV for
decoding checks; --renderer cpu explicitly selects diagnostic CPU rendering.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import platform
import tempfile
import textwrap
import time

import numpy as np

os.environ.setdefault("MPLCONFIGDIR",str(Path(tempfile.gettempdir())/"owb-reference-matplotlib"))

from open_world_bench.reference_tasks import ReferenceTaskEnv
from open_world_bench.reference_policy import ReferencePolicy
from owb.render.presentation import PresentationRenderer, add_render_arguments


def render_source_hashes():
    root=Path(__file__).resolve().parents[2]
    paths=[root/"owb/scripts/teyvat_reference.py",*sorted((root/"owb/render").glob("*.py")),
           *sorted((root/"open_world_bench").glob("*.py")),
           *sorted((root/"open_world_bench/render").glob("*.py"))]
    return {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def jsonable(value):
    if isinstance(value,np.ndarray):
        return value.tolist()
    if isinstance(value,np.generic):
        return value.item()
    if isinstance(value,dict):
        return {k:jsonable(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):
        return [jsonable(v) for v in value]
    return value


def save_json(path,value):
    path.write_text(json.dumps(jsonable(value),indent=2,allow_nan=False)+"\n")


def collect_reference(task,seed,directory,max_steps=9000):
    env=ReferenceTaskEnv(task,seed,render_enabled=False)
    rows=[]
    counts=Counter()
    try:
        obs,_=env.reset(seed=seed)
        policy=ReferencePolicy(env)
        initial_digest=env.state_digest()
        with (directory/"actions.jsonl").open("w") as stream:
            for step in range(max_steps):
                milestone=env.course.current
                op=env.course.op
                action=policy(obs)
                assert env.action_space.contains(action)
                before=env.char.pos.copy()
                obs,reward,terminated,truncated,info=env.step(action)
                counts.update(info["events"])
                row={"step":step+1,"milestone":milestone.id,"title":milestone.title,
                     "operation":op.id,"explanation":policy.explanation,"action":jsonable(action),
                     "position_before":before.tolist(),"position_after":env.char.pos.tolist(),
                     "simulated_seconds":env.world.elapsed,"reward":reward,
                     "events":info["event_details"]}
                if "milestone_complete" in info["events"]:
                    row["state_digest"]=env.state_digest()
                    print(f"Task {task.upper()}: {len(env.course.completed)}/{len(env.course.milestones)} verified",flush=True)
                stream.write(json.dumps(row,allow_nan=False)+"\n")
                rows.append(row)
                if terminated or truncated:
                    break
            verification=env.course.verify()
            if not verification["success"] or not env.quests.all_complete:
                raise RuntimeError(f"Reference did not complete: {verification}")
            if counts["player_defeated"]:
                raise RuntimeError("A successful reference recording must not contain respawn jumps")
        manifest={"task":task,"seed":seed,"controller":"privileged native-action ReferencePolicy",
                  "verification":verification,"initial_digest":initial_digest,
                  "final_digest":env.state_digest(),"decisions":len(rows),
                  "simulated_seconds":env.world.elapsed,"event_counts":dict(counts),
                  "milestones":env.course.completed,"decision_seconds":.2,
                  "limitations":["Compact authored course, not the full island-scale task suite",
                    "Privileged reference controller: no claim about learned memory or model performance",
                    "Ferry/causeway repair represented by item exchanges and access milestones",
                    "Native service lift and vent flights; no teleportation by the controller",
                    "Sentinel trials test phase mechanics without active sentinel damage",
                    "Designed context boundaries logged, not evaluated with a language-model memory store"]}
        save_json(directory/"reference-manifest.json",manifest)
        return rows,manifest
    finally:
        env.close()


class Video:
    def __init__(self,directory,fps):
        import cv2
        from owb.render.video import H264Writer
        self.cv=cv2
        self.fps=fps
        self.frames=0
        path=directory/"walkthrough.partial.mp4"
        self.writers=[(path,H264Writer(path,size=(1280,720),fps=fps))]

    def append(self,rgb,repeat=1):
        for _ in range(repeat):
            for _,writer in self.writers:
                writer.append_data(rgb)
            self.frames+=1

    def close(self):
        for _,writer in self.writers:
            writer.close()

    def publish(self):
        for path,_ in self.writers:
            cap=self.cv.VideoCapture(str(path))
            try:
                assert cap.isOpened(),f"Cannot decode {path}"
                assert int(cap.get(self.cv.CAP_PROP_FRAME_COUNT))==self.frames
                assert int(cap.get(self.cv.CAP_PROP_FRAME_WIDTH))==1280
                assert int(cap.get(self.cv.CAP_PROP_FRAME_HEIGHT))==720
                cap.set(self.cv.CAP_PROP_POS_FRAMES,self.frames-1)
                ok,_=cap.read()
                assert ok,"Final video frame could not be decoded"
            finally:
                cap.release()
            path.rename(path.with_name(path.name.replace(".partial","")))


def load_font(size,bold=False):
    from PIL import ImageFont
    name="DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        return ImageFont.truetype(name,size)
    except OSError:
        import matplotlib
        return ImageFont.truetype(str(Path(matplotlib.get_data_path())/"fonts"/"ttf"/name),size)


def frame_plan(rows,stride,fps):
    """Keep successful swaps and their preceding state visible during fast travel.

    Both recording paths use this plan. Holds repeat real simulator frames;
    they do not interpolate, change actions, or invent a swap animation.
    """
    from open_world_bench.party import DEFAULT_PARTY
    important={"relay_application","relay_interrupted","relay_active","sentinel_exposed",
               "sentinel_interrupted","sentinel_reformed","vent_launch","vent_landed",
               "milestone_complete","item_acquired","lift_arrived","talk","party_swap"}
    plan={}
    previous=None
    for row in rows:
        chapter=row["milestone"]!=previous
        previous=row["milestone"]
        eventful=bool({e["type"] for e in row["events"]} & important)
        if row["step"]%stride==0 or eventful or chapter:
            plan[row["step"]]={"chapter":chapter,
                "repeats":max(1,int(fps*.75)) if eventful or chapter else 1,
                "transition":None}
    active=DEFAULT_PARTY[0]
    for i,row in enumerate(rows):
        swaps=[e for e in row["events"] if e["type"]=="party_swap"]
        if not swaps:
            continue
        target=swaps[-1]["character"]
        transition={"from":active,"to":target,"phase":"after"}
        entry=plan[row["step"]]
        entry["transition"]=transition
        entry["repeats"]=max(entry["repeats"],round(fps*1.2))
        if i:
            before=plan.setdefault(rows[i-1]["step"],
                {"chapter":False,"repeats":1,"transition":None})
            # A consecutive swap keeps its own successful-swap label.
            if before["transition"] is None:
                before["transition"]={**transition,"phase":"before"}
            before["repeats"]=max(before["repeats"],round(fps*.4))
        active=target
    # Always show the final verified state, even when it is off the stride.
    if rows:
        plan.setdefault(rows[-1]["step"],{"chapter":False,"repeats":max(1,fps),"transition":None})
    return plan


class Overlay:
    def __init__(self,task,manifest,render_stride,fps,*,renderer="cycles",blender=None,samples=48,threads=8,view="presentation"):
        self.task,self.manifest,self.view=task,manifest,view
        self.speed=.2*render_stride*fps
        self.fonts={n:load_font(n) for n in (14,16,18,20,23,28)}
        self.bold={n:load_font(n,True) for n in (16,18,20,23,28)}
        self.renderer=PresentationRenderer(backend=renderer,width=864,height=486,
                                          blender=blender,samples=samples,threads=threads,view=view)

    def close(self):
        self.renderer.close()

    @property
    def metadata(self):
        return {**self.renderer.metadata,"layout":"party-continuity-v1",
                "swap_before_hold_seconds":.4,"swap_after_hold_seconds":1.2}

    def frame(self,env,row,chapter=False,transition=None):
        from PIL import Image,ImageDraw
        from open_world_bench import math3d
        image=Image.new("RGB",(1280,720),(18,43,53))
        draw=ImageDraw.Draw(image)
        accent=(222,193,129)
        title=f"{row['milestone']}  {row['title']}"
        draw.text((26,17),title,font=self.bold[28],fill=(243,243,232))
        label="Blender Cycles" if self.renderer.backend=="cycles" else "CPU diagnostic render"
        if self.view=="first_person":
            label+="  |  First-person view (character's eyes)"
        draw.text((27,55),f"TASK {self.task.upper()}  |  Verified reference course  |  Oracle controller  |  {label}",font=self.fonts[16],fill=(174,206,202))
        rgb=self.renderer.render(env)
        image.paste(Image.fromarray(rgb),(24,94))
        # World-anchored annotations explain the actual mechanism state to viewers.
        op=env.course.op
        if op:
            from owb.render.scene_export import teyvat_camera
            view=math3d.look_at(*teyvat_camera(env,self.view))
            projection=math3d.perspective(env.cfg.camera.fov_y,env.cfg.render.aspect,.08,900)
            for eid in op.entities:
                entity=env.world.get(eid)
                if not entity.alive:
                    continue
                point=np.r_[entity.pos+np.array([0,3.0,0]),1]
                clip=projection@view@point
                if clip[3]<=0:
                    continue
                ndc=clip[:3]/clip[3]
                if not(-.96<ndc[0]<.96 and -.9<ndc[1]<.9):
                    continue
                x=int(24+(ndc[0]+1)*432);y=int(94+(1-ndc[1])*243)
                label=entity.label
                if op.kind=="relay":
                    label=next(k for k,v in op.state["roles"].items() if v==eid).upper()
                w=draw.textbbox((0,0),label,font=self.fonts[14])[2]
                draw.rounded_rectangle((x-w//2-5,y-3,x+w//2+5,y+19),4,fill=(22,49,57))
                draw.text((x-w//2,y),label,font=self.fonts[14],fill=(255,239,196))
        x,y=910,103
        draw.text((x,y),"ONE CONTROLLER / PARTY",font=self.bold[16],fill=accent);y+=28
        for i,member in enumerate(env.party.members):
            active=i==env.party.active_index
            color=tuple(round(c*255) for c in member.spec.tunic)
            draw.rounded_rectangle((x,y,1255,y+39),5,
                fill=(41,76,83) if active else (23,49,59),
                outline=accent if active else (53,78,85),width=2 if active else 1)
            draw.rectangle((x+9,y+10,x+15,y+29),fill=color)
            draw.text((x+25,y+3),member.spec.name,font=self.bold[16],fill=(247,246,233))
            draw.text((x+25,y+22),"ACTIVE" if active else "In party",font=self.fonts[14],fill=accent if active else (160,186,190))
            draw.text((x+226,y+10),member.spec.element.name.title(),font=self.fonts[16],fill=(227,235,229))
            y+=44
        y+=12
        draw.text((x,y),"WORLD STATE",font=self.bold[18],fill=accent);y+=29
        state_lines=[f"Completed: {len(env.course.completed)} / {len(env.course.milestones)}",
                     f"Time: {env.world.elapsed:.1f} s | Step: {row['step']}",
                     f"HP: {env.char.hp:.0f} | Stamina: {env.char.stamina:.0f}",
                     f"Movement: {env.char.state.name.lower()}"]
        if op:
            s=op.state
            state_lines+= ["",f"Mechanism: {op.kind}"]
            if op.kind=="relay":
                state_lines += [f"Version: {s['version']}",f"Progress: {['unprimed','primed','routed'][s['progress']]}",
                                f"Overload: {'yes' if s['overloaded'] else 'no'}",f"Drained: {'yes' if s['drained'] else 'no'}"]
            elif op.kind=="sentinel":
                state_lines += [f"Phase: {s['phase']}",f"Sentinel HP: {env.world.get(op.entities[0]).hp:.0f}"]
            elif op.kind=="crossing":
                state_lines += [f"Leg: {s['leg']+1} / 2",f"Charged: {s['charged']}",
                                f"Pulse: {'BRIGHT' if 1<=env.world.elapsed%4<=3.5 else 'dim'}",
                                f"Landings: {s['landings']}"]
        else:
            state_lines += ["","ALL MILESTONES COMPLETE"]
        for value in state_lines:
            draw.text((x,y),value,font=self.fonts[18],fill=(227,235,229));y+=23
        if transition:
            from open_world_bench.party import ROSTER
            old,new=ROSTER[transition["from"]],ROSTER[transition["to"]]
            prefix="NEXT SWAP" if transition["phase"]=="before" else "PARTY SWAP"
            draw.rounded_rectangle((40,510,870,569),7,fill=(23,55,63),outline=accent,width=2)
            draw.text((55,517),f"{prefix}: {old.name} → {new.name}",font=self.bold[20],fill=(247,246,233))
            ability=f"Upcoming: {new.element.name.title()} ability" if transition["phase"]=="before" else f"{new.element.name.title()} ability selected"
            draw.text((55,544),f"{ability} | Same controller, same journey",
                      font=self.fonts[16],fill=accent)
        draw.rectangle((24,599,1255,666),fill=(29,65,73))
        events=[e for e in row['events'] if e['type']=='milestone_complete']
        message=("VERIFIED: "+events[-1]['milestone']+" completed") if events else row['explanation']
        draw.text((39,610),"REFERENCE ACTION" if not events else "COMPLETION CHECK",font=self.bold[14] if 14 in self.bold else self.fonts[14],fill=accent)
        for k,value in enumerate(textwrap.wrap(message,width=102)[:2]):
            draw.text((39,633+k*22),value,font=self.fonts[20],fill=(247,246,233))
        draw.text((26,690),f"Compact prototype | Travel shown at ~{self.speed:g}x; event pauses | Complete action log + replay verification",font=self.fonts[14],fill=(169,192,190))
        if chapter:
            # A quiet objective chip: the header already names the milestone, so the
            # in-world cue stays small, translucent, and out of the scene's centre.
            chip=Image.new("RGBA",image.size,(0,0,0,0))
            label="Next  "+row['milestone']
            w=draw.textbbox((0,0),label,font=self.fonts[14])[2]
            ImageDraw.Draw(chip).rounded_rectangle((36,104,36+w+20,128),5,fill=(16,38,45,165))
            image=Image.alpha_composite(image.convert("RGBA"),chip).convert("RGB")
            draw=ImageDraw.Draw(image)
            draw.text((46,108),label,font=self.fonts[14],fill=accent)
        return np.asarray(image)


def replay_video(task,seed,directory,rows,manifest,*,stride=5,fps=10,renderer="cycles",blender=None,samples=48,threads=8):
    from PIL import Image
    env=ReferenceTaskEnv(task,seed,render_enabled=False,width=384,height=216)
    overlay=Overlay(task,manifest,stride,fps,renderer=renderer,blender=blender,samples=samples,threads=threads)
    writer=Video(directory,fps)
    chapters=[]
    preview=[]
    last_mid=None
    start=time.monotonic()
    plan=frame_plan(rows,stride,fps)
    try:
        env.reset(seed=seed)
        assert env.state_digest()==manifest["initial_digest"]
        for row in rows:
            action={k:np.asarray(v,dtype=np.int8 if k=="buttons" else np.float32) if k in ("move","look","buttons") else v
                    for k,v in row["action"].items()}
            new_chapter=row["milestone"]!=last_mid
            if new_chapter:
                if chapters:
                    chapters[-1]["end_seconds"]=writer.frames/fps
                chapters.append({"id":row["milestone"],"title":row["title"],"start_seconds":writer.frames/fps,
                                 "simulation_start":env.world.elapsed,"action_start":row["step"]})
                last_mid=row["milestone"]
            _,_,_,_,info=env.step(action)
            if "state_digest" in row:
                assert env.state_digest()==row["state_digest"],f"Replay diverged at {row['milestone']}"
            kinds={e["type"] for e in row["events"]}
            if row["step"] in plan:
                entry=plan[row["step"]]
                frame=overlay.frame(env,row,chapter=entry["chapter"],transition=entry["transition"])
                repeats=entry["repeats"]
                writer.append(frame,repeats)
                if row["milestone"] in ({"A03"} if task=="a" else {"B19"}) and len(preview)<90:
                    small=Image.fromarray(frame).resize((640,360),Image.Resampling.LANCZOS)
                    preview.append((small,round(1000*repeats/fps)))
                if new_chapter and row["milestone"] in {"A03","B19"}:
                    Image.fromarray(frame).save(directory/"preview.png")
            if "milestone_complete" in kinds:
                print(f"Task {task.upper()}: replayed {row['milestone']} ({time.monotonic()-start:.0f}s rendering)",flush=True)
        assert env.state_digest()==manifest["final_digest"]
        assert env.course.verify()["success"]
        chapters[-1]["end_seconds"]=writer.frames/fps
        writer.close()
        writer.publish()
        save_json(directory/"chapters.json",chapters)
        if preview:
            preview[0][0].save(directory/"preview.gif",save_all=True,append_images=[p for p,_ in preview[1:]],
                               duration=[d for _,d in preview],loop=0,optimize=True)
        summary={**manifest,"replay_verified":True,"video_frames":writer.frames,"video_fps":fps,
                 "presentation":overlay.metadata,
                 "video_seconds":writer.frames/fps,"render_stride":stride,
                 "render_wall_seconds":time.monotonic()-start,
                 "actions_sha256":hashlib.sha256((directory/"actions.jsonl").read_bytes()).hexdigest()}
        save_json(directory/"summary.json",summary)
        return summary
    finally:
        writer.close()
        overlay.close()
        env.close()


def verify_recording(directory):
    """Independently replay every action, compare events/positions and final state."""
    manifest=json.loads((directory/"reference-manifest.json").read_text())
    rows=[json.loads(line) for line in (directory/"actions.jsonl").read_text().splitlines()]
    env=ReferenceTaskEnv(manifest["task"],manifest["seed"],render_enabled=False)
    try:
        env.reset(seed=manifest["seed"])
        assert env.state_digest()==manifest["initial_digest"]
        for row in rows:
            action={k:np.asarray(v,dtype=np.int8 if k=="buttons" else np.float32) if k in ("move","look","buttons") else v
                    for k,v in row["action"].items()}
            assert env.action_space.contains(action)
            _,_,_,_,info=env.step(action)
            assert env.char.pos.tolist()==row["position_after"],f"Position mismatch at step {row['step']}"
            assert info["event_details"]==row["events"],f"Event mismatch at step {row['step']}"
            if "state_digest" in row:
                assert env.state_digest()==row["state_digest"]
        assert env.state_digest()==manifest["final_digest"]
        assert env.quests.all_complete and env.course.verify()["success"]
        root=Path(__file__).resolve().parents[2]
        sources=["open_world_bench/reference_tasks.py","open_world_bench/reference_policy.py",
                 "open_world_bench/architecture.py","open_world_bench/render/architecture.py",
                 "open_world_bench/camera.py",
                 "open_world_bench/env.py","open_world_bench/world.py","open_world_bench/character.py",
                 "open_world_bench/party.py","open_world_bench/terrain.py","owb/scripts/teyvat_reference.py",
                 "owb/render/presentation.py","owb/render/scene_export.py",
                 "owb/render/blender_scene.py","owb/render/video.py"]
        result={"success":True,"checked_actions":len(rows),"checked_events":True,
                "checked_positions":True,"checked_milestone_hashes":len(manifest["milestones"]),
                "runtime":{"python":platform.python_version(),"numpy":np.__version__},
                "verification":env.course.verify(),
                "actions_sha256":hashlib.sha256((directory/"actions.jsonl").read_bytes()).hexdigest(),
                "validated_source_sha256":{p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in sources},
                "video_sha256":{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in directory.glob("walkthrough.*")}}
        save_json(directory/"verification.json",result)
        return result
    finally:
        env.close()


def render_chunk(directory,cache,first,last,stride=8,fps=10,*,renderer="cycles",blender=None,samples=48,threads=8,max_frames=None,view="presentation"):
    """Replay an interval, resuming only frames matching its source/state contract."""
    from PIL import Image
    manifest=json.loads((directory/"reference-manifest.json").read_text())
    rows=[json.loads(s) for s in (directory/"actions.jsonl").read_text().splitlines()]
    task,seed=manifest["task"],manifest["seed"]
    total=manifest["verification"]["required"]
    if not 1<=first<=last<=total:
        raise ValueError("invalid milestone interval")
    folder=cache/f"{directory.name}-{first:02d}-{last:02d}"
    folder.mkdir(parents=True,exist_ok=True)
    env=ReferenceTaskEnv(task,seed,render_enabled=False)
    overlay=Overlay(task,manifest,stride,fps,renderer=renderer,blender=blender,samples=samples,threads=threads,view=view)
    entries=[]
    previous=None
    plan=frame_plan(rows,stride,fps)
    try:
        contract={"first":first,"last":last,"fps":fps,"stride":stride,
                  "presentation":overlay.metadata,
                  "trace_sha256":hashlib.sha256((directory/"actions.jsonl").read_bytes()).hexdigest(),
                  "sources":render_source_hashes()}
        contract_path=folder/"render-plan.json"
        if contract_path.exists():
            if json.loads(contract_path.read_text())!=contract:
                raise RuntimeError("Cannot resume: render settings, source files, or trace changed")
        else:
            if any(folder.iterdir()):
                raise RuntimeError("Existing cache has no verified render contract; use a new cache")
            save_json(contract_path,contract)
        progress=folder/"progress.json"
        saved={e["step"]:e for e in json.loads(progress.read_text())} if progress.exists() else {}
        rendered=0
        env.reset(seed=seed)
        assert env.state_digest()==manifest["initial_digest"]
        for row in rows:
            number=int(row["milestone"][1:])
            if number>last:
                break
            new=row["milestone"]!=previous
            previous=row["milestone"]
            before=env.world.elapsed
            action={k:np.asarray(v,dtype=np.int8 if k=="buttons" else np.float32) if k in ("move","look","buttons") else v
                    for k,v in row["action"].items()}
            _,_,_,_,info=env.step(action)
            assert env.char.pos.tolist()==row["position_after"],f"Position mismatch at step {row['step']}"
            assert info["event_details"]==row["events"],f"Event mismatch at step {row['step']}"
            if "state_digest" in row:
                assert env.state_digest()==row["state_digest"]
            if number<first:
                continue
            kinds={e["type"] for e in row["events"]}
            if row["step"] in plan:
                entry=plan[row["step"]]
                digest=env.state_digest()
                if row["step"] in saved:
                    cached=saved[row["step"]]
                    if cached["state_digest"]!=digest or hashlib.sha256((folder/cached["file"]).read_bytes()).hexdigest()!=cached["sha256"]:
                        raise RuntimeError("Cached frame no longer matches its replay state or image hash")
                    entries.append(cached)
                    continue
                if max_frames is not None and rendered>=max_frames:
                    return False
                frame=overlay.frame(env,row,chapter=entry["chapter"],transition=entry["transition"])
                assert env.state_digest()==digest,"Rendering changed the simulation"
                filename=f"{row['step']:06d}.png"
                partial=folder/(filename+".partial")
                Image.fromarray(frame).save(partial,format="PNG")
                partial.replace(folder/filename)
                entries.append({"file":filename,"step":row["step"],"milestone":row["milestone"],
                                "title":row["title"],"new_chapter":new,"simulation_start":before,
                                "repeats":entry["repeats"],"transition":entry["transition"],
                                "state_digest":digest,"sha256":hashlib.sha256((folder/filename).read_bytes()).hexdigest()})
                partial_progress=folder/"progress.partial.json"
                save_json(partial_progress,entries)
                partial_progress.replace(progress)
                rendered+=1
                print(f"Task {task.upper()} {row['milestone']}: saved action {row['step']} ({len(entries)} frames in chunk)",flush=True)
            if "milestone_complete" in kinds:
                print(f"Rendered chunk {first:02d}-{last:02d}: {row['milestone']}",flush=True)
        if last==total:
            assert env.state_digest()==manifest["final_digest"]
            assert env.course.verify()["success"] and env.quests.all_complete
        save_json(folder/"index.json",{**contract,
                 "checkpoint":env.state_digest(),"frames":entries})
        return True
    finally:
        overlay.close()
        env.close()


def assemble_chunks(directory,cache,*,renderer="cycles"):
    from PIL import Image
    manifest=json.loads((directory/"reference-manifest.json").read_text())
    indices=sorted(cache.glob(f"{directory.name}-*/index.json"))
    pieces=[(p.parent,json.loads(p.read_text())) for p in indices]
    assert pieces,"No completed render chunks"
    expected=1
    fps=pieces[0][1]["fps"]
    stride=pieces[0][1]["stride"]
    presentation=pieces[0][1].get("presentation",{"backend":"cpu","legacy":True})
    assert presentation["backend"]==renderer,"Cached frames use a different renderer; rerender them or explicitly select that backend"
    digest=hashlib.sha256((directory/"actions.jsonl").read_bytes()).hexdigest()
    for _,piece in pieces:
        assert piece["first"]==expected,"Render intervals must be contiguous and disjoint"
        assert piece["trace_sha256"]==digest and piece["fps"]==fps and piece["stride"]==stride
        assert piece.get("presentation",{"backend":"cpu","legacy":True})==presentation,"Cannot mix render settings"
        assert piece.get("sources")==pieces[0][1].get("sources"),"Cannot mix render source versions"
        expected=piece["last"]+1
    assert expected==manifest["verification"]["required"]+1
    assert pieces[-1][1]["checkpoint"]==manifest["final_digest"]
    assert not (directory/"walkthrough.mp4").exists(),"Refusing to overwrite a published recording"
    writer=Video(directory,fps)
    chapters=[]
    preview=[]
    try:
        for folder,piece in pieces:
            for entry in piece["frames"]:
                if "sha256" in entry:
                    assert hashlib.sha256((folder/entry["file"]).read_bytes()).hexdigest()==entry["sha256"],"Cached image changed after rendering"
                if entry["new_chapter"]:
                    if chapters:
                        chapters[-1]["end_seconds"]=writer.frames/fps
                    chapters.append({"id":entry["milestone"],"title":entry["title"],
                                     "start_seconds":writer.frames/fps,"simulation_start":entry["simulation_start"],
                                     "action_start":entry["step"]})
                with Image.open(folder/entry["file"]) as im:
                    writer.append(np.asarray(im.convert("RGB")),entry["repeats"])
                    preview_id="A03" if manifest["task"]=="a" else "B19"
                    if entry["milestone"]==preview_id and len(preview)<90:
                        preview.append((im.resize((640,360),Image.Resampling.LANCZOS).copy(),entry["repeats"]*1000//fps))
                        if entry["new_chapter"]:
                            im.save(directory/"preview.png")
            print(f"Encoded through milestone {piece['last']}",flush=True)
        chapters[-1]["end_seconds"]=writer.frames/fps
        writer.close()
        writer.publish()
        save_json(directory/"chapters.json",chapters)
        if preview:
            preview[0][0].save(directory/"preview.gif",save_all=True,append_images=[im for im,_ in preview[1:]],
                               duration=[d for _,d in preview],loop=0,optimize=True)
        summary={**manifest,"replay_verified":True,"render_method":"verified contiguous chunks",
                 "presentation":presentation,
                 "render_sources":pieces[0][1].get("sources"),
                 "video_frames":writer.frames,"video_fps":fps,"video_seconds":writer.frames/fps,
                 "render_stride":stride,"actions_sha256":digest}
        save_json(directory/"summary.json",summary)
        print(json.dumps({"success":True,"video_seconds":summary["video_seconds"]}),flush=True)
    finally:
        writer.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task",nargs="+",choices=("a","b"),default=["a","b"])
    parser.add_argument("--seed",type=int,default=7)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--output",type=Path)
    mode.add_argument("--verify",type=Path,help="Replay a recorded action log without rendering")
    mode.add_argument("--render-chunk",type=Path,help="Render a range from an existing verified trace")
    mode.add_argument("--assemble",type=Path,help="Assemble completed chunks for this recording")
    parser.add_argument("--milestones",nargs=2,type=int)
    parser.add_argument("--cache",type=Path,default=Path(tempfile.gettempdir())/"owb-teyvat-render-cache")
    parser.add_argument("--render-stride",type=int,default=5)
    parser.add_argument("--fps",type=int,default=10)
    parser.add_argument("--max-steps",type=int,default=9000)
    parser.add_argument("--max-render-frames",type=int,help="Save this many new frames per chunk invocation, then resume with the same command")
    parser.add_argument("--view",choices=("presentation","first_person"),default="presentation",
                        help="presentation: the policy's orbit camera; first_person: the character's eyes (--render-chunk only)")
    add_render_arguments(parser)
    args=parser.parse_args()
    if min(args.render_stride,args.fps,args.max_steps,args.samples,args.threads)<=0:
        parser.error("stride, fps, max-steps, samples, and threads must be positive")
    if args.max_render_frames is not None and args.max_render_frames<=0:
        parser.error("max-render-frames must be positive")
    rendering=dict(renderer=args.renderer,blender=args.blender,samples=args.samples,threads=args.threads)
    if args.render_chunk:
        if not args.milestones:
            parser.error("--render-chunk requires --milestones FIRST LAST")
        render_chunk(args.render_chunk,args.cache,*args.milestones,args.render_stride,args.fps,
                     max_frames=args.max_render_frames,view=args.view,**rendering)
        return
    if args.assemble:
        assemble_chunks(args.assemble,args.cache,renderer=args.renderer)
        return
    if args.verify:
        result=verify_recording(args.verify)
        print(json.dumps({"success":result["success"],"actions":result["checked_actions"],
                          "milestones":result["checked_milestone_hashes"]}),flush=True)
        return
    # Fail before running the controller if presentation tools are unavailable.
    with PresentationRenderer(backend=args.renderer,blender=args.blender,
                              samples=args.samples,threads=args.threads):
        pass
    from owb.render.video import resolve_ffmpeg
    resolve_ffmpeg()
    for task in args.task:
        directory=args.output/f"task-{task}-seed-{args.seed}"
        directory.mkdir(parents=True,exist_ok=False)
        rows,manifest=collect_reference(task,args.seed,directory,args.max_steps)
        summary=replay_video(task,args.seed,directory,rows,manifest,stride=args.render_stride,fps=args.fps,**rendering)
        print(json.dumps({"task":task,"success":summary["verification"]["success"],
                          "replay_verified":summary["replay_verified"],"video_seconds":summary["video_seconds"]}),flush=True)


if __name__=="__main__":
    main()
