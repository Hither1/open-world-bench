"""Collect, replay-verify, and render Wildlands reference walkthroughs in Cycles."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import textwrap
import time
import numpy as np
from owb.env.wildlands_reference import WildlandsReferenceEnv, plain, COMMANDS
from owb.bench.baselines.wildlands_reference import WildlandsReferencePolicy
from owb.render.cycles_batch import CyclesBatchRenderer
from owb.render.video import H264Writer

ROOT=Path(__file__).resolve().parents[2]
SOURCES=('owb/env/wildlands_reference.py','owb/bench/baselines/wildlands_reference.py',
         'owb/scripts/wildlands_reference.py','owb/env/wildlands.py','owb/physics/character.py',
         'owb/physics/bodies.py','owb/physics/elements.py','owb/render/scene_export.py',
         'owb/render/blender_scene.py','owb/render/dressing.py','owb/render/presentation.py','owb/render/video.py',
         'owb/render/cycles_batch.py','owb/render/blender_batch.py',
         'owb/world/organic.py','owb/world/fauna.py','owb/entities/npc.py','owb/entities/agent.py',
         'owb/sim/riding.py','owb/sim/orchard.py','owb/render/creatures.py','owb/sim/survival.py','owb/world/relief.py')
LIMITATIONS=[
 'Compact authored course within procedural Wildlands terrain; shortened travel, not the full long-duration benchmark.',
 'Privileged reference controller; no claim that an agent learned skills or retained memories.',
 'Counterweights use native magnetic bodies and supported-mass sensors; fixed gate/lift actuators are authored mechanisms.',
 'Thermal flights use native jump/glide physics with authored shaft updraft fields and measured service-deck landings.',
 'Flow and cooling are discrete authored networks, not a general fluid solver.',
 'Training procedures are readable local instructions plus observable practice, not a separately animated teacher.',
 'Role mappings, material/shape transfer variants, regional temperature exposure, and learning-from-memory ablations remain incomplete.',
 'Repair parts use a separate persistent satchel; ordinary food uses native inventory and cooking.',
 'The road and bridge really change voxel collision; no controller teleportation or direct completion edits.',
]


def save(path,data): path.write_text(json.dumps(plain(data),indent=2,allow_nan=False)+'\n')
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def restore_action(a):
    out=dict(a)
    for key in ('move','look','vector'): out[key]=np.array(out[key],np.float32)
    out['buttons']=np.array(out['buttons'],np.int8)
    out['ingredients']=np.array(out['ingredients'],np.int64)
    return out


def collect(task,seed,directory):
    directory.mkdir(parents=True,exist_ok=False)
    env=WildlandsReferenceEnv(task,seed); obs,_=env.reset(seed=seed)
    policy=WildlandsReferencePolicy(env); initial=env.state_digest(); counts=Counter()
    try:
        with (directory/'actions.jsonl').open('w') as stream:
            while not env._ended:
                m=env.current; before=env.agent.pos.copy(); action=policy(obs)
                assert env.action_space.contains(action)
                obs,r,t,tr,info=env.step(action)
                counts.update(e['type'] for e in info['events'])
                row=dict(step=env.steps,milestone=m.id,title=m.title,kind=m.kind,
                         explanation=policy.explanation,action=plain(action),position_before=plain(before),
                         position_after=plain(env.agent.pos),seconds=env.elapsed,events=plain(info['events']),reward=r)
                if any(e['type']=='milestone_complete' for e in info['events']):
                    row['digest']=env.state_digest()
                    print(f'{task.upper()} {env.steps}: {len(env.completed)}/{len(env.milestones)} verified',flush=True)
                stream.write(json.dumps(row,allow_nan=False)+'\n')
        if not env.success or env.agent.deaths: raise RuntimeError('Reference failed or died')
        manifest=dict(task=task,seed=seed,controller='privileged native-action reference policy',
            initial_digest=initial,final_digest=env.state_digest(),decisions=env.steps,
            simulated_seconds=env.elapsed,milestones=env.completed,required=len(env.milestones),
            success=True,events=dict(counts),health=env.agent.health,limitations=LIMITATIONS,
            sources={p:sha(ROOT/p) for p in SOURCES},actions_sha256=sha(directory/'actions.jsonl'))
        save(directory/'reference-manifest.json',manifest)
        return verify(directory)
    finally: env.close()


def read(directory):
    return json.loads((directory/'reference-manifest.json').read_text()),[json.loads(r) for r in (directory/'actions.jsonl').read_text().splitlines()]


def replay(directory):
    manifest,rows=read(directory)
    if sha(directory/'actions.jsonl')!=manifest['actions_sha256']: raise RuntimeError('Action log hash mismatch')
    env=WildlandsReferenceEnv(manifest['task'],manifest['seed']); env.reset(seed=manifest['seed'])
    if env.state_digest()!=manifest['initial_digest']: raise RuntimeError('Initial state differs')
    return env,manifest,rows


def checked_step(env,row):
    if env.agent.pos.tolist()!=row['position_before']: raise RuntimeError(f'Before-position mismatch at {row["step"]}')
    _,r,_,_,info=env.step(restore_action(row['action']))
    if env.agent.pos.tolist()!=row['position_after'] or plain(info['events'])!=row['events'] or r!=row['reward']:
        raise RuntimeError(f'Replay mismatch at action {row["step"]}')
    if 'digest' in row and env.state_digest()!=row['digest']: raise RuntimeError(f'Milestone digest mismatch: {row["milestone"]}')


def verify(directory):
    env,manifest,rows=replay(directory)
    try:
        for row in rows: checked_step(env,row)
        assert env.success and len(env.completed)==manifest['required'] and not env.agent.deaths
        assert env.state_digest()==manifest['final_digest']
        result=dict(success=True,checked_actions=len(rows),checked_positions=True,checked_events=True,
            checked_rewards=True,checked_milestones=len(env.completed),final_digest=env.state_digest(),
            actions_sha256=sha(directory/'actions.jsonl'),sources={p:sha(ROOT/p) for p in SOURCES})
        save(directory/'verification.json',result)
        return result
    finally: env.close()


def font(size,bold=False):
    from PIL import ImageFont
    import matplotlib
    name='DejaVuSans-Bold.ttf' if bold else 'DejaVuSans.ttf'
    return ImageFont.truetype(str(Path(matplotlib.get_data_path())/'fonts/ttf'/name),size)


class Overlay:
    def __init__(self,task,samples=48,threads=8):
        self.renderer=CyclesBatchRenderer(width=960,height=540,samples=samples,threads=threads)
        self.task=task; self.fonts={n:font(n) for n in (14,16,18,20,24)}; self.title=font(26,True)

    def submit(self,env,row):
        """Export this state for Cycles and capture the caption panel; does not wait for the render."""
        env.reference_view_id=row['milestone']
        before=env.state_digest()
        ticket=self.renderer.submit(env)
        assert env.state_digest()==before,'Presentation must not change simulation'
        return ticket,self.panel(env,row)

    def result(self,pending):
        ticket,panel=pending
        return self.compose(panel,self.renderer.result(ticket))

    def frame(self,env,row): return self.result(self.submit(env,row))

    def panel(self,env,row):
        m=env.by_id[row['milestone']]; s=m.state
        lines=[f'Completed {len(env.completed)} / {len(env.milestones)}',f'Simulation {env.elapsed:.1f} s',
               f'Action {env.steps}',f'Health {env.agent.health:.1f}',f'Stamina {env.agent.stamina:.0f}',
               f'Weather: {env.elements.climate.weather}']
        cl=env.elements.climate; hour=(cl.time*24)%24
        out=sum(1 for n in env.npcs.npcs if n.alive and n.data.get('den') and not n.data.get('denned'))
        dens=sum(1 for n in env.npcs.npcs if n.alive and n.data.get('den'))
        ride=env.riding.state()
        lines += [f'Time {int(hour):02}:{int(hour%1*60):02} ({"night" if cl.is_night else "day"})',
                  f'Monsters out: {out} / {dens}',
                  f'Riding: {ride["gait"]}, {ride["distance"]:.0f} m'+(' (mounted)' if ride['mounted'] else ''),
                  '', 'MECHANISM']
        if m.kind=='balance': lines += [f'Load: {s["load"]:.0f} kg',f'Target: {s["count"]*320} kg',f'Brake held: {s["braked"]}',f'Released: {s["released"]}',f'Stable: {s["stable"]:.1f} s']
        elif m.kind=='thermal': lines += [f'Draft open: {s["vent"]}',f'Heat source: {s["heat"]}',f'Launch safe: {s["safe"]}',f'Updraft used: {s["updraft_seen"]}',f'Deck reached: {s["landed"]}']
        elif m.kind in ('flow','cool','prepare'): lines += [f'Inlet: {"OPEN" if s["inlet"] else "CLOSED"}',f'Outlets: '+''.join('O' if v else '-' for v in s['outlets']),f'Stable: {s["stable"]:.1f} s',f'Temperature: {s["temperature"]:.0f} C']
        elif m.kind=='calibrate': lines += [f'Parts installed: {s["installed"]}',f'Mark selected: {s["selected"]}',f'Stable: {s["elapsed"]:.1f} s']
        else: lines += [f'Parts: {sum(env.satchel.values())}',f'Stations online: {len(env.online)}',f'Road drained: {env.drained}',f'Bridge closed: {env.storm}']
        complete=any(ev['type']=='milestone_complete' for ev in row['events'])
        caption=('VERIFIED: '+row['milestone']+' complete') if complete else row['explanation']
        return dict(heading=row['milestone']+'  '+row['title'],lines=lines,caption=caption)

    def compose(self,panel,rgb):
        from PIL import Image,ImageDraw
        image=Image.new('RGB',(1280,720),(18,39,46)); d=ImageDraw.Draw(image)
        d.text((22,14),panel['heading'],font=self.title,fill=(247,237,211))
        d.text((23,54),'WILDLANDS  |  Executed reference course  |  Privileged controller  |  Blender Cycles',font=self.fonts[16],fill=(161,201,194))
        image.paste(Image.fromarray(rgb),(16,93))
        for i,line in enumerate(panel['lines']): d.text((993,104+i*27),line,font=self.fonts[16],fill=(222,231,223))
        caption=panel['caption']
        d.rectangle((16,646,1263,688),fill=(32,66,73))
        d.text((29,655),caption[:105],font=self.fonts[20],fill=(247,237,211))
        d.text((22,698),'Selected simulator states; held for reading | Compact oracle course | Complete actions supplied for replay',font=self.fonts[14],fill=(159,188,183))
        return np.asarray(image)

    def close(self): self.renderer.close()


IMPORTANT={'brake_trip','magnetic_grip','magnetic_drop','counterweight_control','counterweight_open',
 'thermal_control','thermal_landed','clue_exposed','item_exchange','parts_installed','flow_control',
 'milestone_complete','road_drained','station_engaged','cook','gather','light_campfire','bridge_closed','intake_engaged','calibration_selected'}


HOLD,MOTION=30,12   # encoded frames per checkpoint still / per motion moment


def checkpoint_steps(rows):
    """Select actual procedural transitions for a clearly labelled checkpoint film."""
    selected=set()
    for mid in dict.fromkeys(r['milestone'] for r in rows):
        group=[r for r in rows if r['milestone']==mid]
        selected.add(group[-1]['step'])
        kind=group[0]['kind']
        event_types={
            'balance':('brake_trip','magnetic_grip','magnetic_drop','counterweight_control'),
            'thermal':('thermal_control','thermal_landed','parts_installed'),
            'flow':('parts_installed','flow_control'),
            'cool':('flow_control',),'prepare':('flow_control',),
            'calibrate':('parts_installed','calibration_selected','intake_engaged'),
            'station':('station_engaged',),'cook':('cook','light_campfire'),
        }.get(kind,())
        seen=set()
        for row in group:
            for event in row['events']:
                if event['type'] not in event_types: continue
                key=(event['type'],event.get('control'),tuple(event.get('effects',[])))
                # One illustrated lift/drop is enough; the log retains every weight.
                if key not in seen:
                    selected.add(row['step']); seen.add(key)
        if kind=='report' and len(group)>2:
            selected.add(group[len(group)//2]['step'])
    holds={step:HOLD for step in selected}
    # Motion moments, held briefly: glides on the updraft, rides on the roads,
    # learning to handle the horse, and dusk bringing the monsters out.
    flight,ride=[],[]
    for row in rows+[None]:
        gliding=row is not None and bool(row['action']['buttons'][3])
        riding=row is not None and row['explanation'].startswith('Ride the access road')
        if flight and not gliding:
            for f in (.25,.5,.75): holds.setdefault(flight[int(len(flight)*f)]['step'],MOTION)
            flight=[]
        if ride and not riding:
            if len(ride)>=6:
                for f in (.3,.7): holds.setdefault(ride[int(len(ride)*f)]['step'],MOTION)
            ride=[]
        if gliding: flight.append(row)
        if riding: ride.append(row)
    first_seen=set()
    for row in rows:
        for event in row['events']:
            kind=event['type']
            if kind in ('horse_calmed','mounted','riding_improved','equip') or (kind=='npc_emerged' and kind not in first_seen):
                if kind=='mounted' and kind in first_seen: continue
                holds.setdefault(row['step'],HOLD if kind!='mounted' else MOTION)
                first_seen.add(kind)
    return holds


def render_chunk(directory,cache,first,last,*,stride=12,fps=10,samples=48,threads=8,checkpoints=False):
    from PIL import Image
    env,manifest,rows=replay(directory)
    if not 1<=first<=last<=manifest['required']: raise ValueError('Invalid milestone interval')
    folder=cache/f'{directory.name}-{first:02}-{last:02}'
    folder.mkdir(parents=True,exist_ok=True)
    overlay=Overlay(manifest['task'],samples,threads); entries=[]; previous=None
    selected=checkpoint_steps(rows) if checkpoints else None
    contract=dict(first=first,last=last,fps=fps,stride=stride,checkpoints=checkpoints,
                  presentation=overlay.renderer.metadata,trace_sha256=manifest['actions_sha256'],
                  sources={p:sha(ROOT/p) for p in SOURCES})
    plan=folder/'render-plan.json'
    if plan.exists():
        if json.loads(plan.read_text())!=contract: raise RuntimeError('Resume settings or source files changed')
    else: save(plan,contract)
    progress=folder/'progress.json'
    previous_frames={r['step']:r for r in json.loads(progress.read_text())} if progress.exists() else {}
    started=time.monotonic(); pending=None

    def finish():
        # Collect the frame queued earlier; Blender renders it while the replay continues.
        nonlocal pending
        if pending is None: return
        job,row,step,digest,new,important=pending; pending=None
        rgb=overlay.result(job); name=f'{step:06}.png'
        Image.fromarray(rgb).save(folder/name)
        entries.append(dict(file=name,sha256=sha(folder/name),step=step,id=row['milestone'],title=row['title'],
            seconds=row['seconds'],new=new,repeats=selected[step] if checkpoints else (8 if important or new else 1),digest=digest))
        save(progress.with_suffix('.partial'),entries)
        progress.with_suffix('.partial').replace(progress)
        print(f'{row["milestone"]} action {step}: saved Cycles frame; {time.monotonic()-started:.0f} s',flush=True)
    try:
        for row in rows:
            number=int(row['milestone'][1:])
            if number>last: break
            new=row['milestone']!=previous; previous=row['milestone']
            checked_step(env,row)
            if number<first: continue
            kinds={e['type'] for e in row['events']}
            important=bool(kinds&IMPORTANT)
            flight=row['kind']=='thermal' and (env.agent.gliding or abs(env.agent.vel[1])>2)
            movement=row['action']['command']==COMMANDS.index('drag')
            capture=(row['step'] in selected) if checkpoints else (new or important or row['step']%stride==0 or (flight and row['step']%2==0) or (movement and row['step']%4==0))
            if capture:
                saved=previous_frames.get(row['step'])
                if saved:
                    if saved['digest']!=env.state_digest() or sha(folder/saved['file'])!=saved['sha256']:
                        raise RuntimeError('Saved frame no longer matches replay')
                    finish(); entries.append(saved)
                    print(f'{row["milestone"]} action {row["step"]}: reused verified Cycles frame',flush=True)
                else:
                    job=overlay.submit(env,row)
                    finish()
                    pending=(job,row,row['step'],env.state_digest(),new,important)
            if 'milestone_complete' in kinds:
                print(f'{directory.name}: replayed {row["milestone"]}; {len(entries)} frames saved; {time.monotonic()-started:.0f} s',flush=True)
        finish()
        timings=getattr(overlay.renderer,'timings',None)
        if timings:
            build=[t['build_seconds'] for t in timings]; render=[t['render_seconds'] for t in timings]
            print(f'Cycles timing over {len(timings)} frames: scene build mean {np.mean(build):.1f} s, '
                  f'render+denoise mean {np.mean(render):.1f} s; wall {time.monotonic()-started:.0f} s',flush=True)
        if last==manifest['required']:
            assert env.success and env.state_digest()==manifest['final_digest']
        save(folder/'index.json',dict(**contract,frames=entries,checkpoint=env.state_digest()))
    finally: overlay.close(); env.close()


def assemble(directory,cache):
    import cv2
    from PIL import Image
    manifest,_=read(directory)
    pieces=sorted((p.parent,json.loads(p.read_text())) for p in cache.glob(f'{directory.name}-*/index.json'))
    if not pieces: raise ValueError('No completed chunks')
    expected=1; settings=pieces[0][1]; fps=settings['fps']; entries=[]
    for folder,index in pieces:
        assert index['first']==expected and index['trace_sha256']==manifest['actions_sha256']
        assert all(index[k]==settings[k] for k in ('fps','stride','presentation','checkpoints'))
        expected=index['last']+1
        for entry in index['frames']: entries.append((folder,entry))
    assert expected==manifest['required']+1 and pieces[-1][1]['checkpoint']==manifest['final_digest']
    final=directory/'walkthrough.mp4'; partial=directory/'walkthrough.partial.mp4'
    if final.exists() or partial.exists(): raise FileExistsError(final)
    writer=H264Writer(partial,size=(1280,720),fps=fps); chapters=[]; last_id=None; preview_saved=False
    try:
        for folder,entry in entries:
            p=folder/entry['file']; assert sha(p)==entry['sha256']
            if entry['id']!=last_id:
                if chapters: chapters[-1]['end_seconds']=writer.frames/fps
                chapters.append(dict(id=entry['id'],title=entry['title'],start_seconds=writer.frames/fps,
                                     action_start=entry['step'],simulation_start=entry['seconds']))
                last_id=entry['id']
            with Image.open(p) as im:
                rgb=np.asarray(im.convert('RGB'))
                for _ in range(entry['repeats']): writer.append_data(rgb)
                if entry['id'] in ('A03','B17') and not entry['new'] and not preview_saved:
                    im.save(directory/'preview.png'); preview_saved=True
        chapters[-1]['end_seconds']=writer.frames/fps
    finally: writer.close()
    cap=cv2.VideoCapture(str(partial)); decoded=0
    try:
        assert cap.isOpened()
        while True:
            ok,frame=cap.read()
            if not ok: break
            assert frame.shape==(720,1280,3); decoded+=1
    finally: cap.release()
    assert decoded==writer.frames and decoded>0
    partial.rename(final)
    save(directory/'chapters.json',chapters)
    save(directory/'summary.json',dict(**manifest,replay_verified=True,presentation=settings['presentation'],
         video_frames=decoded,video_fps=fps,video_seconds=decoded/fps,render_stride=settings['stride'],
         video_sha256=sha(final),frame_count=len(entries),checkpoint_walkthrough=settings['checkpoints'],
         render_sources={p:sha(ROOT/p) for p in SOURCES}))
    print(f'Published {final}: {decoded/fps:.1f} seconds; all {decoded} frames decoded',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    mode=p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--collect',type=Path);mode.add_argument('--verify',type=Path)
    mode.add_argument('--render-chunk',type=Path);mode.add_argument('--assemble',type=Path)
    p.add_argument('--task',nargs='+',choices=('a','b'),default=['a','b']);p.add_argument('--seed',type=int,default=7)
    p.add_argument('--milestones',nargs=2,type=int);p.add_argument('--cache',type=Path,default=Path('/tmp/owb-wildlands-reference-frames'))
    p.add_argument('--stride',type=int,default=12);p.add_argument('--fps',type=int,default=10)
    p.add_argument('--samples',type=int,default=48);p.add_argument('--threads',type=int,default=8)
    p.add_argument('--checkpoints',action='store_true',help='selected real states held for reading; not continuous motion')
    args=p.parse_args()
    if min(args.stride,args.fps,args.samples,args.threads)<=0: p.error('positive render settings required')
    if args.collect:
        for task in args.task: print(collect(task,args.seed,args.collect/f'task-{task}-seed-{args.seed}'))
    elif args.verify: print(verify(args.verify))
    elif args.render_chunk:
        if not args.milestones: p.error('--milestones FIRST LAST is required')
        render_chunk(args.render_chunk,args.cache,*args.milestones,stride=args.stride,fps=args.fps,samples=args.samples,threads=args.threads,checkpoints=args.checkpoints)
    else: assemble(args.assemble,args.cache)

if __name__=='__main__': main()
