"""Execute, independently replay, and GPU-Cycles-render Skylands reference courses."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import time
import numpy as np
from owb.env.skylands_reference import SkylandsReferenceEnv, COMMANDS, plain
from owb.bench.baselines.skylands_reference import SkylandsReferencePolicy
from owb.render.presentation import PresentationRenderer
from owb.render.video import H264Writer

ROOT=Path(__file__).resolve().parents[2]
SOURCES=('owb/env/skylands_reference.py','owb/bench/baselines/skylands_reference.py','owb/core/working_memory.py',
 'owb/scripts/skylands_reference.py','owb/env/skylands.py','owb/env/wildlands.py',
 'owb/world/skylands.py','owb/physics/character.py','owb/render/scene_export.py',
 'owb/render/blender_scene.py','owb/render/presentation.py','owb/render/video.py')
LIMITATIONS=[
 'Compact authored reference course, not the full island-scale or long-duration benchmark.',
 'Privileged scripted controller, not evidence of learned skills or memory performance.',
 'Cargo craft use four discrete mounts, rated flotation and a straight authored water channel; no general rigid-body vehicle or fluid solver.',
 'Deck movement is grid-stepped and collision-backed; freight lifts and reversible platforms are authored actuators.',
 'Heavy cargo uses a separate physical-position ledger with grounded roller-cradle towing, not the native inventory or a general contact solver.',
 'Task A uses one fixed return mooring rather than several agent-chosen docks; this video does not establish that spatial-memory probe.',
 'Standard cold-weather clothing is supplied at reset; provisioning is not evaluated.',
 'The two reserve items share a selected surface cache in this compact course, rather than separate regional cache networks.',
 'Training is given through inspectable local instructions and executed practice, not an animated teacher.',
 'A maintenance lift shortens inter-layer travel; cross-layer clue assignments and navigation alternatives are limited.',
 'Video samples verified simulator states and omits long walking intervals. Full action logs retain all intermediate steps.',
]


# Working-memory cues as they appear in the observation text after inspecting
# the source. The controller keeps what it read; it never reads probe content
# from the simulator.
WM_CUE_PATTERNS = {'WM-SA12': r'release sequence (\d)-(\d)-(\d)-(\d)',
                   'WM-SB10': r'inner gate code (\d)-(\d)-(\d)-(\d)',
                   'WM-SB17': r'jack order (\d)-(\d)-(\d)-(\d)'}


class WorkingMemoryControl:
    """Reference-controller extension for WM-SA12/SB10/SB17: read the source
    with `inspect`, keep the code it showed in the text observation, travel,
    and key it at the use site with `use`/`option`. Positions and routing are
    privileged as in the base controller; the code itself is taken only from
    the observation text."""

    def wm_init(self):
        self.wm_memory={}

    def observe_cues(self,obs):
        text=str(obs.get('text','')) if isinstance(obs,dict) else ''
        for probe,pattern in WM_CUE_PATTERNS.items():
            found=re.search(pattern,text)
            if found:
                items=[int(v) for v in found.groups()]
                # The schematic states its own rule: wire in the reverse order.
                self.wm_memory[probe]=items[::-1] if probe=='WM-SB17' and 'REVERSE' in text else items

    def wm_pending(self,probe):
        p=getattr(self.env,'wm',{}).get(probe)
        return p is not None and not p.solved

    def wm_read(self,probe):
        e=self.env; tid=e.wm_sites[probe]['source']; p=np.asarray(e.targets[tid]['pos'],float)
        if e._reachable(p,2.8):
            a=self.command('inspect',tid); self.explanation='Read the '+e.targets[tid]['role'].replace('_',' '); return a
        return self.go(p-(0,1.19,0))

    def wm_enter(self,probe):
        e=self.env; tid=e.wm_sites[probe]['use']
        key=self.wm_memory[probe][len(e.wm[probe].buffer)]
        a=self.approach(tid,'use',key)
        if e.commands[int(a['command'])]=='use':
            self.explanation=f'Key {key} on the '+e.targets[tid]['role'].replace('_',' ')+' from memory'
        return a

    def wm_step(self,probe):
        """Read the cue if not yet held, otherwise key it; None once solved."""
        if not self.wm_pending(probe): return None
        if probe not in self.wm_memory: return self.wm_read(probe)
        return self.wm_enter(probe)

    def wm_action(self,obs):
        """A WM action for the current milestone, or None to defer to the base controller."""
        self.observe_cues(obs)
        e=self.env; m=e.current
        if m is None or not getattr(e,'wm',None): return None
        s=m.state
        if m.id=='B10': return self.wm_step('WM-SB10')
        if m.id=='B17' and s['installed']: return self.wm_step('WM-SB17')   # output needs the patch
        if m.id=='A12' and self.wm_pending('WM-SA12'):
            mv=e.movers[s['mover']]
            if mv['moving']: return None
            if mv['cargo']!='core':
                a=self.transport_cargo('core',mv['pos']+(-3,0,0),s['load_target'])
                if a is not None: return a
                # Read the dispatch placard, then board and ride up with the load.
                if 'WM-SA12' not in self.wm_memory: return self.wm_read('WM-SA12')
                if not e.on_deck(mv): return self.go(mv['pos']+(.5,.01,.5))
                return None
            return self.wm_step('WM-SA12')   # held at the receiving station
        return None


class SkylandsWMReferencePolicy(WorkingMemoryControl,SkylandsReferencePolicy):
    def __init__(self,env):
        super().__init__(env); self.wm_init()

    def __call__(self,obs):
        a=self.wm_action(obs)
        return a if a is not None else super().__call__(obs)


def save(p,data): Path(p).write_text(json.dumps(plain(data),indent=2,allow_nan=False)+'\n')
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def sources(): return {p:sha(ROOT/p) for p in SOURCES}
def action_from_json(a):
    a=dict(a)
    for k in ('move','look','build_move'): a[k]=np.array(a[k],np.float32)
    a['buttons']=np.array(a['buttons'],np.int8); a['ingredients']=np.array(a['ingredients'],np.int64)
    return a


def collect(task,seed,directory):
    directory.mkdir(parents=True,exist_ok=False)
    with SkylandsReferenceEnv(task,seed) as env:
        obs,_=env.reset(seed=seed); initial=env.state_digest(); policy=SkylandsWMReferencePolicy(env); counts=Counter()
        with (directory/'actions.jsonl').open('w') as stream:
            while not env._ended:
                m=env.current; before=env.agent.pos.copy(); a=policy(obs)
                assert env.action_space.contains(a)
                obs,reward,terminated,truncated,info=env.step(a)
                counts.update(e['type'] for e in info['events'])
                row=dict(step=env.steps,milestone=m.id,title=m.title,kind=m.kind,action=plain(a),
                    explanation=policy.explanation,position_before=plain(before),position_after=plain(env.agent.pos),
                    seconds=env.elapsed,reward=reward,events=plain(info['events']))
                if any(e['type']=='milestone_complete' for e in info['events']):
                    row['digest']=env.state_digest(); print(task,env.steps,len(env.completed),m.id,flush=True)
                stream.write(json.dumps(row,allow_nan=False)+'\n')
        if not env.success or env.agent.deaths: raise RuntimeError('Reference did not finish successfully')
        save(directory/'reference-manifest.json',dict(task=task,seed=seed,initial_digest=initial,final_digest=env.state_digest(),
            decisions=env.steps,simulated_seconds=env.elapsed,milestones=env.completed,required=len(env.milestones),
            success=env.success,health=env.agent.health,events=dict(counts),limitations=LIMITATIONS,
            working_memory=getattr(env,'wm',None) and env.wm_summary(),wm_control=env.wm_control,
            controller='privileged action-only reference policy',sources=sources(),actions_sha256=sha(directory/'actions.jsonl')))
    return verify(directory)


def replay(directory):
    manifest=json.loads((directory/'reference-manifest.json').read_text())
    if sha(directory/'actions.jsonl')!=manifest['actions_sha256']: raise RuntimeError('Action log hash differs')
    rows=[json.loads(line) for line in (directory/'actions.jsonl').read_text().splitlines()]
    env=SkylandsReferenceEnv(manifest['task'],manifest['seed']); env.reset(seed=manifest['seed'])
    if env.state_digest()!=manifest['initial_digest']: raise RuntimeError('Initial state mismatch')
    return env,manifest,rows


def checked_step(env,row):
    if env.agent.pos.tolist()!=row['position_before']: raise RuntimeError(f'Before-position mismatch at {row["step"]}')
    _,reward,_,_,info=env.step(action_from_json(row['action']))
    if env.agent.pos.tolist()!=row['position_after'] or plain(info['events'])!=row['events'] or reward!=row['reward']:
        raise RuntimeError(f'Action replay mismatch at {row["step"]}')
    if 'digest' in row and env.state_digest()!=row['digest']: raise RuntimeError(f'Milestone state mismatch at {row["step"]}')


def verify(directory):
    env,manifest,rows=replay(directory)
    try:
        for row in rows: checked_step(env,row)
        assert env.success and env.system_operational() and len(env.completed)==manifest['required'] and not env.agent.deaths
        assert env.state_digest()==manifest['final_digest']
        wm=getattr(env,'wm',None) and plain(env.wm_summary())
        if wm!=manifest.get('working_memory'): raise RuntimeError('Working-memory summary differs from the manifest')
        result=dict(success=True,checked_actions=len(rows),checked_positions=True,checked_events=True,checked_rewards=True,
                    checked_milestones=len(env.completed),final_live_predicate=True,final_digest=env.state_digest(),
                    working_memory=wm,checked_working_memory=bool(wm),
                    actions_sha256=manifest['actions_sha256'],sources=sources())
        save(directory/'verification.json',result); return result
    finally: env.close()


def selected_steps(rows):
    selected=set(); seen=set()
    key_types={'part_picked','part_mounted','cargo_loaded','cargo_delivered','gallery_serviced','ascended',
      'core_extracted','raft_staged','dock_lit','freight_loaded','hoist_started','lift_arrived','core_installed',
      'reserves_cached','cache_retrieved','outpost_opened','station_engaged','service_part_recovered',
      'reversal_started','reversal_cancelled','platform_landing_reached','maintenance_bridge_closed',
      'lock_coupling_installed','cargo_berth_opened','milestone_complete','calibration_trip'}
    for row in rows:
        types={e['type'] for e in row['events']}
        if row['milestone'] not in seen:
            selected.add(row['step']); seen.add(row['milestone'])
        if types&key_types: selected.add(row['step'])
        if types&{'craft_moved','platform_reversed'} and row['step']%3==0: selected.add(row['step'])
        if row['explanation'].startswith('Freight lift') and row['step']%8==0: selected.add(row['step'])
    return selected


def render(directory,cache,*,samples=48,device='CUDA',first=1,last=None):
    from PIL import Image,ImageDraw,ImageFont
    import matplotlib
    env,manifest,rows=replay(directory); selected=selected_steps(rows)
    last=last or manifest['required']; folder=cache/directory.name; folder.mkdir(parents=True,exist_ok=True)
    renderer=PresentationRenderer(backend='cycles',width=960,height=540,samples=samples,threads=4,device=device)
    font_path=Path(matplotlib.get_data_path())/'fonts/ttf/DejaVuSans.ttf'
    fonts={n:ImageFont.truetype(str(font_path),n) for n in (15,18,22,26)}
    contract=dict(trace_sha256=manifest['actions_sha256'],presentation=renderer.metadata,sources=sources())
    plan=folder/'render-plan.json'
    if plan.exists() and json.loads(plan.read_text())!=contract: raise RuntimeError('Render source/settings changed')
    if not plan.exists(): save(plan,contract)
    entries=[]; started=time.monotonic()
    try:
        for row in rows:
            checked_step(env,row)
            number=int(row['milestone'][1:])
            if number>last: break
            if number<first or row['step'] not in selected: continue
            frame_path=folder/f'{row["step"]:06}.png'; meta=frame_path.with_suffix('.json')
            digest=env.state_digest()
            if frame_path.exists() and meta.exists():
                record=json.loads(meta.read_text())
                if record['digest']!=digest or record['sha256']!=sha(frame_path): raise RuntimeError('Cached frame mismatch')
                entries.append(record); continue
            env.reference_view_id=row['milestone']
            rgb=renderer.render(env); assert env.state_digest()==digest,'Renderer changed simulation'
            image=Image.new('RGB',(1280,720),(15,32,40)); draw=ImageDraw.Draw(image)
            draw.text((20,14),row['milestone']+'  '+row['title'],font=fonts[26],fill=(242,234,208))
            draw.text((20,55),'SKYLANDS | Executed compact reference course | Privileged controller | Blender Cycles / '+device,font=fonts[15],fill=(154,204,191))
            image.paste(Image.fromarray(rgb),(16,91))
            m=env.by_id[row['milestone']]
            lines=[f'{len(env.completed)} / {len(env.milestones)} milestones',f'Action {env.steps}',f'Simulation {env.elapsed:.1f} s',
                   f'Health {env.agent.health:.1f}',f'Layer: '+('depths','surface','sky')[m.layer],'','LIVE STATE']
            rid=m.state.get('raft') or ('B23' if m.kind=='delivery' else None)
            if rid:
                r=env.rafts[rid]; lines += [f'Mounts: {sum(p is not None for p in r["parts"])}/4',f'Charge: {r["charge"]:.0f}',
                    f'Load: {r["cargo"] or "none"}',f'Travel: {r["travel"]:.0f} m',f'Moored: {r["moored"]}']
            elif m.state.get('mover'):
                mv=env.movers[m.state['mover']]; lines += [f'Moving: {mv["moving"] or mv["reverse"]}',f'History: {len(mv["history"])}',f'Altitude: {mv["pos"][1]:.0f} m']
            else: lines += [f'Installed: {m.state["installed"]}',f'Stable: {m.state["stable"]:.1f} s',f'Stations: {len(env.online)}/3',f'Route updated: {env.closed}']
            for i,line in enumerate(lines): draw.text((992,106+i*28),line[:27],font=fonts[18],fill=(213,228,219))
            complete=any(e['type']=='milestone_complete' for e in row['events'])
            caption='VERIFIED: '+row['milestone']+' complete' if complete else row['explanation']
            draw.rectangle((16,644,1264,687),fill=(28,61,67)); draw.text((28,654),caption[:100],font=fonts[22],fill=(242,234,208))
            draw.text((20,698),'Sampled simulator states; travel omitted and key states held for reading. Complete actions supplied for replay.',font=fonts[15],fill=(153,186,180))
            image.save(frame_path)
            record=dict(step=row['step'],milestone=row['milestone'],title=row['title'],seconds=row['seconds'],digest=digest,
                        file=frame_path.name,sha256=sha(frame_path),repeats=15 if complete else 6)
            save(meta,record); entries.append(record)
            print(row['milestone'],row['step'],'Cycles frame',round(time.monotonic()-started),flush=True)
        save(folder/f'index-{first:02}-{last:02}.json',dict(**contract,first=first,last=last,frames=entries))
    finally: renderer.close(); env.close()


def assemble(directory,cache):
    from PIL import Image
    folder=cache/directory.name; manifest=json.loads((directory/'reference-manifest.json').read_text())
    entries=sorted((json.loads(p.read_text()) for p in folder.glob('[0-9]*.json')),key=lambda r:r['step'])
    assert {r['milestone'] for r in entries}=={m['id'] for m in manifest['milestones']}
    out=directory/'walkthrough.mp4'; assert not out.exists()
    partial=directory/'walkthrough.partial.mp4'; assert not partial.exists()
    writer=H264Writer(partial,size=(1280,720),fps=5); chapters=[]; last=None
    try:
        for entry in entries:
            path=folder/entry['file']; assert sha(path)==entry['sha256']
            if entry['milestone']!=last:
                if chapters: chapters[-1]['end_seconds']=writer.frames/5
                chapters.append(dict(id=entry['milestone'],title=entry['title'],start_seconds=writer.frames/5,action=entry['step']))
                last=entry['milestone']
            rgb=np.asarray(Image.open(path).convert('RGB'))
            for _ in range(entry['repeats']): writer.append_data(rgb)
        chapters[-1]['end_seconds']=writer.frames/5
    finally: writer.close()
    partial.replace(out)
    Image.open(folder/entries[len(entries)//2]['file']).save(directory/'preview.png')
    plan=json.loads((folder/'render-plan.json').read_text())
    save(directory/'chapters.json',chapters)
    save(directory/'summary.json',dict(task=manifest['task'],success=True,milestones=manifest['required'],decisions=manifest['decisions'],
        simulated_seconds=manifest['simulated_seconds'],video_seconds=writer.frames/5,video_frames=writer.frames,
        rendered_states=len(entries),video_sha256=sha(out),presentation=plan['presentation'],limitations=LIMITATIONS))
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('phase',choices=('collect','verify','render','assemble'))
    p.add_argument('--task',choices=('a','b'),required=True); p.add_argument('--seed',type=int,default=7)
    p.add_argument('--root',type=Path,default=ROOT/'assets/demos/skylands-reference')
    p.add_argument('--cache',type=Path,default=ROOT/'.cache/skylands-cycles'); p.add_argument('--device',default='CUDA',choices=('CUDA','OPTIX'))
    p.add_argument('--first',type=int,default=1); p.add_argument('--last',type=int)
    a=p.parse_args(); d=a.root/f'task-{a.task}-seed-{a.seed}'
    if a.phase=='collect': print(collect(a.task,a.seed,d))
    elif a.phase=='verify': print(verify(d))
    elif a.phase=='render': render(d,a.cache,device=a.device,first=a.first,last=a.last)
    else: print(assemble(d,a.cache))

if __name__=='__main__': main()
