"""Export full validated journeys for persistent GPU Cycles recording."""
import argparse
import json
from pathlib import Path
import numpy as np
from owb.scripts.skylands_campaign import replay,sources
from owb.scripts.skylands_reference import checked_step,save,sha,plain
from owb.render.skylands_campaign import static_scene,dynamic_scene,grass_prototypes

KEYS={'part_picked','part_mounted','cargo_loaded','cargo_delivered','gallery_serviced','ascended',
 'core_extracted','raft_staged','dock_lit','freight_loaded','hoist_started','lift_arrived','core_installed',
 'reserves_cached','cache_retrieved','outpost_opened','station_engaged','service_part_recovered',
 'reversal_started','reversal_cancelled','platform_landing_reached','maintenance_bridge_closed',
 'lock_coupling_installed','cargo_berth_opened','milestone_complete','clue_exposed'}

def sampling(rows):
    selected={};last=0
    for i,row in enumerate(rows):
        types={e['type'] for e in row['events']};walking='Walk' in row['explanation'] or 'Approach' in row['explanation']
        craft='craft_moved' in types
        speed=16 if walking else 8 if craft else 1
        stride=40 if walking else 20 if craft else 3
        if types&KEYS or row['step']-last>=stride or i==len(rows)-1:
            span=row['step']-last
            # Source time always advances by the actual trace interval.
            duration=max(.2,span*.2/speed)
            pause=bool(types&KEYS)
            if pause: duration=max(1.2,duration)
            selected[row['step']]=dict(speed=speed,duration=duration,annotation_pause=pause,
                                      source_start_step=last,source_end_step=row['step'])
            last=row['step']
    return selected


def export(directory,output):
    output.mkdir(parents=True,exist_ok=False)
    env,manifest,rows=replay(directory);selected=sampling(rows)
    print('EXPORT STATIC',manifest['task'],flush=True)
    before=env.state_digest();scene,points,variants=static_scene(env)
    assert before==env.state_digest()
    scene.save(output/'static.npz');np.savez_compressed(output/'ground-cover.npz',points=points,variants=variants)
    grass_prototypes().save(output/'grass-prototypes.npz')
    save(output/'geometry.json',scene.metadata['geometry_contract'])
    entries=[];trail=[]
    (output/'states').mkdir()
    for row in rows:
        checked_step(env,row)
        if row['step'] not in selected: continue
        env.reference_view_id=row['milestone'];before=env.state_digest()
        state=dynamic_scene(env);assert before==env.state_digest()
        name=f'{len(entries):06}.npz';state.save(output/'states'/name)
        entry=dict(index=len(entries),scene='states/'+name,scene_sha256=sha(output/'states'/name),step=row['step'],seconds=row['seconds'],
            milestone=row['milestone'],title=row['title'],explanation=row['explanation'],events=row['events'],
            position=row['position_after'],digest=before,**selected[row['step']])
        entries.append(entry);trail.append(row['position_after'])
        if len(entries)%50==0: print('EXPORTED',len(entries),'step',row['step'],flush=True)
    assert env.success and env.state_digest()==manifest['final_digest']
    save(output/'frames.json',dict(task=manifest['task'],seed=manifest['seed'],trace_sha256=manifest['actions_sha256'],
        world_shape=manifest['world_shape'],source_seconds=manifest['simulated_seconds'],
        renderer=dict(engine='Blender Cycles',device='CUDA GPU',width=1280,height=720,samples=64,
                      denoising=True,max_bounces=6,upscaling=False,geometry='exact full-world surfaces',
                      foliage='all trees at detail 2; complete high-tier ground-cover instances'),
        artifact_hashes={name:sha(output/name) for name in ('static.npz','ground-cover.npz','grass-prototypes.npz')},
        playback='Complete journey sampled with labeled 16x walking / 8x craft travel; 1x mechanisms; brief annotation pauses.',
        sources={**sources(),**{p:sha(Path(p)) for p in ('owb/render/skylands_campaign.py','owb/render/blender_campaign.py','owb/scripts/skylands_campaign_video.py')}},entries=entries))
    print('EXPORT COMPLETE',len(entries),flush=True);env.close()


def verified_images(entries,render_dir,trace_sha256):
    """Prefetch a bounded number of independently verified full-resolution frames."""
    from collections import deque
    from concurrent.futures import ThreadPoolExecutor
    from PIL import Image
    import hashlib
    import io
    def read(entry):
        path=render_dir/f'{entry["index"]:06}.png'
        record=json.loads(path.with_suffix('.json').read_text());raw=path.read_bytes()
        if record['digest']!=entry['digest'] or record['trace_sha256']!=trace_sha256 or record['image_sha256']!=hashlib.sha256(raw).hexdigest():
            raise RuntimeError('Rendered frame verification failed')
        with Image.open(io.BytesIO(raw)) as source:rgb=source.convert('RGB')
        return entry,rgb
    remaining=iter(entries)
    with ThreadPoolExecutor(max_workers=4) as pool:
        pending=deque(pool.submit(read,next(remaining)) for _ in range(min(8,len(entries))))
        while pending:
            result=pending.popleft().result()
            entry=next(remaining,None)
            if entry is not None:pending.append(pool.submit(read,entry))
            yield result


def assemble(directory,render_dir,output):
    from PIL import Image,ImageDraw,ImageFont
    import matplotlib
    from owb.render.video import H264Writer
    from owb.render import video as encoder_module
    assembly_sources={str(Path(__file__).relative_to(Path(__file__).resolve().parents[2])):sha(Path(__file__)),str(Path(encoder_module.__file__).relative_to(Path(__file__).resolve().parents[2])):sha(Path(encoder_module.__file__))}
    data=json.loads((directory/'frames.json').read_text());entries=data['entries']
    fontpath=Path(matplotlib.get_data_path())/'fonts/ttf/DejaVuSans.ttf'
    fonts={n:ImageFont.truetype(str(fontpath),n) for n in (17,19,22,27,30)}
    output.mkdir(parents=True,exist_ok=True)
    partial=output/'walkthrough.partial.mp4';final=output/'walkthrough.mp4'
    if final.exists(): raise FileExistsError(final)
    chapters=[];last=None;video_seconds=0;trail=[];frame_count=0
    writer=H264Writer(partial,size=(1600,900),fps=10)
    try:
        for entry,rgb in verified_images(entries,render_dir,data['trace_sha256']):
            if rgb.size!=(1280,720): raise RuntimeError('Rendered resolution differs from contract')
            canvas=Image.new('RGB',(1600,900),(12,28,35));draw=ImageDraw.Draw(canvas)
            canvas.paste(rgb,(0,90))
            draw.text((22,16),entry['milestone']+'  '+entry['title'],font=fonts[30],fill=(244,235,209))
            draw.text((22,57),'SKYLANDS  |  Full-scale executed campaign  |  Privileged reference policy  |  Cycles GPU',font=fonts[19],fill=(159,211,195))
            x=1300;draw.text((x,105),'CAMPAIGN PROGRESS',font=fonts[19],fill=(159,211,195))
            draw.text((x,143),f"{int(entry['milestone'][1:])} / {15 if data['task']=='a' else 24} milestones",font=fonts[22],fill='white')
            t=entry['seconds'];draw.text((x,190),f"Sim time  {int(t//60):02}:{int(t%60):02}",font=fonts[22],fill='white')
            draw.text((x,224),f"Decision  {entry['step']:,}",font=fonts[19],fill=(207,220,221))
            label=f"Travel {entry['speed']}x" if entry['speed']>1 else 'Mechanism 1x'
            if entry['annotation_pause']: label+=' + pause'
            draw.text((x,265),label,font=fonts[19],fill=(246,200,114))
            p=entry['position'];layer='SKY' if p[1]>100 else 'DEPTHS' if p[1]<40 else 'SURFACE'
            draw.text((x,315),layer,font=fonts[22],fill=(151,223,206))
            # A stable whole-world route map makes the complete journey legible.
            mapbox=(x,360,x+270,650);draw.rectangle(mapbox,fill=(24,44,49),outline=(68,99,99))
            def point(q): return (x+10+q[0]/data['world_shape'][0]*250,370+q[2]/data['world_shape'][2]*270)
            trail.append(p)
            if len(trail)>1:draw.line([point(q) for q in trail],fill=(87,154,155),width=2)
            px,pz=point(p);draw.ellipse((px-4,pz-4,px+4,pz+4),fill=(255,217,123))
            draw.text((x,668),'Full action log + replay',font=fonts[17],fill=(163,190,190))
            draw.text((x,694),'verification accompany video.',font=fonts[17],fill=(163,190,190))
            text=entry['explanation'];draw.text((22,828),text[:108],font=fonts[22],fill=(239,231,211))
            events=[e.get('text',e['type'].replace('_',' ')) for e in entry['events'] if e['type'] in KEYS]
            draw.text((22,863),' | '.join(events)[:145],font=fonts[17],fill=(159,211,195))
            if entry['milestone']!=last:
                chapters.append(dict(id=entry['milestone'],title=entry['title'],video_seconds=round(video_seconds,3),
                                     source_seconds=entry['seconds'],step=entry['step']))
                last=entry['milestone']
            repeats=max(2,int(round(entry['duration']*10)))
            arr=np.asarray(canvas)
            for _ in range(repeats):writer.append_data(arr)
            frame_count+=repeats;video_seconds+=repeats/10
            if entry['index']==(24 if data['task']=='a' else 112):canvas.save(output/'preview.png')
    finally:writer.close()
    partial.rename(final)
    save(output/'chapters.json',chapters)
    save(output/'video-manifest.json',dict(renderer={**data['renderer'],**json.loads((render_dir/'renderer-report.json').read_text())},playback=data['playback'],
        duration_seconds=video_seconds,frames=frame_count,rendered_states=len(entries),
        source_seconds=data['source_seconds'],trace_sha256=data['trace_sha256'],video_sha256=sha(final),
        encoded_size=[1600,900],source_render_size=[1280,720],encoded_fps=10,upscaling=False,assembly_sources=assembly_sources))
    print('VIDEO COMPLETE',final,video_seconds,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('operation',choices=['export','assemble'])
    p.add_argument('--directory',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--render-dir',type=Path)
    a=p.parse_args();export(a.directory,a.output) if a.operation=='export' else assemble(a.directory,a.render_dir,a.output)
if __name__=='__main__':main()
