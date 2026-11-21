"""Verify disjoint Cycles chunks and assemble both full campaign recordings."""
import argparse
import json
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from .skylands_reference import ROOT,sha,save
from .skylands_campaign_video import assemble


#: Render array task ids per campaign, in frame order (A: two chunks, B: four).
CHUNK_IDS={'a':(0,1),'b':(2,5,3,4)}


def chunk_ranges(task,count):
    """``(array id, first, last)`` frame intervals that tile ``count`` frames."""
    ids=CHUNK_IDS[task];bounds=[round(i*count/len(ids)) for i in range(len(ids)+1)]
    return [(chunk,bounds[i],bounds[i+1]) for i,chunk in enumerate(ids)]


def chunk_for(array_id):
    """Task and frame interval for one render array task (used by the Slurm script)."""
    task=next(t for t,ids in CHUNK_IDS.items() if array_id in ids)
    count=len(json.loads((ROOT/f'.cache/skylands-campaign/video-{task}/frames.json').read_text())['entries'])
    _,first,last=next(r for r in chunk_ranges(task,count) if r[0]==array_id)
    return task,first,last


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--task',choices=('a','b'));args=parser.parse_args()
    cache=ROOT/'.cache/skylands-campaign'
    for task in ('a','b'):
        if args.task and task!=args.task:continue
        count=len(json.loads((cache/f'video-{task}'/'frames.json').read_text())['entries'])
        ranges=chunk_ranges(task,count)
        chunks=[item[0] for item in ranges]
        source=cache/f'video-camera-{task}';plan=json.loads((source/'frames.json').read_text())
        original=json.loads((cache/f'video-{task}'/'frames.json').read_text())
        plan['original_export_sources']=original['sources']
        for entry in plan['entries']:
            clues=[event['text'] for event in entry['events'] if event['type']=='clue_exposed']
            if clues:
                entry['duration']=max(entry['duration'],3.,len(' '.join(clues))/22.+.5)
                entry['annotation_pause']=True
        plan['playback']='Complete journey sampled with labeled 16x walking / 8x craft travel; 1x mechanisms; annotation pauses, including time to read full clues.'
        plan['export_source_hash_timing']='original_export_sources records scene export; sources also includes the camera visibility correction'
        save(source/'frames.json',plan)
        corrected={entry['index'] for entry in plan['entries'] if 'camera_correction' in entry}
        directories=[cache/f'final-frames-{task}-{chunk}' for chunk in chunks]
        camera_dir=cache/f'camera-frames-{task}'
        if corrected:directories.append(camera_dir)
        reports=[json.loads((path/'renderer-report.json').read_text()) for path in directories]
        for report in reports:
            assert report['device']=='OPTIX GPU' and report['accelerators']
            assert report['samples']==64 and (report['width'],report['height'])==(1280,720)
            assert report['denoising'] and report['max_bounces']==6
            assert report['ground_cover']['full_geometry'] and not report['ground_cover']['distance_culling']
            assert report['render_sources']==reports[0]['render_sources']
            assert report['ground_cover']==reports[0]['ground_cover']
        destination=cache/f'final-frames-{task}';destination.mkdir(exist_ok=True)
        def verify_frame(entry):
            name=f'{entry["index"]:06}.png'
            candidates=[camera_dir] if entry['index'] in corrected else [cache/f'final-frames-{task}-{chunk}' for chunk,first,last in ranges if first<=entry['index']<last]
            found=[path/name for path in candidates if (path/name).exists()]
            assert len(found)==1,(task,name,'missing or overlapping chunks')
            path=found[0];record=json.loads(path.with_suffix('.json').read_text())
            assert record['digest']==entry['digest'] and record['trace_sha256']==plan['trace_sha256']
            assert record['image_sha256']==sha(path)
            assert record['device']=='OPTIX GPU' and record['samples']==64 and record['size']==[1280,720]
            for file in (path,path.with_suffix('.json')):
                target=destination/file.name
                if target.exists():
                    if not os.path.samefile(target,file):assert sha(target)==sha(file)
                else:os.link(file,target)
        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(verify_frame,plan['entries']))
        report=dict(reports[0]);report['accelerators']=sorted({gpu for r in reports for gpu in r['accelerators']})
        report['chunks']=[dict(directory=str(p.relative_to(ROOT)),renderer=r) for p,r in zip(directories,reports)]
        save(destination/'renderer-report.json',report)
        output=ROOT/f'assets/demos/skylands-campaign/task-{task}-seed-7'
        # Persist the complete selected-frame plan and geometry/provenance records.
        for name in ('frames.json','geometry.json'):
            (output/name).write_bytes((source/name).read_bytes())
        save(output/'rendered-frame-verification.json',dict(complete=True,checked_states=len(plan['entries']),
             trace_sha256=plan['trace_sha256'],renderer_report=report))
        if not (output/'walkthrough.mp4').exists():assemble(source,destination,output)
        from .skylands_campaign_index import verify_video
        verify_video(output,json.loads((output/'reference-manifest.json').read_text()),json.loads((output/'video-manifest.json').read_text()))
        print('CAMPAIGN VIDEO VERIFIED',task,flush=True)
    from .skylands_campaign_index import main as index
    # Regeneration keeps the previous manifests beside the archived videos, so
    # build the index only once both new walkthroughs exist.
    if all((ROOT/f'assets/demos/skylands-campaign/task-{task}-seed-7/{name}').exists()
           for task in 'ab' for name in ('video-manifest.json','walkthrough.mp4')):index()


if __name__=='__main__':main()
