"""Package verified render certificates with exact video/source timeline alignment."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from .skylands_reference import ROOT,save


def package(task,render_dir=None):
    directory=ROOT/f'assets/demos/skylands-campaign/task-{task}-seed-7'
    render_dir=render_dir or ROOT/f'.cache/skylands-campaign/final-frames-{task}'
    plan=json.loads((directory/'frames.json').read_text())
    movie=json.loads((directory/'video-manifest.json').read_text())
    playback=json.loads((directory/'playback-verification.json').read_text())
    assert playback['complete'] and playback['video_sha256']==movie['video_sha256']
    def record(entry):
        value=json.loads((render_dir/f'{entry["index"]:06}.json').read_text())
        assert value['index']==entry['index'] and value['step']==entry['step']
        assert value['digest']==entry['digest'] and value['trace_sha256']==plan['trace_sha256']
        return value
    with ThreadPoolExecutor(max_workers=8) as pool:records=list(pool.map(record,plan['entries']))
    rows=[];video_frame=0;source_step=0;source_seconds=0.
    for entry,value in zip(plan['entries'],records):
        assert entry['source_start_step']==source_step and entry['source_end_step']==entry['step']
        repeats=max(2,int(round(entry['duration']*10)))
        rows.append(dict(index=entry['index'],milestone=entry['milestone'],source_step_start=source_step,
            source_step_end=entry['step'],simulated_seconds_start=source_seconds,simulated_seconds_end=entry['seconds'],
            video_frame_start=video_frame,video_frame_end_exclusive=video_frame+repeats,
            video_seconds_start=video_frame/10,video_seconds_end=(video_frame+repeats)/10,
            simulation_digest=entry['digest'],scene_sha256=entry['scene_sha256'],rendered_png_sha256=value['image_sha256'],
            renderer=value['renderer'],device=value['device'],samples=value['samples'],render_size=value['size']))
        video_frame+=repeats;source_step=entry['step'];source_seconds=entry['seconds']
    assert video_frame==movie['frames']==playback['decoded_frames']
    assert source_step==playback['source_actions'] and abs(source_seconds-movie['source_seconds'])<1e-6
    save(directory/'frame-records.json',dict(trace_sha256=plan['trace_sha256'],video_sha256=movie['video_sha256'],
        source_timeline_complete=True,encoded_fps=10,rendered_states=len(rows),frames=rows))
    print('FRAME CERTIFICATES PACKAGED',task,len(rows),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--task',choices=('a','b'));p.add_argument('--render-dir',type=Path)
    a=p.parse_args()
    if a.render_dir and not a.task:p.error('--render-dir requires --task')
    for task in ([a.task] if a.task else 'ab'):package(task,a.render_dir)


if __name__=='__main__':main()
