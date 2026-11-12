"""First-person Cycles recording of a verified full Skylands campaign trace.

The action log, full-world static scene, Blender renderer and render settings
are the campaign walkthrough's own. Only the presentation camera and sampling
change: the camera sits at the traveler's eye height and looks along the
traveler's own yaw and pitch with the policy's field of view. The traveler's
body is omitted so it does not block its own view. States are sampled densely
(every ``--stride`` decisions plus every key event), so the view turns smoothly.

Intermediate states and frames (thousands of files) go to ``--cache`` (default
``$OWB_RENDER_SCRATCH`` or ``.cache``); only the video and its records are
published under ``assets/demos/skylands-campaign-first-person``.

    python -m owb.scripts.skylands_campaign_first_person export --task a
    python -m owb.scripts.skylands_campaign_first_person range --task a --chunk 0 --chunks 8
    (Blender: owb/render/blender_campaign_baked.py over each range)
    python -m owb.scripts.skylands_campaign_first_person assemble --task a --chunks 8
"""
import argparse
import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from .skylands_campaign import replay
from .skylands_reference import ROOT, checked_step, save, sha
from .skylands_campaign_video import KEYS, assemble
from owb.render.skylands_campaign import dynamic_scene

STATIC_SOURCE = ROOT / '.cache/skylands-campaign'
STATIC = ('static.npz', 'ground-cover.npz', 'grass-prototypes.npz', 'geometry.json')
FPS = 10
DECISION_SECONDS = .2
VIEW_LABEL = "First-person view (traveler's eyes)"
MILESTONES = {'a': 15, 'b': 24}


def work_dir(cache, task): return cache / f'first-person-{task}'
def chunk_dir(cache, task, chunk): return cache / f'first-person-frames-{task}-{chunk}'
def publish_dir(task): return ROOT / f'assets/demos/skylands-campaign-first-person/task-{task}-seed-7'
def trace_dir(task): return ROOT / f'assets/demos/skylands-campaign/task-{task}-seed-7'


def sampling(rows, stride):
    """Every ``stride`` decisions, plus every key event (paused for reading)."""
    selected = {}; last = 0
    for i, row in enumerate(rows):
        types = {e['type'] for e in row['events']}
        key = bool(types & KEYS)
        if key or row['step'] - last >= stride or i == len(rows) - 1:
            span = row['step'] - last
            speed = round(span * DECISION_SECONDS * FPS)
            duration = 1 / FPS
            clues = [e['text'] for e in row['events'] if e['type'] == 'clue_exposed']
            if key: duration = 1.2
            if clues: duration = max(duration, 3., len(' '.join(clues)) / 22. + .5)
            selected[row['step']] = dict(speed=speed, duration=duration, annotation_pause=key,
                playback_label='Event pause' if key else f'Playback {speed}x',
                source_start_step=last, source_end_step=row['step'])
            last = row['step']
    return selected


def export(cache, task, stride, stop_step=None):
    """Replay every action (checked) and export first-person dynamic states."""
    source = STATIC_SOURCE / f'video-{task}'
    original = json.loads((source / 'frames.json').read_text())
    output = work_dir(cache, task) if stop_step is None else cache / f'first-person-pilot-{task}'
    output.mkdir(parents=True, exist_ok=False); (output / 'states').mkdir()
    # The verified full-world static scene is camera independent; reuse it exactly.
    for name in STATIC: shutil.copyfile(source / name, output / name)
    for name, expected in original['artifact_hashes'].items():
        if sha(output / name) != expected: raise RuntimeError(f'Static scene artifact changed: {name}')
    env, manifest, rows = replay(trace_dir(task))
    if manifest['actions_sha256'] != original['trace_sha256']: raise RuntimeError('Trace differs from static export')
    selected = sampling(rows, stride); entries = []
    try:
        for row in rows:
            if stop_step is not None and row['step'] > stop_step: break
            checked_step(env, row)
            if row['step'] not in selected: continue
            env.reference_view_id = row['milestone']; before = env.state_digest()
            state = dynamic_scene(env, first_person=True)
            if before != env.state_digest(): raise RuntimeError('Scene export changed the simulation')
            name = f'{len(entries):06}.npz'; state.save(output / 'states' / name)
            entries.append(dict(index=len(entries), scene='states/' + name, scene_sha256=sha(output / 'states' / name),
                step=row['step'], seconds=row['seconds'], milestone=row['milestone'], title=row['title'],
                explanation=row['explanation'], events=row['events'], position=row['position_after'],
                camera=state.metadata['camera'], digest=before, **selected[row['step']]))
            if len(entries) % 200 == 0: print('EXPORTED', len(entries), 'step', row['step'], flush=True)
        if stop_step is None and not (env.success and env.state_digest() == manifest['final_digest']):
            raise RuntimeError('Replay did not reach the verified final state')
    finally:
        env.close()
    playback = (f'First-person view sampled every {stride} decisions ({stride * DECISION_SECONDS:g} s of simulation per '
                f'video frame, {round(stride * DECISION_SECONDS * FPS)}x) plus every key event; 1.2 s event pauses and '
                'time to read full clues. No interpolated frames.')
    save(output / 'frames.json', dict(task=task, seed=manifest['seed'], trace_sha256=manifest['actions_sha256'],
        world_shape=manifest['world_shape'], source_seconds=manifest['simulated_seconds'],
        renderer=original['renderer'], artifact_hashes=original['artifact_hashes'],
        view=dict(kind='first_person', eye='traveler position + sim eye height', direction="traveler's own yaw and pitch",
                  fov='policy horizontal FOV converted to vertical at 16:9', traveler_body='omitted', stride_decisions=stride),
        playback=playback, pilot_stop_step=stop_step,
        sources={p: sha(ROOT / p) for p in ('owb/render/skylands_campaign.py', 'owb/scripts/skylands_campaign_first_person.py',
                                            'owb/scripts/skylands_campaign_video.py', 'owb/scripts/skylands_campaign.py')},
        entries=entries))
    print('EXPORT COMPLETE', len(entries), output, flush=True)


def chunk_range(count, chunk, chunks):
    return count * chunk // chunks, count * (chunk + 1) // chunks


def merge_and_assemble(cache, task, chunks):
    source = work_dir(cache, task); plan = json.loads((source / 'frames.json').read_text())
    if plan['pilot_stop_step'] is not None: raise RuntimeError('Refusing to publish a pilot export')
    count = len(plan['entries'])
    reports = [json.loads((chunk_dir(cache, task, c) / 'renderer-report.json').read_text()) for c in range(chunks)]
    for report in reports:
        assert report['device'] == 'OPTIX GPU' and report['accelerators']
        assert report['samples'] == 64 and (report['width'], report['height']) == (1280, 720)
        assert report['denoising'] and report['max_bounces'] == 6
        assert report['ground_cover']['full_geometry'] and not report['ground_cover']['distance_culling']
        assert report['render_sources'] == reports[0]['render_sources'] and report['ground_cover'] == reports[0]['ground_cover']
    destination = cache / f'first-person-final-{task}'; destination.mkdir(exist_ok=True)

    def verify_frame(entry):
        chunk = next(c for c in range(chunks) if chunk_range(count, c, chunks)[0] <= entry['index'] < chunk_range(count, c, chunks)[1])
        path = chunk_dir(cache, task, chunk) / f'{entry["index"]:06}.png'; record = json.loads(path.with_suffix('.json').read_text())
        assert record['digest'] == entry['digest'] and record['trace_sha256'] == plan['trace_sha256']
        assert record['image_sha256'] == sha(path)
        assert record['device'] == 'OPTIX GPU' and record['samples'] == 64 and record['size'] == [1280, 720]
        for file in (path, path.with_suffix('.json')):
            target = destination / file.name
            if target.exists():
                if not os.path.samefile(target, file): assert sha(target) == sha(file)
            else: os.link(file, target)
    with ThreadPoolExecutor(max_workers=8) as pool: list(pool.map(verify_frame, plan['entries']))
    report = dict(reports[0]); report['accelerators'] = sorted({gpu for r in reports for gpu in r['accelerators']})
    report['chunks'] = [dict(directory=chunk_dir(cache, task, c).name, renderer=r) for c, r in enumerate(reports)]
    save(destination / 'renderer-report.json', report)
    output = publish_dir(task); output.mkdir(parents=True, exist_ok=True)
    trace = trace_dir(task)
    # Hard links: the published trace and its verification are the originals.
    for name in ('actions.jsonl', 'reference-manifest.json', 'verification.json'):
        if not (output / name).exists(): os.link(trace / name, output / name)
    for name in ('frames.json', 'geometry.json'): (output / name).write_bytes((source / name).read_bytes())
    save(output / 'rendered-frame-verification.json', dict(complete=True, checked_states=count,
         trace_sha256=plan['trace_sha256'], renderer_report=report))
    if not (output / 'walkthrough.mp4').exists():
        # Preview: a few states into the second milestone, seen first-hand.
        preview = next(e['index'] for e in plan['entries'] if e['milestone'] != plan['entries'][0]['milestone']) + 5
        assemble(source, destination, output, view_label=VIEW_LABEL, min_repeats=1, preview_index=preview)
    from .skylands_campaign_index import verify_video
    manifest = json.loads((output / 'reference-manifest.json').read_text())
    movie = json.loads((output / 'video-manifest.json').read_text())
    chapters = json.loads((output / 'chapters.json').read_text())
    assert len(chapters) == MILESTONES[task] == manifest['required']
    verify_video(output, manifest, movie)
    print('FIRST-PERSON VIDEO VERIFIED', task, output / 'walkthrough.mp4', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('operation', choices=('export', 'range', 'assemble'))
    p.add_argument('--task', choices=('a', 'b'), required=True)
    p.add_argument('--cache', type=Path, default=Path(os.environ.get('OWB_RENDER_SCRATCH', ROOT / '.cache')) / 'skylands-campaign')
    p.add_argument('--stride', type=int, default=4, help='decisions per rendered state between key events')
    p.add_argument('--stop-step', type=int, help='pilot export: stop after this decision (never published)')
    p.add_argument('--chunk', type=int); p.add_argument('--chunks', type=int, default=8)
    a = p.parse_args()
    if a.operation == 'export': export(a.cache, a.task, a.stride, a.stop_step)
    elif a.operation == 'range':
        count = len(json.loads((work_dir(a.cache, a.task) / 'frames.json').read_text())['entries'])
        print(*chunk_range(count, a.chunk, a.chunks))
    else: merge_and_assemble(a.cache, a.task, a.chunks)


if __name__ == '__main__': main()
