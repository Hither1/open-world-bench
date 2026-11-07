"""Decode-check full campaign videos and create the local milestone player."""
import json
from pathlib import Path
import cv2
from .skylands_reference import ROOT,save,sha

PROCEDURES={
 'clue':'Inspect the commission or survey board. Retain its destination, landmark correspondence, or calibration for later use.',
 'raft':'Gather compatible hulls, fan and cell. Mount the parts, secure the physical payload, board fully, steer through the waterway, and unload at the receiving apron.',
 'stage':'Choose a return dock. Assemble and load its raft, perform an outward-and-return trial, then moor the charged craft there.',
 'return':'Recover the same staged raft, exchange its ballast for the original navigation core, and follow the surface waterway to the hoist berth.',
 'ascend':'Reach the supported launch point beneath the service ceiling, use Ascend, and operate the upper-gallery control.',
 'dock':'Use the service authorization and issued lamp at the depths staging dock.',
 'extract':'Open the vault through its service gallery, release the unique core, and tow it on its transport cradle.',
 'freight':'Bring the original core to the loading deck, board the freight lift, secure the core, and ride to the surface receiving bay.',
 'hoist':'Load the core onto the sky hoist. Ascend into the control gallery and engage the lift; the core travels separately from the traveler.',
 'calibrate':'Ascend through the observatory service shaft, install the original core, select the surveyed calibration, and verify five seconds of correct output.',
 'cache':'Choose a depths cache for the lamp kit and a separate surface cache for the lock coupling. Store each item in its assigned region.',
 'take':'Travel to the remote service yard and recover its unique coupling or sky-shuttle power cell.',
 'outpost':'Return to the chosen depths cache for the lamp kit, light the freight outpost, and release its unique power module.',
 'station':'Tow the original regional module into the station, install it, and verify five seconds of stable output.',
 'ascend_station':'Ascend to the sky station control room, install the recovered spindle, and verify five seconds of stable output.',
 'reverse':'Observe a fresh inbound platform trip, board, reverse its bounded motion history toward the far landing, cancel at the endpoint, and reach the service control.',
 'update':'Inspect the physical bridge closure and use the new route evidence for the lock approach.',
 'retrieve':'Return to the surface cache chosen earlier. Recover the reserved coupling and install it in the lock drive housing.',
 'compose':'Ascend into the control loft and engage the repaired drive. Return to the adjacent workshop, assemble the barge, load relief cargo, and complete its trial.',
 'delivery':'Navigate the loaded barge to the distinct refuge assigned by the commission and unload at that refuge’s receiving bay. The three stations and lock must remain operational.'}


def verify_video(directory,manifest,movie):
    video=directory/'walkthrough.mp4';record=directory/'playback-verification.json'
    checksum=sha(video)
    assert checksum==movie['video_sha256'] and movie['trace_sha256']==manifest['actions_sha256']
    if record.exists():
        cached=json.loads(record.read_text())
        if cached['video_sha256']==checksum and cached['decoded_frames']==movie['frames'] and cached['complete']:
            return cached
    capture=cv2.VideoCapture(str(video));assert capture.isOpened();decoded=0;sizes=set()
    while True:
        ok,frame=capture.read()
        if not ok:break
        decoded+=1;sizes.add(tuple(frame.shape[:2]))
    fps=capture.get(cv2.CAP_PROP_FPS);capture.release()
    assert decoded==movie['frames'] and sizes=={(900,1600)} and fps==10
    result=dict(complete=True,decoded_frames=decoded,size=[1600,900],fps=fps,video_sha256=checksum,
        source_actions=manifest['decisions'],source_simulated_seconds=manifest['simulated_seconds'])
    save(record,result);print('DECODE VERIFIED',manifest['task'],decoded,flush=True);return result


def main():
    root=ROOT/'assets/demos/skylands-campaign';tasks={};playback={}
    for task,count in [('a',15),('b',24)]:
        d=root/f'task-{task}-seed-7';manifest=json.loads((d/'reference-manifest.json').read_text())
        movie=json.loads((d/'video-manifest.json').read_text());verification=json.loads((d/'verification.json').read_text())
        chapters=json.loads((d/'chapters.json').read_text());rows=[json.loads(s) for s in (d/'actions.jsonl').read_text().splitlines()]
        assert manifest['success'] and verification['checked_actions']==len(rows)==manifest['decisions']
        assert verification['checked_milestones']==len(chapters)==count
        for i,chapter in enumerate(chapters):
            group=[row for row in rows if row['milestone']==chapter['id']]
            chapter['instruction']=PROCEDURES[group[0]['kind']]
            chapter['observed_clues']=[e['text'] for row in group for e in row['events'] if e['type']=='clue_exposed']
            chapter['end_seconds']=chapters[i+1]['video_seconds'] if i+1<len(chapters) else movie['duration_seconds']
        playback[task]=verify_video(d,manifest,movie)
        tasks[task]=dict(manifest=manifest,movie=movie,chapters=chapters,video=f'task-{task}-seed-7/walkthrough.mp4')
    save(root/'playback-verification.json',playback);save(root/'chapter-guide.json',tasks)
    data=json.dumps(tasks).replace('<','\\u003c')
    html='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Skylands — full campaign demonstrations</title><style>
:root{font-family:system-ui,sans-serif;color-scheme:dark;background:#10232d;color:#f4edda}body{max-width:1500px;margin:auto;padding:26px}p{line-height:1.6;color:#c0d4cd}a{color:#a3dfca}button{font:inherit;color:inherit;border:1px solid #568078;background:#23453f;padding:12px;border-radius:7px;text-align:left;cursor:pointer}button:hover,button:focus-visible{background:#326759}button[aria-pressed=true]{border-color:#edcb75}nav{display:flex;gap:12px;margin:24px 0}main{display:grid;grid-template-columns:minmax(0,3fr) minmax(280px,1fr);gap:24px}video{width:100%;background:black}#chapters{display:flex;flex-direction:column;gap:8px;max-height:720px;overflow:auto}small{display:block;color:#bad5ca;margin-top:5px}li{margin:8px 0}@media(max-width:900px){main{grid-template-columns:1fr}#chapters{max-height:350px}body{padding:14px}}
</style><h1>Skylands: two full campaign demonstrations</h1>
<p>Complete regional journeys through the surface, depths and sky. The 15- and 24-milestone reference solutions execute 17,372 and 37,321 actions. Both independently replay successfully with all milestones and intact deliveries. These are original simulator environments driven by an action-only oracle.</p>
<nav><button id="a">A · Fallen Observatory · 15 milestones</button><button id="b">B · Three-Layer Lifeline · 24 milestones</button></nav>
<main><section><video id="video" controls preload="metadata"></video><p id="status"></p><article><h2 id="chapter-title"></h2><p id="instruction"></p><p id="clues"></p></article><p><a id="actions">Complete action trace</a> · <a id="verify">Replay verification</a> · <a id="meta">Renderer settings</a></p><details><summary>Simulator scope</summary><ul id="limits"></ul></details></section><aside id="chapters" aria-label="Milestone chapters"></aside></main>
<p>The full journeys use labeled travel timelapse, with slower mechanisms and pauses for annotations. Every decision remains in the action logs. Blender Cycles GPU renders the full scene at 1280 × 720, 64 samples; the 1600 × 900 H.264 video adds captions without upscaling.</p>
<script>const tasks=DATA;let task='a';const video=document.getElementById('video');function show(c){document.getElementById('chapter-title').textContent=c.id+' · '+c.title;document.getElementById('instruction').textContent=c.instruction;document.getElementById('clues').textContent=c.observed_clues.join(' ');}function choose(t){task=t;const d=tasks[t];show(d.chapters[0]);video.src=d.video;video.poster='task-'+t+'-seed-7/preview.png';for(const k of ['a','b'])document.getElementById(k).setAttribute('aria-pressed',k===t);document.getElementById('status').textContent=d.manifest.required+' milestones verified · '+d.manifest.decisions.toLocaleString()+' actions · '+(d.manifest.simulated_seconds/60).toFixed(1)+' simulated minutes · '+(d.movie.duration_seconds/60).toFixed(1)+' minutes of playback';const list=document.getElementById('chapters');list.replaceChildren();d.chapters.forEach(c=>{let b=document.createElement('button');b.textContent=c.id+' · '+c.title;let s=document.createElement('small');s.textContent=Math.floor(c.video_seconds/60)+':'+String(Math.floor(c.video_seconds%60)).padStart(2,'0');b.append(s);b.onclick=()=>{video.currentTime=c.video_seconds;show(c);video.play().catch(()=>{});};list.append(b);});for(const [id,name] of [['actions','actions.jsonl'],['verify','verification.json'],['meta','video-manifest.json']])document.getElementById(id).href='task-'+t+'-seed-7/'+name;const limits=document.getElementById('limits');limits.replaceChildren();d.manifest.limitations.forEach(s=>{let li=document.createElement('li');li.textContent=s;limits.append(li);});}for(const t of ['a','b'])document.getElementById(t).onclick=()=>choose(t);video.addEventListener('timeupdate',()=>{document.querySelectorAll('#chapters button').forEach((b,i)=>{const c=tasks[task].chapters[i];const active=video.currentTime>=c.video_seconds&&video.currentTime<c.end_seconds;b.setAttribute('aria-pressed',active);if(active)show(c);});});choose('a');</script></html>'''
    (root/'index.html').write_text(html.replace('DATA',data));print(root/'index.html')
if __name__=='__main__':main()
