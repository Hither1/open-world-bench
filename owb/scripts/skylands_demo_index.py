"""Build the local chapter player and decode-check both completed MP4 files."""
from pathlib import Path
import json
import cv2
from .skylands_reference import ROOT, save, sha


def main():
    root=ROOT/'assets/demos/skylands-reference'; tasks={}; playback={}
    for task,count in [('a',15),('b',24)]:
        d=root/f'task-{task}-seed-7'; summary=json.loads((d/'summary.json').read_text())
        verification=json.loads((d/'verification.json').read_text()); chapters=json.loads((d/'chapters.json').read_text())
        rows=[json.loads(line) for line in (d/'actions.jsonl').read_text().splitlines()]
        procedures={
            'clue':'Inspect the source and retain the facts it exposes for later use.',
            'update':'Inspect the new closure report and replace the earlier approach with the alternative platform route.',
            'raft':'Join both pontoons, mount the forward fan and charged cell, secure the payload, board fully, then drive to the receiving dock and unload.',
            'stage':'Assemble and load the return raft, make a short outward test, return to the starting dock, and moor it for later retrieval.',
            'return':'Return to the staged raft, remove its test ballast, bring and secure the navigation core, then ferry it to the receiving dock.',
            'ascend':'Reach the inspected launch point below the thin ceiling, ascend into the upper gallery, and operate its service control.',
            'dock':'Use the service authorization and issued lamp to open the depths staging dock.',
            'extract':'After opening the vault service entrance, release the original core and secure it in the transport cradle.',
            'freight':'Tow the core to the loading deck and secure it on the freight lift. The lift carries the cargo upward; the agent can use a separate route.',
            'hoist':'Load the core onto the sky hoist, ascend to the elevated control gallery, and engage the hoist.',
            'calibrate':'Install the original core at the receiving cradle, select the receiver calibration observed during the survey, and verify five seconds of correct output.',
            'cache':'Choose a reserve cache and stow both the lamp kit and the lock coupling. They will be retrieved at different times.',
            'take':'Reach the service yard and recover its unique repair part or charged power cell.',
            'outpost':'Return to the chosen cache for the lamp kit, travel to the depths outpost, light the staging area, and release its power module.',
            'station':'Bring the original module to its station, install it, and verify five seconds of stable output.',
            'ascend_station':'Ascend to the sky control room, install the recovered spindle, and verify five seconds of stable output.',
            'reverse':'Observe a fresh inbound platform trip, board safely, reverse the recorded motion toward the far landing, cancel at the endpoint, and reach the service control.',
            'retrieve':'Return to the earlier chosen cache for the lock coupling, then install it in the lower drive housing.',
            'compose':'Ascend into the control loft and engage the repaired lock drive. Return to the workshop, assemble the final barge, load the relief cargo, and complete a short loaded test.',
            'delivery':'Pilot the loaded barge to the receiving dock and unload into the refuge bay designated by the commission. All stations and the lock must remain operational.',
        }
        for chapter in chapters:
            group=[r for r in rows if r['milestone']==chapter['id']]
            chapter['instruction']=procedures[group[0]['kind']]
            chapter['observed_clues']=[e['text'] for r in group for e in r['events'] if e['type']=='clue_exposed']
        assert summary['milestones']==count and verification['checked_milestones']==count and len(chapters)==count
        video=d/'walkthrough.mp4'; cap=cv2.VideoCapture(str(video)); decoded=0; sizes=set()
        assert cap.isOpened()
        while True:
            ok,frame=cap.read()
            if not ok: break
            decoded+=1; sizes.add(tuple(frame.shape[:2]))
        cap.release()
        assert decoded==summary['video_frames'] and sizes=={(720,1280)}
        assert sha(video)==summary['video_sha256']
        playback[task]=dict(decoded_frames=decoded,size=[1280,720],complete=True,video_sha256=sha(video))
        tasks[task]=dict(summary=summary,chapters=chapters,video=f'task-{task}-seed-7/walkthrough.mp4')
    save(root/'playback-verification.json',playback)
    save(root/'chapter-guide.json',tasks)
    data=json.dumps(tasks).replace('<','\\u003c')
    html='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Skylands — verified reference walkthroughs</title><style>
:root{font-family:system-ui,sans-serif;color-scheme:dark;background:#10232d;color:#f4edda}body{max-width:1400px;margin:auto;padding:24px}p{line-height:1.6;color:#c0d4cd}a{color:#a3dfca}button{font:inherit;color:inherit;border:1px solid #568078;background:#23453f;padding:12px;border-radius:7px;text-align:left;cursor:pointer}button:hover,button:focus-visible{background:#326759}button[aria-pressed=true]{border-color:#edcb75}nav{display:flex;gap:12px;margin:24px 0}main{display:grid;grid-template-columns:minmax(0,3fr) minmax(260px,1fr);gap:24px}video{width:100%;background:black}#chapters{display:flex;flex-direction:column;gap:8px;max-height:680px;overflow:auto}small{display:block;color:#bad5ca;margin-top:5px}li{margin:8px 0}@media(max-width:900px){main{grid-template-columns:1fr}#chapters{max-height:350px}body{padding:14px}}
</style><h1>Skylands reference walkthroughs</h1>
<p>Executed compact courses for the 15- and 24-milestone task designs. Every action is independently replayed and checked. Blender Cycles renders the actual simulation state on a GPU. These are scripted reference solutions, not learned-agent results or the full long-duration campaigns.</p>
<nav><button id="a">A · Fallen Observatory · 15 milestones</button><button id="b">B · Three-Layer Lifeline · 24 milestones</button></nav>
<main><section><video id="video" controls preload="metadata"></video><p id="status"></p><article><h2 id="chapter-title"></h2><p id="instruction"></p><p id="clues"></p></article><p><a id="actions">Full action trace</a> · <a id="verify">Replay verification</a> · <a id="meta">Renderer and scope</a></p><details><summary>What this prototype demonstrates and omits</summary><ul id="limits"></ul></details></section><aside id="chapters" aria-label="Milestone chapters"></aside></main>
<p>Native scene: 960 × 540, 48 Cycles samples, GPU CUDA; captioned video: 1280 × 720 H.264. No scene upscaling. Selected states are held for reading; long walking intervals are omitted.</p>
<script>const tasks=DATA;let task='a';function show(c){document.getElementById('chapter-title').textContent=c.id+' · '+c.title;document.getElementById('instruction').textContent=c.instruction;document.getElementById('clues').textContent=c.observed_clues.join(' ');}const video=document.getElementById('video');function choose(t){task=t;const d=tasks[t];show(d.chapters[0]);video.src=d.video;for(const k of ['a','b'])document.getElementById(k).setAttribute('aria-pressed',k===t);document.getElementById('status').textContent=d.summary.milestones+' milestones verified · '+d.summary.decisions+' actions · '+d.summary.simulated_seconds.toFixed(1)+' simulated seconds · '+d.summary.video_seconds.toFixed(1)+' seconds of walkthrough';const list=document.getElementById('chapters');list.replaceChildren();d.chapters.forEach(c=>{let b=document.createElement('button');b.textContent=c.id+' · '+c.title;let s=document.createElement('small');s.textContent=Math.floor(c.start_seconds/60)+':'+String(Math.floor(c.start_seconds%60)).padStart(2,'0');b.append(s);b.onclick=()=>{video.currentTime=c.start_seconds;show(c);video.play().catch(()=>{});};list.append(b);});for(const [id,name] of [['actions','actions.jsonl'],['verify','verification.json'],['meta','summary.json']])document.getElementById(id).href='task-'+t+'-seed-7/'+name;const limits=document.getElementById('limits');limits.replaceChildren();d.summary.limitations.forEach(s=>{let li=document.createElement('li');li.textContent=s;limits.append(li);});}for(const t of ['a','b'])document.getElementById(t).onclick=()=>choose(t);video.addEventListener('timeupdate',()=>{document.querySelectorAll('#chapters button').forEach((b,i)=>{const c=tasks[task].chapters[i];const active=video.currentTime>=c.start_seconds&&video.currentTime<c.end_seconds;b.setAttribute('aria-pressed',active);if(active)show(c);});});choose('a');</script></html>'''
    (root/'index.html').write_text(html.replace('DATA',data))
    print(root/'index.html')

if __name__=='__main__': main()
