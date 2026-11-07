"""Build a local chapter player and instruction transcript for verified videos."""
from pathlib import Path
import json

ROOT=Path(__file__).resolve().parents[2]/'assets/demos/wildlands-reference'
GUIDANCE={
 'a':[
 ('Remember a location','Inspect the commission. Retain the ranger-cache landmark for A08.'),
 ('Learn support and release','Observe the empty-load brake trip. Engage the brake, magnetically place one slab on the cradle, drop it so its mass is supported, then release. Wait for stable opening.'),
 ('Learn a safe updraft','Read the local shaft instructions. Open the vent, ignite the sheltered heat source, jump and deploy the glider into the updraft. Land on the upper gallery.'),
 ('Practice support and release','Use the same device-family meanings on the second gate. Brake first; establish the required supported load, then release and verify the gate opens.'),
 ('Practice a safe updraft','Open the new shaft vent, enable sheltered heat, and glide to the elevated gallery. Landing awards the salvage kit.'),
 ('Remember a calibration','Inspect the survey source. Save the seeded intake calibration mark and the depot bearings for later use.'),
 ('Intervening work','Bring the salvage kit to the supply locker. Exchange it for the replacement regulator and cache key.'),
 ('Retrieve the earlier location','Return to the cache identified at A01. Spend the key from A07 to obtain the service handle.'),
 ('Recall support and release','At the intake housing, engage the brake and build the larger supported load with two slabs. Release only when the gauge is correct; verify stable opening.'),
 ('Recall a safe updraft','Establish a sheltered draft in rainy weather. Jump, glide through the shaft, and land on the intake gallery.'),
 ('Combine parts and remembered facts','Install both the regulator and service handle. Select the calibration from A06, engage the intake, and maintain valid operation for five simulated seconds. The road drains.'),
 ('Verify the outcome','Walk through the drained road from its south side to its north side, then report at Reedbank while the intake remains operational.'),
 ],
 'b':[
 ('Remember the authorization','Read the floodworks commission. Retain the authorized empty overflow basin; the final outlet symbols will need a separate mapping.'),
 ('Learn support and release','Observe a release with insufficient support trip the brake. Engage the brake, place and drop a slab on the cradle, release, and verify stable lift operation.'),
 ('Practice support and release','Use the same control-family meanings at the second gate. Establish supported mass with the brake held, then release.'),
 ('Learn a safe updraft','Inspect the shaft instructions. Open the draft and ignite the sheltered hearth. Jump, deploy the glider into the updraft, and land on the gallery.'),
 ('Practice a safe updraft','Repeat the vent, protected heat, launch, and landing procedure at the second gallery.'),
 ('Learn route before feeding','Keep the inlet closed. Open only the locally requested outlet, then enable the inlet and wait for five seconds of stable operation.'),
 ('Practice route before feeding','Read the new outlet request. Configure that branch with the inlet isolated, then feed and verify stable operation.'),
 ('Remember a depot and route','Inspect the reserve-depot survey. Remember the eastern causeway depot and the original wooden bridge approach.'),
 ('Intervening supply work','Gather peppers and a chillshroom, light the cooking campfire, and cook both cold-resistance and heat-resistance provisions.'),
 ('Collect a repair part','Use the commission authorization at the wind-coupling locker and collect its coupling.'),
 ('Collect two repair parts','Use the authorization at the second locker to collect the pump and quarry-engine couplings.'),
 ('Remember mapping and applicability','Inspect the channel survey. Retain the outlet-symbol-to-basin mapping and the maximum safe shaft-launch temperature of 45 C.'),
 ('Recall support and release','Engage the ridge brake, establish the required supported mass, and release to open the ridge housing.'),
 ('Recall a safe updraft','Open the wind-deck shaft draft and enable its protected heat source. Glide onto the upper service deck.'),
 ('Commission the wind station','Use the wind coupling on the upper service deck. Hold valid station operation for five simulated seconds.'),
 ('Recall route before feeding','Install the pump coupling. With the inlet shut, configure the requested outlet, then enable feed and verify five seconds of stable flow.'),
 ('Check whether a skill is safe to use','The quarry shaft is too hot to launch. Route the cooling bypass before feeding it, then wait until the measured temperature is at most 45 C.'),
 ('Compose cooling and updraft skills','After the cooling check, open the shaft draft and enable heat. Glide to the service deck, install the engine coupling, and verify stable operation.'),
 ('Update an earlier memory','Inspect the storm report. The wooden bridge is now physically gone; retain the new north-stone-detour route to the same depot.'),
 ('Reuse a location with updated routing','Return to the remembered reserve depot using the detour. Collect the reserve winch after all three stations are online.'),
 ('Compose support and repair','Hold the final lock with its brake, establish the required supported load, and release. Install the reserve winch only after the support lock is valid.'),
 ('Join two distant facts','Combine the basin authorization from B01 with the symbol-to-basin map from B12. Open the matching final outlet while keeping the inlet shut.'),
 ('Compose all three skills','Establish a safe updraft and reach the upper controls. Enable the prepared final inlet, verify stable flow with the repaired stations and lock, and confirm that the road drains.'),
 ('Verify the mission outcome','Traverse the reopened road from south to north and signal the designated refuge while the floodworks remain operational.'),
 ]}


def main():
    tasks={}
    for task in ('a','b'):
        folder=ROOT/f'task-{task}-seed-7'
        chapters=json.loads((folder/'chapters.json').read_text())
        assert len(chapters)==len(GUIDANCE[task])
        for chapter,(role,instruction) in zip(chapters,GUIDANCE[task]):
            chapter.update(memory_role=role,instruction=instruction)
        tasks[task]=dict(chapters=chapters,video=f'task-{task}-seed-7/walkthrough.mp4',
                        summary=json.loads((folder/'summary.json').read_text()))
    (ROOT/'chapter-guide.json').write_text(json.dumps(tasks,indent=2)+'\n')
    data=json.dumps(tasks).replace('<','\\u003c')
    html='''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Wildlands — verified reference walkthroughs</title>
<style>
:root{color-scheme:dark;font-family:system-ui,sans-serif;background:#12272e;color:#f4efdc}
body{max-width:1440px;margin:auto;padding:24px}h1{font-size:30px;margin:6px 0}p{line-height:1.6;color:#bfd5ce}
button,a{font:inherit}button{background:#20434b;border:1px solid #4b7779;color:#f4efdc;border-radius:8px;padding:12px;cursor:pointer;text-align:left}button:hover,button:focus-visible{background:#305861}button[aria-pressed=true]{border-color:#d9bc79;background:#365b60}
nav{display:flex;gap:12px;margin:20px 0}main{display:grid;grid-template-columns:minmax(0,3fr) minmax(280px,1fr);gap:24px}video{width:100%;background:#000;border-radius:8px}#chapters{display:flex;flex-direction:column;gap:8px;max-height:760px;overflow:auto}#chapters small{display:block;color:#bfd5ce;margin-top:5px}article{border-left:3px solid #d9bc79;padding:0 18px;margin:24px 0}article h2{font-size:20px}a{color:#a8dedc}footer{margin-top:32px;font-size:14px}#status{font-variant-numeric:tabular-nums} @media(max-width:900px){main{grid-template-columns:1fr}#chapters{max-height:360px}body{padding:14px}}
</style>
<h1>Wildlands reference walkthroughs</h1>
<p>Two successful, replay-verified compact courses. The videos show selected actual simulator states held for reading, rendered in Blender Cycles. A privileged oracle performs the actions; these are not learned-agent memory results.</p>
<nav aria-label="Choose task"><button id="task-a" aria-pressed="true">A · Frostbound Pass · 12 milestones</button><button id="task-b" aria-pressed="false">B · Valley Before the Storm · 24 milestones</button></nav>
<main><section><video id="video" controls preload="metadata"></video><p id="status"></p>
<article aria-live="polite"><h2 id="title"></h2><strong id="role"></strong><p id="instruction"></p></article>
<p><a id="trace">Complete action log</a> · <a id="verification">Replay verification</a> · <a id="summary">Renderer and scope metadata</a></p>
</section><aside id="chapters" aria-label="Milestone chapters"></aside></main>
<footer>Seed 7. Native scene: 960 × 540, 48 Cycles samples; captions: 1280 × 720, H.264. No scene upscaling. Travel and geography are compact; the full long-duration memory benchmark, varied transfer instances, and memory ablations remain separate work. <a href="../../../docs/wildlands-long-horizon-tasks.md">Full task design</a></footer>
<script>const tasks=DATA;let task='a',active=-1;const v=document.getElementById('video');
function show(i){if(i===active)return;active=i;const c=tasks[task].chapters[i];document.getElementById('title').textContent=c.id+' · '+c.title;document.getElementById('role').textContent=c.memory_role;document.getElementById('instruction').textContent=c.instruction;document.querySelectorAll('#chapters button').forEach((b,j)=>b.setAttribute('aria-pressed',j===i));}
function choose(t){task=t;active=-1;const d=tasks[t];v.src=d.video;document.querySelectorAll('nav button').forEach(b=>b.setAttribute('aria-pressed',b.id==='task-'+t));document.getElementById('status').textContent=d.summary.required+'/'+d.summary.required+' milestones verified · '+d.summary.decisions+' primitive actions · '+d.summary.simulated_seconds.toFixed(1)+' simulated seconds · '+d.summary.video_seconds.toFixed(0)+' seconds of checkpoint video';const list=document.getElementById('chapters');list.replaceChildren();d.chapters.forEach((c,i)=>{let b=document.createElement('button');b.textContent=c.id+' · '+c.title;let s=document.createElement('small');s.textContent=Math.floor(c.start_seconds/60)+':'+String(c.start_seconds%60).padStart(2,'0')+' · '+c.memory_role;b.append(s);b.onclick=()=>{v.currentTime=c.start_seconds;show(i);v.play().catch(()=>{});};list.append(b);});for(const [id,name] of [['trace','actions.jsonl'],['verification','verification.json'],['summary','summary.json']])document.getElementById(id).href='task-'+t+'-seed-7/'+name;show(0);}
v.addEventListener('timeupdate',()=>{const cs=tasks[task].chapters;let i=cs.findIndex(c=>v.currentTime>=c.start_seconds&&v.currentTime<c.end_seconds);if(i>=0)show(i);});for(const t of ['a','b'])document.getElementById('task-'+t).onclick=()=>choose(t);choose('a');</script></html>'''
    (ROOT/'index.html').write_text(html.replace('DATA',data))
    print(ROOT/'index.html')


if __name__=='__main__': main()
