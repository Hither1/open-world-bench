"""Collect and independently replay full-scale Skylands campaign traces."""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
import pickle
import time
import numpy as np
from owb.env.skylands_campaign import SkylandsCampaignEnv
from owb.bench.baselines.skylands_campaign import SkylandsCampaignPolicy
from owb.scripts.skylands_reference import save,sha,plain,checked_step,action_from_json,ROOT,SOURCES as BASE_SOURCES

SOURCES=BASE_SOURCES+('owb/env/skylands_campaign.py','owb/bench/baselines/skylands_campaign.py','owb/scripts/skylands_campaign.py')
LIMITATIONS=[
 'Privileged scripted action-only oracle; demonstrates executability, not learned skills or memory performance.',
 'Rated four-mount cargo craft and collision-backed grid-stepped actuators; no general rigid-body vehicle or fluid solver.',
 'Cargo uses a grounded roller-cradle position ledger; it cannot follow the agent through Ascend.',
 'Authored regional campaign geometry; the scene is not Nintendo game footage.',
 'Standard cold-weather clothing and paraglider supplied at reset.',
 'This reference itinerary does not validate every recovery, alternative construction, branch order, or held-out geometry in the research design.'
]
def sources(): return {p:sha(ROOT/p) for p in SOURCES}


def collect(task,seed,directory,*,resume=None):
    directory.mkdir(parents=True,exist_ok=True)
    if resume:
        with Path(resume).open('rb') as f: env,policy,obs,initial,counts=pickle.load(f)
        mode='a'; print('RESUME',env.steps,env.current.id,flush=True)
    else:
        if (directory/'actions.jsonl').exists(): raise FileExistsError(directory/'actions.jsonl')
        env=SkylandsCampaignEnv(task,seed); print('GENERATE',task,flush=True)
        obs,_=env.reset(seed=seed); print('WORLD',env.world.shape,flush=True)
        initial=env.state_digest(); policy=SkylandsCampaignPolicy(env); counts=Counter(); mode='w'
    last_progress=env.steps; recent=[]; start=time.monotonic()
    try:
        with (directory/'actions.jsonl').open(mode) as stream:
            while not env._ended:
                m=env.current; before=env.agent.pos.copy(); a=policy(obs)
                assert env.action_space.contains(a),a
                obs,reward,terminated,truncated,info=env.step(a)
                counts.update(e['type'] for e in info['events'])
                row=dict(step=env.steps,milestone=m.id,title=m.title,kind=m.kind,action=plain(a),
                    explanation=policy.explanation,position_before=plain(before),position_after=plain(env.agent.pos),
                    seconds=env.elapsed,reward=reward,events=plain(info['events']))
                recent=(recent+[row])[-8:]
                if any(e['type']=='milestone_complete' for e in info['events']):
                    row['digest']=env.state_digest(); last_progress=env.steps
                    print('MILESTONE',task,env.steps,len(env.completed),m.id,'health',env.agent.health,'seconds',round(env.elapsed,1),flush=True)
                stream.write(json.dumps(row,allow_nan=False)+'\n')
                if env.steps%500==0:
                    stream.flush(); print('PROGRESS',env.steps,m.id,env.agent.pos.tolist(),policy.explanation,'health',env.agent.health,'wall',round(time.monotonic()-start),flush=True)
                if env.steps%1000==0:
                    save(directory/'progress.json',dict(step=env.steps,milestone=m.id,position=env.agent.pos.tolist(),explanation=policy.explanation))
                    with (directory/'resume.pkl').open('wb') as f: pickle.dump((env,policy,obs,initial,counts),f,protocol=5)
        if not env.success or env.agent.deaths: raise RuntimeError('Campaign did not finish successfully')
        save(directory/'reference-manifest.json',dict(task=task,seed=seed,initial_digest=initial,final_digest=env.state_digest(),
            decisions=env.steps,simulated_seconds=env.elapsed,milestones=env.completed,required=len(env.milestones),
            success=env.success,health=env.agent.health,events=dict(counts),limitations=LIMITATIONS,
            world_shape=env.world.shape,walked_metres=env.agent.distance_travelled,
            return_dock=env.return_choice,cache_bindings=env.cache_bindings,
            controller='privileged action-only full-campaign reference policy',sources=sources(),actions_sha256=sha(directory/'actions.jsonl')))
    except BaseException:
        save(directory/'failure-context.json',dict(recent=recent,position=env.agent.pos,health=env.agent.health,
            hauling=env.hauling,cargo=env.cargo,rafts=env.rafts,movers={k:{a:(list(b) if a=='history' else b) for a,b in v.items()} for k,v in env.movers.items()}))
        with (directory/'failure.pkl').open('wb') as f: pickle.dump((env,policy,obs,initial,counts),f,protocol=5)
        raise
    finally: env.close()
    return verify(directory)


def replay(directory):
    manifest=json.loads((directory/'reference-manifest.json').read_text())
    if sha(directory/'actions.jsonl')!=manifest['actions_sha256']: raise RuntimeError('Action log hash differs')
    rows=[json.loads(line) for line in (directory/'actions.jsonl').read_text().splitlines()]
    env=SkylandsCampaignEnv(manifest['task'],manifest['seed']); env.reset(seed=manifest['seed'])
    if env.state_digest()!=manifest['initial_digest']: raise RuntimeError('Initial state mismatch')
    return env,manifest,rows


def verify(directory):
    env,manifest,rows=replay(directory)
    try:
        for row in rows: checked_step(env,row)
        assert env.success and env.system_operational() and len(env.completed)==manifest['required'] and not env.agent.deaths
        assert env.state_digest()==manifest['final_digest']
        result=dict(success=True,checked_actions=len(rows),checked_positions=True,checked_events=True,checked_rewards=True,
                    checked_milestones=len(env.completed),final_live_predicate=True,final_digest=env.state_digest(),
                    actions_sha256=manifest['actions_sha256'],sources=sources())
        save(directory/'verification.json',result); print('VERIFIED',manifest['task'],len(rows),flush=True); return result
    finally: env.close()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('operation',choices=('collect','verify'))
    p.add_argument('--task',choices=('a','b'),required=True);p.add_argument('--seed',type=int,default=7)
    p.add_argument('--directory',type=Path);p.add_argument('--resume',type=Path)
    a=p.parse_args(); d=a.directory or Path(f'assets/demos/skylands-campaign/task-{a.task}-seed-{a.seed}')
    collect(a.task,a.seed,d,resume=a.resume) if a.operation=='collect' else verify(d)
if __name__=='__main__': main()
