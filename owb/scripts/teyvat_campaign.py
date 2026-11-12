"""Record and independently replay native-action Teyvat campaign feasibility runs.

No API/model calls and no presentation rendering are performed by this command.
Headless zero RGB is used only for logic validation. Screenshots/videos use the
existing Cycles pipeline on the recorded simulator state.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np

from open_world_bench.campaign_env import TeyvatCampaignEnv, plain
from open_world_bench.campaign_policy import CampaignReferencePolicy
from open_world_bench.campaign_tasks import GOALS, SUITE_VERSION

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sources():
    files = sorted((ROOT/'open_world_bench').rglob('*.py'))
    files += [Path(__file__).resolve(),ROOT/'owb/render/scene_export.py']
    return {str(p.relative_to(ROOT)):sha(p) for p in files}


def write_json(path, value):
    path.write_text(json.dumps(plain(value),indent=2,allow_nan=False)+'\n')


def geometry_digest(env):
    w = env.world
    h = hashlib.sha256()
    for array in (w.terrain.height,w.props.pos,w.props.kind,w.props.scale,w.props.yaw):
        h.update(array.tobytes())
    h.update(json.dumps(plain(dict(buildings=[b.metadata() for b in w.architecture.buildings],
                                  gates=env.campaign.gates)),sort_keys=True).encode())
    return h.hexdigest()


def collect(task, seed, directory, *, geography_scale=2.0, reverse=False, max_steps=40000):
    directory = Path(directory)
    directory.mkdir(parents=True,exist_ok=False)
    started = time.monotonic()
    with TeyvatCampaignEnv(task,seed,render_enabled=False,width=16,height=12,
                          geography_scale=geography_scale,max_steps=max_steps) as env:
        obs,_ = env.reset(seed=seed)
        policy = CampaignReferencePolicy(env,reverse=reverse)
        manifest = dict(suite_version=SUITE_VERSION,task=task,seed=seed,geography_scale=geography_scale,
                        max_steps=max_steps,reverse=reverse,decision_seconds=env.cfg.dt*env.cfg.action_repeat,
                        sources=sources(),initial_digest=env.state_digest(),geometry_digest=geometry_digest(env),
                        goal=GOALS[task],milestones=[vars(m) for m in env.campaign.milestones.values()],
                        observation='Headless zero RGB for logic validation only; final goal plus local text',
                        controller='Privileged reference controller; native actions only; no world writes',
                        artifact_visibility='Evaluator only: contains hidden graph and state hashes',
                        runtime=dict(python=platform.python_version(),numpy=np.__version__))
        write_json(directory/'manifest.json',manifest)
        with (directory/'actions.jsonl').open('w') as stream:
            for step in range(max_steps):
                before = env.state_digest()
                position = env.char.pos.copy()
                action = policy(obs)
                if env.state_digest()!=before: raise RuntimeError('Reference policy mutated simulator state')
                assert env.action_space.contains(action)
                obs,reward,terminated,truncated,info = env.step(action)
                row = dict(step=env.steps,action=action,position_before=position,position_after=env.char.pos,
                           events=info['event_details'],reward=reward,terminated=terminated,truncated=truncated,
                           state_digest=env.state_digest())
                stream.write(json.dumps(plain(row),allow_nan=False)+'\n')
                for event in info['event_details']:
                    if event['type']=='milestone_complete':
                        stream.flush()
                        print(f'{task} seed {seed}: {event["milestone"]} at action {env.steps}',flush=True)
                if terminated or truncated: break
        result = dict(env.campaign.verify(),actions=env.steps,simulated_seconds=env.world.elapsed,
                      milestone_order=env.campaign.completed,final_digest=env.state_digest(),
                      actions_sha256=sha(directory/'actions.jsonl'),geometry_digest=geometry_digest(env),
                      wall_seconds=time.monotonic()-started)
        assert result['geometry_digest']==manifest['geometry_digest']
        write_json(directory/'summary.json',result)
    verification = verify(directory)
    print(f'{task}: success={result["success"]}, actions={result["actions"]}, replay={verification["verified"]}',flush=True)
    return result


def verify(directory):
    directory = Path(directory)
    manifest = json.loads((directory/'manifest.json').read_text())
    expected = json.loads((directory/'summary.json').read_text())
    for path,digest in manifest['sources'].items():
        if sha(ROOT/path)!=digest: raise RuntimeError('Replay source changed: '+path)
    assert sha(directory/'actions.jsonl')==expected['actions_sha256']
    with TeyvatCampaignEnv(manifest['task'],manifest['seed'],render_enabled=False,width=16,height=12,
                          geography_scale=manifest['geography_scale'],max_steps=manifest['max_steps']) as env:
        env.reset(seed=manifest['seed'])
        assert env.state_digest()==manifest['initial_digest']
        assert geometry_digest(env)==manifest['geometry_digest']
        count = 0
        with (directory/'actions.jsonl').open() as stream:
            for line in stream:
                row = json.loads(line)
                action = dict(row['action'])
                for key in ('move','look'): action[key]=np.asarray(action[key],np.float32)
                action['buttons'] = np.asarray(action['buttons'],np.int8)
                assert env.action_space.contains(action)
                assert env.char.pos.tolist()==row['position_before']
                _,reward,terminated,truncated,info = env.step(action)
                count += 1
                assert env.steps==row['step'] and env.char.pos.tolist()==row['position_after']
                assert plain(info['event_details'])==row['events']
                assert reward==row['reward'] and terminated==row['terminated'] and truncated==row['truncated']
                assert env.state_digest()==row['state_digest'],f'replay diverged at action {count}'
        assert count==expected['actions'] and env.state_digest()==expected['final_digest']
        assert env.campaign.completed==expected['milestone_order']
        assert env.campaign.verify()['success']==expected['success']
        result = dict(verified=True,success=expected['success'],checked_actions=count,
                      checked_positions=True,checked_events=True,checked_state_digests=True,
                      geometry_digest=geometry_digest(env),actions_sha256=expected['actions_sha256'],
                      scope='Feasibility and deterministic replay; not evidence of learned planning or memory')
        write_json(directory/'verification.json',result)
        return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task',nargs='+',choices=tuple(GOALS),default=list(GOALS))
    p.add_argument('--seed',nargs='+',type=int,default=[7])
    p.add_argument('--scale',type=float,default=2.0,help='Regional geography scale; 2 is full, 0.5 is compact validation')
    p.add_argument('--reverse',action='store_true',help='Validate a different branch order')
    p.add_argument('--max-steps',type=int,default=40000)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--output',type=Path)
    mode.add_argument('--verify',type=Path)
    args = p.parse_args()
    if args.verify:
        print(json.dumps(verify(args.verify),indent=2));return
    if args.max_steps<=0 or not np.isfinite(args.scale) or args.scale<.5:
        p.error('max-steps must be positive and scale at least 0.5')
    args.output.mkdir(parents=True,exist_ok=False)
    results=[]
    for task in args.task:
        for seed in args.seed:
            name=f'{task}-seed-{seed}-'+('reverse' if args.reverse else 'forward')
            result=collect(task,seed,args.output/name,geography_scale=args.scale,
                           reverse=args.reverse,max_steps=args.max_steps)
            results.append(dict(task=task,seed=seed,**result))
    write_json(args.output/'summary.json',dict(suite_version=SUITE_VERSION,runs=results))
    if not all(r['success'] for r in results): raise SystemExit(1)


if __name__=='__main__':
    main()
