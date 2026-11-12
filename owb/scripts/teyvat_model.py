"""Run bounded observation-only model pilots on the 12/24 milestone courses."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from open_world_bench.reference_tasks import ReferenceTaskEnv
from owb.bench.baselines.teyvat_model import TeyvatModelPolicy,GUIDE
ROOT=Path(__file__).resolve().parents[2]
SOURCES=[str(p.relative_to(ROOT)) for p in (ROOT/'open_world_bench').rglob('*.py')]

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def plain(value):
    if isinstance(value,np.ndarray): return value.tolist()
    if isinstance(value,np.generic): return value.item()
    if isinstance(value,dict): return {k:plain(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [plain(v) for v in value]
    return value

def checked_step(env,row):
    action=dict(row['action'])
    for key in ('move','look'): action[key]=np.asarray(action[key],np.float32)
    action['buttons']=np.asarray(action['buttons'],np.int8)
    # Rows recorded before the world map carry no pointer fields.
    action['pointer']=np.asarray(action.get('pointer',(0,0)),np.float32)
    action.setdefault('click',0)
    assert env.action_space.contains(action)
    assert plain(env.char.pos)==row['position_before']
    _,reward,terminated,truncated,info=env.step(action)
    assert env.steps==row['step'] and reward==row['reward']
    assert plain(env.char.pos)==row['position_after']
    assert plain(info['event_details'])==row['events']
    assert terminated==row['terminated'] and truncated==row['truncated']
    assert env.state_digest()==row['digest']

def policy_image(env):
    env._last_frame=None
    return env.render().copy()


MISSIONS={
 'a':'Restore the relay network and commission the summit beacon.',
 'b':'Restore the regional network and commission the distant outpost.',
}


def save(path,data):
    temp=path.with_suffix(path.suffix+'.partial')
    temp.write_text(json.dumps(plain(data),indent=2,allow_nan=False)+'\n')
    temp.replace(path)


def verify_run(directory):
    manifest=json.loads((directory/'config.json').read_text())
    summary=json.loads((directory/'summary.json').read_text())
    if sha(directory/'actions.jsonl')!=summary['actions_sha256']:
        raise RuntimeError('Action log hash differs')
    for name,digest in manifest['sources'].items():
        if sha(ROOT/name)!=digest: raise RuntimeError('Replay source changed: '+name)
    with ReferenceTaskEnv(manifest['task'],manifest['seed'],render_enabled=False) as env:
        env.reset(seed=manifest['seed'])
        assert env.state_digest()==manifest['initial_digest']
        count=0
        for line in (directory/'actions.jsonl').read_text().splitlines():
            checked_step(env,json.loads(line));count+=1
        assert env.state_digest()==summary['final_digest']
        assert env.course.completed==summary['completed'] and env.course.verify()['success']==summary['success']
        result=dict(replay_verified=True,checked_actions=count,completed=env.course.completed,
                    final_digest=env.state_digest(),actions_sha256=summary['actions_sha256'])
        save(directory/'verification.json',result)
        return result


def run(task,directory,client,*,seed=7,model='gpt-6.1-sol',effort='medium',max_calls=200,
        max_actions=6000,wall_seconds=3600,memory_chars=6000,transport='codex'):
    from PIL import Image
    from owb.bench.baselines.codex_policy import CodexPolicyError
    directory.mkdir(parents=True,exist_ok=False)
    for name in ('observations','requests','responses'): (directory/name).mkdir()
    env=ReferenceTaskEnv(task,seed,render_enabled=False)
    obs,_=env.reset(seed=seed)
    policy=TeyvatModelPolicy(client,model=model,effort=effort,memory_chars=memory_chars,max_calls=max_calls)
    sources={p:sha(ROOT/p) for p in SOURCES}
    for p in ('owb/bench/baselines/teyvat_model.py','owb/scripts/teyvat_model.py','owb/bench/baselines/codex_policy.py'):
        sources[p]=sha(ROOT/p)
    config=dict(task=task,seed=seed,model=model,transport=transport,reasoning_effort=effort,
                max_model_decisions=max_calls,max_primitive_actions=max_actions,max_wall_seconds=wall_seconds,
                memory_chars=memory_chars,history_steps=6,max_action_repeat=30,seconds_per_action=.2,
                max_consecutive_transport_errors=3,request_timeout_seconds=getattr(client,"timeout",None),
                output_token_limit_enforced=transport=='openai',observation_renderer='native policy renderer; no presentation video',
                observation_resolution=[384,216],observation_only=True,oracle_supplied=False,
                initial_digest=env.state_digest(),sources=sources,
                started_utc=datetime.now(timezone.utc).isoformat(),mission=MISSIONS[task],
                scope='Single-seed compact-course pilot; no memory ablation or transfer claim.',
                course_interface='step-feedback-v2: instruction names the current step; hint and '
                                 'off-step feedback are scoped to the current operation',
                limitations=['Native observations lack valid quest target coordinates (target_valid=0).',
                             'Local feedback names an off-step object only after the agent interacts with it.'])
    save(directory/'config.json',config)
    (directory/'control-guide.txt').write_text(GUIDE)
    started=time.monotonic();reason='running';failure=None;transport_errors=0
    def status(state):
        return dict(status=state,task=task,model=model,model_decisions=policy.calls,
                    primitive_actions=env.steps,milestones_completed=len(env.course.completed),required=len(env.course.milestones),
                    completed=env.course.completed,current_objective=env.course.current.title if env.course.current else None,
                    elapsed_wall_seconds=time.monotonic()-started,simulated_seconds=env.world.elapsed,
                    input_tokens=policy.input_tokens,output_tokens=policy.output_tokens)
    save(directory/'status.json',status('running'))
    try:
        with (directory/'actions.jsonl').open('w') as stream:
            while not (env.quests.all_complete or env.char.hp<=0 or env.steps>=env.cfg.max_episode_steps):
                if policy.calls>=max_calls: reason='model_decision_limit';break
                if env.steps>=max_actions: reason='primitive_action_limit';break
                if time.monotonic()-started>=wall_seconds: reason='wall_time_limit';break
                obs=dict(obs,rgb=policy_image(env))
                call=policy.calls+1
                Image.fromarray(obs['rgb']).save(directory/'observations'/f'{call:04}.png')
                save(directory/'status.json',dict(**status('inference'),pending_decision=call))
                try:
                    action,repeat=policy.decide(obs,MISSIONS[task])
                    transport_errors=0
                except CodexPolicyError as error:
                    transport_errors+=1
                    save(directory/'responses'/f'{call:04}-error.json',dict(type=type(error).__name__,reason=str(error)))
                    if transport_errors>=3 or not any(s in str(error) for s in ('timed out','inference failed')):
                        raise
                    # Failed requests count against the original budget; no state reset.
                    continue
                finally:
                    if policy.last_request is not None:
                        save(directory/'requests'/f'{call:04}.json',policy.last_request)
                    if policy.last_response is not None:
                        save(directory/'responses'/f'{call:04}.json',policy.last_response)
                assert env.action_space.contains(action)
                for _ in range(min(repeat,max_actions-env.steps)):
                    previous_text=obs['local_text']
                    m=env.course.current;before=env.char.pos.copy()
                    obs,reward,terminated,truncated,info=env.step(action)
                    row=dict(step=env.steps,decision=call,milestone=m.id if m else None,
                             action=plain(action),position_before=plain(before),position_after=plain(env.char.pos),
                             seconds=env.world.elapsed,events=plain(info['event_details']),reward=reward)
                    completed=any(e['type']=='milestone_complete' for e in info['event_details'])
                    row.update(digest=env.state_digest(),terminated=terminated,truncated=truncated)
                    stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush()
                    if terminated or truncated or completed or obs['local_text']!=previous_text: break
                save(directory/'status.json',status('running'))
                print(f'{task.upper()} decision {call}/{max_calls}: {len(env.course.completed)}/{len(env.course.milestones)} milestones, {env.steps} actions',flush=True)
            if (env.quests.all_complete or env.char.hp<=0 or env.steps>=env.cfg.max_episode_steps):
                reason='success' if env.course.verify()['success'] else 'death' if not (env.char.hp>0) else 'environment_time_limit'
    except Exception as error:
        reason='inference_or_interface_error'
        failure=dict(type=type(error).__name__,code=getattr(error,'code',None),http_status=getattr(error,'status_code',None))
        # Known transport errors are intentionally sanitized by CodexClient.
        from owb.bench.baselines.codex_policy import CodexPolicyError
        if isinstance(error,CodexPolicyError): failure['reason']=str(error)
        save(directory/'error.json',failure)
    finally:
        summary=dict(**status('finished'),stop_reason=reason,success=env.course.verify()['success'],
                     failure=failure,health=env.char.hp,final_digest=env.state_digest(),
                     actions_sha256=sha(directory/'actions.jsonl'),final_memory=policy.memory,
                     score_available=env.steps>0)
        save(directory/'summary.json',summary);save(directory/'status.json',summary)
        env.close()
    verify_run(directory)
    print(json.dumps(dict(task=task,stop_reason=reason,milestones=summary['milestones_completed'],
                          required=summary['required'],score_available=summary['score_available'])),flush=True)
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task',nargs='+',choices=('a','b'),default=['a','b'])
    p.add_argument('--output',type=Path,required=True);p.add_argument('--seed',type=int,default=7)
    p.add_argument('--transport',choices=('codex','openai'),default='codex')
    p.add_argument('--account-home',type=Path);p.add_argument('--codex-executable',default='codex')
    p.add_argument('--model',default='gpt-6.1-sol');p.add_argument('--effort',choices=('low','medium','high','xhigh','max'),default='medium')
    p.add_argument('--max-calls',type=int,default=200);p.add_argument('--max-actions',type=int,default=6000)
    p.add_argument('--wall-seconds',type=float,default=3600);p.add_argument('--timeout',type=float,default=90)
    p.add_argument('--memory-chars',type=int,default=6000)
    args=p.parse_args()
    if min(args.max_calls,args.max_actions,args.wall_seconds,args.timeout,args.memory_chars)<=0:
        p.error('Limits must be positive')
    if len(set(args.task))!=len(args.task): p.error('Each task may appear only once')
    if args.transport=='codex' and args.account_home is None: p.error('--account-home is required for Codex')
    for task in args.task:
        if (args.output/f'task-{task}-seed-{args.seed}').exists(): p.error('Output task directory already exists')
    def worker(task):
        if args.transport=='codex':
            from owb.bench.baselines.codex_policy import CodexClient
            client=CodexClient(args.account_home,executable=args.codex_executable,timeout=args.timeout)
        else:
            from openai import OpenAI
            client=OpenAI(timeout=args.timeout,max_retries=0)
        return run(task,args.output/f'task-{task}-seed-{args.seed}',client,seed=args.seed,
                   model=args.model,effort=args.effort,max_calls=args.max_calls,max_actions=args.max_actions,
                   wall_seconds=args.wall_seconds,memory_chars=args.memory_chars,transport=args.transport)
    with ThreadPoolExecutor(max_workers=len(args.task)) as pool:
        results=list(pool.map(worker,args.task))
    save(args.output/'results.json',results)


if __name__=='__main__': main()
