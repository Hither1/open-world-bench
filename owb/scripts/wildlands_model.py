"""Run bounded observation-only model pilots on the 12/24 milestone courses."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shutil
import time

from owb.env.wildlands_reference import WildlandsReferenceEnv,plain
from owb.bench.baselines.wildlands_model import WildlandsModelPolicy,GUIDE
from owb.scripts.wildlands_reference import checked_step,sha,SOURCES,ROOT

MISSIONS={
 'a':'Repair the mountain intake, drain the flooded pass, and return to Reedbank with the road open.',
 'b':'Restore the valley floodworks, divert the rising water away from the designated refuge, and reopen its evacuation road.',
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
    with WildlandsReferenceEnv(manifest['task'],manifest['seed']) as env:
        env.reset(seed=manifest['seed'])
        assert env.state_digest()==manifest['initial_digest']
        count=0
        for line in (directory/'actions.jsonl').read_text().splitlines():
            checked_step(env,json.loads(line));count+=1
        assert env.state_digest()==summary['final_digest']
        assert env.completed==summary['completed'] and env.success==summary['success']
        result=dict(replay_verified=True,checked_actions=count,completed=env.completed,
                    final_digest=env.state_digest(),actions_sha256=summary['actions_sha256'])
        save(directory/'verification.json',result)
        return result


def run(task,directory,client,*,seed=7,model='gpt-6.1-sol',effort='medium',max_calls=200,
        max_actions=6000,wall_seconds=3600,memory_chars=6000,transport='codex',resume_source=None):
    from PIL import Image
    directory.mkdir(parents=True,exist_ok=False)
    for name in ('observations','requests','responses'): (directory/name).mkdir()
    env=WildlandsReferenceEnv(task,seed,render_enabled=False)
    obs,_=env.reset(seed=seed)
    policy=WildlandsModelPolicy(client,model=model,effort=effort,memory_chars=memory_chars,max_calls=max_calls)
    sources={p:sha(ROOT/p) for p in SOURCES}
    for p in ('owb/bench/baselines/wildlands_model.py','owb/scripts/wildlands_model.py','owb/bench/baselines/codex_policy.py'):
        sources[p]=sha(ROOT/p)
    config=dict(task=task,seed=seed,model=model,transport=transport,reasoning_effort=effort,
                max_model_decisions=max_calls,max_primitive_actions=max_actions,max_wall_seconds=wall_seconds,
                memory_chars=memory_chars,history_steps=6,max_action_repeat=30,seconds_per_action=.2,
                output_token_limit_enforced=transport=='openai',observation_renderer='native policy renderer; no presentation video',
                observation_resolution=[320,180],observation_only=True,oracle_supplied=False,
                initial_digest=env.state_digest(),sources=sources,
                started_utc=datetime.now(timezone.utc).isoformat(),mission=MISSIONS[task],
                scope='Single-seed compact-course pilot; not a full long-horizon memory benchmark or an ablation.')
    previous_actions='';resume_elapsed=0.
    if resume_source is not None:
        resume_source=Path(resume_source)
        old=json.loads((resume_source/'config.json').read_text())
        stopped=json.loads((resume_source/'summary.json').read_text())
        if stopped['stop_reason']!='inference_or_interface_error' or (stopped.get('failure') or {}).get('type')!='CodexPolicyError':
            raise ValueError('Only interrupted Codex transport errors may resume')
        for key in ('task','seed','model','transport','reasoning_effort','max_model_decisions',
                    'max_primitive_actions','max_wall_seconds','memory_chars','history_steps'):
            if config[key]!=old[key]: raise ValueError(f'Resume protocol changed: {key}')
        verify_run(resume_source)
        previous_actions=(resume_source/'actions.jsonl').read_text()
        for line in previous_actions.splitlines(): checked_step(env,json.loads(line))
        assert env.state_digest()==stopped['final_digest']
        obs=env._observation()
        request=json.loads(sorted((resume_source/'requests').glob('*.json'))[-1].read_text())
        policy.memory=request['memory'];policy.history.extend(request['recent_history'])
        policy.calls=stopped['model_decisions']
        policy.input_tokens=stopped['input_tokens'];policy.output_tokens=stopped['output_tokens']
        for name in ('observations','requests','responses'):
            for source in (resume_source/name).iterdir(): shutil.copyfile(source,directory/name/source.name)
        config['started_utc']=old['started_utc']
        config['resumed_from']=str(resume_source.resolve())
        config['prior_summary_sha256']=sha(resume_source/'summary.json')
        config['prior_sources']=old['sources']
        resume_elapsed=(datetime.now(timezone.utc)-datetime.fromisoformat(old['started_utc'])).total_seconds()
    save(directory/'config.json',config)
    (directory/'control-guide.txt').write_text(GUIDE)
    started=time.monotonic()-resume_elapsed;reason='running';failure=None
    def status(state):
        return dict(status=state,task=task,model=model,model_decisions=policy.calls,
                    primitive_actions=env.steps,milestones_completed=len(env.completed),required=len(env.milestones),
                    completed=env.completed,current_objective=env.current.title if env.current else None,
                    elapsed_wall_seconds=time.monotonic()-started,simulated_seconds=env.elapsed,
                    input_tokens=policy.input_tokens,output_tokens=policy.output_tokens)
    save(directory/'status.json',status('running'))
    try:
        with (directory/'actions.jsonl').open('w') as stream:
            stream.write(previous_actions);stream.flush()
            while not env._ended:
                if policy.calls>=max_calls: reason='model_decision_limit';break
                if env.steps>=max_actions: reason='primitive_action_limit';break
                if time.monotonic()-started>=wall_seconds: reason='wall_time_limit';break
                obs=dict(obs,rgb=env.render().copy())
                call=policy.calls+1
                Image.fromarray(obs['rgb']).save(directory/'observations'/f'{call:04}.png')
                try:
                    action,repeat=policy.decide(obs,MISSIONS[task])
                finally:
                    if policy.last_request is not None:
                        save(directory/'requests'/f'{call:04}.json',policy.last_request)
                    if policy.last_response is not None:
                        save(directory/'responses'/f'{call:04}.json',policy.last_response)
                for _ in range(min(repeat,max_actions-env.steps)):
                    m=env.current;before=env.agent.pos.copy()
                    obs,reward,terminated,truncated,info=env.step(action)
                    row=dict(step=env.steps,decision=call,milestone=m.id if m else None,
                             action=plain(action),position_before=plain(before),position_after=plain(env.agent.pos),
                             seconds=env.elapsed,events=plain(info['events']),reward=reward)
                    completed=any(e['type']=='milestone_complete' for e in info['events'])
                    if completed: row['digest']=env.state_digest()
                    stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush()
                    if terminated or truncated or completed: break
                save(directory/'status.json',status('running'))
                print(f'{task.upper()} decision {call}/{max_calls}: {len(env.completed)}/{len(env.milestones)} milestones, {env.steps} actions',flush=True)
            if env._ended:
                reason='success' if env.success else 'death' if not env.agent.alive else 'environment_time_limit'
    except Exception as error:
        reason='inference_or_interface_error'
        failure=dict(type=type(error).__name__,code=getattr(error,'code',None),http_status=getattr(error,'status_code',None))
        # Known transport errors are intentionally sanitized by CodexClient.
        from owb.bench.baselines.codex_policy import CodexPolicyError
        if isinstance(error,CodexPolicyError): failure['reason']=str(error)
        save(directory/'error.json',failure)
    finally:
        summary=dict(**status('finished'),stop_reason=reason,success=env.success,
                     failure=failure,health=env.agent.health,final_digest=env.state_digest(),
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
    p.add_argument('--resume',type=Path,help='Prior task folder stopped by a Codex transport error; preserve state, memory and budgets')
    args=p.parse_args()
    if min(args.max_calls,args.max_actions,args.wall_seconds,args.timeout,args.memory_chars)<=0:
        p.error('Limits must be positive')
    if len(set(args.task))!=len(args.task): p.error('Each task may appear only once')
    if args.resume is not None and len(args.task)!=1: p.error('--resume requires exactly one --task')
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
                   wall_seconds=args.wall_seconds,memory_chars=args.memory_chars,transport=args.transport,resume_source=args.resume)
    with ThreadPoolExecutor(max_workers=len(args.task)) as pool:
        results=list(pool.map(worker,args.task))
    save(args.output/'results.json',results)


if __name__=='__main__': main()
