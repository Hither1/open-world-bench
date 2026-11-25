"""Run bounded observation-only model pilots on the 12/24 milestone courses."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass,replace
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shutil
import time
from typing import Callable

from owb.env.wildlands_reference import COMMANDS,WildlandsReferenceEnv,plain
from owb.bench.baselines.wildlands_model import (GOAL_NOTE,RECALL_NOTE,WildlandsModelPolicy,
    WildlandsPrivilegedPolicy,WildlandsPerceptionPolicy)
from owb.bench.baselines import privileged_view,perception_view
from owb.bench import checkpoints,memory_ladder
from owb.scripts.wildlands_reference import checked_step,sha,SOURCES,ROOT

MISSIONS={
 'a':'Repair the mountain intake, drain the flooded pass, and return to Reedbank with the road open.',
 'b':'Restore the valley floodworks, divert the rising water away from the designated refuge, and reopen its evacuation road.',
}


@dataclass(frozen=True)
class Course:
    """Everything that differs between worlds; the decision protocol is shared."""
    world: str
    make_env: Callable          # (task, seed) -> headless env
    policy: type
    missions: dict
    checked_step: Callable      # independent replay of one recorded row
    sources: tuple
    observe: Callable = lambda env,obs,events: obs   # allowlisted additions; events = previous decision's
    condition: str = 'observation_only'              # see CONDITIONS; every other value is a labelled condition
    commands: tuple = COMMANDS                       # command names, for the evaluator-only search tracker
    recall: bool = False                             # supply memory_ladder.Ledger.view() as observation.recalled
    goal: bool = False                               # supply observation.objective_location
    note: str = ''                                   # appended to the policy's guide


WILDLANDS=Course('wildlands',lambda task,seed: WildlandsReferenceEnv(task,seed,render_enabled=False),
                 WildlandsModelPolicy,MISSIONS,checked_step,
                 SOURCES+('owb/bench/baselines/wildlands_model.py','owb/scripts/wildlands_model.py',
                          'owb/bench/baselines/codex_policy.py','owb/bench/baselines/claude_policy.py',
                          # policy-view renderer: the frames the model actually sees
                          'owb/render/landscape.py','owb/render/models.py','owb/render/raycast.py'),
                 observe=lambda env,obs,events: dict(obs,holding_body=env.held is not None))
WILDLANDS_PRIVILEGED=replace(WILDLANDS,policy=WildlandsPrivilegedPolicy,condition='privileged_state',
    sources=WILDLANDS.sources+('owb/bench/baselines/privileged_view.py',),
    observe=lambda env,obs,events: dict(WILDLANDS.observe(env,obs,events),privileged=privileged_view.wildlands(env,events)))
# Between the two: a perfect detector on the current frame (visual categories and positions only).
WILDLANDS_PERCEPTION=replace(WILDLANDS,policy=WildlandsPerceptionPolicy,condition='perception_oracle',
    sources=WILDLANDS.sources+('owb/bench/baselines/perception_view.py',),
    observe=lambda env,obs,events: dict(WILDLANDS.observe(env,obs,events),perceived=perception_view.wildlands(env)))

CONDITIONS=('observation_only','perception_oracle','perfect_recall','goal_location','privileged_state')


def ladder(perception):
    """The two memory-ladder rungs above a world's perception-oracle course (docs/memory-ladder.md)."""
    recall=replace(perception,condition='perfect_recall',recall=True,note=RECALL_NOTE)
    return recall,replace(recall,condition='goal_location',goal=True,note=RECALL_NOTE+GOAL_NOTE)


CHECKPOINT_MEMORY=('empty','clues')


def takeover_note(start,memory):
    if not start: return ''
    return (f'\nCheckpoint start: you take over this mission partway through. {start} earlier objectives were '
            'already completed by another operator; the world changes, items and satchel contents they produced '
            'persist, but you did not see how they were done.'
            +(' Your memory starts with the clue texts that were read along the way.' if memory=='clues' else
              ' Your memory starts empty.'))


def clue_memory(bank,start,memory_chars):
    lines=[f'{k}. {text}' for k,text in enumerate(checkpoints.clue_notes(bank,start),1)]
    note='Clue texts read earlier in this mission (verbatim):\n'+'\n'.join(lines) if lines else ''
    if len(note)>memory_chars: raise ValueError('Recalled clues exceed the memory budget')
    return note

# Death/recovery rules applied to the env's SimConfig before reset. Runs without a
# recorded 'recovery' predate the option and used permadeath.
RECOVERY={
 'zelda':dict(respawn_on_death=True,max_continues=3,void_out=True,void_out_damage=4.0),
 'permadeath':dict(respawn_on_death=False,void_out=False),
}
RECOVERY_NOTES={
 'zelda':('\nDeath and recovery: if your stamina runs out while swimming you void out: you reappear on the '
          'last safe solid ground you stood on (never a moving deck), lose 1 heart (4 health) and drop what '
          'you hold. At 0 health you die and continue from your checkpoint, the safe ground where you last '
          'completed a milestone, with full health; milestones, items and world changes persist. You have 3 '
          'continues; the 4th death ends the episode. observation.recovery reports continues left.'),
 'permadeath':'\nDeath ends the episode; there is no respawn or continue.',
}


def build_env(course,task,seed,recovery):
    env=course.make_env(task,seed)
    for key,value in RECOVERY[recovery].items(): setattr(env.cfg.sim_cfg,key,value)
    return env


def recovery_view(env):
    s=env.cfg.sim_cfg
    return dict(deaths=env.agent.deaths,void_outs=env.void_outs,continues_used=env.continues_used,
                continues_left=max(0,s.max_continues-env.continues_used) if s.respawn_on_death else 0)


def save(path,data):
    temp=path.with_suffix(path.suffix+'.partial')
    temp.write_text(json.dumps(plain(data),indent=2,allow_nan=False)+'\n')
    temp.replace(path)


def verify_run(directory,course=WILDLANDS):
    manifest=json.loads((directory/'config.json').read_text())
    summary=json.loads((directory/'summary.json').read_text())
    if sha(directory/'actions.jsonl')!=summary['actions_sha256']:
        raise RuntimeError('Action log hash differs')
    with build_env(course,manifest['task'],manifest['seed'],manifest.get('recovery','permadeath')) as env:
        env.reset(seed=manifest['seed'])
        assert env.state_digest()==manifest['initial_digest']
        if manifest.get('checkpoint'):
            c=manifest['checkpoint']
            restored=checkpoints.restore(env,course.checked_step,checkpoints.ROOT/c['bank'],c['start_milestone'],
                                         manifest['initial_digest'])
            assert restored['digest']==c['digest'] and restored['actions_sha256']==c['actions_sha256']
        count=0
        for line in (directory/'actions.jsonl').read_text().splitlines():
            course.checked_step(env,json.loads(line));count+=1
        assert env.state_digest()==summary['final_digest']
        assert env.completed==summary['completed'] and env.success==summary['success']
        result=dict(replay_verified=True,checked_actions=count,completed=env.completed,
                    final_digest=env.state_digest(),actions_sha256=summary['actions_sha256'])
        save(directory/'verification.json',result)
        return result


def run(task,directory,client,*,seed=7,model='gpt-6.1-sol',effort='medium',max_calls=200,
        max_actions=6000,wall_seconds=3600,memory_chars=6000,transport='codex',resume_source=None,
        course=WILDLANDS,stop_after=None,recovery='zelda',start_milestone=0,checkpoint_memory='empty',bank=None):
    """One episode. With start_milestone=K>0 the world is restored from the verified checkpoint
    bank (reference prefix replayed and digest-checked) and the model plays from milestone K+1.
    Budgets, stop_after and the reported action count apply to the model's own actions only."""
    from PIL import Image
    if checkpoint_memory not in CHECKPOINT_MEMORY: raise ValueError(checkpoint_memory)
    directory.mkdir(parents=True,exist_ok=False)
    for name in ('observations','requests','responses'): (directory/name).mkdir()
    env=build_env(course,task,seed,recovery)
    obs,_=env.reset(seed=seed)
    initial_digest=env.state_digest()
    checkpoint=None
    # Evaluator-side: what the policy camera showed and what was read. It feeds the search
    # metrics in every condition and is shown to the model only under perfect_recall/goal_location.
    ledger=memory_ladder.Ledger(course.world)
    bank=Path(bank) if bank else checkpoints.bank_dir(course.world,task,seed)
    if start_milestone:
        checkpoint=checkpoints.restore(env,course.checked_step,bank,start_milestone,initial_digest,
                                       on_row=lambda e,_row: ledger.look(e,0))
        checkpoint['memory']=checkpoint_memory
        obs=env._observation()
    base_steps=env.steps
    tracker=memory_ladder.SearchTracker(course.world,ledger,course.commands,reference_paths(bank,start_milestone,initial_digest))
    tracker.begin(env,env.steps)
    recall_prefix=bool(start_milestone) and checkpoint_memory=='clues'
    policy=course.policy(client,model=model,effort=effort,memory_chars=memory_chars,max_calls=max_calls)
    policy.guide=policy.guide+course.note+RECOVERY_NOTES[recovery]+takeover_note(start_milestone,checkpoint_memory)
    if start_milestone and checkpoint_memory=='clues': policy.memory=clue_memory(bank,start_milestone,memory_chars)
    sources={p:sha(ROOT/p) for p in course.sources}
    if checkpoint: sources['owb/bench/checkpoints.py']=sha(ROOT/'owb/bench/checkpoints.py')
    sources['owb/bench/memory_ladder.py']=sha(ROOT/'owb/bench/memory_ladder.py')
    config=dict(world=course.world,task=task,seed=seed,model=model,transport=transport,reasoning_effort=effort,
                max_model_decisions=max_calls,max_primitive_actions=max_actions,max_wall_seconds=wall_seconds,
                memory_chars=memory_chars,history_steps=6,max_action_repeat=30,seconds_per_action=.2,
                max_consecutive_transport_errors=3,request_timeout_seconds=getattr(client,'timeout',None),
                output_token_limit_enforced=transport=='openai',observation_renderer='native policy renderer; no presentation video',
                observation_resolution=[env.cfg.render_cfg.width,env.cfg.render_cfg.height],
                observation_only=course.condition=='observation_only',information_condition=course.condition,
                oracle_supplied=False,
                initial_digest=initial_digest,sources=sources,
                started_utc=datetime.now(timezone.utc).isoformat(),mission=course.missions[task],
                stop_after_milestones=stop_after,recovery=recovery,recovery_rule=RECOVERY[recovery],
                start_milestone=start_milestone,checkpoint=checkpoint,
                recall_includes_prefix=recall_prefix if course.recall else None,
                search_metrics='search.json (evaluator-only; never shown to the model)',
                scope=('Single-seed pilot; not a full long-horizon memory benchmark or an ablation.'
                       +(f' Checkpoint start after {start_milestone} reference-completed milestones'
                         f' (memory: {checkpoint_memory}); report separately from full episodes.' if start_milestone else '')
                       +(f' Scored on the next {stop_after} milestones only.' if stop_after else '')))
    previous_actions='';resume_elapsed=0.
    if resume_source is not None:
        resume_source=Path(resume_source)
        old=json.loads((resume_source/'config.json').read_text())
        stopped=json.loads((resume_source/'summary.json').read_text())
        if stopped['stop_reason']!='inference_or_interface_error' or (stopped.get('failure') or {}).get('type') not in ('CodexPolicyError','ClaudePolicyError'):
            raise ValueError('Only interrupted Codex/Claude transport errors may resume')
        for key in ('task','seed','model','transport','reasoning_effort','max_model_decisions',
                    'max_primitive_actions','max_wall_seconds','memory_chars','history_steps'):
            if config[key]!=old[key]: raise ValueError(f'Resume protocol changed: {key}')
        if (old.get('world','wildlands')!=course.world or old.get('stop_after_milestones')!=stop_after
                or old.get('recovery','permadeath')!=recovery
                or old.get('information_condition','observation_only')!=course.condition
                or old.get('start_milestone',0)!=start_milestone
                or (old.get('checkpoint') or {}).get('memory')!=(checkpoint or {}).get('memory')):
            raise ValueError('Resume protocol changed: world or milestone stop')
        verify_run(resume_source,course)
        previous_actions=(resume_source/'actions.jsonl').read_text()
        last=None
        for line in previous_actions.splitlines():
            row=json.loads(line)
            if row['decision']!=last: tracker.look(env,row['decision'],env.steps);last=row['decision']
            course.checked_step(env,row);tracker.step(env,row)
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
    (directory/'control-guide.txt').write_text(policy.guide)
    from owb.bench.baselines.codex_policy import CodexPolicyError
    from owb.bench.baselines.claude_policy import ClaudePolicyError
    transport_failures=(CodexPolicyError,ClaudePolicyError)
    started=time.monotonic()-resume_elapsed;reason='running';failure=None;transport_errors=0;last_events=[]
    def status(state):
        return dict(status=state,task=task,model=model,model_decisions=policy.calls,
                    primitive_actions=env.steps-base_steps,milestones_completed=len(env.completed),required=len(env.milestones),
                    start_milestone=start_milestone,milestones_gained=len(env.completed)-start_milestone,
                    completed=env.completed,current_objective=env.current.title if env.current else None,
                    elapsed_wall_seconds=time.monotonic()-started,simulated_seconds=env.elapsed,
                    input_tokens=policy.input_tokens,output_tokens=policy.output_tokens,**recovery_view(env))
    save(directory/'status.json',status('running'))
    try:
        with (directory/'actions.jsonl').open('w') as stream:
            stream.write(previous_actions);stream.flush()
            while not env._ended:
                if policy.calls>=max_calls: reason='model_decision_limit';break
                if env.steps-base_steps>=max_actions: reason='primitive_action_limit';break
                if time.monotonic()-started>=wall_seconds: reason='wall_time_limit';break
                if stop_after and len(env.completed)-start_milestone>=stop_after: reason='milestone_target_reached';break
                obs=course.observe(env,dict(obs,rgb=env.render().copy()),last_events)
                if recovery!='permadeath': obs['recovery']=recovery_view(env)
                call=policy.calls+1
                tracker.look(env,call,env.steps)
                if course.recall: obs['recalled']=ledger.view(include_prefix=recall_prefix)
                if course.goal: obs['objective_location']=memory_ladder.objective_location(env,course.world)
                Image.fromarray(obs['rgb']).save(directory/'observations'/f'{call:04}.png')
                try:
                    action,repeat=policy.decide(obs,course.missions[task])
                    transport_errors=0
                except transport_failures as error:
                    # Transient transport failures retry, as in the Teyvat pilots: each failed
                    # call consumes the original decision budget; no state or memory reset.
                    transport_errors+=1
                    save(directory/'responses'/f'{call:04}-error.json',dict(type=type(error).__name__,reason=str(error)))
                    if transport_errors>=3 or not any(s in str(error) for s in ('timed out','inference failed')):
                        raise
                    continue
                finally:
                    if policy.last_request is not None:
                        save(directory/'requests'/f'{call:04}.json',policy.last_request)
                    if policy.last_response is not None:
                        save(directory/'responses'/f'{call:04}.json',policy.last_response)
                last_events=[]
                for _ in range(min(repeat,max_actions-(env.steps-base_steps))):
                    m=env.current;before=env.agent.pos.copy()
                    obs,reward,terminated,truncated,info=env.step(action)
                    row=dict(step=env.steps,decision=call,milestone=m.id if m else None,
                             action=plain(action),position_before=plain(before),position_after=plain(env.agent.pos),
                             seconds=env.elapsed,events=plain(info['events']),reward=reward)
                    last_events.extend(info['events'])
                    completed=any(e['type']=='milestone_complete' for e in info['events'])
                    if completed: row['digest']=env.state_digest()
                    tracker.step(env,row)
                    stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush()
                    if terminated or truncated or completed: break
                save(directory/'status.json',status('running'))
                print(f'{task.upper()} decision {call}/{max_calls}: {len(env.completed)}/{len(env.milestones)} milestones, {env.steps-base_steps} actions',flush=True)
            if env._ended:
                reason='success' if env.success else 'death' if not env.agent.alive else 'environment_time_limit'
    except Exception as error:
        reason='inference_or_interface_error'
        failure=dict(type=type(error).__name__,code=getattr(error,'code',None),http_status=getattr(error,'status_code',None))
        # Known transport errors are intentionally sanitized by CodexClient/ClaudeClient.
        if isinstance(error,transport_failures): failure['reason']=str(error)
        save(directory/'error.json',failure)
    finally:
        search=tracker.finish(env.steps)
        save(directory/'search.json',search)
        summary=dict(**status('finished'),stop_reason=reason,success=env.success,
                     failure=failure,health=env.agent.health,final_digest=env.state_digest(),
                     actions_sha256=sha(directory/'actions.jsonl'),final_memory=policy.memory,
                     score_available=env.steps>base_steps,search=search['summary'])
        save(directory/'summary.json',summary);save(directory/'status.json',summary)
        env.close()
    verify_run(directory,course)
    print(json.dumps(dict(world=course.world,task=task,stop_reason=reason,milestones=summary['milestones_completed'],
                          required=summary['required'],score_available=summary['score_available'])),flush=True)
    return summary


def episode_name(task,seed,start=0,memory='empty'):
    base=f'task-{task}-seed-{seed}'
    return base if not start else f'{base}-from-{start:02}'+('-clues' if memory=='clues' else '')


def reference_paths(bank,start,initial_digest):
    """Per-segment reference path lengths, when a bank for this exact initial state exists."""
    try:
        index=checkpoints.load(bank)
    except (FileNotFoundError,RuntimeError):     # missing or tampered bank: the yardstick is optional
        return []
    if index['initial_digest']!=initial_digest: return []
    return memory_ladder.reference_paths(checkpoints.read_rows(bank),index,start)


def main(course=WILDLANDS,description=__doc__,privileged=WILDLANDS_PRIVILEGED,perception=WILDLANDS_PERCEPTION):
    p=argparse.ArgumentParser(description=description)
    p.add_argument('--task',nargs='+',choices=('a','b'),default=['a','b'])
    p.add_argument('--output',type=Path,required=True);p.add_argument('--seed',type=int,default=7)
    p.add_argument('--transport',choices=('codex','openai','claude'),default='codex',
                   help='claude: saved Claude Code login (claude -p, no tools), e.g. --model claude-fable-5-2')
    p.add_argument('--account-home',type=Path,help='Saved CLI profile: CODEX_HOME (codex) or CLAUDE_CONFIG_DIR (claude)')
    p.add_argument('--codex-executable',default='codex');p.add_argument('--claude-executable',default='claude')
    p.add_argument('--model',default='gpt-6.1-sol');p.add_argument('--effort',choices=('low','medium','high','xhigh','max'),default='medium')
    p.add_argument('--max-calls',type=int,default=200);p.add_argument('--max-actions',type=int,default=6000)
    p.add_argument('--wall-seconds',type=float,default=3600);p.add_argument('--timeout',type=float,default=90)
    p.add_argument('--memory-chars',type=int,default=6000)
    p.add_argument('--resume',type=Path,help='Prior task folder stopped by a Codex transport error; preserve state, memory and budgets')
    p.add_argument('--stop-after-milestones',type=int,
                   help='End the episode once this many milestones are complete (counted from the start checkpoint)')
    p.add_argument('--recovery',choices=tuple(RECOVERY),default='zelda',
                   help='zelda: void-out to safe ground (-1 heart) and 3 continues from the last milestone checkpoint')
    p.add_argument('--condition',choices=CONDITIONS,default='observation_only',
                   help='perception_oracle: perfect detector on the current frame (visual categories, positions); '
                        'perfect_recall: + every fixture shown so far (last-seen position) and every text read; '
                        'goal_location: + the current objective position only; '
                        'privileged_state: labelled upper bound with evaluator state, rules and step feedback')
    p.add_argument('--privileged',action='store_true',help='Alias for --condition privileged_state')
    p.add_argument('--start-milestone',type=int,nargs='+',default=[0],
                   help='Checkpoint starts: replay the verified reference bank through K milestones, then hand over. '
                        'Several values run one episode each, concurrently.')
    p.add_argument('--checkpoint-memory',choices=CHECKPOINT_MEMORY,default='empty',
                   help="Model memory at a checkpoint start: 'empty' or 'clues' (verbatim clue texts read in the prefix)")
    p.add_argument('--bank',type=Path,help='Checkpoint bank directory (default assets/evaluation/checkpoint-bank/WORLD/task-T-seed-S)')
    args=p.parse_args()
    condition='privileged_state' if args.privileged else args.condition
    recall,goal=ladder(perception)
    course={'observation_only':course,'perception_oracle':perception,'perfect_recall':recall,
            'goal_location':goal,'privileged_state':privileged}[condition]
    if min(args.max_calls,args.max_actions,args.wall_seconds,args.timeout,args.memory_chars,
           args.stop_after_milestones or 1)<=0:
        p.error('Limits must be positive')
    if min(args.start_milestone)<0 or len(set(args.start_milestone))!=len(args.start_milestone):
        p.error('--start-milestone values must be distinct and non-negative')
    if len(set(args.task))!=len(args.task): p.error('Each task may appear only once')
    if args.resume is not None and (len(args.task)!=1 or len(args.start_milestone)!=1):
        p.error('--resume requires exactly one --task and one --start-milestone')
    if args.bank is not None and len(args.task)!=1: p.error('--bank requires exactly one --task')
    if args.transport in ('codex','claude') and args.account_home is None:
        p.error(f'--account-home is required for --transport {args.transport}')
    if args.transport=='claude' and not args.model.startswith('claude-'):
        p.error('--transport claude needs a Claude model, e.g. --model claude-fable-5-2')
    jobs=[(task,start) for task in args.task for start in args.start_milestone]
    for task,start in jobs:
        if (args.output/episode_name(task,args.seed,start,args.checkpoint_memory)).exists():
            p.error('Output task directory already exists')
        if start:   # fail fast, before any model call, if the bank is missing or stale
            index=checkpoints.load(args.bank or checkpoints.bank_dir(course.world,task,args.seed))
            if start>=len(index['checkpoints'])-1: p.error(f'--start-milestone {start} is past the last milestone')
    def worker(job):
        task,start=job
        if args.transport=='codex':
            from owb.bench.baselines.codex_policy import CodexClient
            client=CodexClient(args.account_home,executable=args.codex_executable,timeout=args.timeout)
        elif args.transport=='claude':
            from owb.bench.baselines.claude_policy import ClaudeClient
            client=ClaudeClient(args.account_home,executable=args.claude_executable,timeout=args.timeout)
        else:
            from openai import OpenAI
            client=OpenAI(timeout=args.timeout,max_retries=0)
        return run(task,args.output/episode_name(task,args.seed,start,args.checkpoint_memory),client,seed=args.seed,
                   model=args.model,effort=args.effort,max_calls=args.max_calls,max_actions=args.max_actions,
                   wall_seconds=args.wall_seconds,memory_chars=args.memory_chars,transport=args.transport,resume_source=args.resume,
                   course=course,stop_after=args.stop_after_milestones,recovery=args.recovery,
                   start_milestone=start,checkpoint_memory=args.checkpoint_memory,bank=args.bank)
    with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
        results=list(pool.map(worker,jobs))
    save(args.output/'results.json',results)


if __name__=='__main__': main()
