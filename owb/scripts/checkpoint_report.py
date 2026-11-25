"""Summarise model runs by information condition, checkpoint start and memory condition.

    python -B -m owb.scripts.checkpoint_report runs/NAME [runs/OTHER ...]

Reads every episode folder (config.json + summary.json + verification.json) under the
given run directories and writes ``checkpoint-report.json`` and ``checkpoint-report.md``
into the first one. Episodes whose replay did not verify, or that stopped on an
infrastructure error before acting, are listed but not scored.

Per episode: ``gained`` = milestones completed by the model itself after its start
checkpoint, and ``next`` = whether it completed the first milestone it was handed.
Per (world, task, condition, memory) group: the fraction of starts whose next milestone
was completed and the mean milestones gained. These segment-level numbers are what
checkpoint starts exist for; they are not full-episode success rates.

Search metrics (``search.json``, evaluator-only; runs from 2026-11-24 onward) split the
handed segment into search (until first within reach of the objective's own targets) and
execution. The ladder table puts the memory-ladder conditions side by side for each
(world, task, model, memory): perception oracle -> perfect recall is the memory gap,
perfect recall -> goal location is the memory-guided exploration gap (docs/memory-ladder.md).
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path

LADDER = ('observation_only', 'perception_oracle', 'perfect_recall', 'goal_location')


def episodes(directories):
    for root in directories:
        for config in sorted(Path(root).glob('**/config.json')):
            folder = config.parent
            if not (folder / 'summary.json').exists():
                continue
            yield folder, json.loads(config.read_text()), json.loads((folder / 'summary.json').read_text())


def row(folder, config, summary):
    verification = folder / 'verification.json'
    verified = verification.exists() and json.loads(verification.read_text()).get('replay_verified', False)
    start = config.get('start_milestone', 0)
    checkpoint = config.get('checkpoint') or {}
    gained = summary.get('milestones_gained', summary['milestones_completed'] - start)
    segment = checkpoint.get('reference_segment_actions') or []
    target = config.get('stop_after_milestones')
    search = folder / 'search.json'
    first = (json.loads(search.read_text())['segments'] or [None])[0] if search.exists() else None
    return dict(folder=str(folder), world=config.get('world', 'wildlands'), task=config['task'], seed=config['seed'],
                model=config['model'], condition=config.get('information_condition', 'observation_only'),
                memory=checkpoint.get('memory', 'own') if start else 'own', start=start,
                handed=checkpoint.get('next_milestone'), target=target, gained=gained, next=gained >= 1,
                reached_target=bool(target) and gained >= target, success=summary['success'],
                model_actions=summary['primitive_actions'], decisions=summary['model_decisions'],
                reference_actions_for_target=sum(segment[:target]) if target else sum(segment),
                stop=summary['stop_reason'],
                scored=verified and summary.get('score_available', True),
                # The handed segment only: was its site seen before, was it reached, at what cost.
                first_kind=first and first['kind'], first_reached=first and first['reached_step'] is not None,
                first_search_actions=first and first['search_actions'],
                first_execution_actions=first and first['execution_actions'],
                first_revisit_fraction=first and first['revisit_fraction'])


def table(rows):
    lines = ['| World | Task | Condition | Memory | Start | Gained | Next done | Model actions | Ref. actions | Decisions | Stop '
             '| Site | Reached | Search act. | Exec. act. | Revisit |',
             '| --- | --- | --- | --- | ---: | ---: | --- | ---: | ---: | ---: | --- | --- | --- | ---: | ---: | ---: |']
    for r in sorted(rows, key=lambda r: (r['world'], r['task'], r['condition'], r['memory'], r['start'])):
        gained = r['gained'] if r['scored'] else 'n/a'
        lines.append(f"| {r['world']} | {r['task']} | {r['condition']} | {r['memory']} | {r['start']} | {gained}"
                     f"{'/' + str(r['target']) if r['target'] else ''} | {'yes' if r['next'] and r['scored'] else 'no'}"
                     f" | {r['model_actions']} | {r['reference_actions_for_target']} | {r['decisions']} | {r['stop']} |"
                     + (f" {r['first_kind']} | {'yes' if r['first_reached'] else 'no'} | {r['first_search_actions']} |"
                        f" {r['first_execution_actions']} | {r['first_revisit_fraction']} |" if r['first_kind'] else
                        ' n/a | n/a | n/a | n/a | n/a |'))
    return '\n'.join(lines)


def aggregate(rows):
    groups = defaultdict(list)
    for r in rows:
        if r['scored']:
            groups[(r['world'], r['task'], r['model'], r['condition'], r['memory'])].append(r)
    out = []
    for (world, task, model, condition, memory), rs in sorted(groups.items()):
        out.append(dict(world=world, task=task, model=model, condition=condition, memory=memory, starts=len(rs),
                        next_milestone_rate=round(sum(r['next'] for r in rs) / len(rs), 3),
                        mean_gained=round(sum(r['gained'] for r in rs) / len(rs), 3),
                        reach_rate=rate([r['first_reached'] for r in rs if r['first_kind']]),
                        mean_search_actions=avg([r['first_search_actions'] for r in rs if r['first_kind']]),
                        start_points=sorted(r['start'] for r in rs)))
    return out


def rate(xs):
    return round(sum(map(bool, xs)) / len(xs), 3) if xs else None


def avg(xs):
    return round(sum(xs) / len(xs), 1) if xs else None


def ladder(groups):
    """One line per (world, task, model, memory) with each ladder rung's next-milestone and reach rates."""
    cells = defaultdict(dict)
    for g in groups:
        if g['condition'] in LADDER:
            cells[(g['world'], g['task'], g['model'], g['memory'])][g['condition']] = g
    if not any(len(c) > 1 for c in cells.values()):
        return ''

    def fmt(g):
        if not g:
            return '–'
        reach = 'n/a' if g['reach_rate'] is None else f"{g['reach_rate']:.2f}"
        return f"{g['next_milestone_rate']:.2f} / {reach} (n={g['starts']})"

    def gap(c, hi, lo):
        return (f"{c[hi]['next_milestone_rate'] - c[lo]['next_milestone_rate']:+.2f}"
                if hi in c and lo in c and c[hi]['start_points'] == c[lo]['start_points'] else '–')
    lines = ['Ladder: next-milestone rate / reach rate of the handed segment. Gaps compare identical start sets only.', '',
             '| World | Task | Model | Memory | ' + ' | '.join(LADDER) + ' | Memory gap | Exploration gap |',
             '| --- | --- | --- | --- |' + ' --- |' * len(LADDER) + ' ---: | ---: |']
    for key, c in sorted(cells.items()):
        lines.append('| ' + ' | '.join(map(str, key)) + ' | ' + ' | '.join(fmt(c.get(k)) for k in LADDER)
                     + f" | {gap(c, 'perfect_recall', 'perception_oracle')} | {gap(c, 'goal_location', 'perfect_recall')} |")
    return '\n'.join(lines) + '\n\n'



def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('runs', nargs='+', type=Path)
    a = p.parse_args()
    rows = [row(*e) for e in episodes(a.runs)]
    if not rows:
        raise SystemExit('No finished episodes found')
    groups = aggregate(rows)
    summary = ['| World | Task | Model | Condition | Memory | Starts | Next-milestone rate | Mean gained |',
               '| --- | --- | --- | --- | --- | ---: | ---: | ---: |']
    summary += [f"| {g['world']} | {g['task']} | {g['model']} | {g['condition']} | {g['memory']} | {g['starts']} | "
                f"{g['next_milestone_rate']:.2f} | {g['mean_gained']:.2f} |" for g in groups]
    unscored = [r['folder'] for r in rows if not r['scored']]
    text = '\n'.join(summary) + '\n\n' + ladder(groups) + table(rows) + '\n'
    if unscored:
        text += '\nNot scored (replay unverified or no actions):\n' + '\n'.join(f'- {u}' for u in unscored) + '\n'
    (a.runs[0] / 'checkpoint-report.md').write_text(text)
    (a.runs[0] / 'checkpoint-report.json').write_text(json.dumps(dict(groups=groups, episodes=rows), indent=2) + '\n')
    print(text)


if __name__ == '__main__':
    main()
