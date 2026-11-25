"""Memory-ladder information conditions and evaluator-only search metrics.

The question this module serves: when a model fails a later milestone, is the
bottleneck *remembering* what it has seen and read, or *using* that memory to
decide where to go (memory-guided exploration)? Each rung of the ladder adds one
kind of information to the one below; everything else is identical:

=================== ======================================================================
observation_only    native RGB, sensors, text, visited-cell map, the model's own notes
perception_oracle   + ``perceived``: a perfect detector on the current frame
perfect_recall      + ``recalled``: every fixture the agent has been shown, with the
                    position where it was last seen, and every text it has read
goal_location       + ``objective_location``: the current objective's position only
=================== ======================================================================

``perception_oracle`` -> ``perfect_recall`` measures what memory is worth.
``perfect_recall`` -> ``goal_location`` measures what *using* memory to choose where
to go is worth, once nothing has to be remembered. ``privileged_state`` is off the
ladder: it also reveals roles, rules, mechanism state and failure reasons.

``recalled`` is built from what the policy camera actually showed at the agent's
decision points (or, for a checkpoint start with ``clues`` memory, every frame of
the reference prefix), never from world state the agent did not observe. Movable
objects keep their last *seen* position. ``objective_location`` gives a position
and, in Skylands, a layer: no roles, target IDs, procedures or answers.

The :class:`SearchTracker` runs in every condition and never reaches the model. It
splits each milestone segment into a *search* phase (until the agent is first
within interaction reach of one of the objective's own targets) and an
*execution* phase (from there to completion). It also records whether those
targets had been seen before the segment began, by the agent or by the earlier
operator, so return-to-known-place segments can be told apart from genuine search.
"""
from __future__ import annotations

import math

import numpy as np

from .baselines import perception_view
from .baselines.privileged_view import skylands_owners

CELL_M = 4.0            # horizontal revisit-grid cell
CELL_H = 8.0            # vertical bucket, so Skylands layers stay separate
ARRIVE_M = 6.0          # arrival radius when an objective has no interaction targets
RELATIVE = ('distance_m', 'bearing_deg')   # meaningless once the agent has moved

PERCEIVE = {'wildlands': perception_view.wildlands, 'skylands_campaign': perception_view.skylands}
TARGETED = {
    'wildlands': {'interact', 'inspect', 'operate', 'grip', 'install'},
    'skylands_campaign': {'inspect', 'pick', 'mount', 'load', 'unload', 'drive', 'use', 'take',
                          'cache', 'retrieve', 'reverse', 'cancel', 'haul'},
}
REACH = {'wildlands': 3.2, 'skylands_campaign': 4.5}
LADDER = ('observation_only', 'perception_oracle', 'perfect_recall', 'goal_location')


def readings(env):
    """Texts the player is currently being shown because they read something."""
    out = []
    text = getattr(env, 'last_text', '')
    if text:
        out.append(text)
    cue = getattr(env, 'wm_cue', None)          # Wildlands working-memory cue line
    if cue and cue.get('text'):
        out.append(cue['text'])
    return out


class Ledger:
    """Everything the policy camera has shown and every text read, keyed by stable identity.

    Evaluator-side bookkeeping. Under ``perfect_recall``/``goal_location`` its
    :meth:`view` is supplied to the model; in other conditions it only feeds the
    search metrics.
    """

    def __init__(self, world):
        self.world = world
        self.objects = {}           # key -> entry
        self.texts = {}             # text -> {'prefix': bool, 'agent': bool}; insertion-ordered

    def look(self, env, when):
        """Record the current frame. ``when`` is the decision number, or 0 for the reference prefix.

        Returns the set of target IDs in frame.
        """
        ids = set()
        for e in PERCEIVE[self.world](env, limit=None, creatures=False)['in_frame']:
            key = e['id'] if 'id' in e else (e['kind'], tuple(e['pos']))
            fields = {k: v for k, v in e.items() if k not in RELATIVE and k != 'pos'}
            old = self.objects.get(key) or dict(first_seen=when, agent_first=None)
            self.objects[key] = dict(fields, last_seen_pos=e['pos'], first_seen=old['first_seen'], last_seen=when,
                                     agent_first=old['agent_first'] if old['agent_first'] or not when else when)
            if 'id' in e:
                ids.add(e['id'])
        self.read(env, prefix=not when)
        return ids

    def read(self, env, prefix=False):
        for text in readings(env):
            flags = self.texts.setdefault(text, dict(prefix=False, agent=False))
            flags['prefix' if prefix else 'agent'] = True

    def view(self, include_prefix):
        """What the model is given. Without ``include_prefix`` only the agent's own experience counts."""
        def seen(d):
            return 'before takeover' if d == 0 else d
        objects = []
        for _, e in sorted(self.objects.items(), key=lambda kv: (isinstance(kv[0], tuple), str(kv[0]))):
            first = e['first_seen'] if include_prefix else e['agent_first']
            if first is None:
                continue
            fields = {k: v for k, v in e.items() if k not in ('first_seen', 'last_seen', 'agent_first')}
            objects.append(dict(fields, first_seen_decision=seen(first), last_seen_decision=seen(e['last_seen'])))
        texts = [t for t, f in self.texts.items() if f['agent'] or (include_prefix and f['prefix'])]
        return dict(objects=objects, texts=texts)


def objective_location(env, world):
    """Where the current objective is, and nothing else."""
    m = env.current
    if m is None:
        return None
    if world == 'wildlands':
        from ..env.wildlands_reference import Y
        x, z = m.site
        return dict(position=[float(x), float(Y), float(z)])
    from ..env.skylands import LAYERS
    return dict(position=np.round(m.pos, 1).tolist(), layer=LAYERS[m.layer])


def objective_targets(env, world, m):
    """(target ID, current position) for the milestone's own interaction targets."""
    out = []
    if world == 'wildlands':
        for tid, obj in env.targets.items():
            if obj.get('milestone') != m.id:
                continue
            pos = obj['pos']
            if obj['role'].startswith('weight'):
                body = env.bodies.get(m.state['bodies'][int(obj['role'][6:])])
                if body is None:
                    continue
                pos = body.pos
            out.append((tid, np.asarray(pos, float)))
        return out
    _, owners = skylands_owners(env, m)
    probes = {k for k, p in getattr(env, 'wm', {}).items() if p.milestone == m.id}
    for tid, obj in env.targets.items():
        if obj['owner'] in owners or obj['owner'] in probes:
            out.append((tid, np.asarray(obj['pos'], float)))
    return out


def _cell(pos):
    return (math.floor(pos[0] / CELL_M), math.floor(pos[1] / CELL_H), math.floor(pos[2] / CELL_M))


def _flat(a, b):
    return float(math.hypot(a[0] - b[0], a[2] - b[2]))


class SearchTracker:
    """Per-segment search/execution split, sightings, path and revisits. Evaluator-only."""

    def __init__(self, world, ledger, commands, reference_segments=None):
        self.world, self.ledger, self.commands = world, ledger, commands
        self.reference = list(reference_segments or [])     # per-segment reference path metres
        self.segments, self.seg, self.cells = [], None, set()

    # -- segment boundaries -------------------------------------------------------------
    def _open(self, env, decision, step):
        m = env.current
        if m is None:
            self.seg = None
            return
        targets = objective_targets(env, self.world, m)
        ids = {t for t, _ in targets}
        seen = [self.ledger.objects[t] for t in ids if t in self.ledger.objects]
        goal = objective_location(env, self.world)['position']
        pos = np.asarray(env.agent.pos, float)
        nearest = min((_flat(pos, p) for _, p in targets), default=_flat(pos, goal))
        k = len(self.segments)
        self.seg = dict(milestone=m.id, start_decision=decision, start_step=step, targets=sorted(ids),
                        seen_in_prefix=any(e['first_seen'] == 0 for e in seen),
                        seen_by_agent=any(e['agent_first'] is not None for e in seen),
                        start_distance_m=round(nearest, 1), first_in_frame_decision=None,
                        reached_step=None, reached_decision=None, completed_step=None,
                        path_m=0.0, path_to_reach_m=None, steps=0, cell_entries=0, revisits=0, new_cells=0,
                        first_targeted=None, wrong_targeted_before_reach=0,
                        reference_path_m=self.reference[k] if k < len(self.reference) else None)
        self._last = pos
        self._check_reach(env, decision, step)

    def _check_reach(self, env, decision, step):
        s = self.seg
        if s is None or s['reached_step'] is not None:
            return
        targets = objective_targets(env, self.world, env.by_id[s['milestone']])
        if targets:
            hit = any(env._reachable(p, REACH[self.world]) for _, p in targets)
        else:
            goal = objective_location(env, self.world)['position']
            hit = float(np.linalg.norm(np.asarray(env.agent.pos) - goal)) <= ARRIVE_M
        if hit:
            s['reached_step'], s['reached_decision'] = step, decision
            s['path_to_reach_m'] = round(s['path_m'], 1)

    def _close(self, step, completed):
        s = self.seg
        if s is None:
            return
        if completed:
            s['completed_step'] = step
        s['path_m'] = round(s['path_m'], 1)
        n = step - s['start_step']
        s['actions'] = n
        s['search_actions'] = (s['reached_step'] - s['start_step']) if s['reached_step'] is not None else n
        s['execution_actions'] = (step - s['reached_step']) if s['reached_step'] is not None else 0
        s['revisit_fraction'] = round(s['revisits'] / s['cell_entries'], 3) if s['cell_entries'] else 0.0
        s['completed'] = completed
        s['kind'] = ('known' if s['seen_by_agent'] else 'seen_by_earlier_operator' if s['seen_in_prefix']
                     else 'unseen')
        self.segments.append(s)
        self.seg = None

    # -- hooks called by the runner --------------------------------------------------------
    def begin(self, env, step):
        """Start of model control (after any checkpoint prefix)."""
        self.here = _cell(env.agent.pos)
        self.cells.add(self.here)
        self._open(env, 1, step)

    def look(self, env, decision, step):
        """A frame was shown to the model at this decision."""
        ids = self.ledger.look(env, decision)
        s = self.seg
        if s is not None and s['first_in_frame_decision'] is None and ids & set(s['targets']):
            s['first_in_frame_decision'] = decision

    def step(self, env, row):
        """After one model primitive action (``row`` is the actions.jsonl row)."""
        self.ledger.read(env)
        s = self.seg
        if s is not None:
            pos = np.asarray(row['position_after'], float)
            s['path_m'] += _flat(self._last, pos)
            self._last = pos
            s['steps'] += 1
            cell = _cell(pos)
            if cell != self.here:               # count cell *entries*, not steps spent inside one
                self.here = cell
                s['cell_entries'] += 1
                if cell in self.cells:
                    s['revisits'] += 1
                else:
                    self.cells.add(cell)
                    s['new_cells'] += 1
            action = row['action']
            name = self.commands[int(action['command'])]
            target = int(action.get('target', 0) or 0)
            if name in TARGETED[self.world] and target > 0:
                hit = target in s['targets']
                if s['first_targeted'] is None:
                    s['first_targeted'] = dict(command=name, on_objective=hit, step=row['step'])
                if not hit and s['reached_step'] is None:
                    s['wrong_targeted_before_reach'] += 1
            self._check_reach(env, row['decision'], row['step'])
        done = [e['milestone'] for e in row['events'] if e['type'] == 'milestone_complete']
        if done and s is not None:
            self._close(row['step'], True)
            self._open(env, row['decision'], row['step'])

    def finish(self, step):
        self._close(step, False)
        return dict(segments=self.segments, summary=summarise(self.segments))


def summarise(segments):
    def mean(xs):
        xs = [x for x in xs if x is not None]
        return round(sum(xs) / len(xs), 2) if xs else None
    out = dict(segments=len(segments), completed=sum(s['completed'] for s in segments),
               reached=sum(s['reached_step'] is not None for s in segments),
               search_actions=mean([s['search_actions'] for s in segments]),
               execution_actions=mean([s['execution_actions'] for s in segments if s['reached_step'] is not None]),
               revisit_fraction=mean([s['revisit_fraction'] for s in segments]))
    reached = [s for s in segments if s['reached_step'] is not None]
    # 1.0 = walked straight to the objective; lower = detours or wandering before arrival.
    out['reach_efficiency'] = mean([min(1.0, s['start_distance_m'] / s['path_to_reach_m']) for s in reached
                                    if s['path_to_reach_m'] and s['path_to_reach_m'] > 1])
    done = [s for s in segments if s['completed'] and s['reference_path_m'] and s['reference_path_m'] > 1]
    out['path_ratio_to_reference'] = mean([s['path_m'] / s['reference_path_m'] for s in done])
    out['by_kind'] = {}
    for kind in ('known', 'seen_by_earlier_operator', 'unseen'):
        ks = [s for s in segments if s['kind'] == kind]
        if ks:
            out['by_kind'][kind] = dict(segments=len(ks), reached=sum(s['reached_step'] is not None for s in ks),
                                        completed=sum(s['completed'] for s in ks),
                                        search_actions=mean([s['search_actions'] for s in ks]))
    return out


def reference_paths(rows, index, start):
    """Horizontal path metres the reference walked in each whole segment after checkpoint ``start``."""
    out = []
    entries = index['checkpoints']
    for k in range(start, len(entries) - 1):
        a, b = entries[k]['rows'], entries[k + 1]['rows']
        seg = rows[a:b]
        out.append(round(sum(_flat(r['position_before'], r['position_after']) for r in seg), 1))
    return out
