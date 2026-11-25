"""Milestone checkpoint banks: verified reference states that model episodes can start from.

A bank is a privileged reference trace collected on the *current* simulator code,
plus an index of the exact replay prefix after each completed milestone. A model
episode that starts at checkpoint K replays that prefix through the world's
``checked_step`` (positions, events, rewards and milestone digests all checked),
then hands control to the model. The model never sees the prefix, its actions, or
any evaluator state; it only inherits the world, inventory and satchel that the
reference left behind.

Checkpoint starts measure later segments even when a model cannot get through the
early ones, which otherwise floors every score. They are reported separately from
full-episode runs (``config.json`` records ``checkpoint``).

Layout of a bank directory::

    actions.jsonl            reference rows (same format as the world's reference traces)
    reference-manifest.json  initial/final digests, sources, actions_sha256
    verification.json        independent replay result
    checkpoints.json         one entry per K = 0..required (see ``build_index``)
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BANK_ROOT = ROOT / 'assets' / 'evaluation' / 'checkpoint-bank'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bank_dir(world, task, seed):
    return BANK_ROOT / world / f'task-{task}-seed-{seed}'


def read_rows(directory):
    return [json.loads(line) for line in (Path(directory) / 'actions.jsonl').read_text().splitlines()]


def build_index(directory):
    """Write checkpoints.json: the replay prefix length and digest after each milestone.

    Entry K means "K milestones complete". ``rows`` is how many reference actions to
    replay; ``segment_actions`` is how many the reference then needed for the next
    milestone (a difficulty/budget yardstick, not an expected model length).
    """
    directory = Path(directory)
    manifest = json.loads((directory / 'reference-manifest.json').read_text())
    if sha(directory / 'actions.jsonl') != manifest['actions_sha256']:
        raise RuntimeError('Bank action log hash differs from its manifest')
    rows = read_rows(directory)
    entries = [dict(completed=0, milestone=None, rows=0, digest=manifest['initial_digest'], seconds=0.0)]
    completed_ids = []
    for i, row in enumerate(rows):
        done = [e['milestone'] for e in row['events'] if e['type'] == 'milestone_complete']
        if not done:
            continue
        if 'digest' not in row:
            raise RuntimeError(f'Milestone row {row["step"]} has no digest')
        completed_ids.extend(done)
        entries.append(dict(completed=len(completed_ids), milestone=done[-1], rows=i + 1,
                            digest=row['digest'], seconds=row['seconds']))
    for k, entry in enumerate(entries):
        entry['completed_ids'] = completed_ids[:entry['completed']]
        nxt = entries[k + 1] if k + 1 < len(entries) else None
        entry['next_milestone'] = nxt['milestone'] if nxt else None
        entry['segment_actions'] = (nxt['rows'] - entry['rows']) if nxt else 0
        entry['segment_seconds'] = round(nxt['seconds'] - entry['seconds'], 3) if nxt else 0.0
    index = dict(task=manifest['task'], seed=manifest['seed'], required=manifest['required'],
                 actions_sha256=manifest['actions_sha256'], initial_digest=manifest['initial_digest'],
                 reference_actions=len(rows), checkpoints=entries)
    if entries[-1]['completed'] != manifest['required']:
        raise RuntimeError('Bank trace does not complete every milestone')
    (directory / 'checkpoints.json').write_text(json.dumps(index, indent=2) + '\n')
    return index


def load(directory):
    directory = Path(directory)
    index = json.loads((directory / 'checkpoints.json').read_text())
    if sha(directory / 'actions.jsonl') != index['actions_sha256']:
        raise RuntimeError('Bank action log hash differs from its index')
    return index


def restore(env, checked_step, directory, start, initial_digest, on_row=None):
    """Replay the bank prefix for ``start`` completed milestones into a freshly reset env.

    ``on_row(env, row)``, if given, runs after each replayed row (evaluator-side
    bookkeeping such as the memory-ladder ledger; it must not change the world).

    Returns the provenance dict stored in the episode config. Raises if the bank was
    collected from a different initial state or any replayed action diverges.
    """
    directory = Path(directory)
    index = load(directory)
    if not 0 <= start < len(index['checkpoints']) - 1:
        raise ValueError(f'start milestone must be in [0, {len(index["checkpoints"]) - 2}]')
    if index['initial_digest'] != initial_digest:
        raise RuntimeError('Bank was collected from a different initial state; re-collect it on this code')
    entry = index['checkpoints'][start]
    rows = read_rows(directory)[:entry['rows']]
    for row in rows:
        checked_step(env, row)
        if on_row is not None:
            on_row(env, row)
    if env.state_digest() != entry['digest'] or len(env.completed) != start:
        raise RuntimeError('Checkpoint state differs after replay')
    return dict(bank=str(directory.relative_to(ROOT) if directory.is_relative_to(ROOT) else directory),
                actions_sha256=index['actions_sha256'], start_milestone=start,
                completed_ids=entry['completed_ids'], replayed_actions=entry['rows'],
                digest=entry['digest'], simulated_seconds=entry['seconds'],
                next_milestone=entry['next_milestone'],
                reference_segment_actions=[e['segment_actions'] for e in index['checkpoints'][start:-1]])


def clue_notes(directory, start):
    """Player-visible clue texts read during the prefix, in order (the 'recalled clues' memory condition).

    These are the exact texts the reference exposed by inspecting; nothing evaluator-only.
    """
    index = load(Path(directory))
    rows = read_rows(directory)[:index['checkpoints'][start]['rows']]
    seen, out = set(), []
    for row in rows:
        for e in row['events']:
            if e['type'] == 'clue_exposed' and e.get('text') and e['text'] not in seen:
                seen.add(e['text']); out.append(e['text'])
    return out
