"""Collect, verify and index milestone checkpoint banks on the current simulator code.

    python -B -m owb.scripts.checkpoint_bank collect --world wildlands --task a --seed 7
    python -B -m owb.scripts.checkpoint_bank index   --world skylands_campaign --task b --seed 7

Skylands campaign worlds use about 10 GB; collect them in a Slurm job
(owb/scripts/checkpoint_bank.slurm), never on the login node.
"""
import argparse
from pathlib import Path

from owb.bench import checkpoints


def collect(world, task, seed, directory):
    if world == 'wildlands':
        from owb.scripts.wildlands_reference import collect as run
        run(task, seed, directory)          # creates the directory, collects, verifies
    elif world == 'skylands_campaign':
        from owb.scripts.skylands_campaign import collect as run
        run(task, seed, directory)
    else:
        raise ValueError(world)
    return checkpoints.build_index(directory)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('operation', choices=('collect', 'index'))
    p.add_argument('--world', choices=('wildlands', 'skylands_campaign'), required=True)
    p.add_argument('--task', choices=('a', 'b'), required=True)
    p.add_argument('--seed', type=int, default=7)
    p.add_argument('--directory', type=Path)
    a = p.parse_args()
    directory = a.directory or checkpoints.bank_dir(a.world, a.task, a.seed)
    index = (collect(a.world, a.task, a.seed, directory) if a.operation == 'collect'
             else checkpoints.build_index(directory))
    print(f'{a.world} {a.task}: {len(index["checkpoints"]) - 1} milestones, '
          f'{index["reference_actions"]} reference actions -> {directory}', flush=True)


if __name__ == '__main__':
    main()
