"""Privileged reference/oracle policies (may read simulator state; not benchmark agents).

The scripted reference policies still live in ``owb.bench.baselines``; they move here
in a later refactor step. Anything in this package must report ``privileged=True``.
"""
PRIVILEGED = True
