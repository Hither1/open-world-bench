"""Mechanism protocol shared by the Wildlands reference-course state machines.

A mechanism is stateless. At reset it places its fixture (:meth:`Mechanism.place`);
afterwards it reads and writes the milestone's ``state`` dict (which
``state_digest`` serialises) and the env's world/body/satchel handles, and reports what
happened as native-style event dicts. ``env`` is the hosting ``WildlandsReferenceEnv``;
the course ledger (:mod:`owb.tasks.wildlands.course`) decides completion from
:meth:`Mechanism.complete`.
"""
from __future__ import annotations

import numpy as np

#: walking level of the authored course pads (top of the brick floor at y=28)
Y = 29.0


def plain(x):
    """JSON-ready copy of ``x`` (arrays/numpy scalars to Python, sets sorted)."""
    if isinstance(x, np.ndarray): return x.tolist()
    if isinstance(x, np.generic): return x.item()
    if isinstance(x, dict): return {str(k): plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [plain(v) for v in x]
    if isinstance(x, set): return sorted(x)
    return x


def fail(reason, **kw):
    return [dict(type='action_failed', reason=reason, **kw)]


class Mechanism:
    kinds: tuple = ()

    def place(self, hooks, m, rng, rules):
        """Initialise ``m.state`` and place the fixture's controls and blocks at ``m.site``.

        Called by the course setup (:mod:`owb.tasks.wildlands.setup`) in course order;
        ``rng`` is the course RNG. The default is a single console: an ``inspect`` board
        for clues, an ``interact`` control otherwise (raised onto the gallery deck for the
        elevated service consoles A11/B15).
        """
        x, z = m.site
        elevated = m.id in ('A11', 'B15')
        y = Y+4 if elevated else Y
        hooks.target(m, 'inspect' if m.kind == 'clue' else 'interact', (x+.5, y+1, z+.5))
        return y

    def install_blocked(self, env, m):
        """Reason why parts cannot be installed yet (``None`` when the socket is ready)."""
        return None

    def operate(self, env, m, role, obj):
        """``operate``/``interact`` on one of the milestone's controls; returns events."""
        return fail('inspect this source instead')

    def clue(self, env, m):
        """Text shown by ``inspect`` on one of the milestone's controls."""
        return m.title

    def update(self, env, m, events, kinds, dt):
        """Per-step update after the native ticks; may append events."""

    def complete(self, env, m):
        """The milestone's completion condition (checked only while it is eligible)."""
        return False


__all__ = ["Mechanism", "Y", "fail", "plain"]


def family_roles(env, m):
    """Symbol -> control role of the milestone's primary controls (taught at the lessons)."""
    return {env.targets[v]['symbol']: k for k, v in m.targets.items()
            if k in ('brake', 'release', 'vent', 'heat', 'inlet')}


__all__ += ["family_roles"]
