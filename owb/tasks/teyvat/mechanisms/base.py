"""Mechanism protocol shared by the Teyvat reference-course state machines.

A mechanism is stateless: everything it changes lives in the operation's ``state`` dict
(which ``Course.state()`` and hence ``state_digest`` serialise) or on world entities. It
places its own fixture at setup time (:meth:`Mechanism.place`), then
receives the course runtime (``course``: world handle, rule bindings, ``emit`` /
``feedback`` / ``finish_operation``) and the native inputs the course forwards to it
(interactions, elemental applications, hits, ticks), and it reports progress by emitting
events. The course decides *which* operation is current; a mechanism only ever acts on the
operation it is given.
"""
from __future__ import annotations


class Mechanism:
    kind = ""
    #: elemental skills cast near this operation's entities reach :meth:`apply_element`
    accepts_elements = False
    #: this operation's enemies keep their authored stance instead of pursuing
    holds_position = False

    def place(self, hooks, op, p, rng, rules):
        """Initialise ``op.state`` and spawn the fixture's entities around site ``p``.

        Called by the course setup (:mod:`owb.tasks.teyvat.setup`) in build order;
        ``rng`` is the course RNG, so a mechanism that draws from it changes the course.
        """
        raise NotImplementedError

    def target(self, course, op):
        """The entity the operation currently needs (privileged; used by the oracle)."""
        return course.world.get(op.entities[0])

    def interact(self, course, op, char, entity):
        """The character pressed interact within reach of ``entity``."""

    def apply_element(self, course, op, char, element, entity):
        """A skill of ``element`` was cast within reach of ``entity``."""

    def hit(self, course, op, enemy, damage, element):
        """An attack hit one of this operation's enemies.

        Return ``None`` to let native combat resolve the hit, or the list of native events
        that replaces it.
        """
        return None

    def tick(self, course, op, char, dt):
        """Called once per world update while ``op`` is the current operation."""

    def verify(self, course, milestone, op):
        """Kind-specific end-of-course failures for ``op`` (strings)."""
        return []


__all__ = ["Mechanism"]
