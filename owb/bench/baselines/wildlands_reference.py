"""Deprecated: the privileged Wildlands reference controller moved to
:mod:`owb.agents.oracle.wildlands_reference`. This module re-exports it."""
from owb import _compat

_compat.warn_once(__name__, "owb.agents.oracle.wildlands_reference")

from owb.agents.oracle.wildlands_reference import WildlandsReferencePolicy, neutral  # noqa: E402,F401

__all__ = ["WildlandsReferencePolicy", "neutral"]
