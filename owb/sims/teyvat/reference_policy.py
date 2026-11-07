"""Deprecated: the privileged Teyvat reference controller moved to
:mod:`owb.agents.oracle.teyvat_reference`. This module re-exports it."""
from owb import _compat

_compat.warn_once(__name__, "owb.agents.oracle.teyvat_reference")

from owb.agents.oracle.teyvat_reference import ReferencePolicy  # noqa: E402,F401

__all__ = ["ReferencePolicy"]
