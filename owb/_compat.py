"""Deprecated import paths kept working after the ``owb.sims`` consolidation.

A meta-path finder maps each old module prefix onto its canonical module and registers the
canonical module object under the old name, so identity holds
(``open_world_bench.env.OpenWorldEnv is owb.sims.teyvat.env.OpenWorldEnv``).  One
DeprecationWarning is issued per old prefix.
"""
from __future__ import annotations

import importlib
import importlib.abc
import importlib.machinery
import importlib.util
import sys
import warnings

#: old module prefix -> canonical module prefix
ALIASES = {
    "open_world_bench": "owb.sims.teyvat",
    "owb.core": "owb.sims.voxel.core",
    "owb.world": "owb.sims.voxel.world",
    "owb.physics": "owb.sims.voxel.physics",
    "owb.sim": "owb.sims.voxel.sim",
    "owb.entities": "owb.sims.voxel.entities",
    "owb.env": "owb.sims.voxel.env",
    # voxel sensor renderers that used to live next to the presentation modules
    **{f"owb.render.{name}": f"owb.sims.voxel.render.{name}"
       for name in ("camera", "hud", "landscape", "models", "palette", "raycast", "shadows")},
}

#: aliases backed by a real package: only their submodules go through the finder
#: (``owb.core`` is now the shared helper package; its old submodules still alias voxel core)
_SHIM_PACKAGES = {"open_world_bench", "owb.core"}
#: modules whose frames are skipped when attributing the warning
_SHIM_MODULES = {__name__, "owb", "owb.render", "owb.core", "open_world_bench",
                 # explicit re-export shims for modules that moved to owb.agents.oracle
                 "owb.sims.teyvat.reference_policy", "owb.bench.baselines.wildlands_reference"}

_warned: set[str] = set()


def _stacklevel() -> int:
    """Point the warning at the first frame outside the shims and importlib."""
    def internal(f):  # frames that ``warnings.warn`` itself skips when counting
        return "importlib" in f.f_code.co_filename and "_bootstrap" in f.f_code.co_filename

    frame, level = sys._getframe(1), 1
    while frame is not None and (internal(frame) or frame.f_globals.get("__name__") in _SHIM_MODULES):
        frame = frame.f_back
        while frame is not None and internal(frame):
            frame = frame.f_back
        level += 1
    return level


def warn_once(old: str, new: str) -> None:
    if old in _warned:
        return
    _warned.add(old)
    warnings.warn(f"{old} is deprecated; import {new} instead", DeprecationWarning, stacklevel=_stacklevel())


def _match(fullname: str) -> str | None:
    for old in ALIASES:
        if fullname.startswith(old + ".") or (fullname == old and old not in _SHIM_PACKAGES):
            return old
    return None


class _AliasLoader(importlib.abc.Loader):
    def __init__(self, old: str, target: str):
        self.old, self.target = old, target

    def create_module(self, spec):
        return None  # placeholder; exec_module swaps in the canonical module

    def exec_module(self, module):
        warn_once(self.old, ALIASES[self.old])
        sys.modules[module.__name__] = importlib.import_module(self.target)


class _AliasFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        old = _match(fullname)
        if old is None:
            return None
        canonical = ALIASES[old] + fullname[len(old):]
        if importlib.util.find_spec(canonical) is None:
            return None
        return importlib.machinery.ModuleSpec(fullname, _AliasLoader(old, canonical))


def install() -> None:
    if not any(isinstance(f, _AliasFinder) for f in sys.meta_path):
        sys.meta_path.insert(0, _AliasFinder())
