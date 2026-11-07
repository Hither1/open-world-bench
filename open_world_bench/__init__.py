"""Deprecated alias for :mod:`owb.sims.teyvat`.

``open_world_bench`` and every submodule (``open_world_bench.env``,
``open_world_bench.render.meshes`` ...) resolve to the same module objects under
``owb.sims.teyvat``.  Importing this package still registers the gym ids.
"""
from owb import _compat

_compat.install()
_compat.warn_once("open_world_bench", "owb.sims.teyvat")

from owb.sims.teyvat import *  # noqa: E402,F401,F403
from owb.sims.teyvat import __all__, __version__  # noqa: E402,F401


def __getattr__(name):
    # submodules were attributes of the old package because its __init__ imported them
    if name.startswith("_"):
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    try:
        return importlib.import_module(f"{__name__}.{name}")
    except ModuleNotFoundError as exc:
        if exc.name != f"{__name__}.{name}":
            raise
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
