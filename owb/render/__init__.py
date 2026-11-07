"""Presentation rendering: scene export, Blender Cycles and video encoding.

The voxel sensor renderers moved to :mod:`owb.sims.voxel.render`; ``owb.render.make_renderer``
remains as a deprecated alias.
"""


def __getattr__(name):
    if name == "make_renderer":
        from .._compat import warn_once
        warn_once("owb.render.make_renderer", "owb.sims.voxel.render.make_renderer")
        from ..sims.voxel.render import make_renderer
        return make_renderer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
