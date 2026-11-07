"""Rendering backends.

`make_renderer` prefers the OpenGL backend when a GL context can actually be
created, and silently falls back to the numpy backend otherwise, so the same
code runs on a GPU node and on a driver-less login node.
"""
from __future__ import annotations

from .base import Frame, Renderer, RenderSettings
from .soft_renderer import SoftRenderer
from .style import ToonStyle

__all__ = ["Frame", "Renderer", "RenderSettings", "SoftRenderer", "ToonStyle", "make_renderer", "available_backends"]


def available_backends() -> list[str]:
    out = ["soft"]
    try:  # pragma: no cover - depends on the machine
        from .gl_renderer import gl_available

        if gl_available():
            out.insert(0, "gl")
    except Exception:
        pass
    return out


def make_renderer(settings: RenderSettings | None = None, backend: str = "auto") -> Renderer:
    """Build a renderer. `backend` is one of "auto", "gl", "soft"."""
    if backend in ("auto", "gl"):
        try:  # pragma: no cover - needs a GL driver
            from .gl_renderer import GLRenderer, gl_available

            if gl_available():
                return GLRenderer(settings)
        except Exception as exc:
            if backend == "gl":
                raise RuntimeError(f"OpenGL backend unavailable: {exc}") from exc
    return SoftRenderer(settings)
