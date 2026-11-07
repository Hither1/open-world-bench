"""Wildlands rendering backends."""


def make_renderer(cfg):
    if cfg.style == "landscape":
        from .landscape import LandscapeRenderer
        return LandscapeRenderer(cfg)
    if cfg.style == "voxel":
        from .raycast import Renderer
        return Renderer(cfg)
    raise ValueError("render_cfg.style must be 'landscape' or 'voxel'")
