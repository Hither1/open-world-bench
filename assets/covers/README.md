# Wildlands and Skylands covers

Matching 16:9 covers for the project website and GitHub README. The PNGs are
native **1920 × 1080 Blender Cycles renders**, with no baked-in text, HUD,
retouching, or upscaling. WebP copies provide smaller website downloads.

| Wildlands | Skylands |
| --- | --- |
| ![Wildlands cover](wildlands-cover.png) | ![Skylands cover](skylands-cover.png) |
| **Explore. Survive. Remember.** | **Build. Traverse. Remember.** |
| Explore a changing wilderness, manage supplies, and overcome environmental challenges. Remember landmarks, resources, and past discoveries to plan your next move. | Traverse floating islands, surface landscapes, and underground depths. Build crossings, transport cargo, and reuse earlier discoveries to complete journeys across all three layers. |

The descriptions introduce each environment as a whole. Wildlands' image shows
the full-size procedural sandbox; Skylands' image shows the introductory Aster
Reach construction course, full-size variant. Both use seed 7 at reset. The
Skylands image does not depict the larger cargo or restoration campaigns.

## Files and use

- `wildlands-cover.png`, `skylands-cover.png`: original Cycles outputs for GitHub
  and reuse. Display at 16:9, preferably at least 400 pixels wide.
- `wildlands-cover.webp`, `skylands-cover.webp`: website copies at the same pixel
  dimensions; encoded at quality 92, without resizing or image enhancement.
- [copy.json](copy.json): names, short descriptions, taglines, scope, and alt text.
- [manifest.json](manifest.json): camera positions, renderer settings, source and
  image hashes, and before/after state checks.
- [covers.zip](covers.zip): both formats, copy, and provenance in one download.

Keep the environment name and description in accessible HTML or Markdown beside
the image. The website uses the covers as video posters; its separate cover and
video captions identify the different scenes.

## Rendering and fidelity

Blender **4.5.4 LTS**, **Cycles / OptiX GPU**, 128 samples, denoising enabled,
6 maximum light bounces, AgX with Medium High Contrast, display exposure +0.65,
opaque PNG. This presentation exposure does not change the simulated lighting.
No CPU fallback. Both renders use the existing high-quality scene exporter and
the same materials and lighting pipeline as the repository's Cycles previews.

The cover camera moves independently of the agent. Export verifies that the
simulator state and RNG are unchanged; no actions are executed and the agent,
terrain, collisions, clock, and weather are not moved or edited. Existing
presentation-only horizon ridges and surrounding sea remain decorative. The
exporter retains its usual camera-dependent visibility/detail rules for props
and ground cover. These stills show environment geometry, not task performance.

Reproduce the scene exports and full GPU renders from the repository root:

```bash
sbatch assets/covers/render.slurm .cache/covers-new
```

Choose a new output directory. The script requests one GPU for up to 30 minutes
and preserves the numeric snapshots, editable `.blend` files, full-size PNGs,
and state-verification records there. To export without rendering:

```bash
python -B assets/covers/export.py .cache/covers-export-only
```

Large scene files remain in the local render cache; only delivery images and
provenance are included in the website and repository.
