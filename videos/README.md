# Teyvat demo videos

Both walkthroughs were regenerated with **Blender 4.5.4 Cycles on GPU**, the
updated procedural assets, and explicit party-swap cues. Rendering uses 48
samples per pixel at 864×486 inside a 1280×720 annotated video, without upscaling.

Open the **MP4** files:

- [12 milestones — 60.5 seconds](teyvat-12-milestones.mp4)
- [24 milestones — 2 minutes 46 seconds](teyvat-24-milestones.mp4)

These copies use H.264 Constrained Baseline, `yuv420p`, and fast-start metadata.
Close and reopen any existing preview to load the regenerated file.

[Playback checks and file hashes](playback-verification.json) verify complete
decoding, exact render/toolchain settings, and complete action replays. All 17
and 44 party swaps are shown with labeled before/after pauses. These are compact native
simulator reference courses driven by a scripted oracle; see the
[scope, milestone chapters, and action replay evidence](../docs/teyvat-reference-demos.md).

The original CPU recordings remain archived in `assets/demos/teyvat-reference/`.
The new Cycles evidence packages are in `assets/demos/teyvat-reference-cycles/`.
