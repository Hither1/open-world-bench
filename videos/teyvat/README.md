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
[scope, milestone chapters, and action replay evidence](../../docs/teyvat-reference-demos.md).

The original CPU recordings remain archived in `assets/demos/teyvat-reference/`.
The new Cycles evidence packages are in `assets/demos/teyvat-reference-cycles/`.

## Full-scale Task B (draft)

- [24 milestones on the 1,760 m island, 4 minutes 50 seconds](teyvat-fullscale-24-milestones.mp4)

This is a first draft of the full-scale Task B reference run (seed 7, 8,324 actions,
timelapse), rendered with Blender 4.5.4 Cycles (OptiX, 64 samples) in the opt-in
stylised presentation at commit b36a90a. The file is a 1280×720 copy of the 1600×900
recording; grass under fast motion is soft at this bitrate. The run manifest, replay
verification, chapters and render settings are in
`assets/demos/teyvat-fullscale-b/seed-7/`. Later look fixes (relays, vents, a closer
camera, lighter shadows) are not in this video. See
[the design and results](../../docs/teyvat-full-scale-task-b.md).
