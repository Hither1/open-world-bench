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

# Lemon tart (mobile manipulation)

The current task is [mobile-manipulation-tasks.md](../docs/mobile-manipulation-tasks.md).
A robot trajectory for that procedure is not recorded yet.

- [lemon-tart-reference.mp4](lemon-tart-reference.mp4) is an **older version**.
  The PandaOmron finishes the previous nine milestones with its own 12-dimensional
  actions: seed 1, 22402 control steps, shown at 4x. Summary:
  [lemon-tart-reference.json](lemon-tart-reference.json). It needs `--horizon 24000`;
  the default 20000-step limit cuts it off at 7/9. On-screen captions in this file
  are Chinese and describe those old milestones. Rendered with MuJoCo's native
  offscreen renderer: RoboCasa has no Cycles exporter.
- [State walkthrough](lemon-tart-state-walkthrough.mp4): poses written directly into
  the simulator for the same old milestones. Not robot actions. Captions are Chinese.
- [lemon-tart-mix.mp4](lemon-tart-mix.mp4): bowl, flour, and a sugar cube. The flour
  bag is one rigid body and nothing pours out. Seed 1, 4857 steps.
- [lemon-tart-mix-usable.mp4](lemon-tart-mix-usable.mp4), plus
  [flour](lemon-tart-usable-flour-pour.mp4), [sugar](lemon-tart-usable-sugar-pour.mp4),
  and [egg](lemon-tart-usable-egg-pour.mp4): grains pour. The same usable run
  also cuts the butter and the lemon. Captions are Chinese.
- [lemon-tart-oven-lock.mp4](lemon-tart-oven-lock.mp4) and
  [lemon-tart-oven-knobs.mp4](lemon-tart-oven-knobs.mp4): optional door lock and
  knob order. Captions are Chinese.
- [lemon-tart-tart-pan.png](lemon-tart-tart-pan.png): the metal tart pan used when
  `usable_ingredients=True`. The default task and the old reference video still use a plate.

Details: [../docs/lemon_tart_reference.md](../docs/lemon_tart_reference.md).
