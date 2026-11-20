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

## Full-scale Task B (draft)

- [24 milestones on the 1,760 m island, 4 minutes 50 seconds](teyvat-fullscale-24-milestones.mp4)

This is a first draft of the full-scale Task B reference run (seed 7, 8,324 actions,
timelapse), rendered with Blender 4.5.4 Cycles (OptiX, 64 samples) in the opt-in
stylised presentation at commit b36a90a. The file is a 1280×720 copy of the 1600×900
recording; grass under fast motion is soft at this bitrate. The run manifest, replay
verification, chapters and render settings are in
`assets/demos/teyvat-fullscale-b/seed-7/`. Later look fixes (relays, vents, a closer
camera, lighter shadows) are not in this video. See
[the design and results](../docs/teyvat-full-scale-task-b.md).


# Lemon tart (mobile manipulation)

The current task is [mobile-manipulation-tasks.md](../docs/mobile-manipulation-tasks.md).

- [lemon-tart-full.mp4](lemon-tart-full.mp4) is one continuous run of that procedure from `reset`, at 6x, 7 minutes 41 seconds, MuJoCo's own renderer. Log: [lemon-tart-full.json](lemon-tart-full.json), 20/21 milestones, `success` false.
  From 0:00 to about 6:07 are the first 19 milestones, through `ingredients_restored` at step 44087.
  From 6:07 to 6:41 the tools go into the dishwasher. `tools_washed` latches at step 48075, but the rack placement in this stretch was changed after the recording.
  After 6:41 the run fails: the whisk does not come off the rack, and the last caption says the run stopped. Do not treat that stretch as a result.

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
