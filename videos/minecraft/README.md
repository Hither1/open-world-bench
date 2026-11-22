# Minecraft videos

- [Ender Dragon task, The End fight — 26 seconds](ender-dragon-end-clip.mp4)
  ([provenance](ender-dragon-end-clip.json))

Real Minecraft footage from MFM's harness on MCU, not an `owb.make_env`
environment and not a Cycles render. The source frames are 640×360,
nearest-neighbour upscaled to 1280×720 at 20 fps (real game speed), and
brightened for visibility.

The two segments come from one life (Opus 5, End-with-diamond-kit diagnostic
start, life 3), in step order:

1. Steps 1095–1290: the agent replays a post position and aim tool that the
   previous life wrote to its memory and destroys three end crystals with three
   bow shots.
2. Steps 5144–5350: its own `perch_melee` procedure. The dragon is on screen
   throughout but never perches. It strafes the agent four times (HP 20 → 3) and
   no sword hit lands.

The dragon is not killed in this clip. No MFM run on disk has landed a hit on
it. Task definitions and reference scores are in
[docs/minecraft-long-horizon-tasks.md](../../docs/minecraft-long-horizon-tasks.md).
