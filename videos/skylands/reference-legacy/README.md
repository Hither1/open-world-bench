# Earlier Skylands campaign recordings

These are the complete **earlier rectangular-world** Cycles recordings of
The Fallen Observatory (A, 15 milestones) and The Three-Layer Lifeline (B, 24).
They predate the organic-world and working-memory-probe updates. They depict a
privileged reference controller, not model evaluation results.

Blender 4.5.4 Cycles, OptiX GPU, 64 samples. The native 1280 × 720 rendered scene
sits inside the original 1600 × 900 annotated video. Every original encoded frame
is retained: 4,793 for A (7:59.3), 9,145 for B (15:14.5).

The original MP4s exceed GitHub's single-file limit. `task-*-part-*.mp4` are
keyframe-aligned, FFmpeg stream-copy segments, with fast-start metadata and
reset per-part timestamps. They have **not** been re-encoded, downscaled or
upscaled. Concatenated encoded-packet SHA-256 sequences exactly match the source.

`task-a.json` and `task-b.json` retain source-video hashes, original renderer
reports, chapters, part time intervals and hashes. Posters are unscaled first
frames. The website's task player automatically continues between parts and uses
the original complete-video chapter times.

Prepare again only from the matching original files:

```bash
python website/prepare_reference_videos.py --ffmpeg /path/to/ffmpeg
```

See [campaign scope and validation](../../../docs/skylands-campaign-demos.md)
for the distinction between these recordings and newer simulator traces.
