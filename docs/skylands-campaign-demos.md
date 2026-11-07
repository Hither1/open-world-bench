# Skylands full-scale campaign demonstrations

Status: both full-scale campaign solutions, independent replays, and GPU Cycles
videos are complete. Every encoded video frame passed a playback decode check.

Open the [milestone chapter player](../assets/demos/skylands-campaign/index.html),
or download the individual walkthroughs:

| Walkthrough | Video duration | Full simulated journey |
| --- | ---: | ---: |
| [A — The Fallen Observatory: 15 milestones](../videos/skylands-15-milestones.mp4) | 7 min 59.3 sec | 57 min 54.4 sec |
| [B — The Three-Layer Lifeline: 24 milestones](../videos/skylands-24-milestones.mp4) | 15 min 14.5 sec | 2 hr 4 min 24.2 sec |

These original Tears of the Kingdom-inspired environments implement full regional
journeys for the [15- and 24-milestone task designs](skylands-long-horizon-tasks.md).
The reference controller changes the simulator only through its action interface.
It reads privileged state for planning; these runs establish executable solutions,
not learned-agent skill, retention, or memory results.

| Campaign | Completed milestones | Actions | Simulated duration | World extent (X × Z) |
| --- | ---: | ---: | ---: | ---: |
| A — The Fallen Observatory | 15 / 15 | 17,372 | 57 min 54.4 sec | 960 × 1,056 m |
| B — The Three-Layer Lifeline | 24 / 24 | 37,321 | 124 min 24.2 sec | 1,280 × 1,408 m |

Both seed-7 runs finish with full health and no deaths. Task B exceeds its
provisional 60–120 minute pilot target by 4.4 minutes; its measured duration is
reported without trimming the campaign or adding idle time. The reference
itineraries are valid, not optimized shortest paths.

## Ground-truth evidence

- Task A: [complete action log](../assets/demos/skylands-campaign/task-a-seed-7/actions.jsonl),
  [run manifest](../assets/demos/skylands-campaign/task-a-seed-7/reference-manifest.json),
  [independent replay verification](../assets/demos/skylands-campaign/task-a-seed-7/verification.json),
  [per-frame provenance and timeline](../assets/demos/skylands-campaign/task-a-seed-7/frame-records.json).
- Task B: [complete action log](../assets/demos/skylands-campaign/task-b-seed-7/actions.jsonl),
  [run manifest](../assets/demos/skylands-campaign/task-b-seed-7/reference-manifest.json),
  [independent replay verification](../assets/demos/skylands-campaign/task-b-seed-7/verification.json),
  [per-frame provenance and timeline](../assets/demos/skylands-campaign/task-b-seed-7/frame-records.json).

The [playback verification](../assets/demos/skylands-campaign/playback-verification.json)
records all 4,793 decoded frames for A and 9,145 for B. The videos contain 557
and 1,170 distinct rendered replay states, respectively, held for the annotated
timelapse intervals. Per-frame records map each video interval to its source
steps, simulation digest, scene hash, and rendered-image hash.

Verification starts from a fresh reset and compares every action's before/after
position, events, reward, milestone-state digest, and final world-state digest.
Final success checks the original cargo's identity, position, installation, and
intact state, plus the required live station outputs. The videos are exported by
another checked replay. Scene export asserts that presentation leaves the
simulation digest unchanged. A further checked replay reviews the traveler’s
projected position and camera rays against collision geometry. It corrects
37 of A’s 557 views and 29 of B’s 1,170 views while proving that mesh arrays,
lighting, and simulation state are unchanged. See the [A camera audit](../assets/demos/skylands-campaign/task-a-seed-7/camera-verification.json)
and [B camera audit](../assets/demos/skylands-campaign/task-b-seed-7/camera-verification.json).

A uses a selected persistent return raft, a vault entrance, a flooded depths
route, a physical freight lift, a long surface return journey, and a separate
sky hoist. B stores the lamp kit and lock coupling in different regional caches,
restores three stations, encounters a physical bridge closure, and delivers to
one of two distinct commissioned refuge docks. Both use native character
movement, Ascend, and glider descents. There is no universal maintenance lift in
the full campaigns.

## Recording contract

The approved format retains the entire campaign timeline with **labeled travel
timelapse**: walking at 16×, craft travel at 8×, slower mechanism sequences, and
short pauses for readable annotations. Frames sample actual replay states; the
source-step interval for every rendered frame is recorded. Every intermediate
action remains in the JSONL log. This is not a continuous real-time video.

Rendering uses **Blender 4.5.4 LTS Cycles, OptiX GPU, 1,280 × 720, 64 samples**
on an NVIDIA RTX PRO 6000 Blackwell Server Edition, with denoising, six light
bounces, shadows, water, and full botanical geometry. The final 1,600 × 900,
10 fps H.264 frame adds captions and a route map around the native scene;
it does not upscale it. The renderer refuses a CPU fallback. Exact settings and
source hashes are in the [A video manifest](../assets/demos/skylands-campaign/task-a-seed-7/video-manifest.json)
and [B video manifest](../assets/demos/skylands-campaign/task-b-seed-7/video-manifest.json).

Static terrain uses exact voxel surfaces with coplanar faces merged losslessly.
All regions, trees, and high-tier ground cover remain in the scene. Grass is
batched into fixed meshes without dropping any blade or changing its triangles;
prototype triangulation is checked before batching.
Static geometry is reused between frames; characters, cargo, craft, gates,
platforms, lighting, and camera follow the replay. Visual detail uses stable
world identity and does not consume the simulation RNG. Renderer/device metadata,
source hashes, per-frame state and image hashes, and final video decode checks
accompany the recordings.

## Scope and remaining research work

These are full-scale authored simulator campaigns. Cargo craft have four
compatible mounts, rated flotation, charge, and steering through physical
waterways. They are not general welded rigid-body vehicles or a fluid simulation.
Heavy cargo has a grounded roller-cradle position ledger and parent identity.
Lifts and platforms use collision-backed grid motion; reversal has a bounded
recent trajectory and cannot rewind inventories or the whole world.

The broader design still calls for richer teaching demonstrations, carrier
capacity pressure, more varied held-out geometry and landmark correspondences,
broader construction alternatives, cargo-recovery coverage, flexible branch-order
validation, and paired memory experiments. The full reference itinerary covers
all milestones; general alternative-solution scoring and a model-agent interface
remain separate benchmark work. Standard clothing and a paraglider are provided
at reset. The original introductory `skylands` environment remains unchanged.

The [compact mechanics testbed](skylands-reference-demos.md) is retained for
regression checks. Its short routes and unfinished compact recordings are
separate from these full-scale demonstrations.

## Reproduce

Use fresh output directories. Collection and replay are headless; presentation
is a separate GPU job.

```bash
python -B -m owb.scripts.skylands_campaign collect --task a --directory /tmp/skylands-full-a
python -B -m owb.scripts.skylands_campaign collect --task b --directory /tmp/skylands-full-b
python -B -m owb.scripts.skylands_campaign verify --task a --directory /tmp/skylands-full-a
python -B -m owb.scripts.skylands_campaign verify --task b --directory /tmp/skylands-full-b
python -B -m owb.scripts.skylands_campaign_video export --directory /tmp/skylands-full-a --output /tmp/skylands-scene-a
```

The repository's GPU job performs export, persistent Cycles rendering, and video
assembly for both verified seed-7 runs:

```bash
sbatch owb/scripts/render_skylands_campaign.slurm
# After both recordings finish:
python -B -m owb.scripts.skylands_campaign_index
```

For existing seed-7 exports in `.cache/skylands-campaign/video-a` and
`video-b`, the parallel workflow reviews cameras and renders six disjoint
ranges. The camera job skips execution when no corrections are needed:

```bash
python -B -m owb.scripts.skylands_campaign_camera --task a
python -B -m owb.scripts.skylands_campaign_camera --task b
sbatch owb/scripts/render_skylands_campaign_chunks.slurm
sbatch owb/scripts/render_skylands_campaign_camera.slurm
# On this project’s allocated RTX partition:
# sbatch --account=kempner_sham_lab --partition=kempner_rtx --constraint=rtx6000pro owb/scripts/render_skylands_campaign_chunks.slurm
# After all main and corrected-camera chunks finish:
python -B -m owb.scripts.skylands_campaign_merge
python -B -m owb.scripts.skylands_campaign_frame_records
```

Chunk merging validates every state and image hash before encoding. The player
builder fully decodes both videos, verifies frame counts and hashes, and builds
the local milestone player. Cached renders are reusable only when source,
trace, state, settings, and image hashes match.
