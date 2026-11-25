# Skylands compact mechanics testbed

These are executable **compact reference courses** corresponding to the
[15- and 24-milestone designs](skylands-long-horizon-tasks.md), driven by a
privileged scripted controller. They are not the full long-duration campaigns
or results from a learned agent.

Status (2026-11-24): the traces linked below predate the
[working-memory probes](working-memory-probes.md) `WM-SA12`, `WM-SB10` and
`WM-SB17` and no longer replay on the current simulator. Verified traces with the
probes are in
[`assets/evaluation/working-memory/skylands/reference/`](../assets/evaluation/working-memory/skylands/reference/).

These compact traces are retained as mechanics tests. Their video rendering was
stopped when the requested scope changed to full-scale campaigns. There is no
completed compact walkthrough MP4 or chapter player in this package.

- [Task A action trace](../assets/demos/skylands-reference/task-a-seed-7/actions.jsonl)
  and [replay verification](../assets/demos/skylands-reference/task-a-seed-7/verification.json).
- [Task B action trace](../assets/demos/skylands-reference/task-b-seed-7/actions.jsonl)
  and [replay verification](../assets/demos/skylands-reference/task-b-seed-7/verification.json).

## What is verified

The controller uses the environment's action interface for walking, manipulation,
ascent, cargo loading, driving, lift operation, caching, and platform reversal.
It reads privileged state to choose actions but cannot set milestone completion
or teleport between sites. Inter-layer shortcuts use a moving maintenance lift.

The seed-7 solutions execute **2,001 actions / 400.2 simulated seconds** for A
and **3,127 actions / 625.4 simulated seconds** for B. Each trace is independently
replayed from reset. Verification compares every agent position, event, and
reward, milestone state digests, and the final world-state digest. Success also
requires live cargo and station predicates: moving the delivered core or relief
shipment away makes the final predicate false.

The reference courses also pass full legal-action rollouts on seed 23. Targeted
checks cover remote manipulation, incompatible mounts, overload, missing power,
reversed propulsion, incomplete boarding, descending lift collision, obstructed
reversal, and state-preserving scene export. This establishes executable examples
for these seeds, not a general solvability claim for arbitrary future worlds.

## Differences from the full designs

- Geography and travel are compact. A and B run for about 6.7 and 10.4 simulated
  minutes, not the proposed 30–60 and 60–120 minute campaign budgets.
- Craft use four discrete mounts, load-rated flotation, energy, and straight
  authored channels. They are not general welded rigid-body vehicles or a fluid
  solver. Cargo is tracked by physical position and parent identity, with grounded
  roller-cradle towing; a general contact solver is not implemented.
- Platforms and freight lifts are authored, grid-stepped collision decks. Reversal
  uses bounded recent platform history; it cannot restore inventory or milestones.
- Task A has one fixed return mooring. It does not establish the design's
  agent-chosen-dock memory probe. Task B places both reserves in one selected
  surface cache and retrieves the lamp kit and coupling at different times.
- Training uses inspectable instructions and executed practice, without an animated
  teacher. Standard cold-weather clothing is provided at reset. The oracle's
  success is not evidence of acquisition, retention, transfer, or memory benefit.
- Cross-layer clue variation, broad alternative constructions, independent regional
  cache networks, and the paired memory-ablation experiments remain future work.

The original introductory `skylands` environment is unchanged. These courses
live in `SkylandsReferenceEnv`; their additional actions are not yet part of the
shared model-agent interface.

## Reproduce

Use a fresh output directory for collection. Tests and trace collection are
headless; presentation uses the GPU renderer explicitly.

```bash
python -B -m owb.scripts.skylands_reference collect --task a --root /tmp/skylands-reference
python -B -m owb.scripts.skylands_reference collect --task b --root /tmp/skylands-reference
python -B -m owb.scripts.skylands_reference verify --task a --root /tmp/skylands-reference
python -B -m owb.scripts.skylands_reference verify --task b --root /tmp/skylands-reference
python -B -m owb.scripts.skylands_reference render --task a --root /tmp/skylands-reference --device CUDA
python -B -m owb.scripts.skylands_reference assemble --task a --root /tmp/skylands-reference
```

Repeat render/assemble for task B. On the configured cluster, after collecting
both seed-7 traces at the default repository location:

```bash
sbatch owb/scripts/render_skylands.slurm
# After both jobs finish:
python -B -m owb.scripts.skylands_demo_index
```

The job array requests one GPU per task and refuses a CPU fallback. Existing
verified frame files can be resumed only when their source/settings contract,
simulation digest, and image hash match.
