# Teyvat campaign validation

Suite: `teyvat-campaign-v1`. These are privileged reference-controller feasibility runs, not model or memory-performance results. Every action uses the native control interface; the controller is checked for simulator-state mutation.

| Task | Scale | Seed / branch order | Milestones | Native actions | Simulated minutes | Travel (m) | Success / exact replay |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| summit | Full (2) | 7 / forward | 9 | 2,529 | 8.43 | 2513.1 | [Pass / pass](../assets/evaluation/teyvat-campaign-v1/full-seed7/summit-seed-7-forward/verification.json) |
| cleansing | Full (2) | 7 / forward | 17 | 3,246 | 10.82 | 3326.5 | [Pass / pass](../assets/evaluation/teyvat-campaign-v1/full-seed7/cleansing-seed-7-forward/verification.json) |
| desert | Full (2) | 7 / forward | 15 | 5,052 | 16.84 | 5339.8 | [Pass / pass](../assets/evaluation/teyvat-campaign-v1/full-seed7/desert-seed-7-forward/verification.json) |
| desert | Compact (0.5) | 7 / forward | 15 | 1,602 | 5.34 | 1614.2 | [Pass / pass](../assets/evaluation/teyvat-campaign-v1/compact-seed7/desert-seed-7-forward/verification.json) |
| summit | Compact (0.5) | 19 / reverse | 9 | 999 | 3.33 | 796.3 | [Pass / pass](../assets/evaluation/teyvat-campaign-v1/alternate-seed19/summit-seed-19-reverse/verification.json) |
| cleansing | Compact (0.5) | 19 / reverse | 17 | 1,499 | 5.00 | 1419.1 | [Pass / pass](../assets/evaluation/teyvat-campaign-v1/alternate-seed19/cleansing-seed-19-reverse/verification.json) |
| desert | Compact (0.5) | 19 / reverse | 15 | 1,633 | 5.44 | 1643.3 | [Pass / pass](../assets/evaluation/teyvat-campaign-v1/alternate-seed19/desert-seed-19-reverse/verification.json) |

All **16,560 actions** were independently replayed in fresh environments. The decision interval is 0.2 simulated seconds. Run lengths describe this privileged controller and are not difficulty calibration or expected model performance.

Checks include exact before/after positions, event streams, reward, terminal flags, state digests, initial/final geometry hashes, dependency order, final predicates, and resource accounting. State digests include character/camera state, party health and continuous cooldowns, elemental state, entities, campaign mechanisms and inventory, RNG states, and the completion ledger. Source hashes and Python/NumPy versions are stored with every episode.

Each run awards exactly +1 on its final action and zero on all earlier actions. Mountain traces contain twelve separately charged ice hits. Cleansing traces open exactly two boss exposure windows. Desert traces record three locked-archive revisits and two recharges. Forward and reversed branch orders both reach success.

## Tests

- **19 focused campaign tests passed:** observation/privacy contracts, deterministic seeds, native swing cooldown, quartz consumption and expiry, cold/warmth, shard prerequisites, unique wards, wrong-pattern recovery, remote-cast rejection, unbreakable puzzle seals, two boss phases, clearance/charge gates, final-door geometry, walkable final terrain, Gym/shared adapters, snapshot immutability, reward, truncation, and state-digest coverage.
- **200 tests passed in the broader regression run:** campaigns, legacy Teyvat reference/model behavior, architecture, shared framework, combat, scene export, asset diversity, presentation, and landscape rendering. This run overlaps the focused suite; the counts are not a unique combined total.
- The public benchmark CLI completed a two-action random-policy smoke run using `--env teyvat --task summit`; it correctly truncated with zero reward and no success.
- `git diff --check` passed.

[Campaign test log](../assets/evaluation/teyvat-campaign-v1/campaign-tests.log) · [Regression log](../assets/evaluation/teyvat-campaign-v1/regression-tests.log) · [Machine-readable aggregate](../assets/evaluation/teyvat-campaign-v1/validation.json)

## Artifacts and reproduction

Each linked episode directory contains `manifest.json`, `actions.jsonl`, `summary.json`, and `verification.json`. Manifests contain evaluator-only goals, milestone graphs, source hashes, seed, scale, and controller settings. Action traces contain native controls and simulator evidence. Keep these privileged artifacts outside model context.

```bash
python -m owb.scripts.teyvat_campaign --verify \
  assets/evaluation/teyvat-campaign-v1/full-seed7/desert-seed-7-forward
```

Replay requires matching source files. These results cover the listed seeds and scales, not arbitrary layouts or every policy. The corrected terrain and collision definitions are hashed and used in both collection and replay; no render-only terrain changes are substituted. The old 12/24-step course recordings are separate legacy artifacts.

Validation deliberately uses inexpensive headless zero-RGB observations. No screenshots or videos were generated for these results. User-facing presentation continues to use the actual scene through Blender Cycles, with renderer/settings recorded when rendered.

See [task design and API](teyvat-long-horizon-tasks.md) for the three dependency graphs, local observation contract, and proposed memory-ablation protocol.
