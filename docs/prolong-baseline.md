# PRO-LONG baseline

[PRO-LONG](https://github.com/alexisfox7/PRO-LONG) (Fox et al., 2026,
[arXiv:2607.20064](https://arxiv.org/abs/2607.20064)) is a coding-agent harness for
long-horizon tasks. The runner appends every observation, action and outcome to a
log and never summarises it. A coding-agent CLI searches the log with its file,
grep and shell tools. This page describes the port of the ARC-AGI-3 research
harness to the Wildlands and Skylands milestone courses. The port lives in
[`owb/bench/baselines/prolong.py`](../owb/bench/baselines/prolong.py) and
[`owb/scripts/prolong_model.py`](../owb/scripts/prolong_model.py).

Status: the integration and its offline contract tests are in place
(`tests/test_prolong.py`). No scored live episode has run yet. A live smoke test
ran the full sandbox and CLI path, but its token was revoked (HTTP 401).

```bash
python -u -B -m owb.scripts.prolong_model \
  --world wildlands --task a --seed 7 \
  --backend claude-code --account-home PROFILE_DIR \
  --model claude-opus-5-5 --effort high \
  --max-calls 200 --action-cap 20 --max-actions 6000 \
  --wall-seconds 14400 --timeout 1800 \
  --output runs/prolong-wildlands-YYYYMMDD
```

`--backend codex --account-home ~/.codex-roboharness04 --executable
.cache/codex-sol/node_modules/.bin/codex --model gpt-6.1-sol` selects Codex. That
matches the existing Sol pilots' model and account. The Codex path has not been
exercised live.

## What carries over from PRO-LONG

| PRO-LONG (ARC-AGI-3) | This port |
|---|---|
| `logs.txt`: board after every action, the agent's analyses | `history/logs.txt`: allowlisted sensors as one JSON line and a PNG frame after every plan entry; the agent's response text per call |
| Agent writes `actions.json` with up to 20 actions | Same file and default cap. An entry is one course primitive action with its `repeat` count, and omitted fields are neutral |
| ~30-line prompt, Read/Write/Edit/Bash/Grep/Glob, no subagents | The same prompt structure plus the shared control-interface text from the standard baseline guide. Claude Code: exactly those six tools. Codex: shell, apply_patch, view_image; web search, MCP, apps, plugins and multi-agent are disabled |
| One resumed CLI session per game; native compaction | One resumed session per episode (`--resume` / `codex exec resume`) |
| Queue flushed on score change | Remaining entries discarded on milestone completion, episode end or action budget, and noted as `[PLAN STOPPED]` |
| Invalid `actions.json` → re-prompt with a nudge | Same. An invalid entry and all later entries are rejected, and the reason is logged as `[PLAN ERROR]`. Every call, valid or not, counts against `--max-calls` |
| `--log-window -1` no-log baseline, `--log-window N` truncation | Same flags. No-log puts the current sensors in the prompt and the frame at `/workspace/current_frame.png` |
| Docker container + allowlist egress proxy | bubblewrap sandbox (no Docker daemon on the cluster). **Egress is not restricted** |

The `memory` field of the standard baseline is removed. Workspace files and the log
replace it.

## Information boundary

The log and prompt contain only the observation allowlist used by the standard
model baseline (`describe()` of the course policy) and the native 320 × 180 policy
frame. They also contain the agent's own plan entries and text. Evaluator events,
milestone IDs and state digests stay in the runner's `actions.jsonl`, which the
agent cannot read. A test checks this.

Each CLI process runs under `bwrap` with these mounts:

- `/usr`, `/etc` and the system library directories, read-only;
- the CLI install and the Python prefix (numpy and PIL for frame analysis), read-only;
- `/workspace`, read-write: the agent's notes and `actions.json`;
- `/workspace/history`, read-only: the log and frames;
- a fresh per-episode CLI config directory, read-write.

The repository, the simulator and the user's home are not visible. This includes
other CLI sessions' transcripts. Claude auth comes from a `claude setup-token`
`oauth_token` file in the profile, passed through the environment and not on the
command line. Without that file, the profile's `.credentials.json` is mirrored.
Codex mirrors `auth.json`. A mirrored file that the CLI refreshes is written back
to the profile, unless the profile changed meanwhile.

## Outputs

Each `task-*-seed-*` folder holds the standard pilot files: `config.json`,
`status.json`, `summary.json`, `actions.jsonl`, `verification.json`, and
`control-guide.txt` (the exact system prompt). It also holds:

- `requests/NNNN.json`: the user prompt of each agent call;
- `responses/NNNN.json`: response text, the raw `actions.json`, accepted entry
  count, plan error, tools used, usage, cost (Claude) and latency;
- `prolong/history/`: the log and frames exactly as the agent saw them;
- `prolong/workspace/`: the agent's final workspace;
- `prolong/transcripts/NNNN.jsonl`: the raw CLI event stream of each call, plus stderr.

Replay verification is the same as for the other pilots. Rendering after each
entry does not alter the simulation, and the replay digest check confirms this.

## Comparability caveats

- One agent call returns up to `--action-cap` entries. Compare runs at equal primitive-action and
  wall-clock budgets, not at equal call counts. Report agent calls and tokens alongside milestones.
- A PRO-LONG call takes minutes, so the default wall budget is four hours. The single-action
  baseline's default is one hour.
- `--resume` of an interrupted episode and the privileged-state condition are not implemented for
  this baseline yet.
- Renderer: native policy observations only. No Cycles presentation is produced by this runner.
