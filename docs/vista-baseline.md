# VISTA harness baseline

[VISTA](https://github.com/joshhhhhan/VISTA) ([arXiv 2610.02200](https://arxiv.org/abs/2610.02200))
is a visual harness. It runs Claude Code or Codex CLI as the player inside a sandboxed
container, and keeps every returned frame retrievable. The player acts through
benchmark tools and also gets:

- `inspect` and `read_pixels`, to revisit and zoom into earlier frames;
- `history`, to review prior actions and results;
- `GUIDE.md` and `WORKING.md`, for model-authored notes.

OWB uses VISTA unmodified, vendored at commit `30542e9`, and adds only a task adapter:

- [`owb/bench/baselines/vista_course.py`](../owb/bench/baselines/vista_course.py): a VISTA
  `TaskSession` over any model `Course` (Wildlands and Skylands, observation-only or the
  labelled privileged condition).
- [`owb/scripts/vista_model.py`](../owb/scripts/vista_model.py): the runner.
- [`tests/test_vista_course.py`](../tests/test_vista_course.py): drives the real VISTA
  runtime with a scripted backend. It makes no model calls and needs no container.

## Protocol

The protocol is the same as [Wildlands model pilots](wildlands-model-pilots.md) except
where stated.

- **Shared with the pilots**
  - Mission text and the action interface. The guide's interface paragraphs are kept
    verbatim, and the run fails if that text drifts.
  - The allowlisted sensors (`describe`), the 320 × 180 native policy renderer, and the
    repeat rule: 1..30, only for `none`/`drag` (Skylands: `none`).
  - The budgets: 200 decisions, 6000 primitive actions and one hour of wall time.
  - Milestone interruption, and the host-side `actions.jsonl` log. `verify_run` replays
    the log independently.
- **Single action tool.** The player acts only through `play`, whose arguments are the
  pilot action schema without `memory`. One `play` call is one decision. World time is
  paused between calls.
- **Memory.** The pilot's 6000-character memory field and six-step text history are
  replaced by VISTA's own memory: GUIDE/WORKING notes, plus full retrievable visual
  history. This is the variable that differs from the pilots, and the point of the
  baseline.
- **Invalid actions.** An invalid `play`, such as a repeated `inspect`, spends one
  decision and moves nothing. The pilots instead end the run on an invalid reply.
- **What the player cannot see.** It never receives evaluator events, milestone
  predicates, fixture coordinates or the simulator. One frame is returned per decision.
  Repeated actions do not stream intermediate frames.
- **Run boundary.** The player may call `play` many times in one CLI turn. If a turn
  ends without an action, it is redirected; redirections are bounded by the same
  decision count.
- **Rendering.** Observations come from the native policy renderer, not Cycles. These
  are policy observations, not presentation renders.

## Setup

The runtimes are pinned by VISTA and installed under the gitignored `.cache/`, so the
user's own `claude` and `codex` are untouched.

```bash
git clone https://github.com/joshhhhhan/VISTA.git .cache/vista
git -C .cache/vista checkout 30542e943d21922ad73f91e55888067dbec765bd
# Images are node-local (rootless Podman stores them in /tmp); build on the run node.
(cd .cache/vista && TMPDIR=/tmp docker build -t arc3-codex-player:0.1 -f Dockerfile.codex-player .)
(cd .cache/vista && TMPDIR=/tmp docker build -t arc3-claude-player:0.1 -f Dockerfile.claude-player .)
# Codex CLI 0.145.0
npm install --prefix .cache/vista-runtime/codex-0.145.0 --omit=dev --no-audit --no-fund @openai/codex@0.145.0
# Claude Code 2.1.220: download the linux-x64 binary from
# https://downloads.claude.ai/claude-code-releases/2.1.220/ and check its sha256
# against manifest.json (674f61f2…c863), as .cache/vista-runtime/claude-2.1.220
```

VISTA's other Python dependencies (jsonschema, Pillow, requests, python-dotenv) are
already in the project env. The runner puts `.cache/vista/src` on `sys.path` instead
of installing VISTA, whose pins include numpy 2.5.1.

## Run

Codex:

```bash
python -u -B -m owb.scripts.vista_model --world wildlands --task a --seed 7 \
  --backend codex --auth-file ~/.codex-roboharness04/auth.json \
  --binary .cache/vista-runtime/codex-0.145.0/node_modules/@openai/codex-linux-x64/vendor/x86_64-unknown-linux-musl/bin/codex \
  --model gpt-5.6-sol --effort medium \
  --max-calls 200 --max-actions 6000 --wall-seconds 3600 --task-timeout 3600 \
  --output runs/vista-wildlands-pilot
```

`--binary` must point to the *native* Codex binary. VISTA mounts its vendor directory,
including `codex-resources/bwrap`, into the container.

Claude: `--backend claude --binary .cache/vista-runtime/claude-2.1.220 --model opus`.
This needs `CLAUDE_CODE_OAUTH_TOKEN`, created interactively with
`.cache/vista-runtime/claude-2.1.220 setup-token`.

Other options:

- `--world skylands` runs the Skylands campaign courses.
- `--privileged` runs the labelled upper-bound condition.
- `--tool-calls` bounds every tool, including `inspect` and `history`.

Each task folder contains the same files as the pilots: `config.json`, `actions.jsonl`,
`summary.json`, `verification.json` and `observations/`. VISTA's own artifacts are in
`vista/artifacts` and `vista/transport`: the evidence store, events, GUIDE and the CLI
transcripts.

## Cluster adaptations

`docker` on the FASRC login nodes is rootless Podman 4.9. The runner records these
changes in `config.json` under `transport_provenance.container_adaptation`. All other
VISTA sandbox flags are unchanged (read-only root, `--cap-drop ALL`,
`no-new-privileges`, pid/memory/cpu limits).

- **Player user.** VISTA's `--user <uid>:<gid>` cannot write the run directory, so it
  becomes `--user 0:0`. Container root is the invoking user under rootless Podman.
- **Group access.** `--group-add keep-groups` is added, because `sham_lab` lab storage
  is reachable only through the group.
- **Temp directory.** `TMPDIR=/tmp` keeps VISTA's AF_UNIX socket paths under the
  108-byte limit.
- **Image preflight.** It reads `{{.Id}}`, because Podman rejects the Docker template
  `{{json .Id}}`.

## Status (2026-11-22)

| Run | Result |
| --- | --- |
| `runs/vista-smoke-20261122-gpt61-unsupported` | `gpt-6.1-sol` rejected by Codex 0.145.0 with a ChatGPT login (`model_not_found`). The existing pilots use 0.166.0. |
| `runs/vista-smoke-20261122-gpt56` | `gpt-5.6-sol`, medium: 3 decisions, 25 primitive actions, stopped at the decision limit, replay verified. A transport smoke test, not a score. |

Open items:

- No full-budget run yet.
- Matching the `gpt-6.1-sol` pilots means either moving VISTA's Codex pin, which needs
  a protocol check against app-server changes, or comparing at `gpt-5.6-sol`.
- The Claude backend is untested until a setup token is available.
- Teyvat is not wired up, because it does not use the `Course` abstraction.
