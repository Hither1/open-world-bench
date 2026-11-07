# `owb` command line

One entry point, driven by a run manifest. `pip install -e .` installs the `owb`
console script; `python -m owb.cli` is the same program.

```bash
owb run configs/examples/random-smoke.yaml               # 3 envs x 20 random steps
owb run configs/examples/demo-skylands.yaml --seed 7 11  # one run per seed
owb run configs/examples/random-smoke.yaml --set 'env.image_size=[32,24]' --set record.output_dir=/tmp/smoke
owb replay /tmp/smoke/teyvat-7 --verify
owb tasks list && owb tasks show ender_dragon
owb agents list
owb envs list
owb render runs/demo-presentation/teyvat-7
```

`python -m owb.scripts.benchmark ...` still works: it prints a deprecation notice,
translates its flags into the equivalent manifest (`manifest_from_args`, table
below) and runs it through the same code path.

## Subcommands

| Command | What it does |
| --- | --- |
| `owb run MANIFEST [--seed N ...] [--set KEY=VALUE ...] [--allow-privileged]` | Validates the manifest, checks that no run directory exists yet, then runs one episode per `env.sim` x seed with `owb.eval.run_episode`. Prints one summary line per episode and a final table. `--seed` replaces `task.seeds`; `--set section.key[.sub]=value` overrides any key (the value is parsed as YAML: `--set 'env.sim=[teyvat, skylands]'`). Exit status 2 for an invalid manifest or refused run, 1 for a model/harness failure (message only; details in `error.json`). |
| `owb replay RUN_DIR` | Prints the run's environment, seed, configuration, `summary.json`/`error.json`/`task.json` and whether it was privileged. |
| `owb replay RUN_DIR --verify [--fields proprio reward events info]` | `owb.eval.verify_run`: rebuilds the env from `config.json`, re-steps the recorded actions and compares each transition (default: proprio, reward, events). Prints the report; exit status 1 on divergence. |
| `owb tasks list [--sim SIM]` / `owb tasks show ID` | Shipped `TaskSpec` files (`owb/tasks/specs/`); `env` says whether the spec's sim has a runnable env. `show` prints the spec as YAML. |
| `owb agents list` | Policies, harness kinds with their fixed `HarnessCaps`, the sandbox profiles (which are implemented), and the oracles with their `privileged` flag. |
| `owb envs list` | Registered environments. |
| `owb render RUN_DIR [render_blender options]` | `owb.scripts.render_blender` does not read run directories yet: it exports and Cycles-renders the reset state of a fresh env. This prints the matching command for the run's env, seed and variant (extra options are appended). For per-decision frames, re-run the manifest with `record.video`. Skylands has no Cycles exporter; the command says so instead of substituting another renderer. |

## Run directory

`<record.output_dir>/<sim>-<seed>/`, the same layout as `benchmark.py`. The
recorder files are unchanged: `config.json`, `trajectory.jsonl`, `summary.json`,
`policy.json` (policies with `metadata()`), `error.json` (on failure), optional
`episode.gif`/`episode.mp4`. `owb run` adds:

- `manifest.json`: `{"schema_version", "manifest": Manifest.to_dict(), "environment", "seed", "privileged", "privileged_reasons", "argv"}`. `manifest` is the full manifest after overrides, with defaults filled in, so `owb run` on it reproduces the run.
- `task.json` (with `task.id`): the evaluator summary (`task`, milestones achieved, progress, success). Per-step `info["task"]` is in `trajectory.jsonl`.

## Manifest schema

YAML (`.yaml`/`.yml`) or JSON. Unknown sections or keys are errors. Every error
names the dotted key. "Enforced by" says what actually applies the value.

### `run`

| Key | Default | Enforced by |
| --- | --- | --- |
| `name` | required | Default output directory `runs/<name>`. `run: <name>` is shorthand for `run: {name: <name>}`. |

### `env`

| Key | Default | Enforced by |
| --- | --- | --- |
| `sim` | required | One name or a list (`owb envs list`); one run per sim. `make_env`. |
| `variant` | `small` | `BenchmarkConfig.variant` (`small`/`full`). |
| `image_size` | `[96, 72]` | `BenchmarkConfig.image_size` (policy RGB width, height). |
| `decision_seconds` | `0.2` | `BenchmarkConfig`; must be a multiple of the native timestep (checked by `make_env`). Also `LLMPolicyConfig.decision_seconds`. |
| `render_quality` | `fast` | `BenchmarkConfig.render_quality` (`fast`/`presentation`); policy observations only. |
| `include_goal_hint` | `false` | `BenchmarkConfig`. `true` exposes the oracle objective direction, so the run is **privileged** (needs `--allow-privileged`). |
| `native_config` | `null` | `{sim: {field: value, nested: {field: value}}}`: overrides applied to the native config `make_env` would build (fields checked against the native dataclasses), then passed as `native_config=`. `make_env` still applies its own image size, seed, step limit and frame skip. Recorded in `config.json["native"]`, so replay needs nothing extra. |

### `task`

| Key | Default | Enforced by |
| --- | --- | --- |
| `id` | `null` | A shipped spec id (`owb tasks list`). Its `sim` must match every `env.sim`. The env is wrapped in `owb.tasks.env.TaskEnv` (evaluator, `info["task"]`, optional instruction override). Its `budget.steps` is the default for `budget.max_steps`; its other budget fields are not applied yet. Spec `setup` hooks are not run by `TaskEnv` yet. Today all shipped specs are Minecraft, which has no env, so `task.id` is validated but not yet runnable with a shipped spec. |
| `seeds` | `[0]` | One run per seed (`env.reset(seed=...)`); nonnegative, unique. `--seed` replaces it. |

### `agent`

| Key | Default | Enforced by |
| --- | --- | --- |
| `policy` | required | `random` (`owb.eval.RandomPolicy`), `demo` (`DemoPolicy`), `llm` (`owb.agents.LLMPolicy` with `harness`), `oracle(<name>)` (privileged, see below). |
| `history_steps` | `6` | `LLMPolicyConfig.history_steps` (`llm` only). |
| `memory_chars` | `1600` | `LLMPolicyConfig.memory_chars` / `ActionDecoder` (`llm` only). |
| `harness` | required for `llm` | See next table. Not allowed for other policies. |

### `agent.harness`

| Key | Default | Enforced by |
| --- | --- | --- |
| `kind` | required | `openai_api` (`OpenAIResponsesHarness`), `claude_cli` (`ClaudeCLIHarness`), `codex_cli` (`CodexCLIHarness`), `gemini_cli` (`GeminiCLIHarness`). |
| `model` | `gpt-6-astra` (openai, codex), `claude-opus-5-5`, `gemini-4-argon` | Harness; the CLI harnesses verify the responding model id. |
| `effort` | `low` | Harness reasoning effort (`low`..`max`; Gemini: `low`/`medium`/`high`). |
| `account_home` | required for CLI kinds | Harness (Claude: `.credentials.json`; Codex: `auth.json`; Gemini: a `.gemini` directory). Not used by `openai_api`. |
| `account_email` | required for `gemini_cli` | `GeminiCLIHarness` checks the active Google account. Gemini only. |
| `executable` | `claude` / `codex` / `gemini` | Harness (`shutil.which`). CLI kinds only. |
| `max_output_tokens` | harness default (2048) | Per-response cap; `openai_api` and `gemini_cli` only (rejected for Claude/Codex, which cannot enforce it). |
| `image_detail` | harness default (`auto`) | `openai_api` only. |
| `timeout` | `60` | Seconds per decision (API client timeout or CLI process-tree kill). |
| `profile`, `isolation`, `persistent_session`, `workspace`, `persist_workspace`, `tools`, `mcp_tools`, `network`, `max_turns_per_decision` | `null` (= the harness's own) | `HarnessCaps` fields ([agent-sandbox.md](agent-sandbox.md) §4.1-4.2). Every harness kind has **fixed** caps today (`owb agents list`); a declared value must equal them or validation fails. They document and pin the run's caps; they are not configurable. Beyond the CLI flags each harness already passes (for example Claude `--tools ""`, Codex read-only sandbox), nothing enforces them at the OS level: `network` and `persist_workspace` in particular are declarations. |

### `budget`

| Key | Default | Enforced by |
| --- | --- | --- |
| `max_steps` | required unless the task declares `budget.steps` | `BenchmarkConfig.max_steps` (env truncation, `termination_reason="time_limit"`). |
| `max_model_calls` | `null`; required for `llm` | `owb.eval.Budget` in `run_episode`: the episode ends with `termination_reason="budget"`, `budget_hit="max_model_calls"` (counts `policy.calls`, else policy invocations). Also `LLMPolicy` `Budgets.max_calls`. |
| `max_total_tokens` | `null` | `LLMPolicy` `Budgets.max_total_tokens` (`llm` only): checked before each call; exhaustion raises, the run is recorded with `error.json`. The last call may overshoot. |
| `max_wall_clock_s` | `null` | `owb.eval.Budget` (from the start of `run_episode`, reset included); ends the episode with `termination_reason="budget"`. |

When any runner budget is set, `summary.json` also carries `budget` and `budget_hit`.

### `record`

| Key | Default | Enforced by |
| --- | --- | --- |
| `output_dir` | `runs/<run.name>` | Run directories `<output_dir>/<sim>-<seed>`; all are checked before the first episode and never overwritten. |
| `trajectory` | `true` | `false` records nothing (no run directory; `replay` impossible). |
| `video` | `null` | `null`, `gif` or `mp4` (`owb.eval.VideoSink`); needs `trajectory: true`. |
| `video_renderer` | `null` | `null` keeps the `VideoSink` default: Blender Cycles for Teyvat and Wildlands, the native observation renderer elsewhere. `cycles` or `cpu`; `cpu` is an explicit lightweight render. There is no fallback when Blender fails. |
| `video_size` | `[960, 540]` | Presentation resolution. |
| `samples` | `48` | Cycles samples per pixel. |
| `threads` | `8` | Cycles CPU threads. |
| `blender` | `null` | Blender executable (else `OWB_BLENDER`, `PATH`, `.owb-render.local.json`). |

The renderer and its settings are recorded in `config.json["presentation"]`.

## Privileged runs

A run is privileged when `agent.policy` is `oracle(<name>)` or
`env.include_goal_hint` is true. `owb run` refuses it unless `--allow-privileged`
is passed; then the policy is wrapped so that `policy.json` and
`summary.json["policy"]` report `privileged: true` (with `privileged_reasons`),
`manifest.json` records `privileged: true`, and the summary line and table show
it. Decisions are unchanged.

The built-in oracles (`owb agents list`: Teyvat/Wildlands reference courses,
Skylands reference/campaign/Tempest) drive their native reference-course
environments, not the common-schema env `owb run` builds, so `owb run` refers you
to their scripts. `owb.cli.agents.register_oracle(OracleSpec(..., factory=f))`
adds a runnable one: `f(env, manifest)` gets the live common-schema env and returns
a policy.

## Examples

`configs/examples/`: `random-smoke` (3 envs, 20 random steps), `demo-skylands`,
`demo-presentation` and `demo-shared` (README / framework demos; Cycles videos for
Teyvat/Wildlands), `claude-cli-teyvat` (`claude_cli`, profile `cli-stateless`),
`openai-wildlands` and `openai-trial` (`openai_api`, profile `api-stateless`).
`tests/test_cli.py` loads and validates every file in that directory.

## Migrating from `python -m owb.scripts.benchmark`

| benchmark.py flag | Manifest key |
| --- | --- |
| `--env A B` | `env.sim: [A, B]` (default there: all envs; in a manifest: required) |
| `--seed N` | `task.seeds: [N]` (or `owb run --seed N`) |
| `--steps N` | `budget.max_steps: N` |
| `--size W H` | `env.image_size: [W, H]` (benchmark default 160x120; manifest default 96x72) |
| `--variant` | `env.variant` |
| `--quality` | `env.render_quality` |
| `--decision-seconds` | `env.decision_seconds` |
| `--policy random` / `demo` | `agent.policy: random` / `demo` |
| `--policy openai` | `agent.policy: llm`, `agent.harness.kind: openai_api` |
| `--policy codex` / `claude` / `gemini` | `agent.policy: llm`, `agent.harness.kind: codex_cli` / `claude_cli` / `gemini_cli` |
| `--codex-account-home`, `--claude-account-home`, `--gemini-account-home` | `agent.harness.account_home` |
| `--codex-executable`, `--claude-executable`, `--gemini-executable` | `agent.harness.executable` |
| `--gemini-account-email` | `agent.harness.account_email` |
| `--model` | `agent.harness.model` |
| `--reasoning-effort` | `agent.harness.effort` |
| `--max-output-tokens` | `agent.harness.max_output_tokens` (openai/gemini only; benchmark.py accepted and ignored it for Claude/Codex) |
| `--max-model-calls` | `budget.max_model_calls` |
| `--history-steps` | `agent.history_steps` |
| `--api-timeout` | `agent.harness.timeout` |
| `--video` | `record.video` |
| `--video-renderer` | `record.video_renderer` |
| `--blender` | `record.blender` |
| `--samples` / `--threads` | `record.samples` / `record.threads` |
| `--video-size W H` | `record.video_size: [W, H]` |
| `--output DIR` | `record.output_dir: DIR` |

Differences when benchmark.py runs through the manifest: for model policies,
`max_model_calls` is now also a runner budget, so `summary.json` gains `budget`
and `budget_hit` (`null` while `--steps <= --max-model-calls`, which benchmark.py
still requires), and `policy.json`/`summary.json["policy"]` use the `LLMPolicy`
metadata (`type: llm`, `harness`, `caps`, `privileged`) instead of the legacy
`OpenAIPolicy`/`ClaudePolicy`/... layout. Trajectories are unchanged. Random and
demo recordings are byte-identical to the previous script; every run also gets
`manifest.json`.
