# Embodied vision agents

The `openai` policy connects the shared Teyvat, Wildlands, and Skylands interface
to the OpenAI Responses API. At each decision it sends the current RGB image,
named sensors, supported controls, recent observation/action pairs, and a compact
memory note. The model returns one validated action and a replacement memory note.
The simulator advances, and the policy observes the result before choosing again.

## Use your Google account with Gemini 4 Argon

The `gemini` policy uses the same RGB, sensors, action schema, and bounded memory
through Gemini CLI. Install `@google/gemini-cli`, run `gemini` interactively, and
choose **Sign in with Google** using the intended account. Complete sign-in in
the CLI/browser. A cached Google login is required for headless runs.

```bash
python -B -m owb.scripts.benchmark \
  --env skylands teyvat wildlands --seed 7 --steps 12 --size 320 240 \
  --policy gemini --gemini-account-home ~/.gemini \
  --gemini-account-email you@gmail.com --model gemini-4-argon \
  --reasoning-effort low --max-model-calls 12 --api-timeout 90 --video gif \
  --output runs/gemini4-trial
```

Use `--gemini-executable /path/to/gemini` for a specific installation. The account
directory must be named `.gemini`. The adapter checks its non-secret active-account
metadata against `--gemini-account-email` before and after each decision. The CLI
handles its existing OAuth credentials; credentials are never copied into recordings.
Model access and account limits must be confirmed by an actual inference request.

Each decision runs in a temporary directory containing the observation and local
settings. The CLI expands the PNG attachment before inference. Model tools, MCP,
extensions, skills, and hooks are disabled; the parent account selection is unchanged.
The adapter requests JSON matching the action schema and rejects unexpected tool
activity or model statistics that do not confirm the exact requested model.

Reasoning effort supports `low`, `medium`, or `high`. `--max-output-tokens` is sent
in generation settings, `--api-timeout` bounds each CLI process, and
`--max-model-calls` bounds decisions. CLI retries can make a decision involve more
than one request. This is a Gemini CLI baseline with its runtime instructions;
offline adapter tests alone do not establish account access or live model behavior.

Sources: [Gemini 4 Argon capabilities](https://aistudio.google.com/docs/models/gemini-4-argon),
[Gemini CLI authentication](https://geminicli.com/docs/get-started/authentication/),
[headless output](https://geminicli.com/docs/cli/headless/), and
[configuration](https://geminicli.com/docs/reference/configuration/).

## Use your Claude account with Opus 5.5

The `claude` policy uses the same image, sensors, action schema, recent history,
and memory note through Claude Code. It selects a saved login only for its child
processes; the user's normal account configuration is unchanged.

```bash
python -B -m owb.scripts.benchmark \
  --env skylands teyvat wildlands --seed 7 --steps 12 --size 320 240 \
  --policy claude --claude-account-home ~/.claude --model claude-opus-5-5 \
  --reasoning-effort low --max-model-calls 12 --api-timeout 90 --video gif \
  --output runs/opus55-trial
```

Use `--claude-account-home` to identify the intended saved profile, and
`--claude-executable /path/to/claude` for a specific CLI installation. This Linux
adapter expects `.credentials.json` in the selected directory. The account must
be signed in and entitled to the requested model. Credentials stay in their
profile and are handled by the CLI. API keys, alternate endpoints, and third-party
provider overrides are removed from the child environment.

The adapter sends RGB as an image block over stdin and requests an action through
`--json-schema`. It uses safe/restricted mode with external tools, customizations,
and MCP servers disabled. `StructuredOutput` is the only allowed internal tool.
Every response must identify the exact requested model; fallback to another model
is rejected. An account hold, usage limit, invalid action, or incomplete CLI result
stops the run before another environment step.

Each decision permits at most three CLI turns to produce its structured action.
The process timeout and episode decision cap apply, but `--max-output-tokens` is
only enforced by the `openai` and `gemini` policies. Internal retries and structured-output
turns mean a CLI decision is not necessarily one inference request. Recorded input
token totals include Anthropic's separately reported cache reads and cache writes.
Like the Codex transport, this uses a coding-agent runtime and should be reported
separately from direct API baselines. Matching seeds and twelve short decisions
are an integration check, not a reliable model-quality comparison.

Sources: [Opus 5.5 model and vision support](https://platform.claude.com/docs/en/models/opus-5-5/overview),
[Claude Code CLI options](https://code.claude.com/docs/en/cli-reference).

## Use a saved Codex account

The `codex` policy runs the same observation/action loop through a saved ChatGPT
login in Codex CLI. For the locally configured `roboharness04` account:

```bash
python -B -m owb.scripts.benchmark \
  --env skylands teyvat wildlands --seed 7 --steps 12 --size 320 240 \
  --policy codex --codex-account-home ~/.codex-roboharness04 \
  --model gpt-6-astra --max-model-calls 12 --api-timeout 90 --video gif \
  --output runs/astra-codex-trial
```

This requires Pillow and a recent Codex CLI with access to the requested model.
Use `--codex-executable /path/to/codex` to select a separate installation. If the
service reports that the model requires a newer CLI, upgrade that installation.
The policy never silently switches models.

The subprocess selects only the requested account, ignores the user's Codex
configuration, removes API-key overrides from its environment, and runs each
decision in a temporary directory with the current image and action schema.
Shell, browser, plugins, and delegation are disabled, and unexpected tool events
cause the action to be rejected. Credentials remain in their original profile;
they are not copied into the repository or recordings. Codex may refresh its own
saved login as usual. The parent process's account selection is unchanged.

`--max-model-calls` bounds CLI decisions and `--api-timeout` bounds each process.
The CLI may retry requests internally. **`--max-output-tokens` applies to the
`openai` and `gemini` policies**; Codex exec does not expose that per-response cap. Recorded
Codex metadata marks this explicitly. Account plan limits still apply. This path
uses Codex's model instructions and runtime, so report it as a separate baseline
from direct Responses API runs. See [Codex authentication](https://learn.chatgpt.com/docs/auth)
and [non-interactive structured outputs](https://learn.chatgpt.com/docs/non-interactive-mode).

## Run a trial

```bash
pip install -e ".[agent,video]"
# Set OPENAI_API_KEY securely in your shell before running.
python -B -m owb.scripts.benchmark \
  --env skylands teyvat wildlands --seed 7 --steps 12 --size 320 240 \
  --policy openai --model gpt-6-astra --reasoning-effort low \
  --max-model-calls 12 --max-output-tokens 2048 --video gif \
  --output runs/astra-trial
```

The output directory must be new for each environment/seed pair. Local `runs/`
artifacts are ignored by Git. The SDK reads `OPENAI_API_KEY` and supports its
standard environment configuration, including `OPENAI_BASE_URL`. Access to the
requested model must be available to that API account. Credentials are never
written to recordings. No API calls are made by the random/demo policies or tests.

This trial allows at most 36 requests (12 per environment). `--steps` and
`--max-model-calls` bound the number of decisions; `--max-output-tokens` bounds each
response, including reasoning. There are no automatic request retries. These are
request/token limits, not a dollar budget. Usage is recorded without assuming a
fixed token price. An incomplete response, refusal, invalid action, or API error
stops the run; no scripted policy is substituted. Increase the output token cap
if the model reaches its limit without returning a complete action.

If the API returns `credit_balance_exhausted` or `insufficient_quota`, fund the
API project or configure a funded API key, then rerun with a new output directory.
Model-catalog access alone does not establish that inference credits are available.

Simulation pauses during each API request. By default each action advances only
0.2 simulated seconds, so twelve decisions cover 2.4 seconds even if wall time is
much longer. A short rollout checks the integration, not task-solving ability.
For a longer attempt, set both `--steps 60` and `--max-model-calls 60`. You can
increase `--decision-seconds` to a multiple of the native physics timestep, but
this changes control resolution; Skylands construction still moves only one grid
cell per decision. GIF playback uses simulated time, excluding API latency.

## Recordings

Each `ENV-SEED/` directory contains:

- `episode.gif` or `episode.mp4` when requested, showing the actual policy views.
- `config.json`: environment configuration, seed, and observation/action schema.
- `policy.json`: model, reasoning effort, context and request limits at run start.
- `trajectory.jsonl`: actions, rewards, evaluator events, model response IDs,
  per-request usage/latency, and memory notes.
- `summary.json`: terminal result, metrics, cumulative tokens, and model latency.
- `error.json` on failure, with error type, completed step count, and available
  policy usage. A failed rollout has no successful episode summary.

The recorder may contain privileged diagnostic information for evaluation. The
model receives only the observation passed to its policy, never recorder data,
the simulator handle, native diagnostics, task success flags, or hidden positions.
Goal-direction hints remain disabled by default. Model memory notes are hypotheses,
not evaluator-verified events. The policy retains six recent sensor/action pairs
and at most 1,600 characters of notes by default; older images are not retained.
Memory is reset between episodes, so this is an episode-local baseline, not a
cross-session memory system. A fixed seed reproduces the world, not API outputs.

Skylands has an explicit bridge-and-beacon success predicate. Teyvat measures
quest completion. Wildlands is a survival sandbox with no success predicate, so
its success field is `null`. Time-limit truncation is not success.

## Python interface

```python
from owb import make_env
from owb.bench import run_episode
from owb.bench.baselines import OpenAIPolicy, OpenAIPolicyConfig

policy = OpenAIPolicy(OpenAIPolicyConfig(model="gpt-6-astra", max_calls=30))
with make_env("skylands", max_steps=30, image_size=(320, 240)) as env:
    summary = run_episode(env, policy, seed=7,
                          output_dir="runs/astra-python/skylands-7", video="gif")
```

When changing `decision_seconds` in Python, pass the same value to the environment
and policy configuration. `OpenAIPolicy` also accepts a supplied `client` for
offline testing. API/image packages are optional and loaded when the policy is
constructed.

The implementation uses [image input and structured output supported by
GPT-6-Astra](https://developers.openai.com/api/docs/models/gpt-6-astra) and the
[Responses structured-output format](https://developers.openai.com/api/docs/guides/structured-outputs).

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider \
  tests/test_openai_policy.py tests/test_codex_policy.py tests/test_claude_policy.py \
  tests/test_gemini_policy.py tests/test_framework.py
```
