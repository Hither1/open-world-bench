"""PRO-LONG baseline: a coding agent with an append-only episode log.

Port of the ARC-AGI-3 research harness from https://github.com/alexisfox7/PRO-LONG
(Fox et al., 2026, arXiv:2607.20064) to the Wildlands/Skylands model-pilot courses.

The method, as in the original harness:

* the runner appends every observation, executed plan entry and the agent's own
  analysis to ``logs.txt`` without summarising it;
* a coding-agent CLI (Claude Code or Codex) with file, search and shell tools reads
  that log programmatically, keeps its own notes in a persistent workspace, and
  writes ``actions.json`` with a batch of up to ``action_cap`` entries;
* one agent session per episode is resumed on every call; context growth is left
  to the CLI's native compaction;
* the runner executes the batch, flushes the remainder when a milestone completes
  (PRO-LONG flushes on score change), logs the outcome and calls the agent again.

Adaptations: an entry is one Wildlands/Skylands primitive action with the course's
``repeat`` count (the shared action interface, minus the ``memory`` field, which the
workspace replaces). Observations are the same allowlisted sensors as the standard
model baseline plus the native RGB frame, saved as a PNG next to the log.

Isolation: PRO-LONG runs each agent in a Docker container. Here every CLI process
runs in a bubblewrap sandbox that exposes only system directories, the CLI and
Python installations, a fresh per-episode CLI config directory, the agent workspace
(rw) and the episode history (read-only). The repository, simulator, evaluator and
the user's CLI profiles (with other sessions' transcripts) are not mounted; login
material enters as an environment token or a mirrored copy of the login file.
Unlike PRO-LONG's allowlist proxy, outbound network access is not restricted.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time
import uuid

CLAUDE_TOOLS = ("Read", "Write", "Edit", "Bash", "Grep", "Glob")
# Codex features that would reach beyond the workspace or add agents/retrieval systems.
CODEX_DISABLED_FEATURES = (
    "multi_agent", "apps", "plugins", "remote_plugin", "browser_use", "computer_use",
    "image_generation", "memories", "skill_search", "hooks",
)
SANDBOX_ROOT = "/workspace"
HISTORY = "history"          # read-only mount inside the workspace
LOG_NAME = "logs.txt"
RULE = "=" * 80

NEUTRAL_ENTRY = dict(move=[0, 0], look=[0, 0], vector=[0, 0, 0], build_move=[0, 0, 0], buttons=[],
                     command="none", target=0, option=0, part=0, slot=0,
                     ingredients=[0, 0, 0, 0, 0], repeat=1)


class ProLongError(RuntimeError):
    """A CLI, sandbox, authentication or model failure; never replaced with scripted actions."""
    def __init__(self, message, code="agent_call_failed"):
        super().__init__(message)
        self.code = code


# ----------------------------------------------------------------------------- sandbox

def _install_root(path):
    """Directory to expose for an npm-installed CLI or interpreter."""
    path = Path(path).resolve()
    parts = path.parts
    if "node_modules" in parts:
        return Path(*parts[:parts.index("node_modules")])
    return path.parent.parent if path.parent.name == "bin" else path.parent


@dataclass
class Sandbox:
    """bubblewrap jail: /workspace rw, /workspace/history ro, no repository or home access."""
    workspace: Path
    history: Path | None
    read_only: tuple = ()
    read_write: tuple = ()
    env: dict = field(default_factory=dict)

    def __post_init__(self):
        if shutil.which("bwrap") is None:
            raise ProLongError("bubblewrap (bwrap) is required for the PRO-LONG sandbox", "sandbox_unavailable")

    def wrap(self, argv):
        """(command, environment). The environment is passed to bwrap, never on its command
        line, so credentials do not appear in the host process table."""
        command = ["bwrap", "--die-with-parent", "--new-session", "--unshare-pid", "--unshare-ipc",
                   "--unshare-uts", "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp",
                   "--dir", "/tmp/home"]
        for system in ("/usr", "/bin", "/sbin", "/lib", "/lib64", "/etc"):
            if os.path.exists(system):
                command += ["--ro-bind", system, system]
        for path in self.read_only:
            command += ["--ro-bind", str(path), str(path)]
        for path in self.read_write:
            command += ["--bind", str(path), str(path)]
        command += ["--bind", str(self.workspace.resolve()), SANDBOX_ROOT]
        if self.history is not None:
            command += ["--ro-bind", str(self.history.resolve()), f"{SANDBOX_ROOT}/{HISTORY}"]
        env = dict(HOME="/tmp/home", TMPDIR="/tmp", LANG="C.UTF-8", TERM="dumb", **self.env)
        for name in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "no_proxy", "NO_PROXY",
                     "SSL_CERT_FILE", "SSL_CERT_DIR", "NODE_EXTRA_CA_CERTS"):
            if name in os.environ:
                env.setdefault(name, os.environ[name])
        return command + ["--chdir", SANDBOX_ROOT, "--", *argv], env


class MirroredCredential:
    """Per-episode copy of a CLI login file, so the sandbox never sees the real profile.

    ``pull`` adopts the profile's file when it changed; ``push`` writes a token refresh made
    inside the sandbox back to the profile, unless the profile changed meanwhile.
    """
    def __init__(self, source, target):
        self.source, self.target, self.snapshot = Path(source), Path(target), None

    def pull(self):
        data = self.source.read_bytes()
        if data != self.snapshot:
            self.target.write_bytes(data)
            self.target.chmod(0o600)
            self.snapshot = data

    def push(self):
        if not self.target.is_file():
            return
        data = self.target.read_bytes()
        if data != self.snapshot and self.source.read_bytes() == self.snapshot:
            temp = self.source.with_name(self.source.name + ".prolong-partial")
            temp.write_bytes(data)
            temp.chmod(0o600)
            temp.replace(self.source)
            self.snapshot = data


def _run(wrapped, prompt, transcript, timeout):
    """Stream stdout to the transcript file and bound the whole process group."""
    command, env = wrapped
    with open(transcript, "w") as out, open(str(transcript) + ".stderr", "w") as err:
        with subprocess.Popen(command, stdin=subprocess.PIPE, stdout=out, stderr=err, text=True,
                              env=env, start_new_session=True) as process:
            try:
                process.communicate(prompt, timeout=timeout)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.communicate()
                raise ProLongError(f"agent call timed out after {timeout:.0f}s", "timeout")
            return process.returncode


def _events(transcript):
    events = []
    for line in Path(transcript).read_text(errors="replace").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict):
            events.append(event)
    return events


@dataclass
class AgentResult:
    text: str
    session_id: str | None
    usage: dict
    tools: list
    duration_seconds: float
    models: list = field(default_factory=list)
    cost_usd: float | None = None


# ----------------------------------------------------------------------------- backends

class ClaudeCodeBackend:
    """``claude -p`` with Read/Write/Edit/Bash/Grep/Glob, one resumed session per episode."""
    name = "claude-code"
    view_hint = "View frames with the Read tool (it displays PNG images)"
    tools_line = ", ".join(CLAUDE_TOOLS)

    def __init__(self, account_home, *, executable="claude", model="claude-opus-5-5", effort="high",
                 timeout=1800.0, max_turns=50):
        self.account_home = Path(account_home).expanduser().resolve()
        token, saved = self.account_home / "oauth_token", self.account_home / ".credentials.json"
        if not (token.is_file() or saved.is_file()):
            raise ValueError(f"No oauth_token or .credentials.json in {self.account_home}")
        # A long-lived `claude setup-token` token, as PRO-LONG uses, is preferred over a login file.
        self.auth = "oauth_token" if token.is_file() else "credentials_json"
        found = shutil.which(str(executable))
        if found is None:
            raise ValueError(f"Claude executable not found: {executable}")
        self.executable = str(Path(found).resolve())
        self.model, self.effort, self.timeout, self.max_turns = model, effort, timeout, max_turns
        self.session_id = None
        self.credential = None

    def mounts(self, private):
        """Read-only installs, writable dirs and environment; ``private`` is a fresh config dir."""
        node = shutil.which("node")
        ro = {_install_root(self.executable)} | ({_install_root(node)} if node else set())
        env = dict(CLAUDE_CONFIG_DIR=str(private), DISABLE_AUTOUPDATER="1",
                   CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC="1")
        if self.auth == "oauth_token":
            env["CLAUDE_CODE_OAUTH_TOKEN"] = (self.account_home / "oauth_token").read_text().strip()
        else:
            self.credential = MirroredCredential(self.account_home / ".credentials.json",
                                                 private / ".credentials.json")
        return tuple(sorted(ro)), (private,), env

    def call(self, sandbox, prompt, system_prompt, transcript):
        if self.credential is not None:
            self.credential.pull()
        try:
            return self._call(sandbox, prompt, system_prompt, transcript)
        finally:
            if self.credential is not None:
                self.credential.push()

    def _call(self, sandbox, prompt, system_prompt, transcript):
        resume = self.session_id is not None
        if not resume:
            self.session_id = str(uuid.uuid4())
        argv = [self.executable, "-p", "--model", self.model, "--effort", self.effort,
                "--safe-mode", "--tools", ",".join(CLAUDE_TOOLS),
                "--strict-mcp-config", "--mcp-config", '{"mcpServers":{}}',
                "--disable-slash-commands", "--no-chrome", "--permission-mode", "bypassPermissions",
                "--max-turns", str(self.max_turns), "--output-format", "stream-json", "--verbose",
                "--append-system-prompt", system_prompt,
                *(["--resume", self.session_id] if resume else ["--session-id", self.session_id])]
        start = time.monotonic()
        code = _run(sandbox.wrap(argv), prompt, transcript, self.timeout)
        events = _events(transcript)
        results = [e for e in events if e.get("type") == "result"]
        result = results[-1] if results else {}
        details = str(result.get("result", "")).lower()
        if code or result.get("is_error") or result.get("subtype") not in ("success", "error_max_turns"):
            if "usage limit" in details or "rate limit" in details:
                raise ProLongError("Claude account usage limit reached", "usage_limit")
            if (result.get("api_error_status") in (401, 403) or "not logged in" in details
                    or "authenticat" in details or "login" in details or "revoked" in details):
                raise ProLongError("Claude account needs authentication: " + details[:120], "authentication_failed")
            if "model" in details and any(s in details for s in ("not found", "not available", "not supported")):
                raise ProLongError("Requested model is unavailable to this Claude account/client", "model_unavailable")
            raise ProLongError(f"Claude Code inference failed (exit {code}, result {result.get('subtype')})")
        models, tools = set(), []
        for event in events:
            if event.get("type") == "system" and event.get("subtype") == "init":
                extra = set(event.get("tools", [])) - set(CLAUDE_TOOLS)
                if extra:
                    raise ProLongError(f"Unexpected tools enabled: {sorted(extra)}", "unexpected_tools")
            if event.get("type") == "assistant":
                message = event.get("message", {})
                models.add(message.get("model"))
                tools += [b.get("name") for b in message.get("content", []) if b.get("type") == "tool_use"]
        models.discard(None)
        if models - {self.model}:
            raise ProLongError(f"Responses came from {sorted(models)}, not {self.model}", "model_mismatch")
        self.session_id = result.get("session_id") or self.session_id
        return AgentResult(text=str(result.get("result") or ""), session_id=self.session_id,
                           usage=dict(result.get("usage") or {}, num_turns=result.get("num_turns"),
                                      subtype=result.get("subtype")),
                           tools=tools, duration_seconds=time.monotonic() - start, models=sorted(models),
                           cost_usd=result.get("total_cost_usd"))


class CodexBackend:
    """``codex exec`` with shell, apply_patch and view_image; ``codex exec resume`` per call."""
    name = "codex"
    view_hint = "View frames with the view_image tool"
    tools_line = "shell, apply_patch, view_image"

    def __init__(self, account_home, *, executable="codex", model="gpt-6.1-sol", effort="high",
                 timeout=1800.0):
        self.account_home = Path(account_home).expanduser().resolve()
        if not (self.account_home / "auth.json").is_file():
            raise ValueError(f"No saved Codex login in {self.account_home}")
        found = shutil.which(str(executable))
        if found is None:
            raise ValueError(f"Codex executable not found: {executable}")
        self.executable = str(Path(found).resolve())
        self.model, self.effort, self.timeout = model, effort, timeout
        self.session_id = None
        self.credential = None
        self.auth = "auth_json"

    def mounts(self, private):
        node = shutil.which("node")
        ro = {_install_root(self.executable)} | ({_install_root(node)} if node else set())
        self.credential = MirroredCredential(self.account_home / "auth.json", private / "auth.json")
        return tuple(sorted(ro)), (private,), dict(CODEX_HOME=str(private))

    def call(self, sandbox, prompt, system_prompt, transcript):
        self.credential.pull()
        try:
            return self._call(sandbox, prompt, system_prompt, transcript)
        finally:
            self.credential.push()

    def _call(self, sandbox, prompt, system_prompt, transcript):
        settings = {"model_reasoning_effort": self.effort, "web_search": "disabled",
                    "project_doc_max_bytes": 0, "forced_login_method": "chatgpt",
                    "approval_policy": "never", "developer_instructions": system_prompt}
        for feature in CODEX_DISABLED_FEATURES:
            settings[f"features.{feature}"] = False
        settings["features.view_image"] = True
        argv = [self.executable, "exec"]
        if self.session_id is not None:
            argv += ["resume", self.session_id]
        argv += ["--ignore-user-config", "--skip-git-repo-check", "--json", "--model", self.model,
                 # bubblewrap is the sandbox; Codex's own one cannot nest inside it.
                 "--dangerously-bypass-approvals-and-sandbox"]
        for key, value in settings.items():
            argv += ["-c", f"{key}={json.dumps(value)}"]
        argv.append("-")
        start = time.monotonic()
        code = _run(sandbox.wrap(argv), prompt, transcript, self.timeout)
        events = _events(transcript)
        failures = [e for e in events if e.get("type") in ("error", "turn.failed")]
        completed = [e for e in events if e.get("type") == "turn.completed"]
        if code or failures or not completed:
            details = " ".join(str(e.get("message", e.get("error", ""))) for e in failures)
            if "usage limit" in details or "quota" in details:
                raise ProLongError("Codex account usage limit reached", "usage_limit")
            if "newer version" in details or "not supported" in details or "not found" in details:
                raise ProLongError("Requested model is unavailable to this Codex account/client", "model_unavailable")
            raise ProLongError(f"Codex inference failed (exit {code})")
        tools, text = [], ""
        for event in events:
            if event.get("type") == "thread.started":
                self.session_id = event.get("thread_id") or self.session_id
            if event.get("type") == "item.completed":
                item = event.get("item", {})
                kind = item.get("type")
                if kind in ("web_search", "mcp_tool_call", "collab_tool_call"):
                    raise ProLongError(f"Unexpected Codex tool activity: {kind}", "unexpected_tools")
                if kind == "agent_message":
                    text = item.get("text", text)
                elif kind not in ("reasoning", "error"):
                    tools.append(kind)
        return AgentResult(text=text, session_id=self.session_id, usage=completed[-1].get("usage", {}),
                           tools=tools, duration_seconds=time.monotonic() - start)


# ----------------------------------------------------------------------------- prompt

HARNESS_PROMPT = """\
You are a coding agent controlling an embodied character in a 3D world by writing action plans.

Mission: {mission}
Your primary objective is to complete the mission's milestones. Your secondary objective is to use few primitive actions.

{history_section}

**Tools**: {tools}. {view_hint}; you can also analyze frames with Python (numpy, PIL are installed).

**Workspace**: `/workspace/` persists across calls. `actions.json` is cleared each call; other files accumulate. Feel free to save notes, state, or helper functions. You have no access to the simulator, its source code or any reference solution; the observations are your only information about the world.

**Response format**: a strategic briefing, then
[PLAN]
<2-3 sentence action plan>

**Write `/workspace/actions.json`** with a JSON object `{{"actions": [ENTRY, ...]}}` — a list of 1–{action_cap} entries to execute in order. Entries beyond {action_cap} are discarded. An ENTRY is one primitive action with these fields (omitted fields take the neutral value shown):
{entry_fields}
Prefer short lists (1–2 entries) when testing a new hypothesis so you see the result before committing further; scale up toward {action_cap} for proven sequences. An invalid entry is reported in the log and it and all later entries are discarded.

The runner executes the entries in order, then calls you again{again}. It stops the batch early, discarding the remaining entries, when a milestone is completed or the episode ends.

**Control interface** (shared with the other model baselines):

{interface}
"""

FULL_HISTORY = """\
`/workspace/history/logs.txt` is the episode log: section headers, your prior analyses, every executed entry, and the observation after it (sensors as one JSON line, plus the path of the RGB frame under `/workspace/history/frames/`). {window} It is read-only. Parse it **programmatically** (grep, Python json), as it grows long and reading it whole is costly and imprecise. Cross-call parsing (positions over time, inspection texts, which commands had which effects) is tractable and can be useful for understanding mechanics, including long-horizon ones.

**Log markers**:
    [OBSERVATION] step=N frame=PATH — followed by one line of sensor JSON; the state after the section's entry
    [ANALYSIS] — your response text from the call that produced the following entries
    [ENTRY] — an executed plan entry, with the number of primitive actions that ran
    [PLAN ERROR] / [PLAN STOPPED] — rejected entries / entries discarded at a milestone or episode end"""

NO_HISTORY = """\
The current observation (sensor JSON and the path of the current RGB frame) is in your user prompt each call. Prior observations and analyses are not preserved across calls unless you save them yourself to the workspace."""


def _interface(guide):
    """The control-interface paragraphs of a standard baseline guide.

    The first paragraph frames the single-action, memory-note protocol and is replaced
    by the harness prompt; so is the final instruction to answer with schema JSON.
    """
    paragraphs = guide.strip().split("\n\n")
    framing, body = paragraphs[0], "\n\n".join(paragraphs[1:])
    if "memory" not in framing or "Return only JSON conforming to the supplied schema." not in body:
        raise ValueError("Unrecognised baseline guide layout; update prolong._interface")
    body = body.replace("Return only JSON conforming to the supplied schema.", "")
    body = body.replace("and remembered notes", "and your own notes and log content")
    body = body.replace("with no new model call", "with no new agent call")
    return body.strip()


def _entry_fields(schema):
    lines = []
    for name, spec in schema["properties"].items():
        if name == "memory":
            continue
        if spec.get("type") == "array" and "minItems" in spec:
            kind = f"array of {spec['minItems']} {spec['items']['type']}s in [{spec['items']['minimum']}, {spec['items']['maximum']}]"
        elif spec.get("type") == "array":
            kind = f"list of {', '.join(spec['items']['enum'])}"
        elif "enum" in spec:
            kind = "one of " + ", ".join(spec["enum"])
        else:
            kind = f"integer {spec['minimum']}..{spec['maximum']}"
        lines.append(f"- `{name}`: {kind} (default {json.dumps(NEUTRAL_ENTRY[name])})")
    return "\n".join(lines)


# ----------------------------------------------------------------------------- policy

class ProLongPolicy:
    """Plan-batch policy: one agent call returns up to ``action_cap`` validated entries.

    ``base`` is a standard course policy (WildlandsModelPolicy or SkylandsModelPolicy)
    used only for its observation allowlist, action schema, decoder and interface text.
    ``log_window``: None = full log (PRO-LONG), N>0 = last N entry sections,
    -1 = no harness log (the paper's no-log baseline; the agent keeps its own notes).
    """

    def __init__(self, base, backend, root, *, mission, action_cap=20, log_window=None, max_calls=200):
        if log_window is not None and (type(log_window) is not int or log_window == 0 or log_window < -1):
            raise ValueError("log_window must be None, -1 or a positive integer")
        if type(action_cap) is not int or action_cap <= 0:
            raise ValueError("action_cap must be a positive integer")
        self.base, self.backend, self.mission = base, backend, mission
        self.action_cap, self.log_window, self.max_calls = action_cap, log_window, max_calls
        self.root = Path(root)
        self.history = self.root / "history"          # harness-owned; mounted read-only
        self.master_log = self.history / LOG_NAME
        self.frames = self.history / "frames"
        self.workspace = self.root / "workspace"       # agent-owned
        self.transcripts = self.root / "transcripts"
        for path in (self.frames, self.workspace, self.transcripts):
            path.mkdir(parents=True, exist_ok=False)
        self.visible = self.root / "visible-history"   # truncated copy for log_window > 0
        if log_window is not None and log_window > 0:
            (self.visible / "frames").mkdir(parents=True)
        (self.workspace / HISTORY).mkdir()             # mount point
        private = self.root / "cli-home"                # fresh per-episode CLI config and sessions
        private.mkdir(mode=0o700)
        ro, rw, env = backend.mounts(private.resolve())
        python_root = Path(sys.prefix).resolve()
        path = [str(python_root / "bin"), "/usr/bin", "/bin"]
        node = shutil.which("node")
        if node:
            path.insert(0, str(Path(node).parent))
        mounted = None if log_window == -1 else (self.visible if log_window else self.history)
        self.sandbox = Sandbox(self.workspace, mounted, read_only=tuple(ro) + (python_root,),
                               read_write=tuple(rw), env=dict(env, PATH=":".join(path)))
        self.system_prompt = self._system_prompt()
        self.calls = 0
        self.input_tokens = self.output_tokens = 0
        self.cost_usd = 0.0
        self.memory = ""          # runner compatibility: the workspace replaces the memory note
        self.current = None       # latest (sensors, frame name) for the no-log prompt
        self.last_request = self.last_response = None
        self._nudge = ""
        self._sections = 0

    @property
    def guide(self):
        return self.system_prompt

    def _system_prompt(self):
        if self.log_window == -1:
            history = NO_HISTORY
        else:
            window = ("It contains the full episode history." if self.log_window is None
                      else f"It contains the initial state and the last {self.log_window} entry sections.")
            history = FULL_HISTORY.format(window=window)
        again = "" if self.log_window == -1 else " with the updated log"
        return HARNESS_PROMPT.format(
            mission=self.mission, history_section=history, tools=self.backend.tools_line,
            view_hint=self.backend.view_hint, action_cap=self.action_cap,
            entry_fields=_entry_fields(self.base.schema()), again=again,
            interface=_interface(self.base.guide))

    # -- log writing (harness side) ------------------------------------------------
    def _append(self, text):
        with open(self.master_log, "a") as stream:
            stream.write(text)

    def observe(self, observation, step, header):
        """Append a section: header, then the observation after it (sensors + frame)."""
        from PIL import Image
        sensors = self.base.describe(observation)
        frame = f"frames/{step:06d}.png"
        path = self.history / frame
        if not path.exists():       # an unchanged step (e.g. rejected batch) keeps its frame
            Image.fromarray(observation["rgb"]).save(path)
        self.current = (sensors, frame)
        self._sections += 1
        self._append(f"\n{RULE}\n{header}\n\n[OBSERVATION] step={step} frame={HISTORY}/{frame}\n"
                     f"{json.dumps(sensors, allow_nan=False, separators=(',', ':'))}\n")

    def note(self, marker, text):
        self._append(f"\n[{marker}]\n{text.strip()}\n")

    def _sync_visible(self):
        """For a positive window: the first section plus the last N sections, frames included."""
        if self.log_window is None or self.log_window == -1:
            return
        sections = re.split(rf"(?=\n{RULE}\n)", self.master_log.read_text())
        kept = sections[:2] + sections[2:][-self.log_window:]
        (self.visible / LOG_NAME).write_text("".join(kept))
        wanted = set(re.findall(r"frame=history/(frames/\d+\.png)", "".join(kept)))
        for old in (self.visible / "frames").iterdir():
            if f"frames/{old.name}" not in wanted:
                old.unlink()
        for name in wanted:
            target = self.visible / name
            if not target.exists():
                shutil.copyfile(self.history / name, target)

    # -- agent call ------------------------------------------------------------------
    def _prompt(self):
        if self.log_window == -1:
            sensors, frame = self.current
            shutil.copyfile(self.history / frame, self.workspace / "current_frame.png")
            head = ("This is the first call. " if self.calls == 1 else
                    "Check /workspace/ for anything you saved previously. ")
            prompt = (f"[CURRENT OBSERVATION] frame=/workspace/current_frame.png\n"
                      f"{json.dumps(sensors, allow_nan=False)}\n\n{head}Analyze the observation "
                      f"and write a new /workspace/actions.json.")
        elif self.calls == 1:
            prompt = (f"Read the episode log at /workspace/history/{LOG_NAME}\n\nThis is the first call. "
                      f"Analyze the initial observation and write /workspace/actions.json with your first entries.")
        else:
            prompt = (f"Read the episode log at /workspace/history/{LOG_NAME}\n\nRecent entries and "
                      f"observations are at the end of the log; what changed since the last call (positions, "
                      f"objective text, inspection results, plan adherence) can be informative. Check "
                      f"/workspace/ for anything you saved previously, then write a new /workspace/actions.json.")
        return prompt + (f"\n\n{self._nudge}" if self._nudge else "")

    def plan(self):
        """One agent call. Returns validated [(action, repeat, entry_json)], possibly empty."""
        if self.calls >= self.max_calls:
            raise RuntimeError("Agent call budget exhausted")
        self.calls += 1
        self._sync_visible()
        actions_file = self.workspace / "actions.json"
        if actions_file.exists():
            actions_file.unlink()
        prompt = self._prompt()
        self.last_request = dict(call=self.calls, prompt=prompt, backend=self.backend.name,
                                 session_id=self.backend.session_id)
        self.last_response = None
        transcript = self.transcripts / f"{self.calls:04d}.jsonl"
        try:
            result = self.backend.call(self.sandbox, prompt, self.system_prompt, transcript)
        except ProLongError as error:
            self.last_response = dict(call=self.calls, status="error", code=error.code, reason=str(error))
            raise
        usage = result.usage
        self.input_tokens += int(usage.get("input_tokens", 0)) + int(usage.get("cache_read_input_tokens", 0)) \
            + int(usage.get("cache_creation_input_tokens", 0))
        self.output_tokens += int(usage.get("output_tokens", 0))
        self.cost_usd += result.cost_usd or 0.0
        raw = actions_file.read_text(errors="replace") if actions_file.is_file() else None
        entries, error = self._parse(raw)
        self.last_response = dict(call=self.calls, status="completed", session_id=result.session_id,
                                  models=result.models, usage=usage, cost_usd=result.cost_usd,
                                  latency_seconds=result.duration_seconds, tools=result.tools,
                                  output_text=result.text, actions_json=raw,
                                  accepted_entries=len(entries), plan_error=error)
        if self.log_window != -1 and result.text.strip():
            self.note("ANALYSIS", f"call {self.calls}\n{result.text}")
        if error:
            self.note("PLAN ERROR", f"call {self.calls}: {error}")
        self._nudge = "" if entries else (
            "Your previous call did not produce a valid /workspace/actions.json"
            + (f" ({error})" if error else "") + '. Please write one with the shape {"actions": [...]}.')
        return entries

    def _parse(self, raw):
        if raw is None:
            return [], "no actions.json was written"
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            return [], f"actions.json is not valid JSON: {exc}"
        items = value.get("actions") if isinstance(value, dict) else value
        if not isinstance(items, list):
            return [], 'expected {"actions": [...]}'
        fields = set(self.base.schema()["properties"]) - {"memory"}
        entries, error = [], None
        if len(items) > self.action_cap:
            error = f"{len(items) - self.action_cap} entries beyond the cap of {self.action_cap} were discarded"
        for index, item in enumerate(items[:self.action_cap]):
            try:
                if not isinstance(item, dict):
                    raise ValueError("entry is not an object")
                unknown = set(item) - fields
                if unknown:
                    raise ValueError(f"unknown fields {sorted(unknown)}")
                payload = {k: NEUTRAL_ENTRY[k] for k in fields} | item | {"memory": ""}
                action, repeat = self.base.decode(payload)
            except ValueError as exc:
                discarded = min(len(items), self.action_cap) - index
                error = f"entry {index + 1}: {exc}; it and {discarded - 1} later entries were discarded"
                break
            entries.append((action, repeat, {k: payload[k] for k in sorted(fields)}))
        return entries, error

    def metadata(self):
        return dict(type="prolong", backend=self.backend.name, model=self.backend.model,
                    effort=self.backend.effort, action_cap=self.action_cap,
                    log_window="full" if self.log_window is None else self.log_window,
                    calls=self.calls, input_tokens=self.input_tokens, output_tokens=self.output_tokens,
                    cost_usd=self.cost_usd, session_id=self.backend.session_id)
