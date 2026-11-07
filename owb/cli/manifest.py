"""Run manifest: one YAML/JSON file that fully determines an ``owb run``.

Sections (docs/cli.md lists every field, its default and what enforces it)::

    run:    {name}
    env:    {sim, variant, image_size, decision_seconds, render_quality,
             include_goal_hint, native_config}
    task:   {id, seeds}
    agent:  {policy, harness: {kind, model, effort, ..., HarnessCaps keys}, history_steps, memory_chars}
    budget: {max_steps, max_model_calls, max_total_tokens, max_wall_clock_s}
    record: {output_dir, trajectory, video, video_renderer, video_size, samples, threads, blender}

``Manifest.from_dict`` validates everything that can be checked without building an
environment or a harness (unknown keys, types, ranges, cross-section rules); all
errors are ``ManifestError`` with the dotted key. ``Manifest.to_dict`` returns the
normalised manifest with defaults filled in; ``from_dict(to_dict())`` round-trips.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, fields
import json
import math
from pathlib import Path
import re

POLICIES = ("random", "demo", "llm")
ORACLE_POLICY = re.compile(r"^oracle\(([A-Za-z0-9_.-]+)\)$")
HARNESS_KINDS = ("openai_api", "claude_cli", "codex_cli", "gemini_cli")
CLI_KINDS = ("claude_cli", "codex_cli", "gemini_cli")
EFFORTS = ("low", "medium", "high", "xhigh", "max")
GEMINI_EFFORTS = ("low", "medium", "high")
IMAGE_DETAILS = ("low", "high", "auto")
VARIANTS = ("small", "full")
QUALITIES = ("fast", "presentation")
VIDEOS = (None, "gif", "mp4")
VIDEO_RENDERERS = (None, "cycles", "cpu")
DEFAULT_MODELS = {"openai_api": "gpt-6-astra", "claude_cli": "claude-opus-5-5",
                  "codex_cli": "gpt-6-astra", "gemini_cli": "gemini-4-argon"}
DEFAULT_EXECUTABLES = {"claude_cli": "claude", "codex_cli": "codex", "gemini_cli": "gemini"}
# HarnessCaps fields (docs/agent-sandbox.md §4.1); see HarnessSection.
CAPS_KEYS = ("profile", "isolation", "persistent_session", "workspace", "persist_workspace",
             "tools", "mcp_tools", "network", "max_turns_per_decision")


class ManifestError(ValueError):
    """An invalid manifest; the message starts with the offending dotted key."""


def _fail(where, message):
    raise ManifestError(f"{where}: {message}")


def _is_int(value):
    return type(value) is int


def _positive_int(where, value, *, optional=False):
    if value is None and optional:
        return None
    if not _is_int(value) or value <= 0:
        _fail(where, f"must be a positive integer, got {value!r}")
    return value


def _nonnegative_int(where, value):
    if not _is_int(value) or value < 0:
        _fail(where, f"must be a nonnegative integer, got {value!r}")
    return value


def _positive_number(where, value, *, optional=False):
    if value is None and optional:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        _fail(where, f"must be a positive finite number, got {value!r}")
    return float(value) if isinstance(value, float) else value


def _choice(where, value, allowed):
    if value not in allowed:
        _fail(where, f"must be one of {[a for a in allowed]}, got {value!r}")
    return value


def _string(where, value, *, optional=False):
    if value is None and optional:
        return None
    if not isinstance(value, str) or not value.strip():
        _fail(where, f"must be a non-empty string, got {value!r}")
    return value


def _bool(where, value):
    if not isinstance(value, bool):
        _fail(where, f"must be true or false, got {value!r}")
    return value


def _pair(where, value):
    if (not isinstance(value, (list, tuple)) or len(value) != 2
            or any(not _is_int(n) or n <= 0 for n in value)):
        _fail(where, f"must be [width, height] positive integers, got {value!r}")
    return tuple(value)


def _set(obj, name, value):
    object.__setattr__(obj, name, value)


def _section(cls, data, where):
    """Build section ``cls`` from a mapping, rejecting unknown and missing keys."""
    if data is None:
        data = {}
    if not isinstance(data, Mapping):
        _fail(where, f"must be a mapping, got {type(data).__name__}")
    names = [f.name for f in fields(cls)]
    unknown = sorted(set(data) - set(names))
    if unknown:
        _fail(where, f"unknown key(s) {unknown}; allowed: {names}")
    required = [f.name for f in fields(cls) if f.default is _REQUIRED]
    missing = [name for name in required if name not in data]
    if missing:
        _fail(where, f"missing required key(s) {missing}")
    return cls(**{str(k): v for k, v in data.items()})


class _Required:
    def __repr__(self):
        return "<required>"


_REQUIRED = _Required()


def _plain(value):
    """JSON/YAML-ready copy: tuples become lists, frozensets sorted lists, sections dicts."""
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if isinstance(value, (tuple, list)):
        return [_plain(v) for v in value]
    if isinstance(value, (set, frozenset)):
        return sorted(_plain(v) for v in value)
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    return value


class _Section:
    def to_dict(self):
        return {f.name: _plain(getattr(self, f.name)) for f in fields(self)}


@dataclass(frozen=True)
class RunSection(_Section):
    name: str = _REQUIRED

    def __post_init__(self):
        _string("run.name", self.name)


@dataclass(frozen=True)
class EnvSection(_Section):
    """``sim`` is one environment name or a list; one run per sim and seed.

    ``native_config`` maps a sim name to overrides of the native config that
    ``make_env`` would build (nested dataclass fields as nested mappings).
    """
    sim: tuple[str, ...] = _REQUIRED
    variant: str = "small"
    image_size: tuple[int, int] = (96, 72)
    decision_seconds: float = 0.2
    include_goal_hint: bool = False
    render_quality: str = "fast"
    native_config: dict | None = None

    def __post_init__(self):
        from ..bench.framework import available_environments
        sims = (self.sim,) if isinstance(self.sim, str) else self.sim
        if not isinstance(sims, (list, tuple)) or not sims:
            _fail("env.sim", f"must be an environment name or a non-empty list, got {self.sim!r}")
        for name in sims:
            _choice("env.sim", name, available_environments())
        if len(set(sims)) != len(sims):
            _fail("env.sim", "each environment must appear only once")
        _set(self, "sim", tuple(sims))
        _choice("env.variant", self.variant, VARIANTS)
        _set(self, "image_size", _pair("env.image_size", self.image_size))
        _set(self, "decision_seconds", _positive_number("env.decision_seconds", self.decision_seconds))
        _bool("env.include_goal_hint", self.include_goal_hint)
        _choice("env.render_quality", self.render_quality, QUALITIES)
        if self.native_config is not None:
            if not isinstance(self.native_config, Mapping):
                _fail("env.native_config", "must be a mapping {sim: {field: value}}")
            for name, overrides in self.native_config.items():
                if name not in sims:
                    _fail("env.native_config", f"key {name!r} is not one of env.sim {list(sims)}")
                if not isinstance(overrides, Mapping):
                    _fail(f"env.native_config.{name}", "must be a mapping of native config fields")
            _set(self, "native_config", _plain(self.native_config))


@dataclass(frozen=True)
class TaskSection(_Section):
    """``id`` names a shipped TaskSpec (``owb tasks list``); the env is wrapped in TaskEnv."""
    id: str | None = None
    seeds: tuple[int, ...] = (0,)

    def __post_init__(self):
        _string("task.id", self.id, optional=True)
        seeds = (self.seeds,) if _is_int(self.seeds) else self.seeds
        if not isinstance(seeds, (list, tuple)) or not seeds:
            _fail("task.seeds", f"must be a non-empty list of integers, got {self.seeds!r}")
        for seed in seeds:
            _nonnegative_int("task.seeds", seed)
        if len(set(seeds)) != len(seeds):
            _fail("task.seeds", "each seed must appear only once")
        _set(self, "seeds", tuple(seeds))


def _harness_caps(kind):
    from ..agents.harness import ClaudeCLIHarness, CodexCLIHarness, GeminiCLIHarness, OpenAIResponsesHarness
    return {"openai_api": OpenAIResponsesHarness, "claude_cli": ClaudeCLIHarness,
            "codex_cli": CodexCLIHarness, "gemini_cli": GeminiCLIHarness}[kind].caps


@dataclass(frozen=True)
class HarnessSection(_Section):
    """Model runtime. The HarnessCaps keys (``profile`` ... ``max_turns_per_decision``)
    are declarations: each harness kind has fixed caps today, and a declared value
    must equal the harness's own (None = take the harness's). They are not
    configurable, and nothing beyond the harness's CLI flags enforces them.
    """
    kind: str = _REQUIRED
    model: str | None = None
    effort: str = "low"
    account_home: str | None = None
    account_email: str | None = None
    executable: str | None = None
    max_output_tokens: int | None = None
    image_detail: str | None = None
    timeout: float = 60.0
    profile: str | None = None
    isolation: str | None = None
    persistent_session: bool | None = None
    workspace: str | None = None
    persist_workspace: str | None = None
    tools: tuple[str, ...] | None = None
    mcp_tools: tuple[str, ...] | None = None
    network: str | None = None
    max_turns_per_decision: int | None = None

    def __post_init__(self):
        w = "agent.harness."
        kind = _choice(w + "kind", self.kind, HARNESS_KINDS)
        _string(w + "model", self.model, optional=True)
        _choice(w + "effort", self.effort, GEMINI_EFFORTS if kind == "gemini_cli" else EFFORTS)
        _string(w + "account_home", None if self.account_home is None else str(self.account_home), optional=True)
        if self.account_home is not None:
            _set(self, "account_home", str(self.account_home))
        _string(w + "executable", self.executable, optional=True)
        _string(w + "account_email", self.account_email, optional=True)
        if kind in CLI_KINDS and self.account_home is None:
            _fail(w + "account_home", f"is required for {kind}")
        if kind not in CLI_KINDS:
            for name in ("account_home", "executable"):
                if getattr(self, name) is not None:
                    _fail(w + name, f"is not used by {kind}")
        if kind == "gemini_cli" and self.account_email is None:
            _fail(w + "account_email", "is required for gemini_cli")
        if kind != "gemini_cli" and self.account_email is not None:
            _fail(w + "account_email", "is only used by gemini_cli")
        _positive_int(w + "max_output_tokens", self.max_output_tokens, optional=True)
        if self.max_output_tokens is not None and kind not in ("openai_api", "gemini_cli"):
            _fail(w + "max_output_tokens", f"is not enforced by {kind}; only openai_api and gemini_cli cap output")
        if self.image_detail is not None:
            if kind != "openai_api":
                _fail(w + "image_detail", "is only used by openai_api")
            _choice(w + "image_detail", self.image_detail, IMAGE_DETAILS)
        _set(self, "timeout", _positive_number(w + "timeout", self.timeout))
        for name in ("tools", "mcp_tools"):
            value = getattr(self, name)
            if value is not None:
                if not isinstance(value, (list, tuple, set, frozenset)) or any(
                        not isinstance(t, str) for t in value):
                    _fail(w + name, f"must be a list of tool names, got {value!r}")
                _set(self, name, tuple(sorted(set(value))))
        if self.workspace is not None:
            _set(self, "workspace", str(self.workspace))
        self._check_caps()

    def _check_caps(self):
        from ..agents.harness.base import ISOLATIONS, NETWORK, PROFILES, WORKSPACE_PERSISTENCE
        w = "agent.harness."
        for name, allowed in (("profile", PROFILES), ("isolation", ISOLATIONS),
                              ("persist_workspace", WORKSPACE_PERSISTENCE), ("network", NETWORK)):
            if getattr(self, name) is not None:
                _choice(w + name, getattr(self, name), allowed)
        if self.persistent_session is not None:
            _bool(w + "persistent_session", self.persistent_session)
        _positive_int(w + "max_turns_per_decision", self.max_turns_per_decision, optional=True)
        actual = self.caps()
        for name in CAPS_KEYS:
            declared = getattr(self, name)
            if declared is not None and _plain(declared) != actual[name]:
                _fail(w + name, f"{self.kind} has fixed caps {name}={actual[name]!r}; "
                                f"configurable caps are not implemented yet (declared {_plain(declared)!r})")

    def caps(self) -> dict:
        """The chosen harness kind's HarnessCaps as a dict (what the runtime actually uses)."""
        return _harness_caps(self.kind).to_dict()

    @property
    def resolved_model(self):
        return self.model or DEFAULT_MODELS[self.kind]

    @property
    def resolved_executable(self):
        return self.executable or DEFAULT_EXECUTABLES.get(self.kind)


@dataclass(frozen=True)
class AgentSection(_Section):
    """``policy``: random | demo | llm | oracle(<name>). ``harness``, ``history_steps``
    and ``memory_chars`` apply to ``llm`` only (None = LLMPolicyConfig default)."""
    policy: str = _REQUIRED
    harness: HarnessSection | None = None
    history_steps: int | None = None
    memory_chars: int | None = None

    def __post_init__(self):
        if not isinstance(self.policy, str) or (
                self.policy not in POLICIES and not ORACLE_POLICY.match(self.policy)):
            _fail("agent.policy", f"must be one of {list(POLICIES)} or oracle(<name>), got {self.policy!r}")
        if self.harness is not None and not isinstance(self.harness, HarnessSection):
            _set(self, "harness", _section(HarnessSection, self.harness, "agent.harness"))
        if self.policy == "llm":
            if self.harness is None:
                _fail("agent.harness", "is required for policy llm")
        else:
            for name in ("harness", "history_steps", "memory_chars"):
                if getattr(self, name) is not None:
                    _fail(f"agent.{name}", f"only applies to policy llm, not {self.policy}")
        if self.history_steps is not None:
            _nonnegative_int("agent.history_steps", self.history_steps)
        _positive_int("agent.memory_chars", self.memory_chars, optional=True)

    @property
    def oracle(self) -> str | None:
        match = ORACLE_POLICY.match(self.policy)
        return match.group(1) if match else None


@dataclass(frozen=True)
class BudgetSection(_Section):
    """Per-episode limits. ``max_steps`` is the env time limit (required unless the
    task spec declares ``budget.steps``). ``max_model_calls`` and ``max_wall_clock_s``
    end the episode with ``termination_reason="budget"`` (runner); ``max_total_tokens``
    is the LLM policy's token budget (raises when exhausted, recorded in error.json).
    """
    max_steps: int | None = None
    max_model_calls: int | None = None
    max_total_tokens: int | None = None
    max_wall_clock_s: float | None = None

    def __post_init__(self):
        for name in ("max_steps", "max_model_calls", "max_total_tokens"):
            _positive_int(f"budget.{name}", getattr(self, name), optional=True)
        _positive_number("budget.max_wall_clock_s", self.max_wall_clock_s, optional=True)


@dataclass(frozen=True)
class RecordSection(_Section):
    """``output_dir`` defaults to ``runs/<run.name>``; each run writes ``<sim>-<seed>/``.

    ``trajectory: false`` records nothing (no run directory, no replay). ``video``
    needs the run directory. ``video_renderer: null`` keeps the VideoSink default
    (Blender Cycles for Teyvat/Wildlands, the native observation for other worlds);
    ``cpu`` is an explicit, labelled lightweight render. There is no fallback.
    """
    output_dir: str | None = None
    trajectory: bool = True
    video: str | None = None
    video_renderer: str | None = None
    video_size: tuple[int, int] = (960, 540)
    samples: int = 48
    threads: int = 8
    blender: str | None = None

    def __post_init__(self):
        if self.output_dir is not None:
            _string("record.output_dir", str(self.output_dir))
            _set(self, "output_dir", str(self.output_dir))
        _bool("record.trajectory", self.trajectory)
        _choice("record.video", self.video, VIDEOS)
        if self.video is not None and not self.trajectory:
            _fail("record.video", "needs record.trajectory: true (the video is written to the run directory)")
        _choice("record.video_renderer", self.video_renderer, VIDEO_RENDERERS)
        _set(self, "video_size", _pair("record.video_size", self.video_size))
        _positive_int("record.samples", self.samples)
        _positive_int("record.threads", self.threads)
        _string("record.blender", self.blender, optional=True)


SECTIONS = {"run": RunSection, "env": EnvSection, "task": TaskSection, "agent": AgentSection,
            "budget": BudgetSection, "record": RecordSection}


@dataclass(frozen=True)
class Manifest:
    run: RunSection
    env: EnvSection
    agent: AgentSection
    task: TaskSection = field(default_factory=TaskSection)
    budget: BudgetSection = field(default_factory=BudgetSection)
    record: RecordSection = field(default_factory=RecordSection)

    def __post_init__(self):
        spec = self.task_spec()
        if spec is not None:
            for name in self.env.sim:
                if name != spec.sim:
                    _fail("task.id", f"task {spec.id!r} is for sim {spec.sim!r}, but env.sim includes {name!r}")
        if self.max_steps is None:
            _fail("budget.max_steps", "is required (the task spec declares no budget.steps)")
        if self.agent.policy == "llm" and self.budget.max_model_calls is None:
            _fail("budget.max_model_calls", "is required for policy llm (each call is a paid request)")
        if self.budget.max_total_tokens is not None and self.agent.policy != "llm":
            _fail("budget.max_total_tokens", "only applies to policy llm")

    # -- construction -------------------------------------------------------
    @classmethod
    def from_dict(cls, data) -> "Manifest":
        if not isinstance(data, Mapping):
            raise ManifestError(f"manifest must be a mapping, got {type(data).__name__}")
        unknown = sorted(set(data) - set(SECTIONS))
        if unknown:
            raise ManifestError(f"unknown section(s) {unknown}; allowed: {list(SECTIONS)}")
        for name in ("run", "env", "agent"):
            if name not in data:
                raise ManifestError(f"{name}: missing required section")
        run = data["run"]
        if isinstance(run, str):  # ``run: <name>`` shorthand (docs/architecture-proposal.md §2.6)
            run = dict(name=run)
        return cls(**{name: _section(section, run if name == "run" else data.get(name), name)
                      for name, section in SECTIONS.items() if name in data})

    @classmethod
    def load(cls, path, overrides=()) -> "Manifest":
        """Read a ``.yaml``/``.yml``/``.json`` manifest and apply ``key=value`` overrides."""
        return cls.from_dict(apply_overrides(read(path), overrides))

    def to_dict(self) -> dict:
        return {name: getattr(self, name).to_dict() for name in SECTIONS}

    # -- derived values -----------------------------------------------------
    def task_spec(self):
        """The TaskSpec named by ``task.id`` (None without a task)."""
        if self.task.id is None:
            return None
        from ..tasks.registry import load_task
        try:
            return load_task(self.task.id)
        except KeyError as exc:
            raise ManifestError(f"task.id: {exc.args[0]}") from None

    @property
    def max_steps(self) -> int | None:
        if self.budget.max_steps is not None:
            return self.budget.max_steps
        spec = self.task_spec()
        return spec.budget.steps if spec is not None else None

    @property
    def output_dir(self) -> Path:
        return Path(self.record.output_dir or Path("runs") / self.run.name).expanduser()

    def run_dir(self, sim, seed) -> Path:
        return self.output_dir / f"{sim}-{seed}"

    @property
    def privileged_reasons(self) -> tuple[str, ...]:
        """Why a run would not be a benchmark-agent result (empty for normal runs)."""
        reasons = []
        if self.agent.oracle is not None:
            reasons.append(f"oracle policy {self.agent.oracle!r}")
        if self.env.include_goal_hint:
            reasons.append("env.include_goal_hint exposes the oracle objective direction")
        return tuple(reasons)

    @property
    def privileged(self) -> bool:
        return bool(self.privileged_reasons)


def read(path) -> dict:
    path = Path(path)
    text = path.read_text()
    if path.suffix == ".json":
        data = json.loads(text)
    elif path.suffix in (".yaml", ".yml"):
        import yaml
        data = yaml.safe_load(text)
    else:
        raise ManifestError(f"{path}: manifest must be .yaml, .yml or .json")
    if not isinstance(data, Mapping):
        raise ManifestError(f"{path}: manifest must be a mapping")
    return data


def _parse_value(text):
    import yaml
    try:
        return yaml.safe_load(text) if text.strip() else None
    except yaml.YAMLError as exc:
        raise ManifestError(f"--set value {text!r} is not valid YAML: {exc}") from None


def apply_overrides(data, overrides) -> dict:
    """Copy of ``data`` with each ``"section.key[.sub]=value"`` applied (value parsed as YAML)."""
    data = json.loads(json.dumps(data))  # deep copy of plain data
    for item in overrides:
        key, sep, value = item.partition("=")
        parts = key.strip().split(".")
        if not sep or not all(parts):
            raise ManifestError(f"--set expects section.key=value, got {item!r}")
        if parts[0] not in SECTIONS:
            raise ManifestError(f"--set {key}: unknown section {parts[0]!r}; allowed: {list(SECTIONS)}")
        node = data
        for part in parts[:-1]:
            if isinstance(node.get(part), str) and part == "run" and node is data:
                node[part] = dict(name=node[part])
            if node.get(part) is None:
                node[part] = {}
            if not isinstance(node[part], dict):
                raise ManifestError(f"--set {key}: {part!r} is not a mapping")
            node = node[part]
        node[parts[-1]] = _parse_value(value)
    return data


__all__ = ["AgentSection", "BudgetSection", "EnvSection", "HarnessSection", "Manifest", "ManifestError",
           "RecordSection", "RunSection", "TaskSection", "apply_overrides", "read"]
