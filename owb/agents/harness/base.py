"""Harness contract: the runtime that turns model input parts into one structured decision.

The harness is the research variable (API call, sandboxed coding-agent CLI, ...).
Policies own memory, history and budgets; a harness owns only transport/runtime.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path


PROFILES = ("api-stateless", "cli-stateless", "cli-session", "cli-workspace", "open")
ISOLATIONS = ("process", "sandbox-exec", "docker")
WORKSPACE_PERSISTENCE = ("none", "episode", "run", "suite")
NETWORK = ("none", "model-endpoint", "allowlist")


@dataclass(frozen=True, kw_only=True)
class HarnessCaps:
    """What the model may use during one decision (not what the transport itself uses).

    ``network="model-endpoint"`` means the runtime reaches only its model API; the
    model has no browsing/fetch tool. ``privileged`` must stay False for benchmark agents.
    """
    profile: str
    isolation: str = "process"
    persistent_session: bool
    workspace: Path | None
    persist_workspace: str = "none"
    tools: frozenset[str]
    mcp_tools: frozenset[str] = field(default_factory=frozenset)
    network: str = "model-endpoint"
    max_turns_per_decision: int
    privileged: bool = False

    def __post_init__(self):
        for name, allowed in (("profile", PROFILES), ("isolation", ISOLATIONS),
                              ("persist_workspace", WORKSPACE_PERSISTENCE), ("network", NETWORK)):
            if getattr(self, name) not in allowed:
                raise ValueError(f"{name} must be one of {allowed}")
        if type(self.max_turns_per_decision) is not int or self.max_turns_per_decision <= 0:
            raise ValueError("max_turns_per_decision must be a positive integer")

    def to_dict(self):
        return dict(profile=self.profile, isolation=self.isolation,
                    persistent_session=self.persistent_session,
                    workspace=None if self.workspace is None else str(self.workspace),
                    persist_workspace=self.persist_workspace, tools=sorted(self.tools),
                    mcp_tools=sorted(self.mcp_tools), network=self.network,
                    max_turns_per_decision=self.max_turns_per_decision, privileged=self.privileged)


@dataclass(frozen=True)
class Decision:
    """One harness result.

    ``payload`` is the structured action object when the runtime returns one;
    otherwise ``text`` holds the raw model output and the policy parses it, so
    malformed JSON is recorded as an invalid action rather than a transport error.
    ``usage`` is the runtime's raw usage record; token totals are normalised.
    """
    payload: dict | None
    model: str
    usage: dict | None
    latency_seconds: float
    raw_id: str | None
    status: str
    text: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0


class Harness(ABC):
    caps: HarnessCaps  # set by each concrete harness

    def open(self, episode_seed: int) -> None:
        """Start an episode. Stateless harnesses only record the seed."""
        self.episode_seed = int(episode_seed)

    @abstractmethod
    def decide(self, system: str, parts: list, schema: dict) -> Decision:
        """Return one decision for the given system prompt, input parts and JSON schema."""

    def close(self) -> dict:
        """End the episode; returns harness-side statistics (none for stateless harnesses)."""
        return {}

    def metadata(self) -> dict:
        return dict(name=type(self).__name__)
