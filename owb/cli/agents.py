"""Policies a manifest can name: random, demo, llm (a harness kind) and oracles.

Oracles are privileged (they read simulator state) and are never benchmark agents;
``owb run`` refuses them unless ``--allow-privileged`` is given. The built-in
scripted references drive their native reference-course envs, not the common-schema
env that ``owb run`` builds, so they are listed here with their own scripts and are
not runnable through a manifest yet. ``register_oracle`` adds one that is: its
``factory(env, manifest)`` receives the live common-schema env and returns a policy.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .manifest import DEFAULT_EXECUTABLES, DEFAULT_MODELS, HARNESS_KINDS, ManifestError


@dataclass(frozen=True)
class OracleSpec:
    name: str
    description: str
    entry: str  # "module:Class" of the controller (not imported by ``agents list``)
    script: str | None = None  # how the oracle is run today, if not through owb run
    factory: Callable | None = None  # factory(env, manifest) -> policy, for common-schema envs
    privileged: bool = True

    @property
    def runnable(self):
        return self.factory is not None


_ORACLES: dict[str, OracleSpec] = {}


def register_oracle(spec: OracleSpec, *, replace: bool = False) -> None:
    if not spec.privileged:
        raise ValueError("oracles are privileged by definition (docs/agent-sandbox.md §4.1)")
    if spec.name in _ORACLES and not replace:
        raise ValueError(f"oracle {spec.name!r} is already registered")
    _ORACLES[spec.name] = spec


def unregister_oracle(name: str) -> None:
    _ORACLES.pop(name, None)


def oracles() -> tuple[OracleSpec, ...]:
    return tuple(_ORACLES[name] for name in sorted(_ORACLES))


def oracle(name: str) -> OracleSpec:
    try:
        return _ORACLES[name]
    except KeyError:
        raise ManifestError(f"agent.policy: unknown oracle {name!r}; available: {sorted(_ORACLES)}") from None


for _spec in (
    OracleSpec("teyvat_reference", "Teyvat reference courses A/B (reads course task state)",
               "owb.sims.teyvat.reference_policy:ReferencePolicy", "python -m owb.scripts.teyvat_reference"),
    OracleSpec("wildlands_reference", "Wildlands reference courses A/B (reads course task state)",
               "owb.bench.baselines.wildlands_reference:WildlandsReferencePolicy",
               "python -m owb.scripts.wildlands_reference"),
    OracleSpec("skylands_reference", "Skylands reference levels (privileged geometry reads)",
               "owb.bench.baselines.skylands_reference:SkylandsReferencePolicy",
               "python -m owb.scripts.skylands_reference"),
    OracleSpec("skylands_campaign", "Full Skylands campaign (privileged routing)",
               "owb.bench.baselines.skylands_campaign:SkylandsCampaignPolicy",
               "python -m owb.scripts.skylands_campaign"),
    OracleSpec("skylands_tempest", "Skylands Tempest legal-action references",
               "owb.bench.baselines.skylands_tempest:SkylandsTempestPolicy",
               "python -m owb.scripts.skylands_tempest"),
):
    register_oracle(_spec)


class PrivilegedPolicy:
    """Forwards to ``policy``; ``metadata()`` always reports ``privileged=True``.

    Used for every privileged run (oracle policy or a privileged observation), so
    policy.json and summary.json["policy"] mark the run. Decisions are unchanged.
    """
    def __init__(self, policy, *, name, reasons):
        self.policy, self.name, self.reasons = policy, name, tuple(reasons)

    def reset(self, seed):
        if hasattr(self.policy, "reset"):
            self.policy.reset(seed)

    def __call__(self, observation):
        return self.policy(observation)

    @property
    def last_decision(self):
        return getattr(self.policy, "last_decision", None)

    @property
    def calls(self):
        return getattr(self.policy, "calls", None)

    def close(self):
        close = getattr(self.policy, "close", None)
        return close() if callable(close) else None

    def metadata(self):
        inner = self.policy.metadata() if hasattr(self.policy, "metadata") else {}
        return dict(inner, type=inner.get("type", self.name), privileged=True,
                    privileged_reasons=list(self.reasons))


def build_harness(harness):
    """Construct the harness for a validated ``HarnessSection``."""
    from ..agents.harness import ClaudeCLIHarness, CodexCLIHarness, GeminiCLIHarness, OpenAIResponsesHarness
    common = dict(model=harness.resolved_model, reasoning_effort=harness.effort, timeout=harness.timeout)
    if harness.kind == "openai_api":
        extra = {k: v for k, v in dict(max_output_tokens=harness.max_output_tokens,
                                       image_detail=harness.image_detail).items() if v is not None}
        return OpenAIResponsesHarness(**common, **extra)
    if harness.kind == "claude_cli":
        return ClaudeCLIHarness(harness.account_home, executable=harness.resolved_executable, **common)
    if harness.kind == "codex_cli":
        return CodexCLIHarness(harness.account_home, executable=harness.resolved_executable, **common)
    extra = {} if harness.max_output_tokens is None else dict(max_output_tokens=harness.max_output_tokens)
    return GeminiCLIHarness(harness.account_home, account_email=harness.account_email,
                            executable=harness.resolved_executable, **common, **extra)


def build_policy(manifest, env):
    """The policy for one run of ``manifest`` on ``env`` (privileged runs are wrapped)."""
    agent = manifest.agent
    if agent.oracle is not None:
        spec = oracle(agent.oracle)
        if not spec.runnable:
            raise ManifestError(f"agent.policy: oracle {spec.name!r} drives its native reference-course env, "
                                f"not the common-schema env built by owb run; use {spec.script}")
        policy = spec.factory(env, manifest)
    elif agent.policy == "random":
        from ..eval.policies import RandomPolicy
        policy = RandomPolicy()
    elif agent.policy == "demo":
        from ..eval.policies import DemoPolicy
        policy = DemoPolicy()
    else:
        from ..agents.policy import Budgets, LLMPolicy, LLMPolicyConfig
        settings = {k: v for k, v in dict(history_steps=agent.history_steps,
                                          memory_chars=agent.memory_chars).items() if v is not None}
        config = LLMPolicyConfig(
            budgets=Budgets(max_calls=manifest.budget.max_model_calls,
                            max_total_tokens=manifest.budget.max_total_tokens),
            decision_seconds=manifest.env.decision_seconds, **settings)
        policy = LLMPolicy(build_harness(agent.harness), config=config)
    if manifest.privileged:
        policy = PrivilegedPolicy(policy, name=agent.oracle or agent.policy, reasons=manifest.privileged_reasons)
    return policy


def harness_rows():
    """(kind, default model, default executable, caps dict) for ``owb agents list``."""
    from .manifest import _harness_caps
    return [(kind, DEFAULT_MODELS[kind], DEFAULT_EXECUTABLES.get(kind), _harness_caps(kind).to_dict())
            for kind in HARNESS_KINDS]


__all__ = ["OracleSpec", "PrivilegedPolicy", "build_harness", "build_policy", "harness_rows", "oracle",
           "oracles", "register_oracle", "unregister_oracle"]
