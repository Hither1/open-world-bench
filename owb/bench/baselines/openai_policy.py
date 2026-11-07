"""Vision policy using the Responses API and only common-schema observations.

Compatibility layer: the implementation is ``owb.agents.LLMPolicy`` with an
``OpenAIResponsesHarness``. Names below keep their pre-refactor import paths.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

from ...agents.decoder import action_schema, decode_action
from ...agents.harness.openai_api import OpenAIResponsesHarness
from ...agents.perception import describe_observation
from ...agents.policy import CONTROL_GUIDE, Budgets, LLMPolicy, LLMPolicyConfig

__all__ = ["OpenAIPolicy", "OpenAIPolicyConfig", "CONTROL_GUIDE",
           "action_schema", "decode_action", "describe_observation"]


@dataclass(frozen=True)
class OpenAIPolicyConfig:
    model: str = "gpt-6-astra"
    reasoning_effort: str = "low"
    max_output_tokens: int = 2048
    max_calls: int = 100
    history_steps: int = 6
    memory_chars: int = 1600
    timeout: float = 60.0
    image_detail: str = "auto"
    decision_seconds: float = 0.2

    def __post_init__(self):
        if not self.model.strip():
            raise ValueError("model must not be empty")
        if self.reasoning_effort not in ("low", "medium", "high", "xhigh", "max"):
            raise ValueError("unsupported reasoning effort")
        if self.image_detail not in ("low", "high", "auto"):
            raise ValueError("image_detail must be low, high, or auto")
        for name in ("max_output_tokens", "max_calls", "memory_chars"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.history_steps) is not int or self.history_steps < 0:
            raise ValueError("history_steps must be a nonnegative integer")
        if any(not math.isfinite(x) or x <= 0 for x in (self.timeout, self.decision_seconds)):
            raise ValueError("timeout and decision_seconds must be positive and finite")


class OpenAIPolicy(LLMPolicy):
    """One model request per decision, bounded local memory, no hidden-state access.

    Imports optional API/image dependencies only when constructed. A supplied client
    allows offline contract testing. API errors/refusals stop the run; no demo fallback.
    """
    def __init__(self, config=None, *, client=None):
        config = config or OpenAIPolicyConfig()
        harness = OpenAIResponsesHarness(
            client, model=config.model, reasoning_effort=config.reasoning_effort,
            max_output_tokens=config.max_output_tokens, image_detail=config.image_detail,
            timeout=config.timeout)
        self._init_legacy(config, harness, harness.client)

    def _init_legacy(self, config, harness, client):
        try:
            from PIL import Image
        except ImportError as exc:
            raise ImportError('OpenAI policy requires pip install -e ".[agent]"') from exc
        self._image = Image  # no longer used (PNG encoding moved to owb.agents.Perception)
        self.config = config  # legacy combined config; limits are copied to policy_config
        self.client = client
        super().__init__(harness, config=LLMPolicyConfig(
            budgets=Budgets(max_calls=config.max_calls), history_steps=config.history_steps,
            memory_chars=config.memory_chars, decision_seconds=config.decision_seconds))

    def metadata(self):
        return dict(type="openai", config=asdict(self.config), calls=self.calls,
                    input_tokens=self.input_tokens, output_tokens=self.output_tokens,
                    latency_seconds=self.latency_seconds)
