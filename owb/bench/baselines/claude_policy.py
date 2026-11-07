"""Use a saved Claude Code login for the common RGB/sensor action policy.

Compatibility layer over ``owb.agents.harness.claude_cli.ClaudeCLIHarness``.
"""
from __future__ import annotations

from ...agents.harness.claude_cli import ClaudeCLIHarness, ClaudePolicyError
from ...agents.harness.cli import run_cli as _run_cli
from ._responses_shim import parts_from_request, response_from_decision
from .openai_policy import OpenAIPolicy, OpenAIPolicyConfig

__all__ = ["ClaudePolicy", "ClaudePolicyError", "ClaudeClient"]


def _cli(*args, **kwargs):
    # Look up _run_cli at call time so patching this module's attribute takes effect.
    return _run_cli(*args, **kwargs)


class ClaudeClient:
    """Deprecated Responses-shaped adapter; uses CLI OAuth without extracting account tokens."""
    def __init__(self, account_home, *, executable="claude", timeout=60):
        self.harness = ClaudeCLIHarness(account_home, executable=executable, timeout=timeout, runner=_cli)
        self.account_home, self.executable = self.harness.account_home, self.harness.executable
        self.timeout = timeout
        self.responses = self

    def create(self, **request):
        parts = parts_from_request(request, png_error="Claude policy requires a base64 PNG observation")
        return response_from_decision(self.harness._invoke(
            request["instructions"], parts, request["text"]["format"]["schema"],
            model=request["model"], effort=request["reasoning"]["effort"]))


class ClaudePolicy(OpenAIPolicy):
    """Same sensors, action validation, history and memory as the other policies."""
    def __init__(self, config=None, *, account_home, executable="claude"):
        config = config or OpenAIPolicyConfig(model="claude-opus-5-5")
        harness = ClaudeCLIHarness(account_home, executable=executable, timeout=config.timeout,
                                   model=config.model, reasoning_effort=config.reasoning_effort,
                                   runner=_cli)
        # ``client`` is the harness (exposes account_home/executable like the old ClaudeClient).
        self._init_legacy(config, harness, harness)

    def metadata(self):
        data = super().metadata()
        data.update(type="claude", account_home=str(self.client.account_home),
                    executable=self.client.executable, output_token_limit_enforced=False,
                    max_cli_turns_per_decision=3)
        data["config"].pop("max_output_tokens")
        data["config"].pop("image_detail")
        return data
