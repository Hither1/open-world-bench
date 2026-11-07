"""Gemini CLI Google-login transport for the common vision/action policy.

Compatibility layer over ``owb.agents.harness.gemini_cli.GeminiCLIHarness``.
"""
from __future__ import annotations

from ...agents.harness.cli import run_cli as _run_cli
from ...agents.harness.gemini_cli import GeminiCLIHarness, GeminiPolicyError
from ._responses_shim import parts_from_request, response_from_decision
from .openai_policy import OpenAIPolicy, OpenAIPolicyConfig

__all__ = ["GeminiPolicy", "GeminiPolicyError", "GeminiClient"]


def _cli(*args, **kwargs):
    # Look up _run_cli at call time so patching this module's attribute takes effect.
    return _run_cli(*args, **kwargs)


class GeminiClient:
    """Deprecated Responses-shaped adapter over GeminiCLIHarness."""
    def __init__(self, account_home, *, account_email, executable="gemini", timeout=60):
        self.harness = GeminiCLIHarness(account_home, account_email=account_email,
                                        executable=executable, timeout=timeout, runner=_cli)
        self.account_home, self.account_email = self.harness.account_home, self.harness.account_email
        self.executable = self.harness.executable
        self.timeout = timeout
        self.responses = self

    def check_account(self):
        self.harness.check_account()

    def create(self, **request):
        parts = parts_from_request(request, png_error="Gemini policy requires a base64 PNG observation")
        return response_from_decision(self.harness._invoke(
            request["instructions"], parts, request["text"]["format"]["schema"],
            model=request["model"], effort=request["reasoning"]["effort"],
            max_output_tokens=request["max_output_tokens"]))


class GeminiPolicy(OpenAIPolicy):
    def __init__(self, config=None, *, account_home, account_email, executable="gemini"):
        config = config or OpenAIPolicyConfig(model="gemini-4-argon")
        harness = GeminiCLIHarness(account_home, account_email=account_email, executable=executable,
                                   timeout=config.timeout, model=config.model,
                                   reasoning_effort=config.reasoning_effort,
                                   max_output_tokens=config.max_output_tokens, runner=_cli)
        # ``client`` is the harness (exposes account_home/account_email/executable like GeminiClient).
        self._init_legacy(config, harness, harness)

    def metadata(self):
        data = super().metadata()
        data.update(type="gemini", account_home=str(self.client.account_home),
                    account_email=self.client.account_email, executable=self.client.executable,
                    output_token_limit_enforced=True, max_cli_turns_per_decision=1)
        data["config"].pop("image_detail")
        return data
