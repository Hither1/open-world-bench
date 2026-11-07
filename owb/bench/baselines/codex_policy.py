"""Run the same vision policy through a separately authenticated Codex CLI.

Compatibility layer over ``owb.agents.harness.codex_cli.CodexCLIHarness``.
"""
from __future__ import annotations

from ...agents.harness.cli import run_cli as _run_cli
from ...agents.harness.codex_cli import CodexCLIHarness, CodexPolicyError
from ._responses_shim import parts_from_request, response_from_decision
from .openai_policy import OpenAIPolicy, OpenAIPolicyConfig

__all__ = ["CodexPolicy", "CodexPolicyError", "CodexClient"]


def _cli(*args, **kwargs):
    # Look up _run_cli at call time so patching this module's attribute takes effect.
    return _run_cli(*args, **kwargs)


class CodexClient:
    """Deprecated Responses-shaped adapter over CodexCLIHarness.

    Still used by the Teyvat/Wildlands model scripts, which send their own schemas.
    """
    def __init__(self, account_home, *, executable="codex", timeout=60):
        self.harness = CodexCLIHarness(account_home, executable=executable, timeout=timeout, runner=_cli)
        self.account_home, self.executable = self.harness.account_home, self.harness.executable
        self.timeout = timeout
        self.responses = self

    def create(self, **request):
        parts = parts_from_request(request)
        return response_from_decision(self.harness._invoke(
            request["instructions"], parts, request["text"]["format"]["schema"],
            model=request["model"], effort=request["reasoning"]["effort"]))


class CodexPolicy(OpenAIPolicy):
    def __init__(self, config=None, *, account_home, executable="codex"):
        config = config or OpenAIPolicyConfig()
        harness = CodexCLIHarness(account_home, executable=executable, timeout=config.timeout,
                                  model=config.model, reasoning_effort=config.reasoning_effort,
                                  runner=_cli)
        # ``client`` is the harness (exposes account_home/executable like the old CodexClient).
        self._init_legacy(config, harness, harness)

    def metadata(self):
        data = super().metadata()
        data.update(type="codex", account_home=str(self.client.account_home),
                    executable=self.client.executable, output_token_limit_enforced=False)
        # Codex exec does not expose the Responses API per-response token cap.
        data["config"].pop("max_output_tokens")
        data["config"].pop("image_detail")
        return data
