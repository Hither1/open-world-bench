"""Stateless Claude Code CLI harness using a saved OAuth login (no tokens extracted)."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

from ..perception import IMAGE_PNG_BASE64, TEXT
from .base import Decision, HarnessCaps
from .cli import StatelessCLIHarness, run_cli


class ClaudePolicyError(RuntimeError):
    def __init__(self, message, code="claude_inference_failed"):
        super().__init__(message)
        self.code = code


class ClaudeCLIHarness(StatelessCLIHarness):
    """``claude -p`` in safe/restricted mode with no tools beyond StructuredOutput."""
    name = "claude_cli"
    output_token_limit_enforced = False
    MAX_TURNS = 3
    caps = HarnessCaps(profile="cli-stateless", isolation="process", persistent_session=False,
                       workspace=None, persist_workspace="none", tools=frozenset({"StructuredOutput"}),
                       network="model-endpoint", max_turns_per_decision=MAX_TURNS)

    def __init__(self, account_home, *, executable="claude", timeout=60, model="claude-opus-5-5",
                 reasoning_effort="low", runner=None):
        self.account_home = Path(account_home).expanduser().resolve()
        if not (self.account_home / ".credentials.json").is_file():
            raise ValueError(f"No saved Claude credentials in {self.account_home}")
        self.executable = shutil.which(str(executable))
        if self.executable is None:
            raise ValueError(f"Claude executable not found: {executable}")
        self.timeout = timeout
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.runner = run_cli if runner is None else runner

    def _invoke(self, system, parts, schema, *, model, effort):
        content = []
        for part in parts:
            if part.kind == TEXT:
                content.append(dict(type="text", text=part.data))
            elif part.kind == IMAGE_PNG_BASE64:
                content.append(dict(type="image", source=dict(type="base64", media_type="image/png",
                                                              data=part.data)))
        message = dict(type="user", message=dict(role="user", content=content))
        child_env = os.environ.copy()
        child_env["CLAUDE_CONFIG_DIR"] = str(self.account_home)
        for name in list(child_env):
            if name.startswith("ANTHROPIC_") or name in (
                "CLAUDE_CODE_OAUTH_TOKEN", "CLAUDE_CODE_USE_BEDROCK", "CLAUDE_CODE_USE_VERTEX",
                "CLAUDE_CODE_USE_FOUNDRY", "CLAUDE_CODE_SIMPLE", "CLAUDE_CODE_MAX_OUTPUT_TOKENS",
            ):
                child_env.pop(name, None)
        command = [
            self.executable, "-p", "--safe-mode", "--restricted", "--tools", "",
            "--strict-mcp-config", "--mcp-config", '{"mcpServers":{}}',
            "--setting-sources", "", "--no-session-persistence", "--disable-slash-commands",
            "--no-chrome", "--permission-mode", "dontAsk", "--max-turns", str(self.MAX_TURNS),
            "--model", model, "--effort", effort,
            "--input-format", "stream-json", "--output-format", "stream-json", "--verbose",
            "--json-schema", json.dumps(schema),
            "--system-prompt", system + "\nReturn the action using StructuredOutput. Use no external tools.",
        ]
        # Avoid shared-filesystem cleanup races; the image travels through stdin only.
        with tempfile.TemporaryDirectory(prefix="owb-claude-decision-", dir="/tmp",
                                         ignore_cleanup_errors=True) as directory:
            try:
                process = self.runner(command, input=json.dumps(message, allow_nan=False) + "\n",
                                      env=child_env, cwd=directory, timeout=self.timeout)
            except subprocess.TimeoutExpired as exc:
                raise ClaudePolicyError("Claude decision timed out", "timeout") from exc
        events = []
        for line in process.stdout.splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(event, dict):
                events.append(event)
        results = [e for e in events if e.get("type") == "result"]
        result = results[-1] if results else {}
        if process.returncode or result.get("is_error") or result.get("subtype") != "success":
            details = str(result.get("result", "")).lower()
            if "account is on hold" in details:
                raise ClaudePolicyError("Claude account is on hold; resolve it at https://claude.ai/restricted", "account_on_hold")
            if "usage limit" in details or "rate limit" in details:
                raise ClaudePolicyError("Claude account usage limit reached", "usage_limit")
            if "not logged in" in details or "authentication" in details or "login" in details:
                raise ClaudePolicyError("Claude account needs authentication", "authentication_failed")
            if "model" in details and any(s in details for s in ("not found", "not available", "not supported")):
                raise ClaudePolicyError("Requested model is unavailable to this Claude account/client", "model_unavailable")
            raise ClaudePolicyError(f"Claude inference failed (exit {process.returncode}, result {result.get('subtype')})")
        models = set()
        for event in events:
            if event.get("type") == "system" and event.get("subtype") == "init":
                if set(event.get("tools", [])) - {"StructuredOutput"}:
                    raise ClaudePolicyError("Unexpected tools enabled in Claude session", "unexpected_tools")
            if event.get("type") == "assistant":
                response_message = event.get("message", {})
                models.add(response_message.get("model"))
                for block in response_message.get("content", []):
                    if block.get("type") == "tool_use" and block.get("name") != "StructuredOutput":
                        raise ClaudePolicyError("Unexpected Claude tool activity", "unexpected_tools")
        if models != {model}:
            raise ClaudePolicyError("Claude response did not confirm the exact requested model", "model_mismatch")
        payload = result.get("structured_output")
        if not isinstance(payload, dict):
            raise ClaudePolicyError("Claude returned no structured action", "missing_action")
        usage_data = result.get("usage") or {}
        # Include cache reads/writes, which Anthropic reports separately from input_tokens.
        input_tokens = sum(usage_data.get(key, 0) for key in (
            "input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"))
        return Decision(payload=payload, model=model, usage=dict(usage_data, total_input_tokens=input_tokens),
                        latency_seconds=0.0, raw_id=result.get("session_id"), status="completed",
                        text=json.dumps(payload), input_tokens=input_tokens,
                        output_tokens=usage_data.get("output_tokens", 0))
