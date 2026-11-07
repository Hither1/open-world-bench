"""Stateless Gemini CLI harness using a saved Google login; core tools disabled."""
from __future__ import annotations

import base64
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

from ..perception import IMAGE_PNG_BASE64, TEXT
from .base import Decision, HarnessCaps
from .cli import StatelessCLIHarness, run_cli


class GeminiPolicyError(RuntimeError):
    def __init__(self, message, code="gemini_inference_failed"):
        super().__init__(message)
        self.code = code


class GeminiCLIHarness(StatelessCLIHarness):
    name = "gemini_cli"
    output_token_limit_enforced = True
    MAX_SESSION_TURNS = 1
    caps = HarnessCaps(profile="cli-stateless", isolation="process", persistent_session=False,
                       workspace=None, persist_workspace="none", tools=frozenset(),
                       network="model-endpoint", max_turns_per_decision=MAX_SESSION_TURNS)

    def __init__(self, account_home, *, account_email, executable="gemini", timeout=60,
                 model="gemini-4-argon", reasoning_effort="low", max_output_tokens=2048, runner=None):
        self.account_home = Path(account_home).expanduser().resolve()
        if self.account_home.name != ".gemini":
            raise ValueError("Gemini account directory must be named .gemini")
        self.account_email = account_email.strip().lower()
        if not self.account_email:
            raise ValueError("Gemini account email is required")
        self.executable = shutil.which(str(executable))
        if self.executable is None:
            raise ValueError(f"Gemini executable not found: {executable}")
        self.timeout = timeout
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.max_output_tokens = max_output_tokens
        self.runner = run_cli if runner is None else runner

    def check_account(self):
        """Read only non-secret account metadata, never OAuth token contents."""
        accounts = self.account_home / "google_accounts.json"
        if not accounts.is_file() or not (self.account_home / "oauth_creds.json").is_file():
            raise GeminiPolicyError("Sign in to Gemini CLI with the requested Google account first", "login_required")
        active = json.loads(accounts.read_text()).get("active")
        if not isinstance(active, str) or active.lower() != self.account_email:
            raise GeminiPolicyError("Gemini CLI active Google account does not match the requested email", "account_mismatch")

    def _settings(self):
        return dict(super()._settings(), max_output_tokens=self.max_output_tokens)

    def _invoke(self, system, parts, schema, *, model, effort, max_output_tokens):
        self.check_account()
        if effort not in ("low", "medium", "high"):
            raise ValueError("Gemini 4 Argon supports low, medium, or high reasoning effort")
        text = next(p.data for p in parts if p.kind == TEXT)
        data = next(p.data for p in parts if p.kind == IMAGE_PNG_BASE64)
        child_env = os.environ.copy()
        child_env["GEMINI_CLI_HOME"] = str(self.account_home.parent)
        for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "GOOGLE_APPLICATION_CREDENTIALS",
                     "GOOGLE_GENAI_USE_VERTEXAI", "GOOGLE_CLOUD_PROJECT", "GOOGLE_CLOUD_LOCATION",
                     "GEMINI_MODEL", "GEMINI_CLI_AUTH_OVERRIDE", "GEMINI_SYSTEM_MD"):
            child_env.pop(name, None)
        with tempfile.TemporaryDirectory(prefix="owb-gemini-decision-", dir="/tmp",
                                         ignore_cleanup_errors=True) as directory:
            root = Path(directory)
            (root / "observation.png").write_bytes(base64.b64decode(data, validate=True))
            # These are task-local settings. System/admin policy still takes precedence.
            settings = {
                "security": {"auth": {"selectedType": "oauth-personal"}},
                "tools": {"core": []}, "hooksConfig": {"enabled": False},
                "admin": {"mcp": {"enabled": False}, "extensions": {"enabled": False},
                          "skills": {"enabled": False}},
                "context": {"fileName": "OWB_UNUSED_CONTEXT.md", "includeDirectories": [],
                            "loadFromIncludeDirectories": False},
                "advanced": {"ignoreLocalEnv": True},
                "model": {"name": model, "maxSessionTurns": self.MAX_SESSION_TURNS},
                "modelConfigs": {"customOverrides": [{
                    "match": {"model": model},
                    "modelConfig": {"generateContentConfig": {
                        "thinkingConfig": {"thinkingLevel": effort.upper()},
                        "maxOutputTokens": max_output_tokens,
                        "responseMimeType": "application/json",
                        "responseJsonSchema": schema,
                    }},
                }]},
            }
            (root / ".gemini").mkdir()
            (root / ".gemini" / "settings.json").write_text(json.dumps(settings))
            # The CLI expands this explicit attachment before inference, even with tools disabled.
            # Notes/instructions must not introduce additional CLI @file attachments.
            prompt = (system + "\nReturn only the action JSON matching this schema:\n" +
                      json.dumps(schema) +
                      "\nSensors and memory:\n" + text).replace("@", "＠")
            prompt += "\nCurrent RGB observation: @observation.png"
            command = [self.executable, "--skip-trust", "--model", model,
                       "--output-format", "json", "--prompt", prompt]
            try:
                process = self.runner(command, input="", env=child_env, cwd=directory, timeout=self.timeout)
            except subprocess.TimeoutExpired as exc:
                raise GeminiPolicyError("Gemini decision timed out", "timeout") from exc
        try:
            result = json.loads(process.stdout)
        except json.JSONDecodeError as exc:
            raise GeminiPolicyError("Gemini returned no JSON result; check login and CLI setup", "invalid_cli_output") from exc
        if not isinstance(result, dict):
            raise GeminiPolicyError("Invalid Gemini CLI result", "invalid_cli_output")
        if process.returncode or result.get("error"):
            details = str(result.get("error", "")).lower()
            if "quota" in details or "rate limit" in details:
                raise GeminiPolicyError("Gemini account usage limit reached", "usage_limit")
            if "auth" in details or "login" in details or "sign in" in details:
                raise GeminiPolicyError("Gemini account needs authentication", "login_required")
            if "model" in details and any(s in details for s in ("not found", "not available", "not supported")):
                raise GeminiPolicyError("Requested model is unavailable to this Gemini account/client", "model_unavailable")
            raise GeminiPolicyError(f"Gemini inference failed (exit {process.returncode})")
        stats = result.get("stats") or {}
        models = stats.get("models") or {}
        if set(models) != {model}:
            raise GeminiPolicyError("Gemini statistics did not confirm only the requested model", "model_mismatch")
        if stats.get("tools", {}).get("totalCalls", 0):
            raise GeminiPolicyError("Unexpected Gemini model tool activity", "unexpected_tools")
        if not isinstance(result.get("response"), str) or not result["response"].strip():
            raise GeminiPolicyError("Gemini returned no action", "missing_action")
        self.check_account()
        tokens = models[model].get("tokens") or {}
        return Decision(payload=None, model=model, usage=tokens, latency_seconds=0.0,
                        raw_id=result.get("session_id"), status="completed", text=result["response"],
                        input_tokens=tokens.get("prompt", 0),
                        output_tokens=tokens.get("candidates", 0) + tokens.get("thoughts", 0))

    def metadata(self):
        return dict(super().metadata(), account_email=self.account_email,
                    max_output_tokens=self.max_output_tokens)
