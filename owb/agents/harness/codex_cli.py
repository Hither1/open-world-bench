"""Stateless Codex CLI harness: ``codex exec`` with a read-only sandbox and tools disabled."""
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


class CodexPolicyError(RuntimeError):
    """A CLI/auth/model failure; never replaced with scripted actions."""


class CodexCLIHarness(StatelessCLIHarness):
    """Each call runs in an empty temporary directory, with the supplied image and
    output schema. User configuration and external tools are disabled. The saved
    account's auth stays in its original profile; no tokens are copied or logged.
    """
    name = "codex_cli"
    output_token_limit_enforced = False
    SANDBOX = "read-only"
    DISABLED_FEATURES = ("shell_tool", "shell_snapshot", "multi_agent", "apps", "plugins",
                         "remote_plugin", "browser_use", "computer_use", "image_generation",
                         "view_image", "memories", "skill_search", "hooks")
    caps = HarnessCaps(profile="cli-stateless", isolation="process", persistent_session=False,
                       workspace=None, persist_workspace="none", tools=frozenset(),
                       network="model-endpoint", max_turns_per_decision=1)

    def __init__(self, account_home, *, executable="codex", timeout=60, model="gpt-6-astra",
                 reasoning_effort="low", runner=None):
        self.account_home = Path(account_home).expanduser().resolve()
        if not (self.account_home / "auth.json").is_file():
            raise ValueError(f"No saved Codex login in {self.account_home}")
        self.executable = shutil.which(str(executable))
        if self.executable is None:
            raise ValueError(f"Codex executable not found: {executable}")
        # Inference changes cwd to an empty temporary directory. Preserve a
        # caller-supplied relative installation path before that directory switch.
        self.executable = str(Path(self.executable).resolve())
        self.timeout = timeout
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.runner = run_cli if runner is None else runner

    def _invoke(self, system, parts, schema, *, model, effort):
        text = next(p.data for p in parts if p.kind == TEXT)
        image = next(p.data for p in parts if p.kind == IMAGE_PNG_BASE64)
        child_env = os.environ.copy()
        # Select the requested Codex account only for this subprocess.
        child_env["CODEX_HOME"] = str(self.account_home)
        for name in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "CODEX_API_KEY",
                     "OPENAI_ORG_ID", "OPENAI_PROJECT_ID"):
            child_env.pop(name, None)
        # Host TMPDIR may be NFS: delayed CLI writes can race directory removal there.
        # Cleanup failures must not discard a valid model action.
        with tempfile.TemporaryDirectory(prefix="owb-codex-decision-", dir="/tmp",
                                         ignore_cleanup_errors=True) as directory:
            root = Path(directory)
            frame, schema_path, output = root / "observation.png", root / "action.schema.json", root / "action.json"
            frame.write_bytes(base64.b64decode(image))
            schema_path.write_text(json.dumps(schema))
            command = [self.executable, "exec", "--ignore-user-config", "--ephemeral",
                       "--skip-git-repo-check", "--sandbox", self.SANDBOX, "--json",
                       "--model", model, "--cd", directory,
                       "--image", str(frame), "--output-schema", str(schema_path),
                       "--output-last-message", str(output)]
            settings = {
                "model_reasoning_effort": effort,
                "web_search": "disabled", "project_doc_max_bytes": 0,
                "forced_login_method": "chatgpt", "approval_policy": "never",
            }
            for feature in self.DISABLED_FEATURES:
                settings[f"features.{feature}"] = False
            for key, value in settings.items():
                command.extend(["-c", f"{key}={json.dumps(value)}"])
            command.append("-")
            prompt = system + "\nUse no tools. Return only the action JSON.\n" + text
            try:
                result = self.runner(command, input=prompt, env=child_env,
                                     cwd=directory, timeout=self.timeout)
            except subprocess.TimeoutExpired as exc:
                raise CodexPolicyError("Codex decision timed out") from exc
            events = []
            for line in result.stdout.splitlines():
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
            failures = [e for e in events if e.get("type") in ("error", "turn.failed")]
            if result.returncode or failures or not output.is_file():
                # Return a useful error without retaining raw CLI stdout/stderr in recordings.
                details = " ".join(str(e.get("message", e.get("error", ""))) for e in failures)
                if "newer version" in details:
                    reason = "requested model requires a newer Codex CLI; upgrade or use --codex-executable"
                elif "not supported" in details or "not found" in details:
                    reason = "requested model is unavailable to this Codex account/client"
                elif "usage limit" in details or "quota" in details:
                    reason = "Codex account usage limit reached"
                else:
                    reason = f"Codex inference failed (exit {result.returncode}); check account login and model access"
                raise CodexPolicyError(reason)
            for event in events:
                if event.get("type") in ("item.started", "item.completed"):
                    kind = event.get("item", {}).get("type")
                    if kind not in ("agent_message", "reasoning", "error"):
                        raise CodexPolicyError(f"Unexpected Codex tool activity: {kind}")
            completed = [e for e in events if e.get("type") == "turn.completed"]
            if not completed:
                raise CodexPolicyError("Codex returned no completed turn")
            usage_data = completed[-1].get("usage", {})
            thread_id = next((e.get("thread_id") for e in events if e.get("type") == "thread.started"), None)
            return Decision(payload=None, model=model, usage=usage_data, latency_seconds=0.0,
                            raw_id=thread_id, status="completed", text=output.read_text(),
                            input_tokens=usage_data.get("input_tokens", 0),
                            output_tokens=usage_data.get("output_tokens", 0))

    def metadata(self):
        return dict(super().metadata(), sandbox=self.SANDBOX)
