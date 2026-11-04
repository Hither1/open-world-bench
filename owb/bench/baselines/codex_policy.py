"""Run the same vision policy through a separately authenticated Codex CLI."""
from __future__ import annotations

import base64
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import tempfile
from types import SimpleNamespace

from .openai_policy import OpenAIPolicy


class CodexPolicyError(RuntimeError):
    """A CLI/auth/model failure; never replaced with scripted actions."""


def _run_cli(command, *, input, env, cwd, timeout):
    """Bound the entire CLI process tree, including the npm launcher child."""
    with subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, text=True, env=env, cwd=cwd,
                          start_new_session=os.name == "posix") as process:
        try:
            stdout, stderr = process.communicate(input, timeout=timeout)
        except subprocess.TimeoutExpired:
            if os.name == "posix":
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            else:
                process.kill()
            process.communicate()
            raise
        return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)


class CodexClient:
    """Small Responses-shaped transport for the observation-only policy.

    Each call runs in an empty temporary directory, with the supplied image and
    output schema. User configuration and external tools are disabled. The saved
    account's auth stays in its original profile; no tokens are copied or logged.
    """
    def __init__(self, account_home, *, executable="codex", timeout=60):
        self.account_home = Path(account_home).expanduser().resolve()
        if not (self.account_home / "auth.json").is_file():
            raise ValueError(f"No saved Codex login in {self.account_home}")
        self.executable = shutil.which(str(executable))
        if self.executable is None:
            raise ValueError(f"Codex executable not found: {executable}")
        self.timeout = timeout
        self.responses = self

    def create(self, **request):
        content = request["input"][0]["content"]
        text = next(c["text"] for c in content if c["type"] == "input_text")
        data_url = next(c["image_url"] for c in content if c["type"] == "input_image")
        child_env = os.environ.copy()
        # Select the requested Codex account only for this subprocess.
        child_env["CODEX_HOME"] = str(self.account_home)
        for name in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "CODEX_API_KEY",
                     "OPENAI_ORG_ID", "OPENAI_PROJECT_ID"):
            child_env.pop(name, None)
        with tempfile.TemporaryDirectory(prefix="owb-codex-decision-") as directory:
            root = Path(directory)
            frame, schema, output = root / "observation.png", root / "action.schema.json", root / "action.json"
            frame.write_bytes(base64.b64decode(data_url.split(",", 1)[1]))
            schema.write_text(json.dumps(request["text"]["format"]["schema"]))
            command = [self.executable, "exec", "--ignore-user-config", "--ephemeral",
                       "--skip-git-repo-check", "--sandbox", "read-only", "--json",
                       "--model", request["model"], "--cd", directory,
                       "--image", str(frame), "--output-schema", str(schema),
                       "--output-last-message", str(output)]
            settings = {
                "model_reasoning_effort": request["reasoning"]["effort"],
                "web_search": "disabled", "project_doc_max_bytes": 0,
                "forced_login_method": "chatgpt", "approval_policy": "never",
            }
            for feature in ("shell_tool", "shell_snapshot", "multi_agent", "apps", "plugins",
                            "remote_plugin", "browser_use", "computer_use", "image_generation",
                            "view_image", "memories", "skill_search", "hooks"):
                settings[f"features.{feature}"] = False
            for key, value in settings.items():
                command.extend(["-c", f"{key}={json.dumps(value)}"])
            command.append("-")
            prompt = request["instructions"] + "\nUse no tools. Return only the action JSON.\n" + text
            try:
                result = _run_cli(command, input=prompt, env=child_env,
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
            usage = SimpleNamespace(input_tokens=usage_data.get("input_tokens", 0),
                                    output_tokens=usage_data.get("output_tokens", 0),
                                    model_dump=lambda: usage_data)
            thread_id = next((e.get("thread_id") for e in events if e.get("type") == "thread.started"), None)
            return SimpleNamespace(id=thread_id, model=request["model"], status="completed",
                                   output_text=output.read_text(), usage=usage)


class CodexPolicy(OpenAIPolicy):
    def __init__(self, config=None, *, account_home, executable="codex"):
        from .openai_policy import OpenAIPolicyConfig
        config = config or OpenAIPolicyConfig()
        client = CodexClient(account_home, executable=executable, timeout=config.timeout)
        super().__init__(config, client=client)

    def metadata(self):
        data = super().metadata()
        data.update(type="codex", account_home=str(self.client.account_home),
                    executable=self.client.executable, output_token_limit_enforced=False)
        # Codex exec does not expose the Responses API per-response token cap.
        data["config"].pop("max_output_tokens")
        data["config"].pop("image_detail")
        return data
