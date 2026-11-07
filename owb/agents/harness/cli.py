"""Shared pieces of the stateless coding-agent CLI harnesses."""
from __future__ import annotations

from dataclasses import replace
import os
import signal
import subprocess
import time

from .base import Harness


def run_cli(command, *, input, env, cwd, timeout):
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


class StatelessCLIHarness(Harness):
    """One fresh CLI process per decision in an empty temporary directory.

    ``runner`` replaces :func:`run_cli` (same keyword signature) for offline tests.
    Subclasses validate the account/executable in ``__init__`` and implement
    ``_invoke(system, parts, schema, **settings) -> Decision`` with explicit model
    settings, so legacy Responses-shaped shims can pass per-request values;
    ``decide`` uses the harness's configured settings.
    """
    name = "cli"
    output_token_limit_enforced = False

    def _settings(self):
        return dict(model=self.model, effort=self.reasoning_effort)

    def decide(self, system, parts, schema):
        start = time.monotonic()
        decision = self._invoke(system, parts, schema, **self._settings())
        # Harness-side latency; the policy separately times every call, including failures.
        return replace(decision, latency_seconds=time.monotonic() - start)

    def metadata(self):
        return dict(name=self.name, model=self.model, reasoning_effort=self.reasoning_effort,
                    account_home=str(self.account_home), executable=self.executable,
                    timeout=self.timeout, output_token_limit_enforced=self.output_token_limit_enforced)
