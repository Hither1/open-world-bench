"""Stateless OpenAI Responses API harness: one request per decision, store=False."""
from __future__ import annotations

import time

from ..perception import IMAGE_PNG_BASE64, TEXT
from .base import Decision, Harness, HarnessCaps


class OpenAIResponsesHarness(Harness):
    """Accepts an injected Responses-shaped client (``client.responses.create``) for offline tests."""
    caps = HarnessCaps(profile="api-stateless", isolation="process", persistent_session=False,
                       workspace=None, persist_workspace="none", tools=frozenset(),
                       network="model-endpoint", max_turns_per_decision=1)

    def __init__(self, client=None, *, model="gpt-6-astra", reasoning_effort="low",
                 max_output_tokens=2048, image_detail="auto", timeout=60.0):
        if client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise ImportError('OpenAI policy requires pip install -e ".[agent]"') from exc
            client = OpenAI(timeout=timeout, max_retries=0)
        self.client = client
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.max_output_tokens = max_output_tokens
        self.image_detail = image_detail
        self.timeout = timeout

    def _content(self, parts):
        content = []
        for part in parts:
            if part.kind == TEXT:
                content.append(dict(type="input_text", text=part.data))
            elif part.kind == IMAGE_PNG_BASE64:
                content.append(dict(type="input_image", image_url="data:image/png;base64," + part.data,
                                    detail=self.image_detail))
        return content

    def decide(self, system, parts, schema):
        start = time.monotonic()
        response = self.client.responses.create(
            model=self.model, instructions=system,
            input=[dict(role="user", content=self._content(parts))],
            reasoning=dict(effort=self.reasoning_effort),
            text=dict(format=dict(type="json_schema", name="embodied_action", strict=True,
                                  schema=schema)),
            max_output_tokens=self.max_output_tokens, store=False,
        )
        usage = response.usage
        return Decision(payload=None, model=response.model,
                        usage=usage.model_dump() if usage else None,
                        latency_seconds=time.monotonic() - start, raw_id=response.id,
                        status=response.status, text=response.output_text,
                        input_tokens=usage.input_tokens if usage else 0,
                        output_tokens=usage.output_tokens if usage else 0)

    def metadata(self):
        return dict(name="openai_responses", model=self.model, reasoning_effort=self.reasoning_effort,
                    max_output_tokens=self.max_output_tokens, image_detail=self.image_detail,
                    timeout=self.timeout, store=False, output_token_limit_enforced=True)
