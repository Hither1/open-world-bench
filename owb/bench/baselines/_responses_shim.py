"""Deprecated Responses-API-shaped adapters over the CLI harnesses.

``CodexClient``/``ClaudeClient``/``GeminiClient`` exist for callers that still send
``client.responses.create(**request)`` (e.g. the Teyvat/Wildlands model scripts).
New code should call ``owb.agents.harness.*.decide`` directly.
"""
from __future__ import annotations

from types import SimpleNamespace

from ...agents.perception import IMAGE_PNG_BASE64, TEXT, Part

_PNG_PREFIX = "data:image/png;base64"


def parts_from_request(request, *, png_error=None):
    """Responses ``input`` content -> parts. ``png_error`` rejects non-PNG data URLs."""
    parts = []
    for item in request["input"][0]["content"]:
        if item["type"] == "input_text":
            parts.append(Part(TEXT, item["text"]))
        elif item["type"] == "input_image":
            prefix, data = item["image_url"].split(",", 1)
            if png_error is not None and prefix != _PNG_PREFIX:
                raise ValueError(png_error)
            parts.append(Part(IMAGE_PNG_BASE64, data))
    return parts


def response_from_decision(decision):
    usage = SimpleNamespace(input_tokens=decision.input_tokens, output_tokens=decision.output_tokens,
                            model_dump=lambda: decision.usage)
    return SimpleNamespace(id=decision.raw_id, model=decision.model, status=decision.status,
                           output_text=decision.text, usage=usage)
