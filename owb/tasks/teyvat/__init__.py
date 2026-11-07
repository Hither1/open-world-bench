"""Teyvat task mappings: the native QuestGraph expressed as a TaskSpec."""

from .questline import builtin_questline_spec, questgraph_to_spec

__all__ = ["builtin_questline_spec", "questgraph_to_spec"]
