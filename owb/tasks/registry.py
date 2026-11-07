"""Task registry: every spec file shipped under ``owb/tasks/specs/``."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from .schema import TaskSpec, load

SPECS_DIR = Path(__file__).resolve().parent / "specs"
_SUFFIXES = (".yaml", ".yml", ".json")


@lru_cache(maxsize=None)
def _index() -> dict[str, Path]:
    out: dict[str, Path] = {}
    for path in sorted(SPECS_DIR.rglob("*")):
        if path.suffix not in _SUFFIXES or not path.is_file():
            continue
        spec = load(path)
        if spec.id in out:
            raise ValueError(f"task id {spec.id!r} is defined by both {out[spec.id]} and {path}")
        out[spec.id] = path
    return out


def available_tasks(sim: str | None = None) -> tuple[str, ...]:
    """Ids of the shipped task specs, optionally only those for one sim."""
    return tuple(sorted(i for i in _index() if sim is None or load_task(i).sim == sim))


def task_path(task_id: str) -> Path:
    try:
        return _index()[task_id]
    except KeyError:
        raise KeyError(f"unknown task {task_id!r}; available: {sorted(_index())}") from None


def load_task(task_id: str) -> TaskSpec:
    """Load a shipped spec by id (a fresh, immutable object each call)."""
    return load(task_path(task_id))


__all__ = ["SPECS_DIR", "available_tasks", "load_task", "task_path"]
