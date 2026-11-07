"""Task specifications that are independent of any one simulator backend.

`schema` defines the declarative `TaskSpec` (YAML/JSON), `evaluator` scores it from
events and a read-only state view, `registry` lists the shipped spec files, and
`env.TaskEnv` runs a spec on a common-schema environment. See docs/tasks.md.
"""

from .evaluator import MilestoneEvaluator, State, StateView
from .minecraft import (
    ENDER_DRAGON,
    MINE_DIAMOND_FROM_SCRATCH,
    MINECRAFT_TASKS,
    Milestone,
    MilestoneTracker,
    MinecraftTask,
    minecraft_task,
)
from .registry import available_tasks, load_task
from .schema import Budget, Predicate, TaskSpec, dump, load
from .schema import Milestone as TaskMilestone

__all__ = [
    "ENDER_DRAGON",
    "MINE_DIAMOND_FROM_SCRATCH",
    "MINECRAFT_TASKS",
    "Milestone",
    "MilestoneTracker",
    "MinecraftTask",
    "minecraft_task",
    "Budget",
    "MilestoneEvaluator",
    "Predicate",
    "State",
    "StateView",
    "TaskMilestone",
    "TaskSpec",
    "available_tasks",
    "dump",
    "load",
    "load_task",
]


def __getattr__(name):
    # TaskEnv imports gymnasium and the bench layer; load it only when asked for.
    if name in ("TaskEnv", "TeyvatStateView", "NullStateView"):
        from . import env
        return getattr(env, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
