"""Task specifications that are independent of any one simulator backend."""

from .lemon_tart import (
    MILESTONES as LEMON_TART_MILESTONES,
    LemonTartTracker,
    assign_homes,
)
from .minecraft import (
    ENDER_DRAGON,
    MINE_DIAMOND_FROM_SCRATCH,
    MINECRAFT_TASKS,
    Milestone,
    MilestoneTracker,
    MinecraftTask,
    minecraft_task,
)

__all__ = [
    "ENDER_DRAGON",
    "LEMON_TART_MILESTONES",
    "LemonTartTracker",
    "MINE_DIAMOND_FROM_SCRATCH",
    "MINECRAFT_TASKS",
    "Milestone",
    "MilestoneTracker",
    "MinecraftTask",
    "assign_homes",
    "minecraft_task",
]
