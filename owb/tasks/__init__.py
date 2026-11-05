"""Task specifications that are independent of any one simulator backend."""

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
    "MINE_DIAMOND_FROM_SCRATCH",
    "MINECRAFT_TASKS",
    "Milestone",
    "MilestoneTracker",
    "MinecraftTask",
    "minecraft_task",
]
