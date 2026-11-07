"""Reference policies for the common schema (observation-only, no simulator access)."""
from __future__ import annotations

import numpy as np

from ..bench.framework import BUTTONS, COMMANDS, neutral_action
from ..sims.voxel.env import INVENTORY_COLUMNS, ITEM_IDS


class RandomPolicy:
    """Seeded policy which samples only capabilities supported by the backend."""
    def reset(self, seed):
        self.rng = np.random.default_rng(seed)

    def __call__(self, observation):
        action = neutral_action()
        action["move"] = self.rng.uniform(-1, 1, 2).astype(np.float32)
        action["look"] = self.rng.uniform(-1, 1, 2).astype(np.float32)
        action["buttons"] = (self.rng.random(len(BUTTONS)) < 0.15).astype(np.int8) * observation["button_mask"]
        action["command"] = int(self.rng.choice(np.flatnonzero(observation["command_mask"])))
        action["swap"] = int(self.rng.choice(np.flatnonzero(observation["swap_mask"])))
        action["slot"] = int(self.rng.integers(32))
        return action


class DemoPolicy:
    """A capability-aware demonstration, not a task-solving baseline.

    Reads only policy observations. Uses party skills when available, otherwise
    demonstrates the fixed Skylands bridge course or gathers/cooks/eats at the
    Wildlands starter camp. The bridge sequence is a showcase, not a general solver.
    """
    def reset(self, seed):
        self.step = 0

    def __call__(self, observation):
        action = neutral_action()
        step, self.step = self.step, self.step + 1
        supported = observation["command_mask"]
        if supported[COMMANDS.index("grab")]:
            # Fixed-course showcase. No simulator state or privileged info is read.
            if step in (0, 6):
                action.update(command=COMMANDS.index("grab"), part=1 if step == 0 else 2)
            elif 1 <= step <= 4 or 7 <= step <= 13:
                index = step - (1 if step <= 4 else 7)
                action["build_move"][:] = (1, -1 if index == 0 else 0,
                                            (1 if step <= 4 else -1) if index < 3 else 0)
            elif step in (5, 14):
                action["command"] = COMMANDS.index("place")
            else:
                action["move"][0] = 1
                action["command"] = COMMANDS.index("interact")
            return action
        if supported[COMMANDS.index("skill")]:
            if step % 12 == 0:
                action["command"] = COMMANDS.index("skill")
                slots = np.flatnonzero(observation["swap_mask"])[1:]
                if len(slots):
                    action["swap"] = int(slots[(step // 12) % len(slots)])
        elif step == 0:
            action["command"] = COMMANDS.index("interact")
        elif step == 1 and supported[COMMANDS.index("cook")]:
            pepper = ITEM_IDS.index("spicy_pepper") + 1
            slots = np.flatnonzero(observation["inventory"][:, 0] == pepper)
            if len(slots):
                action["command"] = COMMANDS.index("cook")
                action["ingredients"][0] = int(slots[0]) + 1
        elif step == 2 and supported[COMMANDS.index("eat")]:
            slots = np.flatnonzero(observation["inventory"][:, INVENTORY_COLUMNS.index("dish")])
            if len(slots):
                action["command"] = COMMANDS.index("eat")
                action["slot"] = int(slots[0])
        if step >= 4:
            action["move"][0] = 0.7
            action["look"][0] = 0.25 if step % 24 < 8 else 0.0
            action["buttons"][BUTTONS.index("jump")] = int(step % 15 == 5)
        return action
