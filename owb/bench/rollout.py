"""One episode runner and optional video recorder for every common-schema env."""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from .framework import BUTTONS, COMMANDS, MODALITIES, PROPRIO_NAMES, SCHEMA_VERSION, jsonable, neutral_action
from ..env import INVENTORY_COLUMNS, ITEM_IDS, SURVIVAL_NAMES
from ..sim.construction import PART_COLUMNS, BUILD_STATE_NAMES


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


class EpisodeRecorder:
    """JSONL transitions plus optional GIF/MP4; refuses to overwrite a run."""
    def __init__(self, directory, env, seed, video=None):
        if video not in (None, "gif", "mp4"):
            raise ValueError("video must be None, 'gif', or 'mp4'")
        self.video, self.log = None, None
        if video:
            try:
                import imageio.v2 as imageio
            except ImportError as exc:
                raise ImportError('video recording requires pip install -e ".[video]"') from exc
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        config = dict(schema_version=SCHEMA_VERSION, environment=env.name, seed=seed,
                      benchmark=asdict(env.config), native=asdict(env.native.cfg),
                      schema=dict(buttons=BUTTONS, commands=COMMANDS, modalities=MODALITIES,
                                  proprio=PROPRIO_NAMES, inventory=INVENTORY_COLUMNS,
                                  item_ids=ITEM_IDS, survival=SURVIVAL_NAMES),
                      video=video, decision_seconds=env.config.decision_seconds)
        config["schema"].update(construction=PART_COLUMNS, build_state=BUILD_STATE_NAMES)
        try:
            self._write_json("config.json", config)
            self.log = (self.directory / "trajectory.jsonl").open("w")
            if video == "gif":
                self.video = imageio.get_writer(self.directory / "episode.gif", mode="I",
                                               duration=env.config.decision_seconds * 1000, loop=0)
            elif video == "mp4":
                self.video = imageio.get_writer(self.directory / "episode.mp4",
                                               fps=1 / env.config.decision_seconds, macro_block_size=1)
        except Exception:
            self.close()
            raise

    def _write_json(self, name, value):
        (self.directory / name).write_text(json.dumps(jsonable(value), indent=2, allow_nan=False) + "\n")

    def record(self, kind, observation, info, *, action=None, reward=None):
        row = dict(type=kind, info=info, proprio=observation["proprio"], instruction=observation["instruction"])
        if action is not None:
            row.update(action=action, reward=reward)
        self.log.write(json.dumps(jsonable(row), allow_nan=False) + "\n")
        if self.video is not None:
            self.video.append_data(observation["rgb"])

    def finish(self, summary):
        self._write_json("summary.json", summary)

    def close(self):
        try:
            if self.video is not None:
                self.video.close()
        finally:
            if self.log is not None:
                self.log.close()


def run_episode(env, policy=None, *, seed=None, output_dir=None, video=None) -> dict:
    """Run until termination/truncation. Policies receive observations only.

    A policy is callable and may expose reset(seed) for episode-local memory.
    The caller owns the environment and can reset/reuse or close it afterward.
    """
    if video is not None and output_dir is None:
        raise ValueError("video recording requires output_dir")
    policy = RandomPolicy() if policy is None else policy
    seed = env.config.seed if seed is None else int(seed)
    observation, info = env.reset(seed=seed)
    if hasattr(policy, "reset"):
        policy.reset(seed)
    recorder = EpisodeRecorder(output_dir, env, seed, video) if output_dir is not None else None
    counts = Counter()
    try:
        if recorder:
            recorder.record("reset", observation, info)
        while True:
            action = policy(observation)
            observation, reward, terminated, truncated, info = env.step(action)
            counts.update(event["type"] for event in info["events"])
            if recorder:
                recorder.record("step", observation, info, action=action, reward=reward)
            if terminated or truncated:
                summary = dict(info["episode"], schema_version=SCHEMA_VERSION,
                               metrics=info["metrics"], event_counts=dict(counts))
                if recorder:
                    recorder.finish(summary)
                return summary
    finally:
        if recorder:
            recorder.close()
