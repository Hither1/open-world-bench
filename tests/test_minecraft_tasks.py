from pathlib import Path

import numpy as np
import pytest

from owb.tasks import ENDER_DRAGON, MINE_DIAMOND_FROM_SCRATCH, MilestoneTracker, minecraft_task

MFM_TASKS = (Path(__file__).resolve().parents[2] / "MFM" / "MCU" / "MCU_benchmark"
             / "task_configs" / "tasks")


def test_milestone_counts_and_budgets():
    assert len(MINE_DIAMOND_FROM_SCRATCH.milestones) == 10
    assert len(ENDER_DRAGON.milestones) == 24
    assert MINE_DIAMOND_FROM_SCRATCH.step_budget == 12_000
    assert ENDER_DRAGON.step_budget == 36_000
    for task in (MINE_DIAMOND_FROM_SCRATCH, ENDER_DRAGON):
        assert len(set(task.identities)) == len(task.identities)
        assert task.world_seed == 0
    assert ENDER_DRAGON.identities[-1] == "kill_ender_dragon"
    assert MINE_DIAMOND_FROM_SCRATCH.identities[-1] == "diamond"


def test_unknown_task_names_the_choices():
    with pytest.raises(KeyError, match="ender_dragon"):
        minecraft_task("nether_portal")


@pytest.mark.parametrize("task", [MINE_DIAMOND_FROM_SCRATCH, ENDER_DRAGON],
                         ids=lambda t: t.name)
def test_matches_mfm_configs(task):
    yaml = pytest.importorskip("yaml")
    files = sorted((MFM_TASKS / task.name).glob("*.yaml"))
    if not files:
        pytest.skip("no ../MFM checkout next to this repository")
    data = yaml.safe_load(files[0].read_text(encoding="utf-8"))
    cfg = data["milestone_reward_cfg"]
    assert [(c["identity"], c["event"], tuple(c["objects"])) for c in cfg] == [
        (m.identity, m.event, m.objects) for m in task.milestones]
    assert all(c["reward"] == 1.0 and c["max_reward_times"] == 1 for c in cfg)
    assert data["text"] == task.text
    assert data.get("world_seed", 0) == task.world_seed


def test_tracker_scores_increases_not_holdings():
    tracker = MilestoneTracker(MINE_DIAMOND_FROM_SCRATCH)
    # Statistics present at reset (e.g. carried over or given by a kit) count for nothing.
    tracker.reset({"mine_block": {"oak_log": 3}, "pickup": {"diamond": 1}})
    assert tracker.update({"mine_block": {"oak_log": 3}, "pickup": {"diamond": 1}}, 1) == []
    fired = tracker.update({"mine_block": {"oak_log": np.array(4)},
                            "pickup": {"diamond": 1}}, 2)
    assert fired == ["mine_log"]
    # Out of order is fine; each milestone fires once.
    assert tracker.update({"mine_block": {"oak_log": 5},
                           "pickup": {"diamond": 2}}, 3) == ["diamond"]
    assert tracker.achieved == {"mine_log": 2, "diamond": 3}
    assert tracker.summary()["score"] == 20.0


def test_tracker_alias_objects():
    tracker = MilestoneTracker(ENDER_DRAGON)
    tracker.reset({})
    assert tracker.update({"craft_item": {"ender_eye": 1},
                           "mine_block": {"deepslate_diamond_ore": 1},
                           "kill_entity": {"ender_dragon": 1}}, 7) == [
        "mine_diamond_ore", "eye_of_ender", "kill_ender_dragon"]
    assert tracker.summary()["milestones_achieved"] == 3
