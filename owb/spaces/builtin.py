"""Declarations of the shared controls and the seven modalities of schema version 2.

Spaces, vocabularies and descriptions are exactly those of the former central
tables in ``owb.bench.framework`` and ``owb.agents.perception``. Registration
order is the canonical ``MODALITIES`` order and must not change.
"""
from __future__ import annotations

from gymnasium import spaces
import numpy as np

from ..core import jsonable
from ..sims.voxel.env import INVENTORY_COLUMNS, ITEM_IDS, SURVIVAL_NAMES
from ..sims.voxel.sim.construction import BUILD_COMMANDS, BUILD_STATE_NAMES, MAX_PARTS, PART_COLUMNS
from . import Control, Modality, SpaceRegistry

BUTTONS = ("jump", "sprint", "climb", "glide", "dive", "crouch")
COMMANDS = ("none", "interact", "attack", "skill", "burst", "eat", "cook", "equip", "stow") + BUILD_COMMANDS
PROPRIO_NAMES = (
    "velocity_forward", "velocity_right", "velocity_up", "health_fraction", "stamina_fraction",
    "grounded", "airborne", "swimming", "climbing", "gliding", "day_sin", "day_cos",
)
#: per-member layout of the Teyvat ``party`` vector (4 members x 7, then 2 status values)
PARTY_MEMBER_NAMES = ("active", "health_fraction", "energy_fraction", "skill_ready",
                      "burst_cooldown_ready", "element_id_div_6", "alive")
PARTY_STATUS_NAMES = ("shield_fraction", "swap_ready")


def _named(names, values):
    return dict(zip(names, jsonable(values)))


def _describe_party(party, observation):
    names = PARTY_MEMBER_NAMES
    return {"party": [dict(slot=i + 1, **_named(names, party[i * 7:(i + 1) * 7]))
                      for i in range(4) if observation["swap_mask"][i + 1]],
            "party_status": _named(PARTY_STATUS_NAMES, party[28:])}


def _describe_inventory(inventory, observation):
    entries = []
    for slot, row in enumerate(inventory):
        if row[0] > 0 and row[1] > 0:
            entry = dict(slot=slot, **_named(INVENTORY_COLUMNS, row))
            entry["item_name"] = ITEM_IDS[int(row[0]) - 1]
            entries.append(entry)
    return {"inventory": entries}


def _describe_survival(survival, observation):
    return {"survival": _named(SURVIVAL_NAMES, survival)}


def _describe_minimap(minimap, observation):
    return {"visited_map": {"shape": list(minimap.shape),
                            "nonzero_cells": np.argwhere(minimap > 0).tolist()}}


def _describe_goal(goal, observation):
    return {"oracle_goal_hint": jsonable(goal)}


def _describe_construction(construction, observation):
    return {"construction": [dict(part=i + 1, **_named(PART_COLUMNS, row))
                             for i, row in enumerate(construction) if row[0]]}


def _describe_build_state(build_state, observation):
    return {"build_state": _named(BUILD_STATE_NAMES, build_state)}


PARTY = Modality("party", spaces.Box(0, np.inf, (30,), np.float32), _describe_party)
INVENTORY = Modality("inventory", spaces.Box(0, np.inf, (32, len(INVENTORY_COLUMNS)), np.float32),
                     _describe_inventory, vocab=dict(inventory=INVENTORY_COLUMNS, item_ids=ITEM_IDS))
SURVIVAL = Modality("survival", spaces.Box(-np.inf, np.inf, (len(SURVIVAL_NAMES),), np.float32),
                    _describe_survival, vocab=dict(survival=SURVIVAL_NAMES))
MINIMAP = Modality("minimap", spaces.Box(0, 255, (32, 32), np.uint8), _describe_minimap)
GOAL = Modality("goal", spaces.Box(-np.inf, np.inf, (15,), np.float32), _describe_goal)
CONSTRUCTION = Modality("construction", spaces.Box(-np.inf, np.inf, (MAX_PARTS, len(PART_COLUMNS)), np.float32),
                        _describe_construction, vocab=dict(construction=PART_COLUMNS))
BUILD_STATE = Modality("build_state", spaces.Box(-np.inf, np.inf, (len(BUILD_STATE_NAMES),), np.float32),
                       _describe_build_state, vocab=dict(build_state=BUILD_STATE_NAMES))

#: canonical order of ``modality_mask`` in schema version 2
MODALITY_SPECS = (PARTY, INVENTORY, SURVIVAL, MINIMAP, GOAL, CONSTRUCTION, BUILD_STATE)

CONTROLS = (
    Control("move", spaces.Box(-1, 1, (2,), np.float32), "forward, right in [-1, 1], view-relative"),
    Control("look", spaces.Box(-1, 1, (2,), np.float32), "turn right, look up in [-1, 1]"),
    Control("buttons", spaces.MultiBinary(len(BUTTONS)), "one bit per name in BUTTONS", vocab=BUTTONS),
    Control("command", spaces.Discrete(len(COMMANDS)), "index into COMMANDS; one per decision", vocab=COMMANDS),
    Control("swap", spaces.Discrete(5), "0 keeps the character; 1-4 selects a party slot"),
    Control("slot", spaces.Discrete(32), "zero-based inventory slot for eat/equip"),
    Control("ingredients", spaces.MultiDiscrete([33] * 5), "0 unused; 1-32 selects inventory slot 0-31"),
    Control("part", spaces.Discrete(MAX_PARTS + 1), "Skylands part id; 0 is the nearest"),
    Control("build_move", spaces.Box(-1, 1, (3,), np.float32), "Skylands held-anchor shift [east, up, south]"),
)


def register_builtins(registry: SpaceRegistry) -> None:
    for control in CONTROLS:
        registry.register_control(control)
    for modality in MODALITY_SPECS:
        registry.register_modality(modality)
