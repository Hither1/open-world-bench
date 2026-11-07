"""owb.core: shared helpers, and its old voxel-core submodule paths keep aliasing."""
import subprocess
import sys

import numpy as np

import owb.core
from owb.core import SCHEMA_VERSION, event_type, jsonable


def test_single_jsonable_and_schema_version_are_reexported():
    from owb.bench import framework, rollout
    assert framework.jsonable is rollout.jsonable is jsonable
    assert framework.SCHEMA_VERSION == SCHEMA_VERSION == 2
    value = {1: np.float32(0.5), "a": (np.arange(2), [np.int8(3)])}
    assert jsonable(value) == {"1": 0.5, "a": [[0, 1], [3]]}


def test_event_type_accepts_teyvat_events_and_dicts():
    from owb.sims.teyvat.events import Event
    assert event_type(Event("talk", {"npc": "x"})) == "talk"
    assert event_type(dict(type="cook", item=1)) == "cook"


def test_legacy_core_submodules_alias_voxel_core_with_one_warning():
    code = """
import warnings
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    import owb.core
    from owb.core import jsonable
    assert not caught, [str(w.message) for w in caught]
    import owb.sims.voxel.core.config as config, owb.sims.voxel.core.rng as rng
    assert owb.core.config is config
    import owb.core.rng
    assert owb.core.rng is rng
print(*[str(w.message).split()[0] for w in caught if w.category is DeprecationWarning])
"""
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.split() == ["owb.core"]
