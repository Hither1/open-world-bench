"""Privileged reference controllers live in owb.agents.oracle; old paths re-export them."""
import subprocess
import sys


def _fresh(code):
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    return out.stdout.split()


def test_old_oracle_paths_reexport_with_a_deprecation_warning():
    code = """
import warnings
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    from owb.agents.oracle.teyvat_reference import ReferencePolicy
    from owb.agents.oracle.wildlands_reference import WildlandsReferencePolicy, neutral
    assert not [w for w in caught if w.category is DeprecationWarning], [str(w.message) for w in caught]
    from owb.sims.teyvat.reference_policy import ReferencePolicy as OldTeyvat
    from owb.bench.baselines.wildlands_reference import WildlandsReferencePolicy as OldWild, neutral as old_neutral
    import open_world_bench.reference_policy as alias
assert OldTeyvat is ReferencePolicy and alias.ReferencePolicy is ReferencePolicy
assert OldWild is WildlandsReferencePolicy and old_neutral is neutral
print(*sorted(str(w.message).split()[0] for w in caught if w.category is DeprecationWarning))
"""
    assert _fresh(code) == ["open_world_bench", "owb.bench.baselines.wildlands_reference",
                            "owb.sims.teyvat.reference_policy"]


def test_oracles_report_privileged_metadata():
    from owb.agents.oracle.teyvat_reference import ReferencePolicy
    from owb.agents.oracle.wildlands_reference import WildlandsReferencePolicy

    class Env:
        task = "a"

    for cls in (ReferencePolicy, WildlandsReferencePolicy):
        meta = cls(Env()).metadata()
        assert meta["privileged"] is True and meta["type"] == "oracle" and meta["task"] == "a"
