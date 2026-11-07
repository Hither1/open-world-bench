"""Privileged reference/oracle policies (may read simulator state; not benchmark agents).

* :mod:`.teyvat_reference` -- ``ReferencePolicy`` for the Teyvat reference courses A/B;
* :mod:`.wildlands_reference` -- ``WildlandsReferencePolicy`` for the Wildlands courses A/B.

The Skylands scripted references still live in ``owb.bench.baselines``. Anything in this
package must report ``privileged=True`` from ``metadata()``. Submodules are not imported
here so that importing the package stays cheap.
"""
PRIVILEGED = True
