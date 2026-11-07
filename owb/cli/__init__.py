"""The ``owb`` command line: manifest-driven runs plus replay/render/listing subcommands.

``Manifest`` (``owb.cli.manifest``) is the run configuration; ``run_manifest``
(``owb.cli.run``) executes it with ``owb.eval.run_episode``. See docs/cli.md.
"""
from .manifest import Manifest, ManifestError
from .run import RunError, run_manifest

__all__ = ["Manifest", "ManifestError", "RunError", "run_manifest"]
