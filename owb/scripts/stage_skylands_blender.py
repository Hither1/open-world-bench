"""Stage the pinned official Blender bundle on a compute node's local disk.

Kept for the existing Skylands job scripts; staging now comes from the
checksummed lab-storage mirror in ``stage_blender.py`` instead of a download
from blender.org in every job. The archive and binary checks are unchanged.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage_blender import stage  # noqa: E402


if __name__ == '__main__':
    print(stage(Path(sys.argv[1])))
