"""Stage the pinned official Blender bundle on a compute node's local disk.

The checksummed archive is kept once in a mirror on lab storage
(``.cache/render-tools`` by default, or ``OWB_BLENDER_MIRROR``), so jobs do not
download it from blender.org each time; one large sequential read replaces a
download and thousands of small-file reads from network storage. The archive
SHA-256 is checked on every use, so provenance matches a direct download.

Usage: python owb/scripts/stage_blender.py LOCAL_DIR   (prints the binary path)
       python owb/scripts/stage_blender.py --mirror-only
"""
import hashlib
import json
import os
from pathlib import Path
import sys
import tarfile
import urllib.request

ARCHIVE = 'blender-4.5.4-linux-x64.tar.xz'
URL = 'https://download.blender.org/release/Blender4.5/' + ARCHIVE
SHA256 = '2e6ef8e99fc36327270429ddc8e7bad2859dd878a5a137d2e0bf0f02f6792505'
ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while block := f.read(8 << 20):
            h.update(block)
    return h.hexdigest()


def mirrored_archive():
    mirror = Path(os.environ.get('OWB_BLENDER_MIRROR', ROOT / '.cache/render-tools'))
    archive = mirror / ARCHIVE
    if not archive.exists():
        mirror.mkdir(parents=True, exist_ok=True)
        partial = archive.with_name(f'{ARCHIVE}.partial-{os.getpid()}')
        print('Downloading pinned Blender into the mirror', file=sys.stderr, flush=True)
        with urllib.request.urlopen(URL, timeout=60) as source, partial.open('wb') as out:
            while block := source.read(1 << 20):
                out.write(block)
        if digest(partial) != SHA256:
            partial.unlink()
            raise RuntimeError('Blender archive checksum mismatch')
        partial.replace(archive)
    return archive


def stage(root):
    root.mkdir(parents=True, exist_ok=True)
    binary = root / 'blender-4.5.4-linux-x64/blender'
    record = root / 'staging.json'
    if not record.exists():
        archive = mirrored_archive()
        if digest(archive) != SHA256:
            raise RuntimeError(f'Mirrored Blender archive checksum mismatch: {archive}')
        print('Staging pinned Blender on node-local disk', file=sys.stderr, flush=True)
        with tarfile.open(archive) as source:
            source.extractall(root, filter='data')
        record.write_text(json.dumps(dict(url=URL, archive_sha256=SHA256, binary_sha256=digest(binary)), indent=2))
    if digest(binary) != json.loads(record.read_text())['binary_sha256']:
        raise RuntimeError('Staged Blender changed')
    return binary


def main():
    if sys.argv[1:] == ['--mirror-only']:
        print(mirrored_archive())
    else:
        print(stage(Path(sys.argv[1])))


if __name__ == '__main__':
    main()
