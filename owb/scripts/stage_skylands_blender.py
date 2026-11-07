"""Stage the pinned official Blender bundle on a compute node's local disk."""
import hashlib
import json
from pathlib import Path
import sys
import tarfile
import urllib.request

ARCHIVE='blender-4.5.4-linux-x64.tar.xz'
URL='https://download.blender.org/release/Blender4.5/'+ARCHIVE
SHA256='2e6ef8e99fc36327270429ddc8e7bad2859dd878a5a137d2e0bf0f02f6792505'


def main():
    root=Path(sys.argv[1]);root.mkdir(parents=True,exist_ok=True)
    archive=root/ARCHIVE;binary=root/'blender-4.5.4-linux-x64/blender';record=root/'staging.json'
    if not record.exists():
        print('Staging pinned Blender on node-local disk',file=sys.stderr,flush=True)
        with urllib.request.urlopen(URL,timeout=60) as source,archive.open('wb') as out:
            while block:=source.read(1024*1024):out.write(block)
        if hashlib.sha256(archive.read_bytes()).hexdigest()!=SHA256:raise RuntimeError('Blender archive checksum mismatch')
        with tarfile.open(archive) as source:source.extractall(root,filter='data')
        record.write_text(json.dumps(dict(url=URL,archive_sha256=SHA256,binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest()),indent=2))
    if hashlib.sha256(binary.read_bytes()).hexdigest()!=json.loads(record.read_text())['binary_sha256']:raise RuntimeError('Staged Blender changed')
    print(binary)


if __name__=='__main__':main()
