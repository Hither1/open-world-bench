"""Publish existing recordings without re-rendering or re-encoding their video.

One-time asset preparation, not part of the Pages build. Skylands is remuxed at
keyframes into files below GitHub's per-file limit. Encoded packet hashes must
match, in order, before a manifest is written. Wildlands archives are copied
byte-for-byte with their original chapter/renderer receipts.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(path.read_text())


def run(ffmpeg, *args):
    return subprocess.check_output([ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", *map(str, args)], text=True)


def packets(ffmpeg, path):
    output = run(ffmpeg, "-i", path, "-map", "0:v:0", "-c:v", "copy", "-f", "framehash", "-hash", "sha256", "-")
    return [line.split(",")[-1].strip() for line in output.splitlines() if line and not line.startswith("#")]


def poster(ffmpeg, source, target):
    if not target.exists():
        run(ffmpeg, "-n", "-i", source, "-frames:v", 1, "-update", 1, target)


def prepare(ffmpeg):
    for name in ("a", "b"):
        source = ROOT / f"assets/demos/skylands-campaign/task-{name}-seed-7"
        video = source / "walkthrough.mp4"
        receipt = read(source / "video-manifest.json")
        assert digest(video) == receipt["video_sha256"], f"Source hash mismatch: {video}"
        output = ROOT / "videos/skylands/reference-legacy"
        output.mkdir(parents=True, exist_ok=True)
        manifest = output / f"task-{name}.json"
        if manifest.exists():
            continue
        listing = output / f"task-{name}-segments.csv"
        run(ffmpeg, "-n", "-i", video, "-map", "0:v:0", "-c", "copy", "-f", "segment",
            "-segment_time", "180", "-reset_timestamps", "1", "-segment_start_number", "1",
            "-segment_format", "mp4", "-segment_format_options", "movflags=+faststart",
            "-segment_list", listing, "-segment_list_type", "csv", output / f"task-{name}-part-%02d.mp4")
        parts, published_packets = [], []
        for filename, start, end in csv.reader(listing.read_text().splitlines()):
            path = output / Path(filename).name
            assert path.stat().st_size < 95 * 1024 * 1024, f"Part too large: {path}"
            hashes = packets(ffmpeg, path)
            published_packets.extend(hashes)
            parts.append(dict(path=str(path.relative_to(ROOT)), start=float(start), end=float(end),
                              frames=len(hashes), bytes=path.stat().st_size, sha256=digest(path)))
        original_packets = packets(ffmpeg, video)
        assert original_packets == published_packets, "Remux changed, lost or reordered video packets"
        assert len(original_packets) == receipt["frames"]
        image = output / f"task-{name}-poster.png"
        poster(ffmpeg, video, image)
        data = dict(source=str(video.relative_to(ROOT)), source_sha256=digest(video),
                    source_manifest=str((source / "video-manifest.json").relative_to(ROOT)),
                    processing="FFmpeg stream copy at keyframes; no video re-encoding or rescaling",
                    encoded_packet_hashes_match=True, frames=len(original_packets),
                    duration=receipt["duration_seconds"], renderer=receipt["renderer"],
                    encoded_size=receipt["encoded_size"], source_render_size=receipt["source_render_size"],
                    poster=str(image.relative_to(ROOT)), parts=parts,
                    chapters=read(source / "chapters.json"),
                    scope="Earlier rectangular-world campaign, before organic-world and working-memory-probe updates.")
        manifest.write_text(json.dumps(data, indent=2) + "\n")
        listing.unlink()
        print(f"Skylands {name}: {len(parts)} parts, {len(original_packets)} unchanged encoded frames", flush=True)

    for name in ("a", "b"):
        source = ROOT / f".cache/wildlands-published-before-20261115-130716/task-{name}"
        receipt = read(source / "summary.json")
        video = source / "walkthrough.mp4"
        assert digest(video) == receipt["video_sha256"]
        assert receipt["presentation"]["backend"] == "cycles"
        output = ROOT / "videos/wildlands/reference-legacy"
        output.mkdir(parents=True, exist_ok=True)
        target = output / f"task-{name}.mp4"
        if not target.exists():
            shutil.copyfile(video, target)
        assert digest(target) == receipt["video_sha256"]
        image = output / f"task-{name}-poster.png"
        poster(ffmpeg, target, image)
        data = dict(source_archive=str(source.relative_to(ROOT)), source_sha256=receipt["video_sha256"],
                    video=str(target.relative_to(ROOT)), poster=str(image.relative_to(ROOT)),
                    processing="Byte-for-byte copy of the archived Cycles recording",
                    duration=receipt["video_seconds"], chapters=read(source / "chapters.json"),
                    summary=receipt,
                    scope="Compact checkpoint walkthrough: selected replay states held on screen, not continuous gameplay. Predates the organic-world and working-memory-probe updates.")
        (output / f"task-{name}.json").write_text(json.dumps(data, indent=2) + "\n")
        print(f"Wildlands {name}: restored original Cycles checkpoint walkthrough", flush=True)

    # Exact first frames of the other recordings; no poster rescaling.
    for filename, destination in [
        ("assets/demos/teyvat-reference-cycles/task-a-seed-7/walkthrough.mp4", "videos/teyvat/reference-posters/task-a.png"),
        ("assets/demos/teyvat-reference-cycles/task-b-seed-7/walkthrough.mp4", "videos/teyvat/reference-posters/task-b.png"),
        ("videos/teyvat/teyvat-fullscale-24-milestones.mp4", "videos/teyvat/reference-posters/fullscale-b.png"),
        ("videos/clips/wildlands-a-preview-clip.mp4", "videos/wildlands/reference-legacy/task-a-excerpt-poster.png"),
        ("videos/clips/skylands-b-preview-clip.mp4", "videos/skylands/reference-legacy/task-b-excerpt-poster.png"),
        ("videos/minecraft/ender-dragon-end-clip.mp4", "videos/minecraft/ender-dragon-poster.png"),
        ("videos/lemon-tart/lemon-tart-full.mp4", "videos/lemon-tart/lemon-tart-full-poster.png"),
    ]:
        target = ROOT / destination
        target.parent.mkdir(parents=True, exist_ok=True)
        poster(ffmpeg, ROOT / filename, target)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ffmpeg", required=True)
    prepare(parser.parse_args().ffmpeg)
