"""Extract authentic Minecraft and MuJoCo covers, without resizing or enhancement.

python -B assets/covers/extract_native.py NEW_OUTPUT_DIRECTORY [--ffmpeg PATH]
"""
import argparse
import json
from pathlib import Path
import shutil
import subprocess

from export import ROOT, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--ffmpeg", default=shutil.which("ffmpeg") or str(ROOT / ".cache/render-tools/ffmpeg"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    ffmpeg = str(Path(args.ffmpeg).resolve())
    version = subprocess.check_output([ffmpeg, "-version"], text=True).splitlines()[0]
    source = ROOT / "assets/covers/sources/minecraft-step-0000001.jpg"
    video = ROOT / "videos/lemon-tart/lemon-tart-full.mp4"
    # The MCU wire JPEG has swapped red/blue channels. This restores the
    # original RGB order, following worldmodel_purple_agent._decode_image.
    # It is a decoding correction, with no gamma or brightness adjustment.
    commands = {
        "minecraft": ["-i", str(source), "-vf", "colorchannelmixer=rr=0:rb=1:br=1:bb=0"],
        # Remove the 70-pixel caption band and retain an exact 16:9 pixel crop.
        "mobile-manipulation": ["-ss", "320", "-i", str(video), "-vf", "crop=1152:648:64:72"],
    }
    for name, command in commands.items():
        subprocess.run([ffmpeg, "-hide_banner", "-loglevel", "error", *command,
                        "-frames:v", "1", str(args.output / f"{name}.png")], check=True)
    records = {
        "minecraft": dict(
            scope="Mine Diamond from Scratch, early Overworld frame, step 1",
            renderer="Minecraft client via MineStudio / MCU", cycles_exporter=False,
            source="assets/covers/sources/minecraft-step-0000001.jpg", source_sha256=sha(source),
            original_source="MFM-abl/MCU/output/abl_opus_diamond_flat/episode_001/hires/0000001.jpg",
            source_dimensions=[640, 360], output_dimensions=[640, 360],
            decoding="Restore RGB order from the channel-reversed MCU wire JPEG",
            decoding_reference="MFM-abl/MCU/worldmodel_purple_agent.py::_decode_image",
            cropped=False, upscaled=False, hud_retained=True, brightness_adjusted=False,
            task_performance_claim=False),
        "mobile-manipulation": dict(
            scope="PandaOmron lemon-tart reference run, video timestamp 05:20",
            renderer="MuJoCo native offscreen renderer", cycles_exporter=False,
            source="videos/lemon-tart/lemon-tart-full.mp4", source_sha256=sha(video),
            source_record="videos/lemon-tart/lemon-tart-full.json", seed=1,
            source_dimensions=[1280, 720], output_dimensions=[1152, 648],
            timestamp_seconds=320, crop=dict(x=64, y=72, width=1152, height=648),
            crop_purpose="Remove the video caption while preserving the scene pixels",
            upscaled=False, brightness_adjusted=False, scene_content_replaced=False,
            source_run_success=False, source_run_milestones="20/21",
            task_performance_claim=False),
    }
    for name, record in records.items():
        record.update(extractor=version, extraction_script_sha256=sha(Path(__file__)),
                      image_sha256=sha(args.output / f"{name}.png"))
    (args.output / "sources.json").write_text(json.dumps(records, indent=2) + "\n")


if __name__ == "__main__":
    main()
