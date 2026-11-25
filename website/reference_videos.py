"""Task-to-recording catalogue. Only verified, present media becomes a play link."""
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
SHARED = {
    f"videos/{path}": f"media/video/{Path(path).name}" for path in (
        "teyvat/teyvat-fullscale-24-milestones.mp4", "clips/skylands-b-preview-clip.mp4",
        "clips/wildlands-a-preview-clip.mp4", "minecraft/ender-dragon-end-clip.mp4",
        "lemon-tart/lemon-tart-full.mp4",
    )
}


def catalogue(tasks, output=None):
    copied = set()

    def asset(path):
        source = ROOT / path
        if not source.is_file():
            raise ValueError(f"Recording asset missing: {path}")
        public = SHARED.get(path, f"media/reference/{path}")
        if output and public not in copied:
            target = output / public
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            copied.add(public)
        return public

    def read(path):
        return json.loads((ROOT / path).read_text())

    def chapters(path):
        return [dict(id=c["id"], title=c["title"], time=c.get("start_seconds", c.get("video_seconds"))) for c in read(path)]

    def video(ident, label, source, poster, duration, credit, renderer, scope, provenance, entries=()):
        return dict(id=ident, label=label, duration=duration, credit=credit, renderer=renderer,
                    scope=scope, provenance=asset(provenance), poster=asset(poster),
                    parts=[dict(src=asset(source), start=0, end=duration)], chapters=list(entries))

    missing = {
        "frostfall-spire": ("Reference traces are verified, but this branching campaign has not been rendered yet.", "docs/teyvat-campaign-validation.md"),
        "fivefold-cleansing": ("Reference traces are verified, but this branching campaign has not been rendered yet.", "docs/teyvat-campaign-validation.md"),
        "zenith-archive": ("Reference traces are verified, but this branching campaign has not been rendered yet.", "docs/teyvat-campaign-validation.md"),
        "tempest-engine": ("Verified reference runs and Cycles stills exist; a reference video has not been published yet.", "docs/skylands-tempest.md"),
        "tempest-restoration": ("Verified reference runs exist, but this campaign has not been rendered as a video yet.", "docs/skylands-restoration.md"),
        "mine-diamond": ("No reference recording is available for Mine Diamond from Scratch yet.", "docs/minecraft-long-horizon-tasks.md"),
        "guest-dessert": ("This is a proposed task. Its second room and reference recording are not implemented yet.", "docs/mobile-manipulation-tasks.md"),
    }
    records = {}
    for task_id, letter in (("relay-keeper", "a"), ("three-arts", "b")):
        base = f"assets/demos/teyvat-reference-cycles/task-{letter}-seed-7"
        summary = read(f"{base}/summary.json")
        records[task_id] = [video("compact", "Compact reference walkthrough", f"{base}/walkthrough.mp4",
            f"videos/teyvat/reference-posters/task-{letter}.png", summary["video_seconds"],
            "Privileged reference controller · seed 7 · complete compact course",
            "Blender 4.5.4 Cycles · CUDA GPU · 48 samples · native 864 × 486 scene in a 1280 × 720 annotated frame; no upscaling.",
            "Earlier compact course, before the working-memory probes. This is a reference solution, not a learned-agent or memory-evaluation result.",
            "videos/teyvat/playback-verification.json", chapters(f"{base}/chapters.json"))]
    records["three-arts"].insert(0, video("fullscale", "Full-scale reference draft", "videos/teyvat/teyvat-fullscale-24-milestones.mp4",
        "videos/teyvat/reference-posters/fullscale-b.png", 290.3,
        "Privileged reference controller · seed 7 · 24 milestones · 1,760 m island",
        "Blender 4.5.4 Cycles · OptiX GPU · 64 samples · existing 1280 × 720 delivery copy of the 1600 × 900 annotated recording.",
        "Earlier full-scale draft with edited timelapse and event holds. It predates the working-memory probes and later presentation fixes.",
        "assets/demos/teyvat-fullscale-b/seed-7/compressed-copy.json", chapters("assets/demos/teyvat-fullscale-b/seed-7/chapters.json")))

    for task_id, letter in (("frostbound-pass", "a"), ("valley-before-storm", "b")):
        receipt = f"videos/wildlands/reference-legacy/task-{letter}.json"
        data = read(receipt)
        entries = [dict(id=c["id"], title=c["title"], time=c["start_seconds"]) for c in data["chapters"]]
        records[task_id] = [video("checkpoint", "Complete checkpoint walkthrough", data["video"], data["poster"], data["duration"],
            f"Privileged reference controller · seed 7 · {12 if letter == 'a' else 24} milestones",
            "Blender 4.5.4 Cycles · OptiX GPU · 48 samples · native 960 × 540 scene.",
            "Compact course shown as selected replay states held on screen, not continuous gameplay. This preserved recording predates the organic-world and working-memory-probe updates.", receipt, entries)]
    records["frostbound-pass"].append(video("excerpt", "Continuous-motion excerpt", "videos/clips/wildlands-a-preview-clip.mp4",
        "videos/wildlands/reference-legacy/task-a-excerpt-poster.png", 51.8,
        "Privileged reference controller · seed 7 · A01–A05 and A09–A10 only",
        "Blender 4.5.4 Cycles · OptiX GPU · 48 samples · 960 × 540.",
        "A partial preview of the earlier task, before the working-memory probes. The complete checkpoint walkthrough is available in the recording selector.",
        "videos/clips/wildlands-a-preview-clip.json"))

    for task_id, letter in (("fallen-observatory", "a"), ("three-layer-lifeline", "b")):
        receipt = f"videos/skylands/reference-legacy/task-{letter}.json"
        data = read(receipt)
        records[task_id] = [dict(id="full", label="Complete campaign walkthrough", duration=data["duration"],
            credit=f"Privileged reference controller · seed 7 · {15 if letter == 'a' else 24} milestones",
            renderer="Blender 4.5.4 Cycles · OptiX GPU · 64 samples · native 1280 × 720 scene in a 1600 × 900 annotated frame.",
            scope="Earlier rectangular-world campaign, before the organic-world and working-memory-probe updates. The complete recording plays across consecutive parts with no video re-encoding or resolution change.",
            provenance=asset(receipt), poster=asset(data["poster"]),
            parts=[dict(src=asset(p["path"]), start=p["start"], end=p["end"]) for p in data["parts"]],
            chapters=[dict(id=c["id"], title=c["title"], time=c["video_seconds"]) for c in data["chapters"]])]
    records["three-layer-lifeline"].append(video("excerpt", "Short campaign excerpt", "videos/clips/skylands-b-preview-clip.mp4",
        "videos/skylands/reference-legacy/task-b-excerpt-poster.png", 123.0,
        "Privileged reference controller · seed 7 · B01–B03 and B19–B20 only",
        "Blender 4.5.4 Cycles · OptiX GPU · 64 samples · 1280 × 720 scene in a 1600 × 900 annotated frame.",
        "An excerpt from the earlier rectangular-world campaign. Other milestones are in the complete recording; the current working-memory probes are not shown.",
        "videos/clips/skylands-b-preview-clip.json", chapters("videos/clips/skylands-b-preview-clip-chapters.json")))

    records["ender-dragon"] = [video("model", "Model attempt excerpt · not a reference solution", "videos/minecraft/ender-dragon-end-clip.mp4",
        "videos/minecraft/ender-dragon-poster.png", 26.05,
        "Opus 5 model life · diagnostic diamond-kit start in The End",
        "Native Minecraft capture via MFM / MCU · 640 × 360 source, existing 1280 × 720 upscaled delivery copy · no Cycles exporter.",
        "No successful reference video is available. This model attempt destroys three end crystals; it does not defeat the dragon. It does not show the from-scratch task.",
        "videos/minecraft/ender-dragon-end-clip.json")]
    records["lemon-tart"] = [video("partial", "Continuous procedure run · incomplete", "videos/lemon-tart/lemon-tart-full.mp4",
        "videos/lemon-tart/lemon-tart-full-poster.png", 461.3,
        "Scripted robot controller · seed 1 · 20 of 21 milestones · 6× playback",
        "MuJoCo native offscreen renderer · 1280 × 720 · no Cycles exporter for RoboCasa.",
        "The run fails after 6:41 while trying to remove the whisk from the dishwasher rack. It does not complete tools_restored or the final task; rack placement was changed after this recording.",
        "videos/lemon-tart/lemon-tart-full.json")]

    for task in tasks:
        task_id = task["id"]
        task["recordings"] = records.get(task_id, [])
        if not task["recordings"]:
            reason, source = missing[task_id]  # Every unrecorded task needs an explicit explanation.
            task["recordingUnavailable"] = dict(reason=reason, source=f"https://github.com/Hither1/open-world-bench/blob/main/{source}")
        for recording in task["recordings"]:
            ids = {m["id"] for m in task["milestones"]}
            # Some source chapter lists include a terminal hold; it is not a milestone.
            recording["chapters"] = [c for c in recording["chapters"] if c["id"] in ids]
            entries = recording["chapters"]
            if any(c["time"] is None or not 0 <= c["time"] < recording["duration"] for c in entries):
                raise ValueError(f"Invalid chapter timestamp: {task_id}")
            if len({c["id"] for c in entries}) != len(entries):
                raise ValueError(f"Duplicate chapter: {task_id}")
            if [c["time"] for c in entries] != sorted(c["time"] for c in entries):
                raise ValueError(f"Unordered chapters: {task_id}")
            end = 0
            for part in recording["parts"]:
                if abs(part["start"] - end) > .01 or part["end"] <= part["start"]:
                    raise ValueError(f"Non-contiguous video: {task_id}")
                end = part["end"]
            if abs(end - recording["duration"]) > .01:
                raise ValueError(f"Incomplete recording: {task_id}")
