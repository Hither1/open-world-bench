"""Audit stored campaign evidence and measure delayed use from actual traces."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET

from owb.scripts.skylands_reference import save, sha
from owb.scripts.skylands_restoration import sources


def report(root):
    runs = []
    current_sources = sources()
    for path in sorted(root.glob("*/reference-manifest.json")):
        directory = path.parent
        manifest = json.loads(path.read_text())
        verification = json.loads((directory/"verification.json").read_text())
        if manifest["sources"] != current_sources or verification["sources"] != current_sources:
            raise ValueError(f"Source mismatch: {directory}")
        if sha(directory/"actions.jsonl") != manifest["actions_sha256"]:
            raise ValueError(f"Trace mismatch: {directory}")
        if (not verification["success"] or not verification["final_live_predicate"]
                or verification["final_digest"] != manifest["final_digest"]):
            raise ValueError(f"Unverified final state: {directory}")
        first_read, first_used, cached, recovered = {}, {}, None, None
        categories = Counter()
        with (directory/"actions.jsonl").open() as stream:
            for line in stream:
                row = json.loads(line)
                categories[str(row["action"]["command"])] += 1
                for event in row["events"]:
                    if event["type"] == "clue_exposed":
                        for name in re.findall(r"(Cooling|Anchors|Monitoring) service profile:", event["text"]):
                            first_read.setdefault(name.lower(), row["step"])
                    elif event["type"] == "service_dial_set":
                        first_used.setdefault(event["region"], row["step"])
                    elif event["type"] == "tooling_moved":
                        if event["operation"] == "cache" and cached is None:
                            cached = row["step"]
                        elif event["operation"] == "retrieve" and recovered is None:
                            recovered = row["step"]
        delays = {name: dict(read_step=first_read[name], use_step=step,
                            intervening_actions=step-first_read[name], simulated_seconds=(step-first_read[name])*.2)
                  for name, step in first_used.items() if name in first_read}
        e = manifest["evaluation"]
        runs.append(dict(directory=directory.name, objective=manifest["objective"], seed=manifest["seed"],
            shutdown_method=e["engine_route"], regional_order=manifest["order"],
            success=e["success"], deaths=e["deaths"], actions=e["decisions"],
            simulated_seconds=e["simulated_seconds"], checkpoints=len(e["campaign_milestones"]),
            live_regions=e["live_regions"], manifold_live=e["manifold_live"],
            refuge_stable_seconds=e["refuge_stable_seconds"], profile_reuse_delays=delays,
            tooling_cache_delay_actions=recovered-cached if recovered is not None and cached is not None else None,
            actions_sha256=manifest["actions_sha256"], manifest_sha256=sha(path),
            verification_sha256=sha(directory/"verification.json"), action_command_counts=dict(categories)))
    if not runs:
        raise ValueError("No verified runs found")
    test_path = root/"tests.xml"
    xml = ET.parse(test_path).getroot()
    suites = list(xml.iter("testsuite"))
    tests = {key: sum(int(s.get(key, 0)) for s in suites) for key in ("tests", "errors", "failures", "skipped")}
    if tests["errors"] or tests["failures"]:
        raise ValueError("Regression tests failed")
    tests["sha256"] = sha(test_path)
    result = dict(format="owb-restoration-validation-v1", success=True, source_hashes_match=True,
                  tests=tests, runs=runs,
                  limitations=["Scripted references; no model scores or memory-ablation claim.",
                               "Measured itineraries, not minimum successful horizon bounds.",
                               "Seeded service profiles in authored geometry; finite settings can be searched."])
    save(root/"validation-summary.json", result)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--directory", type=Path, default=Path("assets/demos/skylands-restoration"))
    args = p.parse_args()
    print(json.dumps(report(args.directory), indent=2))
