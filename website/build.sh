#!/usr/bin/env bash
# Assemble the GitHub Pages site into an output directory (default: _site).
#
# The page source lives in website/. Media is copied from the repository at
# build time so GIFs, screenshots and MP4 clips are never duplicated in git.
#
#   bash website/build.sh            # -> _site/
#   bash website/build.sh /tmp/owb   # -> /tmp/owb/
#   python -m http.server -d _site 8000   # local preview
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:-$ROOT/_site}"

rm -rf "$OUT"
mkdir -p "$OUT/media/gif" "$OUT/media/img" "$OUT/media/video"

cp "$ROOT/website/index.html" "$ROOT/website/style.css" "$ROOT/website/main.js" "$OUT/"
touch "$OUT/.nojekyll"

# Animated previews (8 s, 480 px) cut from the MP4s below.
cp "$ROOT"/assets/demos/readme/{teyvat,skylands,wildlands,minecraft-ender-dragon,lemon-tart}.gif "$OUT/media/gif/"

# Blender Cycles stills (seed 7).
cp "$ROOT"/assets/screenshots/{teyvat-diverse-cycles,wildlands-blender}.png "$OUT/media/img/"

# Full clips. All are H.264 with fast-start metadata, so they stream in browsers.
cp "$ROOT/videos/teyvat/teyvat-fullscale-24-milestones.mp4" \
   "$ROOT/videos/clips/skylands-b-preview-clip.mp4" \
   "$ROOT/videos/clips/wildlands-a-preview-clip.mp4" \
   "$ROOT/videos/minecraft/ender-dragon-end-clip.mp4" \
   "$ROOT/videos/lemon-tart/lemon-tart-full.mp4" \
   "$OUT/media/video/"

echo "Built site in $OUT ($(du -sh "$OUT" | cut -f1))"
