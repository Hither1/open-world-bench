# Project page

Source for the GitHub Pages site at https://hither1.github.io/open-world-bench/.

- `index.html`, `style.css`, `main.js`: the page (static, no JavaScript framework).
- `task-explorer.js`, `task-explorer.css`: the five-environment overview and
  interactive task diagrams. Task, stage and milestone selections have shareable
  `#explore/<environment>/<task>/<stage>/<milestone>` URLs. Stages support keyboard
  navigation, and diagrams scroll horizontally on small screens.
- `build_task_data.py`: groups the 15 documented tasks into diagram stages and
  reads milestone descriptions from `docs/`. It generates `_site/task-data.js`
  using Python 3's standard library. The build rejects missing, duplicate or
  unassigned milestone IDs. Edit the source documents for milestone text and
  this file for stage groupings, branches and memory connections.
- `leaderboard-data.js`, `leaderboard.js`: published result data and the
  interactive leaderboard, with condition/world filters and model details.
- `build.sh`: assembles `_site/` by copying the page plus the GIFs, Cycles stills
  and MP4 clips it uses from `assets/` and `videos/`, so media is not duplicated.
- `../.github/workflows/pages.yml`: runs `build.sh` and deploys on pushes to `main`
  that touch the page, task documents or media.

Preview locally:

```bash
bash website/build.sh
python -m http.server -d _site 8000   # open http://localhost:8000
```

Example task link: `http://localhost:8000/#explore/skylands/three-layer-lifeline`.
The explorer is a presentation of task designs, not simulated progress. Its
labels distinguish playable campaigns, legacy courses and unimplemented designs.
Solid edges show the documented structure (or a reference itinerary); dashed
edges show information and procedures carried between stages. The task-specific
note explains branch order and optional routes. Existing demonstration videos,
renderer metadata, the leaderboard and task catalogue remain available below.

To feature a different clip, update the `data-src`, link and `<img>` in
`index.html` and the matching `cp` line in `build.sh`. Keep the renderer and
settings in the caption (see AGENTS.md).

All five worlds use the covers in [`assets/covers`](../assets/covers/README.md),
with WebP sources and PNG fallbacks. Teyvat, Wildlands, and Skylands use Cycles;
Minecraft and mobile manipulation retain their native game/MuJoCo captures.
Their concise descriptions and alt text are collected in
[`copy.json`](../assets/covers/copy.json). Keep the cover caption separate from
the video caption: the cover and clip can show different courses.

## Leaderboard

The leaderboard borrows the horizontal ranking chart, segmented filters, and
world-by-world comparison from [Embodied First Exam](https://www.embodied-first-exam.ai/leaderboard/),
using OWB's existing typography and palette. It is part of the existing GitHub
Pages build; no separate hosting or dependencies are needed.

Edit `leaderboard-data.js` to publish results. The three planned models match the
repository README. Their `results` objects are deliberately empty: demonstrations
and privileged reference solutions are not model evaluations. Missing results
display as pending/unscored, with no numeric rank or filled score bar. An actual
published zero-success evaluation displays as 0%.

Each model's `results` is keyed first by information condition (`native`,
`perception`, `privileged`), then by world (`minecraft`, `teyvat`, `wildlands`,
`skylands`, `mobile`) or `overall`. A result has integer `successes` and `episodes`
fields, with `0 <= successes <= episodes` and `episodes > 0`, plus a `reportUrl`
link to its published evaluation report. The chart displays
`100 * successes / episodes`, sorted descending, with tied scores sharing a rank.
Missing or invalid counts never become zero scores.

Before adding a result, its report must identify the suite/version, task set,
seeds, episode counts and exclusions, information condition, harness/model
configuration, and action/model-call budgets. Compare only matching protocols
within a condition and world. Leave tasks with no success criterion unscored.
`overall` must be an explicitly published aggregate with a documented protocol;
the UI never averages partial world results or mixes information conditions.

The static HTML comparison is also a no-JavaScript fallback. When publishing
results or changing the model roster, update that fallback table in `index.html`
as well. The normal interactive table, bars, counts, and detail panels all read
from `leaderboard-data.js`.
