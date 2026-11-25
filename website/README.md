# Project page

Source for the GitHub Pages site at https://hither1.github.io/open-world-bench/.

- `index.html`, `style.css`, `main.js`: the page (static, no build tools).
- `build.sh`: assembles `_site/` by copying the page plus the GIFs, Cycles stills
  and MP4 clips it uses from `assets/` and `videos/`, so media is not duplicated.
- `../.github/workflows/pages.yml`: runs `build.sh` and deploys on pushes to `main`
  that touch the page or its media.

Preview locally:

```bash
bash website/build.sh
python -m http.server -d _site 8000   # open http://localhost:8000
```

To feature a different clip, update the `data-src`, link and `<img>` in
`index.html` and the matching `cp` line in `build.sh`. Keep the renderer and
settings in the caption (see AGENTS.md).
