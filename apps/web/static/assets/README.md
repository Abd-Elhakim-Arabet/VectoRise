# Web assets

`site-view.mp4` — the seamless loop shown in the `#reel` section between
the explaining and demo sections (`index.html`, served at
`/assets/site-view.mp4` with HTTP Range support).

The file is committed on purpose: git-based deploys (Render, VPS-from-clone)
bake it into the image — a missing clip would 404 the showcase section.
To replace it with any clip, remux with the `moov` atom at the front so
browsers start playback without downloading the whole file first:

```bash
ffmpeg -y -i <clip>.mp4 -c copy -movflags +faststart \
  apps/web/static/assets/site-view.mp4
```

Keep it small (a few MB, a few seconds) — it autoplays on every page load.
