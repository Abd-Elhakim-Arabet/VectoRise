# Web assets (local only, git-ignored)

`site-view.mp4` — the seamless loop shown in the `#reel` section between
the explaining and demo sections (`index.html`, served at
`/assets/site-view.mp4` with HTTP Range support).

The file is **not** committed (see root `.gitignore`). To recreate it from
any clip, remux with the `moov` atom at the front so browsers start
playback without downloading the whole file first:

```bash
ffmpeg -y -i <clip>.mp4 -c copy -movflags +faststart \
  apps/web/static/assets/site-view.mp4
```

Keep it small (a few MB, a few seconds) — it autoplays on every page load.
