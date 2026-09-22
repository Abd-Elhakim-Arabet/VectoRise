# Deploy — vectoriseai.com on a single VPS (site + tool together)

One box runs everything: `server.py` already serves the landing page, the
showcase video, and the conversion API from a single process.

Pick a lane:

| Lane | Cost | Notes |
|---|---|---|
| **A. VPS + Caddy** (below) | $0 Oracle / ~€4 Hetzner | Always on, full CPU, you own it |
| **B. Render free** (repo-root `render.yaml`) | $0, no card | Sleeps 15 min idle (~1 min wake), 512 MB RAM, ephemeral disk (fine — jobs are temp), 750 h/mo. Custom domain + TLS free. Nobody sees `onrender.com` once your domain points at it. |
| **C. This Mac + tunnel** (current) | $0 | Fastest conversions (your CPU beats free tiers ~10×), but sleeps with the Mac |

Vercel/Netlify-style serverless is a deliberate non-option: functions are
stateless with hard timeouts, while the converter keeps jobs in memory and
on local disk for minutes — this app needs a real server (any lane above).

## Lane A: VPS + Caddy

```
visitor → Caddy (:80/:443, auto-TLS) → app:8000 (this repo, Docker)
```

## 0. What you buy / create (only steps needing your accounts)

1. **Domain** — buy `vectoriseai.com` (~$10–14/yr, Cloudflare or Porkbun).
   DNS: `A @ → <server IP>` and `A www → <server IP>`.
2. **VM** — Oracle Always Free, shape `VM.Standard.A1.Flex` (ARM),
   2 OCPUs / 12 GB RAM, Ubuntu 24.04 Minimal. All core deps (numpy,
   opencv, vtracer, …) ship ARM wheels — verified.

## 1. VM prep (Ubuntu)

```bash
# Docker
sudo apt-get update && sudo apt-get install -y docker.io docker-compose-plugin
sudo usermod -aG docker $USER && newgrp docker

# Open HTTP/HTTPS in BOTH places on Oracle:
#  a) OCI console: subnet security list → ingress TCP 80 + 443 from 0.0.0.0/0
#  b) on the VM: Ubuntu's firewall usually allows all out of the box;
#     if you enabled ufw:  sudo ufw allow 80,443/tcp
```

## 2. Ship + run

```bash
git clone https://github.com/Abd-Elhakim-Arabet/VectoRise.git
cd VectoRise
# Showcase clip is git-ignored: copy yours in (needs faststart moov!)
ffmpeg -y -i <clip>.mp4 -c copy -movflags +faststart \
  apps/web/static/assets/site-view.mp4
cd deploy && docker compose up -d --build
docker compose ps          # app + caddy both "running"
curl -s -o /dev/null -w "%{http_code}\n" http://localhost:8000/
```

Open `https://vectoriseai.com` — Caddy provisions the cert on first visit
(needs the DNS records from step 0 to resolve first).

## 3. Operate

```bash
docker compose logs -f app      # conversion errors land on stderr here
docker compose pull && docker compose up -d --build   # update
```

Public-facing knobs (already set stricter in `compose.yml`, defaults in
`server.py` are the local/development values):

| Variable | Compose | Default | Meaning |
|---|---|---|---|
| `VECTORISE_RATE_LIMIT_N` | 5 | 10 | uploads / 10 min / IP |
| `VECTORISE_MAX_QUEUED` | 2 | 4 | queued jobs before `503 busy` |
| `VECTORISE_CONVERT_TIMEOUT_SEC` | 300 | 300 | per-job wall clock |

Uploads cap at 10 MB / ~3 s, job files auto-delete after 30 min.

## Lane B: Render free (standby fallback)

Dashboard → New → Blueprint → select the repo (auto-reads root
`render.yaml`) → Apply with the free plan. You get a
`vectorise-web-xxxx.onrender.com` URL — that stays the fallback address;
only point `vectoriseai.com` at it when you want traffic there instead of
the tunnel (Lane C). First deploy takes several minutes (pip + ffmpeg
layer); the Dockerfile honors Render's `$PORT`.
