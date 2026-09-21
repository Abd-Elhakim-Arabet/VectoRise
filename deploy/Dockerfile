# VectoRise web — production image.
# Pinned to Python 3.12: apps/web/server.py uses `cgi` (removed in 3.13+).
# Multi-arch (amd64/arm64): all core deps ship ARM wheels, incl. vtracer.
FROM python:3.12-slim-bookworm

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv/vectorise

# Core engine first (deps layer changes rarely).
COPY packages/core/pyproject.toml packages/core/
COPY packages/core/src packages/core/src
RUN pip install --no-cache-dir ./packages/core

# Web app (stdlib-only server + static UI).
COPY apps/web apps/web

RUN useradd -m -u 10001 app \
    && mkdir -p /tmp/vectorise-web-jobs \
    && chown app:app /tmp/vectorise-web-jobs
USER app

EXPOSE 8000
# NOTE: 0.0.0.0 is correct here — TLS + internet filtering happen in Caddy
# in front of this container (see Caddyfile). Never expose 8000 directly.
CMD ["python", "apps/web/server.py", "--host", "0.0.0.0", "--port", "8000"]
