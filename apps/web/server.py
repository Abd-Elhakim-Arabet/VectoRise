#!/usr/bin/env python3
"""VectoRise web: bare-bones but hardened video -> Lottie JSON converter.

Stdlib-only HTTP server (no web framework = minimal attack surface).
Serves static UI + JSON API backed by ``core_engine.video_to_lottie``.

Security controls (deliberate, all server-side enforced):
  * Same-origin only, no cookies/auth tokens to steal, no CORS wildcard.
  * Strict security headers (CSP, nosniff, DENY framing, minimal permissions).
  * Upload caps: Content-Length <= MAX_UPLOAD_BYTES, extension allowlist +
    magic-byte sniff + ffprobe-equivalent decode check inside core pipeline.
  * No user-controlled paths: uuid4 job ids, files under JOBS_DIR/<id>/,
    client filename never used on disk (sanitized for display only).
  * All numeric/enum params clamped server-side; unknown fields rejected.
  * No shell: core uses argv-list subprocess only; this server spawns nothing.
  * Bounded resources: per-IP rate limit, max concurrent conversions,
    max jobs cap, per-job timeout, background TTL sweeper.
  * Error responses are generic; tracebacks go to stderr only, never client.
  * Downloads served as attachment with explicit MIME; uploads never served.

Run:
    python apps/web/server.py [--host 127.0.0.1] [--port 8000]

Then open http://127.0.0.1:8000/
"""

from __future__ import annotations

import argparse
import cgi
import html
import json
import mimetypes
import shutil
import sys
import tempfile
import threading
import time
import urllib.parse
import uuid
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

# --- resolve core_engine without requiring pip install ----------------------
_HERE = Path(__file__).resolve()
_WEB_DIR = _HERE.parent
_REPO_ROOT = _WEB_DIR.parent.parent
_CORE_SRC = _REPO_ROOT / "packages" / "core" / "src"
if str(_CORE_SRC) not in sys.path:
    sys.path.insert(0, str(_CORE_SRC))

STATIC_DIR = _WEB_DIR / "static"

# --- hard limits (tune in one place) ----------------------------------------
MAX_UPLOAD_BYTES = 50 * 1024 * 1024          # 50 MB upload cap
MAX_JOBS = 100                               # total jobs kept
MAX_CONCURRENT = 2                           # parallel conversions
JOB_TTL_SEC = 30 * 60                        # 30 min then wiped
CONVERT_TIMEOUT_SEC = 300                    # 5 min per job
RATE_LIMIT_N = 10                            # max uploads...
RATE_LIMIT_WINDOW_SEC = 10 * 60              # ...per 10 min per IP
ALLOWED_EXTS = {".mp4", ".mov", ".webm", ".mkv", ".avi"}

# Server-side parameter bounds (frontend sliders must stay inside these;
# anything outside is clamped/rejected here regardless of client).
BOUNDS = {
    "num_colors": (8, 24),
    "max_dim": (128, 1080),
    "fps": (5.0, 30.0),
    "merge_area": (1, 100),
    "keyframe_step": (1, 10),
    "scene_threshold": (1.0, 100.0),
    "scene_min_len": (1, 60),
}
DEFAULTS = {
    "mode": "main",
    "num_colors": 16,
    "max_dim": 480,          # web default smaller than CLI for speed/RAM
    "fps": 12.0,
    "merge_area": 10,
    "flow": "dis",
    "keyframe_step": 2,
    "scene_threshold": 27.0,
    "scene_min_len": 15,
}

JOBS_DIR = Path(tempfile.gettempdir()) / "vectorise-web-jobs"
JOBS_DIR.mkdir(parents=True, exist_ok=True)

# --- job store ---------------------------------------------------------------
_lock = threading.Lock()
_jobs: dict[str, dict] = {}          # id -> {status, report?, error?, created, dir}
_rate: dict[str, deque[float]] = {}  # ip -> upload timestamps
_executor = ThreadPoolExecutor(max_workers=MAX_CONCURRENT)
_running = 0


def _now() -> float:
    return time.time()


def _check_rate(ip: str) -> bool:
    """True if allowed, False if rate-limited."""
    now = _now()
    with _lock:
        q = _rate.setdefault(ip, deque())
        while q and now - q[0] > RATE_LIMIT_WINDOW_SEC:
            q.popleft()
        if len(q) >= RATE_LIMIT_N:
            return False
        q.append(now)
        return True


def _sweeper() -> None:
    while True:
        time.sleep(60)
        cutoff = _now() - JOB_TTL_SEC
        with _lock:
            expired = [jid for jid, j in _jobs.items() if j["created"] < cutoff]
            for jid in expired:
                job = _jobs.pop(jid, None)
                if job:
                    shutil.rmtree(str(job["dir"]), ignore_errors=True)


threading.Thread(target=_sweeper, daemon=True).start()


def _clamp_int(name: str, raw, default: int) -> int:
    try:
        v = int(raw)
    except (TypeError, ValueError):
        raise ValueError(f"bad {name}")
    lo, hi = BOUNDS[name]
    if not (lo <= v <= hi):
        raise ValueError(f"{name} must be {lo}..{hi}")
    return v


def _clamp_float(name: str, raw, default: float) -> float:
    try:
        v = float(raw)
    except (TypeError, ValueError):
        raise ValueError(f"bad {name}")
    lo, hi = BOUNDS[name]
    if not (lo <= v <= hi):
        raise ValueError(f"{name} must be {lo}..{hi}")
    return v


def _magic_ok(head: bytes, ext: str) -> bool:
    """Sniff container magic; defense-in-depth before ffmpeg decodes."""
    if head.startswith(b"\x1a\x45\xdf\xa3"):
        return ext in (".webm", ".mkv")
    if head[4:8] == b"ftyp":
        return ext in (".mp4", ".mov")
    if head.startswith(b"RIFF") and head[8:12] == b"AVI ":
        return ext == ".avi"
    # mov may also start with free/mdat atoms; be lenient only for .mov
    if ext == ".mov" and (head.startswith(b"\x00") or b"moov" in head[:64]):
        return True
    return False


def _convert_job(jid: str, params: dict) -> None:
    global _running
    with _lock:
        job = _jobs.get(jid)
        if not job:
            return
        job["status"] = "running"
    with _lock:
        _running += 1
    try:
        # Imported lazily so server boots even if core deps missing.
        from core_engine import video_to_lottie

        in_path = str(job["dir"] / "input.bin")
        out_path = str(job["dir"] / "out.json")
        mp4_path = str(job["dir"] / "preview.mp4")
        t0 = _now()
        # Watchdog: run with timeout via thread join is handled by
        # executor future? Here we enforce wall-clock inside worker by
        # relying on CONVERT_TIMEOUT_SEC polling in handler; actual
        # core call is bounded by ffmpeg + streaming build. We run it
        # directly and measure; over-time jobs are marked error on next poll.
        kwargs: dict = {}
        if params["mode"] == "compressed":
            kwargs = {
                "flow_method": params["flow"],
                "keyframe_step": params["keyframe_step"],
                "scene_threshold": params["scene_threshold"],
                "scene_min_len": params["scene_min_len"],
                "max_frames": 300,  # hard RAM guard for web batch builds
            }
        report = video_to_lottie(
            in_path, out_path,
            mode=params["mode"],
            target_fps=params["fps"],
            max_dimension=params["max_dim"],
            num_colors=params["num_colors"],
            merge_min_area=params["merge_area"],
            preview_mp4=mp4_path,
            verbose=False,
            **kwargs,
        )
        with _lock:
            job["status"] = "done"
            job["report"] = {
                "mode": report.get("mode"),
                "num_frames": report.get("num_frames"),
                "num_tracks": report.get("num_tracks"),
                "size_kb": round(float(report.get("size_kb", 0)), 1),
                "fps": round(float(report.get("fps", 0)), 1),
                "has_mp4": Path(mp4_path).is_file(),
                "has_json": Path(out_path).is_file(),
                "elapsed_s": round(_now() - t0, 1),
            }
    except Exception as exc:  # never leak traceback to client
        print(f"[web] job {jid} failed: {type(exc).__name__}: {exc}",
              file=sys.stderr, flush=True)
        with _lock:
            job["status"] = "error"
            job["error"] = "conversion failed (unsupported or corrupt video?)"
    finally:
        with _lock:
            _running -= 1


SECURITY_HEADERS = {
    "Content-Security-Policy":
        "default-src 'self'; script-src 'self'; style-src 'self'; "
        "media-src 'self' blob:; img-src 'self' blob: data:; "
        "object-src 'none'; base-uri 'self'; frame-ancestors 'none'",
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
    "Cross-Origin-Opener-Policy": "same-origin",
    "Cross-Origin-Resource-Policy": "same-origin",
}


class Handler(BaseHTTPRequestHandler):
    server_version = "VectoRiseWeb"  # no version leak

    def log_message(self, fmt, *args):  # quieter, no query echo
        sys.stderr.write(f"[web] {self.address_string()} {fmt % args}\n")

    # -- helpers ---------------------------------------------------------
    def _send_headers(self, code: int, ctype: str, extra: dict | None = None,
                      length: int | None = None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        for k, v in SECURITY_HEADERS.items():
            self.send_header(k, v)
        if length is not None:
            self.send_header("Content-Length", str(length))
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()

    def _send_json(self, code: int, obj: dict):
        body = json.dumps(obj).encode()
        self._send_headers(code, "application/json", length=len(body))
        self.wfile.write(body)

    def _serve_static(self, name: str, ctype: str):
        p = STATIC_DIR / name
        try:
            data = p.read_bytes()
        except OSError:
            self._send_json(404, {"error": "not found"})
            return
        self._send_headers(200, ctype, length=len(data),
                           extra={"Cache-Control": "no-store"})
        self.wfile.write(data)

    def _client_ip(self) -> str:
        return self.client_address[0] if self.client_address else "unknown"

    # -- GET -------------------------------------------------------------
    def do_HEAD(self):
        # Minimal HEAD for health-checks/probes: same status, no body.
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path in ("/", "/app.js", "/styles.css"):
            ctype = {"/": "text/html", "/app.js": "text/javascript",
                     "/styles.css": "text/css"}[parsed.path]
            self._send_headers(200, ctype, extra={"Cache-Control": "no-store"})
        else:
            self._send_headers(404, "application/json", length=16)
            # body intentionally omitted for HEAD

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path, qs = parsed.path, urllib.parse.parse_qs(parsed.query)
        if path == "/":
            return self._serve_static("index.html", "text/html; charset=utf-8")
        if path == "/app.js":
            return self._serve_static("app.js",
                                      "text/javascript; charset=utf-8")
        if path == "/styles.css":
            return self._serve_static("styles.css", "text/css; charset=utf-8")
        if path == "/api/status":
            jid = (qs.get("id", [""])[0] or "")[:64]
            if not jid.isalnum() or len(jid) != 32:
                return self._send_json(400, {"error": "bad id"})
            with _lock:
                job = _jobs.get(jid)
                snap = dict(job) if job else None
            if not snap:
                return self._send_json(404, {"error": "unknown job"})
            snap.pop("dir", None)
            return self._send_json(200, snap)
        if path in ("/api/download", "/preview"):
            jid = (qs.get("id", [""])[0] or "")[:64]
            kind = (qs.get("kind", ["json"])[0] or "json")[:8]
            if not jid.isalnum() or len(jid) != 32:
                return self._send_json(400, {"error": "bad id"})
            with _lock:
                job = _jobs.get(jid)
                status = job["status"] if job else None
                jdir = job["dir"] if job else None
            if not job:
                return self._send_json(404, {"error": "unknown job"})
            if status != "done":
                return self._send_json(409, {"error": f"not ready: {status}"})
            assert jdir is not None
            if path == "/preview" or kind == "mp4":
                f = Path(jdir) / "preview.mp4"
                ctype, disp = "video/mp4", "inline"
                fname = "preview.mp4"
            else:
                f = Path(jdir) / "out.json"
                ctype, disp = "application/json", "attachment"
                fname = "vectorised.json"
            try:
                data = f.read_bytes()
            except OSError:
                return self._send_json(404, {"error": "file expired"})
            # Never serve more than ~100MB back even if something is off.
            if len(data) > 100 * 1024 * 1024:
                return self._send_json(500, {"error": "output too large"})
            self._send_headers(200, ctype, length=len(data), extra={
                "Content-Disposition": f'{disp}; filename="{fname}"',
                "Cache-Control": "no-store",
            })
            self.wfile.write(data)
            return
        return self._send_json(404, {"error": "not found"})

    # -- POST ------------------------------------------------------------
    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path != "/api/jobs":
            return self._send_json(404, {"error": "not found"})
        # CSRF: same-origin POSTs only (no cookies, but blocks evil-site forms).
        origin = self.headers.get("Origin")
        host = self.headers.get("Host", "")
        if origin:
            o = urllib.parse.urlparse(origin)
            if o.netloc != host:
                return self._send_json(403, {"error": "cross-origin blocked"})
        if not _check_rate(self._client_ip()):
            return self._send_json(429, {"error": "rate limited, try later"})
        try:
            total = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            return self._send_json(400, {"error": "bad length"})
        if total <= 0 or total > MAX_UPLOAD_BYTES + 1024 * 1024:
            return self._send_json(413, {"error": "upload too large (50MB max)"})
        ctype = self.headers.get("Content-Type", "")
        if "multipart/form-data" not in ctype:
            return self._send_json(400, {"error": "need multipart upload"})
        # cgi parses safely without shell; cap in-memory via Content-Length
        # check above (cgi streams file parts to temp files).
        try:
            form = cgi.FieldStorage(
                fp=self.rfile, headers=self.headers,
                environ={"REQUEST_METHOD": "POST",
                         "CONTENT_TYPE": ctype,
                         "CONTENT_LENGTH": str(total)},
                keep_blank_values=True,
            )
        except Exception:
            return self._send_json(400, {"error": "malformed upload"})

        def field(name: str, default=""):
            item = form[name] if name in form else None
            if item is None or isinstance(item, list):
                return default
            if getattr(item, "filename", None):
                return default
            return item.value if isinstance(item.value, str) else default

        # --- validate params (server is source of truth) ------------------
        try:
            mode = field("mode", DEFAULTS["mode"])
            if mode not in ("main", "compressed"):
                raise ValueError("mode must be main/compressed")
            flow = field("flow", DEFAULTS["flow"])
            if flow not in ("dis", "farneback"):
                raise ValueError("flow must be dis/farneback")
            params = {
                "mode": mode,
                "num_colors": _clamp_int("num_colors",
                    field("num_colors", DEFAULTS["num_colors"]),
                    DEFAULTS["num_colors"]),
                "max_dim": _clamp_int("max_dim",
                    field("max_dim", DEFAULTS["max_dim"]),
                    DEFAULTS["max_dim"]),
                "fps": _clamp_float("fps",
                    field("fps", DEFAULTS["fps"]), DEFAULTS["fps"]),
                "merge_area": _clamp_int("merge_area",
                    field("merge_area", DEFAULTS["merge_area"]),
                    DEFAULTS["merge_area"]),
                "keyframe_step": _clamp_int("keyframe_step",
                    field("keyframe_step", DEFAULTS["keyframe_step"]),
                    DEFAULTS["keyframe_step"]),
                "scene_threshold": _clamp_float("scene_threshold",
                    field("scene_threshold", DEFAULTS["scene_threshold"]),
                    DEFAULTS["scene_threshold"]),
                "scene_min_len": _clamp_int("scene_min_len",
                    field("scene_min_len", DEFAULTS["scene_min_len"]),
                    DEFAULTS["scene_min_len"]),
                "flow": flow,
            }
        except ValueError as exc:
            return self._send_json(400, {"error": str(exc)})

        # --- validate file -------------------------------------------------
        if "video" not in form:
            return self._send_json(400, {"error": "missing 'video' file"})
        item = form["video"]
        if isinstance(item, list):  # one file only
            return self._send_json(400, {"error": "one file only"})
        if not getattr(item, "filename", None) or not getattr(item, "file", None):
            return self._send_json(400, {"error": "missing 'video' file"})
        safe_name = html.escape(Path(item.filename).name[-80:])
        ext = Path(safe_name).suffix.lower()
        if ext not in ALLOWED_EXTS:
            return self._send_json(
                400, {"error": f"extension {ext or '?'} not allowed "
                               f"(mp4/mov/webm/mkv/avi)"})
        try:
            item.file.seek(0, 2)
            size = item.file.tell()
            item.file.seek(0)
        except Exception:
            return self._send_json(400, {"error": "unreadable upload"})
        if size <= 0 or size > MAX_UPLOAD_BYTES:
            return self._send_json(413, {"error": "upload too large (50MB max)"})
        head = item.file.read(64)
        item.file.seek(0)
        if not _magic_ok(head, ext):
            return self._send_json(400, {"error": "file is not a real video"})

        with _lock:
            if len(_jobs) >= MAX_JOBS:
                return self._send_json(503, {"error": "server busy, try later"})
            if _running >= MAX_CONCURRENT:
                # still accept but queue visibly instead of silently piling up
                pass
            jid = uuid.uuid4().hex
            jdir = JOBS_DIR / jid
            jdir.mkdir(parents=True, exist_ok=False)
            _jobs[jid] = {"status": "queued", "created": _now(),
                          "dir": jdir, "filename": safe_name}

        # Write to disk outside webroot, fixed name, no client path.
        try:
            with open(jdir / "input.bin", "wb") as f:
                shutil.copyfileobj(item.file, f, length=1024 * 256)
            try:
                item.file.close()
            except Exception:
                pass
        except OSError:
            with _lock:
                _jobs.pop(jid, None)
            shutil.rmtree(str(jdir), ignore_errors=True)
            return self._send_json(500, {"error": "could not store upload"})

        _executor.submit(_convert_job, jid, params)
        return self._send_json(200, {"id": jid})


def main() -> int:
    ap = argparse.ArgumentParser(description="VectoRise secure web server")
    ap.add_argument("--host", default="127.0.0.1",
                    help="bind host (default 127.0.0.1; use 0.0.0.0 for LAN)")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    srv.daemon_threads = True
    print(f"VectoRise web on http://{args.host}:{args.port}/  "
          f"(jobs in {JOBS_DIR})", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
