"""get_replays.py — resumable bulk downloader for ballchasing.com replays.

This is stage 1 of the JEPA data pipeline (see CLAUDE.md → "Data pipeline"):
it downloads raw `.replay` binaries from the ballchasing API into a directory.
It does NOT decode them — the decode/tensor stage is a separate script that
drains and deletes the files (stream-decode-discard).

Process design
--------------
* Driver: **date-windowed cursor scraping**. We walk time windows (default 1 day)
  over a playlist (default ranked 1v1), oldest -> newest, fully paging each window
  via the API's `next` cursor. Windowing spans all ranks and makes resumption clean.
* Rate limits (GC Patron tier): metadata listing <= 16 req/s, file downloads <= 2 req/s.
  Both enforced by thread-safe token buckets. File downloads are the hard bottleneck
  (~2/s ~= 172k/day per token).
* Resumable: a SQLite manifest records downloaded ids (dedup across restarts); a JSON
  checkpoint persists {window_start, next_cursor} after every page. Kill/crash -> resume
  with no re-downloads. Files are written to `.part` then atomically renamed.
* 429 / Retry-After honored with exponential backoff.

Usage
-----
    export BALLCHASING_TOKEN=your_gc_token        # or --token, or a token file
    python get_replays.py --out /path/to/replays --start 2023-01-01 --end 2023-02-01

Run `python get_replays.py --help` for all options. Safe to Ctrl-C and re-run.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import queue
import signal
import sqlite3
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

API_BASE = "https://ballchasing.com/api"
DEFAULT_TOKEN_FILE = ".ballchasing_token"
DEFAULT_OUT = "/Users/ck/PycharmProjects/JEPA_rocketleague_pretraining/replays"

log = logging.getLogger("get_replays")


# --------------------------------------------------------------------------- #
# Rate limiting
# --------------------------------------------------------------------------- #
class RateLimiter:
    """Thread-safe token bucket. `acquire()` blocks until a token is available,
    enforcing a global cap of `rate` operations/second across all threads."""

    def __init__(self, rate: float, capacity: float | None = None):
        self.rate = rate
        self.capacity = capacity if capacity is not None else max(1.0, rate)
        self.tokens = self.capacity
        self.timestamp = time.monotonic()
        self.lock = threading.Lock()

    def acquire(self) -> None:
        with self.lock:
            while True:
                now = time.monotonic()
                self.tokens = min(self.capacity, self.tokens + (now - self.timestamp) * self.rate)
                self.timestamp = now
                if self.tokens >= 1.0:
                    self.tokens -= 1.0
                    return
                time.sleep((1.0 - self.tokens) / self.rate)


# --------------------------------------------------------------------------- #
# Manifest (SQLite) + checkpoint (JSON)
# --------------------------------------------------------------------------- #
class Manifest:
    """Persistent record of downloaded replay ids + light metadata."""

    def __init__(self, db_path: Path):
        self.con = sqlite3.connect(db_path, check_same_thread=False)
        self.con.execute(
            "CREATE TABLE IF NOT EXISTS downloaded ("
            "  id TEXT PRIMARY KEY, playlist TEXT, rank TEXT, date TEXT,"
            "  map_code TEXT, duration INTEGER, ts REAL)"
        )
        self.con.commit()
        self.lock = threading.Lock()
        # Load existing ids into memory for O(1) hot-path dedup.
        # NOTE: at many millions of ids this set costs hundreds of MB+; acceptable
        # on the hosted machine. Swap for a Bloom filter if it becomes a problem.
        self.seen: set[str] = {row[0] for row in self.con.execute("SELECT id FROM downloaded")}
        log.info("manifest: %d replays already downloaded", len(self.seen))

    def has(self, replay_id: str) -> bool:
        return replay_id in self.seen

    def record(self, rep: dict) -> None:
        with self.lock:
            self.con.execute(
                "INSERT OR IGNORE INTO downloaded VALUES (?,?,?,?,?,?,?)",
                (
                    rep["id"],
                    rep.get("playlist_id"),
                    (rep.get("min_rank") or {}).get("id"),
                    rep.get("date"),
                    rep.get("map_code"),
                    rep.get("duration"),
                    time.time(),
                ),
            )
            self.con.commit()
            self.seen.add(rep["id"])

    def count(self) -> int:
        return len(self.seen)


class Checkpoint:
    """Persists scrape progress {window_start, next_cursor} atomically."""

    def __init__(self, path: Path):
        self.path = path
        self.data = {"window_start": None, "next_url": None}
        if path.exists():
            self.data = json.loads(path.read_text())

    def save(self, window_start: str | None, next_url: str | None) -> None:
        self.data = {"window_start": window_start, "next_url": next_url}
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data))
        os.replace(tmp, self.path)


# --------------------------------------------------------------------------- #
# ballchasing API helpers
# --------------------------------------------------------------------------- #
def make_session(token: str) -> requests.Session:
    s = requests.Session()
    s.headers.update({"Authorization": token})
    return s


def check_account(session) -> str:
    """Validate the token and report the patron tier up front. Returns the tier
    string ('regular', 'gold', 'diamond', 'champion', 'gc'). Raises on bad token."""
    r = session.get(f"{API_BASE}/", timeout=15)
    if r.status_code == 401:
        raise SystemExit("API token rejected (401). Check .ballchasing_token.")
    r.raise_for_status()
    d = r.json()
    tier = d.get("type", "?")
    log.info("account: %s | patron tier: %s", d.get("name"), tier)
    if tier != "gc":
        log.warning(
            "tier '%s' has a HARD hourly FILE-download cap (~200/hr on regular). "
            "Downloads will 429 once it's hit — this looks like the scraper 'stalling'. "
            "Upgrade to GC for 2/sec with no hourly cap.", tier)
    return tier


def iter_pages(session, meta_limiter, start_url, params, stop):
    """Yield (list_of_replays, next_url) per page, following the `next` cursor.
    `start_url` may already contain query params (a resumed cursor); `params` is
    used only for the first request."""
    url, p = start_url, params
    while url and not stop.is_set():
        meta_limiter.acquire()
        try:
            resp = session.get(url, params=p, timeout=30)
        except requests.RequestException as e:
            log.warning("list error (%s); retrying in 5s", e)
            time.sleep(5)
            continue
        if resp.status_code == 429:
            wait = max(1.0, float(resp.headers.get("Retry-After", 2)))
            log.info("429 on list; sleeping %.1fs", wait)
            time.sleep(wait)
            continue
        resp.raise_for_status()
        data = resp.json()
        next_url = data.get("next")
        yield data.get("list", []), next_url
        url, p = next_url, None


def download_replay(session, file_limiter, replay_id, dest: Path, max_retries=6) -> bool:
    """Stream one replay to `dest` (atomic via .part rename). Returns success."""
    tmp = dest.with_suffix(".part")
    saw_429 = False
    for attempt in range(max_retries):
        file_limiter.acquire()
        try:
            with session.get(f"{API_BASE}/replays/{replay_id}/file", stream=True, timeout=120) as r:
                if r.status_code == 429:
                    saw_429 = True
                    wait = max(1.0, float(r.headers.get("Retry-After", 2 ** attempt)))
                    log.info("429 on file %s; sleeping %.1fs", replay_id, wait)
                    time.sleep(wait)
                    continue
                r.raise_for_status()
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1 << 16):
                        f.write(chunk)
            os.replace(tmp, dest)
            return True
        except requests.RequestException as e:
            wait = min(60, 2 ** attempt)
            log.warning("download %s failed (%s); retry in %ds", replay_id, e, wait)
            time.sleep(wait)
    tmp.unlink(missing_ok=True)
    if saw_429:
        log.error("giving up on %s: repeated 429s — hourly download quota likely "
                  "EXHAUSTED (regular tier ~200/hr). Wait for reset or upgrade to GC.", replay_id)
    else:
        log.error("giving up on %s after %d attempts", replay_id, max_retries)
    return False


# --------------------------------------------------------------------------- #
# Date windowing
# --------------------------------------------------------------------------- #
def parse_day(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc)


def windows(start: datetime, end: datetime, step: timedelta):
    cur = start
    while cur < end:
        nxt = min(cur + step, end)
        yield cur, nxt
        cur = nxt


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run(args) -> None:
    token = resolve_token(args)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    session = make_session(token)
    tier = check_account(session)  # validate token + warn about tier caps before scraping
    # auto-match rates to the tier unless the user overrode them. Running ABOVE the
    # tier's per-second file cap causes ~half the requests to 429 + backoff, which
    # is slower than just running at the cap.
    FILE_CAP = {"gc": 2.0, "champion": 2.0, "diamond": 2.0}      # else 1.0/s
    META_CAP = {"gc": 16.0, "champion": 8.0, "diamond": 4.0, "gold": 2.0}  # else 2.0/s
    file_rate = args.file_rate if args.file_rate is not None else FILE_CAP.get(tier, 1.0)
    meta_rate = args.meta_rate if args.meta_rate is not None else META_CAP.get(tier, 2.0)
    log.info("rate limits (tier=%s): file=%.1f/s, meta=%.1f/s", tier, file_rate, meta_rate)
    if tier != "gc":
        log.warning("tier '%s' also has an HOURLY file cap (~200/hr on regular) — "
                    "throughput will collapse after the first ~200 downloads regardless of rate.", tier)
    meta_limiter = RateLimiter(meta_rate)
    file_limiter = RateLimiter(file_rate)
    manifest = Manifest(out / "manifest.sqlite")
    checkpoint = Checkpoint(out / "checkpoint.json")

    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: (log.info("stopping (finishing in-flight)…"), stop.set()))
    signal.signal(signal.SIGTERM, lambda *_: stop.set())

    # --- download workers pull ids from a bounded queue (backpressure) ------- #
    q: "queue.Queue[dict]" = queue.Queue(maxsize=args.workers * 8)
    downloaded_this_run = {"n": 0}
    dl_lock = threading.Lock()

    def worker():
        while not stop.is_set():
            try:
                rep = q.get(timeout=1)
            except queue.Empty:
                continue
            try:
                dest = out / f"{rep['id']}.replay"
                if download_replay(session, file_limiter, rep["id"], dest):
                    manifest.record(rep)
                    with dl_lock:
                        downloaded_this_run["n"] += 1
                        n = downloaded_this_run["n"]
                        if n % 100 == 0:
                            log.info("downloaded %d this run (%d total)", n, manifest.count())
                        if args.max and n >= args.max:
                            log.info("reached --max %d; stopping", args.max)
                            stop.set()
            finally:
                q.task_done()

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(args.workers)]
    for t in threads:
        t.start()

    # --- resume position ----------------------------------------------------- #
    start = parse_day(args.start)
    end = parse_day(args.end) if args.end else datetime.now(timezone.utc)
    step = timedelta(hours=args.window_hours)

    resume_ws = checkpoint.data.get("window_start")
    resume_cursor = checkpoint.data.get("next_url")

    try:
        for w_start, w_end in windows(start, end, step):
            if stop.is_set():
                break
            ws_iso = w_start.isoformat()
            # Skip windows already completed before the checkpointed one.
            if resume_ws and ws_iso < resume_ws:
                continue

            first_url = f"{API_BASE}/replays"
            params = {
                "playlist": args.playlist,
                "replay-date-after": ws_iso,
                "replay-date-before": w_end.isoformat(),
                "count": 200,
                "sort-by": "replay-date",
                "sort-dir": "asc",
            }
            # If resuming mid-window, jump straight to the saved cursor.
            start_url = first_url
            if resume_ws == ws_iso and resume_cursor:
                start_url, params = resume_cursor, None
            resume_ws = resume_cursor = None  # only applies to the first window

            log.info("window %s -> %s", ws_iso, w_end.isoformat())
            for page, next_url in iter_pages(session, meta_limiter, start_url, params, stop):
                for rep in page:
                    if stop.is_set():
                        break
                    if manifest.has(rep["id"]) or (out / f"{rep['id']}.replay").exists():
                        continue
                    # Stop-aware backpressure: block while the queue is full, but
                    # bail out if a worker (or SIGINT) sets `stop` meanwhile —
                    # otherwise a plain q.put() deadlocks once workers have exited.
                    while not stop.is_set():
                        try:
                            q.put(rep, timeout=0.5)
                            break
                        except queue.Full:
                            continue
                checkpoint.save(ws_iso, next_url)
                if stop.is_set():
                    break
            else:
                # window fully paged; advance checkpoint past it
                checkpoint.save(w_end.isoformat(), None)

        if not stop.is_set():
            log.info("draining download queue…")
            q.join()  # normal completion: wait for all queued downloads to finish
    finally:
        stop.set()
        # Unblock shutdown: discard anything still queued (unprocessed ids are simply
        # re-fetched on the next run, since they were never recorded in the manifest).
        try:
            while True:
                q.get_nowait()
                q.task_done()
        except queue.Empty:
            pass
        for t in threads:
            t.join(timeout=5)
        log.info("done. %d replays in manifest.", manifest.count())


def resolve_token(args) -> str:
    """Token precedence: --token > token file > $BALLCHASING_TOKEN.

    The token file wins over the env var on purpose: the env var has repeatedly
    held a *stale* token that silently overrode the intended one. The file is the
    explicit, freshly-placed credential, so it takes priority.
    """
    if args.token:
        return args.token.strip()
    tf = Path(args.token_file)
    if tf.exists():
        log.info("using token from %s", args.token_file)
        return tf.read_text().strip()
    env = os.environ.get("BALLCHASING_TOKEN")
    if env:
        log.info("using token from $BALLCHASING_TOKEN")
        return env.strip()
    raise SystemExit(
        f"No API token. Put it in {args.token_file}, pass --token, "
        "or set $BALLCHASING_TOKEN"
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Resumable ballchasing.com replay downloader")
    p.add_argument("--out", default=DEFAULT_OUT, help="output directory for .replay files + manifest")
    p.add_argument("--playlist", default="ranked-duels",
                   help="ballchasing playlist id (default ranked-duels = 1v1)")
    p.add_argument("--start", default="2021-01-01", help="earliest replay date, YYYY-MM-DD (UTC)")
    p.add_argument("--end", default=None, help="latest replay date, YYYY-MM-DD (default: now)")
    p.add_argument("--window-hours", type=int, default=24, help="date window size in hours")
    p.add_argument("--workers", type=int, default=4, help="concurrent download workers")
    p.add_argument("--max", type=int, default=0, help="stop after N downloads this run (0 = unlimited)")
    p.add_argument("--file-rate", type=float, default=None,
                   help="max file downloads/sec (default: auto from tier — 2/s GC, 1/s regular)")
    p.add_argument("--meta-rate", type=float, default=None,
                   help="max metadata calls/sec (default: auto from tier)")
    p.add_argument("--token", default=None, help="API token (else env/file)")
    p.add_argument("--token-file", default=DEFAULT_TOKEN_FILE, help="path to token file")
    return p


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
