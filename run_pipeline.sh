#!/usr/bin/env bash
# run_pipeline.sh — couple the downloader + decoder into a stream-decode-discard
# pipeline (see CLAUDE.md). Downloader pulls .replay files at the 2/s cap into
# $RAW; decoder --watch drains them into zstd shards in $OUT and deletes each raw
# once it's packed (peak disk stays flat). Decode (~2.8/s single-core) outpaces
# download, so raw never accumulates.
#
# Usage:
#   ./run_pipeline.sh [DATA_DIR] [MAX_REPLAYS]
#   ./run_pipeline.sh /Volumes/rl/data 100000
#
# DATA_DIR defaults to ./data ; put it on the volume with real space.
# Safe to Ctrl-C: both stages checkpoint and resume with no re-work.
set -euo pipefail
cd "$(dirname "$0")"

DATA="${1:-./data}"
MAX="${2:-100000}"
RAW="$DATA/replays"
OUT="$DATA/shards"
mkdir -p "$RAW" "$OUT"

echo "pipeline: DATA=$DATA  MAX=$MAX  raw=$RAW  shards=$OUT"

# stage 1: downloader (training venv has requests). Logs to $DATA/download.log
.venv/bin/python get_replays.py --out "$RAW" --max "$MAX" \
    >>"$DATA/download.log" 2>&1 &
DL=$!
echo "  downloader pid $DL  -> $DATA/download.log"

# stage 2: decoder watching $RAW, streaming shards to $OUT, deleting raw. Logs to $DATA/decode.log
.venv-decode/bin/python decode_replays.py --replays "$RAW" --out "$OUT" \
    --watch --delete-raw >>"$DATA/decode.log" 2>&1 &
DEC=$!
echo "  decoder    pid $DEC  -> $DATA/decode.log"

# Ctrl-C stops both cleanly (each flushes/checkpoints in its SIGINT handler).
trap 'echo; echo "stopping both…"; kill -INT "$DL" "$DEC" 2>/dev/null || true' INT TERM

# Wait for the download target to be reached, then let decode drain the last
# files before stopping it — otherwise --watch would idle forever.
wait "$DL" || true
echo "download finished; draining remaining raw files…"
while ls "$RAW"/*.replay >/dev/null 2>&1; do sleep 5; done
kill -INT "$DEC" 2>/dev/null || true
wait "$DEC" || true
echo "pipeline done. shards in $OUT"
