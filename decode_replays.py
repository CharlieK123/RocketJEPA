"""decode_replays.py — stage 2+3 of the JEPA data pipeline (see CLAUDE.md).

Reads raw `.replay` files, decodes them with sprocket-carball into per-frame
game state, resamples to a fixed fps, extracts a compact float16 feature tensor
per replay, and packs many replays into zstd-compressed shards for training.

MUST be run with the decode venv, NOT the training venv:
    .venv-decode/bin/python decode_replays.py

(carball needs pandas 3 / py3.11, which conflict with the Torch training .venv.)

Pipeline role
-------------
  get_replays.py  ->  replays/*.replay  ->  [THIS]  ->  shards/*.zst  ->  JEPA loader

Design
------
* Decode via the verified carball bypass (header+network -> DataFrame, skipping
  carball's pandas-3-incompatible events analysis): decompile_replay -> Game ->
  AnalysisManager._initialize_data_frame.
* 1v1 only: state = ball(12) + 2 cars(17 each); action = 2 cars x 3 inputs. 52 feats.
* Resample to --fps using the game clock (cumulative frame delta).
* float16 + zstd shards, each holding --shard-size replays, with a JSON sidecar
  index (per-replay row ranges + rank + the feature schema).
* Resumable: a manifest of decoded ids; already-done replays are skipped.
* stream-decode-discard: with --delete-raw, raw .replay files are removed once
  their shard is durably written (keeps peak disk flat). OFF by default for safety.

Run `.venv-decode/bin/python decode_replays.py --help` for options.
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import signal
import threading
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("decode_replays")

# --- feature schema (fixed order) ------------------------------------------ #
# Layout matches JEPA.ObjectEncoder: ball(12) + player(22) + opponent(16) + env(N).
# Actions (throttle/steer/handbrake + button flags) are NOT a separate tensor —
# they live inside the self-player object. Aerial pitch/yaw/roll are not in replays.
KINEMATICS = ["pos_x", "pos_y", "pos_z", "vel_x", "vel_y", "vel_z",
              "ang_vel_x", "ang_vel_y", "ang_vel_z", "rot_x", "rot_y", "rot_z"]

BALL_FIELDS = KINEMATICS                                        # 12
# physical state (from the decoded DataFrame): kinematics + boost + 3 flags
PLAYER_STATE_FIELDS = KINEMATICS + ["boost", "jump_active", "dodge_active",
                                    "double_jump_active"]        # 16
# all 8 actions, APPROXIMATED by carball's ControlsCreator (from player.controls).
# throttle/steer/pitch/yaw/roll in [-1,1]; jump/boost/handbrake in {0,1}.
# pitch/yaw/roll are inferred from angular velocity and only defined airborne
# (grounded -> 0). These are best-guess inputs, not ground-truth.
ACTION_FIELDS = ["throttle", "steer", "pitch", "yaw", "roll",
                 "jump", "boost", "handbrake"]                  # 8
# self-player object = physical state (16) + actions (8) = 24
# opponent: physical state only, no actions (16)
OPPONENT_FIELDS = PLAYER_STATE_FIELDS                           # 16
# env: timestep + score + kickoff (7). Extra env dims (boost pads etc.) derived downstream.
ENV_FIELDS = ["seconds_remaining", "is_overtime", "ball_has_been_hit",
              "blue_score", "orange_score", "score_diff", "kickoff"]  # 7

OBJ_LENGTHS = (len(BALL_FIELDS),                                # ball      12
               len(PLAYER_STATE_FIELDS) + len(ACTION_FIELDS),  # player    24
               len(OPPONENT_FIELDS),                           # opponent  16
               len(ENV_FIELDS))                                # env        7
FEAT_DIM = sum(OBJ_LENGTHS)                                     # 59


def build_feature_names() -> list[str]:
    return ([f"ball.{f}" for f in BALL_FIELDS]
            + [f"player.{f}" for f in PLAYER_STATE_FIELDS]
            + [f"player.act.{f}" for f in ACTION_FIELDS]
            + [f"opponent.{f}" for f in OPPONENT_FIELDS]
            + [f"env.{f}" for f in ENV_FIELDS])


# --------------------------------------------------------------------------- #
# Decode + feature extraction
# --------------------------------------------------------------------------- #
def _col(df, actor, field):
    """Return a column as float, or zeros if the field is absent for this actor."""
    sub = df[actor]
    if field in sub.columns:
        return sub[field].to_numpy(dtype="float64")
    return np.zeros(len(df), dtype="float64")


def _running_score(game, n):
    """Per-frame cumulative (blue, orange) score from goal events."""
    blue = np.zeros(n, dtype="float64")
    orange = np.zeros(n, dtype="float64")
    for g in getattr(game, "goals", []) or []:
        fr = int(getattr(g, "frame_number", 0))
        if fr >= n:
            fr = n - 1
        if int(getattr(g, "player_team", 0)) == 0:
            blue[fr:] += 1
        else:
            orange[fr:] += 1
    return blue, orange


def _kickoff_flag(df, ball_block):
    """Heuristic kickoff indicator: ball at center & not yet hit (start of play).
    Coarse on purpose — you derive a richer env downstream."""
    not_hit = 1.0 - np.nan_to_num(df[("game", "ball_has_been_hit")].to_numpy(dtype="float64"))
    near_center = (np.abs(ball_block[:, 0]) < 200) & (np.abs(ball_block[:, 1]) < 200)
    return (not_hit * near_center.astype("float64"))


def decode_to_features(replay_path: str, fps: int):
    """Decode one replay -> (feat[n,57] float16, meta). Returns None on skip.

    Frame layout: ball(12) + self-player(22) + opponent(16) + env(7).
    Self = blue player, opponent = orange (team-ordered). Perspective augmentation
    (swapping self/opponent) can be added in the loader later.
    """
    from carball.decompile_replays import decompile_replay
    from carball.json_parser.game import Game
    from carball.analysis.analysis_manager import AnalysisManager
    from carball.controls.controls import ControlsCreator

    _json = decompile_replay(replay_path)
    game = Game()
    game.initialize(loaded_json=_json)

    players = list(game.players)
    if len(players) != 2:
        return None  # not 1v1
    players.sort(key=lambda p: bool(getattr(p, "is_orange", 0)))
    self_player, opp_player = players[0], players[1]
    self_name, opp_name = self_player.name, opp_player.name

    df = AnalysisManager(game)._initialize_data_frame(game)
    n = len(df)
    if n < fps * 5:  # < 5 seconds of data -> skip
        return None
    if not all(pn in {c[0] for c in df.columns} for pn in (self_name, opp_name)):
        return None

    # approximate all 8 controller inputs (adds .controls to each player)
    ControlsCreator().get_controls(game)

    # --- assemble objects at full frame rate --------------------------------
    ball = np.stack([_col(df, "ball", f) for f in BALL_FIELDS], axis=1)
    player_state = np.stack([_col(df, self_name, f) for f in PLAYER_STATE_FIELDS], axis=1)
    # 8 actions from carball's ControlsCreator: reindex to df frames, fill gaps with 0
    ctrl = self_player.controls.reindex(df.index).fillna(0.0)
    actions = np.stack([ctrl[f].to_numpy(dtype="float64") for f in ACTION_FIELDS], axis=1)
    player = np.concatenate([player_state, actions], axis=1)          # 16 + 8 = 24
    opponent = np.stack([_col(df, opp_name, f) for f in OPPONENT_FIELDS], axis=1)

    blue, orange = _running_score(game, n)
    env_map = {
        "seconds_remaining": _col(df, "game", "seconds_remaining"),
        "is_overtime": _col(df, "game", "is_overtime"),
        "ball_has_been_hit": _col(df, "game", "ball_has_been_hit"),
        "blue_score": blue,
        "orange_score": orange,
        "score_diff": blue - orange,
        "kickoff": _kickoff_flag(df, ball),
    }
    env = np.stack([env_map[f] for f in ENV_FIELDS], axis=1)

    feat = np.concatenate([ball, player, opponent, env], axis=1)

    # --- resample to target fps via cumulative game-clock delta -------------
    delta = np.nan_to_num(df[("game", "delta")].to_numpy(dtype="float64"))
    t = np.cumsum(delta)
    targets = np.arange(0.0, t[-1], 1.0 / fps)
    idx = np.clip(np.searchsorted(t, targets), 0, n - 1)
    feat = np.nan_to_num(feat[idx]).astype(np.float16)

    meta = {"players": [self_name, opp_name], "self_team": "blue"}
    return feat, meta


# --------------------------------------------------------------------------- #
# Shard writer (zstd + JSON index)
# --------------------------------------------------------------------------- #
class ShardWriter:
    def __init__(self, out_dir: Path, fps: int, shard_size: int, level: int, delete_raw: bool = False):
        import zstandard as zstd
        self.out = out_dir
        self.out.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self.shard_size = shard_size
        self.delete_raw = delete_raw
        self.cctx = zstd.ZstdCompressor(level=level)
        self.buf_arrays: list[np.ndarray] = []
        self.buf_index: list[dict] = []
        self.buf_raws: list[str] = []          # raw paths to delete on flush
        self.rows = 0
        # continue numbering after any existing shards
        existing = sorted(glob.glob(str(self.out / "shard_*.zst")))
        self.shard_no = (int(Path(existing[-1]).stem.split("_")[1]) + 1) if existing else 0

    def add(self, replay_id, feat, extra_meta, raw_path):
        """Buffer a replay. Returns the list of ids flushed to a shard (so the
        caller can mark them done), or None if still buffering."""
        self.buf_index.append({"id": replay_id, "start": self.rows,
                               "length": int(feat.shape[0]), **extra_meta})
        self.buf_arrays.append(feat)
        self.buf_raws.append(raw_path)
        self.rows += feat.shape[0]
        if len(self.buf_arrays) >= self.shard_size:
            return self.flush()
        return None

    def flush(self):
        """Write the buffered replays as one shard. Returns the flushed ids
        (empty if nothing buffered). Ids are only 'done' once this returns them,
        so a mid-buffer kill loses no persisted data — those replays re-decode."""
        if not self.buf_arrays:
            return []
        flushed_ids = [r["id"] for r in self.buf_index]
        arr = np.concatenate(self.buf_arrays, axis=0).astype(np.float16)
        name = f"shard_{self.shard_no:05d}"
        (self.out / f"{name}.zst").write_bytes(self.cctx.compress(arr.tobytes()))
        (self.out / f"{name}.json").write_text(json.dumps({
            "dtype": "float16", "shape": list(arr.shape), "fps": self.fps,
            "obj_lengths": list(OBJ_LENGTHS), "feat_dim": FEAT_DIM,
            "feature_names": build_feature_names(),
            "replays": self.buf_index,
        }))
        log.info("wrote %s (%d replays, %d frames, %.1f MB compressed)",
                 name, len(self.buf_arrays), arr.shape[0],
                 (self.out / f"{name}.zst").stat().st_size / 1e6)
        if self.delete_raw:
            for rp in self.buf_raws:
                try:
                    os.remove(rp)
                except OSError:
                    pass
        self.shard_no += 1
        self.buf_arrays, self.buf_index, self.buf_raws, self.rows = [], [], [], 0
        return flushed_ids


# --------------------------------------------------------------------------- #
# Manifest (decoded ids) + orchestration
# --------------------------------------------------------------------------- #
def load_manifest(path: Path) -> set[str]:
    return set(path.read_text().split()) if path.exists() else set()


def run(args):
    replays_dir = Path(args.replays)
    out_dir = Path(args.out)
    manifest_path = out_dir / "decoded.txt"
    out_dir.mkdir(parents=True, exist_ok=True)

    done = load_manifest(manifest_path)       # in-memory set, grown as we decode
    writer = ShardWriter(out_dir, args.fps, args.shard_size, args.zstd_level, args.delete_raw)
    manifest_f = manifest_path.open("a")

    # graceful shutdown: finish the current file, flush the partial shard, exit.
    stop = threading.Event()
    def _handle(*_):
        log.info("stopping (finishing current file, flushing shard)…")
        stop.set()
    signal.signal(signal.SIGINT, _handle)
    signal.signal(signal.SIGTERM, _handle)

    def mark_done(ids):
        for x in ids:
            manifest_f.write(x + "\n")
            done.add(x)
        manifest_f.flush()

    # ids decoded and sitting in the shard buffer but not yet flushed. They are
    # NOT in `done` (crash-safety: unflushed work must re-decode after a crash)
    # and their raw file is NOT deleted until flush — so without tracking them,
    # the next --watch scan would re-decode them. Excluded from `todo` below.
    inflight: set[str] = set()

    n_ok = n_skip = n_err = 0
    log.info("start: %d already decoded, fps=%d, watch=%s, delete_raw=%s, out=%s",
             len(done), args.fps, args.watch, args.delete_raw, out_dir)
    try:
        while not stop.is_set():
            files = sorted(glob.glob(str(replays_dir / "*.replay")))
            todo = [f for f in files
                    if Path(f).stem not in done and Path(f).stem not in inflight]
            if todo:
                log.info("decoding %d new replay(s)", len(todo))
            for f in todo:
                if stop.is_set():
                    break
                rid = Path(f).stem
                try:
                    res = decode_to_features(f, args.fps)
                except Exception as e:
                    n_err += 1
                    log.warning("decode failed %s: %s", rid, e)
                    continue
                if res is None:
                    n_skip += 1
                    # skipped replays produce no shard data, so mark them done immediately
                    mark_done([rid])
                    if args.delete_raw:
                        try: os.remove(f)
                        except OSError: pass
                    continue
                feat, meta = res
                # only mark ids done once their shard is actually flushed (crash-safe);
                # until then they're in-flight so a re-scan won't decode them again.
                flushed = writer.add(rid, feat, meta, f)
                if flushed:
                    mark_done(flushed)
                    inflight.difference_update(flushed)
                else:
                    inflight.add(rid)
                n_ok += 1
                if args.max and n_ok >= args.max:
                    log.info("reached --max %d", args.max)
                    stop.set()
                    break
            if not args.watch or stop.is_set():
                break
            stop.wait(args.poll_interval)     # idle until new downloads land
        flushed = writer.flush()  # final partial shard (on completion or Ctrl-C)
        mark_done(flushed)
        inflight.difference_update(flushed)
    finally:
        manifest_f.close()
        log.info("done. decoded=%d skipped=%d errors=%d total=%d",
                 n_ok, n_skip, n_err, len(done))


def build_parser():
    p = argparse.ArgumentParser(description="Decode .replay files into float16 zstd shards")
    p.add_argument("--replays", default="replays", help="dir of .replay files")
    p.add_argument("--out", default="shards", help="output dir for shards")
    p.add_argument("--fps", type=int, default=10, help="resample target frame rate")
    p.add_argument("--shard-size", type=int, default=500, help="replays per shard")
    p.add_argument("--zstd-level", type=int, default=10, help="zstd compression level")
    p.add_argument("--delete-raw", action="store_true",
                   help="delete each .replay after it's packed into a shard (stream-decode-discard)")
    p.add_argument("--watch", action="store_true",
                   help="keep running, re-scanning --replays for new files (couple with the downloader)")
    p.add_argument("--poll-interval", type=int, default=10,
                   help="seconds to wait between re-scans in --watch mode")
    p.add_argument("--max", type=int, default=0, help="stop after N decoded (0=all)")
    return p


if __name__ == "__main__":
    run(build_parser().parse_args())
