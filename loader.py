"""loader.py — read decoded zstd shards for training (runs in the training .venv).

Each shard row is one timestep: at index t you get state_t and the action taken
at t, aligned. `state` = all non-action columns (ball + player physical state +
opponent + env); `action` = the 8 `player.act.*` columns (throttle, steer, pitch,
yaw, roll, jump, boost, handbrake). Pairs/windows are NOT stored on disk — form
them here from the per-replay sequence (row ranges never cross replay boundaries).

Shard format (written by decode_replays.py):
  shard_NNNNN.zst   zstd-compressed float16 bytes, reshape to [total_frames, feat_dim]
  shard_NNNNN.json  {dtype, shape, fps, obj_lengths, feat_dim, feature_names, replays[]}
                    replays[i] = {id, start, length, players, self_team}
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import torch
import zstandard as zstd
from torch.utils.data import IterableDataset, get_worker_info

ACTION_PREFIX = "player.act."


def load_shard(zst_path):
    """Return (arr[frames, feat_dim] float array, meta dict)."""
    zst_path = Path(zst_path)
    meta = json.loads(zst_path.with_suffix(".json").read_text())
    raw = zstd.ZstdDecompressor().decompress(zst_path.read_bytes())
    arr = np.frombuffer(raw, dtype=np.dtype(meta["dtype"])).reshape(meta["shape"])
    return arr, meta


def split_indices(feature_names):
    """(state_idx, action_idx) column indices. action = player.act.* ; state = rest."""
    action_idx = [i for i, n in enumerate(feature_names) if n.startswith(ACTION_PREFIX)]
    aset = set(action_idx)
    state_idx = [i for i in range(len(feature_names)) if i not in aset]
    return state_idx, action_idx


def iter_replays(shards_dir):
    """Yield (replay_id, state[n, state_dim] float32, action[n, action_dim] float32)
    for every replay across all shards. state[t]/action[t] are aligned at timestep t."""
    for zst in sorted(glob.glob(str(Path(shards_dir) / "shard_*.zst"))):
        arr, meta = load_shard(zst)
        s_idx, a_idx = split_indices(meta["feature_names"])
        for r in meta["replays"]:
            seg = arr[r["start"]: r["start"] + r["length"]]
            yield r["id"], seg[:, s_idx].astype(np.float32), seg[:, a_idx].astype(np.float32)


class TimestepDataset(IterableDataset):
    """Streams (state_t, action_t) timesteps across all shards.

    IterableDataset (not map-style) because the full corpus is billions of
    timesteps — building a global row index would blow up memory. Shards are
    decompressed one at a time; set shuffle=True to shuffle shard order and rows
    within each shard (buffered), which is enough decorrelation for SGD.
    """

    def __init__(self, shards_dir, shuffle=True, seed=0):
        self.files = sorted(glob.glob(str(Path(shards_dir) / "shard_*.zst")))
        if not self.files:
            raise FileNotFoundError(f"no shard_*.zst in {shards_dir}")
        self.shuffle = shuffle
        self.seed = seed
        # dims from the first shard's schema
        _, meta = load_shard(self.files[0])
        self.state_idx, self.action_idx = split_indices(meta["feature_names"])
        self.state_dim = len(self.state_idx)
        self.action_dim = len(self.action_idx)
        self.feature_names = meta["feature_names"]

    def __iter__(self):
        info = get_worker_info()
        files = self.files
        if info is not None:  # shard the shard-list across DataLoader workers
            files = files[info.id:: info.num_workers]
        rng = np.random.default_rng(self.seed + (info.id if info else 0))
        order = rng.permutation(len(files)) if self.shuffle else range(len(files))
        for fi in order:
            arr, meta = load_shard(files[fi])
            states = arr[:, self.state_idx].astype(np.float32)
            actions = arr[:, self.action_idx].astype(np.float32)
            rows = rng.permutation(len(arr)) if self.shuffle else range(len(arr))
            for t in rows:
                yield torch.from_numpy(states[t]), torch.from_numpy(actions[t])


def build_loader(shards_dir, batch_size=256, shuffle=True, num_workers=0, seed=0):
    """DataLoader yielding (state[B, state_dim], action[B, action_dim]) batches."""
    ds = TimestepDataset(shards_dir, shuffle=shuffle, seed=seed)
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, num_workers=num_workers), ds


# --------------------------------------------------------------------------- #
# Normalization (empirical z-score) — makes the weird raw scales irrelevant
# --------------------------------------------------------------------------- #
def compute_norm_stats(shards_dir, max_frames=1_000_000, save=True):
    """Streaming per-feature mean/std over up to `max_frames` frames. Constant
    columns get std=1 (no scaling). Saved to shards/norm_stats.npz if save=True."""
    files = sorted(glob.glob(str(Path(shards_dir) / "shard_*.zst")))
    if not files:
        raise FileNotFoundError(f"no shard_*.zst in {shards_dir}")
    s = ss = None
    n = 0
    names = None
    for f in files:
        arr, meta = load_shard(f)
        a = arr.astype(np.float64)
        names = meta["feature_names"]
        s = a.sum(0) if s is None else s + a.sum(0)
        ss = (a * a).sum(0) if ss is None else ss + (a * a).sum(0)
        n += len(a)
        if n >= max_frames:
            break
    mean = s / n
    std = np.sqrt(np.maximum(ss / n - mean ** 2, 1e-12))
    std = np.where(std < 1e-6, 1.0, std)          # constant cols -> leave as-is
    mean, std = mean.astype(np.float32), std.astype(np.float32)
    if save:
        np.savez(Path(shards_dir) / "norm_stats.npz",
                 mean=mean, std=std, feature_names=np.array(names))
    return mean, std


def _load_norm(shards_dir, normalize):
    """Resolve the `normalize` arg -> (mean, std) or None."""
    if normalize is False or normalize is None:
        return None
    if isinstance(normalize, (tuple, list)):
        return np.asarray(normalize[0], np.float32), np.asarray(normalize[1], np.float32)
    p = Path(shards_dir) / "norm_stats.npz"
    if not p.exists():
        raise FileNotFoundError("normalize=True but norm_stats.npz missing — "
                                "run compute_norm_stats(shards_dir) first")
    d = np.load(p, allow_pickle=True)
    return d["mean"], d["std"]


# --------------------------------------------------------------------------- #
# Windowed dataset — for the masked-history JEPA. Experiment with the horizon
# via `window` (#frames) and `gap` (frame spacing; window spans (window-1)*gap+1
# real frames, so gap dilates the temporal horizon without more tokens).
# --------------------------------------------------------------------------- #
def live_play_mask(seg, feature_names, resume="go", freeze_speed=1000.0):
    """Per-frame boolean over ONE replay's rows: True = live play, False = noise.

    Drops the two dead-time noise sources that carry no useful dynamics:
      * the post-goal explosion / cars-flung / celebration + reset window (from
        the frame a goal is scored, detected as a `blue_score`/`orange_score`
        increment);
      * the frozen kickoff countdown, where the game forces the cars stationary.

    `resume` controls where each dead span ENDS (where live play resumes):
      * "go"  (default) — resume the frame the car is unfrozen and free to move,
        i.e. right after the last frozen frame in the span. The kickoff
        drive-to-ball is KEPT. "Frozen" = self-car speed < `freeze_speed` (raw
        stored units; frozen countdown reads ~0-72, driving jumps to >2400, so
        the 1000 default separates them cleanly).
      * "first_touch" — resume only when the ball is next hit
        (`ball_has_been_hit` 0->1). The whole pre-touch kickoff is dropped too.

    Must be called per replay: scores are cumulative within a replay, so diffs
    across a replay boundary would be meaningless.
    """
    def col(name):
        return seg[:, feature_names.index(name)]

    bhh = col("env.ball_has_been_hit") >= 0.5
    total_score = col("env.blue_score") + col("env.orange_score")
    dead = ~bhh                                  # freeze + pre-first-touch
    goal_frames = np.where(np.diff(total_score) > 0)[0] + 1
    for g in goal_frames:                        # goals are few per game
        j = g
        while j < len(seg) and bhh[j]:           # explosion: goal -> reset
            dead[j] = True
            j += 1

    if resume == "go":
        # keep the drive-to-ball: within each dead span, un-drop everything
        # after the last frozen frame (the countdown ends -> car may move).
        vel = np.stack([col("player.vel_x"), col("player.vel_y"),
                        col("player.vel_z")], axis=1).astype(np.float32)
        speed = np.sqrt((vel ** 2).sum(axis=1))   # float32: fp16 vels overflow squared
        frozen = speed < freeze_speed
        d = dead.astype(np.int8)
        edges = np.where(np.diff(np.concatenate([[0], d, [0]])) != 0)[0]
        for k in range(0, len(edges), 2):        # each dead run [a, b)
            a, b = edges[k], edges[k + 1]
            fz = np.where(frozen[a:b])[0]
            if len(fz):                          # resume after last frozen frame
                dead[a + fz[-1] + 1: b] = False
    return ~dead


class WindowDataset(IterableDataset):
    """Yields full-frame windows [window, feat_dim] (float32), never crossing a
    replay boundary. `gap` spaces frames within a window; `step` strides between
    successive windows. `normalize`: True (load shards/norm_stats.npz), a
    (mean,std) tuple, or False.

    `drop_noise=True` (default) discards windows whose time-span overlaps a
    dead region (kickoff freeze / post-goal explosion) via `live_play_mask`. The
    check is on the whole window SPAN, not just its sampled frames, so a window
    can never straddle a goal and contain a discontinuity — critical for a
    dynamics model. Set False to keep every window (old behaviour). `resume`
    ("go"/"first_touch") is forwarded to `live_play_mask` — "go" keeps the
    kickoff drive-to-ball, "first_touch" drops the whole pre-touch kickoff."""

    def __init__(self, shards_dir, window=5, gap=1, step=1,
                 normalize=False, shuffle=True, seed=0, drop_noise=True,
                 resume="go"):
        self.files = sorted(glob.glob(str(Path(shards_dir) / "shard_*.zst")))
        if not self.files:
            raise FileNotFoundError(f"no shard_*.zst in {shards_dir}")
        self.window, self.gap, self.step = window, gap, step
        self.shuffle, self.seed = shuffle, seed
        self.drop_noise = drop_noise
        self.resume = resume
        _, meta = load_shard(self.files[0])
        self.feature_names = meta["feature_names"]
        self.feat_dim = meta["feat_dim"]
        norm = _load_norm(shards_dir, normalize)
        self.mean, self.std = (norm if norm is not None else (None, None))

    def __iter__(self):
        info = get_worker_info()
        files = self.files[info.id:: info.num_workers] if info else self.files
        rng = np.random.default_rng(self.seed + (info.id if info else 0))
        order = rng.permutation(len(files)) if self.shuffle else range(len(files))
        span = (self.window - 1) * self.gap + 1
        offs = np.arange(self.window) * self.gap
        for fi in order:
            arr, meta = load_shard(files[fi])
            a = arr.astype(np.float32)
            # live mask on RAW values (score/hit flags) before any normalization
            live = None
            if self.drop_noise:
                live = np.ones(len(arr), dtype=bool)
                for r in meta["replays"]:
                    lo, L = r["start"], r["length"]
                    live[lo:lo + L] = live_play_mask(arr[lo:lo + L], self.feature_names,
                                                     resume=self.resume)
            if self.mean is not None:
                a = (a - self.mean) / self.std
            # prefix sum of dead frames -> O(1) "is span [st,st+span) all live?"
            dead_ps = (np.concatenate([[0], np.cumsum(~live)])
                       if live is not None else None)
            starts = []
            for r in meta["replays"]:               # windows stay within a replay
                lo, L = r["start"], r["length"]
                if L < span:
                    continue
                cand = lo + np.arange(0, L - span + 1, self.step)
                if dead_ps is not None:
                    # reject any window whose SPAN [st, st+span) touches dead time
                    keep = (dead_ps[cand + span] - dead_ps[cand]) == 0
                    cand = cand[keep]
                starts.extend(cand.tolist())
            if self.shuffle:
                rng.shuffle(starts)
            for st in starts:
                yield torch.from_numpy(a[st + offs])   # [window, feat_dim]


def build_window_loader(shards_dir, window=5, gap=1, step=1, batch_size=64,
                        normalize=False, num_workers=0, shuffle=True, seed=0,
                        drop_noise=True, resume="go"):
    """DataLoader yielding windows [B, window, feat_dim] for the masked-history model.
    `drop_noise=True` filters out kickoff-freeze and post-goal windows; `resume`
    ("go"/"first_touch") sets where each dead span ends."""
    ds = WindowDataset(shards_dir, window, gap, step, normalize, shuffle, seed,
                       drop_noise, resume)
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, num_workers=num_workers), ds


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Sanity-check the shard loader")
    p.add_argument("--shards", default="shards")
    p.add_argument("--batch-size", type=int, default=8)
    args = p.parse_args()
    dl, ds = build_loader(args.shards, batch_size=args.batch_size)
    print(f"state_dim={ds.state_dim} action_dim={ds.action_dim} shards={len(ds.files)}")
    s, a = next(iter(dl))
    print("batch state:", tuple(s.shape), "action:", tuple(a.shape))
