# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Self-supervised pretraining of a JEPA (Joint Embedding Predictive Architecture) on Rocket League game states. The model learns latent dynamics by predicting a future state's embedding from the current state, with no reconstruction and no contrastive negatives — collapse is prevented by an EMA target encoder plus stop-gradient (BYOL/I-JEPA style).

The project is early-stage: model + training loop exist in `models.py`, metric helpers in `functions.py`, and `main.py` is an empty entry point. There is no dataset/DataLoader implementation yet — `train()` expects a `loader` yielding `(states, actions, next_states)` batches, which still needs to be written.

## Environment & running

- Python 3.13, PyTorch 2.13, NumPy 2.5, in the local `.venv` (no `requirements.txt`/`pyproject.toml` — dependencies are only pinned in the venv).
- Run: `.venv/bin/python main.py` (once an entry point is wired up). No build step, no test suite, no linter configured.
- `train(model, epochs, loader, optim, device="cuda")` defaults to CUDA; pass `device="cpu"` or `"mps"` when running on this Mac.

## Architecture (`models.py`)

- **`Transformer`** — pre-norm encoder blocks (RMSNorm → MHA → residual, RMSNorm → GELU FFN → residual). Operates on `(batch, seq, dim)`; `residual_dim` must be divisible by `att_heads`.
- **`FFN`** — the predictor MLP (Tanh on the input layer, GELU on hidden layers, linear output).
- **`JEPA`** — wires an `encoder` (Transformer), a `predictor` (FFN), and a `target_encoder` (frozen `deepcopy` of the encoder). `forward(state_t, state_tk)` returns `(latent_tk_hat, latent_tk)`: the online encoder+predictor produce the predicted future latent, the target encoder produces the true future latent under `no_grad`. `update_target_params()` EMA-updates the target via `lerp_` with `weight = 1 - momentum` — **must be called after every `optim.step()`**.

**Loss:** `smooth_l1_loss` on L2-normalized predicted vs. target latents. Gradients are clipped to `max_norm=1.0`.

**Seq-dim caveat:** `JEPA.forward` feeds `state_t` straight into the Transformer, but the in-loop eval metrics call `model.encoder(states.unsqueeze(1)).squeeze(1)` — i.e. metrics treat each state as `seq=1`. Any DataLoader/state representation must be consistent about whether states carry a sequence dimension.

## Collapse monitoring (`functions.py`)

JEPA's failure mode is representational collapse, so the training loop logs it every epoch:
- `effective_rank(embeddings)` — exp-entropy of the covariance eigenvalue spectrum; ranges 1 (collapsed) → D (full rank).
- `batch_collapse_metrics(embeddings)` — mean off-diagonal cosine similarity of normalized embeddings; near 1.0 means collapse. Expects a decent batch (N ≥ 64).

When touching the model or loss, keep these metrics meaningful — `identity_cosine`, `latent_std`, `effrank_enc`/`effrank_tar`, and `offdiag_sim` in the epoch printout are the primary signal that training hasn't collapsed.

## Data pipeline (planned — not yet implemented)

The pretraining corpus comes from [ballchasing.com](https://ballchasing.com), which hosts ~150M+ Rocket League replays. The pipeline has three stages: **download** raw `.replay` files via the ballchasing API → **decode** the binary with `carball` into structured game-state frames → **process** into the compact tensors the JEPA `loader` consumes. The `(states, actions, next_states)` batches the training loop expects are the output of this pipeline.

**Scope decisions (locked in):**
- **Target mode: 1v1 (Duel) only** — the bot is a 1s bot, so only pull `playlist=ranked-duels` (and optionally unranked duels). This keeps the state schema fixed at *ball + 2 cars* and avoids mixing 2v2/3v3 geometries.
- **All ranks** — no rank filter; the corpus should span bronze→SSL so the model sees the full behavior distribution.
- **Account tier: GC Patron** (highest) — see throughput math below.
- **Output: a single PyTorch tensor per frame-pair** carrying *all* available state information (see stage 3). Downstream RL fine-tuning is handled separately; this pipeline just needs to emit tensors.
- **Data source: the ballchasing API, single account/token** (decided). Faster alternatives were considered and explicitly rejected: RocketSim synthetic generation and pre-decoded Kaggle/HuggingFace datasets. Multi-token scraping is also out of scope for now. This means the ~2 downloads/sec file cap is an accepted, hard throughput constraint — the corpus grows over a multi-week continuous scrape, not in a burst.

### 1. Downloading from the ballchasing.com API

> **Implemented in `get_replays.py`** — a resumable, date-windowed downloader (SQLite manifest + JSON checkpoint, 2/sec file limiter, `429` backoff, atomic `.part` writes). Token via `$BALLCHASING_TOKEN` / `--token` / `.ballchasing_token`. Only dependency: `requests`. It only fills a dir with `{id}.replay` files; decode is separate.


- **Docs:** https://ballchasing.com/doc/api. **Auth:** every request needs an `Authorization: <token>` header (token from the ballchasing account/upload settings page). Store the token in an env var (e.g. `BALLCHASING_TOKEN`), never hardcode it.
- **List replays:** `GET /replays` with filters (`player-name`/`player-id`, `playlist`, `season`, `min-rank`/`max-rank`, `map`, `replay-date-after/before`, `sort-by`, `count` up to 200). Cursor-paginate by following the `next` URL in each response — there is no offset paging.
- **Download raw file:** `GET /replays/{id}/file` returns the binary `.replay`. Stream it to disk in chunks (files are relatively large). Only public replays (or your own) are downloadable. **There is NO bulk/batch/zip download endpoint** — files come one request at a time. Metadata *listing* is batched (200/page) but the file bytes are not.
- **Concurrency does not beat the rate limit:** the 2 downloads/sec cap is a server-side per-token aggregate, so parallel requests on one token just yield `429`s past 2/sec. Use ~2 concurrent connections only to stay *saturated* at the cap. The only way to genuinely multiply file throughput is **multiple GC-tier tokens/accounts run in parallel** (each gets its own 2/sec), or sourcing bulk replays outside ballchasing.
- **Rate limits (per Patreon tier)** — this is the throughput bottleneck for a millions-scale scrape:

  | Tier | Calls/sec | Hourly cap | File-download cap |
  |------|-----------|------------|-------------------|
  | Standard (free) | 2 | 500/hr | ~200/hr |
  | Gold | 2 | 1,000/hr | higher |
  | Diamond | 4 | 2,000/hr | higher |
  | Champion | 8 | unlimited | 1–2/sec |
  | GC | 16 | unlimited | 1–2/sec |

  File downloads are throttled more aggressively than metadata calls. **Confirmed for the GC tier: `GET /replays/{id}/file` is capped at 2 downloads/sec with NO hourly cap** (metadata calls get 16/sec). So the binding constraint is the 2/sec file rate, not an hourly quota:
  - 2 files/sec ≈ **7,200/hr ≈ ~172,000/day** running continuously.
  - e.g. ~1M replays ≈ 6 days; ~5M ≈ ~29 days of uninterrupted scraping.

  This makes "millions" feasible but multi-week — the downloader must be a **long-running, resumable daemon**, not a one-shot script. Build it to **checkpoint progress** (persist the last `next` cursor + a set of already-fetched replay IDs) so it survives interruption and never re-downloads. Respect `429`/`Retry-After` with backoff, and run it on the hosted machine (not the laptop) so it can run 24/7. Downloading and decoding should be decoupled stages (a queue/directory of pending `.replay` files) so decode throughput never blocks the download rate and vice-versa.
- Existing wrappers to consider rather than reimplementing: `pychasing`, `python-ballchasing`, `ballchaser` (all on PyPI). They handle tier-aware rate limiting.

### 2. Decoding with carball — VERIFIED WORKING RECIPE

> **Implemented in `decode_replays.py`** (stages 2 AND 3). Runs in a **separate decode venv** `.venv-decode` (py3.11, sprocket-carball, zstandard) — NOT the training `.venv` (dep conflict). Invoke as `.venv-decode/bin/python decode_replays.py`. Reads `replays/*.replay` → per-frame DataFrame → 10fps float16 feature tensor → zstd shards in `shards/`. Resumable (manifest `decoded.txt`); `--delete-raw` enables stream-decode-discard.

Decode turns the opaque `.replay` binary into a **pandas DataFrame** of per-frame game state. The recipe below was tested end-to-end on a real 1v1 replay (5231 frames × 63 columns) and is the confirmed path — don't re-derive it.

- **Use `sprocket-carball`, NOT `carball`.** The original [SaltieRL/carball](https://github.com/SaltieRL/carball) is dead on modern systems: it hard-pins `numpy==1.18.2`, which won't build with current setuptools on *any* Python (`CCompiler not defined`). The maintained fork **`sprocket-carball`** (PyPI, v1.2.2) uses modern numpy 2.4 / pandas 3.0 / `sprocket-boxcars-py` and installs cleanly. Import name is still `carball`.
- **Python 3.11 in an ISOLATED venv, created with `uv`.** carball's deps (pandas 3, protobuf 5) would conflict with the training `.venv` (Torch 2.13 / NumPy 2.5), so decode runs in its own env. On this Mac, Homebrew's Python 3.11 is broken (`pyexpat`/libexpat mismatch) — use `uv venv --python 3.11` which downloads a clean standalone CPython. (On the Linux host this is a non-issue; prebuilt wheels just work.)
- **Bypass carball's analysis layer** — `carball.analyze_replay_file()` crashes on a pandas-3.0 incompatibility (`fillna(method=...)`) in boost-pickup event detection. We only need the raw DataFrame, not events/stats, so skip it:
  ```python
  from carball.decompile_replays import decompile_replay
  from carball.json_parser.game import Game
  from carball.analysis.analysis_manager import AnalysisManager
  _json = decompile_replay(replay_path)
  game = Game(); game.initialize(loaded_json=_json)
  df = AnalysisManager(game)._initialize_data_frame(game)   # DataFrame only, no events
  ```
- **DataFrame layout:** columns are a MultiIndex `(actor, field)`. Actors = `'ball'`, `'game'`, and one per player (by name). Fields:
  - `ball`: `pos_x/y/z`, `vel_x/y/z`, `ang_vel_x/y/z`, `rot_x/y/z`, `hit_team_no`
  - each player: `pos_*`, `vel_*`, `ang_vel_*`, `rot_*`, `boost`, `throttle`, `steer`, `handbrake`, `jump_active`, `dodge_active`, `double_jump_active`, `boost_active`, `ball_cam`, `ping`
  - `game`: `time`, `delta`, `seconds_remaining`, `is_overtime`, `ball_has_been_hit`
  - **Units caveat (nail down in stage 3):** raw velocities come out large (e.g. ball `vel_x` ~ -22941) — carball reports velocity in a scaled unit, likely ×10 vs uu/s (RL max ball speed ≈ 6000 uu/s). Confirm the scale factor and normalize accordingly when building tensors.
- carball is CPU-heavy per replay. Decode in parallel across processes, and treat it as a **transform feeding stage 3** — decode → extract tensors → delete raw (stream-decode-discard), not a permanent artifact.

### 3. Storage strategy (disk is the hard constraint)

The corpus will exceed **any single local disk**, and the plan is to run download+decode on **hired high-spec cloud/host machines** (not the laptop). The pipeline must avoid hoarding intermediate data and must compress aggressively:

- **Don't keep everything.** Raw `.replay` files are smallish individually but millions reach the TB range; carball's decoded DataFrames are several× larger again. Storing both raw + decoded for the whole corpus is not viable.
- **Stream-decode-discard:** download a batch of raw replays → decode → extract only the JEPA feature tensors → write compact processed shards → **delete the raw `.replay` and the decoded DataFrame**. Only the final tensors persist. This caps peak disk at (rolling in-flight cache + growing processed shards), independent of corpus size.
- **The state tensor (1v1 — capture ALL available info):** for each frame build one float vector concatenating, at minimum:
  - **Ball:** position (x,y,z), linear velocity (x,y,z), angular velocity (x,y,z) — 9.
  - **Each of the 2 cars:** position (3), linear velocity (3), rotation (quaternion 4 or pitch/yaw/roll 3), angular velocity (3), boost amount (1), and boolean flags carball exposes — on-ground, has-jumped/double-jumped, supersonic, demolished (~4–5). ≈ 18–19 per car.
  - **Global:** seconds remaining, ball-possession/kickoff flags if available.
  - **Actions** (for the `(states, actions, next_states)` triple): the per-frame controller inputs carball reconstructs — throttle, steer, pitch, yaw, roll, jump, boost, handbrake.

  Normalize to physical ranges (field bounds ≈ ±4096/±5120/2044 uu, max car speed 2300 uu/s, boost 0–100) so the encoder sees roughly unit-scale inputs. Persist a small schema/version file describing column order so the tensor layout is reproducible.
- **Compact, compressed processed format — IMPLEMENTED.** `decode_replays.py` writes **zstd-compressed float16 shards** (default 500 replays/shard) at **10 fps**. Each shard is `shard_NNNNN.zst` (raw float16 bytes) + `shard_NNNNN.json` sidecar (shape, dtype, fps, `state_dim`/`action_dim`, `feature_names`, and a per-replay row index `{id, start, length, players}`). The training loader (still TODO) decompresses a shard, reshapes to `[frames, 52]`, and samples `(state_t, action_t, state_{t+k})` within each replay's row range. Requires `zstandard` in `.venv` (installed).
  - **Feature layout (59 dims/frame), fixed order:** `ball(12)` `[0:12]` + `self-player(24)` `[12:36]` + `opponent(16)` `[36:52]` + `env(7)` `[52:59]`. `obj_lengths=(12,24,16,7)`. **`env` is emitted as 7 and expanded to 23 downstream** (user derives boost pads etc.); note `self-player` is 24 (not the model's original 22), so `models.py` `obj_lengths` must be updated to match.
    - `ball(12)`: pos/vel/ang_vel/rot xyz.
    - `self-player(24)` = physical state (16: 12 kinematics + boost + jump_active/dodge_active/double_jump_active) + **all 8 actions** (`player.act.*`).
    - `opponent(16)`: 12 kinematics + boost + jump/dodge/double_jump flags. No actions.
    - `env(7)`: seconds_remaining, is_overtime, ball_has_been_hit, blue_score, orange_score, score_diff, kickoff. Score derived per-frame from `game.goals` (cumulative by team); kickoff is a heuristic (ball near center & not hit).
    - Players ordered **blue (self) then orange (opponent)** by `is_orange`.
  - **Actions — all 8, via `carball.controls.ControlsCreator`.** `throttle, steer, pitch, yaw, roll, jump, boost, handbrake`. throttle/steer/pitch/yaw/roll ∈ [-1,1]; jump/boost/handbrake ∈ {0,1} (already normalized). **pitch/yaw/roll are APPROXIMATED** from the car's angular velocity and only defined airborne (grounded → 0) — best-guess inputs, not ground-truth. They are the last 8 columns of the self-player block (`player.act.*`, indices 28–35).

### 3b. Loader — IMPLEMENTED (`loader.py`, training `.venv`)

Reads the zstd shards and yields **`state_t` and `action_t` aligned at each timestep** (row `t` of a shard already is state+action at `t`; pairs/windows are NOT stored on disk — form them here). `state` = all non-action columns (**51 dims**: ball + player physical state + opponent + env); `action` = the 8 `player.act.*` columns (**8 dims**). Key API:
- `iter_replays(shards_dir)` → `(id, state[n,51], action[n,8])` per replay (row ranges never cross replay boundaries).
- `TimestepDataset` (an `IterableDataset` — streams shards one at a time; a global row index over billions of frames would blow up memory) → per-timestep `(state, action)`, with shard-order + in-shard shuffle and DataLoader-worker sharding.
- `build_loader(...)` → `(DataLoader, dataset)` yielding `(state[B,51], action[B,8])`.
Verified on a synthetic shard (split correct, no action leakage into state). **NOTE:** the current model (`models.py`) wants 5-frame masked-history windows `[B,5,73]`, not single `(state, action)` timesteps — extend `TimestepDataset` to yield windows when wiring to `train()`.
  - **Normalization (do at load, values stored RAW in fp16):** positions are in uu and confirmed within field bounds (±4096 / ±5120 / 0–2044). **Velocities come out ~×10 scaled** vs uu/s — pin the exact factor when writing the loader. **Actions are 0–255 bytes** (128 = neutral for steer; 255 = full throttle) — normalize `(x-128)/128`-style, NOT already in [-1,1]. All raw values fit in fp16 (< 65504), so no overflow.
  - **Confirmed size:** ~183 KB/replay (10fps fp16 zstd) ≈ **30% of raw** → ~210 GB per 1M replays, vs ~700 GB raw. Naive float32/full-fps would be *larger* than raw; the win needs fp16 + zstd + 10fps together.
- **Storage tiers:** processed shards are the only long-lived artifact — land them in `data/states` (the additional working dir `/Users/ck/Documents/Projects/RocketLeaguePretraining/data/states`) or an object store / large hosted volume. Keep a small rolling cache for in-flight raw+decoded files on fast local disk. Consider gzip/zstd on shards and, for cloud hosting, object storage (S3/R2/GCS) as the durable tier with local disk as scratch.
- **Resumability & dedup:** persist a manifest of processed replay IDs so re-runs skip finished work; make each stage idempotent and restartable given the disk churn involved.
