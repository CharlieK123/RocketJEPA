# HANDOFF — JEPA collapse investigation (2026-07-21)

Context file for continuing a Claude Code conversation on another machine. Read this
top-to-bottom before doing anything; it contains the verdict, the evidence, the open
threads, and where every artifact lives. The companion files are in `diagnostics/`.

---

## The question that started this

The user (Charlie) observed their JEPA's collapse metrics (effrank / latent_std in the
`main.py` epoch printout) **"collapse fast first, then come back"** and asked whether the
model is collapsing or slowly improving/functional, plus a review of the codebase for
areas of concern, backed by real tests.

## VERDICT (delivered and accepted)

**Not collapsing. The dip-then-recover is the normal two-phase JEPA bootstrapping
transient; the model is functional and learning.** But training runs in a measurably
suboptimal regime with two genuine code issues (ranked fixes below). Evidence was
gathered by actually training the user's exact `main.py` configuration on their real
shards for ~500 steps on their RTX 3060 (12 GB), with rich instrumentation.

### The mechanism (explains the user's observation)

- At init, the batch-mean (common-mode) component of the targets is ~400x the
  sample-specific component: "predict each slot's batch average" already scores
  smooth-L1 ~0.0026 while the untrained model scores 0.94.
- **Phase A (fast, ~first 50 steps):** predictor learns slot means; loss crashes
  0.94 -> 0.01; encoder needs almost no input-dependence, so it sheds dimensions ->
  effrank dives (looks like collapse). Prediction variance ratio (pvr) crashes
  0.022 -> 0.002 (predictions huddle on the mean).
- **Phase B (slow):** mean is exhausted; only sample-specific prediction reduces loss;
  requires input-dependent, higher-rank representations -> rank recovers gradually as
  the EMA target enriches. This is the "comes back."

### Evidence collected (all from the instrumented run; raw logs in `diagnostics/`)

1. **Linear probes never moved through the dip** (information retention). Ridge readout
   from the online encoder's pooled rep (20 tokens mean-pooled -> 512-d) on 8,192
   held-out windows from `shard_00499`, R^2 on held-out 25%:

   | step | ball_pos | ball_vel | player_pos | player_vel | opp_pos | throttle | steer |
   |------|----------|----------|------------|------------|---------|----------|-------|
   | raw ceiling | 1.000 | 0.999 | 1.000 | 0.999 | 1.000 | 0.999 | 1.000 |
   | 0 (random init) | 0.998 | 0.918 | 0.984 | 0.954 | 0.993 | 0.824 | 0.807 |
   | 100 | 0.998 | 0.920 | 0.987 | 0.956 | 0.994 | 0.826 | 0.809 |
   | 250 | 0.998 | 0.919 | 0.985 | 0.954 | 0.994 | 0.825 | 0.808 |
   | 500 | 0.997 | 0.905 | 0.976 | 0.943 | 0.988 | 0.823 | 0.802 |

   Target encoder at step 500 still at init levels (EMA lag). The small online dips at
   500 (~1pt) = encoder starting to reshape (its rep std jumped 0.037 -> 0.065), NOT
   collapse (collapse = tens of points, all targets, accelerating). **Watch this on
   long runs.** Caveat discussed at length: random init probes near-ceiling (random
   465->512-d maps are ~linearly lossless), so probes prove retention, not learning.

2. **Centered prediction-cosine (pcc) rose monotonically 0 -> 0.21** through the entire
   effrank dip (steps: 50:0.024, 100:0.055, 150:0.085, 200:0.123, 250:0.149, 300:0.163,
   400:0.214, 450:0.207). This is cos(z_hat - mean, z - mean) — the sample-specific
   prediction skill a collapsed model cannot fake (pins at 0 under collapse). THE
   decisive anti-collapse witness. Note the raw `pred_cos` logged by main.py is
   dominated by the common mode and is collapse-blind.

3. **No scale collapse:** target norms GREW 40 -> 113 (and climbing) — moving away from
   the zero-vector fixed point. (The growth itself is problem #1 below.)

4. **Dimensional concentration, not death:** effrank fell 32.8 -> 8.5 (still falling at
   step 450, bottom not yet observed when run was stopped) while per-dim latent std
   ROSE 0.032 -> 0.054. Variance concentrating into top directions = benign
   EMA-method rank dip; user's own longer runs showed the recovery.

5. **Raw loss was still ABOVE the batch-mean baseline at step 450** (0.006 vs 0.0025):
   the model had not yet beaten the trivial predictor in raw huber terms — early
   phase B. Partly caused by target-norm drift (EMA lags the inflating z̄). The huber
   "skill" score 1 - loss/base went from -370 (init) to ~-1.4 (step 450), rising.

6. **Data audit: clean.** Zero NaN/Inf in shards and through the WindowDataset
   pipeline; all 93 features healthy variance (flagged ones are binary flags — fine);
   velocity x10 raw scaling CONFIRMED (player vel maxes exactly +/-23008 = 2300 uu/s);
   windows diverse (window-window centered cos ~0.0, p95 0.33); boost-pad block sane.
   The dip is not data-driven.

## Ranked code findings (with fixes)

1. **Loss regresses onto unnormalized, norm-inflating targets.**
   `main.py:66-68` computes `z_hat_n`/`z_n` (normalized) then the loss uses RAW
   `z_hat, z`. CLAUDE.md documents the loss as normalized — likely a regression.
   Encoder (`model/encoder.py`) is pre-norm with NO final norm, so targets are a raw
   residual stream; measured norm inflation 40 -> 113 in 450 steps. I-JEPA layer-norms
   targets for exactly this (its `forward_target` applies `F.layer_norm` to target
   encoder output). **Fix (one line):**
   `loss = F.smooth_l1_loss(z_hat, F.layer_norm(z, (z.size(-1),)))`
   (or actually use `z_hat_n`/`z_n`). Expect shallower dip, faster phase-B entry,
   readable raw-loss.
2. **Batch-shared random mask.** `model/jepa.py:build_mask` draws ONE 5-of-20 mask per
   step for the whole batch. Per-sample masks = 2048x task diversity per step
   (vectorize: per-row randperm/argsort + gather; predictor PE stamping needs the
   per-sample masked indices).
3. **Task favors near-frame copying.** 5 scattered masked tokens of 20 in a 0.5 s
   window (10 fps, gap=1): a masked object is usually reconstructible from the same
   object 100 ms away. Consider masking whole object TRACKS (e.g. ball in all 5
   frames) or contiguous blocks in the (frame x object) grid, and/or `gap>1`.
4. **Flat LR (1e-4, no warmup/schedule) and fixed EMA momentum 0.997.** For long runs:
   1-2k step warmup + cosine; momentum ramp 0.997 -> 1.0 (I-JEPA: 0.996 -> 1.0).
5. **Monitoring can't distinguish collapse from health.** Current printout (loss, raw
   pred_cos, effrank, latent_std, offdiag) contains no discriminating metric; raw
   pred_cos is actively misleading. ADD (formulas below): centered pred-cos (rho),
   pred-variance-ratio (v), r2_res, huber skill. Optionally periodic linear probes.
6. **Dead/broken files:** `training/train.py` targets the OLD API
   `model(states, next_states)` (would crash; undefined names in train_maskh);
   `model/observation.py` has a literal syntax error (`def build` with no body).
   Nothing imports either — delete or quarantine. Import styles are mixed
   (`from model.jepa ...` in main.py vs `from RocketJEPA.model.encoder ...` in
   jepa.py) so the project only runs when BOTH the project parent and the RocketJEPA
   dir are on sys.path (PyCharm does this; plain `python main.py` breaks).
7. Minor: predictor's Transformer allocates an unused ObjectEncoder (~0.7M dead
   params, no grads — harmless); metrics encode all 20 tokens while training always
   sees 15 visible (minor distribution shift, monitoring-only).

## The metric framework developed in this chat (user engaged deeply with this)

Per masked slot, center over batch: r = z - z_mean, r_hat = z_hat - z_hat_mean.
- **rho** = mean cosine(r_hat, r) — direction skill ("pcc"). Collapse pins it at 0.
- **v** = Var(r_hat)/Var(r) — "daring dial"; ~0 during mean-phase, re-expands in
  phase B; loss-optimal value is v* = rho^2 (shrinkage) — model measured v=0.006 <<
  rho^2=0.044 at step 450 = under-committed, expected while chasing a moving target.
- **r2_res = 2*rho*sqrt(v) - v** — explained fraction of sample-specific target
  variance. THE one-glance number: exactly 0 through mean-learning (no matter how
  dramatic the loss curve), lifts off iff real sample-based learning happens
  (measured ~2.6% at step 450, rising from 0 at step ~50). Collapse = never leaves 0.
- **skill = 1 - loss / smooth_l1(z_mean.expand, z)** — huber skill vs the trivial
  predictor; crossing 0 = beating the batch-mean. Was -1.4 at step 450.
- Caveat: computed against the MOVING EMA target distribution — read sign and slope,
  not absolute level. The fixed-benchmark complement is the held-out FUTURE-STATE
  probe: rep(t) -> state(t+k) for k beyond the window; random features do badly there,
  so trained gains show as rising R^2 (retention probes are ceiling-saturated at init
  — random init even beats step-500 by ~1pt on them, discussed and expected).

Drop-in code:
```python
with torch.no_grad():
    rc  = z - z.mean(0, keepdim=True)
    rhc = z_hat - z_hat.mean(0, keepdim=True)
    rho = F.cosine_similarity(rhc, rc, dim=-1).mean()
    v   = rhc.var(0).mean() / rc.var(0).mean().clamp_min(1e-12)
    r2_res = 2 * rho * v.sqrt() - v
    skill  = 1 - loss / F.smooth_l1_loss(z.mean(0, keepdim=True).expand_as(z), z)
```

## Project state (as of this session)

- Machine used here: Windows 10, RTX 3060 12GB, `.venv` = py3.12, torch 2.11.0+cu128.
  Project at `C:\Users\charl\PycharmProjects\RocketJEPA_pc\RocketJEPA` (NOT a git repo).
- Data: `data/shards_250k/` — 500 zstd shards (~100 MB each), 59 base features,
  fp16, 10 fps, ~250k 1v1 replays; loader appends 34 boost-pad cols -> feat_dim 93,
  obj_lengths (12,24,16,41). CLAUDE.md is partially stale (predates the model/
  training/ package restructure; still describes the old single-file models.py).
- Live code path: `main.py` (config: latent 512, enc 7 blocks/hdim 2048/8 heads,
  pred 2 blocks/hdim 128, momentum .997, batch 2048, AdamW 1e-4/wd 1e-5, WINDOW=5,
  MIRROR=False) -> `model/jepa.py`, `model/encoder.py`, `loader.py`
  (build_window_loader, normalize="physical", pad_state=True), `boost_pad_state.py`,
  `training/functions.py` (effective_rank, batch_collapse_metrics).

## Artifacts in `diagnostics/` (copied from the session scratchpad)

- `probe_train.py` — the instrumented reproduction of main.py's training (the run
  behind all numbers above). **Hardcoded Windows paths at the top (sys.path inserts +
  SHARDS + EVAL_SHARD) — fix for the new machine.** Probes at checkpoints, metrics
  every 10 steps.
- `probe_train_fixed.py` — READY BUT NEVER RUN: identical config except
  `loss = smooth_l1(z_hat, layer_norm(z))` (fix #1), 1500-step cap, writes
  metrics_fixed.jsonl / probes_fixed.jsonl. Imports from probe_train.py.
- `analyze.py` — summarizes metrics.jsonl/probes.jsonl into the phase tables.
- `data_audit.py` — the shard/pipeline audit (NaN, per-feature stats, window diversity).
- `metrics.jsonl` — full per-10-step log of the baseline run (steps 1-450ish; run was
  killed at user request at ~step 450-500 after verdict; no model checkpoint saved).
- `probes.jsonl` — probe R^2 at steps 0/100/250/500 (+raw ceiling; online AND target).

## Open threads / natural next steps

1. **A/B the layer-norm fix**: run `probe_train_fixed.py` (~20-30 min on a 3060) and
   compare dip depth / pcc slope / skill crossing vs `metrics.jsonl`. Was offered;
   user hasn't said yes yet.
2. **Future-state probe** (rep(t) -> state(t+k)): ~15-line addition to probe_train.py
   probe targets; gives the "is the encoder itself improving" curve that retention
   probes structurally cannot. Was suggested; user interested but not yet requested.
3. Apply fixes #1/#2 to main.py proper + add the metric block to its epoch printout.
4. Where the effrank bottom/turn actually lands with the fix in (my run was stopped
   mid-dip at 8.5; user's own longer runs are the source for "it comes back").
5. Longer-horizon question from CLAUDE.md: new symmetric-schema corpus + MIRROR=True
   augmentation; loader supports it, untested in training.

## Conversation/user notes (for tone continuity)

- User got frustrated mid-session by long waits + status-update messages that ended
  without answers ("YOU JUST STOP TALKING BEFORE GIVING ME AN ANSWER LIKE 4 TIMES").
  Lesson applied: front-load the verdict with evidence-so-far, don't gate the answer
  on a still-running job, keep waiting-state messages to zero or one line. They're
  technically sharp (caught the random-init-beats-trained subtlety in the probe table
  immediately, independently proposed the explained-variance phase metric) — go deep
  on mechanism, skip hand-holding.
- Training probe pace on the 3060 was ~2-3.5 s/step at batch 2048 (metric overhead +
  shard-transition stalls included) — budget wall-clock accordingly before promising
  step counts.
