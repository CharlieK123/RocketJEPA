"""main.py — JEPA pretraining entry point, wired to the real Rocket League shards.

Run:  python main.py     (set SHARDS to your local shard dir first)

Pipeline: build_window_loader streams the zstd shards -> 5-frame windows in the
canonical (blue) frame, physics-only + boost-pad state, physically normalized,
with the RLGym team-mirror augmentation (2x data). Each window feeds the masked
JEPA, which predicts the latents of randomly masked object-tokens from the rest.

The knobs below are meant for tweaking; the data<->model contract (feat_dim ==
sum(OBJ_LENGTHS)) is asserted so a schema mismatch fails loudly, not silently.
"""
import torch
import torch.nn.functional as F

from model.jepa import JEPA
from loader import build_window_loader
from training.functions import effective_rank, batch_collapse_metrics
import time

# --------------------------- config (tweak me) ----------------------------- #
SHARDS        = r"C:\Users\charl\PycharmProjects\RocketJEPA_pc\RocketJEPA\data\shards_250k"   # <- point at your local shard directory
WINDOW        = 5                    # frames per sample (keep 5: PosEncoding STATES assumes it)
BATCH_SIZE    = 2048
EPOCHS        = 100
LR            = 1e-4
WEIGHT_DECAY  = 1e-5
NUM_WORKERS   = 4
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"

# MIRROR = team-perspective augmentation (2x data). ONLY valid for the new symmetric
# schema (both cars have actions + forward/up orientation). The old 250k corpus is
# asymmetric (opponent has no actions, euler rotation) and CANNOT be mirrored -> keep
# MIRROR=False for it; set True once training on the new-schema corpus.
MIRROR = False
# obj_lengths is auto-derived from the shards' actual schema (ds.obj_lengths), so this
# file trains on EITHER the old 250k (12,24,16,41) or the new corpus (9,30,30,41)
# with no edits — the model just matches whatever the loader loaded.


def build():
    loader, ds = build_window_loader(
        SHARDS, window=WINDOW, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
        pad_state=True, normalize="physical", mirror=MIRROR,
    )
    obj_lengths = ds.obj_lengths          # matches the loaded shards' schema

    model = JEPA(
        latent_dim=512, encoder_blocks=7, encoder_hdim=2048, encoder_attheads=8,
        proj_blocks=2, proj_hdim=128, proj_attheads=4, momentum=0.997,
        obj_lengths=obj_lengths, emb_hdim=256,
    ).to(DEVICE)
    optim = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    return loader, ds, model, optim


def train(loader, model, optim):
    model.train()
    for epoch in range(EPOCHS):
        tot = dict(loss=0., pred_cos=0., latent_std=0., effrank=0., offdiag=0., grad=0.)
        n = 0
        for window in loader:                          # [B, WINDOW, feat_dim]
            window = window.to(DEVICE, non_blocking=True)
            z_hat, z = model(window)                   # masked pred, target [B, n_masked, D]

            #z_hat_n = F.normalize(z_hat, dim=-1)
            #z_n = F.normalize(z, dim=-1)
            loss = F.smooth_l1_loss(z_hat, z)

            optim.zero_grad(set_to_none=True)
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optim.step()
            model.update_target_params()               # EMA target — after every step

            with torch.no_grad():
                b = window.size(0)
                # collapse metrics on the ONLINE encoder's PER-SAMPLE representation:
                # encode all 20 object-tokens (no mask) and mean-pool to one vector
                # per sample. This measures true per-state representational diversity.
                # (The old metric used z.reshape(B*n_masked, D) — dominated by the ~5
                # batch-shared PE slot-clusters, so effrank was pinned near ~5 and
                # couldn't distinguish collapse from healthy. effective_rank centers
                # internally; offdiag is centered here to drop the shared common-mode.)
                rep = model.encoder(model.encoder.embed(window)).mean(1)   # [B, D]
                rep_c = rep - rep.mean(0, keepdim=True)
                tot["loss"]       += loss.item() * b
                tot["pred_cos"]   += F.cosine_similarity(z_hat, z, dim=-1).mean().item() * b
                tot["latent_std"] += rep.std(0, unbiased=False).mean().item() * b
                tot["effrank"]    += effective_rank(rep) * b
                tot["offdiag"]    += batch_collapse_metrics(rep_c) * b
                tot["grad"]       += float(grad) * b
                n += b
                print(f"loss={loss.item():.5f} effrank={effective_rank(rep):.1f} "
                      f"latent_std={rep.std(0).mean().item():.4f}")

        a = {k: v / max(n, 1) for k, v in tot.items()}
        print(f"epoch {epoch + 1:03d} | loss={a['loss']:.5f} | pred_cos={a['pred_cos']:.4f} "
              f"| latent_std={a['latent_std']:.4f} | effrank={a['effrank']:.1f} "
              f"| offdiag={a['offdiag']:.4f} | grad={a['grad']:.3f}")


if __name__ == "__main__":                             # guard required for num_workers>0 on Windows
    loader, ds, model, optim = build()
    print(f"device={DEVICE} | feat_dim={ds.feat_dim} | obj_lengths={ds.obj_lengths} | "
          f"mirror={MIRROR} | shards={len(ds.files)} | params={sum(p.numel() for p in model.parameters()):,}")
    train(loader, model, optim)
