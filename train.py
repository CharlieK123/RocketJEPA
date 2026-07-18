from functions import *

def train_ac(model, epochs, loader, optim, device="cpu"):
    model.to(device)
    model.train()

    for epoch in range(epochs):
        totals = {
            "loss": 0.0,
            "latent_std": 0.0,
            "latent_abs": 0.0,
            "prediction_cosine": 0.0,
            "identity_cosine": 0.0,
            "state_cosine": 0.0,
            "grad_norm": 0.0,
            'offdiag': 0.0,
            'effrank_enc': 0.0,
            'effrank_tar': 0.0
        }

        total_samples = 0

        for states, actions, next_states in loader:
            states = states.to(device, non_blocking=True)
            actions = actions.to(device, non_blocking=True)
            next_states = next_states.to(device, non_blocking=True)

            batch_size = states.shape[0]

            z_hat, z = model(states, next_states)

            z_hat_normalized = F.normalize(z_hat, dim=-1)
            z_normalized = F.normalize(z, dim=-1)

            loss = F.smooth_l1_loss(z_hat_normalized, z_normalized)

            optim.zero_grad(set_to_none=True)
            loss.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optim.step()
            model.update_target_params()

            # eval metrics
            with torch.no_grad():
                online_latent = model.encoder(states.unsqueeze(1)).squeeze(1)
                future_target = model.target_encoder(next_states.unsqueeze(1)).squeeze(1)

                latent_std = online_latent.std(dim=0, unbiased=False).mean()
                latent_abs = online_latent.abs().mean()

                prediction_cosine = F.cosine_similarity(z_hat, z, dim=-1).mean()
                identity_cosine = F.cosine_similarity(online_latent, future_target, dim=-1).mean()
                state_cosine = F.cosine_similarity(states, next_states, dim=-1).mean()

                batch_offdiag_sim = batch_collapse_metrics(online_latent)
                eff_rank = effective_rank(online_latent)
                target_eff_rank = effective_rank(future_target)  # check target encoder separately too

            totals["loss"] += loss.item() * batch_size
            totals["latent_std"] += latent_std.item() * batch_size
            totals["latent_abs"] += latent_abs.item() * batch_size
            totals["prediction_cosine"] += prediction_cosine.item() * batch_size
            totals["identity_cosine"] += identity_cosine.item() * batch_size
            totals["state_cosine"] += state_cosine.item() * batch_size
            totals["grad_norm"] += float(grad_norm) * batch_size
            totals['effrank_enc'] += eff_rank * batch_size
            totals['effrank_tar'] += target_eff_rank * batch_size
            totals['offdiag'] += batch_offdiag_sim * batch_size

            total_samples += batch_size

        avg = {name: value / total_samples for name, value in totals.items()}

        print(
            f"Epoch {epoch + 1:03d} | "
            f"loss={avg['loss']:.6f} | "
            f"predict future latent sim={avg['prediction_cosine']:.4f} | "
            f"latent sim={avg['identity_cosine']:.4f} | "
            f"state sim={avg['state_cosine']:.4f}"
        )

        print(
            f"latent_std={avg['latent_std']:.4f} | "
            f"latent_abs={avg['latent_abs']:.4f} | "
            f"grad_norm={avg['grad_norm']:.4f} | "
            f"eff_rank_enc={avg['effrank_enc']:.2f} | "
            f"eff_rank_tar={avg['effrank_tar']:.2f} | "
            f"offdiag_sim={avg['offdiag']:.4f}"
        )

        print('-----------------------------\n')

def train_mask(model, epochs, loader, optim, device="cpu"):
    model.to(device)
    model.train()

    for epoch in range(epochs):
        totals = {
            "loss": 0.0,
            "latent_std": 0.0,
            "latent_abs": 0.0,
            "prediction_cosine": 0.0,
            "identity_cosine": 0.0,
            "state_cosine": 0.0,
            "grad_norm": 0.0,
            'offdiag': 0.0,
            'effrank_enc': 0.0,
            'effrank_tar': 0.0
        }

        total_samples = 0

        for states, in loader:
            states = states.to(device, non_blocking=True)

            batch_size = states.shape[0]

            z_hat, z = model(states)

            z_hat_normalized = F.normalize(z_hat, dim=-1)
            z_normalized = F.normalize(z, dim=-1)

            loss = F.smooth_l1_loss(z_hat_normalized, z_normalized)

            optim.zero_grad(set_to_none=True)
            loss.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optim.step()
            model.update_target_params()

            # eval metrics
            break
            with torch.no_grad():
                online_latent = model.encoder(states)

                latent_std = online_latent.std(dim=0, unbiased=False).mean()
                latent_abs = online_latent.abs().mean()

                prediction_cosine = F.cosine_similarity(z_hat, z, dim=-1).mean()
                identity_cosine = F.cosine_similarity(online_latent, future_target, dim=-1).mean()

                batch_offdiag_sim = batch_collapse_metrics(online_latent)
                eff_rank = effective_rank(online_latent)
                target_eff_rank = effective_rank(future_target)  # check target encoder separately too

            totals["loss"] += loss.item() * batch_size
            totals["latent_std"] += latent_std.item() * batch_size
            totals["latent_abs"] += latent_abs.item() * batch_size
            totals["prediction_cosine"] += prediction_cosine.item() * batch_size
            totals["identity_cosine"] += identity_cosine.item() * batch_size
            totals["state_cosine"] += state_cosine.item() * batch_size
            totals["grad_norm"] += float(grad_norm) * batch_size
            totals['effrank_enc'] += eff_rank * batch_size
            totals['effrank_tar'] += target_eff_rank * batch_size
            totals['offdiag'] += batch_offdiag_sim * batch_size

            total_samples += batch_size

        avg = {name: value / total_samples for name, value in totals.items()}

        print(
            f"Epoch {epoch + 1:03d} | "
            f"loss={avg['loss']:.6f} | "
            f"predict future latent sim={avg['prediction_cosine']:.4f} | "
            f"latent sim={avg['identity_cosine']:.4f} | "
            f"state sim={avg['state_cosine']:.4f}"
        )

        print(
            f"latent_std={avg['latent_std']:.4f} | "
            f"latent_abs={avg['latent_abs']:.4f} | "
            f"grad_norm={avg['grad_norm']:.4f} | "
            f"eff_rank_enc={avg['effrank_enc']:.2f} | "
            f"eff_rank_tar={avg['effrank_tar']:.2f} | "
            f"offdiag_sim={avg['offdiag']:.4f}"
        )

        print('-----------------------------\n')