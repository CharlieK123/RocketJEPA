import torch
import torch.nn as nn
from copy import deepcopy
from model.encoder import Transformer


class JEPA(nn.Module):
    def __init__(self,
                 latent_dim=128,
                 encoder_blocks=2,
                 encoder_hdim=256,
                 encoder_attheads=4,
                 proj_blocks=2,
                 proj_hdim=128,
                 proj_attheads=4,
                 momentum=0.995,
                 obj_lengths=(12, 22, 16, 23),
                 emb_hdim=256
                 ):
        super().__init__()

        # Encoder model takes in all non masked tokens, then output contextual tokens
        self.encoder = Transformer(encoder_blocks, latent_dim, encoder_hdim, encoder_attheads, obj_lengths, emb_hdim)
        # the predictor takes these contextual tokens with the query masked ones (5 of them) then attemps to use the
        # context to predict how the masked objects were acting
        self.predictor = Transformer(proj_blocks, latent_dim, proj_hdim, proj_attheads, obj_lengths, emb_hdim, proj=True)
        # target takes the masked tokens and outputs their latent state
        self.target_encoder = deepcopy(self.encoder)

        self.target_encoder.requires_grad_(False)
        self.momentum = momentum
        self.objects = len(obj_lengths)

    def forward(self, state_history, use_mask=True):
        # take the state history [B, H, 73] and embed all states
        # resulting in [B, 5, 4, 73], then flatten to have [B, 20, 73], note pos enc needs to be added
        # assume masked is true for JEPA class (only used for training)
        state = self.encoder.embed(state_history)  # -> [B, 20, LATENT]

        # prepare the mask to be used
        if use_mask:
            mask = self.build_mask(state.size(1), state.device)
            visible_indices = (~mask).nonzero(as_tuple=True)[0]
            masked_indices = mask.nonzero(as_tuple=True)[0]

            visible_tokens = state[:, visible_indices]
            masked_tokens = state[:, masked_indices]
        else:
            visible_tokens = state
            masked_tokens = None
            masked_indices = None

        # encode the non masked tokens to get their context
        latent_t = self.encoder(visible_tokens)
        masked_latent_t = self.predictor(latent_t)

        # stop grad EMA target encoders true labels:
        with torch.no_grad():
            true_masked_latent_t = self.target_encoder(state)
            true_masked_latent_t = true_masked_latent_t[:, masked_indices]


        print(visible_tokens.shape, masked_tokens.shape if masked_tokens is not None else None, state.shape,
              true_masked_latent_t.shape, latent_t.shape)

        return masked_latent_t, true_masked_latent_t

    @torch.no_grad()
    def update_target_params(self):
        for new_params, old_params in zip(self.encoder.parameters(), self.target_encoder.parameters()):
            # works to make target new params an EMA of the true encoder.
            # theta_t = (m * theta_t-1) + (1 - m)(theta_t)
            # lerp is Linear Interpolate between the old params and the new params with weight 1-m
            # it ultimately is the same operation as the EMA above
            old_params.lerp_(new_params, weight=1.0 - self.momentum)

    def build_mask(self, num_tokens, device, num_masked=5):
        mask = torch.zeros(num_tokens, dtype=torch.bool, device=device)
        mask[torch.randperm(num_tokens, device=device)[:num_masked]] = True
        return mask