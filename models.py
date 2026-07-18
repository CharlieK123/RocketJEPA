import torch
import torch.nn as nn
from copy import deepcopy
import torch.nn.functional as F
from functions import effective_rank, batch_collapse_metrics
import numpy as np

class ObjectEncoder(nn.Module):
    def __init__(self, obj_lengths, latent_dim, hdim):
        super().__init__()

        # build a unique single layer MLP which takes the length of the object
        # and upscales it to the necessary residual dim to be used as a token
        embedding = lambda i: nn.Sequential(nn.Linear(obj_lengths[i], hdim), nn.GELU(), nn.Linear(hdim, latent_dim))

        self.objects = len(obj_lengths)
        self.object_projections = nn.ModuleList([embedding(i) for i in range(self.objects)])

    def build(self, x):
        # parse all of the data and group it into their respective objects
        # note this line assumes x is a [B, 73] dim vector for simplicity
        objects = ball_vec, player_vec, opponent_vec, env_vec = x[:, :12], x[:, 12:34], x[:, 34:50], x[:, 50:73]

        # take every object and push it through its respective encoder
        # then concat all 4 objects into one tensor [B, OBJS, DIM]
        encoded_state = []
        for i, obj in enumerate(objects):
            proj = self.object_projections[i]
            encoded_state.append(proj(obj))

        return torch.stack(encoded_state, dim=1)


class Transformer(nn.Module):
    def __init__(self, blocks, residual_dim, hidden_dim, att_heads, obj_lengths, emb_dim, proj=False):
        super().__init__()

        # make essential variables class global
        self.blocks = blocks
        self.dim = residual_dim
        self.proj = proj

        # otherwise pytorch will error
        if residual_dim % att_heads != 0:
            raise ValueError("residual_dim must be divisible by att_heads")

        # FFN: 1 layer w GeLU activation
        # MHA: n head attention
        # norm: RMS norm
        ffn = lambda: nn.Sequential(nn.Linear(residual_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, residual_dim))
        att = lambda: nn.MultiheadAttention(self.dim, att_heads, batch_first=True)
        norm = lambda: nn.RMSNorm(self.dim, eps=1e-6)

        # Transformer assumes the data is already encoded in shape [B, OBJS * HIST, DIM]
        self.embedding = ObjectEncoder(obj_lengths, residual_dim, emb_dim)

        self.attention = nn.ModuleList([att() for _ in range(blocks)])
        self.ffn = nn.ModuleList([ffn() for _ in range(blocks)])
        self.norm = nn.ModuleList([norm() for _ in range(blocks * 2)])

        # if this instance is used as the projector then you need to make 5 extra query vectors
        # 5 is a hardcoded number for the history, these are the important ones and only
        # ones used after the projector is done
        if proj is not False:
            self.mask_queries = nn.Parameter(torch.randn(5, residual_dim))

    def block(self, x, i):

        # simple Pre-Norm transformer

        norm_1 = self.norm[2 * i]
        norm_2 = self.norm[2 * i + 1]
        attention = self.attention[i]
        feedforward = self.ffn[i]

        norm_out = norm_1(x)
        att_out, _ = attention(norm_out, norm_out, norm_out, need_weights=False)

        x = x + att_out

        norm_out = norm_2(x)
        ff_out = feedforward(norm_out)

        x = x + ff_out

        return x

    def forward(self, x):
        # check if projection is necessary if so add the query vectors
        if self.proj:
            queries = self.mask_queries.unsqueeze(0).expand(x.size(0), -1, -1)
            x = torch.cat((x, queries), dim=1)

        for i in range(self.blocks):
            x = self.block(x, i)

        # if proj was completed only return the desired context rich query vectors
        if self.proj:
            x = x[:, -self.mask_queries.size(0):]

        return x


class FFN(nn.Module):
    def __init__(self, in_dim, out_dim, h_layers, h_dim):
        super().__init__()

        layers = []

        # input layer
        layers.append(nn.Linear(in_dim, h_dim))
        layers.append(nn.Tanh())

        # hidden layers
        for i in range(h_layers - 1):
            layers.append(nn.Linear(h_dim, h_dim))
            layers.append(nn.GELU())

        # output linear layer
        layers.append(nn.Linear(h_dim, out_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)



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
        tokenised_history = []

        for i in range(state_history.size(1)):
            tokenised_history.append(self.encoder.embedding.build(state_history[:, i, :]))

        state = torch.stack(tokenised_history, dim=1).flatten(1, 2)

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

        # then predict the masked tokens latents
        if use_mask:
            latent_t = self.predictor(latent_t)

        with torch.no_grad():
            true_latent_t = self.target_encoder(state)

            if use_mask:
                true_latent_t = true_latent_t[:, masked_indices]

        print(visible_tokens.shape, masked_tokens.shape if masked_tokens is not None else None, state.shape,
              true_latent_t.shape, latent_t.shape)

        return latent_t, true_latent_t

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