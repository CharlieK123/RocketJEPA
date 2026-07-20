import torch
import torch.nn as nn
import math


STATES = 5

class PosEncoding(nn.Module):
    def __init__(self, dim, states, objects):
        super().__init__()

        self.states = states
        self.objects = objects

        self.register_buffer("state_pe", self.sinusoidal(states, dim))
        self.register_buffer("object_pe", self.sinusoidal(objects, dim))

    @staticmethod
    def sinusoidal(length, dim):
        pe = torch.zeros(length, dim)
        pos = torch.arange(length).float().unsqueeze(1)
        div = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))

        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[:pe[:, 1::2].shape[1]])
        return pe

    def forward(self, x):
        state_ids = torch.arange(self.states, device=x.device).repeat_interleave(self.objects)
        object_ids = torch.arange(self.objects, device=x.device).repeat(self.states)

        pe = self.state_pe[state_ids] + self.object_pe[object_ids]
        return x + pe.unsqueeze(0)


class ObjectEncoder(nn.Module):
    def __init__(self, obj_lengths, latent_dim, hdim):
        super().__init__()

        # build a unique single layer MLP which takes the length of the object
        # and upscales it to the necessary residual dim to be used as a token
        embedding = lambda i: nn.Sequential(nn.Linear(obj_lengths[i], hdim), nn.GELU(), nn.Linear(hdim, latent_dim))

        self.objects = len(obj_lengths)
        self.object_projections = nn.ModuleList([embedding(i) for i in range(self.objects)])
        # cumulative slice boundaries so build() splits the flat frame BY obj_lengths
        # instead of hardcoded widths -> works for any schema (73, 85, ...).
        self.offsets = [0]
        for length in obj_lengths:
            self.offsets.append(self.offsets[-1] + length)

    def build(self, x):
        # slice the flat per-frame vector into its objects using obj_lengths
        # (ball, self, opponent, env), project each, stack -> [B, OBJS, DIM]
        encoded_state = []
        for i in range(self.objects):
            obj = x[:, self.offsets[i]: self.offsets[i + 1]]
            encoded_state.append(self.object_projections[i](obj))

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
        self.pos = PosEncoding(residual_dim, states=STATES, objects=len(obj_lengths))

        self.attention = nn.ModuleList([att() for _ in range(blocks)])
        self.ffn = nn.ModuleList([ffn() for _ in range(blocks)])
        self.norm = nn.ModuleList([norm() for _ in range(blocks * 2)])

        # if this instance is used as the projector then you need to make 5 extra query vectors
        # 5 is a hardcoded number for the history, these are the important ones and only
        # ones used after the projector is done
        if proj is not False:
            self.mask_queries = nn.Parameter(torch.randn(STATES, residual_dim))

            self.proj_encode = lambda x: torch.cat((x, self.mask_queries.unsqueeze(0).expand(x.size(0), -1, -1)), dim=1)
            self.proj_decode = lambda x: x[:, -self.mask_queries.size(0):]

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
        # adds n query tokens to act as the contextual information for masked tokens
        if self.proj: x = self.proj_encode(x)

        for i in range(self.blocks):
            x = self.block(x, i)

        # ensures that only the masked query tokens are returned
        if self.proj: x = self.proj_decode(x)

        return x

    def embed(self, x):
        """
        :param x: tensor input in shape: (B, H, D_S)
        :return: embeds all state information into a usable tensor w pos encoding
        """
        tokenised_history = []

        # for every state in history use the build function to encode it into objects then concatenate
        for i in range(x.size(1)):
            tokenised_history.append(self.embedding.build(x[:, i, :]))

        tokens = torch.stack(tokenised_history, dim=1).flatten(1, 2)  # [B, H*OBJS, D_L]

        pos_tokens = self.pos(tokens)

        return pos_tokens  # [B, H*OBJS, D_L]
