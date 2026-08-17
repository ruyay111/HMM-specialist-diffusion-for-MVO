from src.layers.embed import SinusoidalPosEmb
import torch
from torch import nn


class AdaLayerNorm(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.emb = SinusoidalPosEmb(n_embd)
        self.silu = nn.SiLU()
        self.linear = nn.Linear(n_embd, n_embd*2)
        self.layernorm = nn.LayerNorm(n_embd, elementwise_affine=False)

    def forward(self, x, timestep, cond_embed=None, channel_embed=None):
        # PatchTST will have different dim 0 size
        if x.shape[0] != timestep.shape[0]:
            timestep = timestep.repeat_interleave(x.shape[0] // timestep.shape[0], dim=0)
        emb = self.emb(timestep).unsqueeze(1).repeat(1, x.shape[1], 1)
        if cond_embed is not None:
            emb = emb + cond_embed
        if channel_embed is not None:
            emb = emb + channel_embed
        emb = self.linear(self.silu(emb))
        scale, shift = torch.chunk(emb, 2, dim=2)
        x = self.layernorm(x) * (1 + scale) + shift
        return x