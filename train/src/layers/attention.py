import torch
import torch.nn as nn
from src.layers.embed import RotaryPositionalEmbeddings
import numpy as np
from math import sqrt


class TriangularCausalMask():
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask


class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.mask_flag:
            scores.masked_fill_(attn_mask.unsqueeze(1), -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return V.contiguous(), A
        else:
            return V.contiguous(), None


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, num_patches=None, RoPE=None, d_keys=None,
                 d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

        if RoPE:
            self.RoPE = RotaryPositionalEmbeddings(dim=d_model//n_heads)
            self.num_patches = num_patches
        else:
            self.RoPE = None

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).reshape(B, L, H, -1)
        keys = self.key_projection(keys).reshape(B, S, H, -1)
        values = self.value_projection(values).reshape(B, S, H, -1)

        if self.RoPE:
            q_enc_in = L // self.num_patches
            k_enc_in = S // self.num_patches
            q_index = torch.arange(self.num_patches, device=queries.device).unsqueeze(0).repeat(B, q_enc_in)
            k_index = torch.arange(self.num_patches, device=queries.device).unsqueeze(0).repeat(B, k_enc_in)
            queries = self.RoPE(queries, input_pos=q_index)
            keys = self.RoPE(keys, input_pos=k_index)
            values = self.RoPE(values, input_pos=k_index)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask
        )
        out = out.reshape(B, L, -1)

        return self.out_projection(out), attn


class AttentionLayer_w_dispatchers(nn.Module):
    def __init__(self, attention, d_model, n_heads, dispatchers, num_patches=None, RoPE=None, d_keys=None,
                 d_values=None):
        super(AttentionLayer_w_dispatchers, self).__init__()
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection_1 = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection_1 = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection_1 = nn.Linear(d_model, d_values * n_heads)
        self.query_projection_2 = nn.Linear(d_keys * n_heads, d_keys * n_heads)
        self.key_projection_2 = nn.Linear(d_keys * n_heads, d_keys * n_heads)
        self.value_projection_2 = nn.Linear(d_values * n_heads, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.dispatchers = dispatchers
        self.n_heads = n_heads

        if RoPE:
            self.RoPE = RotaryPositionalEmbeddings(dim=d_model//n_heads)
            self.num_patches = num_patches
        else:
            self.RoPE = None

    def forward(self, queries, keys, values, attn_mask):
        B, L, E = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        d = torch.randn((B, self.dispatchers, E), device=queries.device)
        d_queries = self.query_projection_1(d).reshape(B, self.dispatchers, H, -1)
        keys = self.key_projection_1(keys).reshape(B, S, H, -1)
        values = self.value_projection_1(values).reshape(B, S, H, -1)

        if self.RoPE:
            k_enc_in = S // self.num_patches
            k_index = torch.arange(self.num_patches, device=queries.device).unsqueeze(0).repeat(B, k_enc_in)
            keys = self.RoPE(keys, input_pos=k_index)
            values = self.RoPE(values, input_pos=k_index)

        d_out, d_attn = self.inner_attention(
            d_queries,
            keys,
            values,
            attn_mask
        )

        d_out = d_out.reshape(B, self.dispatchers, -1)
        queries = self.query_projection_2(queries).reshape(B, L, H, -1)
        d_keys = self.key_projection_2(d_out).reshape(B, self.dispatchers, H, -1)
        d_values = self.value_projection_2(d_out).reshape(B, self.dispatchers, H, -1)

        if self.RoPE:
            q_enc_in = L // self.num_patches
            q_index = torch.arange(self.num_patches, device=queries.device).unsqueeze(0).repeat(B, q_enc_in)
            queries = self.RoPE(queries, input_pos=q_index)

        out, attn = self.inner_attention(
            queries,
            d_keys,
            d_values,
            attn_mask
        )
        out = out.reshape(B, L, -1)

        return self.out_projection(out), attn
