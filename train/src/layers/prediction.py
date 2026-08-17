import torch
import torch.nn as nn
from math import sqrt


class FlattenHead(nn.Module):
    def __init__(self, nf, target_window, head_dropout=0, individual=0):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=-2)
        if individual == 0:
            self.linear = nn.Linear(nf, target_window)
        else:
            self.linear = nn.ModuleList([nn.Linear(nf, target_window) for _ in range(individual)])
        self.dropout = nn.Dropout(head_dropout)
        self.target_window = target_window

    def forward(self, x):
        x_enc = self.flatten(x)
        if isinstance(self.linear, nn.ModuleList):
            x_out = torch.zeros((x_enc.size(0), x_enc.size(1), self.target_window), device=x.device)
            for i, linear in enumerate(self.linear):
                x_out[:, i, :] = linear(x_enc[:, i, :])
        else:
            x_out = self.linear(x_enc)
        x_out = self.dropout(x_out)
        return x_out, x_enc

    def embed_predict(self, x_embed):
        if isinstance(self.linear, nn.ModuleList):
            x_out = torch.zeros((x_embed.size(0), x_embed.size(1), self.target_window), device=x_embed.device)
            for i, linear in enumerate(self.linear):
                x_out[:, i, :] = linear(x_embed[:, i, :])
        else:
            x_out = self.linear(x_embed)
        x_out = self.dropout(x_out)
        return x_out


class PredictionHead(nn.Module):
    def __init__(self, d_in, d_out, head_dropout=0, individual=0):
        super().__init__()
        self.d_out = d_out
        if individual == 0:
            self.linear = nn.Linear(d_in, d_out)
        else:
            self.linear = nn.ModuleList([nn.Linear(d_in, d_out) for _ in range(individual)])
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        if isinstance(self.linear, nn.ModuleList):
            x_out = torch.zeros((x.size(0), x.size(1), self.d_out), device=x.device)
            for i, linear in enumerate(self.linear):
                x_out[:, i, :] = linear(x[:, i, :])
        else:
            x_out = self.linear(x)
        x_out = self.dropout(x_out)
        return x_out, x

    def embed_predict(self, x_embed):
        return self.forward(x_embed)


class AttentivePredictionHead(nn.Module):
    def __init__(self, d_model, d_out, n_heads, d_keys=None, d_values=None, head_dropout=0, individual=0):
        super().__init__()
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.n_heads = n_heads
        self.d_keys = d_keys
        self.d_values = d_values
        self.scale = 1.0 / sqrt(d_keys)

        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)

        if individual == 0:
            self.out_projection = nn.Linear(d_values * n_heads, d_out)
        else:
            self.out_projection = nn.ModuleList([nn.Linear(d_values * n_heads, d_out) for _ in range(individual)])
        self.dropout = nn.Dropout(head_dropout)

        self.pool_query = nn.Parameter(torch.randn(1, 1, d_model))
        self.d_out = d_out

    def forward(self, x):
        B, C, D, L = x.shape  # batch, channel, d_model, num_patches
        H = self.n_heads

        x = x.permute(0, 1, 3, 2).reshape(B * C, L, D)  # (batch * channel, num_patches, d_model)

        pool_query = self.pool_query.expand(B * C, -1, -1)  # (batch * channel, 1, d_model)

        queries = self.query_projection(pool_query).reshape(B * C, 1, H, self.d_keys)  # (batch * channel, 1, heads, d_keys)
        keys = self.key_projection(x).reshape(B * C, L, H, self.d_keys)  # (batch * channel, num_patches, heads, d_keys)
        values = self.value_projection(x).reshape(B * C, L, H, self.d_values)  # (batch * channel, num_patches, heads, d_values)

        scores = torch.einsum("blhd,bshd->bhls", queries, keys)  # (batch * channel, heads, 1, num_patches)
        scores *= self.scale

        attention_weights = self.dropout(torch.softmax(scores, dim=-1))  # (batch * channel, heads, 1, num_patches)

        pooled_values = torch.einsum("bhls,bshd->blhd", attention_weights, values)  # (batch * channel, 1, heads, d_values)

        pooled_values = pooled_values.reshape(B * C, -1).reshape(B, C, D)
        if isinstance(self.out_projection, nn.ModuleList):
            x_out = torch.zeros((B, C, self.d_out), device=x.device)
            for i, linear in enumerate(self.out_projection):
                x_out[:, i, :] = linear(pooled_values[:, i, :])
        else:
            x_out = self.out_projection(pooled_values)

        return x_out, pooled_values

    def embed_predict(self, x_embed):
        if isinstance(self.out_projection, nn.ModuleList):
            x_out = torch.zeros((x_embed.size(0), x_embed.size(1), self.d_out), device=x_embed.device)
            for i, linear in enumerate(self.out_projection):
                x_out[:, i, :] = linear(x_embed[:, i, :])
        else:
            x_out = self.out_projection(x_embed)

        return x_out


