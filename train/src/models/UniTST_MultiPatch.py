import copy

import torch
from torch import nn
import math
from src.layers.encoder import Encoder, EncoderLayer
from src.layers.attention import FullAttention, AttentionLayer, AttentionLayer_w_dispatchers
from src.layers.embed import PatchEmbedding, MLPEmbedding
from src.layers.prediction import PredictionHead


class Model(nn.Module):
    """
    Chenkai's Implementation of UniTST, No open-source implementation available
    Paper link: https://arxiv.org/abs/2406.04975
    """

    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.dispatchers = configs.dispatchers
        self.cond_type = configs.cond_type
        self.causal_mask = configs.causal_mask

        # calculate number of paddings
        powers_of_2 = []
        power = 2
        while (result := 2 ** power) <= configs.seq_len:
            powers_of_2.append(result)
            power += 1
        self.patch_len = copy.deepcopy(powers_of_2)
        self.stride = copy.deepcopy(powers_of_2)
        self.num_patches = [math.ceil((configs.seq_len - pl) / st + 1) for pl, st in zip(self.patch_len, self.stride)]
        total_len = [(np - 1) * st + pl for np, st, pl in zip(self.num_patches, self.stride, self.patch_len)]
        padding = [tl - configs.seq_len for tl in total_len]

        # patching and embedding
        self.patch_embedding_list = nn.ModuleList()
        for pl, st, pad in zip(self.patch_len, self.stride, padding):
            if configs.RoPE:
                self.patch_embedding_list .append(PatchEmbedding(
                    configs.d_model, pl, st, pad, configs.dropout, flatten=True, pos_embed=False))
            else:
                self.patch_embedding_list .append(PatchEmbedding(
                    configs.d_model, pl, st, pad, configs.dropout, flatten=True))

        if 'conditional' in configs.task_name:
            self.cond_embedding = nn.Embedding(configs.n_regimes + 1, configs.d_model)
            self.prevts_embedding = MLPEmbedding(configs.prev_len, configs.d_ff, configs.d_model, configs.dropout)
        else:
            self.cond_embedding = None
            self.prevts_embedding = None

        # Encoder
        self.encoder_list = nn.ModuleList()
        for i in range(len(self.patch_len)):
            if configs.dispatchers > 0:
                attention = FullAttention(False, attention_dropout=configs.dropout)
                if configs.RoPE:
                    attention_layer = AttentionLayer_w_dispatchers(attention, configs.d_model, configs.n_heads, configs.dispatchers,
                                                                   num_patches=self.num_patches[i], RoPE=True)
                else:
                    attention_layer = AttentionLayer_w_dispatchers(attention, configs.d_model, configs.n_heads, configs.dispatchers)
            else:
                if configs.causal_mask:
                    attention = FullAttention(True, attention_dropout=configs.dropout)
                else:
                    attention = FullAttention(False, attention_dropout=configs.dropout)

                if configs.RoPE:
                    attention_layer = AttentionLayer(attention, configs.d_model, configs.n_heads,
                                                     num_patches=self.num_patches[i], RoPE=True)
                else:
                    attention_layer = AttentionLayer(attention, configs.d_model, configs.n_heads)

            encoder = Encoder(
                [
                    EncoderLayer(
                        attention_layer,
                        configs.d_model,
                        configs.d_ff,
                        dropout=configs.dropout,
                        activation=configs.activation
                    ) for _ in range(configs.e_layers)
                ],
                norm_layer=torch.nn.LayerNorm(configs.d_model)
            )
            self.encoder_list.append(encoder)

        # Prediction Head
        self.head_nf = [configs.d_model * np for np in self.num_patches]
        self.total_nf = self.embed_size = sum(self.head_nf)
        self.flatten = nn.Flatten(start_dim=-2)
        if configs.ind_proj:
            self.head = PredictionHead(self.total_nf, self.seq_len, head_dropout=configs.dropout, individual=configs.enc_in)
        else:
            self.head = PredictionHead(self.total_nf, self.seq_len, head_dropout=configs.dropout)

        # causal mask
        self.attn_mask_list = []
        self.batch_size = configs.batch_size
        self.num_assets = 1 if configs.individual else configs.enc_in
        for np in self.num_patches:
            if self.causal_mask and self.dispatchers == 0:
                attn_mask = self.create_batch_causal_mask(np)
                self.attn_mask_list.append(attn_mask)
            else:
                L = np * self.num_assets
                self.attn_mask_list.append(torch.ones((L, L), dtype=torch.bool))

        # channel embedding
        if configs.channel_embed and configs.individual == False:
            self.channel_embedding = nn.Embedding(configs.enc_in, configs.d_model)
        else:
            self.channel_embedding = None

    def basic_diffusion_forward(self, x_enc, t):
        x_enc = x_enc.permute(0, 2, 1)
        dec_out_list = []
        for i in range(len(self.patch_len)):
            enc_out, n_vars = self.patch_embedding_list[i](x_enc)

            if self.channel_embedding is not None:
                channel_embed = self.channel_embedding.weight.repeat_interleave(self.num_patches[i], dim=0).unsqueeze(
                    0).repeat(enc_out.shape[0], 1, 1)
            else:
                channel_embed = None

            enc_out, attns = self.encoder_list[i](enc_out, t, attn_mask=self.attn_mask_list[i].to(x_enc.device), channel_embed=channel_embed)
            enc_out = torch.reshape(
                enc_out, (-1, n_vars, enc_out.shape[-2] // n_vars, enc_out.shape[-1]))
            enc_out = enc_out.permute(0, 1, 3, 2)
            dec_out_list.append(self.flatten(enc_out))

        dec_out = torch.cat(dec_out_list, dim=-1)
        dec_out, embed = self.head(dec_out)

        return dec_out.permute(0, 2, 1), embed

    def conditional_diffusion_forward(self, x_enc, t, cond, cond_prob, prev_x):
        x_enc = x_enc.permute(0, 2, 1)
        dec_out_list = []
        for i in range(len(self.patch_len)):
            enc_out, n_vars = self.patch_embedding_list[i](x_enc)

            if self.cond_type == 'ridx':
                cond_embed = self.cond_embedding(cond)
                cond_embed = cond_embed.mean(dim=1).unsqueeze(1)
            elif self.cond_type == 'rprob':
                cond_embed = torch.matmul(cond_prob, self.cond_embedding.weight)
                cond_embed = cond_embed.mean(dim=1).unsqueeze(1)
            elif self.cond_type == 'prevts':
                cond_embed = self.prevts_embedding(prev_x.permute(0, 2, 1))
                cond_embed = cond_embed.repeat_interleave(self.num_patches[i], dim=1)
            else:
                cond_embed = None

            if self.channel_embedding is not None:
                channel_embed = self.channel_embedding.weight.repeat_interleave(self.num_patches[i], dim=0).unsqueeze(
                    0).repeat(enc_out.shape[0], 1, 1)
            else:
                channel_embed = None

            enc_out, attns = self.encoder(enc_out, t, cond_embed=cond_embed, attn_mask=self.attn_mask.to(x_enc.device),
                                          channel_embed=channel_embed)
            enc_out = torch.reshape(
                enc_out, (-1, n_vars, enc_out.shape[-2] // n_vars, enc_out.shape[-1]))
            enc_out = enc_out.permute(0, 1, 3, 2)
            dec_out_list.append(self.flatten(enc_out))

        dec_out = torch.cat(dec_out_list, dim=-1)
        dec_out, embed = self.head(dec_out)

        return dec_out.permute(0, 2, 1), embed

    def forward(self, x_enc, t, cond=None, cond_prob=None, prev_x=None):
        basic_diffusion = ['basic_diffusion', 'diffusion_ts', 'diffusion_denoised_x', 'diffusion_direct_x']
        if self.task_name in basic_diffusion:
            return self.basic_diffusion_forward(x_enc, t)
        elif 'conditional' in self.task_name:
            return self.conditional_diffusion_forward(x_enc, t, cond, cond_prob, prev_x)

    def embed_predict(self, x_embed):
        return self.head.embed_predict(x_embed).permute(0, 2, 1)

    def create_batch_causal_mask(self, num_patches):
        L = num_patches * self.num_assets
        single_mask = torch.ones((L, L), dtype=torch.bool)
        start_idxs = [i * num_patches for i in range(self.num_assets)]

        for i in range(L):
            idx = i % num_patches
            end_idxs = [start_idxs[j] + idx + 1 for j in range(self.num_assets)]
            for j in range(len(start_idxs)):
                single_mask[i, start_idxs[j]:end_idxs[j]] = False

        return single_mask.unsqueeze(0)
