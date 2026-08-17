import torch
from torch import nn
import math
from src.layers.encoder import Encoder, EncoderLayer
from src.layers.attention import FullAttention, AttentionLayer, AttentionLayer_w_dispatchers
from src.layers.embed import PatchEmbedding, MLPEmbedding
from src.layers.prediction import FlattenHead, AttentivePredictionHead


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
        self.num_patches = math.ceil((configs.seq_len - configs.patch_len) / configs.stride + 1)
        total_len = (self.num_patches - 1) * configs.stride + configs.patch_len
        padding = total_len - configs.seq_len

        # patching and embedding
        if configs.RoPE:
            self.patch_embedding = PatchEmbedding(
                configs.d_model, configs.patch_len, configs.stride, padding, configs.dropout, flatten=True, pos_embed=False)
        else:
            self.patch_embedding = PatchEmbedding(
                configs.d_model, configs.patch_len, configs.stride, padding, configs.dropout, flatten=True)

        if 'conditional' in configs.task_name:
            self.cond_embedding = nn.Embedding(configs.n_regimes + 1, configs.d_model)
            self.prevts_embedding = MLPEmbedding(configs.prev_len, configs.d_ff, configs.d_model, configs.dropout)
        else:
            self.cond_embedding = None
            self.prevts_embedding = None

        # Encoder
        if configs.dispatchers > 0:
            attention = FullAttention(False, attention_dropout=configs.dropout)
            if configs.RoPE:
                attention_layer = AttentionLayer_w_dispatchers(attention, configs.d_model, configs.n_heads, configs.dispatchers,
                                                               num_patches=self.num_patches, RoPE=True)
            else:
                attention_layer = AttentionLayer_w_dispatchers(attention, configs.d_model, configs.n_heads, configs.dispatchers)
        else:
            if configs.causal_mask:
                attention = FullAttention(True, attention_dropout=configs.dropout)
            else:
                attention = FullAttention(False, attention_dropout=configs.dropout)

            if configs.RoPE:
                attention_layer = AttentionLayer(attention, configs.d_model, configs.n_heads,
                                                 num_patches=self.num_patches, RoPE=True)
            else:
                attention_layer = AttentionLayer(attention, configs.d_model, configs.n_heads)

        self.encoder = Encoder(
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

        # Prediction Head
        self.head_nf = configs.d_model * self.num_patches
        if configs.ind_proj:
            if configs.attn_proj:
                self.head = AttentivePredictionHead(configs.d_model, self.seq_len, configs.n_heads, head_dropout=configs.dropout, individual=configs.enc_in)
                self.embed_size = configs.d_model
            else:
                self.head = FlattenHead(self.head_nf, self.seq_len, head_dropout=configs.dropout, individual=configs.enc_in)
                self.embed_size = self.head_nf
        else:
            if configs.attn_proj:
                self.head = AttentivePredictionHead(configs.d_model, self.seq_len, configs.n_heads, head_dropout=configs.dropout)
                self.embed_size = configs.d_model
            else:
                self.head = FlattenHead(self.head_nf, self.seq_len, head_dropout=configs.dropout)
                self.embed_size = self.head_nf

        # causal mask
        self.batch_size = configs.batch_size
        self.num_assets = 1 if configs.individual else configs.enc_in
        if self.causal_mask and self.dispatchers == 0:
            attn_mask = self.create_batch_causal_mask()
        else:
            L = self.num_patches * self.num_assets
            attn_mask = torch.ones((L, L), dtype=torch.bool)
        self.register_buffer('attn_mask', attn_mask)

        # channel embedding
        if configs.channel_embed and configs.individual == False:
            self.channel_embedding = nn.Embedding(configs.enc_in, configs.d_model)
        else:
            self.channel_embedding = None

    def basic_diffusion_forward(self, x_enc, t):
        # do patching and embedding
        x_enc = x_enc.permute(0, 2, 1)
        # [bs x nvars * patch_num x d_model]
        enc_out, n_vars = self.patch_embedding(x_enc)

        # Encoder
        # [bs x nvars * patch_num x d_model]
        if self.channel_embedding is not None:
            channel_embed = self.channel_embedding.weight.repeat_interleave(self.num_patches, dim=0).unsqueeze(
                0).repeat(enc_out.shape[0], 1, 1)
        else:
            channel_embed = None
        enc_out, attns = self.encoder(enc_out, t, attn_mask=self.attn_mask, channel_embed=channel_embed)
        # [bs x nvars x patch_num x d_model]
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2] // n_vars, enc_out.shape[-1]))
        # z: [bs x nvars x d_model x patch_num]
        enc_out = enc_out.permute(0, 1, 3, 2)

        # Decoder
        dec_out, embed = self.head(enc_out)  # z: [bs x nvars x target_window]
        dec_out = dec_out.permute(0, 2, 1)

        return dec_out, embed

    def conditional_diffusion_forward(self, x_enc, t, cond, cond_prob, prev_x):
        # do patching and embedding
        x_enc = x_enc.permute(0, 2, 1)
        # [bs x nvars * patch_num x d_model]
        enc_out, n_vars = self.patch_embedding(x_enc)
        if self.cond_type == 'ridx':
            cond_embed = self.cond_embedding(cond)  # [bs x time x cond]
            cond_embed = cond_embed.mean(dim=1).unsqueeze(1)  # [bs x 1 x cond]
        elif self.cond_type == 'rprob':
            cond_embed = torch.matmul(cond_prob, self.cond_embedding.weight)
            cond_embed = cond_embed.mean(dim=1).unsqueeze(1)  # [bs x 1 x cond]
        elif self.cond_type == 'prevts':
            cond_embed = self.prevts_embedding(prev_x.permute(0, 2, 1))  # [bs x channel x d_model]
            cond_embed = cond_embed.repeat_interleave(self.num_patches, dim=1)  # [bs x channel * patch_num x d_model]
        else:
            cond_embed = None

        # Encoder
        # [bs x nvars * patch_num x d_model]
        if self.channel_embedding is not None:
            channel_embed = self.channel_embedding.weight.repeat_interleave(self.num_patches, dim=0).unsqueeze(
                0).repeat(enc_out.shape[0], 1, 1)
        else:
            channel_embed = None
        enc_out, attns = self.encoder(enc_out, t, cond_embed=cond_embed, attn_mask=self.attn_mask,
                                      channel_embed=channel_embed)
        # [bs x nvars x patch_num x d_model]
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2] // n_vars, enc_out.shape[-1]))
        # z: [bs x nvars x d_model x patch_num]
        enc_out = enc_out.permute(0, 1, 3, 2)

        # Decoder
        dec_out, embed = self.head(enc_out)  # z: [bs x nvars x target_window]
        dec_out = dec_out.permute(0, 2, 1)

        return dec_out, embed

    def forward(self, x_enc, t, cond=None, cond_prob=None, prev_x=None):
        basic_diffusion = ['basic_diffusion', 'diffusion_ts', 'diffusion_denoised_x', 'diffusion_direct_x']
        if self.task_name in basic_diffusion:
            return self.basic_diffusion_forward(x_enc, t)
        elif 'conditional' in self.task_name:
            return self.conditional_diffusion_forward(x_enc, t, cond, cond_prob, prev_x)

    def embed_predict(self, x_embed):
        dec_out = self.head.embed_predict(x_embed)
        dec_out = dec_out.permute(0, 2, 1)
        return dec_out

    def create_batch_causal_mask(self):
        L = self.num_patches * self.num_assets
        single_mask = torch.ones((L, L), dtype=torch.bool)
        start_idxs = [i * self.num_patches for i in range(self.num_assets)]

        for i in range(L):
            idx = i % self.num_patches
            end_idxs = [start_idxs[j] + idx + 1 for j in range(self.num_assets)]
            for j in range(len(start_idxs)):
                single_mask[i, start_idxs[j]:end_idxs[j]] = False

        return single_mask.unsqueeze(0)
