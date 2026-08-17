import torch
from torch import nn
from src.layers.encoder import Encoder, EncoderLayer
from src.layers.attention import FullAttention, AttentionLayer
from src.layers.embed import PatchEmbedding
from src.layers.prediction import FlattenHead


class Model(nn.Module):
    """
    Paper link: https://arxiv.org/pdf/2211.14730.pdf
    """

    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        padding = configs.stride

        # patching and embedding
        self.patch_embedding = PatchEmbedding(
            configs.d_model, configs.patch_len, configs.stride, padding, configs.dropout)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, attention_dropout=configs.dropout),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for _ in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )

        # Prediction Head
        self.head_nf = self.embed_size = configs.d_model * int((configs.seq_len - configs.patch_len) / configs.stride + 2)
        self.head = FlattenHead(self.head_nf, self.seq_len, head_dropout=configs.dropout)

    def basic_diffusion_forward(self, x_enc, t):
        x_enc = x_enc.permute(0, 2, 1)
        # u: [bs * nvars x patch_num x d_model]
        enc_out, n_vars = self.patch_embedding(x_enc)

        # Encoder
        # z: [bs * nvars x patch_num x d_model]
        enc_out, attns = self.encoder(enc_out, t)
        # z: [bs x nvars x patch_num x d_model]
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        # z: [bs x nvars x d_model x patch_num]
        enc_out = enc_out.permute(0, 1, 3, 2)

        # Decoder
        dec_out, embed = self.head(enc_out)  # z: [bs x nvars x target_window]
        dec_out = dec_out.permute(0, 2, 1)

        return dec_out, embed

    def forward(self, x_enc, t):
        if self.task_name == 'basic_diffusion' or self.task_name == 'diffusion_ts' or self.task_name == 'diffusion_denoised_x':
            return self.basic_diffusion_forward(x_enc, t)

    def embed_predict(self, x_embed):
        return self.head.embed_predict(x_embed).permute(0, 2, 1)