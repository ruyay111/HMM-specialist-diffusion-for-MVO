import torch
import torch.nn as nn
from src.layers.encoder import Encoder, EncoderLayer
from src.layers.attention import FullAttention, AttentionLayer
from src.layers.embed import DataEmbedding_inverted


class Model(nn.Module):
    """
    Paper link: https://arxiv.org/abs/2310.06625
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.embed_size = configs.d_model
        self.args = configs

        # Embedding
        self.enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.dropout)
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

        # Decoder
        self.projection = nn.Linear(configs.d_model, self.seq_len, bias=True)

    def basic_diffusion_forward(self, x_enc, t):
        _, _, N = x_enc.shape

        enc_out = self.enc_embedding(x_enc)
        enc_out, attns = self.encoder(enc_out, t)
        dec_out = self.projection(enc_out).permute(0, 2, 1)

        return dec_out, enc_out

    def forward(self, x_enc, t):
        if self.task_name == 'basic_diffusion' or self.task_name == 'diffusion_ts' or self.task_name == 'diffusion_denoised_x':
            return self.basic_diffusion_forward(x_enc, t)

    def embed_predict(self, x_embed):
        return self.projection(x_embed).permute(0, 2, 1)

    def update_d_model(self, d_model, device):
        self.embed_size = d_model
        d_ff = d_model * 4
        # Embedding
        self.enc_embedding = DataEmbedding_inverted(d_model, d_model, self.args.dropout).to(device)
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, attention_dropout=self.args.dropout),
                        d_model, self.args.n_heads),
                    d_model,
                    d_ff,
                    dropout=self.args.dropout,
                    activation=self.args.activation
                ) for _ in range(self.args.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        ).to(device)

        # Decoder
        self.projection = nn.Linear(d_model, d_model, bias=True).to(device)