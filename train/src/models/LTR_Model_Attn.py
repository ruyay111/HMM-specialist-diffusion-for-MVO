import torch
from torch import nn
from src.layers.encoder import *
from src.layers.attention import FullAttention, AttentionLayer


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        # Embedding
        self.embed = nn.Sequential(
            nn.Linear(configs.enc_in, configs.d_model),
            nn.SiLU(),
            nn.Linear(configs.d_model, configs.d_model)
        )

        # Encoder
        self.encoder = Encoder_wot(
            [
                EncoderLayer_wot(
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
        self.projection = nn.Linear(configs.d_model, 1, bias=True)

    def forward(self, x_enc):
        # Embedding
        x_enc = self.embed(x_enc)

        # Encoder
        enc_out, _ = self.encoder(x_enc)

        # Prediction Head
        return self.projection(enc_out).squeeze(-1)