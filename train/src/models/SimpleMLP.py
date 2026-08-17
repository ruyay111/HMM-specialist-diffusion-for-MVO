import torch.nn as nn
import torch.nn.functional as F
from src.layers.norm import AdaLayerNorm


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.embed_size = configs.d_model

        self.d_model = configs.d_model
        self.dropout = nn.Dropout(configs.dropout)
        self.norm1 = AdaLayerNorm(self.d_model)
        self.norm2 = AdaLayerNorm(self.d_model)
        self.linear1 = nn.Linear(self.seq_len, configs.d_model)
        self.linear2 = nn.Linear(configs.d_model, configs.d_ff)
        self.linear3 = nn.Linear(configs.d_ff, configs.d_model)
        self.linear4 = nn.Linear(configs.d_model, self.seq_len)
        self.activation = F.relu if configs.activation == "relu" else F.gelu

    def basic_diffusion_forward(self, x_enc, t):
        y = x_embed = self.dropout(self.activation(self.norm1(self.linear1(x_enc.permute(0, 2, 1)), t)))
        y = self.dropout(self.activation(self.linear2(y)))
        y = self.dropout(self.linear3(y))
        x_embed = self.norm2(x_embed + y, t)
        x = self.linear4(x_embed)

        return x.permute(0, 2, 1), x_embed

    def forward(self, x_enc, t):
        if self.task_name == 'basic_diffusion' or self.task_name == 'diffusion_ts' or self.task_name == 'diffusion_denoised_x':
            return self.basic_diffusion_forward(x_enc, t)

    def embed_predict(self, x_embed):
        return self.linear4(x_embed).permute(0, 2, 1)

