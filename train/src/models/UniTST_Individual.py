import torch
from torch import nn
from src.models.UniTST import Model as UniTST


# For each channel, we have a separate model
class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.num_models = configs.enc_in
        self.models = nn.ModuleList([UniTST(configs) for _ in range(self.num_models)])

    def forward(self, x_enc, t, cond=None, cond_prob=None):
        dec_out = torch.zeros_like(x_enc).to(x_enc.device)
        # [bs x channel x embed_size]
        embed = torch.zeros(x_enc.size(0), x_enc.size(2), self.models[0].embed_size).to(x_enc.device)
        for i in range(self.num_models):
            dec_out[:, :, [i]], embed[:, [i], :] = self.models[i](x_enc[:, :, [i]], t, cond, cond_prob)
        return dec_out, embed

    def embed_predict(self, x_embed):
        dec_out = []
        for i in range(self.num_models):
            dec_out.append(self.models[i].embed_predict(x_embed[:, [i], :]))
        dec_out = torch.cat(dec_out, dim=1).permute(0, 2, 1)
        return dec_out