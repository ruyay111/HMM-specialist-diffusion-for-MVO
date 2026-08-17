# https://github.com/sattarov/FinDiff/tree/main
from torch import nn
from src.layers.embed import timestep_embedding
from src.layers.encoder import MLP


class Model(nn.Module):
    """ Feed Forward Network used as a synthesizer in the diffusion process."""

    def __init__(self, configs):
        super(Model, self).__init__()
        hidden_layers = [1024, 1024, 1024, 1024]
        self.dim_t = 64
        self.mlp = MLP([self.dim_t, *hidden_layers], activation=configs.activation)

        # projection used for the input data
        self.proj = nn.Sequential(
            nn.Linear(configs.enc_in, self.dim_t),
            nn.SiLU(),
            nn.Linear(self.dim_t, self.dim_t)
        )

        # projection for the time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(self.dim_t, self.dim_t),
            nn.SiLU(),
            nn.Linear(self.dim_t, self.dim_t)
        )

        # used for the output layer
        self.head = nn.Linear(hidden_layers[-1], configs.enc_in)

    def forward(self, x, timesteps):
        # time embeddings
        emb = self.time_embed(timestep_embedding(timesteps, self.dim_t))

        # aggeregated data projection with time & label embeddings
        x = self.proj(x) + emb

        # additional mlp layers
        x = embed = self.mlp(x)
        x = self.head(x)
        return x, embed
