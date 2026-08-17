import torch
from torch import nn
import torch.nn.functional as F
from src.layers.norm import AdaLayerNorm


class Model(nn.Module):
    """
    Paper link: https://arxiv.org/pdf/1505.04597
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.value_embedding = nn.Linear(configs.seq_len, configs.d_model)
        self.dropout = nn.Dropout(configs.dropout)
        self.normt = AdaLayerNorm(configs.d_model)
        # Contracting Path (Encoder)
        self.enc_conv1 = nn.Conv1d(configs.enc_in, 64, kernel_size=3, padding=1)
        self.enc_conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.enc_conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)  # Reduce sequence length by half

        # Bottleneck
        self.bottleneck_conv = nn.Conv1d(256, 512, kernel_size=3, padding=1)

        # Expanding Path (Decoder)
        self.upconv3 = nn.ConvTranspose1d(512, 256, kernel_size=2, stride=2)
        self.dec_conv3 = nn.Conv1d(512, 256, kernel_size=3, padding=1)

        self.upconv2 = nn.ConvTranspose1d(256, 128, kernel_size=2, stride=2)
        self.dec_conv2 = nn.Conv1d(256, 128, kernel_size=3, padding=1)

        self.upconv1 = nn.ConvTranspose1d(128, 64, kernel_size=2, stride=2)
        self.dec_conv1 = nn.Conv1d(128, 64, kernel_size=3, padding=1)

        # Final layer to map back to original input channels (features)
        self.final_conv = nn.Conv1d(64, configs.enc_in, kernel_size=1)

    def basic_diffusion_forward(self, x_enc, t):
        # Input is (batch_size, sequence_length, feature_dims) and needs to be permuted
        x = x_enc.permute(0, 2, 1)  # Now (batch_size, feature_dims, sequence_length)
        x = self.value_embedding(x)
        x = self.dropout(x)
        x = self.normt(x, t)

        # Contracting Path
        c1 = F.relu(self.enc_conv1(x))  # (batch_size, 64, sequence_length)
        p1 = self.pool(c1)  # (batch_size, 64, sequence_length // 2)

        c2 = F.relu(self.enc_conv2(p1))  # (batch_size, 128, sequence_length // 2)
        p2 = self.pool(c2)  # (batch_size, 128, sequence_length // 4)

        c3 = F.relu(self.enc_conv3(p2))  # (batch_size, 256, sequence_length // 4)
        p3 = self.pool(c3)  # (batch_size, 256, sequence_length // 8)

        # Bottleneck
        embed = b = F.relu(self.bottleneck_conv(p3))  # (batch_size, 512, sequence_length // 8)

        # Expanding Path
        u3 = self.upconv3(b)  # (batch_size, 256, sequence_length // 4)
        u3 = torch.cat([u3, c3], dim=1)  # Concatenate with corresponding encoder output
        u3 = F.relu(self.dec_conv3(u3))  # (batch_size, 256, sequence_length // 4)

        u2 = self.upconv2(u3)  # (batch_size, 128, sequence_length // 2)
        u2 = torch.cat([u2, c2], dim=1)  # Concatenate with corresponding encoder output
        u2 = F.relu(self.dec_conv2(u2))  # (batch_size, 128, sequence_length // 2)

        u1 = self.upconv1(u2)  # (batch_size, 64, sequence_length)
        u1 = torch.cat([u1, c1], dim=1)  # Concatenate with corresponding encoder output
        u1 = F.relu(self.dec_conv1(u1))  # (batch_size, 64, sequence_length)

        # Final output layer
        output = self.final_conv(u1)  # (batch_size, out_channels, sequence_length)

        # Transpose back to (batch_size, sequence_length, out_channels)
        output = output.permute(0, 2, 1)

        return output, embed

    def conditional_diffusion_forward(self, x_enc, t, cond, cond_prob, prev_x):
        pass

    def forward(self, x_enc, t, cond=None, cond_prob=None, prev_x=None):
        basic_diffusion = ['basic_diffusion', 'diffusion_ts', 'diffusion_denoised_x', 'diffusion_direct_x']
        if self.task_name in basic_diffusion:
            return self.basic_diffusion_forward(x_enc, t)
        elif 'conditional' in self.task_name:
            return self.conditional_diffusion_forward(x_enc, t, cond, cond_prob, prev_x)
