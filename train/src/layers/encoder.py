from src.layers.norm import AdaLayerNorm
import torch.nn as nn
import torch.nn.functional as F


class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        # Linear FFN (equivalent to kernel_size=1 Conv1d; avoids MPS backward issues with transpose+conv)
        self.conv1 = nn.Linear(d_model, d_ff)
        self.conv2 = nn.Linear(d_ff, d_model)
        self.normt = AdaLayerNorm(d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, t, attn_mask=None, cond_embed=None, channel_embed=None):
        x = self.normt(x, t, cond_embed, channel_embed)  # create a timestep embedding and apply layer norm
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask
        )
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y)))
        y = self.dropout(self.conv2(y))

        return self.norm2(x + y), attn


class EncoderLayer_wot(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer_wot, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        # Linear FFN (equivalent to kernel_size=1 Conv1d; avoids MPS backward issues with transpose+conv)
        self.conv1 = nn.Linear(d_model, d_ff)
        self.conv2 = nn.Linear(d_ff, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask
        )
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y)))
        y = self.dropout(self.conv2(y))

        return self.norm2(x + y), attn


class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, t, attn_mask=None, cond_embed=None, channel_embed=None):
        # x [B, L, D]
        attns = []
        if self.conv_layers is not None:
            for i, (attn_layer, conv_layer) in enumerate(zip(self.attn_layers, self.conv_layers)):
                x, attn = attn_layer(x, t, attn_mask=attn_mask, cond_embed=cond_embed, channel_embed=channel_embed)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x, attn_mask=attn_mask, cond_embed=cond_embed, channel_embed=channel_embed)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, t, attn_mask=attn_mask, cond_embed=cond_embed, channel_embed=channel_embed)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns


class Encoder_wot(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder_wot, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        # x [B, L, D]
        attns = []
        if self.conv_layers is not None:
            for i, (attn_layer, conv_layer) in enumerate(zip(self.attn_layers, self.conv_layers)):
                x, attn = attn_layer(x, attn_mask=attn_mask)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x, attn_mask=attn_mask)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns


# FinDiff
def init_linear_layer(input_size, hidden_size):
    linear = nn.Linear(input_size, hidden_size, bias=True)
    nn.init.xavier_uniform_(linear.weight)
    nn.init.constant_(linear.bias, 0.0)
    return linear


class MLP(nn.Module):
    """ Base FeedForward Network
    """

    def __init__(self, hidden_size, activation='lrelu'):
        super(MLP, self).__init__()
        # init encoder architecture
        self.layers = self.init_layers(hidden_size)
        if activation == 'lrelu':
            self.activation = nn.LeakyReLU(negative_slope=0.4, inplace=True)
        elif activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        else:
            print('WRONG bottleneck function name !!!')

    def init_layers(self, layer_dimensions):
        layers = []
        for i in range(len(layer_dimensions) - 1):
            linear_layer = init_linear_layer(layer_dimensions[i], layer_dimensions[i + 1])
            layers.append(linear_layer)

            self.add_module('linear_' + str(i), linear_layer)
        return layers

    def forward(self, x):
        # Define the forward pass
        for i in range(len(self.layers)):
            x = self.activation(self.layers[i](x))
        return x