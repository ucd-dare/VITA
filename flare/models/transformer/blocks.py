import torch.nn as nn
import torch
from torch import nn

from flare.models.attention import Attention
from timm.layers import Mlp


def approx_gelu(): return nn.GELU(approximate="tanh")


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.,
        dropout=0.,
        rope_max_seq_length=16,
    ):
        super().__init__()

        self.dim = dim
        self.num_heads = num_heads

        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = Attention(
            dim,
            num_heads=num_heads,
            attn_drop=dropout,
            proj_drop=dropout,
            max_seq_len=rope_max_seq_length,
        )

        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=approx_gelu,
            drop=dropout
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        skip = x
        x = self.norm1(x)
        x = self.self_attn(x)
        x = skip + x

        skip = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = skip + x

        return x


class ScaleBlock(nn.Module):
    def __init__(
        self,
        dim,
        output_dim,
        num_heads,
        mlp_ratio=4.,
        dropout=0.,
        rope_max_seq_length=16,
    ):
        super().__init__()

        self.dim = dim
        self.num_heads = num_heads
        self.output_dim = output_dim

        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = Attention(
            dim,
            num_heads=num_heads,
            attn_drop=dropout,
            proj_drop=dropout,
            max_seq_len=rope_max_seq_length,
        )

        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=approx_gelu,
            drop=dropout
        )
        self.norm2 = nn.LayerNorm(dim)

        self.sampler = nn.Linear(dim, output_dim)

    def forward(self, x):
        skip = x
        x = self.norm1(x)
        x = self.self_attn(x)
        x = skip + x

        skip = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = skip + x

        x = self.sampler(x)

        return x
