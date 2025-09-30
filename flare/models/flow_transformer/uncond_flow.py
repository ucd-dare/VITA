import torch
import torch.nn as nn
from timm.layers import Mlp

from flare.models.pos_embed import SinusoidalPosEmbed
from flare.models.attention import Attention


class AdaLNBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=4.):
        super().__init__()
        def approx_gelu(): return nn.GELU(approximate="tanh")

        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=approx_gelu,
            drop=0
        )

        self.ada_ln = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 3*dim)
        )
        self.dim = dim

        self._init_weights()

    def _init_weights(self):
        nn.init.constant_(self.ada_ln[-1].weight, 0)
        nn.init.constant_(self.ada_ln[-1].bias, 0)

    def forward(self, x, t):
        B = x.shape[0]
        features = self.ada_ln(t).view(B, 3, self.dim).unbind(1)
        gamma, scale, shift = features

        x_norm = self.norm(x)
        x_norm = x_norm.mul(scale.add(1)).add_(shift)
        x = x + self.mlp(x_norm).mul_(gamma)

        return x


class AdaLNTransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.,
        dropout=0.,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            attn_drop=dropout,
            proj_drop=dropout
        )

        def approx_gelu(): return nn.GELU(approximate="tanh")

        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=approx_gelu,
            drop=0
        )

        self.ada_ln = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6*dim)
        )
        self.dim = dim

        self._init_weights()

    def _init_weights(self):
        nn.init.constant_(self.ada_ln[-1].weight, 0)
        nn.init.constant_(self.ada_ln[-1].bias, 0)

    def forward(self, x, t):
        B = x.shape[0]
        features = self.ada_ln(t).view(B, 6, 1, self.dim).unbind(1)
        gamma1, gamma2, scale1, scale2, shift1, shift2 = features

        x_norm1 = self.norm1(x)
        x_norm1 = x_norm1.mul(scale1.add(1)).add_(shift1)
        x = x + self.attn(x_norm1).mul_(gamma1)

        x_norm2 = self.norm2(x)
        x_norm2 = x_norm2.mul(scale2.add(1)).add_(shift2)
        x = x + self.mlp(x_norm2).mul_(gamma2)

        return x


class UncondFlowTransformer(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        num_layers,
        num_heads,
        mlp_ratio=4.0,
        dropout=0.1,
        time_embed_dim=256,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.time_embed = nn.Sequential(
            SinusoidalPosEmbed(time_embed_dim),
            nn.Linear(time_embed_dim, time_embed_dim * 4),
            nn.Mish(),
            nn.Linear(time_embed_dim * 4, hidden_dim),
        )

        def approx_gelu(): return nn.GELU(approximate="tanh")
        self.mlp = Mlp(
            in_features=hidden_dim,
            hidden_features=int(hidden_dim * mlp_ratio),
            act_layer=approx_gelu,
            drop=0
        )

        self.blocks = nn.ModuleList([
            AdaLNTransformerBlock(
                dim=hidden_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout
            ) for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, output_dim)

        self._init_weights()

    def _init_weights(self):
        # Basic initialization
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize timestep embedding MLP
        nn.init.normal_(self.time_embed[1].weight, std=0.02)
        nn.init.normal_(self.time_embed[3].weight, std=0.02)

    def forward(self, x, t):
        x = self.input_proj(x)
        t = self.time_embed(t)

        for block in self.blocks:
            x = block(x, t)

        x = self.norm(x)
        x = self.out_proj(x)
        return x
