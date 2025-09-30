import torch.nn as nn

from flare.models.attention import Attention, CrossAttention
from timm.layers import Mlp


class CrossAttentionAdaLNBlock(nn.Module):
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

        self.ada_ln = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 9*dim)
        )

        # Self-attention
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.self_attn = Attention(
            dim,
            num_heads=num_heads,
            attn_drop=dropout,
            proj_drop=dropout,
            max_seq_len=rope_max_seq_length,
        )

        # Cross-attention
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.cross_attn = CrossAttention(
            dim,
            num_heads=num_heads,
            attn_drop=dropout,
            proj_drop=dropout,
            max_seq_len=rope_max_seq_length,
        )

        # MLP
        def approx_gelu(): return nn.GELU(approximate="tanh")
        self.norm3 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=approx_gelu,
            drop=0
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.constant_(self.ada_ln[-1].weight, 0)
        nn.init.constant_(self.ada_ln[-1].bias, 0)

    def forward(self, x, t, c):
        B, _, D = x.shape
        t = t.unsqueeze(1)  # (B, 1, D)
        cond = t + c
        mod = self.ada_ln(cond).view(B, 9, D).unsqueeze(2)  # (B, 9, 1, D)

        (
            gamma_sa, gamma_ca, gamma_mlp,
            scale_sa, scale_ca, scale_mlp,
            shift_sa, shift_ca, shift_mlp
        ) = mod.unbind(dim=1)

        # Self-attention
        x_sa = self.norm1(x) * (1 + scale_sa) + shift_sa
        x = x + self.self_attn(x_sa) * gamma_sa

        # Cross-attention
        x_ca = self.norm2(x) * (1 + scale_ca) + shift_ca
        x = x + self.cross_attn(x_ca, c) * gamma_ca

        # MLP
        x_mlp = self.norm3(x) * (1 + scale_mlp) + shift_mlp
        x = x + self.mlp(x_mlp) * gamma_mlp

        return x


class CrossAttentionBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.,
        dropout=0.,
        rope_max_seq_length=16  # TODO: rope length
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads

        # Self-attention
        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = Attention(
            dim,
            num_heads=num_heads,
            attn_drop=dropout,
            proj_drop=dropout,
            max_seq_len=rope_max_seq_length,
        )

        # Cross-attention
        self.norm2 = nn.LayerNorm(dim)
        self.cross_attn = CrossAttention(
            dim,
            num_heads=num_heads,
            attn_drop=dropout,
            proj_drop=dropout,
            max_seq_len=rope_max_seq_length,
        )

        # Feedforward / MLP
        self.norm3 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)

        def approx_gelu(): return nn.GELU(approximate="tanh")
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=hidden_dim,
            act_layer=approx_gelu,
            drop=dropout
        )

    def forward(self, x, t, c):
        skip = x
        x = self.norm1(x)
        x = self.self_attn(x)
        x = skip + x

        skip = x
        x = self.norm2(x)
        x = self.cross_attn(x, c + t.unsqueeze(1))
        x = skip + x

        skip = x
        x = self.norm3(x)
        x = self.mlp(x)
        x = skip + x

        return x


class AdaLNBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.,
        dropout=0.,
    ):
        super().__init__()
        # self.norm1 = RMSNorm(dim)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            attn_drop=dropout,
            proj_drop=dropout
        )

        def approx_gelu(): return nn.GELU(approximate="tanh")

        # self.norm2 = RMSNorm(dim)
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

    def forward(self, x, t, c):
        B = x.shape[0]
        features = self.ada_ln(nn.SiLU()(t+c)).view(B, 6, 1, self.dim).unbind(1)
        gamma1, gamma2, scale1, scale2, shift1, shift2 = features

        x_norm1 = self.norm1(x)
        x_norm1 = x_norm1.mul(scale1.add(1)).add_(shift1)
        x = x + self.attn(x_norm1).mul_(gamma1)

        x_norm2 = self.norm2(x)
        x_norm2 = x_norm2.mul(scale2.add(1)).add_(shift2)
        x = x + self.mlp(x_norm2).mul_(gamma2)

        return x


class SimpleAdaLNBlock(nn.Module):
    def __init__(self, dim, num_heads=0.0, mlp_ratio=4., dropout=0.):
        super().__init__()
        def approx_gelu(): return nn.GELU(approximate="tanh")

        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=approx_gelu,
            drop=0
        )
        
        self.cond_mlp = Mlp(
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

    def forward(self, x, t, cond):
        B = x.shape[0]
        cond = self.cond_mlp(cond)
        features = self.ada_ln(cond + t).view(B, 3, self.dim).unbind(1)
        gamma, scale, shift = features

        x_norm = self.norm(x)
        x_norm = x_norm.mul(scale.add(1)).add_(shift)
        x = x + self.mlp(x_norm).mul_(gamma)

        return x


FLOW_TRANFORMER_BLOCK_TYPES = {
    "adaln": AdaLNBlock,
    "simple_adaln": SimpleAdaLNBlock,
    "cross": CrossAttentionBlock,
    "cross_adaln": CrossAttentionAdaLNBlock,
}


def get_flow_transformer_block(block_type): return FLOW_TRANFORMER_BLOCK_TYPES[block_type]
