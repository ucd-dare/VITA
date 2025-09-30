import torch
import torch.nn as nn
import torch.nn.functional as F

from flare.models.pos_embed import RotaryPosEmbed


class RMSNorm(nn.Module):
    def __init__(self, dim, eps: float = 1e-6, weight=False):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim)) if weight else None

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        if self.weight is None:
            return output
        return output * self.weight


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=True,
        attn_drop=0.,
        proj_drop=0.,
        max_seq_len=16,
        qk_norm=True,
    ):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        # self.attn_norm = self.head_dim ** -0.5
        self.attn_drop = attn_drop
        self.proj_drop = nn.Dropout(proj_drop)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = nn.LayerNorm(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = nn.LayerNorm(self.head_dim) if qk_norm else nn.Identity()
        self.proj = nn.Linear(dim, dim)
        self.rope = RotaryPosEmbed(dim, max_seq_len=max_seq_len)

    def forward(self, x, mask=None):
        B, S, C = x.shape
        qkv = self.qkv(x).reshape(B, S, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # (B, num_heads, S, head_dim)
        q, k = self.q_norm(q), self.k_norm(k)

        q = self.rope(q)
        k = self.rope(k)
        if self.training:
            x = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=self.attn_drop)
        else:
            x = F.scaled_dot_product_attention(q, k, v)

        x = x.transpose(1, 2).reshape(B, S, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CrossAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        attn_drop=0.,
        proj_drop=0.,
        max_seq_len=16,
        qk_norm=True,
    ):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.attn_drop = attn_drop
        self.proj_drop = nn.Dropout(proj_drop)

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv_proj = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.q_norm = nn.LayerNorm(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = nn.LayerNorm(self.head_dim) if qk_norm else nn.Identity()
        self.proj = nn.Linear(dim, dim)
        self.rope = RotaryPosEmbed(dim, max_seq_len=max_seq_len)

    def forward(self, x, context, mask=None):
        """
        x: [B, S, D] - Query input
        context: [B, S_ctx, D] - Key/Value input
        mask: Optional attention mask
        """
        B, S, C = x.shape
        _, S_ctx, _ = context.shape

        # Project queries from x
        q = self.q_proj(x).reshape(B, S, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # Project keys and values from context
        kv = self.kv_proj(context).reshape(B, S_ctx, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)  # Each [B, num_heads, S_ctx, head_dim]
        q, k = self.q_norm(q), self.k_norm(k)

        q = self.rope(q)
        # Condition is not time-series. There there is no k = self.rope(k)
        if self.training:
            x = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=self.attn_drop)
        else:
            x = F.scaled_dot_product_attention(q, k, v)

        x = x.transpose(1, 2).reshape(B, S, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
