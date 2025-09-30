import torch
import torch.nn as nn

from flare.models.pos_embed import SinusoidalPosEmbed
from flare.models.flow_transformer.blocks import get_flow_transformer_block


class FlowTransformer(nn.Module):
    def __init__(
        self,
        input_dim,
        condition_dim,
        hidden_dim,
        output_dim,
        num_layers,
        num_heads,
        block_type="adaln",
        mlp_ratio=4.0,
        dropout=0.1,
        time_embed_dim=256,
    ):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.time_embed = nn.Sequential(
            SinusoidalPosEmbed(time_embed_dim),
            nn.Linear(time_embed_dim, time_embed_dim * 4),
            nn.Mish(),
            nn.Linear(time_embed_dim * 4, hidden_dim),
        )
        self.cond_embed = nn.Linear(condition_dim, hidden_dim)

        self.transformer_blocks = nn.ModuleList([
            get_flow_transformer_block(block_type)(
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
        
    def forward(self, x, t, cond):
        x = self.input_proj(x)
        t = self.time_embed(t)
        c = self.cond_embed(cond)

        for block in self.transformer_blocks:
            x = block(x, t, c)

        x = self.norm(x)
        x = self.out_proj(x)
        return x
