import torch
import torch.nn as nn


class SimpleMeanFlowNet(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        num_layers,
        mlp_ratio=4.0,
        dropout=0.0
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        self.t_embed = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.h_embed = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.hidden_layers = nn.ModuleList()
        dim = int(hidden_dim * mlp_ratio)

        for _ in range(num_layers):
            self.hidden_layers.append(
                nn.Sequential(
                    nn.Linear(hidden_dim, dim),
                    nn.SiLU(),
                    nn.Dropout(p=dropout),
                    nn.Linear(dim, hidden_dim)
                )
            )

        self.u_output_layer = nn.Linear(hidden_dim, output_dim)
        self.v_output_layer = nn.Linear(hidden_dim, output_dim)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        nn.init.constant_(self.u_output_layer.weight, 0)
        nn.init.constant_(self.u_output_layer.bias, 0)
        nn.init.constant_(self.v_output_layer.weight, 0)
        nn.init.constant_(self.v_output_layer.bias, 0)

    def forward(self, x, timestep, h, **kwargs):
        t_emb = self.t_embed(timestep.unsqueeze(-1))  # (B, H)
        h_emb = self.h_embed(h.unsqueeze(-1))  # (B, H)

        time_cond = t_emb + h_emb

        x = self.input_layer(x)  # (B, H)

        internal_features = []
        for layer in self.hidden_layers:
            x_cond = x + time_cond
            x = layer(x_cond) + x
            internal_features.append(x)

        u = self.u_output_layer(x)
        v = self.v_output_layer(x)
        internal_features = torch.stack(internal_features, dim=0)

        return u, v, internal_features
