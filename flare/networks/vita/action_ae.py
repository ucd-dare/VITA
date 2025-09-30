import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import einops
from timm.layers import Mlp

from flare.models.transformer.blocks import TransformerBlock


def weights_init_encoder(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d):
        nn.init.orthogonal_(m.weight.data)
        if m.bias is not None:
            m.bias.data.fill_(0.0)


def create_sinusoidal_pos_embedding(num_positions: int, dimension: int):
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / dimension) for hid_j in range(dimension)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(num_positions)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
    return torch.from_numpy(sinusoid_table).float()


class TransformerActionEncoder(nn.Module):
    def __init__(
        self,
        enc_hidden_dim: int,
        latent_dim: int,
        num_heads: int,
        pred_horizon: int,
        action_dim: int,
        num_layers: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        # Token embeddings
        self.cls_embed = nn.Embedding(1, enc_hidden_dim)
        self.action_input_proj = nn.Linear(action_dim, enc_hidden_dim)

        # Transformer encoder layers
        self.layers = nn.ModuleList([
            TransformerBlock(
                dim=enc_hidden_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                rope_max_seq_length=pred_horizon + 1
            ) for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(enc_hidden_dim)

        # Output projection to latent parameters (mu and log_var for each dimension)
        self.latent_output_proj = nn.Linear(enc_hidden_dim, latent_dim)

        # Fixed positional embeddings
        num_tokens = 1 + pred_horizon
        self.register_buffer(
            "pos_embed",
            create_sinusoidal_pos_embedding(num_tokens, enc_hidden_dim).unsqueeze(0)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, actions):
        batch_size = actions.shape[0]

        # Prepare tokens
        cls_token = einops.repeat(
            self.cls_embed.weight, "1 d -> b 1 d", b=batch_size
        )
        action_tokens = self.action_input_proj(actions)
        tokens = torch.cat([cls_token, action_tokens], dim=1)

        # Add positional embeddings
        tokens = tokens + self.pos_embed[:, :tokens.shape[1]]

        # Pass through transformer layers
        x = tokens
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)

        # Extract CLS token output and project to latent parameters
        cls_output = x[:, 0]
        z = self.latent_output_proj(cls_output)

        return z


class SimpleActionEncoder(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        pred_horizon: int,
        action_dim: int,
        num_layers: int = 4,
        use_attention: bool = False,
    ):
        super().__init__()
        self.action_latent_dim = latent_dim // pred_horizon
        self.input_proj = nn.Linear(action_dim, self.action_latent_dim)

        self.layers = nn.ModuleList()
        if use_attention:
            for _ in range(num_layers):
                self.layers.append(
                    TransformerBlock(
                        dim=self.action_latent_dim,
                        num_heads=8,
                        mlp_ratio=4.0,
                        dropout=0.0,
                        rope_max_seq_length=pred_horizon
                    )
                )
        else:
            for _ in range(num_layers):
                self.layers.append(
                    Mlp(
                        in_features=self.action_latent_dim,
                        hidden_features=4 * self.action_latent_dim,
                        out_features=self.action_latent_dim,
                        norm_layer=None,
                        bias=True,
                        drop=0.0
                    )
                )

        self.pred_horizon = pred_horizon
        self.action_dim = action_dim

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, actions):
        batch_size = actions.shape[0]

        actions = actions.view(batch_size, self.pred_horizon, self.action_dim)
        x = self.input_proj(actions)  # (B, pred_horizon, action_latent_dim)
        for layer in self.layers:
            x = layer(x)
        x = x.view(batch_size, -1)

        return x


class SimpleActionDecoder_(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        pred_horizon: int,
        action_dim: int,
        num_layers: int = 4,
        use_attention: bool = False,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.pred_horizon = pred_horizon
        self.action_dim = action_dim
        self.action_latent_dim = latent_dim // pred_horizon

        self.output_proj = nn.Linear(self.action_latent_dim, action_dim)

        self.layers = nn.ModuleList()
        if use_attention:
            for _ in range(num_layers):
                self.layers.append(
                    TransformerBlock(
                        dim=self.action_latent_dim,
                        num_heads=8,
                        mlp_ratio=4.0,
                        dropout=0.0,
                        rope_max_seq_length=pred_horizon
                    )
                )
        else:
            for _ in range(num_layers):
                self.layers.append(
                    Mlp(
                        in_features=self.action_latent_dim,
                        hidden_features=4 * self.action_latent_dim,
                        out_features=self.action_latent_dim,
                        norm_layer=None,
                        bias=True,
                        drop=0.0
                    )
                )

        self.output_proj = nn.Linear(self.action_latent_dim, action_dim)

    def forward(self, z):
        batch_size = z.shape[0]
        z = z.view(batch_size, self.pred_horizon, self.action_latent_dim)
        for layer in self.layers:
            z = layer(z)
        z = self.output_proj(z)  # (B, pred_horizon, action_dim)
        return z


class SimpleActionDecoder(nn.Module):
    def __init__(
        self,
        dec_hidden_dim: int,
        latent_dim: int,
        pred_horizon: int,
        action_dim: int,
        num_layers: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        output_dim = pred_horizon * action_dim

        self.input_proj = nn.Linear(latent_dim, dec_hidden_dim)
        self.output_proj = nn.Linear(dec_hidden_dim, output_dim)

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                Mlp(
                    in_features=dec_hidden_dim,
                    hidden_features=dec_hidden_dim,
                    out_features=dec_hidden_dim,
                    norm_layer=None,
                    bias=True,
                    drop=dropout
                )
            )

        self.pred_horizon = pred_horizon
        self.action_dim = action_dim

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, z):
        """
        Args:
            z: (B, latent_dim)
        Returns:
            actions: (B, pred_horizon, action_dim)
        """
        x = self.input_proj(z)
        for layer in self.layers:
            x = layer(x)
        x = self.output_proj(x)
        actions = x.view(-1, self.pred_horizon, self.action_dim)
        return actions


class CNNActionEncoder(nn.Module):
    def __init__(
        self,
        pred_horizon: int,
        action_dim: int,
        latent_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
    ):
        super().__init__()

        self.pred_horizon = pred_horizon
        self.action_dim = action_dim
        self.latent_dim = latent_dim

        # CNN encoder layers
        layers = []
        current_dim = action_dim

        for i in range(num_layers):
            if i == 0:
                layers.append(nn.Conv1d(current_dim, hidden_dim, kernel_size=5, stride=2, padding=2))
            else:
                layers.append(nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, stride=2, padding=2))
            layers.append(nn.ReLU())
            current_dim = hidden_dim

        self.encoder = nn.Sequential(*layers)

        conv_output_length = pred_horizon // (2 ** num_layers)
        conv_output_dim = hidden_dim * conv_output_length

        self.latent_proj = nn.Linear(conv_output_dim, latent_dim)

        self.apply(weights_init_encoder)

    def forward(self, actions, deterministic=False):
        batch_size = actions.shape[0]

        x = actions.transpose(1, 2)  # (B, action_dim, pred_horizon)
        x = self.encoder(x)  # (B, hidden_dim, conv_output_length)
        x = x.view(batch_size, -1)  # (B, hidden_dim * conv_output_length)

        z = self.latent_proj(x)  # (B, latent_dim)

        return z


def get_autoencoder(config, network_config):
    from flare.networks.vita.action_ae import (
        TransformerActionEncoder,
        SimpleActionEncoder,
        SimpleActionDecoder,
    )

    ae_config = network_config.action_ae
    net_config = ae_config.net
    pred_horizon = network_config.pred_horizon
    action_dim = config.task.action_dim
    latent_dim = net_config.latent_dim

    encoder_type = net_config.encoder_type
    decoder_type = net_config.decoder_type

    if encoder_type == 'cnn':
        encoder = CNNActionEncoder(
            pred_horizon=pred_horizon,
            action_dim=action_dim,
            latent_dim=latent_dim,
            hidden_dim=net_config.enc_hidden_dim,
            num_layers=net_config.num_layers
        )
    elif encoder_type == 'transformer':
        encoder = TransformerActionEncoder(
            enc_hidden_dim=net_config.enc_hidden_dim,
            latent_dim=latent_dim,
            num_heads=net_config.num_heads,
            pred_horizon=pred_horizon,
            action_dim=action_dim,
            num_layers=net_config.num_layers,
            mlp_ratio=net_config.mlp_ratio,
            dropout=net_config.dropout,
        )
    elif encoder_type == 'simple':
        encoder = SimpleActionEncoder(
            latent_dim=latent_dim,
            pred_horizon=pred_horizon,
            action_dim=action_dim,
            num_layers=net_config.num_layers,
            use_attention=net_config.get('use_attention', False),
        )
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")

    if decoder_type == 'simple':
        # decoder = SimpleActionDecoder_(
        #     latent_dim=latent_dim,
        #     pred_horizon=pred_horizon,
        #     action_dim=action_dim,
        #     num_layers=net_config.num_layers,
        #     use_attention=net_config.get('use_attention', False),
        # )
        decoder = SimpleActionDecoder(
            dec_hidden_dim=net_config.dec_hidden_dim,
            latent_dim=latent_dim,
            pred_horizon=pred_horizon,
            action_dim=action_dim,
            num_layers=net_config.num_layers,
            dropout=net_config.get('dropout', 0.0)
        )
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")

    return encoder, decoder
