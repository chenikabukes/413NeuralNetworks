import torch
import torch.nn as nn
import math


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim=16, M=10000):
        super().__init__()
        self.dim = dim
        self.M = M

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(self.M) / half_dim
        emb = torch.exp(torch.arange(half_dim, device=device) * (-emb))
        emb = x[..., None] * emb[None, ...]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class RNA_Model(nn.Module):
    def __init__(
        self,
        num_embeddings=4,
        emb_dim=192,
        cnn_layers=3,
        cnn_out_channels=192,
        kernel_size=3,
        dropout_rate=0.1,
        transformer_depth=6,
        head_size=8,
        pool_size=None,
    ):
        super().__init__()
        self.emb = nn.Embedding(num_embeddings, emb_dim)
        self.pos_enc = SinusoidalPosEmb(emb_dim)

        # Adjustable CNN layers for feature extraction
        cnn_layers_list = [
            nn.Conv1d(
                emb_dim if i == 0 else cnn_out_channels,
                cnn_out_channels,
                kernel_size,
                padding=kernel_size // 2,
            )
            for i in range(cnn_layers)
        ]
        self.cnn = nn.Sequential(
            *cnn_layers_list,
            nn.ReLU(),
            nn.BatchNorm1d(cnn_out_channels),
            nn.Dropout(dropout_rate)
        )

        # Optional pooling layer
        self.pool = nn.AdaptiveMaxPool1d(pool_size) if pool_size else nn.Identity()

        # Transformer encoder with configurable depth
        transformer_layers = nn.TransformerEncoderLayer(
            d_model=cnn_out_channels,
            nhead=head_size,
            dim_feedforward=4 * cnn_out_channels,
            dropout=dropout_rate,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            transformer_layers, num_layers=transformer_depth
        )

        self.drop_out = nn.Dropout(dropout_rate)
        self.proj_out = nn.Linear(cnn_out_channels, 2)

    def forward(self, x0):
        mask = x0["mask"]
        Lmax = mask.sum(-1).max()
        mask = mask[:, :Lmax]
        x = x0["seq"][:, :Lmax]

        x = self.emb(x)
        pos = self.pos_enc(torch.arange(Lmax, device=x.device).unsqueeze(0))
        x += pos

        x = x.permute(0, 2, 1)  # (batch, channels, length)
        x = self.cnn(x)
        x = self.pool(x)

        x = x.permute(0, 2, 1)  # (batch, length, channels)
        x = self.transformer(x, src_key_padding_mask=~mask)

        x = self.drop_out(x)
        x = self.proj_out(x)
        return x

    @staticmethod
    def name() -> str:
        return "<Multi CNN Transformer>"
