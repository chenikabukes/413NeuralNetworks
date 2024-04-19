import torch
from torch import nn
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
        dim=192,
        depth=12,
        head_size=32,
        dropout_rate=0.1,
        cnn_out_channels=192,
        kernel_size=3,
        pool_size=96,
    ):
        super().__init__()
        self.emb = nn.Embedding(4, dim)
        self.pos_enc = SinusoidalPosEmb(dim)

        # CNN with BatchNorm and Dropout
        self.cnn = nn.Sequential(
            nn.Conv1d(dim, cnn_out_channels, kernel_size, padding=kernel_size // 2),
            nn.ReLU(),
            nn.BatchNorm1d(cnn_out_channels),
            nn.Dropout(dropout_rate),
        )

        # Pooling layer
        self.pool = nn.AdaptiveMaxPool1d(pool_size) if pool_size else None

        # Transformer encoder
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=cnn_out_channels if not pool_size else pool_size,
                nhead=cnn_out_channels // head_size,
                dim_feedforward=4 * cnn_out_channels,
                dropout=dropout_rate,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            ),
            num_layers=depth,
        )

        self.dropout = nn.Dropout(dropout_rate)
        self.output = nn.Linear(cnn_out_channels if not pool_size else pool_size, 2)

    def forward(self, x0):
        mask = x0["mask"]
        Lmax = mask.sum(-1).max()
        x = x0["seq"][:, :Lmax]

        x = self.emb(x)
        pos = self.pos_enc(torch.arange(Lmax, device=x.device).unsqueeze(0))
        x += pos

        x = x.permute(0, 2, 1)  # (batch, channels, length)
        x = self.cnn(x)

        if self.pool:
            x = self.pool(x)

        x = x.permute(0, 2, 1)  # (batch, length, channels)

        new_Lmax = x.size(1)
        if new_Lmax != Lmax:
            pass

        x = self.transformer(x, src_key_padding_mask=~mask[:, :new_Lmax])
        x = self.dropout(x)
        x = self.output(x)

        return x

    @staticmethod
    def name() -> str:
        return "<CNN Transformer>"
