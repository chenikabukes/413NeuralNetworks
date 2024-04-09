import torch
from torch import nn
import math

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim, M=10000):
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

class RNA_CNN_Transformer(nn.Module):
    def __init__(self, dim=192, depth=12, head_size=32, dropout_rate=0.1, nkmers=64):
        super().__init__()
        self.conv1 = nn.Conv1d(4, nkmers, kernel_size=3, stride=1, padding=1)
        self.emb = nn.Embedding(4, dim)
        self.pos_enc = SinusoidalPosEmb(nkmers)
        encoder_layers = nn.TransformerEncoderLayer(d_model=nkmers, nhead=nkmers//head_size,
                                                    dim_feedforward=4*nkmers, dropout=dropout_rate,
                                                    activation='gelu', batch_first=True,
                                                    norm_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer=encoder_layers, num_layers=depth)
        self.drop_out = nn.Dropout(dropout_rate)
        self.proj_out = nn.Linear(nkmers, 2)

    def forward(self, x0):
        mask = x0['mask']
        Lmax = mask.sum(-1).max()
        mask = mask[:, :Lmax]
        x = x0['seq'][:, :Lmax]

        x = self.emb(x)  # Embedding layer
        x = x.permute(0, 2, 1)  # Changing shape to [batch_size, embedding_dim, seq_length]
        x = self.conv1(x)  # Apply convolution
        x = x.permute(0, 2, 1)  # Reverting shape to [batch_size, seq_length, embedding_dim]

        pos = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        pos = self.pos_enc(pos)
        x = x + pos  # Adding positional encoding

        x = self.transformer(x, src_key_padding_mask=~mask)
        x = self.drop_out(x)
        x = self.proj_out(x)

        return x
