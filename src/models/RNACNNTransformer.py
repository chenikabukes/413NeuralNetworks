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

class RNA_CNN_Transformer(nn.Module):
    def __init__(self, dim=192, depth=12, head_size=32, dropout_rate=0.1, cnn_out_channels=32, kernel_size=3, **kwargs):
        super().__init__()
        self.emb = nn.Embedding(4, dim)
        self.pos_enc = SinusoidalPosEmb(dim)
        
        # CNN layer added to process the input sequence
        self.cnn = nn.Conv1d(dim, cnn_out_channels, kernel_size, padding=kernel_size//2)
        
        encoder_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=cnn_out_channels,  # Adjusted to cnn_out_channels
                                       nhead=cnn_out_channels // head_size,
                                       dim_feedforward=4 * cnn_out_channels,  # Adjusted to cnn_out_channels
                                       dropout=dropout_rate,
                                       activation='gelu',
                                       batch_first=True,
                                       norm_first=True) for _ in range(depth)])
        self.transformer = nn.TransformerEncoder(encoder_layer=encoder_layers[0], num_layers=depth)
        
        self.drop_out = nn.Dropout(dropout_rate)
        self.proj_out = nn.Linear(cnn_out_channels, 2)  # Adjusted to cnn_out_channels

    def forward(self, x0):
        mask = x0['mask']
        Lmax = mask.sum(-1).max()
        mask = mask[:, :Lmax]
        x = x0['seq'][:, :Lmax]

        pos = torch.arange(Lmax, device=x.device).unsqueeze(0)
        pos = self.pos_enc(pos)
        x = self.emb(x)
        x = x + pos
        
        # Apply CNN on the sequence
        x = x.permute(0, 2, 1)  # Change to (batch, channels, sequence length) for CNN
        x = self.cnn(x)
        x = x.permute(0, 2, 1)  # Change back to (batch, sequence length, channels)

        x = self.transformer(x, src_key_padding_mask=~mask)
        x = self.drop_out(x)
        x = self.proj_out(x)

        return x
