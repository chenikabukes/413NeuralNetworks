import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class RNA_CNN_Transformer(nn.Module):
    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, nkmers, dropout=0.5):
        super(RNA_CNN_Transformer, self).__init__()
        # 1D convolutional layer
        self.conv1 = nn.Conv1d(1, nkmers, kernel_size=3, stride=1, padding=1)
        # Embedding layer for k-mers
        self.encoder = nn.Embedding(ntoken, ninp)
        # Positional Encoding
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        # Transformer Encoder
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.ninp = ninp
        # Output linear layer
        self.decoder = nn.Linear(ninp, ntoken)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, src_mask):
        src = self.conv1(src)
        src = F.relu(src)
        src = src.permute(2, 0, 1)  # Reshape for embedding layer
        src = self.encoder(src) + self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        output = self.decoder(output)
        return output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
