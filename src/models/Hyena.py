"""
Adapted from https://github.com/HazyResearch/safari/blob/02220c69d247e5473616cd053a443ad99fd2559b/standalone_hyena.py

Simplified standalone version of Hyena: https://arxiv.org/abs/2302.10866, designed for quick experimentation.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


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


def fftconv(u, k, D):
    seqlen = u.shape[-1]
    fft_size = 2 * seqlen

    k_f = torch.fft.rfft(k, n=fft_size) / (fft_size + 1e-16)
    u_f = torch.fft.rfft(u.to(dtype=k.dtype), n=fft_size) + 1e-16

    if len(u.shape) > 3:
        k_f = k_f.unsqueeze(1)
    uk = u_f * k_f
    y = torch.fft.irfft(uk, n=fft_size, norm="forward")[..., :seqlen]

    out = y + u * D.unsqueeze(-1)
    return out.to(dtype=u.dtype)


@torch.jit.script
def mul_sum(q, y):
    return (q * y).sum(dim=1)


class OptimModule(nn.Module):
    """Interface for Module that allows registering buffers/parameters with configurable optimizer hyperparameters"""

    def register(self, name, tensor, lr=None, wd=0.0):
        """Register a tensor with a configurable learning rate and 0 weight decay"""

        if lr == 0.0:
            self.register_buffer(name, tensor)
        else:
            self.register_parameter(name, nn.Parameter(tensor))

            optim = {}
            if lr is not None:
                optim["lr"] = lr
            if wd is not None:
                optim["weight_decay"] = wd
            setattr(getattr(self, name), "_optim", optim)


class Sin(nn.Module):
    def __init__(self, dim, w=10, train_freq=True):
        super().__init__()
        self.freq = (
            nn.Parameter(w * torch.ones(1, dim))
            if train_freq
            else w * torch.ones(1, dim)
        )

    def forward(self, x):
        return torch.sin(self.freq * x)


class PositionalEmbedding(OptimModule):
    def __init__(self, emb_dim: int, seq_len: int, lr_pos_emb: float = 1e-5, **kwargs):
        """Complex exponential positional embeddings for Hyena filters."""
        super().__init__()

        self.seq_len = seq_len
        # The time embedding fed to the filteres is normalized so that t_f = 1
        t = torch.linspace(0, 1, self.seq_len)[None, :, None]  # 1, L, 1

        if emb_dim > 1:
            bands = (emb_dim - 1) // 2
        # To compute the right embeddings we use the "proper" linspace
        t_rescaled = torch.linspace(0, seq_len - 1, seq_len)[None, :, None]
        w = 2 * math.pi * t_rescaled / seq_len  # 1, L, 1

        f = torch.linspace(1e-4, bands - 1, bands)[None, None]
        z = torch.exp(-1j * f * w)
        z = torch.cat([t, z.real, z.imag], dim=-1)
        self.register("z", z, lr=lr_pos_emb)
        self.register("t", t, lr=0.0)

    def forward(self, L):
        return self.z[:, :L], self.t[:, :L]


class ExponentialModulation(OptimModule):
    def __init__(
        self,
        d_model,
        fast_decay_pct=0.3,
        slow_decay_pct=1.5,
        target=1e-2,
        modulation_lr=0.0,
        modulate: bool = True,
        shift: float = 0.0,
        **kwargs,
    ):
        super().__init__()
        self.modulate = modulate
        self.shift = shift
        max_decay = math.log(target + 1e-16) / fast_decay_pct
        min_decay = math.log(target + 1e-16) / slow_decay_pct
        deltas = torch.linspace(min_decay, max_decay, d_model)[None, None]
        self.register("deltas", deltas, lr=modulation_lr)

    def forward(self, t, x):
        if self.modulate:
            decay = torch.exp(-t * self.deltas.abs() + 1e-16)
            x = x * (decay + self.shift)
        return x


class HyenaFilter(OptimModule):
    def __init__(
        self,
        d_model,
        emb_dim=3,  # dim of input to MLP, augments with positional encoding
        order=16,  # width of the implicit MLP
        fused_fft_conv=False,
        seq_len=768,
        lr=1e-3,
        lr_pos_emb=1e-5,
        dropout=0.0,
        w=1,  # frequency of periodic activations
        wd=0,  # weight decay of kernel parameters
        bias=True,
        num_inner_mlps=2,
        normalized=False,
        **kwargs,
    ):
        """
        Implicit long filter with modulation.

        Args:
            d_model: number of channels in the input
            emb_dim: dimension of the positional encoding (`emb_dim` - 1) // 2 is the number of bands
            order: width of the FFN
            num_inner_mlps: number of inner linear layers inside filter MLP
        """
        super().__init__()
        self.d_model = d_model
        self.use_bias = bias
        self.fused_fft_conv = fused_fft_conv
        self.bias = nn.Parameter(torch.randn(self.d_model))
        self.dropout = nn.Dropout(dropout)

        act = Sin(dim=order, w=w)
        self.emb_dim = emb_dim
        assert (
            emb_dim % 2 != 0 and emb_dim >= 3
        ), "emb_dim must be odd and greater or equal to 3 (time, sine and cosine)"
        self.seq_len = seq_len

        self.pos_emb = PositionalEmbedding(emb_dim, seq_len, lr_pos_emb)

        self.implicit_filter = nn.Sequential(
            nn.Linear(emb_dim, order),
            act,
        )
        for i in range(num_inner_mlps):
            self.implicit_filter.append(nn.Linear(order, order))
            self.implicit_filter.append(act)

        self.implicit_filter.append(nn.Linear(order, d_model, bias=False))

        self.modulation = ExponentialModulation(d_model, **kwargs)

        self.normalized = normalized
        for c in self.implicit_filter.children():
            for name, v in c.state_dict().items():
                optim = {"weight_decay": wd, "lr": lr}
                setattr(getattr(c, name), "_optim", optim)

    def filter(self, L, *args, **kwargs):
        z, t = self.pos_emb(L)
        h = self.implicit_filter(z)
        h = self.modulation(t, h)
        # if self.normalized:
        #     h = 
        return h

    def forward(self, x, L, k=None, bias=None, *args, **kwargs):
        if k is None:
            k = self.filter(L)

        # Ensure compatibility with filters that return a tuple
        k = k[0] if type(k) is tuple else k

        y = fftconv(x, k, bias)
        return y


class HyenaOperator(nn.Module):
    def __init__(
        self,
        d_model,
        l_max,
        order=2,
        filter_order=64,
        dropout=0.0,
        filter_dropout=0.0,
        **filter_args,
    ):
        """
        Hyena operator described in the paper https://arxiv.org/pdf/2302.10866.pdf

        Args:
            d_model (int): Dimension of the input and output embeddings (width of the layer)
            l_max: (int): Maximum input sequence length. Defaults to None
            order: (int): Depth of the Hyena recurrence. Defaults to 2
            dropout: (float): Dropout probability. Defaults to 0.0
            filter_dropout: (float): Dropout probability for the filter. Defaults to 0.0
        """
        super().__init__()
        self.d_model = d_model
        self.l_max = l_max
        self.order = order
        inner_width = d_model * (order + 1)
        self.dropout = nn.Dropout(dropout)
        self.in_proj = nn.Linear(d_model, inner_width)
        self.out_proj = nn.Linear(d_model, d_model)

        self.short_filter = nn.Conv1d(
            inner_width, inner_width, 3, padding=2, groups=inner_width
        )
        self.filter_fn = HyenaFilter(
            d_model * (order - 1),
            order=filter_order,
            seq_len=l_max,
            channels=1,
            dropout=filter_dropout,
            **filter_args,
        )

    def forward(self, u, *args, **kwargs):
        l = u.size(-2)
        l_filter = min(l, self.l_max)
        u = self.in_proj(u)
        u = rearrange(u, "b l d -> b d l")

        uc = self.short_filter(u)[..., :l_filter]
        *x, v = uc.split(self.d_model, dim=1)

        k = self.filter_fn.filter(l_filter)[0]
        k = rearrange(k, "l (o d) -> o d l", o=self.order - 1)
        bias = rearrange(self.filter_fn.bias, "(o d) -> o d", o=self.order - 1)

        for o, x_i in enumerate(reversed(x[1:])):
            v = self.dropout(v * x_i)
            v = self.filter_fn(v, l_filter, k=k[o], bias=bias[o])

        y = rearrange(v * x[0], "b d l -> b l d")

        y = self.out_proj(y)
        return y


class Mlp(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, activation=F.gelu,
                 return_residual=False, device=None, dtype=None):
        """
        From https://github.com/HazyResearch/flash-attention/blob/main/flash_attn/modules/mlp.py
        """
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.return_residual = return_residual
        self.fc1 = nn.Linear(in_features, hidden_features, **factory_kwargs)
        self.activation = activation
        self.fc2 = nn.Linear(hidden_features, out_features, **factory_kwargs)

    def forward(self, x):
        y = self.fc1(x)
        y = self.activation(y)
        y = self.fc2(y)
        return y if not self.return_residual else (y, x)


class RNA_Transformer_Model(nn.Module):
    def __init__(
        self,
        num_embeddings=4,
        emb_dim=192,
        l_max=768,
        order=2,
        filter_order=64,
        head_size=32,
        dropout_rate=0.1,
        depth=12,
    ):
        super().__init__()
        self.emb = nn.Embedding(num_embeddings, emb_dim)
        self.pos_enc = SinusoidalPosEmb(emb_dim)

        # Initialize a Hyena Operator
        self.hyena_operator = HyenaOperator(d_model=emb_dim, 
                                            l_max=l_max, order=order, 
                                            filter_order=filter_order)

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=emb_dim,
                nhead=emb_dim // head_size,
                dim_feedforward=4 * emb_dim,
                dropout=dropout_rate,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            ),
            num_layers=depth,
        )

        self.drop_out = nn.Dropout(0.1)
        self.proj_out = nn.Linear(
            emb_dim, 2
        )  # Output dim is 2 for reactivity prediction

    def forward(self, x0):
        mask = x0["mask"]
        Lmax = mask.sum(-1).max()
        mask = mask[:, :Lmax].bool()
        x = x0["seq"][:, :Lmax]

        x = self.emb(x)
        pos = self.pos_enc(torch.arange(Lmax, device=x.device).unsqueeze(0))
        x += pos

        x = self.hyena_operator(x)
        x = self.transformer(x)

        x = self.drop_out(x)
        x = self.proj_out(x)
        return x

    @staticmethod
    def name() -> str:
        return "<Hyena Transformer Model>"


class RNA_MLP_Model(nn.Module):
    def __init__(
        self,
        num_embeddings=4,
        emb_dim=192,
        l_max=768,
        order=2,
        filter_order=64,
    ):
        super().__init__()
        self.emb = nn.Embedding(num_embeddings, emb_dim)
        self.pos_enc = SinusoidalPosEmb(emb_dim)

        # Initialize a Hyena Operator
        self.hyena_operator = HyenaOperator(d_model=emb_dim, 
                                            l_max=l_max, order=order, 
                                            filter_order=filter_order)

        self.mlp_layer = Mlp(emb_dim)

        self.drop_out = nn.Dropout(0.1)
        self.proj_out = nn.Linear(
            emb_dim, 2
        )  # Output dim is 2 for reactivity prediction

    def forward(self, x0):
        mask = x0["mask"]
        Lmax = mask.sum(-1).max()
        mask = mask[:, :Lmax].bool()
        x = x0["seq"][:, :Lmax]

        x = self.emb(x)
        pos = self.pos_enc(torch.arange(Lmax, device=x.device).unsqueeze(0))
        x += pos

        x = self.hyena_operator(x)
        x = self.mlp_layer(x)

        x = self.drop_out(x)
        x = self.proj_out(x)
        return x

    @staticmethod
    def name() -> str:
        return "<Hyena MLP Model>"


class RNA_CNN_Model(nn.Module):
    def __init__(
        self,
        num_embeddings=4,
        emb_dim=192,
        l_max=768,
        order=2,
        filter_order=64,
        kernel_size=3,
        dropout_rate=0.1,
        cnn_out_channels=192,
    ):
        super().__init__()
        self.emb = nn.Embedding(num_embeddings, emb_dim)
        self.pos_enc = SinusoidalPosEmb(emb_dim)

        self.cnn = nn.Sequential(
            nn.Conv1d(emb_dim, cnn_out_channels, kernel_size, padding=kernel_size // 2),
            nn.ReLU(),
            nn.BatchNorm1d(cnn_out_channels),
            nn.Dropout(dropout_rate),
        )

        # Initialize a Hyena Operator
        self.hyena_operator = HyenaOperator(d_model=cnn_out_channels, 
                                            l_max=l_max, order=order, 
                                            filter_order=filter_order)

        self.drop_out = nn.Dropout(0.1)
        self.proj_out = nn.Linear(
            emb_dim, 2
        )  # Output dim is 2 for reactivity prediction

    def forward(self, x0):
        mask = x0["mask"]
        Lmax = mask.sum(-1).max()
        mask = mask[:, :Lmax].bool()
        x = x0["seq"][:, :Lmax]

        x = self.emb(x)
        pos = self.pos_enc(torch.arange(Lmax, device=x.device).unsqueeze(0))
        x += pos
        
        x = x.permute(0, 2, 1)
        x = self.cnn(x)

        x = x.permute(0, 2, 1)
        x = self.hyena_operator(x)

        x = self.drop_out(x)
        x = self.proj_out(x)
        return x

    @staticmethod
    def name() -> str:
        return "<Hyena CNN Model>"
