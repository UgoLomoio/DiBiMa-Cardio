import math
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import os
from functools import partial
from mamba_ssm.models.mixer_seq_simple import _init_weights
from mamba_ssm.ops.triton.layer_norm import RMSNorm
from mamba_ssm import Mamba, Mamba2
try:  # Mamba3 is unused by the Mamba2 DiBiMa models; optional in the Space
    from mamba_ssm import Mamba3
except Exception:  # pragma: no cover
    Mamba3 = None
from mamba_ssm.modules.block import Block
import matplotlib.pyplot as plt
from src.utils import QRSAwareMSELoss
import neurokit2 as nk

project_path = os.path.dirname(os.path.abspath(__file__))

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SpatialEmbedding(nn.Module):
    def __init__(self, d_model, hidden_dim=32):
        super().__init__()
        self.d_model = d_model  
       
        self.pos_proj = nn.Sequential(  # Pos -> channel embeds
            nn.Linear(3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, d_model)  # Fixed 62->64 adapt
        )
        self.channel_adapt = nn.Linear(d_model, 64)  # Adapt src ch to mamba_dim seq

    def forward(self, positions=None):
        pos = positions # Assume positions is (B,C,3)
        pos_emb = self.pos_proj(pos.reshape(-1,3)).reshape(pos.shape[0], pos.shape[1], -1)  # (B,C,64)
        if pos_emb.shape[1] != 64:
            seq_emb = self.channel_adapt(pos_emb.mean(-1))  # (B,C) -> (B,64) via Linear
        else:
            seq_emb = pos_emb.mean(-1)  # (B,64)
        return seq_emb.unsqueeze(-1)  # (B,64,1) — expand to L later

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):  # x: (B, seq_len, d_model)
        return x + self.pe[:, :x.size(1)]


class ConvSpatialEmbedding(nn.Module):
    def __init__(self, d_model=64, hidden_dim=32, kernel_size=3):  # d_model=64 for Mamba
        super().__init__()
        # CoordConv: Append x,y,z as extra channels
        self.conv = nn.Sequential(
            nn.Conv1d(3, hidden_dim, kernel_size, padding="same"),  # in_ch=3
            nn.SiLU(),
            nn.Conv1d(hidden_dim, d_model, 1),  # To d_model
            #nn.Dropout(0.2)
        )
        #self.norm = nn.LayerNorm(d_model)

    def forward(self, positions=None):
        pos = positions  # Assume positions is (B,C,3)
        pos_flat = pos.transpose(1,2)  # (B,3,C)
        conv_out = self.conv(pos_flat)  # Assume (B,64,C)
        #conv_out = self.norm(conv_out.transpose(1,2)).transpose(1,2)  # (B,C,64)
        pooled = conv_out.mean(dim=-1)  # Pool electrodes → (B,64)
        return pooled.unsqueeze(-1)  # (B,64,1) ✓ — now expands to L=1600

class TemporalEmbedding(nn.Module):

    def __init__(self, time_dim, d_model):
        
        super().__init__()
        self.time_dim = time_dim
        self.d_model = d_model
        self.time_mlp = nn.Sequential(
            nn.Linear(self.time_dim, self.time_dim),
            nn.SiLU(),
            nn.Linear(self.time_dim, self.time_dim),
        )
        self.time_proj = nn.Conv1d(self.time_dim, self.d_model, kernel_size=1)

    def _timestep_embedding(self, t, dim):

        # t can be scalar () or 1-D (B,)
        if t.dim() == 0:
            t = t.unsqueeze(0)          # -> (1,)
        # now t is (B,)

        device = t.device
        half_dim = dim // 2
        emb_factor = math.log(10000) / (half_dim - 1)
        freqs = torch.exp(torch.arange(half_dim, device=device) * -emb_factor)  # (half_dim,)

        t = t.float()                   # (B,)
        emb = t[:, None] * freqs[None, :]   # (B, half_dim)

        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)  # (B, dim)
        if dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb

    def forward(self, t):  # t: (B, d_model, T)
        
        temb = self._timestep_embedding(t, self.time_dim)  # [B, time_dim]
        temb = self.time_mlp(temb).unsqueeze(-1) # (B,64,1)
        temb = self.time_proj(temb)  # (B, d_model, 1)
        return temb

def handle_norm(x, norm_layer, before_mamba=True):
    # x enters as (B, D, L) if before_mamba=True
    # x enters as (B, L, D) if before_mamba=False (coming out of Mamba)

    if norm_layer is None: 
        return x.transpose(1, 2) 
    
    if hasattr(norm_layer, 'normalized_shape'):      # LayerNorm, RMSNorm, InstanceNorm1d
        if before_mamba:
            x = x.transpose(1, 2)                    # (B, D, L) → (B, L, D)
        x = norm_layer(x)                            # norm over last dim, shape unchanged
        if not before_mamba:
            x = x.transpose(1, 2)                    # (B, L, D) → (B, D, L)

    elif hasattr(norm_layer, 'num_features'):         # BatchNorm1d
        if not before_mamba:
            x = x.transpose(1, 2)                    # (B, L, D) → (B, D, L)
        x = norm_layer(x)                            # BN needs (B, D, L)
        if before_mamba:
            x = x.transpose(1, 2)                    # (B, D, L) → (B, L, D)

    elif isinstance(norm_layer, nn.Identity):
        pass  # No operation
    
    else:
        raise ValueError(f"Unsupported norm: {type(norm_layer)}")

    return x   # (B, L, D) if before_mamba=True, (B, D, L) if before_mamba=False
    
def get_norm_layer(norm_type, dim):
    if norm_type == "layer":
        return nn.LayerNorm(dim)
    elif norm_type == "batch":
        return nn.BatchNorm1d(dim)
    elif norm_type == "rms":
        return nn.RMSNorm(dim)
    elif norm_type == "instance":
        return nn.InstanceNorm1d(dim)
    elif norm_type == "none":
        return None
    else:
        raise ValueError(f"Unsupported norm type: {norm_type}")

class SignalEncoder(nn.Module):

    def __init__(self, in_channels, out_dim, mamba_params, kernel_size=3, signal_type="ppg", norm_types=["layer", "batch"], do=0.2):

        super().__init__()

        self.conv1 = nn.Conv1d(in_channels, out_dim, kernel_size, padding=kernel_size//2)
        self.norm_types = norm_types
        if isinstance(self.norm_types, str):
            self.norm_types = [self.norm_types, self.norm_types]
        self.before_norm = get_norm_layer(self.norm_types[0], out_dim)
        self.after_norm = get_norm_layer(self.norm_types[1], out_dim)

        self.mamba_params = mamba_params.copy()

        if signal_type == "ppg":
            self.mamba_params["n_mamba_blocks"] = 2
            self.mamba_params["n_layers"] = 3
            self.mamba_params["d_state"] = 16 if self.mamba_params["version"]==1 else 64
        elif signal_type == "ecg":
            self.mamba_params["n_mamba_blocks"] = 2
            self.mamba_params["n_layers"] = 3
            self.mamba_params["d_state"] = 16 if self.mamba_params["version"]==1 else 64
        else: #bp
            self.mamba_params["n_mamba_blocks"] = 2
            self.mamba_params["n_layers"] = 3
            self.mamba_params["d_state"] = 16 if self.mamba_params["version"]==1 else 64

        # BiMamba expects [B,L,D] seq-first
        self.bimamba = BidirectionalMamba(
            d_model=out_dim, d_state=self.mamba_params["d_state"], n_mamba_blocks=self.mamba_params["n_mamba_blocks"], 
            n_layers=self.mamba_params["n_layers"], mamba_version=self.mamba_params["version"], merge_type='add',
            headdim=self.mamba_params.get("headdim", None), is_mimo=self.mamba_params.get("is_mimo", False), mimo_rank=self.mamba_params.get("mimo_rank", None),
            chunk_size=self.mamba_params.get("chunk_size", None), is_outproj_norm=self.mamba_params.get("is_outproj_norm", False), dtype=self.mamba_params.get("dtype", torch.float32),
            attention=self.mamba_params.get("attention", False)
        )
        self.dropout = nn.Dropout(do)        
        #self.conv_out = nn.Conv1d(out_dim, out_dim, kernel_size=1)  # Final conv to ensure consistent output dim
    
    def forward(self, x, add=None):
        # x: (B, D, L) throughout, except inside bimamba call
        x = self.conv1(x)                             # (B, D, L)
        if add is not None:                           # early fusion (Oct 30 2026): time-aligned other-modality term
            x = x + add

        x_res = x.clone()                                   # (B, D, L) for residual
        x = handle_norm(x, self.before_norm, before_mamba=True)          # (B, D, L) → norm → (B, D, L)
        x_mamba = self.bimamba(x)     # (B, D, L)→(B, L, D) for Mamba → out (B, L, D)
        x = handle_norm(x_mamba, self.after_norm, before_mamba=False)     # (B, D, L) → norm → (B, D, L)
        x = x + x_res                                 # Residual connection

        x = self.dropout(x)
        #x = self.conv_out(x)                          # (B, D, L)
        return x

class PixelShuffle1D(nn.Module):
    """1D PixelShuffle: rearranges (B, C*r, L) to (B, C, L*r)."""
    def __init__(self, upscale_factor):
        super().__init__()
        self.upscale_factor = upscale_factor

    def forward(self, x):
        r = self.upscale_factor
        b, c, l = x.shape
        x = x.view(b, c // r, r, l).permute(0, 1, 3, 2).reshape(b, c // r, l * r)
        return x

class BidirectionalMamba(nn.Module):
    def __init__(
        self,
        d_model=64,
        d_state=16,
        d_conv=4,
        expand=2,
        n_layers=1,
        n_mamba_blocks=1,
        mamba_version=1,
        merge_type="add",
        device="cuda:0" if torch.cuda.is_available() else "cpu",
        headdim=None,
        is_mimo=False,
        mimo_rank=None,
        chunk_size=None,
        is_outproj_norm=False,
        dtype=torch.float32,
        attention=False,
        num_attention_heads=4,
        attention_dropout=0.1,
    ):
        super().__init__()

        if merge_type not in {"add", "concat"}:
            raise ValueError(
                f"merge_type must be either 'add' or 'concat', got '{merge_type}'."
            )

        self.d_model = d_model
        self.n_layers = n_layers
        self.n_mamba_blocks = n_mamba_blocks
        self.merge_type = merge_type
        self.headdim = headdim
        self.is_mimo = is_mimo
        self.mimo_rank = mimo_rank
        self.chunk_size = chunk_size
        self.is_outproj_norm = is_outproj_norm
        self.dtype = dtype
        self.device = device

        self.use_attention = attention
        self.num_attention_heads = num_attention_heads

        self.forward_layers = nn.ModuleList([])
        self.backward_layers = nn.ModuleList([])

        self.forward_convs = nn.ModuleList([])
        self.backward_convs = nn.ModuleList([])

        self.residual_convs = nn.ModuleList([])

        self.attention_layers = nn.ModuleList([])
        self.attention_norms = nn.ModuleList([])
        self.attention_dropouts = nn.ModuleList([])

        # Residual projections. Required when concat doubles d_model.
        residual_in_channels = d_model

        for i in range(self.n_layers):
            residual_out_channels = (
                residual_in_channels * 2
                if self.merge_type == "concat"
                else residual_in_channels
            )

            self.residual_convs.append(
                nn.Conv1d(
                    in_channels=residual_in_channels,
                    out_channels=residual_out_channels,
                    kernel_size=3,
                    padding=1,
                ).to(self.device)
            )

            residual_in_channels = residual_out_channels

        # Resolve Mamba implementation once.
        if mamba_version == 1:
            mamba_instance = Mamba
        elif mamba_version == 2:
            mamba_instance = Mamba2
        elif mamba_version == 3:
            mamba_instance = Mamba3
        else:
            mamba_instance = Mamba
            print(
                f"Warning: Invalid mamba_version={mamba_version}. "
                "Defaulting to Mamba version 1."
            )

        current_d_model = d_model

        for i in range(self.n_layers):
            # From the second layer onward, concat doubles features.
            if i > 0 and self.merge_type == "concat":
                current_d_model *= 2

            if self.use_attention:
                if current_d_model % self.num_attention_heads != 0:
                    raise ValueError(
                        f"Layer {i}: d_model={current_d_model} must be divisible by "
                        f"num_attention_heads={self.num_attention_heads}."
                    )

                self.attention_layers.append(
                    nn.MultiheadAttention(
                        embed_dim=current_d_model,
                        num_heads=self.num_attention_heads,
                        dropout=attention_dropout,
                        batch_first=True,
                    ).to(self.device)
                )

                self.attention_norms.append(
                    RMSNorm(current_d_model, eps=1e-5).to(self.device)
                )

                self.attention_dropouts.append(
                    nn.Dropout(attention_dropout)
                )

            fwd_blocks = nn.ModuleList([])
            bwd_blocks = nn.ModuleList([])

            fwd_conv = nn.Conv1d(
                current_d_model,
                current_d_model,
                kernel_size=3,
                padding=1,
            ).to(self.device)

            bwd_conv = nn.Conv1d(
                current_d_model,
                current_d_model,
                kernel_size=3,
                padding=1,
            ).to(self.device)

            for j in range(self.n_mamba_blocks):
                fwd_blocks.append(
                    Block(
                        current_d_model,
                        mixer_cls=partial(
                            mamba_instance,
                            layer_idx=j,
                            d_state=d_state,
                            d_conv=d_conv,
                            expand=expand,
                        ),
                        mlp_cls=partial(
                            nn.Linear,
                            current_d_model,
                            current_d_model,
                        ),
                        norm_cls=partial(RMSNorm, eps=1e-5),
                        fused_add_norm=False,
                    ).to(self.device)
                )

                bwd_blocks.append(
                    Block(
                        current_d_model,
                        mixer_cls=partial(
                            mamba_instance,
                            layer_idx=j,
                            d_state=d_state,
                            d_conv=d_conv,
                            expand=expand,
                        ),
                        mlp_cls=partial(
                            nn.Linear,
                            current_d_model,
                            current_d_model,
                        ),
                        norm_cls=partial(RMSNorm, eps=1e-5),
                        fused_add_norm=False,
                    ).to(self.device)
                )

            self.forward_layers.append(fwd_blocks)
            self.backward_layers.append(bwd_blocks)

            self.forward_convs.append(fwd_conv)
            self.backward_convs.append(bwd_conv)

        # Initialize all registered submodules after construction.
        self.apply(partial(_init_weights, n_layer=n_layers))

    def _bimamba_layer(
        self,
        x,
        fwd_blocks,
        bwd_blocks,
        layer_idx,
        debug=False,
    ):
        """
        Args:
            x: Tensor of shape [B, L, D]

        Returns:
            Tensor of shape:
                - [B, L, D] for merge_type='add'
                - [B, L, 2D] for merge_type='concat'
        """

        # Forward sequence processing.
        forward_residual = None
        forward_x = x

        for block in fwd_blocks:
            forward_x, forward_residual = block(
                forward_x,
                forward_residual,
                inference_params=None,
            )

        forward_x = self.forward_convs[layer_idx](
            forward_x.transpose(1, 2)
        ).transpose(1, 2)

        forward_output = (
            forward_x + forward_residual
            if forward_residual is not None
            else forward_x
        )

        # Reverse the sequence for backward processing.
        if bwd_blocks is None:
            return forward_output

        backward_residual = None
        backward_x = torch.flip(x, dims=[1])

        for block in bwd_blocks:
            backward_x, backward_residual = block(
                backward_x,
                backward_residual,
                inference_params=None,
            )

        backward_x = self.backward_convs[layer_idx](
            backward_x.transpose(1, 2)
        ).transpose(1, 2)

        backward_output = (
            backward_x + backward_residual
            if backward_residual is not None
            else backward_x
        )

        # Restore original temporal order.
        backward_output = torch.flip(backward_output, dims=[1])

        if self.merge_type == "concat":
            return torch.cat([forward_output, backward_output], dim=-1)

        return forward_output + backward_output

    def forward(self, x, key_padding_mask=None, debug=False):
        """
        Args:
            x:
                Input tensor of shape [B, L, D].

            key_padding_mask:
                Optional boolean tensor of shape [B, L].
                Positions containing True are ignored by self-attention.
                It is used only when attention=True.

            debug:
                Print intermediate tensor shapes.

        Returns:
            Output tensor of shape:
                - [B, L, D] if merge_type='add'
                - [B, L, D * 2**n_layers] if merge_type='concat'
        """

        residual_outer = x

        for i in range(self.n_layers):
            residual_inner = x

            if debug:
                print(f"Layer {i} input shape: {x.shape}")

            # Bidirectional Mamba + temporal convolution.
            x = self._bimamba_layer(
                x=x,
                fwd_blocks=self.forward_layers[i],
                bwd_blocks=self.backward_layers[i],
                layer_idx=i,
                debug=debug,
            )

            if debug:
                print(f"Layer {i} Bi-Mamba output shape: {x.shape}")

            # Pre-norm multi-head self-attention.
            if self.use_attention:
                attention_input = self.attention_norms[i](x)

                attention_output, _ = self.attention_layers[i](
                    query=attention_input,
                    key=attention_input,
                    value=attention_input,
                    key_padding_mask=key_padding_mask,
                    need_weights=False,
                )

                x = x + self.attention_dropouts[i](attention_output)

                if debug:
                    print(f"Layer {i} attention output shape: {x.shape}")

            # Project the input skip connection, especially necessary for concat.
            residual_projection = self.residual_convs[i](
                residual_inner.transpose(1, 2)
            ).transpose(1, 2)

            x = x + residual_projection

            if debug:
                print(f"Layer {i} final output shape: {x.shape}")

        # An outer residual can only be added if concat did not alter D.
        if x.shape == residual_outer.shape:
            x = x + residual_outer

        return x

class ResidualConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        dropout=0.0,
    ):
        super().__init__()

        if in_channels != out_channels:
            raise ValueError(
                "ResidualConvBlock requires "
                "in_channels == out_channels"
            )

        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=kernel_size // 2,
        )
        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size,
            padding=kernel_size // 2,
        )
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, x):
        residual = x

        x = self.act(self.conv1(x))
        x = self.dropout(x)
        x = self.act(self.conv2(x))

        x = self.norm(
            x.transpose(1, 2)
        ).transpose(1, 2)

        return x + residual

class SignalDecoder(nn.Module):
    def __init__(
        self,
        in_dim,
        out_channels=1,
        mamba_params=None,
        kernel_size=5,
        norm_types=("layer", "batch"),
        signal_type="ppg",
        activation=None,
        attention=False,
        mamba=False,
    ):
        super().__init__()

        if out_channels != 1:
            raise ValueError(
                "This decoder is configured for one output signal channel."
            )

        self.out_channels = out_channels
        self.mamba = mamba

        if isinstance(norm_types, str):
            norm_types = [norm_types, norm_types]

        self.residual_conv = ResidualConvBlock(
            in_channels=in_dim,
            out_channels=in_dim,
            kernel_size=kernel_size,
            dropout=0.0,
        )

        self.final_conv = nn.Conv1d(
            in_channels=in_dim,
            out_channels=out_channels,
            kernel_size=1,
        )

        if activation == "tanh":
            self.act_in = nn.Tanh()
        elif activation == "relu":
            self.act_in = nn.ReLU()
        else:
            self.act_in = nn.Identity()

        if self.mamba:
            self.mamba_params = mamba_params.copy()

            if signal_type == "ppg":
                self.mamba_params["n_mamba_blocks"] = 2
                self.mamba_params["n_layers"] = 3
            elif signal_type == "ecg":
                self.mamba_params["n_mamba_blocks"] = 2
                self.mamba_params["n_layers"] = 3
            else:
                self.mamba_params["n_mamba_blocks"] = 2
                self.mamba_params["n_layers"] = 3

            version = self.mamba_params["version"]
            self.mamba_params["d_state"] = (
                16 if version == 1 else 64
            )

            self.before_norm = get_norm_layer(
                norm_types[0],
                in_dim,
            )
            self.after_norm = get_norm_layer(
                norm_types[1],
                in_dim,
            )

            self.bima = BidirectionalMamba(
                d_model=in_dim,
                d_state=self.mamba_params["d_state"],
                n_mamba_blocks=self.mamba_params["n_mamba_blocks"],
                n_layers=self.mamba_params["n_layers"],
                mamba_version=version,
                merge_type="add",
                headdim=self.mamba_params.get("headdim"),
                is_mimo=self.mamba_params.get("is_mimo", False),
                mimo_rank=self.mamba_params.get("mimo_rank"),
                chunk_size=self.mamba_params.get("chunk_size"),
                is_outproj_norm=self.mamba_params.get(
                    "is_outproj_norm",
                    False,
                ),
                dtype=(
                    self.mamba_params.get("dtype")
                    or torch.float32
                ),
                attention=attention
            )

    def forward(self, x):
        return self.forward_head(self.forward_mamba(x))

    def forward_mamba(self, x):
        """Bi-Mamba half of the decoder (identity if mamba=False); split out for decoder fusion (Oct 31 2026)."""
        if x.ndim != 3:
            raise ValueError(
                f"Expected x with shape [B, C, L], "
                f"got {tuple(x.shape)}"
            )

        if self.mamba:
            x_res = x

            x = handle_norm(
                x,
                self.before_norm,
                before_mamba=True,
            )

            x = self.bima(x)

            x = handle_norm(
                x,
                self.after_norm,
                before_mamba=False,
            )

            x = x + x_res
        return x

    def forward_head(self, x):
        x = self.act_in(x)
        x = self.residual_conv(x)
        x = self.final_conv(x)

        if x.shape[1] != self.out_channels:
            raise RuntimeError(
                f"Expected decoder output with "
                f"{self.out_channels} channels, "
                f"got {tuple(x.shape)}"
            )

        return x
    
class LocalCrossAttention(nn.Module):
    """Nov 3 2026. Target features attend to the other two modalities' features within +/-window samples, with a
    learned relative-position bias per head, source and lag (init 0). Absent sources are masked out; if no source
    is available the output is 0. Output projection zero-initialised, so the module outputs exactly 0 at init."""
    def __init__(self, dim=64, attn_dim=32, heads=4, window=62):
        super().__init__()
        if attn_dim % heads:
            raise ValueError("attn_dim must be divisible by heads")
        self.h, self.dk, self.w = heads, attn_dim, int(window)
        self.q = nn.Linear(dim, attn_dim)
        self.k = nn.Linear(dim, attn_dim)
        self.v = nn.Linear(dim, attn_dim)
        self.o = nn.Linear(attn_dim, dim)
        nn.init.zeros_(self.o.weight); nn.init.zeros_(self.o.bias)
        self.rel_bias = nn.Parameter(torch.zeros(heads, 2, 2 * self.w + 1))
        self._cache = {}

    def _lag(self, L, device):
        key = (L, str(device))
        hit = self._cache.get(key)
        if hit is not None and hit[0].is_inference() and not torch.is_inference_mode_enabled():
            hit = None      # built under inference_mode: unusable for autograd, rebuild
        if hit is None:
            i = torch.arange(L, device=device)
            lag = i[None, :] - i[:, None]                                   # [L(query), L(key)]
            self._cache[key] = (lag.clamp(-self.w, self.w) + self.w, lag.abs() <= self.w)
        return self._cache[key]

    def forward(self, x, sources, avail):
        """x: [B,C,L] target features; sources: two [B,C,L]; avail: [B,2] bool availability of the sources."""
        B, C, L = x.shape
        H, dh = self.h, self.dk // self.h
        q = self.q(x.transpose(1, 2)).view(B, L, H, dh).transpose(1, 2)                    # [B,H,L,dh]
        kv_in = torch.cat([s_.transpose(1, 2) for s_ in sources], dim=1)                    # [B,2L,C]
        k = self.k(kv_in).view(B, 2 * L, H, dh).transpose(1, 2)
        v = self.v(kv_in).view(B, 2 * L, H, dh).transpose(1, 2)
        idx, band = self._lag(L, x.device)
        neg = torch.finfo(q.dtype).min / 4
        bias = torch.cat([torch.where(band, self.rel_bias[:, j][:, idx], torch.full_like(self.rel_bias[:, j][:, idx], neg))
                          for j in range(2)], dim=-1).to(q.dtype)                            # [H,L,2L]
        any_av = avail.any(dim=1)
        av = torch.where(any_av[:, None], avail, torch.ones_like(avail))                    # rows with none: allow, zero later
        key_ok = av.repeat_interleave(L, dim=1)                                             # [B,2L]
        mask = bias[None] + torch.where(key_ok, 0.0, neg).to(q.dtype)[:, None, None, :]    # [B,H,L,2L]
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        o = self.o(o.transpose(1, 2).reshape(B, L, self.dk)).transpose(1, 2)               # [B,C,L]
        return o * any_av.to(o.dtype)[:, None, None]


class DecoderFusionAdapter(nn.Module):
    """Conv1d(k) -> SiLU -> Conv1d(1), last layer zero-initialised (output exactly 0 at init)."""
    def __init__(self, in_ch, out_ch, hidden=64, kernel_size=9):
        super().__init__()
        self.conv_in = nn.Conv1d(in_ch, hidden, kernel_size, padding=kernel_size // 2)
        self.act = nn.SiLU()
        self.conv_out = nn.Conv1d(hidden, out_ch, 1)
        nn.init.zeros_(self.conv_out.weight)
        nn.init.zeros_(self.conv_out.bias)

    def forward(self, x):
        return self.conv_out(self.act(self.conv_in(x)))


class ResidualFilter(nn.Module):
    """Nov 5 2026: learned correction of one modality's raw-input decoder residual (z + x_in).
    Input [x_in, mask] (B,2,L) -> Conv(k)-SiLU-Conv(k)-SiLU-Conv(1) -> (B,1,L), added to x_in. The last conv is
    zero-initialised, so the residual is exactly unchanged at init. It sees only the signal itself, so it can learn
    to remove noise it recognises in the waveform (no task or corruption flag is needed)."""
    def __init__(self, hidden=16, kernel_size=15):
        super().__init__()
        self.net = nn.Sequential(nn.Conv1d(2, hidden, kernel_size, padding=kernel_size // 2), nn.SiLU(),
                                 nn.Conv1d(hidden, hidden, kernel_size, padding=kernel_size // 2), nn.SiLU(),
                                 nn.Conv1d(hidden, 1, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x, mask):
        return x + self.net(torch.cat([x, mask], dim=1))


class CrossModalAttention(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.0, aware=False, rope_min_period=8.0, rope_max_period=1000.0,
                 corruption_aware=False):
        """aware (Nov 1 2026): time-, modality- and mask-aware attention, all terms zero-initialised
        (identical outputs at init up to kernel rounding). Acts on queries/keys only, values unchanged:
          * time:     rotary position encoding (period range rope_min/max_period samples) mixed in per head
                      through a gate g_h (init 0): q <- q + g_h * (rope(q) - q), same for k
          * modality: learned embedding per modality (ECG, PPG, BP) added to the q/k input (init 0)
          * mask:     learned embedding per token state (0 observed, 1 missing = mask 0 in an available
                      modality, 2 absent modality) added to the q/k input (init 0); absent modalities keep
                      the hard key-padding mask."""
        super().__init__()
        self.num_heads = num_heads
        self.aware = bool(aware)
        if self.aware:
            if (dim // num_heads) % 2 != 0:
                raise ValueError("aware attention needs an even head dimension")
            self.mod_emb = nn.Parameter(torch.zeros(3, dim))     # token order [ECG, PPG, BP]
            self.state_emb = nn.Parameter(torch.zeros(3, dim))   # observed / missing / absent
            self.rope_gate = nn.Parameter(torch.zeros(num_heads))
            half = (dim // num_heads) // 2
            periods = torch.logspace(math.log10(rope_min_period), math.log10(rope_max_period), half)
            self.register_buffer("rope_omega", 2 * math.pi / periods, persistent=False)
        # Nov 3 2026: corruption flags (per sample and modality: 0 none, 1 noisy, 2 low-resolution), supplied by
        # the caller. Zero-initialised embeddings added to the q/k input (corrupt_emb) and to the residual stream
        # (corrupt_res) of every token of a corrupted modality. Requires aware=True.
        self.corruption_aware = bool(corruption_aware)
        if self.corruption_aware:
            if not self.aware:
                raise ValueError("corruption_aware requires aware=True")
            self.corrupt_emb = nn.Parameter(torch.zeros(2, dim))
            self.corrupt_res = nn.Parameter(torch.zeros(2, dim))

        if dim % num_heads != 0:
            raise ValueError(
                f"dim={dim} must be divisible by num_heads={num_heads}"
            )

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm = nn.LayerNorm(dim)

    def forward(
        self,
        ecg_feat,
        ppg_feat,
        bp_feat,
        modality_mask=None,
        token_masks=None,
        corruption=None,
    ):
        """
        Args:
            ecg_feat, ppg_feat, bp_feat:
                Feature tensors with shape [B, C, L].

            modality_mask:
                Optional availability mask with shape [B, 3].

                Modality order:
                    [PPG, ECG, BP]

                Values:
                    1 = available
                    0 = unavailable

        Returns:
            ecg_fused, ppg_fused, bp_fused:
                Each with shape [B, C, L].
        """

        if not (
            ecg_feat.ndim == ppg_feat.ndim == bp_feat.ndim == 3
        ):
            raise ValueError(
                "All feature tensors must have shape [B, C, L]"
            )

        B, C, L = ecg_feat.shape

        if ppg_feat.shape != (B, C, L):
            raise ValueError(
                f"ppg_feat has shape {tuple(ppg_feat.shape)}, "
                f"expected {(B, C, L)}"
            )

        if bp_feat.shape != (B, C, L):
            raise ValueError(
                f"bp_feat has shape {tuple(bp_feat.shape)}, "
                f"expected {(B, C, L)}"
            )

        # Sequence order is explicitly:
        # [ECG tokens, PPG tokens, BP tokens]
        q = torch.cat(
            [ecg_feat, ppg_feat, bp_feat],
            dim=-1,
        ).transpose(1, 2)  # [B, 3L, C]

        key_padding_mask = None

        if modality_mask is not None:
            if modality_mask.shape != (B, 3):
                raise ValueError(
                    f"modality_mask must have shape {(B, 3)}, "
                    f"got {tuple(modality_mask.shape)}"
                )

            modality_mask = modality_mask.to(
                device=q.device,
                dtype=torch.bool,
            )

            # modality_mask uses 1=available, 0=unavailable.
            # MultiheadAttention uses True=masked/ignored.
            unavailable = ~modality_mask

            # Current q order: ECG, PPG, BP.
            unavailable_ecg = unavailable[:, 1:2]
            unavailable_ppg = unavailable[:, 0:1]
            unavailable_bp = unavailable[:, 2:3]

            key_padding_mask = torch.cat(
                [
                    unavailable_ecg.expand(-1, L),
                    unavailable_ppg.expand(-1, L),
                    unavailable_bp.expand(-1, L),
                ],
                dim=1,
            )  # [B, 3L]

        cor_tok = None
        if getattr(self, "corruption_aware", False) and corruption is not None:
            c = corruption.to(q.device).round().long()[:, [1, 0, 2]]                 # [B,3] -> ECG, PPG, BP order
            onehot = torch.stack([(c == 1), (c == 2)], dim=-1).to(q.dtype)          # [B,3,2]
            cor_tok = onehot.repeat_interleave(L, dim=1)                             # [B,3L,2]
        if getattr(self, "aware", False):
            out = self._aware_attention(q, B, L, modality_mask, key_padding_mask, token_masks, cor_tok)
            if cor_tok is not None:
                out = out + cor_tok @ self.corrupt_res.to(q.dtype)
        else:
            out, _ = self.cross_attn(
                q,
                q,
                q,
                key_padding_mask=key_padding_mask,
                need_weights=False,
            )

        out = self.norm(q + out)  # [B, 3L, C]
        out = out.transpose(1, 2)  # [B, C, 3L]

        ecg_fused, ppg_fused, bp_fused = out.split(L, dim=-1)

        return ecg_fused, ppg_fused, bp_fused

    def _rope(self, x, pos):
        """x: [B, H, N, d] (d even), pos: [N] sample index. Rotates feature pairs (2i, 2i+1)."""
        ang = pos[:, None].to(x.dtype) * self.rope_omega.to(x.dtype)[None, :]      # [N, d/2]
        cos, sin = ang.cos(), ang.sin()
        x1, x2 = x[..., 0::2], x[..., 1::2]
        return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).flatten(-2)

    def _aware_attention(self, x, B, L, modality_mask, key_padding_mask, token_masks, cor_tok=None):
        """Same projections/out_proj as self.cross_attn; queries/keys get modality + state embeddings and gated RoPE."""
        C, H = x.shape[-1], self.num_heads
        d = C // H
        dev, dt = x.device, x.dtype
        mod_idx = torch.arange(3, device=dev).repeat_interleave(L)                  # [3L], order ECG, PPG, BP
        if modality_mask is not None:
            avail = modality_mask.to(device=dev, dtype=torch.bool)[:, [1, 0, 2]]     # -> ECG, PPG, BP
        else:
            avail = torch.ones(B, 3, dtype=torch.bool, device=dev)
        if token_masks is not None:
            obs = torch.cat([m.reshape(B, -1)[:, :L] > 0 for m in token_masks], dim=1)   # [B, 3L]
        else:
            obs = torch.ones(B, 3 * L, dtype=torch.bool, device=dev)
        absent = (~avail).repeat_interleave(L, dim=1)
        state = torch.where(absent, 2, torch.where(obs, 0, 1))                      # [B, 3L]
        qk_in = x + self.mod_emb.to(dt)[mod_idx][None] + self.state_emb.to(dt)[state]
        if cor_tok is not None:
            qk_in = qk_in + cor_tok @ self.corrupt_emb.to(dt)
        W, b = self.cross_attn.in_proj_weight, self.cross_attn.in_proj_bias
        q = F.linear(qk_in, W[:C], None if b is None else b[:C])
        k = F.linear(qk_in, W[C:2 * C], None if b is None else b[C:2 * C])
        v = F.linear(x, W[2 * C:], None if b is None else b[2 * C:])
        q, k, v = (t.view(B, 3 * L, H, d).transpose(1, 2) for t in (q, k, v))        # [B, H, 3L, d]
        pos = torch.arange(L, device=dev).repeat(3)
        g = self.rope_gate.to(dt).view(1, H, 1, 1)
        q = q + g * (self._rope(q, pos) - q)
        k = k + g * (self._rope(k, pos) - k)
        attn_mask = None if key_padding_mask is None else (~key_padding_mask)[:, None, None, :]
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask,
                                           dropout_p=self.cross_attn.dropout if self.training else 0.0)
        o = o.transpose(1, 2).reshape(B, 3 * L, C)
        return self.cross_attn.out_proj(o)

# Conditional generation (FiLM, AdaIN)
class FiLMConditioning(nn.Module):
    def __init__(self, in_dim, cond_dim, out_dim):
        super().__init__()
        self.gamma = nn.Linear(cond_dim, out_dim)
        self.beta = nn.Linear(cond_dim, out_dim)
    
    def forward(self, x, cond):
        gamma = self.gamma(cond).unsqueeze(-1)
        beta = self.beta(cond).unsqueeze(-1)
        return gamma * x + beta

# Reconstruction loss on source modality (auto-encoding regularization)
# Mutual information maximization between latent representations
# Contrastive loss: pull together (ecg, bvp) pairs from same timestamp,
#                   push apart pairs from different timestamps

class CrossModalLoss(nn.Module):
    def __init__(self, recon_weight=1.0, contrastive_weight=0.1, temperature=0.1):
        super().__init__()
        self.recon_weight = recon_weight
        self.contrastive_weight = contrastive_weight
        self.temperature = temperature

    def forward(self, pred, target, source_feat, target_feat, timestamps):
        recon_loss = nn.MSELoss()(pred, target)
        contrastive_loss = self.contrastive_loss(source_feat, target_feat, timestamps)
        return self.recon_weight * recon_loss + self.contrastive_weight * contrastive_loss

    def contrastive_loss(self, source_feat, target_feat, timestamps=None):
        """
        source_feat, target_feat: (B, C, L)
        timestamps: unused at batch level, kept for call-site compatibility
        """
        B = source_feat.shape[0]

        src = source_feat.mean(dim=-1)  # (B, C)
        tgt = target_feat.mean(dim=-1)  # (B, C)

        src = F.normalize(src, dim=-1)
        tgt = F.normalize(tgt, dim=-1)

        logits = torch.matmul(src, tgt.T) / self.temperature  # (B, B)

        labels = torch.arange(B, device=logits.device)

        loss_i2t = F.cross_entropy(logits, labels)
        loss_t2i = F.cross_entropy(logits.T, labels)

        return (loss_i2t + loss_t2i) / 2
    
# -------------------------------
# DiBiMa_SubPixelRes Model
# -------------------------------

class BiMa_nn(nn.Module):
    def __init__(self, target_channels=3, num_channels=3, seconds=10,
                 use_mamba=True, use_diffusion=False, use_positional_encoding=False,
                 residual_internal=True, n_mamba_blocks = 1, n_mamba_layers=1, mamba_dim=64, mamba_d_state=16,
                 mamba_version=1, merge_type='add', do=0.2, norm_layers = ["layer", "batch"], 
                 headdim=None, is_mimo=False, mimo_rank=None, chunk_size=None,
                 is_outproj_norm=False, dtype=None, mask_cond=False, attention=False,
                 cross=False, use_mamba_decoder=False, prior_channels=0,
                 bottleneck=None, latent_concat="sample", attention_wired=False, cross_mask="none",
                 bottleneck_blocks=None, bottleneck_layers=None, bottleneck_expand=None, bottleneck_attention=None,
                 early_fusion=False, decoder_fusion=False, decoder_fusion_kernel=9, decoder_fusion_hidden=64,
                 cross_aware=False, corruption_aware=False, decoder_local_attn=False, decoder_local_window=62,
                 decoder_local_heads=4, decoder_local_dim=32, residual_filter=False, residual_filter_kernel=15,
                 residual_filter_hidden=16):
        """
        New options (Oct 2026; all defaults reproduce the original architecture/weights):
          bottleneck:      None -> legacy ("mamba" if use_mamba else the old conv path),
                           "mamba" = Bi-Mamba bottleneck, "none" = identity (no bottleneck; the
                           internal pre-bottleneck residual is skipped so the latent is not doubled).
          latent_concat:   "sample"  -> modality latents concatenated on the time axis, (B, D, 3L) [default]
                           "channel" -> concatenated on the channel axis, (B, 3D, L)
          attention_wired: if True, `attention` is applied in the Bi-Mamba blocks of the encoders,
                           decoders and (if present) bottleneck. If False (legacy), `attention` only
                           reaches the Mamba bottleneck (and train.build_backbone used to drop it).
          bottleneck_blocks / bottleneck_layers / bottleneck_expand / bottleneck_attention (Oct 29 2026):
                           size of the Bi-Mamba bottleneck only ('mamba' / 'mamba_gated'); None = the shared
                           n_mamba_blocks / n_mamba_layers / expand 2 / attention, i.e. the original sizing.
          early_fusion (Oct 30 2026): if True, each modality encoder also receives the other two modalities'
                           signal (+ mask) channels through a zero-initialised Conv1d (fusion_<mod>) whose output is
                           added to conv1's output. Zero init = identical outputs to the model without it.
          cross_mask:      "none"  -> legacy: CrossModalAttention gets all-available unless cond_mask is given
                           "input" -> modality availability derived from the input mask channels
                                      (a modality whose mask is all zero is excluded as key/value)
        """

        super(BiMa_nn, self).__init__()
        # Template-prior ablation: K extra conditioning channels per modality,
        # appended after the 6 signal/mask channels as [ppg x K, ecg x K, bp x K].
        # 0 keeps the original architecture (and weight compatibility).
        self.prior_channels = int(prior_channels or 0)

        self.target_channels = target_channels
        self.num_channels = num_channels
        self.seconds = seconds

        self.use_mamba = use_mamba
        self.n_mamba_layers = n_mamba_layers    
        self.mamba_dim = mamba_dim
        self.mamba_d_state = mamba_d_state
        self.mamba_version = mamba_version
        self.n_mamba_blocks = n_mamba_blocks
        self.merge_type = merge_type
        self.norm_layers = norm_layers
        self.headdim = headdim
        self.is_mimo = is_mimo
        self.mimo_rank = mimo_rank
        self.chunk_size = chunk_size
        self.is_outproj_norm = is_outproj_norm
        self.dtype = dtype
        self.mask_cond = mask_cond
        self.cross = cross
        self.attention = attention
        self.use_mamba_decoder = use_mamba_decoder

        if bottleneck is None:
            bottleneck = "mamba" if use_mamba else "legacy"
        if bottleneck not in {"mamba", "none", "legacy", "mamba_gated"}:
            raise ValueError(f"bottleneck must be 'mamba', 'mamba_gated' or 'none', got {bottleneck!r}")
        if bottleneck in ("mamba", "mamba_gated") and not use_mamba:
            raise ValueError(f"bottleneck={bottleneck!r} requires use_mamba=True")
        if latent_concat not in {"sample", "channel"}:
            raise ValueError(f"latent_concat must be 'sample' or 'channel', got {latent_concat!r}")
        if cross_mask not in {"none", "input"}:
            raise ValueError(f"cross_mask must be 'none' or 'input', got {cross_mask!r}")
        self.bottleneck_type = bottleneck
        self.latent_concat = latent_concat
        self.attention_wired = bool(attention_wired)
        self.cross_mask = cross_mask
        # attention used inside encoder/decoder Bi-Mamba blocks (only when explicitly wired)
        block_attention = bool(attention) and self.attention_wired

        if self.use_mamba:
            enc_out = self.mamba_dim
        else:   
            enc_out = self.target_channels

        self.use_diffusion = use_diffusion
        self.use_positional_encoding = use_positional_encoding
        
        if self.use_diffusion:
            self.time_mlp = TemporalEmbedding(enc_out, enc_out)
        
            if self.use_positional_encoding:        
                self.pos_enc = PositionalEncoding(d_model=enc_out)
        
        # Latent width seen by the bottleneck: D for sample-axis concat, 3D for channel-axis concat
        lat_dim = enc_out * 3 if self.latent_concat == "channel" else enc_out
        self.lat_dim = lat_dim
        if self.bottleneck_type == "none":
            self.before_norm = self.after_norm = None  # bottleneck norms are not used without a bottleneck
        else:
            self.before_norm = get_norm_layer(self.norm_layers[0], lat_dim) 
            self.after_norm = get_norm_layer(self.norm_layers[1], lat_dim) 
        
        if self.mask_cond:
            in_ch = 2 # Signal + Mask
        else:
            in_ch = 1 # Signal only
        in_ch += self.prior_channels  # + template prior channels (0 if disabled)

        mamba_params = {
            "version": self.mamba_version,
            "d_model": self.mamba_dim,
            "d_state": self.mamba_d_state,
            "n_mamba_blocks": self.n_mamba_blocks,
            "n_layers": self.n_mamba_layers,
            "headdim": self.headdim,
            "is_mimo": self.is_mimo,
            "mimo_rank": self.mimo_rank,
            "chunk_size": self.chunk_size,
            "is_outproj_norm": self.is_outproj_norm,
            "dtype": self.dtype,
            "attention": block_attention,  # read by SignalEncoder via mamba_params.get("attention")
        }

        self.ppg_condition_encoder = SignalEncoder(in_ch, enc_out, mamba_params, kernel_size=3, signal_type="ppg", norm_types=self.norm_layers, do=do)
        self.ecg_condition_encoder = SignalEncoder(in_ch, enc_out, mamba_params, kernel_size=5, signal_type="ecg", norm_types=self.norm_layers, do=do)
        self.bp_condition_encoder = SignalEncoder(in_ch, enc_out, mamba_params, kernel_size=3, signal_type="bp", norm_types=self.norm_layers, do=do)
        self.early_fusion = bool(early_fusion)
        if self.early_fusion:
            other_ch = 2 * (2 if self.mask_cond else 1)   # the two other modalities: signal (+ mask) each
            self.fusion_ppg = nn.Conv1d(other_ch, enc_out, 3, padding=1, bias=False)
            self.fusion_ecg = nn.Conv1d(other_ch, enc_out, 5, padding=2, bias=False)
            self.fusion_bp  = nn.Conv1d(other_ch, enc_out, 3, padding=1, bias=False)
            for conv in (self.fusion_ppg, self.fusion_ecg, self.fusion_bp):
                nn.init.zeros_(conv.weight)

        self.residual_internal = residual_internal

        # --- Bottleneck (residual internal) ---
        if self.bottleneck_type == "none":
            self.bottleneck = nn.Identity()
        elif self.bottleneck_type in ("mamba", "mamba_gated"):
            #print("Using Bi-Mamba bottleneck with residual internal")
            b_att = self.attention if bottleneck_attention is None else bool(bottleneck_attention)
            self.bottleneck = BidirectionalMamba(d_model=lat_dim,
                                                     d_state=self.mamba_d_state,
                                                     expand=2 if bottleneck_expand is None else int(bottleneck_expand),
                                                     n_mamba_blocks=self.n_mamba_blocks if bottleneck_blocks is None else int(bottleneck_blocks),
                                                     n_layers=self.n_mamba_layers if bottleneck_layers is None else int(bottleneck_layers),
                                                     mamba_version=self.mamba_version,
                                                     merge_type=self.merge_type, 
                                                     headdim=self.headdim, is_mimo=self.is_mimo, mimo_rank=self.mimo_rank,
                                                     chunk_size=self.chunk_size, is_outproj_norm=self.is_outproj_norm, dtype=self.dtype,
                                                     attention=b_att
                                                )
            
        else:
            #print("Using simple bottleneck with residual internal")
            self.bottleneck = nn.Sequential(
                    nn.Conv1d(enc_out, enc_out, kernel_size=1, padding=0),
                    nn.LayerNorm(enc_out),
                    nn.SiLU(),
                    nn.Dropout(0.3)
            )

        # 'mamba_gated' (Oct 29 2026): multimodal fusion stage for fine-tuning a bottleneck-free model.
        #   z = x + gate * after_norm(BiMamba(before_norm(x)))   (gate per latent channel, initialised to 0)
        # With gate = 0 the latent is exactly the 'none' latent, so a pretrained bottleneck='none'
        # checkpoint loaded into this architecture (only bottleneck/norm/gate keys missing) reproduces the
        # original outputs at step 0. With latent_concat='channel' the Bi-Mamba mixes all three modality
        # blocks (B, 3*D, L). No internal residual is added in decoder_forward (the gate path is the residual).
        if self.bottleneck_type == "mamba_gated":
            self.bottleneck_gate = nn.Parameter(torch.zeros(1, lat_dim, 1))

        if self.cross:
            self.cross_aware = bool(cross_aware)   # Nov 1 2026: time/modality/mask-aware cross attention
            self.corruption_aware = bool(corruption_aware)   # Nov 3 2026: caller-supplied corruption flags
            self.cross_attention = CrossModalAttention(enc_out, aware=self.cross_aware,
                                                       corruption_aware=self.corruption_aware)

        # --- Decoder ---
        out_chs = 1
        self.decoder_ppg = SignalDecoder(enc_out, out_chs, mamba_params, kernel_size=3, norm_types=self.norm_layers, mamba=self.use_mamba_decoder, signal_type="ppg", attention=block_attention)
        self.decoder_ecg = SignalDecoder(enc_out, out_chs, mamba_params, kernel_size=5, norm_types=self.norm_layers, mamba=self.use_mamba_decoder, signal_type="ecg", attention=block_attention)
        self.decoder_bp = SignalDecoder(enc_out, out_chs, mamba_params, kernel_size=3, norm_types=self.norm_layers, mamba=self.use_mamba_decoder, signal_type="bp", attention=block_attention)

        # Oct 31 2026 decoder fusion: per target modality, two zero-initialised adapters that add a time-local
        # conv of the other two modalities' features (gated by availability) at (a) the decoder input and
        # (b) after the decoder Bi-Mamba (if any). Exact identity at init (last conv weight and bias = 0).
        self.decoder_fusion = bool(decoder_fusion)
        if self.decoder_fusion:
            if self.latent_concat != "channel":
                raise ValueError("decoder_fusion requires latent_concat='channel'")
            mk = lambda: DecoderFusionAdapter(2 * enc_out, enc_out, decoder_fusion_hidden, decoder_fusion_kernel)  # noqa: E731
            self.dfus_in_ppg, self.dfus_in_ecg, self.dfus_in_bp = mk(), mk(), mk()
            if self.use_mamba_decoder:
                self.dfus_mid_ppg, self.dfus_mid_ecg, self.dfus_mid_bp = mk(), mk(), mk()
        # Nov 3 2026: local cross-modal attention in the decoder (+/- decoder_local_window samples), next to the
        # decoder-fusion adapters (same two positions). Zero output at init.
        self.decoder_local_attn = bool(decoder_local_attn)
        if self.decoder_local_attn:
            if not self.decoder_fusion:
                raise ValueError("decoder_local_attn requires decoder_fusion")
            mkl = lambda: LocalCrossAttention(enc_out, decoder_local_dim, decoder_local_heads, decoder_local_window)  # noqa: E731
            self.dlat_in_ppg, self.dlat_in_ecg, self.dlat_in_bp = mkl(), mkl(), mkl()
            if self.use_mamba_decoder:
                self.dlat_mid_ppg, self.dlat_mid_ecg, self.dlat_mid_bp = mkl(), mkl(), mkl()
        # Nov 5 2026: learned, zero-initialised filter on each raw-input decoder residual (see ResidualFilter).
        self.residual_filter = bool(residual_filter)
        if self.residual_filter:
            mkr = lambda: ResidualFilter(residual_filter_hidden, residual_filter_kernel)  # noqa: E731
            self.rf_ppg, self.rf_ecg, self.rf_bp = mkr(), mkr(), mkr()

    def encoder_forward(self, x, t=None, cond_mask=None, debug=False, corruption=None):
        """
        x: [B, C_in, L_in] - input with 3 modalities concatenated + mask
        t: [B] timestep
        cond_mask: [B, 3] - 1=available, 0=missing for [ppg,ecg,bp]
        """
        if debug:
            print(f"Input shape: {x.shape}, cond_mask: {cond_mask}")

        if x.ndim == 2:
            x = x.unsqueeze(0)

        # Split into 4 or 6 modalities (shared by both branches)
        K = self.prior_channels
        prior_ppg = prior_ecg = prior_bp = None
        if K > 0 and x.shape[1] == 6 + 3 * K:
            pri = x[:, 6:]
            prior_ppg, prior_ecg, prior_bp = pri[:, :K], pri[:, K:2 * K], pri[:, 2 * K:]
            x = x[:, :6]
        if x.shape[1] == 4:
            ppg, ecg, bp, mask = x.chunk(4, dim=1)
            mask_ppg = mask_ecg = mask_bp = mask
        elif x.shape[1] == 6:
            ppg, ecg, bp, mask_ppg, mask_ecg, mask_bp = x.chunk(6, dim=1)
        else:
            raise ValueError(f"Expected input with 4, 6 or {6 + 3 * K} channels (3 signals + masks [+ 3x{K} priors]), got {x.shape[1]} channels")
        if K > 0 and prior_ppg is None:
            # No prior supplied (denoising, SR, translation, legacy callers): zero prior
            prior_ppg = prior_ecg = prior_bp = torch.zeros(ppg.shape[0], K, ppg.shape[-1], device=ppg.device, dtype=ppg.dtype)

        if self.use_diffusion:
            temb = self.time_mlp(t)  # (B, 64, 1)
            if ppg.ndim == 4: ppg = ppg.squeeze(2)
            if ecg.ndim == 4: ecg = ecg.squeeze(2)
            if bp.ndim == 4:  bp  = bp.squeeze(2)

        if self.mask_cond:
            input_ppg = torch.cat([ppg, mask_ppg], dim=1)
            input_ecg = torch.cat([ecg, mask_ecg], dim=1)
            input_bp  = torch.cat([bp,  mask_bp],  dim=1)
        else:
            input_ppg, input_ecg, input_bp = ppg, ecg, bp
        if K > 0:
            input_ppg = torch.cat([input_ppg, prior_ppg.to(input_ppg.dtype)], dim=1)
            input_ecg = torch.cat([input_ecg, prior_ecg.to(input_ecg.dtype)], dim=1)
            input_bp  = torch.cat([input_bp,  prior_bp.to(input_bp.dtype)],  dim=1)

        if self.use_positional_encoding:
            input_ppg = self.pos_enc(input_ppg)
            input_ecg = self.pos_enc(input_ecg)
            input_bp  = self.pos_enc(input_bp)

        if getattr(self, "early_fusion", False):
            if self.mask_cond:
                o_ppg = torch.cat([ecg, mask_ecg, bp, mask_bp], dim=1)
                o_ecg = torch.cat([ppg, mask_ppg, bp, mask_bp], dim=1)
                o_bp  = torch.cat([ppg, mask_ppg, ecg, mask_ecg], dim=1)
            else:
                o_ppg, o_ecg, o_bp = torch.cat([ecg, bp], 1), torch.cat([ppg, bp], 1), torch.cat([ppg, ecg], 1)
            add_ppg, add_ecg, add_bp = self.fusion_ppg(o_ppg), self.fusion_ecg(o_ecg), self.fusion_bp(o_bp)
        else:
            add_ppg = add_ecg = add_bp = None
        ppg_cond = self.ppg_condition_encoder(input_ppg, add=add_ppg)  # (B, 64, L)
        ecg_cond = self.ecg_condition_encoder(input_ecg, add=add_ecg)  # (B, 64, L)
        bp_cond  = self.bp_condition_encoder(input_bp, add=add_bp)     # (B, 64, L)

        if self.cross:
            if cond_mask is not None:
                modality_mask = cond_mask
            elif self.cross_mask == "input":
                modality_mask = self.modality_availability(mask_ppg, mask_ecg, mask_bp)
            else:
                modality_mask = torch.ones(ppg_cond.shape[0], 3, device=ppg_cond.device)
            if getattr(self, "cross_aware", False):
                ecg_cond, ppg_cond, bp_cond = self.cross_attention(ecg_cond, ppg_cond, bp_cond, modality_mask=modality_mask,
                                                                   token_masks=(mask_ecg, mask_ppg, mask_bp),
                                                                   corruption=corruption)
            else:
                ecg_cond, ppg_cond, bp_cond = self.cross_attention(ecg_cond, ppg_cond, bp_cond, modality_mask=modality_mask)

        if self.latent_concat == "channel":
            x = torch.cat([ppg_cond, ecg_cond, bp_cond], dim=1)   # (B, 3*64, L)
            if self.use_diffusion:
                x = x + temb.repeat(1, 3, 1)  # same time embedding added to each modality slot
        else:
            x = torch.cat([ppg_cond, ecg_cond, bp_cond], dim=-1)  # (B, 64, 3*L)
            if self.use_diffusion:
                x = x + temb  # broadcast time embedding

        if self.bottleneck_type == "none":
            # No bottleneck: the latent is the (cross-attended) encoder output itself. Skip the
            # internal residual, which would otherwise just double z in decoder_forward.
            return x, None

        if self.bottleneck_type == "mamba_gated":
            # Zero-initialised gated residual fusion (see __init__); identical to 'none' while gate == 0.
            x_mamba = handle_norm(x, self.before_norm, before_mamba=True)
            x_mamba = self.bottleneck(x_mamba, debug=debug)
            x_mamba = handle_norm(x_mamba, self.after_norm, before_mamba=False)
            return x + self.bottleneck_gate.to(x.dtype) * x_mamba, None

        residual_premamba = x.clone() if self.residual_internal else None

        # Mamba bottleneck
        if self.bottleneck_type == "mamba":
            x_mamba = handle_norm(x, self.before_norm, before_mamba=True)
            x_mamba = self.bottleneck(x_mamba, debug=debug)
            x = handle_norm(x_mamba, self.after_norm, before_mamba=False)
        else:
            x = self.bottleneck(x)

        return x, residual_premamba
    
    @staticmethod
    def modality_availability(mask_ppg, mask_ecg, mask_bp):
        """[B, 3] bool, order [PPG, ECG, BP]: a modality is available if any of its mask samples is > 0.
        Rows with no available modality fall back to all-available (avoids NaN attention)."""
        avail = torch.stack([m.reshape(m.shape[0], -1).amax(dim=-1) > 0 for m in (mask_ppg, mask_ecg, mask_bp)], dim=1)
        none = ~avail.any(dim=1, keepdim=True)
        return avail | none

    def decoder_forward(self, z, residual_premamba, inputs, model_flags=None, debug=False):
        """
        DiBiMa-Cardio decoder with multimodal source flags.

        Args:
            z: encoded latents, shape (B, C, L_total)
            residual_premamba: optional residual tensor added to z
            inputs: raw input signals, expected shape (B, >=3, L)
                    first 3 channels correspond to [PPG, ECG, BP]
            model_flags: list[str], e.g.
                ["0_1"]     -> PPG -> ECG
                ["0_0"]     -> PPG -> PPG
                ["012_0"]   -> PPG+ECG+BP -> PPG
                ["12_0"]    -> ECG+BP -> PPG
            debug: bool

        Returns:
            [ppg_out, ecg_out, bp_out]
        """

        if model_flags is None or len(model_flags) == 0:
            model_flags = ["0_0", "1_1", "2_2"]

        if debug:
            print(f"decoder_forward: z={z.shape}, inputs={inputs.shape}, model_flags={model_flags}")

        # --------------------------------------------------
        # 1) Optional residual
        # --------------------------------------------------
        if self.residual_internal and residual_premamba is not None:
            z = z + residual_premamba

        # --------------------------------------------------
        # 2) Split latent into 3 modality slots
        # --------------------------------------------------
        n_signals = 3
        B, C, L_total = z.shape

        if self.latent_concat == "channel":
            if C % n_signals != 0:
                raise ValueError(f"z.shape[1]={C} is not divisible by {n_signals}")
            z_ppg, z_ecg, z_bp = z.chunk(n_signals, dim=1)   # each (B, C/3, L)
        else:
            if L_total % n_signals != 0:
                raise ValueError(f"z.shape[-1]={L_total} is not divisible by {n_signals}")

            unit_L = L_total // n_signals
            z_ppg = z[:, :, 0*unit_L:1*unit_L]
            z_ecg = z[:, :, 1*unit_L:2*unit_L]
            z_bp  = z[:, :, 2*unit_L:3*unit_L]

        # --------------------------------------------------
        # 3) Extract raw signal residuals
        # --------------------------------------------------
        if inputs.shape[1] < 3:
            raise ValueError(f"inputs must have at least 3 channels/modalities, got {inputs.shape}")

        
        signal_inputs = inputs[:, :3, :]   # [PPG, ECG, BP]
        ppg_in = signal_inputs[:, 0:1, :]
        ecg_in = signal_inputs[:, 1:2, :]
        bp_in  = signal_inputs[:, 2:3, :]
        if getattr(self, "residual_filter", False):
            if inputs.shape[1] < 6:
                raise ValueError("residual_filter needs the mask channels (inputs[:, 3:6])")
            ppg_in = self.rf_ppg(ppg_in, inputs[:, 3:4, :])
            ecg_in = self.rf_ecg(ecg_in, inputs[:, 4:5, :])
            bp_in = self.rf_bp(bp_in, inputs[:, 5:6, :])

        # Early residual fusion
        z_ppg = z_ppg + ppg_in
        z_ecg = z_ecg + ecg_in
        z_bp  = z_bp + bp_in

        valid_modalities = {"0", "1", "2"}
        route_to_bp = []
        route_to_ecg = []
        route_to_ppg = []
        
        for flag in model_flags:
            if not isinstance(flag, str) or "_" not in flag:
                raise ValueError(f"Invalid flag '{flag}'. Expected format like '012_0'.")

            src_str, tgt_str = flag.split("_")

            if len(tgt_str) != 1 or tgt_str not in valid_modalities:
                raise ValueError(f"Invalid target in flag '{flag}'.")

            if len(src_str) == 0 or any(ch not in valid_modalities for ch in src_str):
                raise ValueError(f"Invalid source modalities in flag '{flag}'.")

            src_modalities = [int(ch) for ch in src_str]
            tgt_modality = int(tgt_str)
            primary_src = src_modalities[0]

            if debug:
                print(
                    f"flag={flag} | src_modalities={src_modalities} | "
                    f"primary_src={primary_src} | tgt_modality={tgt_modality}"
                )
            
            # Self-denoising (e.g. 0_0, 012_0): fuse ALL non-target sources into target
            # or 
            # Cross-modal generation (e.g. 0_1, 02_1): route only specified sources into target, leave rest as is
            for src_mod in src_modalities:
                if src_mod == tgt_modality:
                    continue
                if src_mod == 1 and tgt_modality == 0:
                    route_to_ppg.append("ecg")
                elif src_mod == 2 and tgt_modality == 0:
                    route_to_ppg.append("bp")
                elif src_mod == 0 and tgt_modality == 1:
                    route_to_ecg.append("ppg")
                elif src_mod == 2 and tgt_modality == 1:
                    route_to_ecg.append("bp")
                elif src_mod == 0 and tgt_modality == 2:
                    route_to_bp.append("ppg")
                elif src_mod == 1 and tgt_modality == 2:
                    route_to_bp.append("ecg")
    
        if debug:
            print(f"z_ppg={z_ppg.shape}, z_ecg={z_ecg.shape}, z_bp={z_bp.shape}")


        # --------------------------------------------------
        # 4) Apply cross-modal routing: fuse source latents into target
        # --------------------------------------------------
        modality_map = {"ppg": z_ppg, "ecg": z_ecg, "bp": z_bp}

        # PPG target: accumulate sources into z_ppg
        if route_to_ppg:
            fused = z_ppg.clone()
            for src_name in route_to_ppg:
                fused = fused + modality_map[src_name]
            z_ppg = fused / (len(route_to_ppg) + 1)  # normalize by total number of contributions

        # ECG target
        if route_to_ecg:
            fused = z_ecg.clone()
            for src_name in route_to_ecg:
                fused = fused + modality_map[src_name]
            z_ecg = fused / (len(route_to_ecg) + 1)

        # BP target
        if route_to_bp:
            fused = z_bp.clone()
            for src_name in route_to_bp:
                fused = fused + modality_map[src_name]
            z_bp = fused / (len(route_to_bp) + 1)

        # --------------------------------------------------
        # 6) Decode all modality branches
        # --------------------------------------------------
        if getattr(self, "decoder_fusion", False):
            outs = self._fused_decode(modality_map, (z_ppg, z_ecg, z_bp), inputs)
        else:
            ppg_out = self.decoder_ppg(z_ppg)
            ecg_out = self.decoder_ecg(z_ecg)
            bp_out  = self.decoder_bp(z_bp)
            outs = [ppg_out, ecg_out, bp_out]

        if debug:
            print(f"outs: {[o.shape for o in outs]}")

        return outs
    
    def _fused_decode(self, src, tgt, inputs):
        """Decoder fusion (see __init__). src: pre-routing per-modality latents {ppg,ecg,bp};
        tgt: routed decoder inputs (ppg, ecg, bp). Sources are gated by modality availability (input masks)."""
        names = ("ppg", "ecg", "bp")
        B = inputs.shape[0]
        if inputs.shape[1] >= 6:
            av = self.modality_availability(inputs[:, 3:4], inputs[:, 4:5], inputs[:, 5:6])
        else:
            av = torch.ones(B, 3, dtype=torch.bool, device=inputs.device)
        g = av.to(tgt[0].dtype).view(B, 3, 1, 1)
        def others(feats, i):
            return torch.cat([feats[j] * g[:, j] for j in range(3) if j != i], dim=1)
        s = [src[n] for n in names]
        h = [tgt[i] + getattr(self, f"dfus_in_{n}")(others(s, i)) for i, n in enumerate(names)]
        local = getattr(self, "decoder_local_attn", False)
        def oth(i):
            return [j for j in range(3) if j != i]
        if local:
            h = [h[i] + getattr(self, f"dlat_in_{n}")(h[i], [s[j] for j in oth(i)], av[:, oth(i)])
                 for i, n in enumerate(names)]
        decs = (self.decoder_ppg, self.decoder_ecg, self.decoder_bp)
        if self.use_mamba_decoder:
            h = [d.forward_mamba(x) for d, x in zip(decs, h)]
            h0 = h
            h = [h0[i] + getattr(self, f"dfus_mid_{n}")(others(h0, i)) for i, n in enumerate(names)]
            if local:
                h = [h[i] + getattr(self, f"dlat_mid_{n}")(h[i], [h0[j] for j in oth(i)], av[:, oth(i)])
                     for i, n in enumerate(names)]
        return [d.forward_head(x) for d, x in zip(decs, h)]

    # ---------- regression SR path ----------
    def forward_regression(self, x, t=None, debug=False, model_flags=None, return_latent = False, corruption=None): 
        """
        lr: low-res input  (B, C_lr, L_lr)
        hr: high-res target (B, C_hr, L_hr), optional (needed for loss)
        returns:
            pred, loss, log_dict
        """
        x_copy = x.clone()  # Avoid in-place issues
        if x_copy.ndim == 4: 
            x_copy = x_copy[:, :, 0, :]  # If input is (B, C, 1, L), squeeze to (B, C, L)
        # encode
        if debug:
            print(f"Input LR shape: {x_copy.shape}")

        z, residual_premamba = self.encoder_forward(x_copy, debug=debug, corruption=corruption)          # (B, 256, L_enc)
        if debug:
            print(f"Encoded tokens shape: {z.shape}")

        # decode SR
        ppg, ecg, bp = self.decoder_forward(z, inputs=x_copy, residual_premamba=residual_premamba, debug=debug, model_flags=model_flags)   # (B, C_hr, L_hr)
        if debug:
            print(f"Final SR shape: {ppg.shape}, {ecg.shape}, {bp.shape}")
        
        if return_latent:
            return ppg, ecg, bp, z
        return ppg, ecg, bp

    def forward_diffusion(self, x, t, cond_mask=None, model_flags=None, debug=False, return_latent=False, base_seed=None,
                          corruption=None):
        """
        DiBiMa-Cardio forward + model_flags!
        x: noised (B, K=4, L)
        t: (B,)
        cond_mask: (1,1,L) conditions quarters
        model_flags: str of maximum lenght 3 containing modalities to use in input
        Returns: pred_noise_list [ppg, ecg, bp], optionally latent z if return_latent=True
        """
        if base_seed is not None:
            torch.manual_seed(base_seed)
        if debug:
            print(f"forward_diffusion: x {x.shape}, t {t.shape}, mask {cond_mask.shape if cond_mask else None}, model_flags{model_flags}")
        
        x_copy = x.clone()  # Avoid in-place issues
        if x_copy.ndim == 4: 
            x_copy = x_copy[:, :, 0, :]  # If input is (B, C, 1, L), squeeze to (B, C, L)
        
        model_input = x_copy.clone()  # (B, C, L)

        z, residual_premamba = self.encoder_forward(model_input, t=t, cond_mask=cond_mask, debug=debug, corruption=corruption)
        ppg, ecg, bp = self.decoder_forward(z, inputs=model_input, residual_premamba=residual_premamba, model_flags=model_flags, debug=debug)

        if return_latent:
            return ppg, ecg, bp, z
        
        return ppg, ecg, bp

    # ---------- unified forward ----------
    def forward(self, *args, **kwargs):
        """
        If use_diffusion = False:
            expect forward(lr, hr=None) -> SR
        If use_diffusion = True:
            expect forward(x_t_hr, t, cond_lr) -> pred_hr_noise_or_sample
        """
        if self.use_diffusion:
            return self.forward_diffusion(*args, **kwargs)
        else:
            return self.forward_regression(*args, **kwargs)

class BiMa(pl.LightningModule):
    def __init__(
        self, 
        model,
        total_steps,
        epochs=10,
        learning_rate=1e-4,
        criterion=nn.MSELoss(),
        criterion_impute=nn.MSELoss(reduction='none'),  # Per-sample loss for imputation
        modality_sampling=False,
        debug=False,
        is_moe=False,
        plot=False, 
        norm_layers = ["layer", "batch"],
        name="bima", 
        cross=False
    ):  
        super().__init__()
        self.save_hyperparameters(ignore=['model', 'criterion', 'criterion_impute', 'model_list'])
        
        self.automatic_optimization = False
        self.is_moe = is_moe
        self.total_steps=total_steps
        if not self.is_moe:
            self.model = model
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.final_learning_rate = 1e-4  # end of the linear LR decay (overridden by train.py)
        self.criterion = criterion
        self.criterion_impute = criterion_impute 
        self.modality_sampling = modality_sampling
        self.norm_layers = norm_layers  
        self.use_diffusion = False 
        self.name = name
        self.cross = cross
        if self.cross:
            self.cross_loss = nn.MSELoss() #CrossModalLoss(recon_weight=1.0, contrastive_weight=0.1)
        else:
            self.cross_loss = nn.MSELoss()  # Default to MSE if not using cross-modal loss
            
        self.train_losses_list = []
        self.val_losses_list = []

        self.train_losses = {"ppg": {}, "ecg": {}, "bp": {}}
        self.val_losses = {"ppg": {}, "ecg": {}, "bp": {}}
        self.train_metrics = {"ppg": {}, "ecg": {}, "bp": {}}
        self.val_metrics = {"ppg": {}, "ecg": {}, "bp": {}}

        self.val_preds = {"ppg": {}, "ecg": {}, "bp": {}}
        self.val_targets = {"ppg": {}, "ecg": {}, "bp": {}}
        self.train_preds = {"ppg": {}, "ecg": {}, "bp": {}}
        self.train_targets = {"ppg": {}, "ecg": {}, "bp": {}}

        self.train_losses_epoch = {"ppg": {}, "ecg": {}, "bp": {}}
        self.val_losses_epoch = {"ppg": {}, "ecg": {}, "bp": {}}

        self.inputs, self.targets, self.outputs, self.masks = None, None, None, None

        self.debug = debug
        self.to_plot = plot
        self.fig_inout, self.ax_inout = None, None
        if self.to_plot:
            self.N = 4 
            self.fig_inout, self.ax_inout = plt.subplots(self.N, 3, figsize=(48, 12), sharex=True, sharey=False)
            self.fig_loss, self.ax_loss = plt.subplots(3, 2, figsize=(24, 12)) # 3 modalities, 2 columns for train/val
            self.fig_metrics, self.ax_metrics = plt.subplots(3, 2, figsize=(24, 12)) # 3 modalities, 2 columns for train/val
            plt.ion()
            plt.show()
                    
    def forward(self, *args, **kwargs):
        
        return self.model.forward_regression(*args, debug=self.debug, **kwargs)

    def on_fit_end(self):
        """Close GUI after final validation"""
        if hasattr(self, 'fig_inout') and self.fig_inout is not None:
            plt.close(self.fig_inout)
        if hasattr(self, 'fig_loss') and self.fig_loss is not None:
            plt.close(self.fig_loss)
        if hasattr(self, 'fig_metrics') and self.fig_metrics is not None:
            plt.close(self.fig_metrics)  
        print("Plot window closed.")


    def compute_loss_chunk(self, outputs, targets, chunk_type, masks=None, tgt_mean=0.0, tgt_std=1.0):
        criterion = self.criterion
        criterion_impute = self.criterion_impute

        losses = {}
        scales = {"ppg": 1.0, "ecg": 1.5 if self.use_diffusion else 1.0, "bp": 1.0}

        if chunk_type in ["denoised", "sr", "imputed", "forecast"]:
            for mod in ["ppg", "ecg", "bp"]:
                pred_key = f"{mod}_{chunk_type}"
                if pred_key not in outputs:
                    continue

                if mod not in losses:
                    losses[mod] = {}

                pred = outputs[pred_key]
                tgt = targets[mod]
                ecg_tgt = targets["ecg"] if "ecg" in targets else None
                rpeaks = self.safe_ecg_findpeaks_batch(ecg_tgt, sampling_rate=125, tgt_mean=tgt_mean, tgt_std=tgt_std) 

                if chunk_type in ["imputed", "forecast"] and masks is not None:
                    mask = masks[chunk_type][mod]
                    while mask.ndim > pred.ndim and mask.shape[1] == 1:
                        mask = mask.squeeze(1)

                    if mask.ndim == pred.ndim - 1:
                        mask = mask.unsqueeze(1)

                    if mask.shape != pred.shape:
                        raise RuntimeError(
                            f"{pred_key}: mask shape={tuple(mask.shape)}, "
                            f"pred shape={tuple(pred.shape)}"
                        )

                    mask = mask.to(
                        device=pred.device,
                        dtype=pred.dtype,
                    )
                    assert mask.shape == pred.shape, f"{pred_key}: mask shape mismatch"
                    missing_mask = (mask == 0)
                    assert missing_mask.sum() > 0, f"{pred_key}: empty mask"

                    if isinstance(self.criterion_impute, nn.MSELoss):
                        element_loss = criterion_impute(pred, tgt)
                    elif isinstance(self.criterion_impute, QRSAwareMSELoss):
                        element_loss = criterion_impute(pred, tgt, rpeaks)
                    if mod != "ecg" and self.use_diffusion:
                        element_loss = element_loss * 2.0  # Emphasize PPG/BP imputation/forecasting losses
                    loss = (element_loss * missing_mask).sum() / (missing_mask.sum() + 1e-8)
                else:
                    loss = criterion(pred, tgt).mean()

                loss = loss * scales[mod]
                losses[mod][pred_key] = loss

        elif chunk_type == "cross":
            cross_pairs = [
                ("ppg2ecg", "ecg", 3.0 if self.use_diffusion else 1.0),
                ("ppg2bp", "bp", 2.0),
                ("ecg2ppg", "ppg", 2.0),
                ("ecg2bp", "bp", 2.0),
                ("bp2ppg", "ppg", 1.5),
                ("bp2ecg", "ecg", 3.0 if self.use_diffusion else 1.0),
            ]
            criterion = self.cross_loss
            for pred_key, tgt_mod, weight in cross_pairs:
                if pred_key not in outputs:
                    continue

                if tgt_mod not in losses:
                    losses[tgt_mod] = {}
                
                if self.cross and self.cross_loss.__class__.__name__ == "CrossModalLoss":
                    timestamps = torch.arange(outputs[pred_key].shape[-1], device=outputs[pred_key].device).unsqueeze(0).repeat(outputs[pred_key].shape[0], 1)
                    losses[tgt_mod][pred_key] = criterion(outputs[pred_key], targets[tgt_mod], outputs[pred_key], targets[tgt_mod], timestamps).mean() * weight
                else:
                    losses[tgt_mod][pred_key] = criterion(outputs[pred_key], targets[tgt_mod]).mean() * weight

        losses_flat = [loss for mod_losses in losses.values() for loss in mod_losses.values()]
        if len(losses_flat) == 0:
            raise ValueError(f"No losses computed for chunk_type={chunk_type}")

        total_loss = torch.stack(losses_flat).mean()
        return total_loss, losses

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        opt.zero_grad()

        targets, noised, masked, masked_sr, masked_forecast, masks, tgt_means, tgt_stds = batch

        ecg_mean = tgt_means["ecg"].to(self.device, non_blocking=True)
        ecg_std = tgt_stds["ecg"].to(self.device, non_blocking=True)

        ppg = targets["ppg"].to(self.device, non_blocking=True)
        ecg = targets["ecg"].to(self.device, non_blocking=True)
        bp  = targets["bp"].to(self.device, non_blocking=True)

        targets_gpu = {
            "ppg": ppg,
            "ecg": ecg,
            "bp": bp,
        }

        if self.modality_sampling:
            modality_dice = random.randint(0, 1)
        else:
            modality_dice = None

        mask_ones = torch.ones_like(ppg)
        mask_forecast = masks["forecast"] if masks else {
            "ppg": mask_ones, "ecg": mask_ones, "bp": mask_ones
        }
        mask_impute = masks["imputed"] if masks else {
            "ppg": mask_ones, "ecg": mask_ones, "bp": mask_ones
        }

        total_loss = torch.zeros((), device=self.device)
        total_terms = 0
        losses_for_logging = {}

        def merge_losses(dst, src):
            for mod, mod_losses in src.items():
                if mod not in dst:
                    dst[mod] = {}
                for k, v in mod_losses.items():
                    dst[mod][k] = v.detach()

        def append_cpu_copy(outputs_chunk):
            for task, pred in outputs_chunk.items():
                if "2" in task:
                    signal_name = task.split("2")[-1]
                else:
                    signal_name = task.split("_")[0]

                if signal_name not in self.train_preds:
                    self.train_preds[signal_name] = {}
                if signal_name not in self.train_targets:
                    self.train_targets[signal_name] = {}

                if task not in self.train_preds[signal_name]:
                    self.train_preds[signal_name][task] = []
                if task not in self.train_targets[signal_name]:
                    self.train_targets[signal_name][task] = []

                self.train_preds[signal_name][task].append(pred.detach().cpu())
                self.train_targets[signal_name][task].append(targets_gpu[signal_name].detach().cpu())

        def backward_chunk(outputs_chunk, chunk_type, tgt_mean=0.0, tgt_std=1.0, masks_chunk=None, save_preds=True):
            nonlocal total_loss, total_terms, losses_for_logging

            chunk_loss, chunk_losses = self.compute_loss_chunk(
                outputs_chunk,
                targets_gpu,
                chunk_type=chunk_type,
                masks=masks_chunk,
                tgt_mean=tgt_mean,
                tgt_std=tgt_std
            )

            n_terms = sum(len(v) for v in chunk_losses.values())
            scaled_loss = chunk_loss * n_terms

            self.manual_backward(scaled_loss)

            total_loss = total_loss + scaled_loss.detach()
            total_terms += n_terms
            merge_losses(losses_for_logging, chunk_losses)

            if save_preds:
                append_cpu_copy(outputs_chunk)

        if modality_dice == 0 or not self.modality_sampling:
            sig_denoise = self.stack_signals(noised, mask_ones)
            ppg_den, ecg_den, bp_den = self.model(sig_denoise, model_flags=["0_0", "1_1", "2_2"])
            backward_chunk(
                {
                    "ppg_denoised": ppg_den,
                    "ecg_denoised": ecg_den,
                    "bp_denoised": bp_den,
                },
                chunk_type="denoised",
                tgt_mean=ecg_mean,
                tgt_std=ecg_std,
                masks_chunk=None,
                save_preds=True
            )
            del sig_denoise, ppg_den, ecg_den, bp_den

            sig_sr = self.stack_signals(masked_sr, mask_ones)
            ppg_sr, ecg_sr, bp_sr = self.model(sig_sr, model_flags=["0_0", "1_1", "2_2"])
            backward_chunk(
                {
                    "ppg_sr": ppg_sr,
                    "ecg_sr": ecg_sr,
                    "bp_sr": bp_sr,
                },
                chunk_type="sr",
                tgt_mean=ecg_mean,
                tgt_std=ecg_std,
                masks_chunk=None,
                save_preds=True
            )
            del sig_sr, ppg_sr, ecg_sr, bp_sr

            sig_impute = self.with_prior(self.stack_signals(masked, mask_impute), masks.get("prior_imputed") if masks else None)
            ppg_imp, ecg_imp, bp_imp = self.model(sig_impute, model_flags=["0_0", "1_1", "2_2"])
            backward_chunk(
                {
                    "ppg_imputed": ppg_imp,
                    "ecg_imputed": ecg_imp,
                    "bp_imputed": bp_imp,
                },
                chunk_type="imputed",
                tgt_mean=ecg_mean,
                tgt_std=ecg_std,
                masks_chunk={"imputed": mask_impute},
                save_preds=True
            )
            del sig_impute, ppg_imp, ecg_imp, bp_imp

            sig_forecast = self.with_prior(self.stack_signals(masked_forecast, mask_forecast), masks.get("prior_forecast") if masks else None)
            ppg_fc, ecg_fc, bp_fc = self.model(sig_forecast, model_flags=["0_0", "1_1", "2_2"])
            backward_chunk(
                {
                    "ppg_forecast": ppg_fc,
                    "ecg_forecast": ecg_fc,
                    "bp_forecast": bp_fc,
                },
                chunk_type="forecast",
                tgt_mean=ecg_mean,
                tgt_std=ecg_std,
                masks_chunk={"forecast": mask_forecast},
                save_preds=True
            )
            del sig_forecast, ppg_fc, ecg_fc, bp_fc

        if modality_dice == 1 or not self.modality_sampling:
            zeros = torch.zeros_like(ppg)
            ones = torch.ones_like(ppg)

            masks_cm = {"ppg": ones, "ecg": zeros, "bp": zeros}
            input_ppg = self.stack_signals(
                {"ppg": ppg, "ecg": zeros, "bp": zeros},
                masks_cm
            )
            _, ppg2ecg, ppg2bp = self.model(input_ppg, model_flags=["0_0", "0_1", "0_2"])
            backward_chunk(
                {
                    "ppg2ecg": ppg2ecg,
                    "ppg2bp": ppg2bp,
                },
                chunk_type="cross",
                tgt_mean=ecg_mean,
                tgt_std=ecg_std,
                masks_chunk=None,
                save_preds=True
            )
            del input_ppg, ppg2ecg, ppg2bp

            masks_cm = {"ppg": zeros, "ecg": ones, "bp": zeros}
            input_ecg = self.stack_signals(
                {"ppg": zeros, "ecg": ecg, "bp": zeros},
                masks_cm
            )
            ecg2ppg, _, ecg2bp = self.model(input_ecg, model_flags=["1_0", "1_1", "1_2"])
            backward_chunk(
                {
                    "ecg2ppg": ecg2ppg,
                    "ecg2bp": ecg2bp,
                },
                chunk_type="cross",
                tgt_mean=ecg_mean,
                tgt_std=ecg_std,
                masks_chunk=None,
                save_preds=True
            )
            del input_ecg, ecg2ppg, ecg2bp

            masks_cm = {"ppg": zeros, "ecg": zeros, "bp": ones}
            input_bp = self.stack_signals(
                {"ppg": zeros, "ecg": zeros, "bp": bp},
                masks_cm
            )
            bp2ppg, bp2ecg, _ = self.model(input_bp, model_flags=["2_0", "2_1", "2_2"])
            backward_chunk(
                {
                    "bp2ppg": bp2ppg,
                    "bp2ecg": bp2ecg,
                },
                chunk_type="cross",
                tgt_mean=ecg_mean,
                tgt_std=ecg_std,
                masks_chunk=None,
                save_preds=True
            )
            del input_bp, bp2ppg, bp2ecg

        opt.step()
        self._step_lr_scheduler()

        train_loss = total_loss / max(total_terms, 1)

        self.train_losses = self.update_loss_dict(self.train_losses, losses_for_logging)
        self.train_losses_list.append(train_loss.item())
        self.log("train_loss", train_loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=ppg.size(0))

        return train_loss

    def update_loss_dict(self, loss_dict, new_losses):
        for modality in ['ppg', 'ecg', 'bp']:
            if modality in new_losses:
                for task, value in new_losses[modality].items():
                    if task not in loss_dict[modality]:
                        #print("Task '{}' not found in loss_dict for modality '{}'. Initializing new list.".format(task, modality))
                        #print(loss_dict[modality].keys())
                        loss_dict[modality][task] = []
                    loss_dict[modality][task].append(value)
            #else:
            #    print(f"Warning: Modality '{modality}' not found in new_losses. Skipping update for this modality.")
        return loss_dict
        
    def compute_metrics(self, outputs, targets, metrics):
    
        modalities = targets.keys()  # Get modalities from targets
        for modality in modalities:
            if modality not in outputs or modality not in targets:
                continue
            if modality not in metrics:
                metrics[modality] = {}
            for task, pred_list in outputs[modality].items():
                true = torch.cat(targets[modality][task], dim=0)  # Concatenate all targets for this task
                pred = torch.cat(pred_list, dim=0)  # Concatenate all predictions for this task
                rmse = torch.sqrt(self.criterion(pred, true)).item()
                if task not in metrics[modality]:
                    metrics[modality][task] = []
                metrics[modality][task].append(rmse)  # Append the computed metric
        return metrics

    def plot_loss(self):
        for j, split in enumerate(['train', 'val']):
            for i, modality in enumerate(['ppg', 'ecg', 'bp']):
                losses_dict = self.train_losses_epoch if split == 'train' else self.val_losses_epoch
                if modality in losses_dict:
                    tasks = losses_dict[modality].keys()
                ax = self.ax_loss[i, j]
                ax.clear()
                for task in tasks:
                    losses = losses_dict[modality][task]               
                    ax.plot(losses, label=f'{split}-{task} loss')
                ax.set_title(f'{modality.upper()} {split} Loss')
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Loss')
                ax.legend()
        self.fig_loss.canvas.draw()
        self.fig_loss.canvas.flush_events()

    def plot_metrics(self):
        for i, modality in enumerate(['ppg', 'ecg', 'bp']):
            for j, split in enumerate(['train', 'val']):
                metrics_dict = self.train_metrics if split == 'train' else self.val_metrics
                ax = self.ax_metrics[i, j]
                ax.clear()

                if modality not in metrics_dict:
                    ax.set_title(f'{modality.upper()} {split} Metrics')
                    ax.axis('off')
                    continue

                for task, values in metrics_dict[modality].items():
                    ax.plot(values, label=f'{task}-rmse')

                ax.set_title(f'{modality.upper()} {split} Metrics')
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Value')
                ax.legend(fontsize=8)

        self.fig_metrics.canvas.draw()
        self.fig_metrics.canvas.flush_events()

    def on_train_end(self):
        #save loss and metric plots at 600 dpi, save also loss and metric dictionaries as .pt files
        if self.to_plot:
            self.fig_loss.savefig(f'imgs/{self.name}_loss_plot.png', dpi=600)
            self.fig_metrics.savefig(f'imgs/{self.name}_metrics_plot.png', dpi=600)
        torch.save(self.train_losses_epoch, f'imgs/{self.name}_train_losses_epoch.pt')
        torch.save(self.val_losses_epoch, f'imgs/{self.name}_val_losses_epoch.pt')
        torch.save(self.train_metrics, f'imgs/{self.name}_train_metrics_epoch.pt')
        torch.save(self.val_metrics, f'imgs/{self.name}_val_metrics_epoch.pt')

    def plot(self):
    
        self.plot_example()
        self.plot_loss()
        self.plot_metrics() 
    
    def plot_example(self):
        
        targets, inputs, outputs, masks = self.targets, self.inputs, self.outputs, self.masks

        if self.to_plot:

            fig, axes = self.fig_inout, self.ax_inout
            for ax in axes.flatten():
                ax.clear()  # Clear previous plots
                        
            # Create 3xN subplot grid (adjust N for all signals)
            hr_len = targets['ppg'].shape[-1]  # Assuming all have same length
            time = np.linspace(0, self.model.seconds, hr_len)
            
            ppg_target = targets['ppg'] if 'ppg' in targets else np.zeros_like(time)
            ecg_target = targets['ecg'] if 'ecg' in targets else np.zeros_like(time)
            bp_target = targets['bp'] if 'bp' in targets else np.zeros_like(time)

            if ppg_target.ndim == 3: 
                ppg_target = ppg_target[0,0,:].cpu().numpy()  # Extract first sample for plotting
            if ecg_target.ndim == 3:
                ecg_target = ecg_target[0,0,:].cpu().numpy()
            if bp_target.ndim == 3:
                bp_target = bp_target[0,0,:].cpu().numpy()

            # Row 1: Targets (repeat BP for alignment)
            axes[0,0].plot(time, ppg_target, 'k-', linewidth=2, label='PPG target')
            axes[0,0].legend(loc='upper right'); axes[0,0].set_title('PPG Target')
            axes[0,1].plot(time[:len(ecg_target)], ecg_target, 'r-', linewidth=2, label='ECG target')
            axes[0,1].legend(loc='upper right'); axes[0,1].set_title('ECG Target')
            axes[0,2].plot(time[:len(bp_target)], bp_target, 'b-', linewidth=2, label='BP target')
            axes[0,2].legend(loc='upper right'); axes[0,2].set_title('BP Target')

            # Row 2: Inputs (noised/masked/masked_sr/masked_forecast for each modality, repeat pattern)
            input_keys = list(inputs.keys())
            map_signals = {'ppg': 0, 'ecg': 1, 'bp': 2}  # Map signal type to column index
            for i, key in enumerate(input_keys): 
                #print(key)
                if "_" in key:
                   target_name = key.split('_')[0]  # Extract 'ppg', 'ecg', or 'bp' from key
                   task_type = key.split('_')[1]  # Extract 'noised', 'masked', etc.
                elif "2" in key:
                    target_name = key.split('2')[0]  # Source signal for cross-modal
                    task_type = "cross-modal"
                else:                    continue  # Skip if key format not recognized

                if task_type in ["cross-modal"]:
                    continue  # Skip cross-modal and forecast inputs for this row

                row = 1  # Inputs row
                col = map_signals[target_name]
                input = inputs[key]
                if input.ndim == 3:
                    input = input[0,0,:].cpu().numpy()  # Extract first sample for plotting                
                axes[row, col].plot(time[:len(input)], input, linewidth=1.5, label=task_type)
                #print(f"input: {key}, min: {input.min()}, max: {input.max()}")

            for target_name in map_signals.keys():
                col = map_signals[target_name]
                signal = targets[target_name][0,0,:].cpu().numpy()
                axes[1, col].plot(time[:len(input)], signal, 'k-', linewidth=1, label='target')  # Add target for reference
                axes[1, col].set_title(f"{target_name.upper()} - Inputs")
                handles, labels = axes[1, col].get_legend_handles_labels()
            axes[1, 1].legend(handles, labels, loc='upper center', ncols=len(handles))
                #print(f"target: {target_name}, min: {signal.min()}, max: {signal.max()}")
            
            # Row 3: Outputs (denoised/imputed/SR/forecast)
            output_keys = list(outputs.keys())
            for _, key in enumerate(output_keys):
                row = 2  # Outputs row
                if "_" in key:
                    signal_name = key.split('_')[0]  # 'ppg', 'ecg', or 'bp'
                    target_name = signal_name  # For non-cross-modal, target is same as signal
                    task_type = key.split('_')[1]  # 'denoised', 'imputed', 'sr', 'forecast', or cross-modal
                elif "2" in key:
                    signal_name = key.split('2')[0]  # Source signal for cross-modal
                    target_name = key.split('2')[1]  # Target signal for cross-modal
                    task_type = "cross-modal"
                else:                    continue  # Skip if key format not recognized
                
                if signal_name in map_signals:
                    col = map_signals[target_name]  # Use target signal for column mapping in cross-modal
                else:                    continue  # Skip if signal type not recognized
                
                if task_type in ["cross-modal"]:
                    continue  # Skip cross-modal and forecast outputs for this row
                elif task_type in ["forecast", "imputed"]:
                    #plot masked regions for reference
                    if task_type == "forecast": color = 'red' 
                    else: color = 'orange'
                    t = time[:len(input)]
                    mask = np.asarray(masks[task_type][signal_name]).ravel()  # Or ~ for missing
                    mask = np.logical_not(mask)  # Invert mask to highlight missing regions
                    min_y, max_y = axes[row, col].get_ylim()
                    axes[row, col].fill_between(t, min_y, max_y, where=mask, color=color, alpha=0.3, label=f'Mask {task_type}')

                output = outputs[key]
                if output.ndim == 3:
                    output = output[0,0,:].cpu().numpy()  # Extract first sample for plotting

                axes[row, col].plot(time[:len(output)], output, '-', linewidth=1.5, label=task_type)
                #print(f"output single-modal: {key}, min: {output.min()}, max: {output.max()}")

            for target_name in map_signals.keys():
                col = map_signals[target_name]
                axes[2, col].plot(time[:len(output)], targets[target_name][0,0,:].cpu().numpy(), 'k--', linewidth=1, label='target')  # Add target for reference
                axes[2, col].set_title(f"{target_name.upper()} - Outputs Single-Modal")
                handles, labels = axes[2, col].get_legend_handles_labels()
            axes[2, 1].legend(handles, labels, loc='upper center', ncols=len(handles))
            
            # Row 4: Outputs (Cross-modal)
            output_keys = list(outputs.keys())
            for _, key in enumerate(output_keys):
                row = 3  # Outputs row
                if "2" in key:
                    signal_name = key.split('2')[0]  # Source signal for cross-modal
                    target_name = key.split('2')[1]  # Target signal for cross-modal
                    task_type = "cross-modal"
                else:                    continue  # Skip if key format not recognized
                
                if signal_name in map_signals:
                    col = map_signals[target_name]  # Use target signal for column mapping in cross-modal
                else:                    continue  # Skip if signal type not recognized
                
                output = outputs[key]
                if output.ndim == 3:
                    output = output[0,0,:].cpu().numpy()  # Extract first sample for plotting

                axes[row, col].plot(time[:len(output)], output, '-', linewidth=1.5, label=key)
                #print(f"output cross-modal: {key}, min: {output.min()}, max: {output.max()}")

            for target_name in map_signals.keys():
                col = map_signals[target_name]
                axes[3, col].plot(time[:len(output)], targets[target_name][0,0,:].cpu().numpy(), 'k--', linewidth=1, label='target')  # Add target for reference
                axes[3, col].set_title(f"{target_name.upper()} - Outputs Cross-Modal")
                handles, labels = axes[3, col].get_legend_handles_labels()
                axes[3, col].legend(handles, labels, loc='upper center', ncols=len(handles))
            
            fig.suptitle(f'Validation Epoch {self.current_epoch}: Targets / Inputs / Outputs (Denoise/Impute/SR) / Outputs Cross-Modal)')
            plt.tight_layout()
            
            # Update the plot window
            fig.canvas.draw()     
            fig.canvas.flush_events()
            plt.pause(0.001)

        return super().on_validation_epoch_end()
        
    def safe_ecg_findpeaks_batch(self, ecg, sampling_rate=125, min_len=500, tgt_mean=0.0, tgt_std=1.0):
        """
        ecg: Tensor [B, T] or [B, 1, T]
        returns: list[np.ndarray], len == B, each element = 1D array of R-peak indices
        """
        # Normalize to [B, T]
        #print(f"ECG shape before processing: {ecg.shape}")
        if ecg.dim() == 3:
            if ecg.shape[1] == 1:
                ecg = ecg.squeeze(1)  # [B, T]
            else:
                # If multiple channels, you need to decide which to use; here we take first
                ecg = ecg[:, 0, :]    # [B, T]
        elif ecg.dim() == 2:
            pass  # already [B, T]
        else:
            raise ValueError(f"Unexpected ECG shape: {ecg.shape}, expected [B, T] or [B, 1, T]")

        B, T = ecg.shape
        device = ecg.device

        rpeaks_list = []

        for b in range(B):
            ecg_1d = ecg[b].cpu() * tgt_std[b].cpu() + tgt_mean[b].cpu()  # Denormalize if needed
            ecg_1d = ecg_1d.numpy()  # Convert to numpy for NeuroKit2

            #print(f"Processing ECG sample {b}: length={ecg_1d.shape}")
            if ecg_1d.shape[0] <= min_len:
                rpeaks_list.append(np.array([], dtype=int))
                continue
            try:
                peaks = nk.ecg_findpeaks(ecg_1d, sampling_rate=sampling_rate)['ECG_R_Peaks'].values
                # Ensure peaks are within [0, T)
                peaks = peaks[(peaks >= 0) & (peaks < T)]
                rpeaks_list.append(peaks)
            except Exception:
                rpeaks_list.append(np.array([], dtype=int))

        #print(f"Found R-peaks for batch: {[len(peaks) for peaks in rpeaks_list]}")
        return rpeaks_list

    def compute_loss(self, outputs, targets, tgt_mean=0.0, tgt_std=1.0, modality_dice=None, masks=None):

        criterion = self.criterion  # MSE for denoising/SR/translation
        criterion_impute = self.criterion_impute  # SoftDTW for imputation/forecasting
        
        losses = {}
        scales = {'ppg': 1.0, 'ecg': 1.0, 'bp': 1.0}  # Adjust if needed to balance modalities
        
        ecg_tgt = targets['ecg'] if 'ecg' in targets else None
        rpeaks = self.safe_ecg_findpeaks_batch(ecg_tgt, sampling_rate=125, tgt_mean=tgt_mean, tgt_std=tgt_std)

        # 1. Denoising/SR/Imputation/Forecasting losses (single modality or all modalities if no sampling)
        if modality_dice == 0 or modality_dice is None or not self.modality_sampling:
            #print("Computing losses for single modality tasks...")
            for task in ['denoised', 'sr', 'imputed', 'forecast']:
                for mod in ['ppg', 'ecg', 'bp']:
                    if mod not in losses:
                        losses[mod] = {}
    
                    pred = outputs[f'{mod}_{task}']          # (B, 1, L)
                    tgt  = targets[f'{mod}']          # (B, 1, L) 
                    if task in ['imputed', 'forecast'] and masks is not None:
                        mask = masks[task][mod] if masks else torch.ones_like(pred)  # (B, 1, L), 1=observed, 0=missing
                        assert mask.shape == pred.shape, f"{mod}_{task}: mask shape mismatch"

                        # missing positions: where mask == 0
                        missing_mask = (mask == 0)  # bool tensor: True where signal is missing
                        assert missing_mask.sum() > 0, f"{mod}_{task}: empty mask, no supervision signal"

                        # element-wise MSE — criterion_impute MUST be nn.MSELoss(reduction='none')
                        if isinstance(criterion_impute, nn.MSELoss):
                            element_loss = criterion_impute(pred, tgt)  # (B, 1, L)
                        elif isinstance(criterion_impute, QRSAwareMSELoss):
                            element_loss = criterion_impute(pred, tgt, rpeaks)  # (B, 1, L)
                        # compute loss ONLY on missing positions
                        loss = (element_loss * missing_mask).sum() / (missing_mask.sum() + 1e-8) 

                    else:
                        loss = criterion(pred, tgt).mean() # Global mean

                    loss = loss * scales[mod] 
                    losses[mod][mod+"_"+task] = loss

        if modality_dice == 1 or modality_dice is None or not self.modality_sampling:
            #print("Computing losses for cross-modal tasks...")
            # 3. Cross-modal: Supervised reconstruction (cycle‑consistency style) with MSE
            cross_pairs = [
                    ('ppg2ecg', 'ecg', 1.0), ('ppg2bp', 'bp', 1.0),
                    ('ecg2ppg', 'ppg', 1.0), ('ecg2bp', 'bp', 1.0),
                    ('bp2ppg', 'ppg', 1.0), ('bp2ecg', 'ecg', 1.0)
            ]
            for pred_key, tgt_mod, weight in cross_pairs:
                if tgt_mod not in losses:
                    losses[tgt_mod] = {}
                losses[tgt_mod][pred_key] = criterion(outputs[pred_key], targets[tgt_mod]).mean() * weight   
        
        # Multi-task average
        losses_flat = [loss for mod_losses in losses.values() for loss in mod_losses.values()]
        loss_list = torch.stack(losses_flat)
        total_loss = loss_list.mean()
        
        return total_loss, losses  # Return both total loss and dict of individual losses for logging/analysis
    
    def with_prior(self, x, prior):
        """Template-prior models: append [ppg x K, ecg x K, bp x K] after the 6 signal/mask channels
        (same convention as src.metrics.evaluate_all_tasks). Unchanged for models without prior input;
        a prior model zero-fills missing priors itself."""
        K = getattr(self.model, "prior_channels", 0)
        if K == 0 or prior is None:
            return x
        if x.shape[1] == 4:  # shared mask -> per-modality masks so the prior lands at channels 6:
            x = torch.cat([x[:, :3], x[:, 3:4].expand(-1, 3, *x.shape[2:])], dim=1)
        pri = torch.cat([prior[m].to(x.device, non_blocking=True).float().reshape(x.shape[0], K, -1)
                         for m in ("ppg", "ecg", "bp")], dim=1)
        if x.ndim == 4:
            pri = pri.unsqueeze(2)
        return torch.cat([x, pri.to(x.dtype)], dim=1)

    def stack_signals(self, signals_dict, mask):
        if isinstance(mask, dict):
            return torch.stack([signals_dict['ppg'], signals_dict['ecg'], signals_dict['bp'], mask['ppg'], mask['ecg'], mask['bp']], dim=1)
        else:
            return torch.stack([signals_dict['ppg'], signals_dict['ecg'], signals_dict['bp'], mask], dim=1)
        
    def move_to_cpu(self, tensor_dict):
        moved = {}
        for k, v in tensor_dict.items():
            moved[k] = {}
            if isinstance(v, dict):
                for sub_k, sub_v in v.items():
                    moved[k][sub_k] = sub_v.cpu()
            else:                moved[k] = v.cpu()
        return moved
    
    @torch.no_grad()
    def validation_step(self, batch, batch_idx):

        """
        Here, we do only One-Input, One-Output validation with model_flags: 0, 1, 2. No DiBiMa-M or BiMa-M.
        """
        targets, noised, masked, masked_sr, masked_forecast, masks, tgt_means, tgt_stds = batch
        ppg, ecg, bp = [t.to(self.device) for t in [targets['ppg'], targets['ecg'], targets['bp']]]
        
        mask_ones = torch.ones_like(ppg)  # Dummy mask of ones for denoising/SR inputs
        mask_forecast = masks['forecast'] if masks else {"ppg": mask_ones, "ecg": mask_ones, "bp": mask_ones}  # (B, 1, L) or dict of masks
        mask_impute = masks['imputed'] if masks else {"ppg": mask_ones, "ecg": mask_ones, "bp": mask_ones}  # (B, 1, L) or dict of masks

        ecg_mean = tgt_means['ecg'] if tgt_means else 0.0
        ecg_std = tgt_stds['ecg'] if tgt_stds else 1.0

        self.model.eval()
        
        # All tasks
        input = self.stack_signals(noised, mask_ones)  # Add dummy mask of ones for denoising
        ppg_den, ecg_den, bp_den = self.model(input, model_flags=["0_0", "1_1", "2_2"])  # Denoising
        self.move_to_cpu({"input": input, "ppg_den": ppg_den, "ecg_den": ecg_den, "bp_den": bp_den})

        input = self.with_prior(self.stack_signals(masked, mask_impute), masks.get("prior_imputed") if masks else None)
        ppg_imp, ecg_imp, bp_imp = self.model(input, model_flags=["0_0", "1_1", "2_2"])  # Imputation
        self.move_to_cpu({"input": input, "ppg_imp": ppg_imp, "ecg_imp": ecg_imp, "bp_imp": bp_imp})

        # Cross-modal (zeros conditioning)
        zeros = torch.zeros_like(ppg)
        ones = torch.ones_like(ppg)
        
        masks_cm = {"ppg": ones, "ecg": zeros, "bp": zeros}
        _, ppg2ecg, ppg2bp = self.model(self.stack_signals({"ppg": ppg, "ecg": zeros, "bp": zeros}, masks_cm), model_flags=["0_0", "0_1", "0_2"])
        self.move_to_cpu({"input": input, "ppg2ecg": ppg2ecg, "ppg2bp": ppg2bp})

        masks_cm = {"ppg": zeros, "ecg": ones, "bp": zeros}
        ecg2ppg, _, ecg2bp = self.model(self.stack_signals({"ppg": zeros, "ecg": ecg, "bp": zeros}, masks_cm), model_flags=["1_0", "1_1", "1_2"])
        self.move_to_cpu({"input": input, "ecg2ppg": ecg2ppg, "ecg2bp": ecg2bp})

        masks_cm = {"ppg": zeros, "ecg": zeros, "bp": ones}
        bp2ppg, bp2ecg, _ = self.model(self.stack_signals({"ppg": zeros, "ecg": zeros, "bp": bp}, masks_cm), model_flags=["2_0", "2_1", "2_2"])
        self.move_to_cpu({"input": input, "bp2ppg": bp2ppg, "bp2ecg": bp2ecg})

        input = self.stack_signals(masked_sr, mask_ones)  # Add dummy mask of ones for super-resolution
        ppg_sr_out, ecg_sr_out, bp_sr_out = self.model(input, model_flags=["0_0", "1_1", "2_2"])  # SR
        self.move_to_cpu({"input": input, "ppg_sr": ppg_sr_out, "ecg_sr": ecg_sr_out, "bp_sr": bp_sr_out})

        input = self.with_prior(self.stack_signals(masked_forecast, mask_forecast), masks.get("prior_forecast") if masks else None)
        ppg_fc_out, ecg_fc_out, bp_fc_out = self.model(input, model_flags=["0_0", "1_1", "2_2"])  # Forecast
        self.move_to_cpu({"input": input, "ppg_fc": ppg_fc_out, "ecg_fc": ecg_fc_out, "bp_fc": bp_fc_out})

        # Dict for compute_loss (matches training_step)
        outputs = {
            'ppg_denoised': ppg_den, 'ppg_imputed': ppg_imp, 'ppg_sr': ppg_sr_out, 'ppg_forecast': ppg_fc_out,
            'ecg_denoised': ecg_den, 'ecg_imputed': ecg_imp, 'ecg_sr': ecg_sr_out, 'ecg_forecast': ecg_fc_out,
            'bp_denoised': bp_den, 'bp_imputed': bp_imp, 'bp_sr': bp_sr_out, 'bp_forecast': bp_fc_out,
            'ppg2ecg': ppg2ecg, 'ppg2bp': ppg2bp, 'ecg2ppg': ecg2ppg, 'ecg2bp': ecg2bp, 'bp2ppg': bp2ppg, 'bp2ecg': bp2ecg
        }

        if batch_idx == 0:
            self.targets = {k: v[0:1].cpu() for k, v in targets.items()}  # First sample
            self.outputs = {k: v[0:1].cpu() for k, v in outputs.items()}
            self.inputs = {
                'ppg_noised': noised['ppg'][0:1].cpu(),
                'ppg_masked': masked['ppg'][0:1].cpu(),
                'ppg_sr_input': masked_sr['ppg'][0:1].cpu(),
                "ppg_forecast_input": masked_forecast['ppg'][0:1].cpu(),
                'ecg_noised': noised['ecg'][0:1].cpu(),
                'ecg_masked': masked['ecg'][0:1].cpu(), 
                'ecg_sr_input': masked_sr['ecg'][0:1].cpu(),
                "ecg_forecast_input": masked_forecast['ecg'][0:1].cpu(),
                'bp_noised': noised['bp'][0:1].cpu(),
                'bp_masked': masked['bp'][0:1].cpu(),
                'bp_sr_input': masked_sr['bp'][0:1].cpu(),
                "bp_forecast_input": masked_forecast['bp'][0:1].cpu()
            }  
            self.masks = {
                "imputed": {k: v[0:1].cpu() for k, v in masks['imputed'].items()},
                "forecast": {k: v[0:1].cpu() for k, v in masks['forecast'].items()}
            }  

        val_loss, losses = self.compute_loss(outputs, targets, modality_dice=None, masks=masks, tgt_mean=ecg_mean, tgt_std=ecg_std)
        self.val_losses = self.update_loss_dict(self.val_losses, losses)
        for task in outputs.keys():
            if '2' in task:
                signal_name = task.split('2')[-1]  # Extract target modality from keys like 'ppg2ecg'
            else:
                signal_name = task.split('_')[0]  # Extract 'ppg', 'ecg', or 'bp' from the key

            if signal_name not in self.val_preds:
                self.val_preds[signal_name] = {}
            if signal_name not in self.val_targets:
                self.val_targets[signal_name] = {}

            if task not in self.val_preds[signal_name]:
                self.val_preds[signal_name][task] = []
            if task not in self.val_targets[signal_name]:
                self.val_targets[signal_name][task] = []

            # Append prediction for this task (once per task)
            self.val_preds[signal_name][task].append(outputs[task].detach().cpu())

            # Append target for this task (once per task)
            self.val_targets[signal_name][task].append(targets[signal_name].detach().cpu())

        self.val_losses_list.append(val_loss.item())
        self.log('val_loss', val_loss)
        return val_loss

    def on_train_epoch_start(self):
        
        self.model.train()
        self.train()
        self.train_losses_list = []
        return super().on_train_epoch_start()

    def on_validation_epoch_start(self):
        
        self.model.eval()
        self.eval()
        self.val_losses_list = []
        self.inputs, self.targets, self.outputs = None, None, None

        return super().on_validation_epoch_start()
        
    def on_train_epoch_end(self):

        self.log("avg_train_loss", np.mean(self.train_losses_list), prog_bar=True, on_epoch=True)
        sch = self.lr_schedulers()
        if sch:
            self.log("lr", sch.get_last_lr()[0], prog_bar=True)

        return super().on_train_epoch_end()

    def on_validation_epoch_end(self):
        
        mean_val_loss = np.mean(self.val_losses_list) if self.val_losses_list else 0.0
        self.log("avg_val_loss", mean_val_loss, prog_bar=True, on_epoch=True)
        with torch.no_grad():
           self.val_metrics = self.compute_metrics(self.val_preds, self.val_targets, self.val_metrics)
        
        for modality in ['ppg', 'ecg', 'bp']: #mean loss per modality for plotting
            if modality in self.val_losses:
                for task in self.val_losses[modality].keys():
                    if modality not in self.val_losses_epoch:
                        self.val_losses_epoch[modality] = {}
                    if task not in self.val_losses_epoch[modality]:
                        self.val_losses_epoch[modality][task] = []
                    losses = self.val_losses[modality][task]
                    mean_loss = torch.stack(losses).mean()
                    #print(f"Validation epoch {self.current_epoch}, modality {modality}, task {task}, mean loss: {mean_loss}")
                    self.val_losses_epoch[modality][task].append(mean_loss.item())
        
        with torch.no_grad():
            self.train_metrics = self.compute_metrics(self.train_preds, self.train_targets, self.train_metrics)

        for modality in ['ppg', 'ecg', 'bp']: #mean loss per modality for plotting
            if modality in self.train_losses:
                for task in self.train_losses[modality].keys():
                    if modality not in self.train_losses_epoch:
                        self.train_losses_epoch[modality] = {}
                    if task not in self.train_losses_epoch[modality]:
                        self.train_losses_epoch[modality][task] = []
                    losses = self.train_losses[modality][task]
                    mean_loss = torch.stack(losses).mean()
                    #print(f"Train epoch {self.current_epoch}, modality {modality}, task {task}, mean loss: {mean_loss}")
                    self.train_losses_epoch[modality][task].append(mean_loss.item())

        if self.to_plot:
            self.plot()

        self.train_losses_list.clear()            
        self.val_losses_list.clear()

        for m in self.val_preds:
            tasks = self.val_preds[m].keys()
            for task in tasks:
                self.val_preds[m][task] = []
                self.val_targets[m][task] = []
    
        for m in self.train_preds:
            tasks = self.train_preds[m].keys()
            for task in tasks:
                self.train_preds[m][task] = []
                self.train_targets[m][task] = []

        return super().on_validation_epoch_end()
        

    def configure_optimizers(self):
        
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=1e-6
        )

        # Linear per-step decay learning_rate -> final_learning_rate over total_steps (then held).
        # automatic_optimization=False, so Lightning does NOT step it: _step_lr_scheduler() is called
        # after every opt.step() in training_step. (The former ReduceLROnPlateau was never stepped:
        # the LR stayed constant at learning_rate in all earlier runs.)
        final_lr = getattr(self, "final_learning_rate", None) or self.learning_rate
        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=final_lr / self.learning_rate,
            total_iters=max(int(self.total_steps), 1),
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                        'scheduler': lr_scheduler,
                        'interval': 'step',
            }
        }

    def _step_lr_scheduler(self):
        sch = self.lr_schedulers()
        if sch is not None:
            sch.step()

class DiBiMa(BiMa):
    def __init__(
        self, 
        model,
        train_scheduler,
        total_steps,
        learning_rate=1e-4,
        predict_type="epsilon",  # "epsilon" or "sample" (x0)
        criterion=nn.MSELoss(),
        criterion_impute=nn.MSELoss(reduction='none'),
        modality_sampling=False,
        debug=False,
        epochs=10,
        use_diffusion=True,
        norm_layers = ["layer", "batch"],
        plot=False,
        name="dibima",
        cross=False
    ):
        super().__init__(model=model, total_steps=total_steps, plot=plot, name=name, cross=cross)  # Call parent constructor for plotting setup
        self.save_hyperparameters(ignore=['model'])
        
        # Core model
        self.model = model
        self.name = name
        self.learning_rate = learning_rate
        self.predict_type = predict_type
        self.debug = debug
        self.epochs = epochs
        self.use_diffusion = use_diffusion
        self.train_scheduler = train_scheduler
        self.modality_sampling = modality_sampling
        self.criterion = criterion
        self.criterion_impute = criterion_impute
        self.norm_layers = norm_layers

        if use_diffusion and train_scheduler is not None:
            self.T = train_scheduler.num_train_timesteps
            self.alphas_cumprod = torch.tensor(train_scheduler.alphas_cumprod).cpu()  # Precompute for efficiency
            self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod).cpu()
            self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - self.alphas_cumprod).cpu()
        else:
            self.T = None

    def forward(self, *args, **kwargs):
        def safe_to_dtype(x):
            if torch.is_tensor(x):
                # Sync with model dtype (handles AMP fp16)
                return x.to(dtype=self.dtype, device=self.device)
            return x
        
        args = tuple(safe_to_dtype(arg) for arg in args)
        kwargs = {k: safe_to_dtype(v) for k, v in kwargs.items()}
        
        if self.debug:
            shapes = [str(a.shape) if torch.is_tensor(a) else str(type(a)) for a in args]
            print(f"Forward shapes: {shapes}, kwargs: {list(kwargs.keys())}")
        
        # Route to correct forward
        if 't' in kwargs and kwargs['t'] is not None:
            return self.model.forward_diffusion(*args, debug=self.debug, **kwargs)
        else:
            return self.model.forward_regression(*args, debug=self.debug, **kwargs)

    def q_sample(self, x0, t, noise=None):
        """FIXED: Robust broadcasting for (B,C,L)"""
        
        #print(f"q_sample called with x0 shape: {x0.shape}, t shape: {t.shape}")
        x0 = x0.cpu()  # Ensure on CPU for indexing
        B, C, L = x0.shape
        if noise is None:
            noise = torch.randn_like(x0).to(x0.device)  # (B,C,L)
        
        # Pre-expand alphas for broadcasting
        alpha_bar = self.alphas_cumprod[t.cpu()]  # (B,) values in [0,1]
        sqrt_alpha_bar = torch.sqrt(alpha_bar).view(B, 1, 1)  # (B,1,1)
        sqrt_one_minus_alpha_bar = torch.sqrt(1 - alpha_bar).view(B, 1, 1)  # (B,1,1)
        
        # Explicit expansion to match x0.shape
        sqrt_alpha_bar = sqrt_alpha_bar.expand(B, C, L)
        sqrt_one_minus_alpha_bar = sqrt_one_minus_alpha_bar.expand(B, C, L)
        
        x_t = sqrt_alpha_bar * x0 + sqrt_one_minus_alpha_bar * noise
        return x_t, noise

    def one_step_sample_from_noise(self, batch_size, length, device=None):
        """ONE_STEP sampling from max noise (t=T-1) -> x_0"""
        if device is None:
            device = self.device
        t = torch.full((batch_size,), self.T-1, dtype=torch.long, device=device)
        x_T = torch.randn(batch_size, 3, length, device=device)
        pred = self.forward(x_T, t=t)
        if self.predict_type == "epsilon":
            eps_theta = pred
            alpha_bar = self.alphas_cumprod[self.T-1]
            x0_pred = (x_T - torch.sqrt(1-alpha_bar) * eps_theta) / torch.sqrt(alpha_bar)
        else:
            x0_pred = pred
        return x0_pred

    def _prior_tensor(self, prior, B, L):
        """(B, 3K, L) prior block [ppg x K, ecg x K, bp x K]; zeros if prior is None. None if the model has no prior input."""
        K = getattr(self.model, "prior_channels", 0)
        if K == 0:
            return None
        if prior is None:
            return torch.zeros(B, 3 * K, L, device=self.device)
        parts = []
        for mod in ("ppg", "ecg", "bp"):
            p = prior[mod].to(self.device, non_blocking=True).float()
            if p.ndim == 2:
                p = p.unsqueeze(1)
            parts.append(p)
        return torch.cat(parts, dim=1)

    def stack_and_noise(self, signals_dict, mask, t, prior=None):
        signals = torch.cat(
            [
                signals_dict["ppg"].to(self.device, non_blocking=True),
                signals_dict["ecg"].to(self.device, non_blocking=True),
                signals_dict["bp"].to(self.device, non_blocking=True),
            ],
            dim=1,
        )

        if signals.ndim == 4:
            signals = signals.squeeze(2)

        x_t, _ = self.q_sample(signals, t)

        if isinstance(mask, dict):
            masks = torch.cat(
                [
                    mask["ppg"].to(self.device, non_blocking=True),
                    mask["ecg"].to(self.device, non_blocking=True),
                    mask["bp"].to(self.device, non_blocking=True),
                ],
                dim=1,
            )

            if masks.ndim == 4:
                masks = masks.squeeze(2)
        else:
            masks = mask.to(self.device, non_blocking=True)
            if masks.ndim == 4:
                masks = masks.squeeze(2)

        out = torch.cat([x_t.to(self.device, non_blocking=True), masks], dim=1)
        pri = self._prior_tensor(prior, out.shape[0], out.shape[-1])
        if pri is not None:
            if out.shape[1] == 4:  # shared mask -> per-modality masks so the prior block lands at channels 6:
                out = torch.cat([out[:, :3], out[:, 3:4].expand(-1, 3, -1)], dim=1)
            out = torch.cat([out, pri.to(out.dtype)], dim=1)
        return out

    def move_mask_dict(self, mask_dict, device):
        if isinstance(mask_dict, dict):
            return {
                key: value.to(device, non_blocking=True)
                for key, value in mask_dict.items()
            }
        return mask_dict.to(device, non_blocking=True)

    
    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        opt.zero_grad()

        targets, noised, masked, masked_sr, masked_forecast, masks, tgt_means, tgt_stds = batch
        ecg_mean = tgt_means['ecg'] if tgt_means else 0.0
        ecg_std = tgt_stds['ecg'] if tgt_stds else 1.0

        if self.modality_sampling:
            modality_dice = random.randint(0, 1)  # 0=single-modality, 1=cross-modal
        else:
            modality_dice = None

        ppg = targets["ppg"].to(self.device, non_blocking=True)
        ecg = targets["ecg"].to(self.device, non_blocking=True)
        bp = targets["bp"].to(self.device, non_blocking=True)

        targets_gpu = {"ppg": ppg, "ecg": ecg, "bp": bp}
    
        mask_ones = torch.ones_like(ppg)
        if masks:
            mask_forecast = self.move_mask_dict(masks["forecast"], self.device)
            mask_impute = self.move_mask_dict(masks["imputed"], self.device)
        else:
            mask_forecast = {
                "ppg": torch.ones_like(ppg),
                "ecg": torch.ones_like(ecg),
                "bp": torch.ones_like(bp),
            }
            mask_impute = {
                "ppg": torch.ones_like(ppg),
                "ecg": torch.ones_like(ecg),
                "bp": torch.ones_like(bp),
            }

        B = ppg.size(0)
        t = torch.randint(
            low=0,
            high=self.T,
            size=(B,),
            device=self.device,
            dtype=torch.long,
        )
        
        total_loss = torch.zeros((), device=self.device)
        total_terms = 0
        losses_for_logging = {}

        def merge_losses(dst, src):
            for mod, mod_losses in src.items():
                if mod not in dst:
                    dst[mod] = {}
                for k, v in mod_losses.items():
                    dst[mod][k] = v.detach()
        
        def append_cpu_copy(outputs_chunk):
            for task, pred in outputs_chunk.items():
                if "2" in task:
                    signal_name = task.split("2")[-1]
                else:
                    signal_name = task.split("_")[0]

                if signal_name not in self.train_preds:
                    self.train_preds[signal_name] = {}
                if signal_name not in self.train_targets:
                    self.train_targets[signal_name] = {}

                if task not in self.train_preds[signal_name]:
                    self.train_preds[signal_name][task] = []
                if task not in self.train_targets[signal_name]:
                    self.train_targets[signal_name][task] = []

                self.train_preds[signal_name][task].append(pred.detach().cpu())
                self.train_targets[signal_name][task].append(targets_gpu[signal_name].detach().cpu())

        def backward_chunk(outputs_chunk, chunk_type, tgt_mean=0.0, tgt_std=1.0, masks_chunk=None, save_preds=True):
            nonlocal total_loss, total_terms, losses_for_logging

            _, chunk_losses = self.compute_loss_chunk(
                outputs_chunk,
                targets_gpu,
                chunk_type=chunk_type,
                masks=masks_chunk,
                tgt_mean=tgt_mean,
                tgt_std=tgt_std
            )

            losses = [
                loss
                for modality_losses in chunk_losses.values()
                for loss in modality_losses.values()
            ]

            if not losses:
                raise RuntimeError(
                    f"No losses computed for chunk_type={chunk_type}"
                )

            total_loss = total_loss + torch.stack(losses).sum()
            total_terms += len(losses)

            merge_losses(losses_for_logging, chunk_losses)

            if save_preds:
                append_cpu_copy(outputs_chunk)

        if modality_dice == 0 or not self.modality_sampling:
            x_t_denoise = self.stack_and_noise(noised, mask_ones, t)
            ppg_den, ecg_den, bp_den = self.forward(
                x_t_denoise, t=t, model_flags=["0_0", "1_1", "2_2"]
            )
            backward_chunk(
                {
                    "ppg_denoised": ppg_den,
                    "ecg_denoised": ecg_den,
                    "bp_denoised": bp_den,
                },
                chunk_type="denoised",
                tgt_mean=ecg_mean,
                tgt_std=ecg_std,
                masks_chunk=None
            )

            del x_t_denoise, ppg_den, ecg_den, bp_den

            x_t_sr = self.stack_and_noise(masked_sr, mask_ones, t)
            ppg_sr, ecg_sr, bp_sr = self.forward(
                x_t_sr, t=t, model_flags=["0_0", "1_1", "2_2"]
            )
            backward_chunk(
                {
                    "ppg_sr": ppg_sr,
                    "ecg_sr": ecg_sr,
                    "bp_sr": bp_sr,
                },
                chunk_type="sr",
                tgt_mean=ecg_mean,
                tgt_std=ecg_std,
                masks_chunk=None
            )
            del x_t_sr, ppg_sr, ecg_sr, bp_sr

            x_t_impute = self.stack_and_noise(masked, mask_impute, t, prior=masks.get("prior_imputed") if masks else None)
            ppg_imp, ecg_imp, bp_imp = self.forward(
                x_t_impute, t=t, model_flags=["0_0", "1_1", "2_2"]
            )
            backward_chunk(
                {
                    "ppg_imputed": ppg_imp,
                    "ecg_imputed": ecg_imp,
                    "bp_imputed": bp_imp,
                },
                chunk_type="imputed",
                tgt_mean=ecg_mean,
                tgt_std=ecg_std,
                masks_chunk={"imputed": mask_impute}
            )
            del x_t_impute, ppg_imp, ecg_imp, bp_imp

            x_t_forecast = self.stack_and_noise(masked_forecast, mask_forecast, t, prior=masks.get("prior_forecast") if masks else None)
            ppg_fc, ecg_fc, bp_fc = self.forward(
                x_t_forecast, t=t, model_flags=["0_0", "1_1", "2_2"]
            )
            backward_chunk(
                {
                    "ppg_forecast": ppg_fc,
                    "ecg_forecast": ecg_fc,
                    "bp_forecast": bp_fc,
                },
                chunk_type="forecast",
                tgt_mean=ecg_mean,
                tgt_std=ecg_std,
                masks_chunk={"forecast": mask_forecast}
            )
            del x_t_forecast, ppg_fc, ecg_fc, bp_fc

        if modality_dice == 1 or not self.modality_sampling:
            zeros = torch.zeros_like(ppg)
            ones = torch.ones_like(ppg)

            masks_cm = {"ppg": ones, "ecg": zeros, "bp": zeros}
            x_t_ppg_cross = self.stack_and_noise(
                {"ppg": ppg, "ecg": zeros, "bp": zeros}, masks_cm, t=t
            )
            _, ppg2ecg, ppg2bp = self.forward(
                x_t_ppg_cross, model_flags=["0_0", "0_1", "0_2"], t=t
            )
            backward_chunk(
                {
                    "ppg2ecg": ppg2ecg,
                    "ppg2bp": ppg2bp,
                },
                chunk_type="cross",
                tgt_mean=ecg_mean,
                tgt_std=ecg_std,
                masks_chunk=None
            )
            del x_t_ppg_cross, ppg2ecg, ppg2bp

            masks_cm = {"ppg": zeros, "ecg": ones, "bp": zeros}
            x_t_ecg_cross = self.stack_and_noise(
                {"ppg": zeros, "ecg": ecg, "bp": zeros}, masks_cm, t=t
            )
            ecg2ppg, _, ecg2bp = self.forward(
                x_t_ecg_cross, model_flags=["1_0", "1_1", "1_2"], t=t
            )
            backward_chunk(
                {
                    "ecg2ppg": ecg2ppg,
                    "ecg2bp": ecg2bp,
                },
                chunk_type="cross",
                tgt_mean=ecg_mean,
                tgt_std=ecg_std,
                masks_chunk=None
            )
            del x_t_ecg_cross, ecg2ppg, ecg2bp

            masks_cm = {"ppg": zeros, "ecg": zeros, "bp": ones}
            x_t_bp_cross = self.stack_and_noise(
                {"ppg": zeros, "ecg": zeros, "bp": bp}, masks_cm, t=t
            )
            bp2ppg, bp2ecg, _ = self.forward(
                x_t_bp_cross, model_flags=["2_0", "2_1", "2_2"], t=t
            )
            backward_chunk(
                {
                    "bp2ppg": bp2ppg,
                    "bp2ecg": bp2ecg,
                },
                chunk_type="cross",
                tgt_mean=ecg_mean,
                tgt_std=ecg_std,
                masks_chunk=None
            )
            del x_t_bp_cross, bp2ppg, bp2ecg

        if total_terms == 0:
            raise RuntimeError("No training losses were accumulated")

        final_loss = total_loss / total_terms

        self.manual_backward(final_loss)
        opt.step()
        self._step_lr_scheduler()

        self.train_losses = self.update_loss_dict(
            self.train_losses,
            losses_for_logging,
        )

        self.train_losses_list.append(final_loss.detach().item())

        self.log(
            "train_loss",
            final_loss.detach(),
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            batch_size=B,
        )

        return final_loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        targets, noised, masked, masked_sr, masked_forecast, masks, tgt_means, tgt_stds = batch
        ecg_mean = tgt_means['ecg'] if tgt_means else 0.0
        ecg_std = tgt_stds['ecg'] if tgt_stds else 1

        # Extract to variables
        ppg, ecg, bp = [t.to(self.device) for t in [targets["ppg"], targets["ecg"], targets["bp"]]]
        
        mask_ones = torch.ones_like(ppg)  # Dummy mask of ones for denoising/SR inputs
        if masks:
            mask_forecast = self.move_mask_dict(masks["forecast"], self.device)
            mask_impute = self.move_mask_dict(masks["imputed"], self.device)
        else:
            mask_forecast = {"ppg": mask_ones, "ecg": mask_ones, "bp": mask_ones}
            mask_impute = {"ppg": mask_ones, "ecg": mask_ones, "bp": mask_ones}

        self.model.eval()
        
        # All tasks
        t = torch.full((ppg.size(0),), self.T-1, dtype=torch.long, device=self.device)  # Max noise for visualization

        input_den = self.stack_and_noise(noised, mask_ones, t)
        ppg_den, ecg_den, bp_den = self.forward(input_den, model_flags=["0_0", "1_1", "2_2"], t=t)  # Denoising
        self.move_to_cpu({"input": input_den, "ppg_den": ppg_den, "ecg_den": ecg_den, "bp_den": bp_den})

        input_impute = self.stack_and_noise(masked, mask_impute, t, prior=masks.get("prior_imputed") if masks else None)
        ppg_imp, ecg_imp, bp_imp = self.forward(input_impute, model_flags=["0_0", "1_1", "2_2"], t=t)  # Imputation
        self.move_to_cpu({"input": input_impute, "ppg_imp": ppg_imp, "ecg_imp": ecg_imp, "bp_imp": bp_imp})

        # Cross-modal (zeros conditioning)
        zeros = torch.zeros_like(ppg)
        ones = torch.ones_like(ppg)
        
        masks_cm = {"ppg": ones, "ecg": zeros, "bp": zeros}
        input = self.stack_and_noise({"ppg": ppg, "ecg": zeros, "bp": zeros}, masks_cm, t)
        _, ppg2ecg, ppg2bp = self.forward(input, t=t, model_flags=["0_0", "0_1", "0_2"])
        self.move_to_cpu({"input": input, "masks": masks_cm, "ppg2ecg": ppg2ecg, "ppg2bp": ppg2bp})

        masks_cm = {"ppg": zeros, "ecg": ones, "bp": zeros}
        ecg2ppg, _, ecg2bp = self.forward(self.stack_and_noise({"ppg": zeros, "ecg": ecg, "bp": zeros}, masks_cm, t), t=t, model_flags=["1_0", "1_1", "1_2"])
        masks_cm = {"ppg": zeros, "ecg": zeros, "bp": ones}
        bp2ppg, bp2ecg, _ = self.forward(self.stack_and_noise({"ppg": zeros, "ecg": zeros, "bp": bp}, masks_cm, t), t=t, model_flags=["2_0", "2_1", "2_2"])
        self.move_to_cpu({"masks": masks_cm, "ecg2ppg": ecg2ppg, "ecg2bp": ecg2bp, "bp2ppg": bp2ppg, "bp2ecg": bp2ecg})

        input = self.stack_and_noise(masked_sr, mask_ones, t)
        ppg_sr_out, ecg_sr_out, bp_sr_out = self.forward(input, model_flags=["0_0", "1_1", "2_2"], t=t)  # SR
        self.move_to_cpu({"input": input, "ppg_sr": ppg_sr_out, "ecg_sr": ecg_sr_out, "bp_sr": bp_sr_out})

        input = self.stack_and_noise(masked_forecast, mask_forecast, t, prior=masks.get("prior_forecast") if masks else None)
        ppg_fc_out, ecg_fc_out, bp_fc_out = self.forward(input, model_flags=["0_0", "1_1", "2_2"], t=t)  # Forecast
        self.move_to_cpu({"input": input, "ppg_fc": ppg_fc_out, "ecg_fc": ecg_fc_out, "bp_fc": bp_fc_out})

        # Dict for compute_loss (matches training_step)
        outputs = {
            'ppg_denoised': ppg_den, 'ppg_imputed': ppg_imp, 'ppg_sr': ppg_sr_out, 'ppg_forecast': ppg_fc_out,
            'ecg_denoised': ecg_den, 'ecg_imputed': ecg_imp, 'ecg_sr': ecg_sr_out, 'ecg_forecast': ecg_fc_out,
            'bp_denoised': bp_den, 'bp_imputed': bp_imp, 'bp_sr': bp_sr_out, 'bp_forecast': bp_fc_out,
            'ppg2ecg': ppg2ecg, 'ppg2bp': ppg2bp, 'ecg2ppg': ecg2ppg, 'ecg2bp': ecg2bp, 'bp2ppg': bp2ppg, 'bp2ecg': bp2ecg, 
        }
        
        # First batch → visualization storage 
        if batch_idx == 0:
            self.targets = {k: v[0:1].cpu() for k, v in targets.items()}  # First sample
            self.outputs = {k: v[0:1].cpu() for k, v in outputs.items()}
            self.inputs = {
                'ppg_noised': noised["ppg"][0:1].cpu(),
                'ppg_masked': masked["ppg"][0:1].cpu(),
                'ppg_sr_input': masked_sr["ppg"][0:1].cpu(),
                "ppg_forecast_input": masked_forecast["ppg"][0:1].cpu(),
                'ecg_noised': noised["ecg"][0:1].cpu(),
                'ecg_masked': masked["ecg"][0:1].cpu(), 
                'ecg_sr_input': masked_sr["ecg"][0:1].cpu(),
                "ecg_forecast_input": masked_forecast["ecg"][0:1].cpu(),
                'bp_noised': noised["bp"][0:1].cpu(),
                'bp_masked': masked["bp"][0:1].cpu(),
                'bp_sr_input': masked_sr["bp"][0:1].cpu(),
                "bp_forecast_input": masked_forecast["bp"][0:1].cpu()
            }       
            self.masks = {
                "imputed": {k: v[0:1].cpu() for k, v in masks['imputed'].items()},
                "forecast": {k: v[0:1].cpu() for k, v in masks['forecast'].items()}
            }
        
        val_loss, val_losses = self.compute_loss(outputs, targets, modality_dice=None, masks=masks, tgt_mean=ecg_mean, tgt_std=ecg_std)
        
        self.val_losses = self.update_loss_dict(self.val_losses, val_losses)

        for task in outputs.keys():
            if '2' in task:
                signal_name = task.split('2')[-1]  # Extract target modality from keys like 'ppg2ecg'
            else:
                signal_name = task.split('_')[0]  # Extract 'ppg', 'ecg', or 'bp' from the key

            if signal_name not in self.val_preds:
                self.val_preds[signal_name] = {}
            if signal_name not in self.val_targets:
                self.val_targets[signal_name] = {}

            if task not in self.val_preds[signal_name]:
                self.val_preds[signal_name][task] = []
            if task not in self.val_targets[signal_name]:
                self.val_targets[signal_name][task] = []

            # Append prediction for this task (once per task)
            self.val_preds[signal_name][task].append(outputs[task].detach().cpu())

            # Append target for this task (once per task)
            self.val_targets[signal_name][task].append(targets[signal_name].detach().cpu())

        # Lightning logging
        self.log('val_loss', val_loss, prog_bar=True, sync_dist=True)
        self.val_losses_list.append(val_loss.item())
        return val_loss

    def generate_one_step(self, signal_name="ppg", batch_size=1, fs=125, seconds=4, base_seed=42, device=None):
        if device is None:
            device = self.device
        
        torch.manual_seed(base_seed)
        self.eval()

        map_signal_to_index = {"ppg": 0, "ecg": 1, "bp": 2}
        if signal_name not in map_signal_to_index:
            raise ValueError(f"signal_name must be one of {list(map_signal_to_index.keys())}, got {signal_name}")
        i = map_signal_to_index[signal_name]
        sig_len = fs * seconds
        x_T = torch.randn(batch_size, 1, sig_len, device=device)
        mask_ones = torch.zeros(batch_size, 1, sig_len, device=device)
        model_input = torch.cat([torch.zeros(batch_size, 3, sig_len, device=device), mask_ones], dim=1)
        model_input[:, i:i+1, :] = x_T  # Place noise in the
        model_input[:, 3:, :] = 1.0  # Mask part is all ones
        t = torch.full((batch_size,), self.T-1, dtype=torch.long, device=device)  # Max noise level
        model_flags = [f"{i}_{i}"]  # Use the appropriate model flag for the selected signal
        with torch.no_grad():
            pred_tuple = self.forward(model_input, t=t, model_flags=model_flags)
            if not isinstance(pred_tuple, tuple):
                raise TypeError(f"Expected self.forward(...) to return a tuple (ppg_pred, ecg_pred, bp_pred), got {type(pred_tuple)}")
            if len(pred_tuple) < 3:
                raise ValueError(f"Expected tuple of length 3: (ppg_pred, ecg_pred, bp_pred), got length {len(pred_tuple)}")
            pred = pred_tuple[i]
            if not torch.is_tensor(pred):
                raise TypeError(f"Selected prediction must be a tensor, got {type(pred)}")
            if pred.ndim != 3:
                raise ValueError(f"Expected pred shape [B, C, L], got {pred.shape}")
            if pred.shape[1] != 1:
                raise ValueError(f"Expected selected prediction to have 1 channel for {signal_name}, got shape {pred.shape}")
            return pred
    
    def generate_ddpm(
        self,
        signal_name="ppg",
        batch_size=1,
        fs=125,
        seconds=4,
        base_seed=42,
        device=None,
    ):
        if device is None:
            device = self.device

        torch.manual_seed(base_seed)
        self.eval()

        map_signal_to_index = {"ppg": 0, "ecg": 1, "bp": 2}
        if signal_name not in map_signal_to_index:
            raise ValueError(
                f"signal_name must be one of {list(map_signal_to_index.keys())}, got {signal_name}"
            )

        i = map_signal_to_index[signal_name]
        sig_len = fs * seconds

        x_t = torch.randn(batch_size, 1, sig_len, device=device)

        signal_part = torch.zeros(batch_size, 3, sig_len, device=device)
        mask_part = torch.zeros(batch_size, 3, sig_len, device=device)

        signal_part[:, i:i+1, :] = x_t
        mask_part[:, i:i+1, :] = 1.0

        with torch.no_grad():
            for t in reversed(range(self.T)):
                t_batch = torch.full((batch_size,), t, dtype=torch.long, device=device)

                model_input = torch.cat([signal_part, mask_part], dim=1)
                pred_tuple = self.forward(model_input, t=t_batch, model_flags=[f"{i}_{i}"])

                if not isinstance(pred_tuple, tuple):
                    raise TypeError(
                        f"Expected self.forward(...) to return a tuple (ppg_pred, ecg_pred, bp_pred), got {type(pred_tuple)}"
                    )

                if len(pred_tuple) < 3:
                    raise ValueError(
                        f"Expected tuple of length 3: (ppg_pred, ecg_pred, bp_pred), got length {len(pred_tuple)}"
                    )

                pred = pred_tuple[i]

                if not torch.is_tensor(pred):
                    raise TypeError(f"Selected prediction must be a tensor, got {type(pred)}")

                if pred.ndim != 3:
                    raise ValueError(f"Expected pred shape [B, C, L], got {pred.shape}")

                if pred.shape[1] != 1:
                    raise ValueError(
                        f"Expected selected prediction to have 1 channel for {signal_name}, got shape {pred.shape}"
                    )

                if self.predict_type == "epsilon":
                    eps_theta = pred
                    alpha_t = self.train_scheduler.alphas[t].to(device)
                    alpha_bar_t = self.train_scheduler.alphas_cumprod[t].to(device)
                    beta_t = self.train_scheduler.betas[t].to(device)

                    x0_pred = (
                        x_t - torch.sqrt(1.0 - alpha_bar_t) * eps_theta
                    ) / torch.sqrt(alpha_bar_t)

                    mean = (
                        1.0 / torch.sqrt(alpha_t)
                    ) * (
                        x_t - (beta_t / torch.sqrt(1.0 - alpha_bar_t)) * eps_theta
                    )

                    if t > 0:
                        noise = torch.randn_like(x_t)
                        sigma_t = torch.sqrt(beta_t)
                        x_t = mean + sigma_t * noise
                    else:
                        x_t = mean

                elif self.predict_type == "x0":
                    x0_pred = pred

                    if t > 0:
                        alpha_t = self.train_scheduler.alphas[t].to(device)
                        alpha_bar_t = self.train_scheduler.alphas_cumprod[t].to(device)
                        beta_t = self.train_scheduler.betas[t].to(device)

                        eps_theta = (
                            x_t - torch.sqrt(alpha_bar_t) * x0_pred
                        ) / torch.sqrt(1.0 - alpha_bar_t)

                        mean = (
                            1.0 / torch.sqrt(alpha_t)
                        ) * (
                            x_t - (beta_t / torch.sqrt(1.0 - alpha_bar_t)) * eps_theta
                        )

                        noise = torch.randn_like(x_t)
                        sigma_t = torch.sqrt(beta_t)
                        x_t = mean + sigma_t * noise
                    else:
                        x_t = x0_pred

                else:
                    raise ValueError(f"Unsupported predict_type: {self.predict_type}")

                signal_part[:, i:i+1, :] = x_t

        return x_t

    def on_train_end(self):
        #save loss and metric plots at 600 dpi, save also loss and metric dictionaries as .pt files
        if self.to_plot:
            self.fig_loss.savefig(f'imgs/{self.name}_loss_plot.png', dpi=600)
            self.fig_metrics.savefig(f'imgs/{self.name}_metrics_plot.png', dpi=600)
        torch.save(self.train_losses_epoch, f'imgs/{self.name}_train_losses_epoch.pt')
        torch.save(self.val_losses_epoch, f'imgs/{self.name}_val_losses_epoch.pt')
        torch.save(self.train_metrics, f'imgs/{self.name}_train_metrics_epoch.pt')
        torch.save(self.val_metrics, f'imgs/{self.name}_val_metrics_epoch.pt')
    