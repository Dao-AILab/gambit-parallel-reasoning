# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
import torch.nn as nn
import torch.nn.functional as F


class HiddenstateClassifier(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int] | None = None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512]
        layers: list[torch.nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(torch.nn.Linear(prev, h))
            layers.append(torch.nn.ReLU())
            prev = h
        layers.append(torch.nn.Linear(prev, 1))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# Sequence scorer: causal Transformer (RoPE attention + SwiGLU FFN)

class RoPEAttention(nn.Module):
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.0,
                 rope_base: int = 10000):
        super().__init__()
        assert d_model % nhead == 0
        self.nhead     = nhead
        self.d_head    = d_model // nhead
        self.qkv       = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj  = nn.Linear(d_model, d_model, bias=False)
        self.dropout_p = dropout

        inv_freq = 1.0 / (
            rope_base ** (torch.arange(0, self.d_head, 2, dtype=torch.float32) / self.d_head)
        )
        self.register_buffer("inv_freq", inv_freq)

    def _rope_cos_sin(self, T: int, device, dtype):
        t     = torch.arange(T, device=device, dtype=dtype)
        freqs = torch.outer(t, self.inv_freq.to(dtype))
        emb   = torch.cat([freqs, freqs], dim=-1)
        return emb.cos(), emb.sin()

    @staticmethod
    def _rotate_half(x):
        half = x.shape[-1] // 2
        return torch.cat([-x[..., half:], x[..., :half]], dim=-1)

    def forward(self, x, key_padding_mask=None):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.nhead, self.d_head).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        cos, sin = self._rope_cos_sin(T, x.device, x.dtype)
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
        q = q * cos + self._rotate_half(q) * sin
        k = k * cos + self._rotate_half(k) * sin

        attn_bias = torch.zeros(B, 1, T, T, device=x.device, dtype=x.dtype)
        causal = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        attn_bias = attn_bias.masked_fill(causal.unsqueeze(0).unsqueeze(0), float('-inf'))
        if key_padding_mask is not None:
            pad_bias = torch.zeros(B, 1, 1, T, device=x.device, dtype=x.dtype)
            pad_bias = pad_bias.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf')
            )
            attn_bias = attn_bias + pad_bias

        dp  = self.dropout_p if self.training else 0.0
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias, dropout_p=dp)
        out = out.transpose(1, 2).reshape(B, T, C)
        return self.out_proj(out)


class SwiGLUFFN(nn.Module):
    def __init__(self, d_model: int, dim_feedforward: int, dropout: float = 0.0):
        super().__init__()
        self.gate = nn.Linear(d_model, dim_feedforward, bias=False)
        self.up   = nn.Linear(d_model, dim_feedforward, bias=False)
        self.down = nn.Linear(dim_feedforward, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.down(self.drop(F.silu(self.gate(x)) * self.up(x)))


class SequenceTransformerLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int,
                 dropout: float = 0.0, rope_base: int = 10000):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn  = RoPEAttention(d_model, nhead, dropout, rope_base=rope_base)
        self.ffn   = SwiGLUFFN(d_model, dim_feedforward, dropout)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x, key_padding_mask=None):
        x = x + self.drop(self.attn(self.norm1(x), key_padding_mask))
        x = x + self.drop(self.ffn(self.norm2(x)))
        return x


class SequenceScorer(nn.Module):
    """History-aware sequence scorer: N-layer causal Transformer over per-step hidden states.

    Input:  x  [B, T, input_dim]
    Output: logits  [B, T, 1]
    """

    def __init__(
        self,
        input_dim: int = 4096,
        d_model: int = 256,
        nhead: int = 4,
        dim_feedforward: int = 0,
        dropout: float = 0.1,
        max_len: int = 4096,
        num_layers: int = 1,
        rope_base: int = 10000,
    ):
        super().__init__()
        if dim_feedforward <= 0:
            dim_feedforward = d_model * 4

        self.input_proj = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, d_model),
            nn.GELU(),
        )
        self.max_len = max_len
        self.layers  = nn.ModuleList([
            SequenceTransformerLayer(d_model, nhead, dim_feedforward, dropout,
                                   rope_base=rope_base)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x, key_padding_mask=None):
        x = self.input_proj(x)
        for layer in self.layers:
            x = layer(x, key_padding_mask)
        x = self.norm(x)
        return self.head(x)   # [B, T, 1]

