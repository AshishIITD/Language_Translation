"""
Modern, highly optimized Transformer blocks for Causal Multilingual GPT.

Includes:
    - RMSNorm (Root Mean Square Layer Normalization)
    - RoPE (Rotary Position Embeddings)
    - GQA (Grouped-Query Attention)
    - SwiGLU (Swish Gated Linear Unit Feed-Forward Network)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (Zhang & Sennrich, 2019).
    Removes mean-centering (unnecessary in modern networks) for a 10-15% speedup.
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight


def precompute_rope_freqs(dim: int, max_len: int, theta: float = 10000.0) -> torch.Tensor:
    """Precompute complex rotary position frequencies."""
    assert dim % 2 == 0, "Dimension must be even for RoPE"
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(max_len)
    freqs = torch.outer(t, freqs)  # (max_len, dim//2)
    return torch.polar(torch.ones_like(freqs), freqs)  # complex tensor representation


def apply_rope(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """Apply precomputed RoPE frequencies to query/key tensors."""
    # x shape: (B, T, n_heads, head_dim)
    # freqs_cis: (T, head_dim//2) -> reshape to (1, T, 1, head_dim//2)
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    freqs_cis = freqs_cis[:x.size(1)].view(1, x.size(1), 1, -1).to(x.device)
    x_out = torch.view_as_real(x_complex * freqs_cis).flatten(3)
    return x_out.type_as(x)


class GroupedQueryAttention(nn.Module):
    """
    Grouped-Query Attention (Ainslie et al., 2023).
    Saves KV cache memory footprint by replicating KV heads across n_rep query heads.
    """
    def __init__(
        self,
        dim: int,
        n_heads: int,
        n_kv_heads: int,
        head_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.n_rep = n_heads // n_kv_heads

        assert n_heads % n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"

        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.out_proj = nn.Linear(n_heads * head_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, _ = x.shape

        # Linear projections
        xq = self.q_proj(x).view(B, T, self.n_heads, self.head_dim)
        xk = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim)
        xv = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim)

        # Apply Rotary Position Embeddings (RoPE)
        xq = apply_rope(xq, freqs_cis)
        xk = apply_rope(xk, freqs_cis)

        # Expand Key and Value heads if using GQA
        if self.n_rep > 1:
            xk = xk.repeat_interleave(self.n_rep, dim=2)
            xv = xv.repeat_interleave(self.n_rep, dim=2)

        # Transpose to (B, n_heads, T, head_dim)
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(xq, xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask

        attn_weights = F.softmax(scores.float(), dim=-1).type_as(xq)
        attn_weights = self.dropout(attn_weights)

        output = torch.matmul(attn_weights, xv)  # (B, n_heads, T, head_dim)
        output = output.transpose(1, 2).contiguous().view(B, T, -1)
        return self.out_proj(output)


class SwiGLUFFN(nn.Module):
    """
    SwiGLU Gated Feed-Forward Network (Shazeer, 2020).
    Outperforms standard GELU/ReLU in causal language modeling.
    """
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
