"""
Part II: Modern Transformer Building Blocks

Implements the three required architectural modifications:
  1. RMSNorm — Root Mean Square Layer Normalization (Zhang & Sennrich, 2019)
  2. RoPE   — Rotary Positional Embeddings (Su et al., 2021)
  3. GQA    — Grouped Query Attention (Ainslie et al., 2023)

Each is implemented from scratch with detailed comments explaining the
mathematical motivation and engineering trade-offs.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


# ═══════════════════════════════════════════════════════════════════════════════
# 1. RMSNorm
# ═══════════════════════════════════════════════════════════════════════════════

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (Zhang & Sennrich, NeurIPS 2019).

    Standard LayerNorm: y = (x - mean) / sqrt(var + ε) * γ + β
    RMSNorm:           y = x / RMS(x) * γ       where RMS(x) = sqrt(mean(x²) + ε)

    Why RMSNorm over LayerNorm?
      - Removes mean-centering (subtracting mean): empirically, the re-centering
        invariance of LayerNorm is not essential for Transformers.
      - Fewer operations → ~10-15% faster than LayerNorm in practice.
      - No β (bias) parameter → fewer parameters, less overfitting risk.
      - Identical or better performance in practice (used in LLaMA, Mistral, Gemma).
      - Numerically: avoids potential instability when mean is near the scale of ε.

    The γ (gain) parameter is initialized to 1.0, allowing the network to learn
    appropriate scaling from that identity-like starting point.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))   # γ

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., dim)
        # RMS = sqrt( mean(x²) + ε )
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Upcast to float32 for numerical stability, then cast back
        normed = self._norm(x.float()).to(x.dtype)
        return normed * self.weight


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Rotary Positional Embeddings (RoPE)
# ═══════════════════════════════════════════════════════════════════════════════

def precompute_rope_freqs(
    dim: int,
    max_seq_len: int = 2048,
    base: float = 10000.0,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Precompute cos and sin tables for RoPE.

    RoPE encodes position m into the attention by rotating query/key vectors:
        q_m' = R(mθ) q_m    where R is a rotation matrix parameterized by
                             frequency θ_i = base^(-2i/d)

    Mathematical motivation:
        For two positions m and n, the attention score q_m'^T k_n' depends
        only on (q_m, k_n, m-n) — i.e., only relative position matters.
        This gives RoPE its key advantage over absolute positional embeddings:
        it can generalize to sequences longer than those seen during training
        (with some caveats).

    Implementation uses complex number rotation:
        [x1, x2] → [x1 cos(mθ) - x2 sin(mθ), x1 sin(mθ) + x2 cos(mθ)]

    Args:
        dim: head dimension (must be even — we rotate pairs of dimensions)
        max_seq_len: maximum sequence length to precompute for
        base: frequency base (10000 in original RoPE, sometimes 500000 for long context)

    Returns:
        cos_table: (max_seq_len, dim//2) — cosine components
        sin_table: (max_seq_len, dim//2) — sine components
    """
    assert dim % 2 == 0, "RoPE requires even head dimension"

    # Inverse frequencies: θ_i = 1 / base^(2i / dim), i = 0, 1, ..., dim/2 - 1
    # Shape: (dim//2,)
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))

    # Position indices: (max_seq_len,)
    positions = torch.arange(max_seq_len, dtype=torch.float32, device=device)

    # Outer product: (max_seq_len, dim//2)
    freqs = torch.outer(positions, inv_freq)

    cos_table = freqs.cos()   # (max_seq_len, dim//2)
    sin_table = freqs.sin()

    return cos_table, sin_table


def apply_rope(
    x: torch.Tensor,           # (B, n_heads, seq_len, head_dim)
    cos: torch.Tensor,         # (seq_len, head_dim//2) or (1, 1, seq_len, head_dim//2)
    sin: torch.Tensor,
    position_ids: Optional[torch.Tensor] = None,  # (B, seq_len) for KV cache
) -> torch.Tensor:
    """
    Apply rotary embeddings to query or key tensor.

    We split the head_dim into two halves [x1 | x2] and apply:
        x_rot = [x1 cos - x2 sin | x1 sin + x2 cos]

    This is equivalent to treating pairs (x[2i], x[2i+1]) as real/imaginary
    parts of a complex number and multiplying by e^{i * m * θ_i}.
    """
    B, H, L, D = x.shape
    half_D = D // 2

    # Split into two halves
    x1 = x[..., :half_D]   # (B, H, L, D//2)
    x2 = x[..., half_D:]

    if position_ids is not None:
        # KV cache mode: select specific positions
        # Clamp to available table size
        pos_clamped = position_ids.clamp(max=cos.shape[0] - 1)
        cos_ = cos[pos_clamped].unsqueeze(1)   # (B, 1, L, D//2)
        sin_ = sin[pos_clamped].unsqueeze(1)
    else:
        # Clamp L to the precomputed table size
        L_clamped = min(L, cos.shape[0])
        cos_ = cos[:L_clamped].unsqueeze(0).unsqueeze(0)  # (1, 1, L_clamped, D//2)
        sin_ = sin[:L_clamped].unsqueeze(0).unsqueeze(0)
        # If sequence is longer than precomputed, extend with the last entry
        if L > L_clamped:
            pad_cos = cos_[:, :, -1:, :].expand(1, 1, L - L_clamped, -1)
            pad_sin = sin_[:, :, -1:, :].expand(1, 1, L - L_clamped, -1)
            cos_ = torch.cat([cos_, pad_cos], dim=2)
            sin_ = torch.cat([sin_, pad_sin], dim=2)

    # Rotate: [x1 cos - x2 sin, x1 sin + x2 cos]
    x_rot = torch.cat([
        x1 * cos_ - x2 * sin_,
        x1 * sin_ + x2 * cos_,
    ], dim=-1)

    return x_rot


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Grouped Query Attention (GQA)
# ═══════════════════════════════════════════════════════════════════════════════

class GroupedQueryAttention(nn.Module):
    """
    Grouped Query Attention (Ainslie et al., EMNLP 2023).

    Standard MHA: n_heads query heads, n_heads key heads, n_heads value heads.
    MQA:          n_heads query heads, 1 key head, 1 value head.
    GQA:          n_heads query heads, n_kv_heads key heads, n_kv_heads value heads.
                  (where n_kv_heads divides n_heads)

    Why GQA over MHA?
      - MHA's KV cache grows as O(n_heads × seq_len × head_dim) per layer.
        In large models, this dominates memory at inference time.
      - GQA reduces KV cache size by n_heads/n_kv_heads factor.
      - Minimal quality degradation vs MHA when n_kv_heads >= 4 (empirically).
      - Better compute efficiency: fewer K/V projections.
      - Used in LLaMA-2, Mistral, Gemma — well-validated at scale.

    Implementation:
      - Each KV group serves n_heads // n_kv_heads query heads.
      - KV tensors are expanded (repeated) before attention computation.
        This is memory-equivalent to MHA during forward pass but saves
        parameters and KV cache at inference.

    For our 110M BERT-like model:
      - n_heads = 12, n_kv_heads = 4 (3 query heads share 1 KV head)
    For our 124M GPT-like model:
      - n_heads = 12, n_kv_heads = 4
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        dropout: float = 0.0,
        max_seq_len: int = 2048,
        rope_base: float = 10000.0,
        is_causal: bool = False,
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        assert n_heads % n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"

        self.n_heads    = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_rep      = n_heads // n_kv_heads   # repetition factor
        self.head_dim   = d_model // n_heads
        self.d_model    = d_model
        self.is_causal  = is_causal
        self.scale      = self.head_dim ** -0.5

        # Query projection: full n_heads
        self.q_proj = nn.Linear(d_model, n_heads    * self.head_dim, bias=False)
        # KV projections: only n_kv_heads
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

        self.attn_dropout = nn.Dropout(dropout)

        # Precompute RoPE frequencies
        cos, sin = precompute_rope_freqs(self.head_dim, max_seq_len, rope_base)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def forward(
        self,
        x: torch.Tensor,                           # (B, L, d_model)
        key_value: Optional[torch.Tensor] = None,  # (B, S, d_model) for cross-attn
        attention_mask: Optional[torch.Tensor] = None,  # (B, 1, L, S)
        position_ids: Optional[torch.Tensor] = None,    # (B, L)
    ) -> torch.Tensor:
        """
        If key_value is None: self-attention.
        If key_value is not None: cross-attention (decoder uses encoder output as K,V).
        """
        B, L, _ = x.shape
        is_cross = key_value is not None
        kv_src = key_value if is_cross else x
        S = kv_src.size(1)

        # Project
        q = self.q_proj(x)          # (B, L, n_heads * head_dim)
        k = self.k_proj(kv_src)     # (B, S, n_kv_heads * head_dim)
        v = self.v_proj(kv_src)

        # Reshape to (B, n_heads, L, head_dim)
        q = q.view(B, L, self.n_heads,    self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE to Q and K (not V, not cross-attention K)
        if not is_cross:
            q = apply_rope(q, self.rope_cos, self.rope_sin, position_ids)
            k = apply_rope(k, self.rope_cos, self.rope_sin, position_ids)

        # Expand K and V from n_kv_heads to n_heads
        # (B, n_kv_heads, S, head_dim) → (B, n_heads, S, head_dim)
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        # Scaled dot-product attention
        # Use PyTorch's efficient implementation if available (Flash Attention path)
        if hasattr(F, "scaled_dot_product_attention"):
            # Build causal mask if needed
            is_causal_flag = self.is_causal and not is_cross and (attention_mask is None)
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attention_mask,
                dropout_p=self.attn_dropout.p if self.training else 0.0,
                is_causal=is_causal_flag,
            )
        else:
            # Fallback manual implementation
            scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, H, L, S)

            if self.is_causal and not is_cross:
                causal_mask = torch.triu(
                    torch.ones(L, S, dtype=torch.bool, device=x.device), diagonal=1
                )
                scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

            if attention_mask is not None:
                scores = scores + attention_mask

            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = self.attn_dropout(attn_weights)
            out = torch.matmul(attn_weights, v)

        # (B, n_heads, L, head_dim) → (B, L, d_model)
        out = out.transpose(1, 2).contiguous().view(B, L, self.d_model)
        return self.o_proj(out)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Feed-Forward Network (SwiGLU)
# ═══════════════════════════════════════════════════════════════════════════════

class SwiGLUFFN(nn.Module):
    """
    SwiGLU feed-forward network (Shazeer, 2020).

    SwiGLU: FFN(x) = (SiLU(W1 x) ⊙ W3 x) W2

    Why SwiGLU over standard FFN with ReLU?
      - Gated activations empirically outperform standard activations in
        Transformer FFNs (PaLM, LLaMA all use SwiGLU).
      - SiLU (swish) is smoother than ReLU, reducing dead neuron problem.
      - The gate (W3 x) acts as a per-dimension learnable mask, allowing the
        network to selectively amplify or suppress information.

    Standard expansion ratio: 4x hidden → intermediate dim.
    We use ~8/3 × d_model for the gate dimension to keep param count comparable.
    """

    def __init__(self, d_model: int, expansion: float = 4.0):
        super().__init__()
        # Use 8/3 multiplier to match standard FFN param count with SwiGLU
        intermediate = int(d_model * expansion * 2 / 3)
        # Round to multiple of 64 for hardware efficiency
        intermediate = ((intermediate + 63) // 64) * 64

        self.w1 = nn.Linear(d_model, intermediate, bias=False)   # gate input
        self.w3 = nn.Linear(d_model, intermediate, bias=False)   # value input
        self.w2 = nn.Linear(intermediate, d_model, bias=False)   # output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Transformer Blocks
# ═══════════════════════════════════════════════════════════════════════════════

class EncoderBlock(nn.Module):
    """
    BERT-like encoder transformer block.
    Pre-norm (RMSNorm applied before sublayer, not after).

    Pre-norm vs Post-norm:
      Post-norm (original Transformer): norm after residual add.
        → Better final performance IF training converges.
        → Numerically unstable early in training for deep models.
      Pre-norm: norm before sublayer, residual is added to unnormalized output.
        → Gradient flows more cleanly through the skip connection.
        → Easier to train without LR warmup.
        → Used by GPT-2, LLaMA, PaLM, etc.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        ffn_expansion: float = 4.0,
        dropout: float = 0.1,
        max_seq_len: int = 512,
    ):
        super().__init__()
        self.attn_norm = RMSNorm(d_model)
        self.attn = GroupedQueryAttention(
            d_model, n_heads, n_kv_heads,
            dropout=dropout, max_seq_len=max_seq_len, is_causal=False
        )
        self.ffn_norm = RMSNorm(d_model)
        self.ffn = SwiGLUFFN(d_model, ffn_expansion)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Pre-norm self-attention with residual
        residual = x
        x = self.attn_norm(x)
        x = self.attn(x, attention_mask=attention_mask)
        x = self.dropout(x) + residual

        # Pre-norm FFN with residual
        residual = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = self.dropout(x) + residual

        return x


class DecoderBlock(nn.Module):
    """
    GPT-like causal decoder block with optional cross-attention.

    Three sublayers:
      1. Causal self-attention (masked so position i only attends to ≤ i)
      2. Cross-attention over encoder outputs (only when used as seq2seq decoder)
      3. FFN

    The cross-attention gate (α) is initialized to 0.0 so the block starts as
    a pure language model and gradually learns to incorporate encoder context.
    This is a simple form of the "LoRA-like" initialization for cross-attention
    (similar to LLaMA-Adapter / LIMA approaches).
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        ffn_expansion: float = 4.0,
        dropout: float = 0.1,
        max_seq_len: int = 1024,
        use_cross_attention: bool = False,
    ):
        super().__init__()
        self.use_cross_attention = use_cross_attention

        self.self_attn_norm = RMSNorm(d_model)
        self.self_attn = GroupedQueryAttention(
            d_model, n_heads, n_kv_heads,
            dropout=dropout, max_seq_len=max_seq_len, is_causal=True
        )

        if use_cross_attention:
            self.cross_attn_norm = RMSNorm(d_model)
            self.cross_attn = GroupedQueryAttention(
                d_model, n_heads, n_kv_heads,
                dropout=dropout, max_seq_len=max_seq_len, is_causal=False
            )
            # Learnable gate for cross-attention (starts at 0 = pure LM)
            self.cross_attn_gate = nn.Parameter(torch.zeros(1))

        self.ffn_norm = RMSNorm(d_model)
        self.ffn = SwiGLUFFN(d_model, ffn_expansion)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: Optional[torch.Tensor] = None,
        self_attn_mask: Optional[torch.Tensor] = None,
        cross_attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # 1. Causal self-attention
        residual = x
        x = self.self_attn_norm(x)
        x = self.self_attn(x, attention_mask=self_attn_mask)
        x = self.dropout(x) + residual

        # 2. Cross-attention (if enabled and encoder output provided)
        if self.use_cross_attention and encoder_output is not None:
            residual = x
            x_normed = self.cross_attn_norm(x)
            cross_out = self.cross_attn(
                x_normed, key_value=encoder_output,
                attention_mask=cross_attn_mask
            )
            # Gated addition — gate starts at 0 (tanh gate ∈ [-1, 1])
            gate = torch.tanh(self.cross_attn_gate)
            x = self.dropout(gate * cross_out) + residual

        # 3. FFN
        residual = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = self.dropout(x) + residual

        return x
