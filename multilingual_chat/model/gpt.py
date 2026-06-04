"""
Multilingual Decoder-Only Conversational GPT Model.

Implements:
    - Standard causal autoregressive forward pass
    - Parameter configurations for 'full' (~124M) and 'small' (~10M) variants
    - Top-k / Top-p (Nucleus) temperature sampling for creative dialogue generation
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple, List

from model.blocks import RMSNorm, GroupedQueryAttention, SwiGLUFFN, precompute_rope_freqs


@dataclass
class GPTConfig:
    vocab_size: int = 8000
    dim: int = 768
    n_layers: int = 12
    n_heads: int = 12
    n_kv_heads: int = 4  # GQA ratio = 3 (12/4)
    multiple_of: int = 256
    max_seq_len: int = 256
    dropout: float = 0.1
    head_dim: int = 64
    hidden_dim: Optional[int] = None

    def __post_init__(self):
        # SwiGLU FFN hidden dimension sizing (similar to LLaMA)
        if self.hidden_dim is None:
            hidden_dim = int(2 * (self.dim * 4) / 3)
            self.hidden_dim = self.multiple_of * ((hidden_dim + self.multiple_of - 1) // self.multiple_of)


# Configurations matching standard GPT scales
GPT_FULL = GPTConfig(
    dim=768, n_layers=12, n_heads=12, n_kv_heads=4, max_seq_len=256
)  # ~124M parameters

GPT_SMALL = GPTConfig(
    dim=256, n_layers=6, n_heads=8, n_kv_heads=2, max_seq_len=256
)  # ~10M parameters (optimized for quick training)


class TransformerBlock(nn.Module):
    """Causal Transformer Decoder Block utilizing pre-norm architecture."""
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.attn = GroupedQueryAttention(
            dim=cfg.dim,
            n_heads=cfg.n_heads,
            n_kv_heads=cfg.n_kv_heads,
            head_dim=cfg.head_dim,
            dropout=cfg.dropout,
        )
        self.ffn = SwiGLUFFN(dim=cfg.dim, hidden_dim=cfg.hidden_dim)
        self.attn_norm = RMSNorm(dim=cfg.dim)
        self.ffn_norm = RMSNorm(dim=cfg.dim)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Pre-normalization residual paths (stable gradient flow)
        x = x + self.attn(self.attn_norm(x), freqs_cis, mask)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class MultilingualGPT(nn.Module):
    """
    Main Decoder-Only Conversational GPT Model.
    """
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg

        self.tok_embeddings = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.dropout = nn.Dropout(cfg.dropout)

        self.layers = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])

        self.norm = RMSNorm(cfg.dim)
        self.output = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)

        # Weight tying (Press & Wolf, 2017)
        self.tok_embeddings.weight = self.output.weight

        # Precompute RoPE complex frequencies
        self.freqs_cis = precompute_rope_freqs(cfg.head_dim, cfg.max_seq_len)

        # Causal triangular attention mask
        mask = torch.full((cfg.max_seq_len, cfg.max_seq_len), float("-inf"))
        mask = torch.triu(mask, diagonal=1)
        self.register_buffer("causal_mask", mask.view(1, 1, cfg.max_seq_len, cfg.max_seq_len))

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        tokens: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        _, T = tokens.shape
        x = self.dropout(self.tok_embeddings(tokens))

        # Retrieve cached RoPE frequencies and transfer to the correct device
        freqs_cis = self.freqs_cis[:T].to(tokens.device)

        # Retrieve causal triangular mask
        mask = self.causal_mask[:, :, :T, :T]

        for layer in self.layers:
            x = layer(x, freqs_cis, mask)

        x = self.norm(x)
        logits = self.output(x)  # (B, T, V)

        loss = None
        if targets is not None:
            # targets: (B, T)
            # Standard next-token CrossEntropy prediction loss
            loss = F.cross_entropy(
                logits.view(-1, self.cfg.vocab_size),
                targets.view(-1),
                ignore_index=-100,  # We ignore user turn prompts in loss calculation
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: List[int],
        max_new_tokens: int = 150,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        eos_id: int = 3,
    ) -> List[int]:
        """
        Interactive autoregressive generation supporting Top-K, Top-P (nucleus),
        and Temperature sampling.
        """
        self.eval()
        device = next(self.parameters()).device
        generated = list(prompt_ids)

        for _ in range(max_new_tokens):
            # Keep prompt context inside standard maximum sequence window
            tokens_in = torch.tensor([generated[-self.cfg.max_seq_len:]], dtype=torch.long, device=device)
            logits, _ = self(tokens_in)
            next_token_logits = logits[0, -1, :] / max(temperature, 1e-5)

            # 1. Apply Top-K filtering
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float("-inf")

            # 2. Apply Top-P (Nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift indices to keep the first token that exceeds top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_token_logits[indices_to_remove] = float("-inf")

            # Calculate probability distribution and sample
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()

            generated.append(next_token)
            if next_token == eos_id:
                break

        return generated
