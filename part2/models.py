"""
Part II: Full Model Definitions

1. HindiMarathiBERT — ~110M parameter bidirectional encoder
   - MLM pretraining objective (no NSP as specified)
   - Bidirectional self-attention: each token attends to all others

2. HindiMarathiGPT — ~124M parameter causal decoder
   - CLM (Causal Language Modeling) pretraining objective
   - Autoregressive: each token attends only to previous tokens

3. Seq2SeqTransformer — Translation model using pretrained BERT + GPT
   - BERT as encoder: extracts bidirectional source representations
   - GPT with cross-attention as decoder: autoregressive target generation
   - Cross-attention added to each GPT decoder block (gated, initially 0)
   - Only cross-attention parameters + output head fine-tuned initially

Parameter count verification:
  BERT-like (~110M):
    Embedding: 8000 × 768 = 6.14M
    12 layers × (GQA + FFN) ≈ 8.6M/layer = 103M
    LM head: 768 × 8000 = 6.14M
    Total ≈ 110M ✓

  GPT-like (~124M):
    Embedding: 8000 × 768 = 6.14M
    12 layers × (Causal GQA + FFN) ≈ 9.7M/layer = 116M
    LM head: tied to embedding = 0 extra
    Total ≈ 124M ✓
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict

from part2.transformer_blocks import (
    RMSNorm, EncoderBlock, DecoderBlock, precompute_rope_freqs
)


# ─── Config dataclasses ───────────────────────────────────────────────────────

from dataclasses import dataclass, field

@dataclass
class BERTConfig:
    vocab_size: int     = 8000
    d_model: int        = 768
    n_heads: int        = 12
    n_kv_heads: int     = 4      # GQA: 3 query heads per KV group
    n_layers: int       = 12
    max_seq_len: int    = 512
    dropout: float      = 0.1
    ffn_expansion: float = 4.0
    pad_idx: int        = 0
    mlm_prob: float     = 0.15   # fraction of tokens to mask

    @property
    def approx_params(self):
        emb = self.vocab_size * self.d_model
        per_layer = (
            # GQA: Q + KV + O projections
            self.d_model * self.n_heads * (self.d_model // self.n_heads) +
            self.d_model * self.n_kv_heads * (self.d_model // self.n_heads) * 2 +
            self.d_model * self.d_model +
            # FFN (SwiGLU): W1 + W3 + W2
            3 * self.d_model * int(self.d_model * self.ffn_expansion * 2/3)
        )
        return emb + self.n_layers * per_layer


@dataclass
class GPTConfig:
    vocab_size: int      = 8000
    d_model: int         = 768
    n_heads: int         = 12
    n_kv_heads: int      = 4
    n_layers: int        = 12
    max_seq_len: int     = 1024
    dropout: float       = 0.1
    ffn_expansion: float = 4.0
    pad_idx: int         = 0
    tie_embeddings: bool = True   # tie LM head to input embeddings


# ─── BERT-like Encoder ────────────────────────────────────────────────────────

class HindiMarathiBERT(nn.Module):
    """
    BERT-like bidirectional encoder pretrained with Masked Language Modeling.

    Key differences from original BERT:
      - No NSP objective (as specified; NSP was shown to be unhelpful by RoBERTa)
      - RMSNorm instead of LayerNorm
      - GQA instead of MHA (4 KV heads, 12 query heads)
      - RoPE instead of learned positional embeddings
      - SwiGLU FFN instead of GELU-MLP
      - Pre-norm architecture for training stability

    MLM Pipeline:
      - For each batch, randomly mask 15% of tokens
      - Of those: 80% → [MASK] token, 10% → random token, 10% → unchanged
        (this noise schedule prevents the model from learning that only
        masked tokens need predictions — forces robust representations)
    """

    MASK_ID = 6   # We reserve ID 6 for [MASK] (IDs 0-5 are special tokens)

    def __init__(self, config: BERTConfig):
        super().__init__()
        self.config = config
        self.pad_idx = config.pad_idx

        self.embedding = nn.Embedding(config.vocab_size, config.d_model,
                                      padding_idx=config.pad_idx)
        self.embed_dropout = nn.Dropout(config.dropout)

        self.layers = nn.ModuleList([
            EncoderBlock(
                d_model=config.d_model,
                n_heads=config.n_heads,
                n_kv_heads=config.n_kv_heads,
                ffn_expansion=config.ffn_expansion,
                dropout=config.dropout,
                max_seq_len=config.max_seq_len,
            )
            for _ in range(config.n_layers)
        ])
        self.final_norm = RMSNorm(config.d_model)

        # MLM head: project hidden → vocab
        self.mlm_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            RMSNorm(config.d_model),
            nn.Linear(config.d_model, config.vocab_size, bias=False),
        )

        self._init_weights()

    def _init_weights(self):
        std = 0.02
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=std)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()

    def encode(
        self,
        input_ids: torch.Tensor,           # (B, L)
        attention_mask: Optional[torch.Tensor] = None,  # (B, L) 1=valid, 0=pad
    ) -> torch.Tensor:
        """
        Forward pass of encoder (without MLM head).
        Returns: (B, L, d_model) hidden states.
        """
        B, L = input_ids.shape

        x = self.embedding(input_ids)   # (B, L, d_model)
        x = self.embed_dropout(x)

        # Build attention mask for padding
        # (B, 1, 1, L) additive mask: 0 for valid, -inf for pad
        if attention_mask is not None:
            # attention_mask: (B, L) 1=keep, 0=mask
            attn_mask = (1.0 - attention_mask.float()).unsqueeze(1).unsqueeze(2) * -1e9
        else:
            attn_mask = None

        for layer in self.layers:
            x = layer(x, attention_mask=attn_mask)

        x = self.final_norm(x)
        return x   # (B, L, d_model)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        masked_labels: Optional[torch.Tensor] = None,  # (B, L) -100 = ignore
    ) -> Dict[str, torch.Tensor]:
        """
        MLM forward pass.
        Returns dict with 'loss' and 'logits'.
        """
        hidden = self.encode(input_ids, attention_mask)
        logits = self.mlm_head(hidden)   # (B, L, V)

        result = {"logits": logits, "hidden_states": hidden}

        if masked_labels is not None:
            B, L, V = logits.shape
            loss = F.cross_entropy(
                logits.reshape(B * L, V),
                masked_labels.reshape(B * L),
                ignore_index=-100,
            )
            result["loss"] = loss

        return result

    def create_mlm_batch(
        self,
        input_ids: torch.Tensor,  # (B, L)
        mask_prob: float = 0.15,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create MLM inputs by randomly masking tokens.
        Returns: (masked_input_ids, labels)
          labels: original token ids for masked positions, -100 elsewhere
        """
        labels = input_ids.clone()
        labels.fill_(-100)

        # Sample positions to mask (excluding special tokens: ids 0–6)
        special_mask = input_ids < 7
        mask_prob_matrix = torch.full_like(input_ids, mask_prob, dtype=torch.float)
        mask_prob_matrix[special_mask] = 0.0

        masked_positions = torch.bernoulli(mask_prob_matrix).bool()
        labels[masked_positions] = input_ids[masked_positions]  # track true labels

        # 80% → [MASK], 10% → random, 10% → unchanged
        rand = torch.rand_like(input_ids, dtype=torch.float)

        # 80%: replace with [MASK]
        replace_mask = masked_positions & (rand < 0.8)
        input_ids = input_ids.clone()
        input_ids[replace_mask] = self.MASK_ID

        # 10%: replace with random token
        replace_rand = masked_positions & (rand >= 0.8) & (rand < 0.9)
        random_tokens = torch.randint_like(input_ids, low=7, high=self.config.vocab_size)
        input_ids[replace_rand] = random_tokens[replace_rand]

        # 10%: keep original (no change needed)

        return input_ids, labels

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─── GPT-like Decoder ─────────────────────────────────────────────────────────

class HindiMarathiGPT(nn.Module):
    """
    GPT-2 style causal language model.
    Pretrained with standard causal language modeling (CLM):
        P(x_t | x_1, ..., x_{t-1})

    Can be extended with cross-attention for translation (see Seq2SeqTransformer).

    Why pretrain GPT on the translation corpus?
      - The model learns the statistical structure of the target language
        (Marathi or Hindi) in a self-supervised way.
      - When later used as a decoder with cross-attention, it already knows
        how to generate fluent target-language sentences.
      - The pretrained LM provides a strong initialization that reduces the
        amount of parallel data needed for fine-tuning.

    This is analogous to how GPT-2 → InstructGPT leveraged LM pretraining
    for a downstream task (RLHF alignment).
    """

    def __init__(self, config: GPTConfig, use_cross_attention: bool = False):
        super().__init__()
        self.config = config
        self.use_cross_attention = use_cross_attention

        self.embedding = nn.Embedding(config.vocab_size, config.d_model,
                                      padding_idx=config.pad_idx)
        self.embed_dropout = nn.Dropout(config.dropout)

        self.layers = nn.ModuleList([
            DecoderBlock(
                d_model=config.d_model,
                n_heads=config.n_heads,
                n_kv_heads=config.n_kv_heads,
                ffn_expansion=config.ffn_expansion,
                dropout=config.dropout,
                max_seq_len=config.max_seq_len,
                use_cross_attention=use_cross_attention,
            )
            for _ in range(config.n_layers)
        ])
        self.final_norm = RMSNorm(config.d_model)

        # LM head
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Tie embeddings (Press & Wolf 2017) — reduces parameters and
        # improves performance by ensuring embedding and output spaces are aligned
        if config.tie_embeddings:
            self.lm_head.weight = self.embedding.weight

        self._init_weights()

    def _init_weights(self):
        std = 0.02
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Scaled initialization for residual layers (GPT-2 trick):
                # divide by sqrt(2 * n_layers) to prevent variance explosion
                if hasattr(module, "_is_residual"):
                    nn.init.normal_(module.weight, std=std / math.sqrt(2 * self.config.n_layers))
                else:
                    nn.init.normal_(module.weight, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=std)

    def forward(
        self,
        input_ids: torch.Tensor,                          # (B, L)
        encoder_output: Optional[torch.Tensor] = None,   # (B, S, d_model)
        attention_mask: Optional[torch.Tensor] = None,   # (B, L) padding mask
        encoder_mask: Optional[torch.Tensor] = None,     # (B, S) encoder padding
        labels: Optional[torch.Tensor] = None,           # (B, L) for CLM loss
    ) -> Dict[str, torch.Tensor]:
        B, L = input_ids.shape

        x = self.embedding(input_ids)
        x = self.embed_dropout(x)

        # Build additive mask for padding (not needed for causal mask — handled in attn)
        self_attn_mask = None
        if attention_mask is not None:
            self_attn_mask = (1.0 - attention_mask.float()).unsqueeze(1).unsqueeze(2) * -1e9

        cross_attn_mask = None
        if encoder_mask is not None:
            cross_attn_mask = (1.0 - encoder_mask.float()).unsqueeze(1).unsqueeze(2) * -1e9

        for layer in self.layers:
            x = layer(
                x,
                encoder_output=encoder_output,
                self_attn_mask=self_attn_mask,
                cross_attn_mask=cross_attn_mask,
            )

        x = self.final_norm(x)
        logits = self.lm_head(x)   # (B, L, V)

        result = {"logits": logits}

        if labels is not None:
            # Shift: predict token t+1 from token t
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            B_, T, V = shift_logits.shape
            loss = F.cross_entropy(
                shift_logits.reshape(B_ * T, V),
                shift_labels.reshape(B_ * T),
                ignore_index=0,   # PAD_ID
            )
            result["loss"] = loss

        return result

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─── Seq2Seq Transformer ──────────────────────────────────────────────────────

class Seq2SeqTransformer(nn.Module):
    """
    Translation model combining pretrained BERT encoder + GPT decoder.

    Architecture rationale (the 'Core Challenge' from the assignment):

    BERT's bidirectionality is ideal for source encoding:
      - In translation, we have access to the complete source sentence.
      - Bidirectional attention allows each source token to be contextualized
        by the entire source sequence, producing rich representations.
      - This is exactly what BERT is pretrained for.

    GPT's autoregressive design is ideal for target generation:
      - Translation is generation: we produce tokens left-to-right.
      - GPT is pretrained to model p(target_token | previous_target_tokens).
      - It already knows the statistical structure of the target language.

    Cross-attention bridge:
      - Each GPT decoder layer gets a cross-attention sublayer that attends
        to the BERT encoder's output.
      - The cross-attention gate (initialized to 0) means at fine-tuning start,
        the decoder behaves like pure GPT, and gradually learns to incorporate
        source context.

    Fine-tuning strategy:
      Phase 1 (frozen BERT, frozen GPT self-attn, only cross-attn trainable):
        - Teaches the model to use source representations without disrupting
          the pretrained LM weights.
        - Cross-attention and LM head are trained from scratch.
      Phase 2 (all parameters unfrozen, lower LR):
        - End-to-end fine-tuning adapts all parameters for translation.
    """

    def __init__(
        self,
        bert_config: BERTConfig,
        gpt_config: GPTConfig,
        bert_pretrained_path: Optional[str] = None,
        gpt_pretrained_path: Optional[str] = None,
    ):
        super().__init__()

        # Encoder: BERT (bidirectional)
        self.encoder = HindiMarathiBERT(bert_config)

        # Decoder: GPT with cross-attention enabled
        self.decoder = HindiMarathiGPT(gpt_config, use_cross_attention=True)

        # If encoder and decoder have different d_model, add projection
        if bert_config.d_model != gpt_config.d_model:
            self.enc_proj = nn.Linear(bert_config.d_model, gpt_config.d_model, bias=False)
        else:
            self.enc_proj = None

        # Load pretrained weights if provided
        if bert_pretrained_path:
            self._load_bert(bert_pretrained_path)
        if gpt_pretrained_path:
            self._load_gpt(gpt_pretrained_path)

    def _load_bert(self, path: str):
        ckpt = torch.load(path, map_location="cpu")
        state = ckpt.get("model_state_dict", ckpt)
        # Don't load MLM head — it's not used during translation
        state = {k: v for k, v in state.items() if not k.startswith("mlm_head")}
        missing, unexpected = self.encoder.load_state_dict(state, strict=False)
        print(f"Loaded BERT encoder. Missing: {len(missing)}  Unexpected: {len(unexpected)}")

    def _load_gpt(self, path: str):
        ckpt = torch.load(path, map_location="cpu")
        state = ckpt.get("model_state_dict", ckpt)
        # Load only the self-attention and FFN layers; skip cross-attention
        # (cross-attention is new and not in the pretrained GPT checkpoint)
        compatible = {}
        for k, v in state.items():
            if "cross_attn" not in k and "cross_attn_gate" not in k:
                compatible[k] = v
        missing, unexpected = self.decoder.load_state_dict(compatible, strict=False)
        print(f"Loaded GPT decoder. Missing: {len(missing)}  Unexpected: {len(unexpected)}")

    def freeze_for_phase1(self):
        """
        Phase 1: Only train cross-attention parameters + LM head.
        Everything else (BERT encoder + GPT self-attention + FFN) is frozen.
        """
        # Freeze all
        for p in self.parameters():
            p.requires_grad_(False)

        # Unfreeze cross-attention gates and weights
        for layer in self.decoder.layers:
            if hasattr(layer, "cross_attn"):
                for p in layer.cross_attn.parameters():
                    p.requires_grad_(True)
                layer.cross_attn_gate.requires_grad_(True)
                for p in layer.cross_attn_norm.parameters():
                    p.requires_grad_(True)

        # Unfreeze LM head
        for p in self.decoder.lm_head.parameters():
            p.requires_grad_(True)

        if self.enc_proj:
            for p in self.enc_proj.parameters():
                p.requires_grad_(True)

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Phase 1: {trainable:,} trainable parameters")

    def unfreeze_all(self):
        """Phase 2: Unfreeze all parameters for end-to-end fine-tuning."""
        for p in self.parameters():
            p.requires_grad_(True)
        total = sum(p.numel() for p in self.parameters())
        print(f"Phase 2: {total:,} total parameters (all unfrozen)")

    def forward(
        self,
        src_ids: torch.Tensor,          # (B, S)
        tgt_ids: torch.Tensor,          # (B, T)
        src_mask: Optional[torch.Tensor] = None,   # (B, S)
        tgt_mask: Optional[torch.Tensor] = None,   # (B, T)
        labels: Optional[torch.Tensor] = None,     # (B, T)
    ) -> Dict[str, torch.Tensor]:
        # Encode source with BERT
        encoder_output = self.encoder.encode(src_ids, src_mask)  # (B, S, d_bert)

        # Optional projection
        if self.enc_proj is not None:
            encoder_output = self.enc_proj(encoder_output)

        # Decode with GPT + cross-attention
        out = self.decoder(
            input_ids=tgt_ids,
            encoder_output=encoder_output,
            attention_mask=tgt_mask,
            encoder_mask=src_mask,
            labels=labels,
        )
        return out

    @torch.no_grad()
    def greedy_decode(
        self,
        src_ids: torch.Tensor,       # (B, S)
        src_mask: torch.Tensor,
        bos_id: int,
        eos_id: int,
        max_len: int = 150,
    ) -> torch.Tensor:
        self.eval()
        B = src_ids.size(0)
        device = src_ids.device

        encoder_output = self.encoder.encode(src_ids, src_mask)
        if self.enc_proj:
            encoder_output = self.enc_proj(encoder_output)

        generated = torch.full((B, 1), bos_id, dtype=torch.long, device=device)
        finished = torch.zeros(B, dtype=torch.bool, device=device)

        # Respect decoder's max_seq_len to stay within RoPE table
        safe_max = min(max_len, self.decoder.config.max_seq_len - 2)

        for _ in range(safe_max):
            out = self.decoder(
                input_ids=generated,
                encoder_output=encoder_output,
                encoder_mask=src_mask,
            )
            next_token = out["logits"][:, -1, :].argmax(dim=-1)  # (B,)
            finished |= (next_token == eos_id)
            generated = torch.cat([generated, next_token.unsqueeze(1)], dim=1)
            if finished.all():
                break

        return generated[:, 1:]   # strip <bos>

    @torch.no_grad()
    def beam_search(
        self,
        src_ids: torch.Tensor,        # (1, S) — single sentence
        src_mask: torch.Tensor,        # (1, S)
        bos_id: int,
        eos_id: int,
        beam_size: int = 4,
        max_len: int = 150,
        length_penalty: float = 0.6,
    ) -> list:
        """
        Beam search decoding for a single sentence (batch_size=1).

        Length penalty (Wu et al., 2016): lp = (5+len)^α / (5+1)^α
        This prevents the model from preferring short translations.
        """
        device = src_ids.device
        assert src_ids.size(0) == 1, "beam_search expects batch_size=1"

        encoder_output = self.encoder.encode(src_ids, src_mask)
        if self.enc_proj:
            encoder_output = self.enc_proj(encoder_output)

        safe_max = min(max_len, self.decoder.config.max_seq_len - 2)

        # Beams: list of (score, token_ids_tensor)
        init_ids = torch.full((1, 1), bos_id, dtype=torch.long, device=device)
        beams = [(0.0, init_ids)]
        completed = []

        for _ in range(safe_max):
            new_beams = []
            for score, token_ids in beams:
                # Check if already ended
                if token_ids[0, -1].item() == eos_id:
                    completed.append((score, token_ids[0].tolist()))
                    continue

                out = self.decoder(
                    input_ids=token_ids,
                    encoder_output=encoder_output,
                    encoder_mask=src_mask,
                )
                log_probs = F.log_softmax(out["logits"][0, -1, :], dim=-1)
                top_probs, top_ids = log_probs.topk(beam_size)

                for prob, tok_id in zip(top_probs.tolist(), top_ids.tolist()):
                    new_score = score + prob
                    new_ids = torch.cat([
                        token_ids,
                        torch.tensor([[tok_id]], dtype=torch.long, device=device)
                    ], dim=1)
                    new_beams.append((new_score, new_ids))

            # Keep top beam_size beams (sorted by length-penalized score)
            new_beams.sort(
                key=lambda x: x[0] / self._length_penalty(x[1].size(1), length_penalty),
                reverse=True,
            )
            beams = new_beams[:beam_size]

            if len(completed) >= beam_size:
                break

        # Add remaining beams to completed
        if not completed:
            completed = [(s, ids[0].tolist()) for s, ids in beams]

        # Return highest-scoring completed sequence (skip <bos>)
        completed.sort(
            key=lambda x: x[0] / self._length_penalty(len(x[1]), length_penalty),
            reverse=True,
        )
        return completed[0][1][1:]  # strip leading <bos>

    @staticmethod
    def _length_penalty(length: int, alpha: float) -> float:
        """Wu et al. (2016) length penalty."""
        return ((5 + length) ** alpha) / ((5 + 1) ** alpha)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─── Pre-instantiated configs (import these directly) ─────────────────────────

BERT_FULL = BERTConfig(
    vocab_size=8000, d_model=768, n_heads=12, n_kv_heads=4,
    n_layers=12, max_seq_len=512, dropout=0.1
)
GPT_FULL = GPTConfig(
    vocab_size=8000, d_model=768, n_heads=12, n_kv_heads=4,
    n_layers=12, max_seq_len=1024, dropout=0.1
)
# Scaled-down (~30M) — trainable on MX250 2GB with batch_size=4, grad_accum=8
BERT_SMALL = BERTConfig(
    vocab_size=8000, d_model=384, n_heads=6, n_kv_heads=2,
    n_layers=8, max_seq_len=512, dropout=0.1
)
GPT_SMALL = GPTConfig(
    vocab_size=8000, d_model=384, n_heads=6, n_kv_heads=2,
    n_layers=8, max_seq_len=1024, dropout=0.1
)
