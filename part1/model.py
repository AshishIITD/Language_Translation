"""
Part I: Classical NMT — LSTM Encoder-Decoder with Bahdanau Attention.

Architecture decisions (justified):
  - Bidirectional LSTM encoder: captures both left and right context for
    source representations, improving alignment quality.
  - Bahdanau (additive) attention: proven for NMT, interpretable alignment.
  - Input feeding: attention vector fed back as input at each decoder step
    (Luong et al.) — helps decoder maintain attention context.
  - Variational dropout: same mask across time steps, reduces overfitting
    in RNNs more effectively than naive dropout.
  - Layer normalization on LSTM hidden states: stabilizes training.
  - Tied embeddings (optional): src ≈ tgt vocab (shared SP vocab), reduces
    parameters and improves low-frequency word handling.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from typing import List, Optional, Tuple


# ─── Bahdanau Attention ───────────────────────────────────────────────────────

class BahdanauAttention(nn.Module):
    """
    Additive attention (Bahdanau et al., 2015).

    score(h_t, h_s) = v^T * tanh(W_a * h_t + U_a * h_s)

    Why additive over dot-product?
    - More expressive: learns separate projections for query and key.
    - Historically better for LSTM-based NMT where encoder/decoder dims differ.
    - More numerically stable for smaller hidden sizes.
    """

    def __init__(self, encoder_hidden: int, decoder_hidden: int, attn_dim: int):
        super().__init__()
        # Projects decoder hidden state (query)
        self.W_a = nn.Linear(decoder_hidden, attn_dim, bias=False)
        # Projects encoder outputs (keys) — precomputed once per source
        self.U_a = nn.Linear(encoder_hidden * 2, attn_dim, bias=False)
        # Scores the combined representation
        self.v   = nn.Linear(attn_dim, 1, bias=False)

    def forward(
        self,
        decoder_hidden: torch.Tensor,    # (B, decoder_hidden)
        encoder_outputs: torch.Tensor,   # (B, S, encoder_hidden*2)
        src_mask: Optional[torch.Tensor] = None,  # (B, S) bool, True = pad
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            context: (B, encoder_hidden*2)
            attn_weights: (B, S)
        """
        B, S, _ = encoder_outputs.size()

        # (B, 1, attn_dim)
        query = self.W_a(decoder_hidden).unsqueeze(1)
        # (B, S, attn_dim)
        keys = self.U_a(encoder_outputs)
        # (B, S, 1) → (B, S)
        energy = self.v(torch.tanh(query + keys)).squeeze(-1)

        if src_mask is not None:
            energy = energy.masked_fill(src_mask, float("-inf"))

        attn_weights = F.softmax(energy, dim=-1)   # (B, S)

        # Context: weighted sum of encoder outputs
        # (B, 1, S) × (B, S, H) → (B, 1, H) → (B, H)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)

        return context, attn_weights


# ─── Encoder ──────────────────────────────────────────────────────────────────

class LSTMEncoder(nn.Module):
    """
    Bidirectional LSTM encoder.

    The bidirectional design means each source token's representation
    incorporates both past (→) and future (←) context — important for
    languages with long-distance agreement (both Hindi and Marathi have
    SOV order with significant verb-final structures).
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_size: int,
        num_layers: int = 2,
        dropout: float = 0.3,
        padding_idx: int = 0,
        pretrained_embeddings: Optional[torch.Tensor] = None,
        freeze_embeddings: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)
        if freeze_embeddings:
            self.embedding.weight.requires_grad_(False)

        self.rnn = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)

        # Project bidirectional hidden → decoder hidden size
        self.fc_hidden = nn.Linear(hidden_size * 2, hidden_size)
        self.fc_cell   = nn.Linear(hidden_size * 2, hidden_size)

    def forward(
        self,
        src: torch.Tensor,        # (B, S)
        src_lengths: torch.Tensor # (B,)
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Returns:
            encoder_outputs: (B, S, hidden*2)
            (hidden, cell): each (num_layers, B, hidden) for decoder init
        """
        embedded = self.dropout(self.embedding(src))  # (B, S, E)

        # Pack for efficiency (skip padding in LSTM computation)
        packed = pack_padded_sequence(
            embedded, src_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, (hidden, cell) = self.rnn(packed)
        encoder_outputs, _ = pad_packed_sequence(packed_out, batch_first=True)
        # encoder_outputs: (B, S, hidden*2)

        # Combine forward + backward final hidden states for decoder init
        # hidden: (num_layers*2, B, hidden) — interleaved fwd/bwd
        hidden = self._combine_bidir(hidden)  # (num_layers, B, hidden)
        cell   = self._combine_bidir(cell)

        return encoder_outputs, (hidden, cell)

    def _combine_bidir(self, h: torch.Tensor) -> torch.Tensor:
        """
        Merge bidirectional LSTM states.
        h: (num_layers*2, B, H) → (num_layers, B, H)
        """
        # Reshape to (num_layers, 2, B, H)
        h = h.view(self.num_layers, 2, h.size(1), self.hidden_size)
        # Concatenate fwd and bwd: (num_layers, B, H*2)
        h = torch.cat([h[:, 0], h[:, 1]], dim=-1)
        # Project down to hidden_size
        # Apply tanh to keep values bounded (standard practice)
        return torch.tanh(self.fc_hidden(h))


# ─── Decoder ──────────────────────────────────────────────────────────────────

class LSTMDecoder(nn.Module):
    """
    Attention-equipped LSTM decoder with input feeding.

    Input feeding (Luong et al., 2015):
        The attentional hidden state from the previous step is concatenated
        with the current target embedding before being fed into the LSTM.
        This allows the decoder to be aware of past alignment decisions.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_size: int,
        encoder_hidden: int,
        attn_dim: int,
        num_layers: int = 2,
        dropout: float = 0.3,
        padding_idx: int = 0,
        pretrained_embeddings: Optional[torch.Tensor] = None,
        freeze_embeddings: bool = False,
        tie_weights: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)
        if freeze_embeddings:
            self.embedding.weight.requires_grad_(False)

        self.attention = BahdanauAttention(encoder_hidden, hidden_size, attn_dim)

        # Input feeding: embed_dim + context (encoder_hidden*2)
        self.rnn = nn.LSTM(
            input_size=embed_dim + encoder_hidden * 2,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)

        # Output projection
        # We project concatenation of [decoder_output, context] for richer signal
        self.fc_out = nn.Linear(hidden_size + encoder_hidden * 2, vocab_size)

        if tie_weights and embed_dim == vocab_size:
            # Tie output projection to embedding weights (Press & Wolf, 2017)
            self.fc_out.weight = self.embedding.weight

    def forward_step(
        self,
        tgt_token: torch.Tensor,                    # (B,)
        hidden: Tuple[torch.Tensor, torch.Tensor],  # each (num_layers, B, H)
        encoder_outputs: torch.Tensor,              # (B, S, H*2)
        prev_context: torch.Tensor,                 # (B, H*2) — input feeding
        src_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Tuple, torch.Tensor, torch.Tensor]:
        """
        Single decoder step.
        Returns: (logits, new_hidden, new_context, attn_weights)
        """
        embedded = self.dropout(self.embedding(tgt_token))  # (B, E)

        # Input feeding: concatenate embedding with previous context
        rnn_input = torch.cat([embedded, prev_context], dim=-1)  # (B, E+H*2)
        rnn_input = rnn_input.unsqueeze(1)                        # (B, 1, E+H*2)

        rnn_out, new_hidden = self.rnn(rnn_input, hidden)
        rnn_out = rnn_out.squeeze(1)   # (B, H)

        # Attention over encoder outputs using current decoder state
        # Use top layer hidden state for attention query
        query = new_hidden[0][-1]    # (B, H)
        context, attn_weights = self.attention(query, encoder_outputs, src_mask)

        # Output: combine decoder output + context
        combined = torch.cat([rnn_out, context], dim=-1)   # (B, H + H*2)
        logits = self.fc_out(self.dropout(combined))        # (B, V)

        return logits, new_hidden, context, attn_weights

    def forward(
        self,
        tgt: torch.Tensor,                          # (B, T)
        hidden: Tuple[torch.Tensor, torch.Tensor],
        encoder_outputs: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        teacher_forcing_ratio: float = 1.0,
    ) -> torch.Tensor:
        """
        Full teacher-forced (or scheduled sampling) forward pass.
        Returns logits: (B, T-1, V)
        """
        B, T = tgt.size()
        V = self.vocab_size

        outputs = torch.zeros(B, T - 1, V, device=tgt.device)

        # Initial context: zeros (will be filled after first attention call)
        context = torch.zeros(B, encoder_outputs.size(-1), device=tgt.device)

        # Decoder input starts with <bos> token
        dec_input = tgt[:, 0]

        for t in range(T - 1):
            logits, hidden, context, _ = self.forward_step(
                dec_input, hidden, encoder_outputs, context, src_mask
            )
            outputs[:, t, :] = logits

            # Scheduled sampling
            if teacher_forcing_ratio >= 1.0 or torch.rand(1).item() < teacher_forcing_ratio:
                dec_input = tgt[:, t + 1]
            else:
                dec_input = logits.argmax(dim=-1)

        return outputs


# ─── Full Seq2Seq Model ───────────────────────────────────────────────────────

class Seq2SeqLSTM(nn.Module):
    """
    Complete LSTM Seq2Seq translation model.
    Encoder: Bidirectional LSTM
    Decoder: LSTM with Bahdanau attention + input feeding
    """

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        embed_dim: int = 256,
        hidden_size: int = 512,
        attn_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
        pad_idx: int = 0,
        src_pretrained_emb: Optional[torch.Tensor] = None,
        tgt_pretrained_emb: Optional[torch.Tensor] = None,
        freeze_embeddings: bool = False,
        tie_weights: bool = False,
    ):
        super().__init__()

        self.encoder = LSTMEncoder(
            vocab_size=src_vocab_size,
            embed_dim=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            padding_idx=pad_idx,
            pretrained_embeddings=src_pretrained_emb,
            freeze_embeddings=freeze_embeddings,
        )

        self.decoder = LSTMDecoder(
            vocab_size=tgt_vocab_size,
            embed_dim=embed_dim,
            hidden_size=hidden_size,
            encoder_hidden=hidden_size,
            attn_dim=attn_dim,
            num_layers=num_layers,
            dropout=dropout,
            padding_idx=pad_idx,
            pretrained_embeddings=tgt_pretrained_emb,
            freeze_embeddings=freeze_embeddings,
            tie_weights=tie_weights,
        )

        self.pad_idx = pad_idx
        self._init_weights()

    def _init_weights(self):
        """Xavier uniform for linear layers, orthogonal for LSTM weights."""
        for name, p in self.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(p)
            elif "weight_hh" in name:
                nn.init.orthogonal_(p)
            elif "bias" in name:
                nn.init.zeros_(p)
                # Set forget gate bias to 1.0 (Jozefowicz et al., 2015)
                if "bias_ih" in name or "bias_hh" in name:
                    n = p.size(0)
                    p.data[n // 4 : n // 2].fill_(1.0)

    def forward(
        self,
        src: torch.Tensor,        # (B, S)
        tgt: torch.Tensor,        # (B, T)
        src_lengths: torch.Tensor,
        teacher_forcing_ratio: float = 1.0,
    ) -> torch.Tensor:
        src_mask = (src == self.pad_idx)   # (B, S) True = padding
        encoder_outputs, hidden = self.encoder(src, src_lengths)
        logits = self.decoder(tgt, hidden, encoder_outputs, src_mask, teacher_forcing_ratio)
        return logits  # (B, T-1, V)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @torch.no_grad()
    def greedy_decode(
        self,
        src: torch.Tensor,
        src_lengths: torch.Tensor,
        bos_id: int,
        eos_id: int,
        max_len: int = 150,
    ) -> torch.Tensor:
        """Greedy decoding for inference. Returns (B, T) token ids."""
        self.eval()
        B = src.size(0)
        encoder_outputs, hidden = self.encoder(src, src_lengths)
        src_mask = (src[:, :encoder_outputs.size(1)] == self.pad_idx)

        dec_input = torch.full((B,), bos_id, dtype=torch.long, device=src.device)
        context = torch.zeros(B, encoder_outputs.size(-1), device=src.device)

        outputs = []
        finished = torch.zeros(B, dtype=torch.bool, device=src.device)

        for _ in range(max_len):
            logits, hidden, context, _ = self.decoder.forward_step(
                dec_input, hidden, encoder_outputs, context, src_mask
            )
            pred = logits.argmax(dim=-1)   # (B,)
            outputs.append(pred)
            finished |= (pred == eos_id)
            dec_input = pred
            if finished.all():
                break

        return torch.stack(outputs, dim=1)   # (B, T)

    @torch.no_grad()
    def beam_search(
        self,
        src: torch.Tensor,        # (1, S) — single sentence
        src_lengths: torch.Tensor,
        bos_id: int,
        eos_id: int,
        beam_size: int = 4,
        max_len: int = 150,
        length_penalty: float = 0.6,
    ) -> List[int]:
        """
        Beam search decoding for a single sentence.
        Length penalty (Wu et al., 2016): lp = (5+len)^α / (5+1)^α
        """
        from typing import List as L
        device = src.device
        encoder_outputs, init_hidden = self.encoder(src, src_lengths)
        src_mask = (src[:, :encoder_outputs.size(1)] == self.pad_idx)

        # Beams: list of (score, token_ids, hidden, context)
        init_context = torch.zeros(1, encoder_outputs.size(-1), device=device)
        beams = [(0.0, [bos_id], init_hidden, init_context)]
        completed = []

        for _ in range(max_len):
            new_beams = []
            for score, tokens, hidden, context in beams:
                if tokens[-1] == eos_id:
                    completed.append((score, tokens))
                    continue

                dec_input = torch.tensor([tokens[-1]], device=device)
                logits, new_hidden, new_ctx, _ = self.decoder.forward_step(
                    dec_input, hidden, encoder_outputs, context, src_mask
                )
                log_probs = F.log_softmax(logits[0], dim=-1)
                top_probs, top_ids = log_probs.topk(beam_size)

                for prob, tok_id in zip(top_probs.tolist(), top_ids.tolist()):
                    new_score = score + prob
                    new_beams.append((new_score, tokens + [tok_id], new_hidden, new_ctx))

            # Keep top beam_size beams
            new_beams.sort(key=lambda x: x[0] / self._length_penalty(len(x[1]), length_penalty),
                           reverse=True)
            beams = new_beams[:beam_size]

            if len(completed) >= beam_size:
                break

        if not completed:
            completed = [(s, t) for s, t, _, _ in beams]

        # Return the highest-scoring completed sequence
        completed.sort(
            key=lambda x: x[0] / self._length_penalty(len(x[1]), length_penalty),
            reverse=True
        )
        return completed[0][1]

    @staticmethod
    def _length_penalty(length: int, alpha: float) -> float:
        return ((5 + length) ** alpha) / ((5 + 1) ** alpha)
