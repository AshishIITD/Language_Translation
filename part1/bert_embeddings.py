"""
BERT-based embedding extraction for Part I experiments.

Design rationale:
    Using pretrained multilingual BERT embeddings (Hindi BERT / Marathi BERT
    from l3cube-pune) as initialization for the Seq2Seq encoder/decoder
    embedding tables has several expected benefits:

    1. Semantic initialization: subword units already carry meaning, so the
       model starts from a semantically meaningful point rather than random
       noise. This typically accelerates convergence.

    2. Morphological awareness: both Hindi and Marathi have rich inflectional
       morphology. BERT's subword tokenizer (WordPiece) was trained on large
       monolingual corpora and thus captures morphological structure well.

    3. Low-frequency word handling: rare tokens in the parallel corpus get
       representations grounded in large monolingual pretraining data, reducing
       the impact of data sparsity.

    4. Challenges with BERT embeddings in Seq2Seq:
       - Vocabulary mismatch: BERT uses WordPiece; our Seq2Seq uses SentencePiece.
         We must project BERT embeddings into our SP vocabulary space.
       - Dimension mismatch: BERT hidden dim (768) ≠ our embed_dim.
         We add a learned linear projection.
       - Frozen vs trainable: frozen embeddings risk being too general; fully
         trainable may overfit. Warmup-then-unfreeze is a good middle ground.
"""

import torch
import torch.nn as nn
from typing import List, Optional
import numpy as np


def extract_bert_embeddings_for_vocab(
    sp_vocab: List[str],
    bert_model_name: str,
    device: str = "cpu",
    batch_size: int = 64,
) -> torch.Tensor:
    """
    For each token in our SentencePiece vocabulary, get a BERT-derived embedding.

    Strategy:
        - For each SP token, encode it with BERT's tokenizer and take the mean
          of BERT's last-hidden-state over the resulting subword tokens.
        - This maps SP vocabulary items → BERT embedding space (dim 768).
        - A learned projection layer then maps 768 → our embed_dim.

    Args:
        sp_vocab: list of SP vocabulary items (length = vocab_size)
        bert_model_name: HuggingFace model name
            e.g. "l3cube-pune/hindi-bert-v2" or "l3cube-pune/marathi-bert-v2"
        device: "cpu" or "cuda"
        batch_size: number of tokens to encode at once

    Returns:
        embeddings: (vocab_size, 768) float tensor
    """
    try:
        from transformers import AutoTokenizer, AutoModel
    except ImportError:
        raise ImportError("Install transformers: pip install transformers")

    print(f"Loading BERT: {bert_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
    model = AutoModel.from_pretrained(bert_model_name).to(device)
    model.eval()

    embed_dim = model.config.hidden_size  # typically 768
    vocab_size = len(sp_vocab)
    embeddings = torch.zeros(vocab_size, embed_dim)

    print(f"Extracting embeddings for {vocab_size} SP tokens...")

    with torch.no_grad():
        for i in range(0, vocab_size, batch_size):
            batch_tokens = sp_vocab[i : i + batch_size]

            # Encode each token piece as a standalone string
            # We use "▁" prefix (SP uses this for word boundaries) → strip it
            cleaned = [t.replace("▁", " ").strip() or t for t in batch_tokens]

            encoded = tokenizer(
                cleaned,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=16,
            ).to(device)

            out = model(**encoded)
            # Take mean of last hidden state (ignoring padding positions)
            mask = encoded["attention_mask"].unsqueeze(-1).float()
            token_emb = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            # token_emb: (batch, 768)

            embeddings[i : i + len(batch_tokens)] = token_emb.cpu()

            if (i // batch_size) % 10 == 0:
                print(f"  {i}/{vocab_size} tokens processed")

    print(f"BERT embedding extraction complete. Shape: {embeddings.shape}")
    return embeddings   # (vocab_size, 768)


class BERTEmbeddingProjector(nn.Module):
    """
    Projects BERT-space embeddings (768-dim) into our model's embed_dim.

    We use a 2-layer MLP with LayerNorm rather than a bare linear projection
    because:
      - BERT space and our SP embedding space have different statistical properties
      - The non-linearity allows the projector to learn a more complex mapping
      - LayerNorm prevents scale mismatches from disrupting early training
    """

    def __init__(self, bert_dim: int = 768, embed_dim: int = 256):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(bert_dim, embed_dim * 2),
            nn.LayerNorm(embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


def prepare_bert_embeddings_for_model(
    sp_vocab: List[str],
    bert_model_name: str,
    embed_dim: int,
    device: str = "cpu",
    save_path: Optional[str] = None,
) -> torch.Tensor:
    """
    Full pipeline: extract BERT embeddings → project to embed_dim.

    Args:
        sp_vocab: SentencePiece vocabulary list
        bert_model_name: HuggingFace model name
        embed_dim: target embedding dimension for the Seq2Seq model
        device: computation device
        save_path: if provided, save the projected embeddings here (for caching)

    Returns:
        projected_embeddings: (vocab_size, embed_dim) — ready to load into
        nn.Embedding.weight
    """
    # Check cache
    if save_path:
        import os
        if os.path.exists(save_path):
            print(f"Loading cached embeddings from {save_path}")
            return torch.load(save_path, map_location="cpu")

    bert_embs = extract_bert_embeddings_for_vocab(sp_vocab, bert_model_name, device)

    # Project to target dim
    projector = BERTEmbeddingProjector(bert_dim=bert_embs.size(-1), embed_dim=embed_dim)
    projector.eval()
    with torch.no_grad():
        projected = projector(bert_embs.to(device)).cpu()

    # L2-normalize (common practice for embedding initialization)
    projected = nn.functional.normalize(projected, p=2, dim=-1)

    if save_path:
        torch.save(projected, save_path)
        print(f"Cached embeddings saved to {save_path}")

    return projected   # (vocab_size, embed_dim)


def get_sp_vocab_list(sp_model_path: str) -> List[str]:
    """Return the list of vocabulary pieces from a SentencePiece model."""
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor()
    sp.Load(sp_model_path)
    return [sp.IdToPiece(i) for i in range(sp.GetPieceSize())]
