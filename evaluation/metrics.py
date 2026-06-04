"""
Evaluation metrics for NMT: BLEU-100 and CHRF++-100.

Both metrics are reported on a 0–100 scale as required by the assignment.

BLEU (Papineni et al., 2002):
    Modified n-gram precision with brevity penalty.
    We use sacrebleu for standardized, reproducible BLEU computation.
    BLEU-100 = BLEU * 100

CHRF++ (Popović, 2017):
    Character n-gram F-score with word unigrams and bigrams added.
    More robust than BLEU for morphologically rich languages like
    Hindi/Marathi because it evaluates at character level — partial
    credit for morphological variants is captured.
    CHRF++-100 = CHRF++ * 100
"""

import torch
from typing import List, Tuple
from sacrebleu.metrics import BLEU, CHRF


# ─── Metric Wrappers ──────────────────────────────────────────────────────────

_BLEU_SCORER = BLEU(effective_order=True)
_CHRF_SCORER = CHRF(word_order=2)   # word_order=2 → CHRF++


def compute_bleu(hypotheses: List[str], references: List[str]) -> float:
    """
    Compute corpus-level BLEU score (0–100).

    Args:
        hypotheses: list of model-generated translations
        references: list of reference translations (same order)
    Returns:
        BLEU score in [0, 100]
    """
    assert len(hypotheses) == len(references), (
        f"hypotheses ({len(hypotheses)}) and references ({len(references)}) must match"
    )
    # sacrebleu expects [[ref1, ref2, ...]] per sentence → wrap in list
    refs = [[r] for r in references]
    result = _BLEU_SCORER.corpus_score(hypotheses, list(zip(*refs)) if refs else [[]])
    return result.score


def compute_chrf_pp(hypotheses: List[str], references: List[str]) -> float:
    """
    Compute corpus-level CHRF++ score (0–100).
    CHRF++ = CHRF with word n-gram order 2 (adds word unigrams and bigrams).
    """
    assert len(hypotheses) == len(references)
    refs = [[r] for r in references]
    result = _CHRF_SCORER.corpus_score(hypotheses, list(zip(*refs)) if refs else [[]])
    return result.score


def compute_metrics(hypotheses: List[str], references: List[str]) -> dict:
    """Compute both BLEU-100 and CHRF++-100."""
    return {
        "bleu": compute_bleu(hypotheses, references),
        "chrf_pp": compute_chrf_pp(hypotheses, references),
    }


# ─── Model Evaluation Loop ────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_model(
    model,
    dataloader,
    tokenizer,
    device: torch.device,
    use_beam: bool = False,
    beam_size: int = 4,
    max_decode_len: int = 150,
) -> Tuple[float, float, float]:
    """
    Run model on a dataloader and compute BLEU + CHRF++.
    Also returns mean loss if model supports it.

    Returns:
        (loss, bleu, chrf_pp) all as floats
    """
    import torch.nn as nn
    model.eval()

    criterion = nn.CrossEntropyLoss(
        ignore_index=tokenizer.PAD_ID,
        label_smoothing=0.1,
    )

    hypotheses = []
    references_text = []
    total_loss = 0.0
    total_tokens = 0
    n_batches = 0

    for src, tgt, src_lengths in dataloader:
        src = src.to(device)
        tgt = tgt.to(device)
        src_lengths = src_lengths.to(device)

        # Forward pass for loss
        logits = model(src, tgt, src_lengths, teacher_forcing_ratio=1.0)
        # logits: (B, T-1, V)   tgt_out: (B, T-1)
        tgt_out = tgt[:, 1:]    # remove <bos>
        B, T, V = logits.size()
        loss = criterion(logits.reshape(B * T, V), tgt_out.reshape(B * T))

        non_pad = (tgt_out != tokenizer.PAD_ID).sum().item()
        total_loss += loss.item() * non_pad
        total_tokens += non_pad
        n_batches += 1

        # Generate translations
        if use_beam:
            for i in range(B):
                single_src = src[i:i+1]
                single_len = src_lengths[i:i+1]
                hyp_ids = model.beam_search(
                    single_src, single_len,
                    bos_id=tokenizer.BOS_ID,
                    eos_id=tokenizer.EOS_ID,
                    beam_size=beam_size,
                    max_len=max_decode_len,
                )
                hypotheses.append(tokenizer.decode(hyp_ids))
        else:
            pred_ids = model.greedy_decode(
                src, src_lengths,
                bos_id=tokenizer.BOS_ID,
                eos_id=tokenizer.EOS_ID,
                max_len=max_decode_len,
            )
            for i in range(B):
                hypotheses.append(tokenizer.decode(pred_ids[i].tolist()))

        # Reference text (decode from tgt, skip <bos>)
        for i in range(B):
            ref_ids = tgt[i, 1:].tolist()
            references_text.append(tokenizer.decode(ref_ids))

    avg_loss = total_loss / max(total_tokens, 1)
    metrics = compute_metrics(hypotheses, references_text)
    return avg_loss, metrics["bleu"], metrics["chrf_pp"]
