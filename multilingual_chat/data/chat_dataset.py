"""
Dataset and Tokenizer for Multilingual Generative Dialogues.

Implements:
    - SPTokenizer trained jointly with custom chat control tokens
    - MultilingualChatDataset with prompt-loss masking (masking user turns with -100)
    - macOS picklable collator for stable spawn-based multiprocessing
"""

import os
import re
import random
import unicodedata
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import torch
from torch.utils.data import Dataset, DataLoader
import sentencepiece as spm


# ─── Unicode Cleaning ─────────────────────────────────────────────────────────

def normalize_chat_text(text: str) -> str:
    """NFC normalization for Devanagari, Urdu, and regional scripts."""
    text = unicodedata.normalize("NFC", text.strip())
    # Clean zero-width space characters
    text = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", text)
    # Collapse consecutive spaces
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


# ─── Joint Conversational Tokenizer ──────────────────────────────────────────

class SPChatTokenizer:
    """
    SentencePiece BPE Tokenizer customized for Conversational AI.

    Control tokens:
        <pad> = 0
        <unk> = 1
        <bos> = 2
        <eos> = 3
        <user> = 4
        <assistant> = 5
        <end_turn> = 6
    """
    PAD_ID = 0
    UNK_ID = 1
    BOS_ID = 2
    EOS_ID = 3
    USER_ID = 4
    ASSISTANT_ID = 5
    END_TURN_ID = 6

    SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>", "<user>", "<assistant>", "<end_turn>"]

    def __init__(self, model_path: str):
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(model_path)
        self.model_path = model_path

    @classmethod
    def train(
        cls,
        sentences: List[str],
        model_prefix: str,
        vocab_size: int = 8000,
        character_coverage: float = 0.9999,
    ) -> "SPChatTokenizer":
        """Train a joint conversational BPE SentencePiece model."""
        tmp_path = model_prefix + "_train_chat_tmp.txt"
        with open(tmp_path, "w", encoding="utf-8") as f:
            for s in sentences:
                f.write(s + "\n")

        spm.SentencePieceTrainer.Train(
            input=tmp_path,
            model_prefix=model_prefix,
            vocab_size=vocab_size,
            character_coverage=character_coverage,
            model_type="bpe",
            pad_id=cls.PAD_ID,
            unk_id=cls.UNK_ID,
            bos_id=cls.BOS_ID,
            eos_id=cls.EOS_ID,
            user_defined_symbols="<user>,<assistant>,<end_turn>",
            shuffle_input_sentence=True,
            num_threads=4,
        )
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        print(f"Conversational Tokenizer trained and saved: {model_prefix}.model")
        return cls(model_prefix + ".model")

    def encode(self, text: str) -> List[int]:
        return self.sp.EncodeAsIds(text)

    def decode(self, ids: List[int]) -> str:
        # Filter out system conversational markers during final output decoding
        ids = [i for i in ids if i not in (self.PAD_ID, self.BOS_ID, self.EOS_ID, self.USER_ID, self.ASSISTANT_ID, self.END_TURN_ID)]
        return self.sp.DecodeIds(ids)

    @property
    def vocab_size(self) -> int:
        return self.sp.GetPieceSize()


# ─── Chat Dataset ────────────────────────────────────────────────────────────

class MultilingualChatDataset(Dataset):
    """
    Torch Dataset formatting multi-turn conversational sequences.
    
    Format:
        [BOS, <user>, ...user_tokens..., <end_turn>, <assistant>, ...reply_tokens..., EOS]

    Prompt Masking:
        Targets are constructed where all user turn tokens are set to -100.
        This forces PyTorch CrossEntropyLoss to ignore user tokens, so gradients are
        only updated based on the model's capacity to generate assistant replies.
    """
    def __init__(
        self,
        dialogues: List[Tuple[str, str]],  # (user_prompt, assistant_response)
        tokenizer: SPChatTokenizer,
        max_seq_len: int = 256,
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.input_ids: List[List[int]] = []
        self.target_ids: List[List[int]] = []

        for user_text, assistant_text in dialogues:
            u_ids = tokenizer.encode(normalize_chat_text(user_text))
            a_ids = tokenizer.encode(normalize_chat_text(assistant_text))

            # seq = [BOS, <user>] + u_ids + [<end_turn>, <assistant>] + a_ids + [EOS]
            seq = (
                [tokenizer.BOS_ID, tokenizer.USER_ID]
                + u_ids
                + [tokenizer.END_TURN_ID, tokenizer.ASSISTANT_ID]
                + a_ids
                + [tokenizer.EOS_ID]
            )

            if len(seq) > max_seq_len:
                # Keep prompt context and truncate long responses if they exceed seq window
                seq = seq[:max_seq_len]

            # Shift targets to the left by 1 for autoregressive modeling
            inp = seq[:-1]
            tgt = seq[1:]

            # Mask out the user prompt tokens by setting target values to -100
            # User turn spans from start up to index: len(u_ids) + 3 in the target sequence
            # (which aligns with the ASSISTANT control token in targets)
            mask_boundary = len(u_ids) + 3
            masked_tgt = []
            for idx, token_val in enumerate(tgt):
                if idx < mask_boundary:
                    masked_tgt.append(-100)
                else:
                    masked_tgt.append(token_val)

            self.input_ids.append(inp)
            self.target_ids.append(masked_tgt)

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.tensor(self.input_ids[idx], dtype=torch.long),
            torch.tensor(self.target_ids[idx], dtype=torch.long),
        )


# ─── Picklable Collator for macOS Spawn Multiprocessing ──────────────────────

class ChatPadCollator:
    """Picklable collator padding input sequences with pad_id, and targets with -100."""
    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
        inputs, targets = zip(*batch)
        max_len = max(x.size(0) for x in inputs)

        B = len(inputs)
        padded_inputs = torch.full((B, max_len), self.pad_id, dtype=torch.long)
        padded_targets = torch.full((B, max_len), -100, dtype=torch.long)

        for i, (inp, tgt) in enumerate(zip(inputs, targets)):
            padded_inputs[i, :inp.size(0)] = inp
            padded_targets[i, :tgt.size(0)] = tgt

        return padded_inputs, padded_targets


def make_chat_dataloader(
    dataset: MultilingualChatDataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 2,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=ChatPadCollator(SPChatTokenizer.PAD_ID),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
