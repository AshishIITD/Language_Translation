"""
Dataset loading and preprocessing for Hindi-Marathi parallel corpus.
Supports SentencePiece BPE tokenization with shared/separate vocabularies.
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


# ─── Text Cleaning ────────────────────────────────────────────────────────────

def normalize_unicode(text: str) -> str:
    """NFC normalize unicode — critical for Devanagari."""
    return unicodedata.normalize("NFC", text.strip())


def clean_text(text: str) -> str:
    """
    Clean a single line of Hindi or Marathi text.
    - Normalize unicode
    - Collapse whitespace
    - Remove zero-width characters
    - Strip leading/trailing spaces
    """
    text = normalize_unicode(text)
    # Remove zero-width non-joiner, zero-width joiner, etc.
    text = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", text)
    # Collapse multiple spaces/tabs
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def _numeric_latin_ratio(text: str) -> float:
    """Return fraction of characters that are Latin letters or digits."""
    if not text:
        return 0.0
    count = sum(1 for ch in text if ch.isascii() and (ch.isalpha() or ch.isdigit()))
    return count / len(text)


def filter_pair(src: str, tgt: str,
                min_len: int = 3,
                max_len: int = 150,
                max_ratio: float = 3.0,
                max_numeric_latin_ratio: float = 0.20) -> bool:
    """
    Return True if the sentence pair should be KEPT.
    Filters:
      - Too short or too long (by token count, rough split on spaces)
      - Extreme length ratio (likely misaligned)
      - Empty lines
      - Copy pairs (source == target, adds no translation signal)
      - High numeric/Latin content (>20% — likely misaligned or code-mixed)
    """
    if not src or not tgt:
        return False

    # Skip copy pairs — identical source and target add no translation signal
    if src.strip() == tgt.strip():
        return False

    src_len = len(src.split())
    tgt_len = len(tgt.split())
    if src_len < min_len or tgt_len < min_len:
        return False
    if src_len > max_len or tgt_len > max_len:
        return False
    ratio = max(src_len, tgt_len) / max(min(src_len, tgt_len), 1)
    if ratio > max_ratio:
        return False

    # Skip pairs with too many numeric/Latin characters (likely misaligned)
    if (_numeric_latin_ratio(src) > max_numeric_latin_ratio or
            _numeric_latin_ratio(tgt) > max_numeric_latin_ratio):
        return False

    return True


# ─── Corpus Loading ───────────────────────────────────────────────────────────

def load_parallel_corpus(
    data_dir: str,
    src_lang: str = "hi",
    tgt_lang: str = "mr",
    max_samples: Optional[int] = None,
    seed: int = 42,
) -> Tuple[List[str], List[str]]:
    """
    Load a parallel corpus from a directory.

    Expected file naming conventions (any of):
      - {src_lang}.txt / {tgt_lang}.txt
      - train.{src_lang} / train.{tgt_lang}
      - corpus.hi / corpus.mr
      - A single TSV file with two columns

    Returns:
        (src_sentences, tgt_sentences) — cleaned, filtered, parallel lists.
    """
    data_dir = Path(data_dir)
    src_lines, tgt_lines = [], []

    # Try common file patterns
    patterns = [
        (data_dir / f"{src_lang}.txt", data_dir / f"{tgt_lang}.txt"),
        (data_dir / f"train.{src_lang}", data_dir / f"train.{tgt_lang}"),
        (data_dir / f"corpus.{src_lang}", data_dir / f"corpus.{tgt_lang}"),
        (data_dir / f"hi-mr.{src_lang}", data_dir / f"hi-mr.{tgt_lang}"),
    ]

    loaded = False
    for src_path, tgt_path in patterns:
        if src_path.exists() and tgt_path.exists():
            with open(src_path, "r", encoding="utf-8") as f:
                src_lines = f.read().splitlines()
            with open(tgt_path, "r", encoding="utf-8") as f:
                tgt_lines = f.read().splitlines()
            loaded = True
            print(f"Loaded from: {src_path.name} / {tgt_path.name}")
            break

    # Try TSV
    if not loaded:
        for tsv_path in data_dir.glob("*.tsv"):
            with open(tsv_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) >= 2:
                        src_lines.append(parts[0])
                        tgt_lines.append(parts[1])
            if src_lines:
                loaded = True
                print(f"Loaded from TSV: {tsv_path.name}")
                break

    if not loaded:
        raise FileNotFoundError(
            f"Could not find parallel corpus in {data_dir}. "
            f"Expected files like {src_lang}.txt / {tgt_lang}.txt or a .tsv file."
        )

    assert len(src_lines) == len(tgt_lines), (
        f"Source ({len(src_lines)}) and target ({len(tgt_lines)}) line counts differ!"
    )

    # Clean
    src_lines = [clean_text(s) for s in src_lines]
    tgt_lines = [clean_text(t) for t in tgt_lines]

    # Filter
    pairs = [(s, t) for s, t in zip(src_lines, tgt_lines)
             if filter_pair(s, t)]
    print(f"After filtering: {len(pairs):,} / {len(src_lines):,} pairs retained")

    # Optional subsampling
    if max_samples and max_samples < len(pairs):
        rng = random.Random(seed)
        pairs = rng.sample(pairs, max_samples)
        print(f"Subsampled to {max_samples:,} pairs")

    src_out, tgt_out = zip(*pairs)
    return list(src_out), list(tgt_out)


def split_corpus(
    src: List[str],
    tgt: List[str],
    train_ratio: float = 0.90,
    val_ratio: float = 0.05,
    seed: int = 42,
) -> Tuple[Tuple[List[str], List[str]], Tuple[List[str], List[str]], Tuple[List[str], List[str]]]:
    """Split into train / val / test."""
    n = len(src)
    indices = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(indices)

    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    def pick(idx_list):
        s = [src[i] for i in idx_list]
        t = [tgt[i] for i in idx_list]
        return s, t

    train = pick(indices[:train_end])
    val = pick(indices[train_end:val_end])
    test = pick(indices[val_end:])
    print(f"Split → train: {len(train[0]):,}  val: {len(val[0]):,}  test: {len(test[0]):,}")
    return train, val, test


# ─── SentencePiece Tokenizer ──────────────────────────────────────────────────

class SPTokenizer:
    """
    Wrapper around SentencePiece for shared Hindi+Marathi BPE vocabulary.

    Using a SHARED vocabulary is well-motivated for Hindi↔Marathi because:
    - Both use Devanagari script → massive character overlap
    - Many root words are shared (they are related Indo-Aryan languages)
    - Shared vocab reduces total vocabulary size, improving coverage
    - Enables zero-shot cross-lingual transfer within subword space

    Special tokens:
        <pad>=0, <unk>=1, <bos>=2, <eos>=3
        <hi>=4  (language tag for Hindi)
        <mr>=5  (language tag for Marathi)
    """

    PAD_ID = 0
    UNK_ID = 1
    BOS_ID = 2
    EOS_ID = 3
    HI_ID = 4
    MR_ID = 5

    SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>", "<hi>", "<mr>"]

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
        model_type: str = "bpe",
    ) -> "SPTokenizer":
        """Train a SentencePiece model on a list of sentences."""
        # Write sentences to a temp file
        tmp_path = model_prefix + "_train_tmp.txt"
        with open(tmp_path, "w", encoding="utf-8") as f:
            for s in sentences:
                f.write(s + "\n")

        special_str = ",".join(cls.SPECIAL_TOKENS)
        spm.SentencePieceTrainer.Train(
            input=tmp_path,
            model_prefix=model_prefix,
            vocab_size=vocab_size,
            character_coverage=character_coverage,
            model_type=model_type,
            pad_id=cls.PAD_ID,
            unk_id=cls.UNK_ID,
            bos_id=cls.BOS_ID,
            eos_id=cls.EOS_ID,
            user_defined_symbols="<hi>,<mr>",
            shuffle_input_sentence=True,
            num_threads=4,
        )
        os.remove(tmp_path)
        print(f"SentencePiece model saved: {model_prefix}.model  (vocab={vocab_size})")
        return cls(model_prefix + ".model")

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = True) -> List[int]:
        ids = self.sp.EncodeAsIds(text)
        if add_bos:
            ids = [self.BOS_ID] + ids
        if add_eos:
            ids = ids + [self.EOS_ID]
        return ids

    def decode(self, ids: List[int]) -> str:
        # Strip special token ids before decoding
        ids = [i for i in ids if i not in (self.PAD_ID, self.BOS_ID, self.EOS_ID)]
        return self.sp.DecodeIds(ids)

    @property
    def vocab_size(self) -> int:
        return self.sp.GetPieceSize()


# ─── PyTorch Dataset ──────────────────────────────────────────────────────────

class TranslationDataset(Dataset):
    """
    Torch Dataset for parallel translation pairs.
    Each item is (src_ids, tgt_ids) as lists of ints (no padding yet).
    Padding is handled by the collate_fn for efficiency.
    """

    def __init__(
        self,
        src_sentences: List[str],
        tgt_sentences: List[str],
        tokenizer: SPTokenizer,
        max_src_len: int = 128,
        max_tgt_len: int = 128,
    ):
        assert len(src_sentences) == len(tgt_sentences)
        self.tokenizer = tokenizer
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len

        self.src_ids: List[List[int]] = []
        self.tgt_ids: List[List[int]] = []

        for src, tgt in zip(src_sentences, tgt_sentences):
            s = tokenizer.encode(src, add_bos=False, add_eos=True)[:max_src_len]
            t = tokenizer.encode(tgt, add_bos=True, add_eos=True)[:max_tgt_len]
            self.src_ids.append(s)
            self.tgt_ids.append(t)

    def __len__(self):
        return len(self.src_ids)

    def __getitem__(self, idx):
        return self.src_ids[idx], self.tgt_ids[idx]


def collate_fn(batch, pad_id: int = 0):
    """
    Collate a batch of (src_ids, tgt_ids) pairs.
    Pads sequences to the max length in the batch.
    Returns:
        src: (B, S)  LongTensor
        tgt: (B, T)  LongTensor
        src_lengths: (B,) LongTensor — actual lengths (for pack_padded_sequence)
    """
    src_batch, tgt_batch = zip(*batch)

    src_lengths = torch.tensor([len(s) for s in src_batch], dtype=torch.long)
    max_src = src_lengths.max().item()
    max_tgt = max(len(t) for t in tgt_batch)

    B = len(src_batch)
    src_tensor = torch.full((B, max_src), pad_id, dtype=torch.long)
    tgt_tensor = torch.full((B, max_tgt), pad_id, dtype=torch.long)

    for i, (s, t) in enumerate(zip(src_batch, tgt_batch)):
        src_tensor[i, :len(s)] = torch.tensor(s, dtype=torch.long)
        tgt_tensor[i, :len(t)] = torch.tensor(t, dtype=torch.long)

    return src_tensor, tgt_tensor, src_lengths


class PadCollate:
    """Picklable collate function wrapper for macOS multiprocessing."""
    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, batch):
        return collate_fn(batch, pad_id=self.pad_id)


def make_dataloader(
    dataset: TranslationDataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 2,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=PadCollate(SPTokenizer.PAD_ID),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )


# ─── Corpus Stats ─────────────────────────────────────────────────────────────

def print_corpus_stats(src: List[str], tgt: List[str], name: str = ""):
    src_lens = [len(s.split()) for s in src]
    tgt_lens = [len(t.split()) for t in tgt]
    print(f"\n{'─'*50}")
    print(f"Corpus stats: {name}")
    print(f"  Pairs     : {len(src):,}")
    print(f"  Src len   : mean={sum(src_lens)/len(src_lens):.1f}  max={max(src_lens)}  min={min(src_lens)}")
    print(f"  Tgt len   : mean={sum(tgt_lens)/len(tgt_lens):.1f}  max={max(tgt_lens)}  min={min(tgt_lens)}")
    print(f"{'─'*50}\n")
