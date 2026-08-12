"""FormulaLite's deterministic compact WordLevel tokenizer."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from tokenizers import Tokenizer, models, processors
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import PreTrainedTokenizerFast

from .normalizer import NORMALIZER_VERSION, baseline_tokens, normalize

BOS_TOKEN: Final = "<bos>"
PAD_TOKEN: Final = "<pad>"
EOS_TOKEN: Final = "<eos>"
UNK_TOKEN: Final = "<unk>"
SPECIAL_TOKENS: Final = (BOS_TOKEN, PAD_TOKEN, EOS_TOKEN, UNK_TOKEN)
SPECIAL_TOKEN_IDS: Final = {BOS_TOKEN: 0, PAD_TOKEN: 1, EOS_TOKEN: 2, UNK_TOKEN: 3}
VOCAB_SIZE: Final = 687
TOKENIZER_ARTIFACT_VERSION: Final = "1.0.0"

def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_vocab_bytes(tokens: Sequence[str]) -> bytes:
    return ("\n".join(tokens) + "\n").encode()


def _artifact_sha256(directory: Path) -> str:
    digest = hashlib.sha256()
    for name in ("special_tokens_map.json", "tokenizer.json", "tokenizer_config.json"):
        payload = (directory / name).read_bytes()
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(payload)
        digest.update(b"\0")
    return digest.hexdigest()


def build_tokenizer() -> PreTrainedTokenizerFast:
    """Build the fixed 687-token WordLevel tokenizer in memory."""

    ordered_tokens = (*SPECIAL_TOKENS, *baseline_tokens())
    if len(ordered_tokens) != VOCAB_SIZE or len(set(ordered_tokens)) != VOCAB_SIZE:
        msg = "FormulaLite vocabulary must contain exactly 687 unique tokens"
        raise RuntimeError(msg)

    vocabulary = {token: token_id for token_id, token in enumerate(ordered_tokens)}
    backend = Tokenizer(models.WordLevel(vocabulary, unk_token=UNK_TOKEN))
    backend.pre_tokenizer = WhitespaceSplit()
    backend.post_processor = processors.TemplateProcessing(
        single=f"{BOS_TOKEN} $A {EOS_TOKEN}",
        pair=f"{BOS_TOKEN} $A {EOS_TOKEN} $B {EOS_TOKEN}",
        special_tokens=[(BOS_TOKEN, 0), (EOS_TOKEN, 2)],
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        bos_token=BOS_TOKEN,
        pad_token=PAD_TOKEN,
        eos_token=EOS_TOKEN,
        unk_token=UNK_TOKEN,
        model_max_length=1027,
        clean_up_tokenization_spaces=False,
    )
    _validate_tokenizer(tokenizer)
    return tokenizer


def _validate_tokenizer(tokenizer: PreTrainedTokenizerFast) -> None:
    if len(tokenizer) != VOCAB_SIZE:
        msg = f"unexpected vocabulary size: {len(tokenizer)}"
        raise ValueError(msg)
    actual = {token: tokenizer.convert_tokens_to_ids(token) for token in SPECIAL_TOKENS}
    if actual != SPECIAL_TOKEN_IDS:
        msg = f"unexpected special-token mapping: {actual}"
        raise ValueError(msg)


@dataclass(frozen=True)
class FormulaTokenizer:
    """Small normalizing facade over a Hugging Face fast tokenizer artifact."""

    tokenizer: PreTrainedTokenizerFast

    @classmethod
    def build(cls) -> FormulaTokenizer:
        return cls(build_tokenizer())

    @classmethod
    def load_pretrained(cls, directory: str | Path) -> FormulaTokenizer:
        tokenizer = PreTrainedTokenizerFast.from_pretrained(directory)
        _validate_tokenizer(tokenizer)
        return cls(tokenizer)

    def encode(self, text: str, *, add_special_tokens: bool = True) -> list[int]:
        return self.tokenizer.encode(normalize(text), add_special_tokens=add_special_tokens)

    def decode(self, token_ids: Iterable[int], *, skip_special_tokens: bool = True) -> str:
        return self.tokenizer.decode(
            list(token_ids),
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=False,
        )

    def save_pretrained(self, directory: str | Path) -> dict[str, Any]:
        return save_pretrained(self.tokenizer, directory)

    @property
    def vocab_size(self) -> int:
        return len(self.tokenizer)


def save_pretrained(tokenizer: PreTrainedTokenizerFast, directory: str | Path) -> dict[str, Any]:
    """Save a deterministic Hugging Face artifact and contract manifest."""

    _validate_tokenizer(tokenizer)
    output = Path(directory)
    output.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(output)

    ordered_tokens = [""] * VOCAB_SIZE
    for token, token_id in tokenizer.get_vocab().items():
        ordered_tokens[token_id] = token
    if any(not token for token in ordered_tokens):
        raise ValueError("tokenizer vocabulary IDs must be contiguous")
    manifest: dict[str, Any] = {
        "artifact_version": TOKENIZER_ARTIFACT_VERSION,
        "vocab_size": VOCAB_SIZE,
        "special_tokens": {
            "bos_token": {"token": BOS_TOKEN, "id": 0},
            "pad_token": {"token": PAD_TOKEN, "id": 1},
            "eos_token": {"token": EOS_TOKEN, "id": 2},
            "unk_token": {"token": UNK_TOKEN, "id": 3},
        },
        "normalizer_version": NORMALIZER_VERSION,
        "vocab_sha256": _sha256(_canonical_vocab_bytes(ordered_tokens)),
        "artifact_sha256": _artifact_sha256(output),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def load_pretrained(directory: str | Path) -> FormulaTokenizer:
    return FormulaTokenizer.load_pretrained(directory)
