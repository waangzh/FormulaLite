import json
from pathlib import Path

import pytest
from transformers import PreTrainedTokenizerFast

from formulalite.data.normalizer import baseline_tokens, normalize
from formulalite.data.tokenizer import (
    BOS_TOKEN,
    EOS_TOKEN,
    PAD_TOKEN,
    UNK_TOKEN,
    VOCAB_SIZE,
    FormulaTokenizer,
)

CURATED_FORMULAS = [
    r"x+y=2",
    r"\frac{a+b}{c-d}",
    r"\sqrt{x^2+y^2}",
    r"\alpha_i+\beta^2=\Gamma",
    r"\sum_{i=0}^{n}i",
    r"\begin{array}{cc}a&b\\c&d\end{array}",
    r"\left(\frac{x}{y}\right)",
]


@pytest.fixture(scope="module")
def tokenizer() -> FormulaTokenizer:
    return FormulaTokenizer.build()


def test_special_token_ids(tokenizer: FormulaTokenizer) -> None:
    backend = tokenizer.tokenizer
    assert backend.convert_tokens_to_ids(BOS_TOKEN) == 0
    assert backend.convert_tokens_to_ids(PAD_TOKEN) == 1
    assert backend.convert_tokens_to_ids(EOS_TOKEN) == 2
    assert backend.convert_tokens_to_ids(UNK_TOKEN) == 3
    assert backend.bos_token_id == 0
    assert backend.pad_token_id == 1
    assert backend.eos_token_id == 2
    assert backend.unk_token_id == 3


def test_vocab_size(tokenizer: FormulaTokenizer) -> None:
    assert tokenizer.vocab_size == VOCAB_SIZE == 687


@pytest.mark.parametrize("formula", CURATED_FORMULAS)
def test_encode_decode_round_trip(tokenizer: FormulaTokenizer, formula: str) -> None:
    encoded = tokenizer.encode(formula)
    assert encoded[0] == 0
    assert encoded[-1] == 2
    assert tokenizer.decode(encoded) == normalize(formula)


def test_unknown_token_behavior(tokenizer: FormulaTokenizer) -> None:
    encoded = tokenizer.encode(r"\definitelyunknown{x}")
    assert tokenizer.tokenizer.unk_token_id in encoded
    assert UNK_TOKEN in tokenizer.decode(encoded, skip_special_tokens=False)


def test_save_load_parity(tokenizer: FormulaTokenizer, tmp_path: Path) -> None:
    output = tmp_path / "tokenizer"
    tokenizer.save_pretrained(output)
    loaded = FormulaTokenizer.load_pretrained(output)
    for formula in CURATED_FORMULAS:
        assert loaded.encode(formula) == tokenizer.encode(formula)
        assert loaded.decode(loaded.encode(formula)) == normalize(formula)

    generic = PreTrainedTokenizerFast.from_pretrained(output)
    assert generic.encode(normalize(CURATED_FORMULAS[0])) == tokenizer.encode(CURATED_FORMULAS[0])


def test_deterministic_artifact_build(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    FormulaTokenizer.build().save_pretrained(first)
    FormulaTokenizer.build().save_pretrained(second)
    filenames = {
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "manifest.json",
    }
    assert {path.name for path in first.iterdir()} == filenames
    assert {path.name for path in second.iterdir()} == filenames
    for name in filenames:
        assert (first / name).read_bytes() == (second / name).read_bytes()


def test_vocabulary_id_contract(tokenizer: FormulaTokenizer) -> None:
    formulalite_mapping = {BOS_TOKEN: 0, PAD_TOKEN: 1, EOS_TOKEN: 2, UNK_TOKEN: 3}
    assert tokenizer.tokenizer.convert_ids_to_tokens([0, 1, 2, 3]) == list(formulalite_mapping)
    for token_id, token in enumerate(baseline_tokens(), start=4):
        assert tokenizer.tokenizer.convert_tokens_to_ids(token) == token_id


@pytest.mark.parametrize("formula", CURATED_FORMULAS)
def test_curated_formulas_have_no_unknown_tokens(
    tokenizer: FormulaTokenizer, formula: str
) -> None:
    assert tokenizer.tokenizer.unk_token_id not in tokenizer.encode(formula)


def test_manifest_records_required_contract(tokenizer: FormulaTokenizer, tmp_path: Path) -> None:
    manifest = tokenizer.save_pretrained(tmp_path)
    loaded = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert loaded == manifest
    assert loaded["vocab_size"] == 687
    assert loaded["normalizer_version"] == "1.0.0"
    assert len(loaded["vocab_sha256"]) == 64
    assert len(loaded["artifact_sha256"]) == 64
