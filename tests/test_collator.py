import numpy as np
import torch
from PIL import Image

from formulalite.data.collator import FormulaCollator
from formulalite.data.normalizer import normalize
from formulalite.data.schema import FormulaSample
from formulalite.data.tokenizer import FormulaTokenizer


def sample(sample_id: str, latex: str, subset: str = "SPE") -> FormulaSample:
    pixels = np.full((12, 24, 3), 255, dtype=np.uint8)
    pixels[4:8, 6:18] = 0
    return FormulaSample(
        id=sample_id,
        image=Image.fromarray(pixels),
        raw_latex=latex,
        normalized_latex=normalize(latex),
        subset=subset,
        source="unit",
        normalizer_version="1.0.0",
    )


def test_decoder_target_alignment_contract() -> None:
    # Input decoder_input_ids = [BOS, x1, x2, EOS, PAD]
    # Expected labels          = [x1,  x2, EOS, -100, -100]
    tokenizer = FormulaTokenizer.build()
    collator = FormulaCollator(tokenizer)
    batch = collator([sample("short", "x"), sample("long", "x + y")])
    short_ids = tokenizer.encode("x")
    long_ids = tokenizer.encode("x + y")

    assert batch.decoder_input_ids.tolist() == [
        [*short_ids, 1, 1],
        long_ids,
    ]
    assert batch.decoder_attention_mask.tolist() == [
        [1, 1, 1, 0, 0],
        [1, 1, 1, 1, 1],
    ]
    assert batch.labels.tolist() == [
        [short_ids[1], 2, -100, -100, -100],
        [*long_ids[1:], -100],
    ]


def test_single_and_multi_token_bos_eos_semantics() -> None:
    tokenizer = FormulaTokenizer.build()
    batch = FormulaCollator(tokenizer)([sample("one", "x"), sample("many", "x + y = 2")])
    for row, latex in enumerate(("x", "x + y = 2")):
        ids = tokenizer.encode(latex)
        assert batch.decoder_input_ids[row, 0].item() == 0
        assert batch.decoder_input_ids[row, len(ids) - 1].item() == 2
        assert batch.labels[row, len(ids) - 2].item() == 2
        assert batch.labels[row, len(ids) - 1].item() == -100


def test_pixel_values_and_metadata_contract() -> None:
    batch = FormulaCollator(FormulaTokenizer.build())(
        [sample("a", "x", "SPE"), sample("b", "x + y", "HWE")]
    )
    assert batch.pixel_values.shape == (2, 3, 384, 384)
    assert batch.pixel_values.dtype == torch.float32
    assert batch.sample_ids == ("a", "b")
    assert batch.subsets == ("SPE", "HWE")
    assert batch.sources == ("unit", "unit")
