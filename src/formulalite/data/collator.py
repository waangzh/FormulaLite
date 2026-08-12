"""The single owner of image preprocessing, token padding, and target alignment."""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import numpy as np
import torch

from formulalite.processor import FormulaImageProcessor

from .normalizer import normalize
from .schema import FormulaBatch, FormulaSample
from .tokenizer import FormulaTokenizer

LABEL_IGNORE_INDEX = -100


class FormulaCollator:
    """Create the frozen decoder-input/next-token-target batch contract."""

    def __init__(
        self,
        tokenizer: FormulaTokenizer,
        image_processor: FormulaImageProcessor | None = None,
        *,
        max_sequence_length: int = 1027,
    ) -> None:
        if max_sequence_length < 2:
            raise ValueError("max_sequence_length must fit at least BOS and EOS")
        self.tokenizer = tokenizer
        self.image_processor = image_processor or FormulaImageProcessor()
        self.max_sequence_length = max_sequence_length

    def __call__(self, samples: Sequence[FormulaSample]) -> FormulaBatch:
        if not samples:
            raise ValueError("FormulaCollator requires a non-empty sample sequence")

        normalized = tuple(
            sample.normalized_latex
            if sample.normalized_latex is not None
            else normalize(sample.raw_latex)
            for sample in samples
        )
        token_sequences = [self.tokenizer.encode(text) for text in normalized]
        for sample, token_ids in zip(samples, token_sequences, strict=True):
            if len(token_ids) > self.max_sequence_length:
                raise ValueError(
                    f"sample {sample.id!r} has tokenized length {len(token_ids)}, exceeding "
                    f"collator max_sequence_length={self.max_sequence_length}"
                )

        backend = self.tokenizer.tokenizer
        bos_id = cast(int, backend.bos_token_id)
        pad_id = cast(int, backend.pad_token_id)
        eos_id = cast(int, backend.eos_token_id)
        if (bos_id, pad_id, eos_id) != (0, 1, 2):
            raise RuntimeError("FormulaLite decoder requires BOS/PAD/EOS IDs 0/1/2")
        if any(ids[0] != bos_id or ids[-1] != eos_id for ids in token_sequences):
            raise RuntimeError("FormulaTokenizer must add exactly the declared BOS/EOS boundaries")

        batch_size = len(samples)
        padded_length = max(map(len, token_sequences))
        decoder_input_ids = torch.full(
            (batch_size, padded_length), pad_id, dtype=torch.long
        )
        decoder_attention_mask = torch.zeros(
            (batch_size, padded_length), dtype=torch.long
        )
        labels = torch.full(
            (batch_size, padded_length), LABEL_IGNORE_INDEX, dtype=torch.long
        )
        for row, token_ids in enumerate(token_sequences):
            length = len(token_ids)
            decoder_input_ids[row, :length] = torch.tensor(token_ids, dtype=torch.long)
            decoder_attention_mask[row, :length] = 1
            # FormulaLite supplies already-aligned next-token targets. Future model code
            # must not ask a third-party wrapper to shift these labels a second time.
            labels[row, : length - 1] = torch.tensor(token_ids[1:], dtype=torch.long)

        processed = [self.image_processor.preprocess(sample.image) for sample in samples]
        pixel_values = torch.from_numpy(np.stack(processed)).to(dtype=torch.float32)
        spec = self.image_processor.spec
        expected_shape = (batch_size, spec.channels, spec.image_height, spec.image_width)
        if tuple(pixel_values.shape) != expected_shape:
            raise RuntimeError(
                "FormulaImageProcessor returned "
                f"{tuple(pixel_values.shape)}, expected {expected_shape}"
            )

        return FormulaBatch(
            pixel_values=pixel_values,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
            sample_ids=tuple(sample.id for sample in samples),
            subsets=tuple(sample.subset for sample in samples),
            sources=tuple(sample.source for sample in samples),
            raw_latex=tuple(sample.raw_latex for sample in samples),
            normalized_latex=normalized,
        )
