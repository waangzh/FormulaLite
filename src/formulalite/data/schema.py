"""Canonical storage, decoded-sample, and batch schemas for FormulaLite."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

from PIL import Image
from torch import Tensor, device

from formulalite.processor.image import ImageInput

ImageReference: TypeAlias = str | Path | bytes | bytearray


@dataclass(frozen=True)
class FormulaRecord:
    """Storage-facing formula record independent of a dataset directory layout."""

    id: str
    image_ref: ImageReference
    raw_latex: str
    subset: str | None = None
    source: str | None = None

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("record id must not be empty")
        if not isinstance(self.raw_latex, str):
            raise TypeError("raw_latex must be a string")


@dataclass(frozen=True)
class FormulaSample:
    """Decoded sample passed from a dataset to the collator."""

    id: str
    image: ImageInput
    raw_latex: str
    normalized_latex: str | None
    subset: str | None = None
    source: str | None = None
    normalizer_version: str | None = None


@dataclass(frozen=True)
class FormulaBatch:
    """Tensor inputs plus non-forward diagnostic metadata."""

    pixel_values: Tensor
    decoder_input_ids: Tensor
    decoder_attention_mask: Tensor
    labels: Tensor
    sample_ids: tuple[str, ...]
    subsets: tuple[str | None, ...]
    sources: tuple[str | None, ...]
    raw_latex: tuple[str, ...]
    normalized_latex: tuple[str, ...]

    def to(self, target: device) -> FormulaBatch:
        """Return an immutable batch copy with only model tensors moved to a device."""

        return FormulaBatch(
            pixel_values=self.pixel_values.to(target),
            decoder_input_ids=self.decoder_input_ids.to(target),
            decoder_attention_mask=self.decoder_attention_mask.to(target),
            labels=self.labels.to(target),
            sample_ids=self.sample_ids,
            subsets=self.subsets,
            sources=self.sources,
            raw_latex=self.raw_latex,
            normalized_latex=self.normalized_latex,
        )


DecodedImage: TypeAlias = Image.Image
