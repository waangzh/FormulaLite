"""Map-style FormulaLite datasets and local manifest backend."""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping, Sequence
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image
from torch.utils.data import Dataset

from .normalizer import NORMALIZER_VERSION, normalize
from .policy import FormulaDataPolicy, FormulaDataStatistics, tokenizer_vocab_sha256
from .schema import FormulaRecord, FormulaSample, ImageReference


def record_from_mapping(value: Mapping[str, Any], *, base_dir: Path) -> FormulaRecord:
    try:
        image_value = value["image"]
        image_ref: ImageReference
        if isinstance(image_value, str):
            image_path = Path(image_value)
            image_ref = image_path if image_path.is_absolute() else base_dir / image_path
        elif isinstance(image_value, (bytes, bytearray)):
            image_ref = image_value
        else:
            raise TypeError("manifest image must be a path string or encoded bytes")
        return FormulaRecord(
            id=str(value["id"]),
            image_ref=image_ref,
            raw_latex=str(value["raw_latex"]),
            subset=str(value["subset"]) if value.get("subset") is not None else None,
            source=str(value["source"]) if value.get("source") is not None else None,
        )
    except KeyError as error:
        raise ValueError(f"manifest record is missing required field {error.args[0]!r}") from error


def iter_manifest_records(path: str | Path) -> Iterator[FormulaRecord]:
    manifest = Path(path)
    with manifest.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"manifest line {line_number} must be a JSON object")
            try:
                yield record_from_mapping(value, base_dir=manifest.parent)
            except (TypeError, ValueError) as error:
                raise ValueError(f"invalid manifest line {line_number}: {error}") from error


def decode_record(record: FormulaRecord, normalized_latex: str) -> FormulaSample:
    reference = record.image_ref
    if isinstance(reference, (bytes, bytearray)):
        with Image.open(BytesIO(reference)) as opened:
            image = opened.copy()
    else:
        with Image.open(reference) as opened:
            image = opened.copy()
    return FormulaSample(
        id=record.id,
        image=image,
        raw_latex=record.raw_latex,
        normalized_latex=normalized_latex,
        subset=record.subset,
        source=record.source,
        normalizer_version=NORMALIZER_VERSION,
    )


class FormulaDataset(Dataset[FormulaSample]):
    """Policy-aware random-access dataset over canonical storage records."""

    def __init__(
        self,
        records: Sequence[FormulaRecord],
        *,
        policy: FormulaDataPolicy,
        source: str = "in-memory",
    ) -> None:
        self.policy = policy
        self.source = source
        self.statistics = FormulaDataStatistics(
            dataset_source=source,
            tokenizer_sha256=tokenizer_vocab_sha256(policy.tokenizer),
        )
        accepted: list[tuple[FormulaRecord, str]] = []
        for record in records:
            normalized = normalize(record.raw_latex)
            sample = FormulaSample(
                id=record.id,
                image=record.image_ref,
                raw_latex=record.raw_latex,
                normalized_latex=normalized,
                subset=record.subset,
                source=record.source,
                normalizer_version=NORMALIZER_VERSION,
            )
            if policy.evaluate(sample, self.statistics):
                accepted.append((record, normalized))
        self._accepted = tuple(accepted)

    def __len__(self) -> int:
        return len(self._accepted)

    def __getitem__(self, index: int) -> FormulaSample:
        record, normalized = self._accepted[index]
        return decode_record(record, normalized)


class ManifestFormulaDataset(FormulaDataset):
    """Map-style adapter for a local JSON Lines manifest."""

    def __init__(self, path: str | Path, *, policy: FormulaDataPolicy) -> None:
        manifest = Path(path)
        super().__init__(
            tuple(iter_manifest_records(manifest)), policy=policy, source=str(manifest)
        )
