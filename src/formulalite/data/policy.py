"""Explicit token coverage and tokenized sequence-length policies."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, cast

from .normalizer import NORMALIZER_VERSION
from .schema import FormulaSample
from .tokenizer import FormulaTokenizer


class UnknownTokenPolicy(StrEnum):
    ERROR = "error"
    DROP = "drop"
    KEEP = "keep"


class OverlengthPolicy(StrEnum):
    ERROR = "error"
    DROP = "drop"


class DataPolicyError(ValueError):
    """Base class for sample-level policy failures."""


class UnknownTokenError(DataPolicyError):
    pass


class OverlengthError(DataPolicyError):
    pass


def tokenizer_vocab_sha256(tokenizer: FormulaTokenizer) -> str:
    vocabulary = tokenizer.tokenizer.get_vocab()
    ordered = sorted(vocabulary, key=vocabulary.__getitem__)
    return hashlib.sha256(("\n".join(ordered) + "\n").encode()).hexdigest()


@dataclass
class FormulaDataStatistics:
    """Online statistics; length storage is a histogram, never a per-sample table."""

    dataset_source: str
    tokenizer_sha256: str
    normalizer_version: str = NORMALIZER_VERSION
    total_samples: int = 0
    normalized_samples: int = 0
    unknown_samples: int = 0
    unknown_token_count: int = 0
    dropped_unknown_samples: int = 0
    overlength_samples: int = 0
    dropped_overlength_samples: int = 0
    total_token_count: int = 0
    subset_distribution: Counter[str] = field(default_factory=Counter)
    _length_histogram: Counter[int] = field(default_factory=Counter, repr=False)

    def observe(
        self,
        sample: FormulaSample,
        *,
        sequence_length: int,
        unknown_token_count: int,
        overlength: bool,
        drop_unknown: bool,
        drop_overlength: bool,
    ) -> None:
        self.total_samples += 1
        self.normalized_samples += int(sample.normalized_latex is not None)
        self.total_token_count += sequence_length
        self._length_histogram[sequence_length] += 1
        self.subset_distribution[sample.subset or "unspecified"] += 1
        if unknown_token_count:
            self.unknown_samples += 1
            self.unknown_token_count += unknown_token_count
            self.dropped_unknown_samples += int(drop_unknown)
        if overlength:
            self.overlength_samples += 1
            self.dropped_overlength_samples += int(drop_overlength)

    def _percentile(self, quantile: float) -> int | None:
        if self.total_samples == 0:
            return None
        rank = max(1, math.ceil(quantile * self.total_samples))
        seen = 0
        for length, count in sorted(self._length_histogram.items()):
            seen += count
            if seen >= rank:
                return length
        raise RuntimeError("length histogram does not match sample count")

    def sequence_length_statistics(self) -> dict[str, int | float | None]:
        maximum = max(self._length_histogram, default=None)
        mean = self.total_token_count / self.total_samples if self.total_samples else None
        return {
            "count": self.total_samples,
            "max": maximum,
            "mean": mean,
            "p50": self._percentile(0.50),
            "p90": self._percentile(0.90),
            "p95": self._percentile(0.95),
            "p99": self._percentile(0.99),
            "overlength_sample_count": self.overlength_samples,
        }

    def to_report(self) -> dict[str, Any]:
        unknown_rate = self.unknown_samples / self.total_samples if self.total_samples else 0.0
        unknown_token_rate = (
            self.unknown_token_count / self.total_token_count if self.total_token_count else 0.0
        )
        return {
            "schema_version": "1.0.0",
            "dataset_source": self.dataset_source,
            "tokenizer_sha256": self.tokenizer_sha256,
            "normalizer_version": self.normalizer_version,
            "total_samples": self.total_samples,
            "normalized_samples": self.normalized_samples,
            "unknown_samples": self.unknown_samples,
            "unknown_token_count": self.unknown_token_count,
            "dropped_unknown_samples": self.dropped_unknown_samples,
            "unknown_sample_rate": unknown_rate,
            "unknown_token_rate": unknown_token_rate,
            "overlength_samples": self.overlength_samples,
            "dropped_overlength_samples": self.dropped_overlength_samples,
            "subset_distribution": dict(sorted(self.subset_distribution.items())),
            "sequence_length": self.sequence_length_statistics(),
        }

    def write_json(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(self.to_report(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


@dataclass(frozen=True)
class FormulaDataPolicy:
    tokenizer: FormulaTokenizer
    max_sequence_length: int
    unknown_policy: UnknownTokenPolicy = UnknownTokenPolicy.ERROR
    overlength_policy: OverlengthPolicy = OverlengthPolicy.ERROR

    def __post_init__(self) -> None:
        if self.max_sequence_length < 2:
            raise ValueError("max_sequence_length must fit at least BOS and EOS")

    def evaluate(self, sample: FormulaSample, statistics: FormulaDataStatistics) -> bool:
        if sample.normalized_latex is None:
            raise ValueError(f"sample {sample.id!r} has not been normalized")
        token_ids = self.tokenizer.encode(sample.normalized_latex)
        unknown_id = cast(int | None, self.tokenizer.tokenizer.unk_token_id)
        if unknown_id is None:
            raise RuntimeError("FormulaTokenizer must define unk_token_id")
        unknown_count = token_ids.count(unknown_id)
        overlength = len(token_ids) > self.max_sequence_length
        drop_unknown = bool(unknown_count) and self.unknown_policy is UnknownTokenPolicy.DROP
        drop_overlength = overlength and self.overlength_policy is OverlengthPolicy.DROP
        statistics.observe(
            sample,
            sequence_length=len(token_ids),
            unknown_token_count=unknown_count,
            overlength=overlength,
            drop_unknown=drop_unknown,
            drop_overlength=drop_overlength,
        )
        if unknown_count and self.unknown_policy is UnknownTokenPolicy.ERROR:
            raise UnknownTokenError(
                f"sample {sample.id!r} contains {unknown_count} unknown token(s)"
            )
        if overlength and self.overlength_policy is OverlengthPolicy.ERROR:
            raise OverlengthError(
                f"sample {sample.id!r} has tokenized length {len(token_ids)}, "
                f"exceeding max_sequence_length={self.max_sequence_length}"
            )
        return not (drop_unknown or drop_overlength)
